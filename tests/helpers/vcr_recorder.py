"""VCR-style LLM response recorder and replayer.

:class:`LLMRecorder` saves LLM responses as JSON cassettes keyed by
prompt hash so integration tests can replay deterministic responses
without calling a real API.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tests.helpers.fake_llm import FakeLLMResponse, _OpenAIResponse
from tooluse_gen.core.cache import PromptCache

# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class CassetteNotFoundError(Exception):
    """Raised when no cassette exists for a given prompt hash."""


# ---------------------------------------------------------------------------
# OpenAI-compatible nested helpers
# ---------------------------------------------------------------------------


class _Completions:
    def __init__(self, owner: LLMRecorder) -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> _OpenAIResponse:
        messages = kwargs.pop("messages", [])
        model = kwargs.pop("model", "gpt-4o")
        tools = kwargs.pop("tools", None)
        result = self._owner.chat_completion(messages, tools=tools, model=model, **kwargs)
        resp = FakeLLMResponse(
            content=result.get("content"),
            tool_calls=result.get("tool_calls"),
            finish_reason=result.get("finish_reason", "stop"),
        )
        return _OpenAIResponse(resp)


class _Chat:
    def __init__(self, owner: LLMRecorder) -> None:
        self.completions = _Completions(owner)


# ---------------------------------------------------------------------------
# LLMRecorder
# ---------------------------------------------------------------------------


class LLMRecorder:
    """Records and replays LLM responses for deterministic integration tests."""

    def __init__(
        self,
        cassette_dir: Path | str,
        record: bool = False,
        replay: bool = True,
        client: Any | None = None,
    ) -> None:
        self._cassette_dir = Path(cassette_dir)
        self._cassette_dir.mkdir(parents=True, exist_ok=True)
        self._record = record
        self._replay = replay
        self._client = client
        self._call_count = 0
        self._replay_count = 0
        self._record_count = 0
        self.chat = _Chat(self)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def replay_count(self) -> int:
        return self._replay_count

    @property
    def record_count(self) -> int:
        return self._record_count

    @property
    def is_recording(self) -> bool:
        return self._record

    @property
    def is_replaying(self) -> bool:
        return self._replay

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        model: str = "gpt-4o",
        **kwargs: Any,
    ) -> dict[str, Any]:
        prompt_hash = PromptCache.hash_prompt(messages, model, tools=tools, **kwargs)
        self._call_count += 1

        # Replay path.
        if self._replay:
            cached = self.load_cassette(prompt_hash)
            if cached is not None:
                self._replay_count += 1
                return cached

        # Record path.
        if self._record and self._client is not None:
            response = self._client.chat_completion(messages, tools=tools, model=model, **kwargs)
            self.save_cassette(prompt_hash, response)
            self._record_count += 1
            return response

        raise CassetteNotFoundError(f"No cassette for hash {prompt_hash}")

    def chat_completion_with_functions(
        self,
        messages: list[dict[str, str]],
        functions: list[dict[str, Any]],
        model: str = "gpt-4o",
        **kwargs: Any,
    ) -> dict[str, Any]:
        tools = [{"type": "function", "function": f} for f in functions]
        return self.chat_completion(messages, tools=tools, model=model, **kwargs)

    # ------------------------------------------------------------------
    # Cassette management
    # ------------------------------------------------------------------

    def load_cassette(self, prompt_hash: str) -> dict[str, Any] | None:
        path = self._cassette_dir / f"{prompt_hash}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return data.get("response")  # type: ignore[no-any-return]
        except (json.JSONDecodeError, OSError):
            return None

    def save_cassette(self, prompt_hash: str, response: dict[str, Any]) -> None:
        path = self._cassette_dir / f"{prompt_hash}.json"
        data = {
            "response": response,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(data, default=str))

    def list_cassettes(self) -> list[str]:
        return sorted(p.stem for p in self._cassette_dir.glob("*.json"))

    def clear_cassettes(self) -> int:
        count = 0
        for p in self._cassette_dir.glob("*.json"):
            p.unlink()
            count += 1
        return count

    def stats(self) -> dict[str, Any]:
        return {
            "calls": self._call_count,
            "replays": self._replay_count,
            "records": self._record_count,
            "cassettes": len(self.list_cassettes()),
        }
