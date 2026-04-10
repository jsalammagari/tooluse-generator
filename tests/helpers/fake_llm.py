"""Deterministic mock LLM for testing.

:class:`FakeLLM` provides pre-configured responses and supports both the
raw OpenAI ``client.chat.completions.create`` interface (used by agents)
and the higher-level :class:`LLMClient` ``chat_completion`` dict interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Response container
# ---------------------------------------------------------------------------


@dataclass
class FakeLLMResponse:
    """A single pre-configured response."""

    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str = "stop"


# ---------------------------------------------------------------------------
# Pre-built fixtures
# ---------------------------------------------------------------------------

USER_REQUEST_RESPONSE = FakeLLMResponse(
    content="I need to book a hotel in Paris for next weekend."
)

ASSISTANT_TOOL_CALL_RESPONSE = FakeLLMResponse(
    content=None,
    tool_calls=[
        {"name": "hotels/search", "arguments": '{"city": "Paris"}', "id": "call_1"},
    ],
    finish_reason="tool_calls",
)

DISAMBIGUATION_RESPONSE = FakeLLMResponse(content="What is your budget range?")

JUDGE_RESPONSE = FakeLLMResponse(
    content=(
        '{"tool_correctness": 4, "argument_grounding": 3,'
        ' "task_completion": 5, "naturalness": 4,'
        ' "reasoning": "Good conversation"}'
    ),
)

FINAL_ANSWER_RESPONSE = FakeLLMResponse(
    content="I've booked Hotel du Marais for you. Confirmation: BK-123."
)

GENERIC_RESPONSE = FakeLLMResponse(content="I'll help you with that.")


# ---------------------------------------------------------------------------
# OpenAI-compatible response objects
# ---------------------------------------------------------------------------


class _Function:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, tc: dict[str, Any]) -> None:
        args = tc.get("arguments", "{}")
        self.function = _Function(tc.get("name", ""), args if isinstance(args, str) else str(args))
        self.id = tc.get("id", "")


class _Message:
    def __init__(self, resp: FakeLLMResponse) -> None:
        self.content = resp.content
        if resp.tool_calls is not None:
            self.tool_calls: list[_ToolCall] | None = [_ToolCall(tc) for tc in resp.tool_calls]
        else:
            self.tool_calls = None


class _Choice:
    def __init__(self, resp: FakeLLMResponse) -> None:
        self.message = _Message(resp)
        self.finish_reason = resp.finish_reason


class _Usage:
    def __init__(self) -> None:
        self.prompt_tokens = 10
        self.completion_tokens = 5
        self.total_tokens = 15


class _OpenAIResponse:
    def __init__(self, resp: FakeLLMResponse) -> None:
        self.choices = [_Choice(resp)]
        self.usage = _Usage()


# ---------------------------------------------------------------------------
# FakeLLM
# ---------------------------------------------------------------------------


class _Completions:
    """Nested helper — ``chat.completions.create(...)``."""

    def __init__(self, owner: FakeLLM) -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> _OpenAIResponse:
        resp = self._owner._generate(kwargs)
        return _OpenAIResponse(resp)


class _Chat:
    """Nested helper — ``chat.completions``."""

    def __init__(self, owner: FakeLLM) -> None:
        self.completions = _Completions(owner)


class FakeLLM:
    """Deterministic mock LLM — drop-in for OpenAI client or LLMClient."""

    def __init__(
        self,
        responses: list[FakeLLMResponse] | None = None,
        default_response: FakeLLMResponse | None = None,
        pattern_responses: dict[str, FakeLLMResponse] | None = None,
    ) -> None:
        self._responses = responses or []
        self._default = default_response or FakeLLMResponse(content="Mock response")
        self._patterns = pattern_responses or {}
        self._call_history: list[dict[str, Any]] = []
        self._call_index = 0

        # OpenAI client interface
        self.chat = _Chat(self)

    # ------------------------------------------------------------------
    # LLMClient interface
    # ------------------------------------------------------------------

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """LLMClient-compatible dict response."""
        resp = self._generate({"messages": messages, "tools": tools, **kwargs})
        tc_list: list[dict[str, Any]] | None = None
        if resp.tool_calls:
            tc_list = [
                {"name": tc.get("name", ""), "arguments": tc.get("arguments", "{}"), "id": tc.get("id", "")}
                for tc in resp.tool_calls
            ]
        return {
            "content": resp.content,
            "tool_calls": tc_list,
            "finish_reason": resp.finish_reason,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    def chat_completion_with_functions(
        self,
        messages: list[dict[str, Any]],
        functions: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Wraps functions as tools and delegates."""
        tools = [{"type": "function", "function": f} for f in functions]
        return self.chat_completion(messages, tools=tools, **kwargs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def call_count(self) -> int:
        """Number of calls made."""
        return len(self._call_history)

    @property
    def call_history(self) -> list[dict[str, Any]]:
        """Copy of all call kwargs."""
        return list(self._call_history)

    def last_call(self) -> dict[str, Any] | None:
        """Most recent call kwargs, or None."""
        return self._call_history[-1] if self._call_history else None

    def reset(self) -> None:
        """Clear history and reset response index."""
        self._call_history.clear()
        self._call_index = 0

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def _generate(self, kwargs: dict[str, Any]) -> FakeLLMResponse:
        self._call_history.append(dict(kwargs))

        # Pattern matching on last user message.
        messages = kwargs.get("messages") or []
        last_user = ""
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                last_user = msg.get("content", "")
                break

        for pattern, resp in self._patterns.items():
            if pattern in last_user:
                return resp

        # Sequential responses.
        if self._call_index < len(self._responses):
            resp = self._responses[self._call_index]
            self._call_index += 1
            return resp

        return self._default
