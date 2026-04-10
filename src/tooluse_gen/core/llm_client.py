"""LLM client with retry logic and error handling.

:class:`LLMClient` wraps the OpenAI chat-completions API with
exponential-backoff retries for transient errors, request/response
logging, and a unified interface for both plain chat and function
calling.
"""

from __future__ import annotations

import json
import time
from typing import Any

import openai
from pydantic import BaseModel, ConfigDict, Field

from tooluse_gen.utils.logging import get_logger

logger = get_logger("core.llm_client")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class LLMClientConfig(BaseModel):
    """Configuration for :class:`LLMClient`."""

    model_config = ConfigDict(use_enum_values=True)

    model: str = Field(default="gpt-4o", description="Default model.")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=1)
    max_retries: int = Field(default=3, ge=0)
    retry_base_delay: float = Field(default=1.0, ge=0.0)
    timeout: float = Field(default=60.0, ge=1.0)


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class LLMClientError(Exception):
    """Raised when an LLM call fails after retries."""

    def __init__(
        self,
        message: str = "",
        original_error: Exception | None = None,
        retries_attempted: int = 0,
    ) -> None:
        super().__init__(message)
        self.original_error = original_error
        self.retries_attempted = retries_attempted


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

# Error codes / types that should trigger a retry.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class LLMClient:
    """OpenAI API client with error handling and retry logic."""

    def __init__(
        self,
        api_key: str | None = None,
        config: LLMClientConfig | None = None,
    ) -> None:
        self._config = config or LLMClientConfig()
        self._logger = logger
        self._request_count = 0
        self._error_count = 0

        if api_key is not None:
            self._client: openai.OpenAI | None = openai.OpenAI(api_key=api_key)
        else:
            self._client = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def request_count(self) -> int:
        """Total API call attempts."""
        return self._request_count

    @property
    def error_count(self) -> int:
        """Total errors encountered."""
        return self._error_count

    @property
    def is_available(self) -> bool:
        """True when an API key has been provided."""
        return self._client is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Run a chat completion, with optional tool definitions."""
        if self._client is None:
            raise LLMClientError("LLM client not initialised (no API key)")

        resolved_model = model or self._config.model
        resolved_temp = temperature if temperature is not None else self._config.temperature
        resolved_max = max_tokens if max_tokens is not None else self._config.max_tokens

        kwargs: dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
            "temperature": resolved_temp,
            "max_tokens": resolved_max,
        }
        if tools:
            kwargs["tools"] = tools

        self._logger.debug(
            "Request: model=%s, msgs=%d, tools=%s",
            resolved_model,
            len(messages),
            len(tools) if tools else 0,
        )

        response = self._call_with_retry(**kwargs)
        result = self._parse_response(response)

        self._logger.debug(
            "Response: finish=%s, tokens=%s",
            result.get("finish_reason"),
            result.get("usage", {}).get("total_tokens"),
        )
        return result

    def chat_completion_with_functions(
        self,
        messages: list[dict[str, str]],
        functions: list[dict[str, Any]],
        model: str | None = None,
    ) -> dict[str, Any]:
        """Convenience wrapper that converts *functions* to tool format."""
        tools = [{"type": "function", "function": f} for f in functions]
        return self.chat_completion(messages, tools=tools, model=model)

    # ------------------------------------------------------------------
    # Retry logic
    # ------------------------------------------------------------------

    def _call_with_retry(self, **kwargs: Any) -> Any:
        """Call the API with exponential-backoff retries for transient errors."""
        assert self._client is not None
        max_attempts = self._config.max_retries + 1
        last_error: Exception | None = None

        for attempt in range(max_attempts):
            self._request_count += 1
            try:
                return self._client.chat.completions.create(**kwargs)
            except openai.RateLimitError as exc:
                last_error = exc
                self._error_count += 1
                self._logger.warning("Rate limit (attempt %d/%d)", attempt + 1, max_attempts)
            except openai.APITimeoutError as exc:
                last_error = exc
                self._error_count += 1
                self._logger.warning("Timeout (attempt %d/%d)", attempt + 1, max_attempts)
            except openai.APIStatusError as exc:
                last_error = exc
                self._error_count += 1
                if exc.status_code in _RETRYABLE_STATUS:
                    self._logger.warning(
                        "Server error %d (attempt %d/%d)", exc.status_code, attempt + 1, max_attempts
                    )
                else:
                    self._logger.error("Non-retryable API error: %s", exc)
                    raise LLMClientError(
                        str(exc), original_error=exc, retries_attempted=attempt + 1
                    ) from exc
            except openai.APIConnectionError as exc:
                last_error = exc
                self._error_count += 1
                self._logger.warning("Connection error (attempt %d/%d)", attempt + 1, max_attempts)

            # Exponential backoff before next retry.
            if attempt < max_attempts - 1:
                delay = self._config.retry_base_delay * (2**attempt)
                time.sleep(delay)

        raise LLMClientError(
            f"All {max_attempts} attempts failed",
            original_error=last_error,
            retries_attempted=max_attempts,
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(response: Any) -> dict[str, Any]:
        choice = response.choices[0]
        content = choice.message.content
        raw_tc = choice.message.tool_calls

        tool_calls: list[dict[str, Any]] | None = None
        if raw_tc:
            tool_calls = []
            for tc in raw_tc:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError, AttributeError):
                    args = {}
                tool_calls.append({
                    "name": tc.function.name,
                    "arguments": args,
                    "id": getattr(tc, "id", None),
                })

        usage: dict[str, int] = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return {
            "content": content,
            "tool_calls": tool_calls,
            "finish_reason": choice.finish_reason,
            "usage": usage,
        }
