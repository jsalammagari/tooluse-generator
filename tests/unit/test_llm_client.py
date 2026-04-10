"""Tests for LLMClient (Task 53)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import openai
import pytest

from tooluse_gen.core.llm_client import LLMClient, LLMClientConfig, LLMClientError

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_response(
    content: str | None = "Hello",
    tool_calls: list[MagicMock] | None = None,
    finish_reason: str = "stop",
) -> MagicMock:
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = content
    mock.choices[0].message.tool_calls = tool_calls
    mock.choices[0].finish_reason = finish_reason
    mock.usage = MagicMock()
    mock.usage.prompt_tokens = 10
    mock.usage.completion_tokens = 5
    mock.usage.total_tokens = 15
    return mock


def _make_client() -> LLMClient:
    """Client with a mocked OpenAI backend."""
    client = LLMClient(api_key="sk-test", config=LLMClientConfig(retry_base_delay=0.0))
    return client


def _inject_response(client: LLMClient, response: MagicMock) -> None:
    assert client._client is not None
    client._client.chat.completions.create = MagicMock(return_value=response)  # type: ignore[union-attr]


# ===================================================================
# LLMClientConfig
# ===================================================================


class TestConfig:
    def test_defaults(self):
        c = LLMClientConfig()
        assert c.model == "gpt-4o"
        assert c.temperature == 0.7
        assert c.max_tokens == 1000
        assert c.max_retries == 3

    def test_custom(self):
        c = LLMClientConfig(model="gpt-4o-mini", temperature=0.3, max_retries=5)
        assert c.model == "gpt-4o-mini"

    def test_temperature_bounds(self):
        LLMClientConfig(temperature=0.0)
        LLMClientConfig(temperature=2.0)
        with pytest.raises(ValueError):
            LLMClientConfig(temperature=-0.1)
        with pytest.raises(ValueError):
            LLMClientConfig(temperature=2.1)

    def test_max_retries_bounds(self):
        LLMClientConfig(max_retries=0)
        with pytest.raises(ValueError):
            LLMClientConfig(max_retries=-1)


# ===================================================================
# Construction
# ===================================================================


class TestConstruction:
    def test_with_key(self):
        c = LLMClient(api_key="sk-test")
        assert c.is_available

    def test_without_key(self):
        c = LLMClient()
        assert not c.is_available

    def test_custom_config(self):
        cfg = LLMClientConfig(model="gpt-3.5-turbo")
        c = LLMClient(config=cfg)
        assert c._config.model == "gpt-3.5-turbo"


# ===================================================================
# chat_completion — success
# ===================================================================


class TestChatSuccess:
    def test_returns_content(self):
        c = _make_client()
        _inject_response(c, _make_mock_response("Hi there"))
        r = c.chat_completion([{"role": "user", "content": "hello"}])
        assert r["content"] == "Hi there"

    def test_returns_tool_calls(self):
        tc = MagicMock()
        tc.function.name = "search"
        tc.function.arguments = '{"city": "Paris"}'
        tc.id = "tc_1"
        c = _make_client()
        _inject_response(c, _make_mock_response(content=None, tool_calls=[tc]))
        r = c.chat_completion([{"role": "user", "content": "x"}])
        assert r["tool_calls"] is not None
        assert r["tool_calls"][0]["name"] == "search"

    def test_returns_finish_reason(self):
        c = _make_client()
        _inject_response(c, _make_mock_response(finish_reason="length"))
        r = c.chat_completion([{"role": "user", "content": "x"}])
        assert r["finish_reason"] == "length"

    def test_uses_config_defaults(self):
        c = _make_client()
        _inject_response(c, _make_mock_response())
        c.chat_completion([{"role": "user", "content": "x"}])
        call_kwargs = c._client.chat.completions.create.call_args.kwargs  # type: ignore[union-attr]
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["temperature"] == 0.7

    def test_overrides_params(self):
        c = _make_client()
        _inject_response(c, _make_mock_response())
        c.chat_completion(
            [{"role": "user", "content": "x"}],
            model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=50,
        )
        kw = c._client.chat.completions.create.call_args.kwargs  # type: ignore[union-attr]
        assert kw["model"] == "gpt-3.5-turbo"
        assert kw["temperature"] == 0.1
        assert kw["max_tokens"] == 50


# ===================================================================
# chat_completion — tools
# ===================================================================


class TestTools:
    def test_passes_tools(self):
        c = _make_client()
        _inject_response(c, _make_mock_response())
        tools = [{"type": "function", "function": {"name": "f"}}]
        c.chat_completion([{"role": "user", "content": "x"}], tools=tools)
        kw = c._client.chat.completions.create.call_args.kwargs  # type: ignore[union-attr]
        assert kw["tools"] == tools

    def test_with_functions_wraps(self):
        c = _make_client()
        _inject_response(c, _make_mock_response())
        funcs = [{"name": "search", "parameters": {}}]
        c.chat_completion_with_functions(
            [{"role": "user", "content": "x"}], functions=funcs
        )
        kw = c._client.chat.completions.create.call_args.kwargs  # type: ignore[union-attr]
        assert kw["tools"] == [{"type": "function", "function": funcs[0]}]

    def test_tool_calls_parsed(self):
        tc = MagicMock()
        tc.function.name = "book"
        tc.function.arguments = '{"hotel_id": "h1"}'
        tc.id = "tc_2"
        c = _make_client()
        _inject_response(c, _make_mock_response(content=None, tool_calls=[tc]))
        r = c.chat_completion([{"role": "user", "content": "x"}])
        assert r["tool_calls"][0]["arguments"] == {"hotel_id": "h1"}
        assert r["tool_calls"][0]["id"] == "tc_2"


# ===================================================================
# Error handling — retries
# ===================================================================


class TestRetries:
    @patch("tooluse_gen.core.llm_client.time.sleep")
    def test_rate_limit_retries(self, mock_sleep: MagicMock):
        c = _make_client()
        assert c._client is not None
        c._client.chat.completions.create = MagicMock(  # type: ignore[union-attr]
            side_effect=[
                openai.RateLimitError("rate limit", response=MagicMock(), body=None),
                _make_mock_response(),
            ]
        )
        r = c.chat_completion([{"role": "user", "content": "x"}])
        assert r["content"] == "Hello"
        assert c.request_count == 2

    @patch("tooluse_gen.core.llm_client.time.sleep")
    def test_server_error_retries(self, mock_sleep: MagicMock):
        c = _make_client()
        assert c._client is not None
        resp_500 = MagicMock()
        resp_500.status_code = 500
        c._client.chat.completions.create = MagicMock(  # type: ignore[union-attr]
            side_effect=[
                openai.APIStatusError("server", response=resp_500, body=None),
                _make_mock_response(),
            ]
        )
        r = c.chat_completion([{"role": "user", "content": "x"}])
        assert r["content"] == "Hello"

    @patch("tooluse_gen.core.llm_client.time.sleep")
    def test_succeeds_after_retry(self, mock_sleep: MagicMock):
        c = _make_client()
        assert c._client is not None
        c._client.chat.completions.create = MagicMock(  # type: ignore[union-attr]
            side_effect=[
                openai.RateLimitError("rl", response=MagicMock(), body=None),
                openai.RateLimitError("rl", response=MagicMock(), body=None),
                _make_mock_response("recovered"),
            ]
        )
        r = c.chat_completion([{"role": "user", "content": "x"}])
        assert r["content"] == "recovered"
        assert c.request_count == 3

    @patch("tooluse_gen.core.llm_client.time.sleep")
    def test_max_retries_raises(self, mock_sleep: MagicMock):
        c = LLMClient(api_key="sk-test", config=LLMClientConfig(max_retries=1, retry_base_delay=0.0))
        assert c._client is not None
        c._client.chat.completions.create = MagicMock(  # type: ignore[union-attr]
            side_effect=openai.RateLimitError("rl", response=MagicMock(), body=None)
        )
        with pytest.raises(LLMClientError) as exc_info:
            c.chat_completion([{"role": "user", "content": "x"}])
        assert exc_info.value.retries_attempted == 2

    def test_non_retryable_raises_immediately(self):
        c = _make_client()
        assert c._client is not None
        resp_401 = MagicMock()
        resp_401.status_code = 401
        c._client.chat.completions.create = MagicMock(  # type: ignore[union-attr]
            side_effect=openai.AuthenticationError("bad key", response=resp_401, body=None)
        )
        with pytest.raises(LLMClientError):
            c.chat_completion([{"role": "user", "content": "x"}])
        assert c.request_count == 1  # no retry


# ===================================================================
# Error handling — timeout
# ===================================================================


class TestTimeout:
    @patch("tooluse_gen.core.llm_client.time.sleep")
    def test_timeout_retries(self, mock_sleep: MagicMock):
        c = _make_client()
        assert c._client is not None
        c._client.chat.completions.create = MagicMock(  # type: ignore[union-attr]
            side_effect=[
                openai.APITimeoutError(request=MagicMock()),
                _make_mock_response("ok"),
            ]
        )
        r = c.chat_completion([{"role": "user", "content": "x"}])
        assert r["content"] == "ok"

    @patch("tooluse_gen.core.llm_client.time.sleep")
    def test_timeout_exhausted(self, mock_sleep: MagicMock):
        c = LLMClient(api_key="sk-test", config=LLMClientConfig(max_retries=1, retry_base_delay=0.0))
        assert c._client is not None
        c._client.chat.completions.create = MagicMock(  # type: ignore[union-attr]
            side_effect=openai.APITimeoutError(request=MagicMock())
        )
        with pytest.raises(LLMClientError):
            c.chat_completion([{"role": "user", "content": "x"}])


# ===================================================================
# Counters
# ===================================================================


class TestCounters:
    def test_request_count(self):
        c = _make_client()
        _inject_response(c, _make_mock_response())
        c.chat_completion([{"role": "user", "content": "x"}])
        c.chat_completion([{"role": "user", "content": "y"}])
        assert c.request_count == 2

    @patch("tooluse_gen.core.llm_client.time.sleep")
    def test_error_count(self, mock_sleep: MagicMock):
        c = _make_client()
        assert c._client is not None
        c._client.chat.completions.create = MagicMock(  # type: ignore[union-attr]
            side_effect=[
                openai.RateLimitError("rl", response=MagicMock(), body=None),
                _make_mock_response(),
            ]
        )
        c.chat_completion([{"role": "user", "content": "x"}])
        assert c.error_count == 1

    def test_is_available(self):
        assert LLMClient(api_key="sk-test").is_available
        assert not LLMClient().is_available


# ===================================================================
# LLMClientError
# ===================================================================


class TestError:
    def test_message(self):
        e = LLMClientError("oops")
        assert str(e) == "oops"

    def test_original_error(self):
        orig = ValueError("inner")
        e = LLMClientError("wrap", original_error=orig)
        assert e.original_error is orig

    def test_retries_attempted(self):
        e = LLMClientError("fail", retries_attempted=5)
        assert e.retries_attempted == 5


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    def test_none_content(self):
        c = _make_client()
        _inject_response(c, _make_mock_response(content=None))
        r = c.chat_completion([{"role": "user", "content": "x"}])
        assert r["content"] is None

    def test_empty_messages(self):
        c = _make_client()
        _inject_response(c, _make_mock_response())
        r = c.chat_completion([])
        assert r["content"] == "Hello"

    def test_usage_stats(self):
        c = _make_client()
        _inject_response(c, _make_mock_response())
        r = c.chat_completion([{"role": "user", "content": "x"}])
        assert r["usage"]["total_tokens"] == 15
        assert r["usage"]["prompt_tokens"] == 10

    def test_no_client_raises(self):
        c = LLMClient()
        with pytest.raises(LLMClientError, match="not initialised"):
            c.chat_completion([{"role": "user", "content": "x"}])
