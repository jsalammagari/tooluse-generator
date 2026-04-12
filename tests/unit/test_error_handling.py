"""Tests for LLM API error handling (Task 77)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import openai
import pytest

from tooluse_gen.agents.conversation_models import Conversation, Message
from tooluse_gen.core.llm_client import (
    LLMClient,
    LLMClientConfig,
    LLMClientError,
    classify_error,
    is_retryable,
)
from tooluse_gen.evaluation.judge import JudgeAgent
from tooluse_gen.evaluation.models import JudgeScores

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _status_error(code: int) -> openai.APIStatusError:
    resp = MagicMock()
    resp.status_code = code
    return openai.APIStatusError(f"HTTP {code}", response=resp, body=None)


def _make_conversation() -> Conversation:
    return Conversation(messages=[
        Message(role="user", content="hi"),
        Message(role="assistant", content="hello"),
    ])


def _make_mock_response(content: str = "Hello") -> MagicMock:
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = content
    mock.choices[0].message.tool_calls = None
    mock.choices[0].finish_reason = "stop"
    mock.usage = MagicMock()
    mock.usage.prompt_tokens = 10
    mock.usage.completion_tokens = 5
    mock.usage.total_tokens = 15
    return mock


# ===================================================================
# classify_error
# ===================================================================


class TestClassifyError:
    def test_rate_limit(self) -> None:
        err = openai.RateLimitError("rl", response=MagicMock(), body=None)
        assert classify_error(err) == "retryable"

    def test_timeout(self) -> None:
        err = openai.APITimeoutError(request=MagicMock())
        assert classify_error(err) == "retryable"

    def test_connection_error(self) -> None:
        err = openai.APIConnectionError(request=MagicMock())
        assert classify_error(err) == "retryable"

    def test_server_500(self) -> None:
        assert classify_error(_status_error(500)) == "retryable"

    def test_server_502(self) -> None:
        assert classify_error(_status_error(502)) == "retryable"

    def test_server_503(self) -> None:
        assert classify_error(_status_error(503)) == "retryable"

    def test_server_504(self) -> None:
        assert classify_error(_status_error(504)) == "retryable"

    def test_auth_error_401(self) -> None:
        assert classify_error(_status_error(401)) == "fatal"

    def test_bad_request_400(self) -> None:
        assert classify_error(_status_error(400)) == "fatal"

    def test_not_found_404(self) -> None:
        assert classify_error(_status_error(404)) == "fatal"

    def test_llm_client_error(self) -> None:
        assert classify_error(LLMClientError("fail")) == "retryable"

    def test_unknown_error(self) -> None:
        assert classify_error(ValueError("oops")) == "unknown"

    def test_runtime_error_unknown(self) -> None:
        assert classify_error(RuntimeError("crash")) == "unknown"


# ===================================================================
# is_retryable
# ===================================================================


class TestIsRetryable:
    def test_retryable_true(self) -> None:
        err = openai.RateLimitError("rl", response=MagicMock(), body=None)
        assert is_retryable(err) is True

    def test_fatal_false(self) -> None:
        assert is_retryable(_status_error(401)) is False

    def test_unknown_false(self) -> None:
        assert is_retryable(ValueError("x")) is False

    def test_timeout_retryable(self) -> None:
        err = openai.APITimeoutError(request=MagicMock())
        assert is_retryable(err) is True


# ===================================================================
# Graceful degradation — JudgeAgent
# ===================================================================


class TestGracefulDegradation:
    def test_judge_falls_back_on_error(self) -> None:
        """JudgeAgent falls back to offline when LLM call fails."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")
        judge = JudgeAgent(llm_client=mock_client)
        conv = _make_conversation()
        scores = judge.score(conv)
        assert isinstance(scores, JudgeScores)
        assert scores.tool_correctness >= 1

    def test_judge_falls_back_on_rate_limit(self) -> None:
        """JudgeAgent handles rate limit gracefully."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = openai.RateLimitError(
            "rl", response=MagicMock(), body=None
        )
        judge = JudgeAgent(llm_client=mock_client)
        scores = judge.score(_make_conversation())
        assert isinstance(scores, JudgeScores)

    def test_judge_falls_back_on_timeout(self) -> None:
        """JudgeAgent handles timeout gracefully."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = openai.APITimeoutError(
            request=MagicMock()
        )
        judge = JudgeAgent(llm_client=mock_client)
        scores = judge.score(_make_conversation())
        assert isinstance(scores, JudgeScores)

    def test_judge_batch_continues_on_error(self) -> None:
        """score_batch returns scores for all conversations even if LLM fails."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("fail")
        judge = JudgeAgent(llm_client=mock_client)
        convs = [_make_conversation(), _make_conversation()]
        results = judge.score_batch(convs)
        assert len(results) == 2
        assert all(isinstance(s, JudgeScores) for s in results)

    def test_judge_offline_mode_unaffected(self) -> None:
        """Offline mode (no client) works as before."""
        judge = JudgeAgent()  # no client
        scores = judge.score(_make_conversation())
        assert isinstance(scores, JudgeScores)
        assert scores.average > 0


# ===================================================================
# Backoff timing
# ===================================================================


class TestBackoffTiming:
    @patch("tooluse_gen.core.llm_client.time.sleep")
    def test_exponential_delays(self, mock_sleep: MagicMock) -> None:
        """Backoff delays increase exponentially: base, base*2, base*4."""
        client = LLMClient(
            api_key="sk-test",
            config=LLMClientConfig(max_retries=3, retry_base_delay=1.0),
        )
        assert client._client is not None
        client._client.chat.completions.create = MagicMock(  # type: ignore[union-attr]
            side_effect=[
                openai.RateLimitError("rl", response=MagicMock(), body=None),
                openai.RateLimitError("rl", response=MagicMock(), body=None),
                openai.RateLimitError("rl", response=MagicMock(), body=None),
                _make_mock_response(),
            ]
        )
        client.chat_completion([{"role": "user", "content": "x"}])
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert delays == [1.0, 2.0, 4.0]

    @patch("tooluse_gen.core.llm_client.time.sleep")
    def test_configurable_base_delay(self, mock_sleep: MagicMock) -> None:
        """Base delay uses config value."""
        client = LLMClient(
            api_key="sk-test",
            config=LLMClientConfig(max_retries=1, retry_base_delay=0.5),
        )
        assert client._client is not None
        client._client.chat.completions.create = MagicMock(  # type: ignore[union-attr]
            side_effect=[
                openai.RateLimitError("rl", response=MagicMock(), body=None),
                _make_mock_response(),
            ]
        )
        client.chat_completion([{"role": "user", "content": "x"}])
        assert mock_sleep.call_args_list[0].args[0] == 0.5


# ===================================================================
# Edge cases
# ===================================================================


class TestErrorEdgeCases:
    def test_malformed_response_no_usage(self) -> None:
        """Response with None usage returns empty dict."""
        client = LLMClient(
            api_key="sk-test", config=LLMClientConfig(retry_base_delay=0.0)
        )
        mock_resp = _make_mock_response()
        mock_resp.usage = None
        assert client._client is not None
        client._client.chat.completions.create = MagicMock(return_value=mock_resp)  # type: ignore[union-attr]
        result = client.chat_completion([{"role": "user", "content": "x"}])
        assert result["usage"] == {}

    def test_json_parse_error_in_tool_calls(self) -> None:
        """Malformed tool call arguments produce empty dict, not crash."""
        client = LLMClient(
            api_key="sk-test", config=LLMClientConfig(retry_base_delay=0.0)
        )
        tc = MagicMock()
        tc.function.name = "search"
        tc.function.arguments = "NOT VALID JSON"
        tc.id = "tc_1"
        mock_resp = _make_mock_response(content=None)
        mock_resp.choices[0].message.tool_calls = [tc]
        assert client._client is not None
        client._client.chat.completions.create = MagicMock(return_value=mock_resp)  # type: ignore[union-attr]
        result = client.chat_completion([{"role": "user", "content": "x"}])
        assert result["tool_calls"][0]["arguments"] == {}

    def test_no_client_raises_descriptive_error(self) -> None:
        """Client without API key raises clear error."""
        client = LLMClient()
        with pytest.raises(LLMClientError, match="not initialised"):
            client.chat_completion([{"role": "user", "content": "x"}])
