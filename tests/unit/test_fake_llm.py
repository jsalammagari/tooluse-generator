"""Tests for FakeLLM (Task 55)."""

from __future__ import annotations

import json

import pytest

from tests.helpers.fake_llm import (
    ASSISTANT_TOOL_CALL_RESPONSE,
    DISAMBIGUATION_RESPONSE,
    FINAL_ANSWER_RESPONSE,
    GENERIC_RESPONSE,
    JUDGE_RESPONSE,
    USER_REQUEST_RESPONSE,
    FakeLLM,
    FakeLLMResponse,
)

pytestmark = pytest.mark.unit


# ===================================================================
# FakeLLMResponse
# ===================================================================


class TestFakeLLMResponse:
    def test_defaults(self):
        r = FakeLLMResponse()
        assert r.content is None
        assert r.tool_calls is None
        assert r.finish_reason == "stop"

    def test_custom(self):
        r = FakeLLMResponse(content="hi", finish_reason="length")
        assert r.content == "hi"
        assert r.finish_reason == "length"

    def test_with_tool_calls(self):
        r = FakeLLMResponse(
            tool_calls=[{"name": "f", "arguments": "{}", "id": "c1"}],
            finish_reason="tool_calls",
        )
        assert r.tool_calls is not None
        assert len(r.tool_calls) == 1


# ===================================================================
# Construction
# ===================================================================


class TestConstruction:
    def test_default(self):
        f = FakeLLM()
        assert f.call_count == 0

    def test_with_responses(self):
        f = FakeLLM(responses=[FakeLLMResponse(content="a")])
        assert f.call_count == 0

    def test_with_patterns(self):
        f = FakeLLM(pattern_responses={"hotel": FakeLLMResponse(content="found")})
        assert f.call_count == 0


# ===================================================================
# Sequential responses
# ===================================================================


class TestSequential:
    def test_returns_in_order(self):
        f = FakeLLM(responses=[
            FakeLLMResponse(content="first"),
            FakeLLMResponse(content="second"),
        ])
        r1 = f.chat_completion([{"role": "user", "content": "a"}])
        r2 = f.chat_completion([{"role": "user", "content": "b"}])
        assert r1["content"] == "first"
        assert r2["content"] == "second"

    def test_exhausted_falls_back(self):
        f = FakeLLM(
            responses=[FakeLLMResponse(content="only")],
            default_response=FakeLLMResponse(content="default"),
        )
        f.chat_completion([{"role": "user", "content": "a"}])
        r = f.chat_completion([{"role": "user", "content": "b"}])
        assert r["content"] == "default"

    def test_index_increments(self):
        f = FakeLLM(responses=[
            FakeLLMResponse(content="0"),
            FakeLLMResponse(content="1"),
            FakeLLMResponse(content="2"),
        ])
        for i in range(3):
            r = f.chat_completion([{"role": "user", "content": str(i)}])
            assert r["content"] == str(i)

    def test_reset_resets_index(self):
        f = FakeLLM(responses=[FakeLLMResponse(content="r")])
        f.chat_completion([{"role": "user", "content": "a"}])
        f.reset()
        r = f.chat_completion([{"role": "user", "content": "b"}])
        assert r["content"] == "r"


# ===================================================================
# Pattern matching
# ===================================================================


class TestPatternMatching:
    def test_matches_substring(self):
        f = FakeLLM(pattern_responses={"hotel": FakeLLMResponse(content="Hotel!")})
        r = f.chat_completion([{"role": "user", "content": "Find a hotel"}])
        assert r["content"] == "Hotel!"

    def test_first_match_wins(self):
        f = FakeLLM(pattern_responses={
            "hotel": FakeLLMResponse(content="hotel match"),
            "book": FakeLLMResponse(content="book match"),
        })
        r = f.chat_completion([{"role": "user", "content": "book a hotel"}])
        # dict ordering — first key that matches
        assert r["content"] in ("hotel match", "book match")

    def test_no_match_falls_through(self):
        f = FakeLLM(
            pattern_responses={"xyz": FakeLLMResponse(content="xyz")},
            responses=[FakeLLMResponse(content="sequential")],
        )
        r = f.chat_completion([{"role": "user", "content": "hello"}])
        assert r["content"] == "sequential"

    def test_pattern_with_tool_calls(self):
        tc = [{"name": "search", "arguments": "{}", "id": "c1"}]
        f = FakeLLM(pattern_responses={
            "search": FakeLLMResponse(content=None, tool_calls=tc, finish_reason="tool_calls"),
        })
        r = f.chat_completion([{"role": "user", "content": "search hotels"}])
        assert r["tool_calls"] is not None
        assert r["tool_calls"][0]["name"] == "search"

    def test_case_sensitive(self):
        f = FakeLLM(
            pattern_responses={"Hotel": FakeLLMResponse(content="matched")},
            default_response=FakeLLMResponse(content="default"),
        )
        r = f.chat_completion([{"role": "user", "content": "hotel"}])
        assert r["content"] == "default"  # lowercase doesn't match


# ===================================================================
# OpenAI client interface
# ===================================================================


class TestOpenAIInterface:
    def test_create_returns_response(self):
        f = FakeLLM(responses=[FakeLLMResponse(content="hi")])
        resp = f.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "x"}]
        )
        assert resp.choices[0].message.content == "hi"

    def test_content(self):
        f = FakeLLM(responses=[FakeLLMResponse(content="hello")])
        resp = f.chat.completions.create(messages=[{"role": "user", "content": "x"}])
        assert resp.choices[0].message.content == "hello"

    def test_tool_calls(self):
        tc = [{"name": "f", "arguments": '{"a": 1}', "id": "tc1"}]
        f = FakeLLM(responses=[FakeLLMResponse(tool_calls=tc, finish_reason="tool_calls")])
        resp = f.chat.completions.create(messages=[{"role": "user", "content": "x"}])
        assert resp.choices[0].message.tool_calls is not None
        assert resp.choices[0].message.tool_calls[0].function.name == "f"
        assert resp.choices[0].message.tool_calls[0].function.arguments == '{"a": 1}'
        assert resp.choices[0].message.tool_calls[0].id == "tc1"

    def test_finish_reason(self):
        f = FakeLLM(responses=[FakeLLMResponse(finish_reason="length")])
        resp = f.chat.completions.create(messages=[{"role": "user", "content": "x"}])
        assert resp.choices[0].finish_reason == "length"

    def test_usage(self):
        f = FakeLLM(responses=[FakeLLMResponse(content="x")])
        resp = f.chat.completions.create(messages=[{"role": "user", "content": "x"}])
        assert resp.usage.prompt_tokens == 10
        assert resp.usage.completion_tokens == 5
        assert resp.usage.total_tokens == 15


# ===================================================================
# LLMClient interface
# ===================================================================


class TestLLMClientInterface:
    def test_returns_dict(self):
        f = FakeLLM(responses=[FakeLLMResponse(content="test")])
        r = f.chat_completion([{"role": "user", "content": "x"}])
        assert isinstance(r, dict)

    def test_dict_keys(self):
        f = FakeLLM(responses=[FakeLLMResponse(content="test")])
        r = f.chat_completion([{"role": "user", "content": "x"}])
        assert set(r.keys()) == {"content", "tool_calls", "finish_reason", "usage"}

    def test_with_functions(self):
        f = FakeLLM(responses=[FakeLLMResponse(content="ok")])
        r = f.chat_completion_with_functions(
            [{"role": "user", "content": "x"}],
            functions=[{"name": "f", "parameters": {}}],
        )
        assert r["content"] == "ok"
        # Check tools were passed
        last = f.last_call()
        assert last is not None
        assert last["tools"] is not None


# ===================================================================
# Call tracking
# ===================================================================


class TestCallTracking:
    def test_count_increments(self):
        f = FakeLLM()
        f.chat_completion([{"role": "user", "content": "a"}])
        f.chat_completion([{"role": "user", "content": "b"}])
        assert f.call_count == 2

    def test_history_records(self):
        f = FakeLLM()
        f.chat_completion([{"role": "user", "content": "test"}])
        assert len(f.call_history) == 1
        assert f.call_history[0]["messages"][0]["content"] == "test"

    def test_last_call(self):
        f = FakeLLM()
        f.chat_completion([{"role": "user", "content": "first"}])
        f.chat_completion([{"role": "user", "content": "second"}])
        lc = f.last_call()
        assert lc is not None
        assert lc["messages"][0]["content"] == "second"

    def test_reset_clears(self):
        f = FakeLLM()
        f.chat_completion([{"role": "user", "content": "x"}])
        f.reset()
        assert f.call_count == 0
        assert f.last_call() is None


# ===================================================================
# Pre-built fixtures
# ===================================================================


class TestFixtures:
    def test_user_request(self):
        assert isinstance(USER_REQUEST_RESPONSE, FakeLLMResponse)
        assert USER_REQUEST_RESPONSE.content is not None

    def test_tool_call(self):
        assert isinstance(ASSISTANT_TOOL_CALL_RESPONSE, FakeLLMResponse)
        assert ASSISTANT_TOOL_CALL_RESPONSE.tool_calls is not None
        assert ASSISTANT_TOOL_CALL_RESPONSE.tool_calls[0]["name"] == "hotels/search"

    def test_judge_parseable(self):
        assert isinstance(JUDGE_RESPONSE, FakeLLMResponse)
        assert JUDGE_RESPONSE.content is not None
        scores = json.loads(JUDGE_RESPONSE.content)
        assert scores["tool_correctness"] == 4

    def test_all_are_responses(self):
        for fix in (
            USER_REQUEST_RESPONSE,
            ASSISTANT_TOOL_CALL_RESPONSE,
            DISAMBIGUATION_RESPONSE,
            JUDGE_RESPONSE,
            FINAL_ANSWER_RESPONSE,
            GENERIC_RESPONSE,
        ):
            assert isinstance(fix, FakeLLMResponse)
