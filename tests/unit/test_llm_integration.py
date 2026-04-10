"""Tests for LLM module integration (Task 57).

Exercises FakeLLM + PromptCache + LLMRecorder + agents together,
verifying the modules compose correctly without real API calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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
from tests.helpers.vcr_recorder import CassetteNotFoundError, LLMRecorder
from tooluse_gen.agents.conversation_models import Conversation, Message
from tooluse_gen.core import (
    LLMClient,
    LLMClientConfig,
    LLMClientError,
    PromptCache,
)
from tooluse_gen.evaluation.judge import JudgeAgent

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MSGS = [{"role": "user", "content": "hello"}]
_MODEL = "gpt-4o"


def _make_conversation(msg_count: int = 4) -> Conversation:
    """Build a minimal conversation for judge scoring."""
    msgs = [
        Message(role="user", content="Find me a hotel in Paris."),
        Message(
            role="assistant",
            content=None,
            tool_calls=None,
        ),
        Message(role="assistant", content="I found Hotel du Marais for $120/night."),
    ]
    if msg_count >= 4:
        msgs.append(Message(role="user", content="Book it please."))
    return Conversation(messages=msgs)


# ===================================================================
# Import completeness
# ===================================================================


class TestImportCompleteness:
    """Verify all LLM-related symbols are importable."""

    def test_core_llm_symbols(self):
        assert LLMClient is not None
        assert LLMClientConfig is not None
        assert LLMClientError is not None
        assert PromptCache is not None

    def test_fake_llm_symbols(self):
        assert FakeLLM is not None
        assert FakeLLMResponse is not None

    def test_vcr_symbols(self):
        assert LLMRecorder is not None
        assert CassetteNotFoundError is not None

    def test_fixture_symbols(self):
        for fixture in (
            USER_REQUEST_RESPONSE,
            ASSISTANT_TOOL_CALL_RESPONSE,
            DISAMBIGUATION_RESPONSE,
            JUDGE_RESPONSE,
            FINAL_ANSWER_RESPONSE,
            GENERIC_RESPONSE,
        ):
            assert isinstance(fixture, FakeLLMResponse)

    def test_core_all_has_llm_exports(self):
        from tooluse_gen.core import __all__

        for name in ("LLMClient", "LLMClientConfig", "LLMClientError", "PromptCache"):
            assert name in __all__


# ===================================================================
# LLMClient + PromptCache integration
# ===================================================================


class TestLLMClientCacheIntegration:
    """Test PromptCache with LLMClient-style dict responses."""

    def test_cache_miss_returns_none(self, tmp_path: Path):
        cache = PromptCache(tmp_path)
        h = PromptCache.hash_prompt(_MSGS, _MODEL)
        assert cache.get(h) is None

    def test_cache_put_get_roundtrip(self, tmp_path: Path):
        cache = PromptCache(tmp_path)
        h = PromptCache.hash_prompt(_MSGS, _MODEL)
        response = {"content": "cached", "tool_calls": None, "finish_reason": "stop"}
        cache.put(h, response)
        assert cache.get(h) == response

    def test_cache_stats_track_hits_misses(self, tmp_path: Path):
        cache = PromptCache(tmp_path)
        h = PromptCache.hash_prompt(_MSGS, _MODEL)
        cache.get(h)  # miss
        cache.put(h, {"content": "x"})
        cache.get(h)  # hit
        s = cache.stats()
        assert s["hits"] == 1
        assert s["misses"] == 1

    def test_cache_disabled_skips(self, tmp_path: Path):
        cache = PromptCache(tmp_path, enabled=False)
        h = PromptCache.hash_prompt(_MSGS, _MODEL)
        cache.put(h, {"content": "nope"})
        assert cache.get(h) is None

    def test_cache_size_increments(self, tmp_path: Path):
        cache = PromptCache(tmp_path)
        assert cache.size == 0
        h1 = PromptCache.hash_prompt([{"role": "user", "content": "a"}], "m")
        h2 = PromptCache.hash_prompt([{"role": "user", "content": "b"}], "m")
        cache.put(h1, {"content": "a"})
        cache.put(h2, {"content": "b"})
        assert cache.size == 2

    def test_cache_clear_resets(self, tmp_path: Path):
        cache = PromptCache(tmp_path)
        h = PromptCache.hash_prompt(_MSGS, _MODEL)
        cache.put(h, {"content": "x"})
        deleted = cache.clear()
        assert deleted == 1
        assert cache.size == 0


# ===================================================================
# FakeLLM as agent client
# ===================================================================


class TestFakeLLMAsAgentClient:
    """Test FakeLLM used as the llm_client for agents."""

    def test_judge_with_fake_llm(self):
        fake = FakeLLM(responses=[JUDGE_RESPONSE])
        judge = JudgeAgent(llm_client=fake, model="gpt-4o")
        conv = _make_conversation()
        scores = judge.score(conv)
        assert scores.tool_correctness == 4
        assert scores.argument_grounding == 3
        assert scores.task_completion == 5
        assert scores.naturalness == 4

    def test_judge_batch_with_fake_llm(self):
        fake = FakeLLM(responses=[
            JUDGE_RESPONSE,
            JUDGE_RESPONSE,
        ])
        judge = JudgeAgent(llm_client=fake, model="gpt-4o")
        convs = [_make_conversation(), _make_conversation()]
        scores = judge.score_batch(convs)
        assert len(scores) == 2
        assert all(s.tool_correctness == 4 for s in scores)

    def test_judge_aggregate(self):
        from tooluse_gen.evaluation.models import JudgeScores as EvalJudgeScores

        s1 = EvalJudgeScores(
            tool_correctness=4, argument_grounding=3,
            task_completion=5, naturalness=4,
        )
        s2 = EvalJudgeScores(
            tool_correctness=2, argument_grounding=3,
            task_completion=3, naturalness=4,
        )
        judge = JudgeAgent()
        agg = judge.aggregate_scores([s1, s2])
        assert agg.tool_correctness == 3  # round((4+2)/2) = 3
        assert agg.task_completion == 4  # round((5+3)/2) = 4

    def test_fake_llm_pattern_for_judge(self):
        """Pattern matching routes different prompts to different responses."""
        fake = FakeLLM(
            pattern_responses={
                "hotel": FakeLLMResponse(content="Hotel response"),
                "weather": FakeLLMResponse(content="Weather response"),
            },
            default_response=JUDGE_RESPONSE,
        )
        # Non-matching input falls back to JUDGE_RESPONSE (default)
        r = fake.chat_completion([{"role": "user", "content": "evaluate this"}])
        assert "tool_correctness" in r["content"]

    def test_fake_llm_sequential_for_multi_turn(self):
        """Sequential responses simulate multi-turn conversations."""
        fake = FakeLLM(responses=[
            USER_REQUEST_RESPONSE,
            ASSISTANT_TOOL_CALL_RESPONSE,
            FINAL_ANSWER_RESPONSE,
        ])
        r1 = fake.chat_completion([{"role": "user", "content": "start"}])
        assert "hotel" in r1["content"].lower() or "book" in r1["content"].lower()

        r2 = fake.chat_completion([{"role": "user", "content": "next"}])
        assert r2["tool_calls"] is not None

        r3 = fake.chat_completion([{"role": "user", "content": "done"}])
        assert "booked" in r3["content"].lower() or "confirmation" in r3["content"].lower()

    def test_fake_llm_call_tracking(self):
        fake = FakeLLM(responses=[GENERIC_RESPONSE, GENERIC_RESPONSE])
        fake.chat_completion(_MSGS)
        fake.chat_completion([{"role": "user", "content": "second"}])
        assert fake.call_count == 2
        assert len(fake.call_history) == 2
        assert fake.last_call()["messages"][0]["content"] == "second"

    def test_fake_llm_openai_interface_for_judge(self):
        """JudgeAgent uses chat.completions.create — verify it works with FakeLLM."""
        fake = FakeLLM(responses=[JUDGE_RESPONSE])
        resp = fake.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "rubric"}, {"role": "user", "content": "conv"}],
            temperature=0.3,
            max_tokens=500,
        )
        content = resp.choices[0].message.content
        parsed = json.loads(content)
        assert parsed["tool_correctness"] == 4


# ===================================================================
# LLMRecorder as agent client
# ===================================================================


class TestLLMRecorderAsAgentClient:
    """Test LLMRecorder used with agents for deterministic replays."""

    def test_record_and_replay_judge(self, tmp_path: Path):
        # Record
        fake = FakeLLM(responses=[JUDGE_RESPONSE])
        rec = LLMRecorder(tmp_path, record=True, replay=False, client=fake)
        judge = JudgeAgent(llm_client=rec, model="gpt-4o")
        conv = _make_conversation()
        scores1 = judge.score(conv)

        # Replay (no fake client needed)
        replay = LLMRecorder(tmp_path, replay=True, record=False)
        judge2 = JudgeAgent(llm_client=replay, model="gpt-4o")
        scores2 = judge2.score(conv)

        assert scores1.tool_correctness == scores2.tool_correctness
        assert scores1.naturalness == scores2.naturalness

    def test_recorder_stats_after_agent_use(self, tmp_path: Path):
        fake = FakeLLM(responses=[JUDGE_RESPONSE, JUDGE_RESPONSE])
        rec = LLMRecorder(tmp_path, record=True, replay=True, client=fake)

        judge = JudgeAgent(llm_client=rec, model="gpt-4o")
        judge.score(_make_conversation())  # records
        judge.score(_make_conversation())  # replays (same conversation → same hash)

        stats = rec.stats()
        assert stats["records"] == 1
        assert stats["replays"] == 1
        assert stats["calls"] == 2

    def test_recorder_cassette_not_found(self, tmp_path: Path):
        rec = LLMRecorder(tmp_path, replay=True, record=False)
        with pytest.raises(CassetteNotFoundError):
            rec.chat_completion(_MSGS, model=_MODEL)

    def test_recorder_openai_interface(self, tmp_path: Path):
        fake = FakeLLM(responses=[FakeLLMResponse(content="via recorder")])
        rec = LLMRecorder(tmp_path, record=True, replay=True, client=fake)
        resp = rec.chat.completions.create(
            model="gpt-4o",
            messages=_MSGS,
        )
        assert resp.choices[0].message.content == "via recorder"


# ===================================================================
# Cache + FakeLLM pipeline
# ===================================================================


class TestCacheFakeLLMPipeline:
    """Test PromptCache used alongside FakeLLM in a manual pipeline."""

    def test_manual_cache_pipeline(self, tmp_path: Path):
        """Simulate: check cache → miss → call FakeLLM → populate cache → hit."""
        cache = PromptCache(tmp_path)
        fake = FakeLLM(responses=[FakeLLMResponse(content="fresh")])

        h = PromptCache.hash_prompt(_MSGS, _MODEL)

        # Miss
        cached = cache.get(h)
        assert cached is None

        # Call FakeLLM
        response = fake.chat_completion(_MSGS)
        cache.put(h, response)

        # Hit
        cached = cache.get(h)
        assert cached is not None
        assert cached["content"] == "fresh"

    def test_cache_ttl_expiry(self, tmp_path: Path):
        cache = PromptCache(tmp_path, ttl_seconds=0)  # immediate expiry
        h = PromptCache.hash_prompt(_MSGS, _MODEL)
        cache.put(h, {"content": "expired"})
        # With TTL=0, entry should be treated as expired
        result = cache.get(h)
        assert result is None

    def test_shared_cache_across_instances(self, tmp_path: Path):
        """Two PromptCache instances sharing a directory see same data."""
        cache1 = PromptCache(tmp_path)
        cache2 = PromptCache(tmp_path)

        h = PromptCache.hash_prompt(_MSGS, _MODEL)
        cache1.put(h, {"content": "shared"})
        assert cache2.get(h) == {"content": "shared"}

    def test_different_prompts_different_hashes(self):
        h1 = PromptCache.hash_prompt([{"role": "user", "content": "a"}], "m")
        h2 = PromptCache.hash_prompt([{"role": "user", "content": "b"}], "m")
        assert h1 != h2

    def test_same_prompt_same_hash(self):
        h1 = PromptCache.hash_prompt(_MSGS, _MODEL)
        h2 = PromptCache.hash_prompt(_MSGS, _MODEL)
        assert h1 == h2

    def test_hash_includes_model(self):
        h1 = PromptCache.hash_prompt(_MSGS, "gpt-4o")
        h2 = PromptCache.hash_prompt(_MSGS, "gpt-3.5-turbo")
        assert h1 != h2


# ===================================================================
# Full pipeline integration
# ===================================================================


class TestFullPipelineIntegration:
    """End-to-end: FakeLLM → JudgeAgent → scores, then cache + replay."""

    def test_judge_score_cache_replay(self, tmp_path: Path):
        """Score with FakeLLM, cache result, verify cache hit returns same."""
        fake = FakeLLM(responses=[JUDGE_RESPONSE])
        judge = JudgeAgent(llm_client=fake, model="gpt-4o")
        conv = _make_conversation()

        scores = judge.score(conv)
        assert scores.tool_correctness == 4

        # Cache the scores dict
        cache = PromptCache(tmp_path)
        h = "judge_result_1"
        cache.put(h, scores.model_dump())

        # Retrieve and verify
        cached = cache.get(h)
        assert cached is not None
        assert cached["tool_correctness"] == 4

    def test_recorder_with_multiple_judge_calls(self, tmp_path: Path):
        """Multiple different conversations recorded and replayed."""
        conv1 = _make_conversation(msg_count=3)
        conv2 = Conversation(messages=[
            Message(role="user", content="Different question about weather"),
            Message(role="assistant", content="Let me check the weather for you."),
        ])

        judge_resp1 = FakeLLMResponse(
            content='{"tool_correctness": 5, "argument_grounding": 4,'
            ' "task_completion": 5, "naturalness": 5,'
            ' "reasoning": "Excellent"}'
        )
        judge_resp2 = FakeLLMResponse(
            content='{"tool_correctness": 2, "argument_grounding": 2,'
            ' "task_completion": 1, "naturalness": 3,'
            ' "reasoning": "Poor"}'
        )

        fake = FakeLLM(responses=[judge_resp1, judge_resp2])
        rec = LLMRecorder(tmp_path, record=True, replay=False, client=fake)
        judge = JudgeAgent(llm_client=rec, model="gpt-4o")

        s1 = judge.score(conv1)
        s2 = judge.score(conv2)
        assert s1.tool_correctness == 5
        assert s2.tool_correctness == 2
        assert rec.record_count == 2

    def test_offline_judge_no_llm_needed(self):
        """JudgeAgent works in offline mode without any LLM client."""
        judge = JudgeAgent()  # no llm_client
        conv = _make_conversation()
        scores = judge.score(conv)
        # Offline heuristics produce valid scores
        assert 1 <= scores.tool_correctness <= 5
        assert 1 <= scores.naturalness <= 5

    def test_fake_llm_reset_between_tests(self):
        """Verify FakeLLM reset works for test isolation."""
        fake = FakeLLM(responses=[
            FakeLLMResponse(content="first"),
            FakeLLMResponse(content="second"),
        ])
        fake.chat_completion(_MSGS)
        assert fake.call_count == 1

        fake.reset()
        assert fake.call_count == 0
        r = fake.chat_completion(_MSGS)
        assert r["content"] == "first"  # index reset to 0


# ===================================================================
# Error handling
# ===================================================================


class TestErrorHandling:
    def test_llm_client_error_creation(self):
        e = LLMClientError("timeout", retries_attempted=3)
        assert "timeout" in str(e)
        assert e.retries_attempted == 3

    def test_llm_client_error_original_error(self):
        orig = ValueError("bad value")
        e = LLMClientError("wrapped", original_error=orig, retries_attempted=1)
        assert e.original_error is orig

    def test_cassette_not_found_error(self):
        e = CassetteNotFoundError("hash abc123")
        assert "abc123" in str(e)

    def test_llm_client_not_available_without_key(self):
        client = LLMClient(api_key=None)
        assert not client.is_available

    def test_llm_client_config_defaults(self):
        cfg = LLMClientConfig()
        assert cfg.model == "gpt-4o"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 1000
        assert cfg.max_retries == 3


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    def test_empty_conversation_judge(self):
        """Judge handles empty conversation."""
        judge = JudgeAgent()
        conv = Conversation(messages=[])
        scores = judge.score(conv)
        assert 1 <= scores.tool_correctness <= 5

    def test_fake_llm_tool_calls_in_dict_response(self):
        tc = [{"name": "search", "arguments": '{"q": "paris"}', "id": "c1"}]
        fake = FakeLLM(responses=[FakeLLMResponse(tool_calls=tc, finish_reason="tool_calls")])
        r = fake.chat_completion(_MSGS)
        assert r["tool_calls"] is not None
        assert r["tool_calls"][0]["name"] == "search"
        assert r["finish_reason"] == "tool_calls"

    def test_recorder_hybrid_mode(self, tmp_path: Path):
        """Hybrid: replay on hit, record on miss."""
        fake = FakeLLM(responses=[
            FakeLLMResponse(content="recorded_a"),
            FakeLLMResponse(content="recorded_b"),
        ])
        rec = LLMRecorder(tmp_path, record=True, replay=True, client=fake)

        # First call: miss → record
        r1 = rec.chat_completion(_MSGS, model=_MODEL)
        assert r1["content"] == "recorded_a"
        assert rec.record_count == 1

        # Same call: hit → replay
        r2 = rec.chat_completion(_MSGS, model=_MODEL)
        assert r2["content"] == "recorded_a"
        assert rec.replay_count == 1

        # Different call: miss → record
        r3 = rec.chat_completion([{"role": "user", "content": "new"}], model=_MODEL)
        assert r3["content"] == "recorded_b"
        assert rec.record_count == 2

    def test_cache_hash_deterministic(self):
        """Same inputs always produce the same hash."""
        msgs = [{"role": "user", "content": "deterministic test"}]
        hashes = [PromptCache.hash_prompt(msgs, "gpt-4o") for _ in range(10)]
        assert len(set(hashes)) == 1

    def test_cache_hash_length(self):
        h = PromptCache.hash_prompt(_MSGS, _MODEL)
        assert len(h) == 16  # first 16 hex chars of SHA-256

    def test_judge_parse_malformed_json(self):
        """Judge handles malformed JSON from LLM gracefully."""
        fake = FakeLLM(responses=[FakeLLMResponse(content="not valid json at all")])
        judge = JudgeAgent(llm_client=fake, model="gpt-4o")
        conv = _make_conversation()
        scores = judge.score(conv)
        # Should return default scores with reasoning about parse failure
        assert scores.reasoning is not None

    def test_fake_llm_default_response(self):
        """Exhausted responses fall back to default."""
        fake = FakeLLM(
            responses=[FakeLLMResponse(content="only")],
            default_response=FakeLLMResponse(content="fallback"),
        )
        fake.chat_completion(_MSGS)  # consumes "only"
        r = fake.chat_completion(_MSGS)
        assert r["content"] == "fallback"
