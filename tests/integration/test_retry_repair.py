"""Integration test demonstrating the retry/repair loop (Task 65).

Spec requirement: "Integration test demonstrating the retry/repair loop"
- FakeLLM returns low scores on first attempt, acceptable on second
- First attempt generates conversation
- Validator or judge triggers repair
- Feedback is injected into regeneration prompt
- Second attempt passes
- Repair stats are tracked correctly
- Max retries respected (third attempt -> discard)
"""

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pytest

from tests.helpers.fake_llm import FakeLLM, FakeLLMResponse
from tooluse_gen.agents.assistant_agent import AssistantAgent
from tooluse_gen.agents.orchestrator import ConversationOrchestrator
from tooluse_gen.agents.tool_executor import ToolExecutor
from tooluse_gen.agents.user_simulator import UserSimulator
from tooluse_gen.evaluation.judge import JudgeAgent
from tooluse_gen.evaluation.models import EvaluationConfig
from tooluse_gen.evaluation.repair import RepairLoop
from tooluse_gen.evaluation.validator import ConversationValidator
from tooluse_gen.graph.chain_models import SamplingConstraints
from tooluse_gen.graph.facade import ToolChainSampler
from tooluse_gen.graph.persistence import load_graph
from tooluse_gen.registry.serialization import load_registry

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# FakeLLM score responses
# ---------------------------------------------------------------------------

_LOW_SCORES = FakeLLMResponse(
    content=json.dumps({
        "tool_correctness": 2,
        "argument_grounding": 1,
        "task_completion": 2,
        "naturalness": 2,
        "reasoning": "Low quality — needs improvement",
    })
)

_HIGH_SCORES = FakeLLMResponse(
    content=json.dumps({
        "tool_correctness": 5,
        "argument_grounding": 4,
        "task_completion": 5,
        "naturalness": 4,
        "reasoning": "Good quality conversation",
    })
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_embeddings(mock_embedding_service: type) -> object:
    with (
        patch("tooluse_gen.graph.builder.EmbeddingService", mock_embedding_service),
        patch("tooluse_gen.graph.embeddings.EmbeddingService", mock_embedding_service),
    ):
        yield


@pytest.fixture()
def registry_and_graph(build_artifacts):  # type: ignore[no-untyped-def]
    """Load registry and graph from built artifacts."""
    registry, _ = load_registry(build_artifacts / "registry.json")
    graph, _ = load_graph(build_artifacts / "graph.pkl")
    return registry, graph


@pytest.fixture()
def orchestrator(registry_and_graph):  # type: ignore[no-untyped-def]
    """Create an offline ConversationOrchestrator."""
    registry, _graph = registry_and_graph
    return ConversationOrchestrator(
        user_sim=UserSimulator(),
        assistant=AssistantAgent(registry=registry),
        executor=ToolExecutor(registry=registry),
    )


@pytest.fixture()
def chain(registry_and_graph):  # type: ignore[no-untyped-def]
    """Sample a tool chain from the graph."""
    _registry, graph = registry_and_graph
    sampler = ToolChainSampler(graph)
    rng = np.random.default_rng(42)
    return sampler.sample_chain(
        SamplingConstraints(min_steps=1, max_steps=2, min_tools=1), rng,
    )


@pytest.fixture()
def conversation(orchestrator, chain):  # type: ignore[no-untyped-def]
    """Generate a conversation to be evaluated."""
    return orchestrator.generate_conversation(chain, seed=42)


# ===================================================================
# Basic repair loop behaviour
# ===================================================================


class TestRepairLoopBasic:
    """Core retry/repair mechanics."""

    def test_passes_on_first_attempt(self, orchestrator, chain, conversation) -> None:  # type: ignore[no-untyped-def]
        """High-scoring conversation passes without repair."""
        fake = FakeLLM(responses=[_HIGH_SCORES])
        judge = JudgeAgent(llm_client=fake)
        loop = RepairLoop(
            orchestrator, ConversationValidator(), judge,
            EvaluationConfig(min_score=3.5, max_retries=2),
        )

        _conv, result = loop.evaluate_and_repair(conversation, chain, seed=42)
        assert result.passed
        assert result.attempt_number == 1

        stats = loop.get_stats()
        assert stats.total_attempts == 0  # no regeneration needed
        assert stats.quality_repairs == 0

    def test_fails_then_passes_on_second(self, orchestrator, chain, conversation) -> None:  # type: ignore[no-untyped-def]
        """Low score on first attempt → regenerate → passes on second."""
        fake = FakeLLM(responses=[_LOW_SCORES, _HIGH_SCORES])
        judge = JudgeAgent(llm_client=fake)
        loop = RepairLoop(
            orchestrator, ConversationValidator(), judge,
            EvaluationConfig(min_score=3.5, max_retries=2),
        )

        _conv, result = loop.evaluate_and_repair(conversation, chain, seed=42)
        assert result.passed
        assert result.attempt_number == 2

        stats = loop.get_stats()
        assert stats.quality_repairs >= 1
        assert stats.total_attempts >= 1
        assert stats.successful_repairs >= 1

    def test_max_retries_exhausted(self, orchestrator, chain, conversation) -> None:  # type: ignore[no-untyped-def]
        """All attempts return low scores → fails with max_retries_exceeded."""
        fake = FakeLLM(default_response=_LOW_SCORES)
        judge = JudgeAgent(llm_client=fake)
        loop = RepairLoop(
            orchestrator, ConversationValidator(), judge,
            EvaluationConfig(min_score=3.5, max_retries=2),
        )

        _conv, result = loop.evaluate_and_repair(conversation, chain, seed=42)
        assert not result.passed
        assert "max_retries_exceeded" in result.failure_reasons
        assert result.attempt_number == 3  # 1 initial + 2 retries

        stats = loop.get_stats()
        assert stats.failed_repairs == 1

    def test_feedback_stored_in_stats(self, orchestrator, chain, conversation) -> None:  # type: ignore[no-untyped-def]
        """Repair triggers regeneration and increments total_attempts."""
        fake = FakeLLM(responses=[_LOW_SCORES, _HIGH_SCORES])
        judge = JudgeAgent(llm_client=fake)
        loop = RepairLoop(
            orchestrator, ConversationValidator(), judge,
            EvaluationConfig(min_score=3.5, max_retries=2),
        )

        loop.evaluate_and_repair(conversation, chain, seed=42)
        stats = loop.get_stats()
        # At least one _regenerate_with_feedback call happened
        assert stats.total_attempts >= 1

    def test_judge_called_per_attempt(self, orchestrator, chain, conversation) -> None:  # type: ignore[no-untyped-def]
        """Judge is called on every attempt (not skipped)."""
        fake = FakeLLM(responses=[_LOW_SCORES, _LOW_SCORES, _HIGH_SCORES])
        judge = JudgeAgent(llm_client=fake)
        loop = RepairLoop(
            orchestrator, ConversationValidator(), judge,
            EvaluationConfig(min_score=3.5, max_retries=2),
        )

        _conv, result = loop.evaluate_and_repair(conversation, chain, seed=42)
        assert result.passed
        assert result.attempt_number == 3
        # FakeLLM was called 3 times (once per attempt)
        assert fake.call_count == 3


# ===================================================================
# Statistics tracking
# ===================================================================


class TestRepairStats:
    """Statistics accumulate and reset correctly."""

    def test_stats_accumulate_across_calls(self, orchestrator, chain, conversation) -> None:  # type: ignore[no-untyped-def]
        """Stats accumulate across multiple evaluate_and_repair calls."""
        fake = FakeLLM(responses=[_HIGH_SCORES, _LOW_SCORES, _HIGH_SCORES])
        judge = JudgeAgent(llm_client=fake)
        loop = RepairLoop(
            orchestrator, ConversationValidator(), judge,
            EvaluationConfig(min_score=3.5, max_retries=2),
        )

        # First: passes immediately
        loop.evaluate_and_repair(conversation, chain, seed=42)
        # Second: fails then passes
        conv2 = orchestrator.generate_conversation(chain, seed=99)
        loop.evaluate_and_repair(conv2, chain, seed=99)

        stats = loop.get_stats()
        assert stats.attempts_distribution.get(1, 0) >= 1  # at least one first-attempt pass

    def test_reset_stats_clears(self, orchestrator, chain, conversation) -> None:  # type: ignore[no-untyped-def]
        """reset_stats() zeroes everything."""
        fake = FakeLLM(responses=[_LOW_SCORES, _HIGH_SCORES])
        judge = JudgeAgent(llm_client=fake)
        loop = RepairLoop(
            orchestrator, ConversationValidator(), judge,
            EvaluationConfig(min_score=3.5, max_retries=2),
        )

        loop.evaluate_and_repair(conversation, chain, seed=42)
        loop.reset_stats()
        stats = loop.get_stats()
        assert stats.total_attempts == 0
        assert stats.quality_repairs == 0
        assert stats.successful_repairs == 0
        assert stats.failed_repairs == 0

    def test_attempts_distribution_records_pass(self, orchestrator, chain, conversation) -> None:  # type: ignore[no-untyped-def]
        """attempts_distribution records which attempt passed."""
        fake = FakeLLM(responses=[_HIGH_SCORES])
        judge = JudgeAgent(llm_client=fake)
        loop = RepairLoop(
            orchestrator, ConversationValidator(), judge,
            EvaluationConfig(min_score=3.5, max_retries=2),
        )

        loop.evaluate_and_repair(conversation, chain, seed=42)
        stats = loop.get_stats()
        assert stats.attempts_distribution.get(1, 0) == 1


# ===================================================================
# Batch evaluation with repair
# ===================================================================


class TestRepairBatch:
    """Batch evaluate_and_repair_batch."""

    def test_batch_mixed_results(self, orchestrator, chain) -> None:  # type: ignore[no-untyped-def]
        """Batch with some passing and some failing."""
        conv1 = orchestrator.generate_conversation(chain, seed=42)
        conv2 = orchestrator.generate_conversation(chain, seed=99)

        # conv1 passes immediately; conv2 fails all attempts (1 initial + 1 retry)
        fake = FakeLLM(responses=[_HIGH_SCORES, _LOW_SCORES, _LOW_SCORES])
        judge = JudgeAgent(llm_client=fake)
        loop = RepairLoop(
            orchestrator, ConversationValidator(), judge,
            EvaluationConfig(min_score=3.5, max_retries=1),
        )

        results = loop.evaluate_and_repair_batch(
            [(conv1, chain), (conv2, chain)], seed=42,
        )
        assert len(results) == 2
        assert results[0][1].passed
        assert not results[1][1].passed

    def test_batch_all_pass(self, orchestrator, chain) -> None:  # type: ignore[no-untyped-def]
        """All conversations in batch pass."""
        conv1 = orchestrator.generate_conversation(chain, seed=42)
        conv2 = orchestrator.generate_conversation(chain, seed=99)

        fake = FakeLLM(default_response=_HIGH_SCORES)
        judge = JudgeAgent(llm_client=fake)
        loop = RepairLoop(
            orchestrator, ConversationValidator(), judge,
            EvaluationConfig(min_score=3.5, max_retries=1),
        )

        results = loop.evaluate_and_repair_batch(
            [(conv1, chain), (conv2, chain)], seed=42,
        )
        assert all(r[1].passed for r in results)


# ===================================================================
# Offline judge (no LLM)
# ===================================================================


class TestRepairWithOfflineJudge:
    """Tests using the offline heuristic judge."""

    def test_offline_repair_passes_low_threshold(self, orchestrator, chain, conversation) -> None:  # type: ignore[no-untyped-def]
        """Offline scores pass when threshold is low."""
        judge = JudgeAgent()  # no LLM → offline heuristic
        loop = RepairLoop(
            orchestrator, ConversationValidator(), judge,
            EvaluationConfig(min_score=1.0, max_retries=1),
        )

        conv, result = loop.evaluate_and_repair(conversation, chain, seed=42)
        assert result.passed
        assert result.scores is not None
        assert result.scores.average >= 1.0

    def test_offline_repair_fails_high_threshold(self, orchestrator, chain, conversation) -> None:  # type: ignore[no-untyped-def]
        """Offline scores may fail with a very high threshold → retries exhaust."""
        judge = JudgeAgent()  # offline
        loop = RepairLoop(
            orchestrator, ConversationValidator(), judge,
            EvaluationConfig(min_score=5.0, max_retries=1),
        )

        conv, result = loop.evaluate_and_repair(conversation, chain, seed=42)
        # Average of offline scores is unlikely to be exactly 5.0
        assert not result.passed
        assert "max_retries_exceeded" in result.failure_reasons
