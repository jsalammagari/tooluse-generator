"""Tests for the RepairLoop (Task 46)."""

from __future__ import annotations

import pytest

from tooluse_gen.agents.assistant_agent import AssistantAgent
from tooluse_gen.agents.conversation_models import Conversation, Message
from tooluse_gen.agents.execution_models import ToolCallRequest
from tooluse_gen.agents.orchestrator import ConversationOrchestrator, OrchestratorConfig
from tooluse_gen.agents.tool_executor import ToolExecutor
from tooluse_gen.agents.user_simulator import UserSimulator
from tooluse_gen.evaluation.judge import JudgeAgent
from tooluse_gen.evaluation.models import EvaluationConfig, EvaluationResult, JudgeScores
from tooluse_gen.evaluation.repair import RepairLoop, RepairStats
from tooluse_gen.evaluation.validator import ConversationValidator
from tooluse_gen.graph.chain_models import ChainPattern, ChainStep, ToolChain
from tooluse_gen.registry.models import Endpoint, HttpMethod, Parameter, ParameterType, Tool
from tooluse_gen.registry.registry import ToolRegistry

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.add_tool(
        Tool(
            tool_id="hotels",
            name="Hotels API",
            domain="Travel",
            endpoints=[
                Endpoint(
                    endpoint_id="hotels/search",
                    tool_id="hotels",
                    name="Search",
                    method=HttpMethod.GET,
                    path="/s",
                    parameters=[
                        Parameter(name="city", param_type=ParameterType.STRING, required=True),
                    ],
                    required_parameters=["city"],
                ),
                Endpoint(
                    endpoint_id="hotels/book",
                    tool_id="hotels",
                    name="Book",
                    method=HttpMethod.POST,
                    path="/b",
                    parameters=[
                        Parameter(name="hotel_id", param_type=ParameterType.STRING, required=True),
                        Parameter(name="guest_name", param_type=ParameterType.STRING, required=True),
                    ],
                    required_parameters=["hotel_id", "guest_name"],
                ),
            ],
        )
    )
    return reg


@pytest.fixture()
def chain() -> ToolChain:
    return ToolChain(
        chain_id="test",
        steps=[
            ChainStep(
                endpoint_id="hotels/search",
                tool_id="hotels",
                tool_name="Hotels API",
                endpoint_name="Search",
                method=HttpMethod.GET,
                domain="Travel",
                expected_params=["city"],
            ),
            ChainStep(
                endpoint_id="hotels/book",
                tool_id="hotels",
                tool_name="Hotels API",
                endpoint_name="Book",
                method=HttpMethod.POST,
                domain="Travel",
                expected_params=["hotel_id", "guest_name"],
            ),
        ],
        pattern=ChainPattern.SEQUENTIAL,
    )


@pytest.fixture()
def orchestrator(registry: ToolRegistry) -> ConversationOrchestrator:
    return ConversationOrchestrator(
        user_sim=UserSimulator(),
        assistant=AssistantAgent(registry=registry),
        executor=ToolExecutor(registry),
        config=OrchestratorConfig(require_disambiguation=False),
    )


@pytest.fixture()
def repair_loop(orchestrator: ConversationOrchestrator) -> RepairLoop:
    return RepairLoop(
        orchestrator=orchestrator,
        validator=ConversationValidator(),
        judge=JudgeAgent(),
    )


def _make_valid_conversation(
    orchestrator: ConversationOrchestrator, chain: ToolChain, seed: int = 42
) -> Conversation:
    return orchestrator.generate_conversation(chain, seed=seed)


def _make_invalid_conversation() -> Conversation:
    """Conversation with structural issues."""
    return Conversation(messages=[])


# ===================================================================
# RepairStats
# ===================================================================


class TestRepairStats:
    def test_defaults(self):
        s = RepairStats()
        assert s.total_attempts == 0
        assert s.structural_repairs == 0
        assert s.quality_repairs == 0
        assert s.successful_repairs == 0
        assert s.failed_repairs == 0
        assert s.attempts_distribution == {}

    def test_custom(self):
        s = RepairStats(
            total_attempts=5,
            structural_repairs=2,
            quality_repairs=3,
            successful_repairs=4,
            failed_repairs=1,
            attempts_distribution={1: 3, 2: 1},
        )
        assert s.total_attempts == 5
        assert s.attempts_distribution[1] == 3

    def test_serialization_round_trip(self):
        s = RepairStats(total_attempts=3, failed_repairs=1)
        data = s.model_dump()
        restored = RepairStats.model_validate(data)
        assert restored.total_attempts == 3


# ===================================================================
# Construction
# ===================================================================


class TestConstruction:
    def test_default_config(self, orchestrator):
        loop = RepairLoop(
            orchestrator=orchestrator,
            validator=ConversationValidator(),
            judge=JudgeAgent(),
        )
        assert loop._config.min_score == 3.5
        assert loop._config.max_retries == 3

    def test_custom_config(self, orchestrator):
        cfg = EvaluationConfig(min_score=4.0, max_retries=5)
        loop = RepairLoop(
            orchestrator=orchestrator,
            validator=ConversationValidator(),
            judge=JudgeAgent(),
            config=cfg,
        )
        assert loop._config.min_score == 4.0
        assert loop._config.max_retries == 5


# ===================================================================
# evaluate_and_repair — passing first try
# ===================================================================


class TestPassFirstTry:
    def test_returns_tuple(self, repair_loop, orchestrator, chain):
        conv = _make_valid_conversation(orchestrator, chain)
        result_conv, result = repair_loop.evaluate_and_repair(conv, chain)
        assert isinstance(result_conv, Conversation)
        assert isinstance(result, EvaluationResult)

    def test_passed(self, repair_loop, orchestrator, chain):
        conv = _make_valid_conversation(orchestrator, chain)
        _, result = repair_loop.evaluate_and_repair(conv, chain)
        assert result.passed

    def test_attempt_number_one(self, repair_loop, orchestrator, chain):
        conv = _make_valid_conversation(orchestrator, chain)
        _, result = repair_loop.evaluate_and_repair(conv, chain)
        assert result.attempt_number == 1


# ===================================================================
# evaluate_and_repair — structural failure
# ===================================================================


class TestStructuralFailure:
    def test_invalid_gets_regenerated(self, orchestrator, chain):
        loop = RepairLoop(
            orchestrator=orchestrator,
            validator=ConversationValidator(),
            judge=JudgeAgent(),
            config=EvaluationConfig(max_retries=2),
        )
        conv = _make_invalid_conversation()
        result_conv, result = loop.evaluate_and_repair(conv, chain)
        # Regenerated conversation should have messages
        assert result_conv.turn_count > 0

    def test_regenerated_passes(self, orchestrator, chain):
        loop = RepairLoop(
            orchestrator=orchestrator,
            validator=ConversationValidator(),
            judge=JudgeAgent(),
            config=EvaluationConfig(max_retries=2),
        )
        conv = _make_invalid_conversation()
        _, result = loop.evaluate_and_repair(conv, chain)
        assert result.passed

    def test_stats_track_structural(self, orchestrator, chain):
        loop = RepairLoop(
            orchestrator=orchestrator,
            validator=ConversationValidator(),
            judge=JudgeAgent(),
            config=EvaluationConfig(max_retries=2),
        )
        conv = _make_invalid_conversation()
        loop.evaluate_and_repair(conv, chain)
        stats = loop.get_stats()
        assert stats.structural_repairs >= 1


# ===================================================================
# evaluate_and_repair — quality failure
# ===================================================================


class TestQualityFailure:
    def test_low_quality_triggers_retry(self, orchestrator, chain):
        # Set threshold impossibly high so quality always fails
        config = EvaluationConfig(min_score=5.0, max_retries=1)
        loop = RepairLoop(
            orchestrator=orchestrator,
            validator=ConversationValidator(),
            judge=JudgeAgent(),
            config=config,
        )
        conv = _make_valid_conversation(orchestrator, chain)
        _, result = loop.evaluate_and_repair(conv, chain)
        assert not result.passed

    def test_stats_track_quality(self, orchestrator, chain):
        config = EvaluationConfig(min_score=5.0, max_retries=1)
        loop = RepairLoop(
            orchestrator=orchestrator,
            validator=ConversationValidator(),
            judge=JudgeAgent(),
            config=config,
        )
        conv = _make_valid_conversation(orchestrator, chain)
        loop.evaluate_and_repair(conv, chain)
        stats = loop.get_stats()
        assert stats.quality_repairs >= 1

    def test_different_seed_different_conv(self, orchestrator, chain):
        config = EvaluationConfig(min_score=5.0, max_retries=2)
        loop = RepairLoop(
            orchestrator=orchestrator,
            validator=ConversationValidator(),
            judge=JudgeAgent(),
            config=config,
        )
        conv = _make_valid_conversation(orchestrator, chain, seed=42)
        result_conv, _ = loop.evaluate_and_repair(conv, chain, seed=42)
        # The returned conversation was regenerated with a different seed,
        # so it should still have messages (structurally valid).
        assert result_conv.turn_count > 0


# ===================================================================
# evaluate_and_repair — max retries
# ===================================================================


class TestMaxRetries:
    def test_returns_failed(self, orchestrator, chain):
        config = EvaluationConfig(min_score=5.0, max_retries=2)
        loop = RepairLoop(
            orchestrator=orchestrator,
            validator=ConversationValidator(),
            judge=JudgeAgent(),
            config=config,
        )
        conv = _make_valid_conversation(orchestrator, chain)
        _, result = loop.evaluate_and_repair(conv, chain)
        assert not result.passed

    def test_failure_reasons(self, orchestrator, chain):
        config = EvaluationConfig(min_score=5.0, max_retries=2)
        loop = RepairLoop(
            orchestrator=orchestrator,
            validator=ConversationValidator(),
            judge=JudgeAgent(),
            config=config,
        )
        conv = _make_valid_conversation(orchestrator, chain)
        _, result = loop.evaluate_and_repair(conv, chain)
        assert "max_retries_exceeded" in result.failure_reasons

    def test_stats_track_failed(self, orchestrator, chain):
        config = EvaluationConfig(min_score=5.0, max_retries=1)
        loop = RepairLoop(
            orchestrator=orchestrator,
            validator=ConversationValidator(),
            judge=JudgeAgent(),
            config=config,
        )
        conv = _make_valid_conversation(orchestrator, chain)
        loop.evaluate_and_repair(conv, chain)
        assert loop.get_stats().failed_repairs >= 1


# ===================================================================
# _regenerate_with_feedback
# ===================================================================


class TestRegenerateWithFeedback:
    def test_returns_conversation(self, repair_loop, chain):
        conv = repair_loop._regenerate_with_feedback(chain, seed=99, feedback=["test"])
        assert isinstance(conv, Conversation)
        assert conv.turn_count > 0

    def test_different_seed_different_result(self, repair_loop, chain):
        c1 = repair_loop._regenerate_with_feedback(chain, seed=10, feedback=[])
        c2 = repair_loop._regenerate_with_feedback(chain, seed=20, feedback=[])
        # Both valid, possibly different content
        assert c1.turn_count > 0
        assert c2.turn_count > 0

    def test_increments_total_attempts(self, repair_loop, chain):
        before = repair_loop.get_stats().total_attempts
        repair_loop._regenerate_with_feedback(chain, seed=1, feedback=["x"])
        after = repair_loop.get_stats().total_attempts
        assert after == before + 1


# ===================================================================
# _identify_problematic_turn
# ===================================================================


class TestIdentifyProblematicTurn:
    def _make_conv(self) -> Conversation:
        tc = ToolCallRequest(
            endpoint_id="x", tool_id="x", tool_name="X", endpoint_name="X"
        )
        return Conversation(
            messages=[
                Message(role="user", content="hi"),
                Message(role="assistant", tool_calls=[tc]),
                Message(role="tool", tool_output={"id": "1"}),
                Message(role="assistant", content="done"),
            ]
        )

    def test_low_naturalness(self, repair_loop):
        conv = self._make_conv()
        idx = repair_loop._identify_problematic_turn(
            conv, JudgeScores(naturalness=1, tool_correctness=4, argument_grounding=4, task_completion=4)
        )
        assert idx == 0  # first user

    def test_low_grounding(self, repair_loop):
        conv = self._make_conv()
        idx = repair_loop._identify_problematic_turn(
            conv, JudgeScores(argument_grounding=1, tool_correctness=4, naturalness=4, task_completion=4)
        )
        assert idx == 1  # first assistant tc

    def test_low_completion(self, repair_loop):
        conv = self._make_conv()
        idx = repair_loop._identify_problematic_turn(
            conv, JudgeScores(task_completion=1, tool_correctness=4, naturalness=4, argument_grounding=4)
        )
        assert idx == 3  # last message

    def test_all_good(self, repair_loop):
        conv = self._make_conv()
        idx = repair_loop._identify_problematic_turn(
            conv, JudgeScores(tool_correctness=4, argument_grounding=4, task_completion=4, naturalness=4)
        )
        assert idx is None


# ===================================================================
# evaluate_and_repair_batch
# ===================================================================


class TestBatch:
    def test_returns_list(self, repair_loop, orchestrator, chain):
        conv = _make_valid_conversation(orchestrator, chain)
        results = repair_loop.evaluate_and_repair_batch([(conv, chain), (conv, chain)])
        assert isinstance(results, list)

    def test_length_matches(self, repair_loop, orchestrator, chain):
        conv = _make_valid_conversation(orchestrator, chain)
        results = repair_loop.evaluate_and_repair_batch([(conv, chain)] * 3)
        assert len(results) == 3

    def test_each_has_conversation_id(self, repair_loop, orchestrator, chain):
        conv = _make_valid_conversation(orchestrator, chain)
        results = repair_loop.evaluate_and_repair_batch([(conv, chain)])
        for result_conv, result in results:
            assert result.conversation_id == result_conv.conversation_id


# ===================================================================
# Stats tracking
# ===================================================================


class TestStatsTracking:
    def test_get_stats(self, repair_loop):
        stats = repair_loop.get_stats()
        assert isinstance(stats, RepairStats)

    def test_reset_stats(self, repair_loop, orchestrator, chain):
        conv = _make_valid_conversation(orchestrator, chain)
        repair_loop.evaluate_and_repair(conv, chain)
        repair_loop.reset_stats()
        stats = repair_loop.get_stats()
        assert stats.total_attempts == 0
        assert stats.structural_repairs == 0
        assert stats.quality_repairs == 0
        assert stats.successful_repairs == 0
        assert stats.failed_repairs == 0

    def test_stats_accumulate(self, orchestrator, chain):
        config = EvaluationConfig(min_score=5.0, max_retries=1)
        loop = RepairLoop(
            orchestrator=orchestrator,
            validator=ConversationValidator(),
            judge=JudgeAgent(),
            config=config,
        )
        conv1 = _make_valid_conversation(orchestrator, chain, seed=42)
        conv2 = _make_valid_conversation(orchestrator, chain, seed=99)
        loop.evaluate_and_repair(conv1, chain, seed=42)
        loop.evaluate_and_repair(conv2, chain, seed=99)
        stats = loop.get_stats()
        assert stats.failed_repairs >= 2
        assert stats.quality_repairs >= 2
