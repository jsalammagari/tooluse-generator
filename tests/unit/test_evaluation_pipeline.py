"""Tests for the EvaluationPipeline (Task 47)."""

from __future__ import annotations

import pytest

from tooluse_gen.agents.assistant_agent import AssistantAgent
from tooluse_gen.agents.conversation_models import Conversation, Message
from tooluse_gen.agents.execution_models import ToolCallRequest
from tooluse_gen.agents.orchestrator import ConversationOrchestrator, OrchestratorConfig
from tooluse_gen.agents.tool_executor import ToolExecutor
from tooluse_gen.agents.user_simulator import UserSimulator
from tooluse_gen.evaluation.judge import JudgeAgent
from tooluse_gen.evaluation.models import EvaluationConfig, EvaluationReport, EvaluationResult
from tooluse_gen.evaluation.pipeline import EvaluationPipeline
from tooluse_gen.evaluation.repair import RepairLoop
from tooluse_gen.evaluation.validator import ConversationValidator
from tooluse_gen.graph.chain_models import ChainPattern, ChainStep, ToolChain
from tooluse_gen.registry.models import Endpoint, HttpMethod, Parameter, ParameterType, Tool
from tooluse_gen.registry.registry import ToolRegistry

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------


def _valid_conversation() -> Conversation:
    tc = ToolCallRequest(
        endpoint_id="hotels/search", tool_id="hotels",
        tool_name="Hotels API", endpoint_name="Search",
        arguments={"city": "Paris"},
    )
    return Conversation(messages=[
        Message(role="user", content="Find me a hotel"),
        Message(role="assistant", tool_calls=[tc]),
        Message(role="tool", tool_call_id=tc.call_id, tool_output={"id": "htl_1"}),
        Message(role="assistant", content="Found hotel htl_1."),
    ])


@pytest.fixture()
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.add_tool(Tool(
        tool_id="hotels", name="Hotels API", domain="Travel",
        endpoints=[
            Endpoint(endpoint_id="hotels/search", tool_id="hotels", name="Search",
                     method=HttpMethod.GET, path="/s",
                     parameters=[Parameter(name="city", param_type=ParameterType.STRING, required=True)],
                     required_parameters=["city"]),
            Endpoint(endpoint_id="hotels/book", tool_id="hotels", name="Book",
                     method=HttpMethod.POST, path="/b",
                     parameters=[
                         Parameter(name="hotel_id", param_type=ParameterType.STRING, required=True),
                         Parameter(name="guest_name", param_type=ParameterType.STRING, required=True),
                     ], required_parameters=["hotel_id", "guest_name"]),
        ],
    ))
    return reg


@pytest.fixture()
def chain() -> ToolChain:
    return ToolChain(
        chain_id="test",
        steps=[
            ChainStep(endpoint_id="hotels/search", tool_id="hotels",
                      tool_name="Hotels API", endpoint_name="Search",
                      method=HttpMethod.GET, domain="Travel", expected_params=["city"]),
            ChainStep(endpoint_id="hotels/book", tool_id="hotels",
                      tool_name="Hotels API", endpoint_name="Book",
                      method=HttpMethod.POST, domain="Travel",
                      expected_params=["hotel_id", "guest_name"]),
        ],
        pattern=ChainPattern.SEQUENTIAL,
    )


@pytest.fixture()
def orchestrator(registry: ToolRegistry) -> ConversationOrchestrator:
    return ConversationOrchestrator(
        user_sim=UserSimulator(), assistant=AssistantAgent(registry=registry),
        executor=ToolExecutor(registry),
        config=OrchestratorConfig(require_disambiguation=False),
    )


@pytest.fixture()
def pipeline() -> EvaluationPipeline:
    return EvaluationPipeline(
        validator=ConversationValidator(), judge=JudgeAgent(),
        config=EvaluationConfig(min_score=3.0),
    )


@pytest.fixture()
def pipeline_with_repair(
    orchestrator: ConversationOrchestrator,
) -> EvaluationPipeline:
    repair = RepairLoop(
        orchestrator=orchestrator,
        validator=ConversationValidator(),
        judge=JudgeAgent(),
    )
    return EvaluationPipeline(
        validator=ConversationValidator(), judge=JudgeAgent(),
        repair_loop=repair,
    )


# ===================================================================
# Construction
# ===================================================================


class TestConstruction:
    def test_without_repair(self):
        p = EvaluationPipeline(
            validator=ConversationValidator(), judge=JudgeAgent(),
        )
        assert p._repair_loop is None

    def test_with_repair(self, orchestrator):
        repair = RepairLoop(
            orchestrator=orchestrator, validator=ConversationValidator(),
            judge=JudgeAgent(),
        )
        p = EvaluationPipeline(
            validator=ConversationValidator(), judge=JudgeAgent(),
            repair_loop=repair,
        )
        assert p._repair_loop is not None

    def test_custom_config(self):
        cfg = EvaluationConfig(min_score=4.0, max_retries=5)
        p = EvaluationPipeline(
            validator=ConversationValidator(), judge=JudgeAgent(),
            config=cfg,
        )
        assert p._config.min_score == 4.0


# ===================================================================
# evaluate_single — passing
# ===================================================================


class TestEvaluateSinglePassing:
    def test_valid_passes(self, pipeline):
        result = pipeline.evaluate_single(_valid_conversation())
        assert result.passed

    def test_has_scores(self, pipeline):
        result = pipeline.evaluate_single(_valid_conversation())
        assert result.scores is not None

    def test_conversation_id(self, pipeline):
        conv = _valid_conversation()
        result = pipeline.evaluate_single(conv)
        assert result.conversation_id == conv.conversation_id

    def test_result_type(self, pipeline):
        result = pipeline.evaluate_single(_valid_conversation())
        assert isinstance(result, EvaluationResult)


# ===================================================================
# evaluate_single — structural failure
# ===================================================================


class TestStructuralFailure:
    def test_empty_fails(self, pipeline):
        result = pipeline.evaluate_single(Conversation(messages=[]))
        assert not result.passed

    def test_failure_reason(self, pipeline):
        result = pipeline.evaluate_single(Conversation(messages=[]))
        assert any("structural" in r for r in result.failure_reasons)

    def test_with_repair_gets_fixed(self, pipeline_with_repair, chain):
        result = pipeline_with_repair.evaluate_single(
            Conversation(messages=[]), chain=chain, seed=42,
        )
        assert result.passed

    def test_without_repair_stays_failed(self, pipeline):
        result = pipeline.evaluate_single(Conversation(messages=[]))
        assert not result.passed
        assert result.scores is None


# ===================================================================
# evaluate_single — quality failure
# ===================================================================


class TestQualityFailure:
    def test_high_threshold_fails(self):
        pipe = EvaluationPipeline(
            validator=ConversationValidator(), judge=JudgeAgent(),
            config=EvaluationConfig(min_score=5.0),
        )
        result = pipe.evaluate_single(_valid_conversation())
        assert not result.passed

    def test_failure_reason_quality(self):
        pipe = EvaluationPipeline(
            validator=ConversationValidator(), judge=JudgeAgent(),
            config=EvaluationConfig(min_score=5.0),
        )
        result = pipe.evaluate_single(_valid_conversation())
        assert any("quality_below_threshold" in r for r in result.failure_reasons)

    def test_with_repair_triggers(self, orchestrator, chain):
        repair = RepairLoop(
            orchestrator=orchestrator, validator=ConversationValidator(),
            judge=JudgeAgent(), config=EvaluationConfig(min_score=5.0, max_retries=1),
        )
        pipe = EvaluationPipeline(
            validator=ConversationValidator(), judge=JudgeAgent(),
            repair_loop=repair, config=EvaluationConfig(min_score=5.0),
        )
        conv = orchestrator.generate_conversation(chain, seed=42)
        result = pipe.evaluate_single(conv, chain=chain, seed=42)
        # Repair tried but threshold is impossible → still fails
        assert not result.passed

    def test_without_repair_stays_failed(self):
        pipe = EvaluationPipeline(
            validator=ConversationValidator(), judge=JudgeAgent(),
            config=EvaluationConfig(min_score=5.0),
        )
        result = pipe.evaluate_single(_valid_conversation())
        assert not result.passed
        assert result.scores is not None


# ===================================================================
# evaluate_batch
# ===================================================================


class TestEvaluateBatch:
    def test_returns_report(self, pipeline):
        report = pipeline.evaluate_batch([_valid_conversation()])
        assert isinstance(report, EvaluationReport)

    def test_total_matches(self, pipeline):
        convs = [_valid_conversation() for _ in range(3)]
        report = pipeline.evaluate_batch(convs)
        assert report.total == 3

    def test_passed_count(self, pipeline):
        convs = [_valid_conversation() for _ in range(3)]
        report = pipeline.evaluate_batch(convs)
        assert report.passed == 3

    def test_mixed_batch(self, pipeline):
        pipeline.reset()
        convs = [_valid_conversation(), Conversation(messages=[]), _valid_conversation()]
        report = pipeline.evaluate_batch(convs)
        assert report.total == 3
        assert report.passed >= 1
        assert report.failed >= 1

    def test_results_accumulated(self, pipeline):
        pipeline.reset()
        pipeline.evaluate_batch([_valid_conversation(), _valid_conversation()])
        assert len(pipeline.get_results()) == 2


# ===================================================================
# generate_report
# ===================================================================


class TestGenerateReport:
    def test_returns_report(self, pipeline):
        pipeline.reset()
        pipeline.evaluate_single(_valid_conversation())
        report = pipeline.generate_report()
        assert isinstance(report, EvaluationReport)

    def test_reflects_results(self, pipeline):
        pipeline.reset()
        pipeline.evaluate_single(_valid_conversation())
        pipeline.evaluate_single(_valid_conversation())
        report = pipeline.generate_report()
        assert report.total == 2
        assert report.passed == 2

    def test_score_distribution(self, pipeline):
        pipeline.reset()
        pipeline.evaluate_single(_valid_conversation())
        report = pipeline.generate_report()
        if report.average_scores is not None:
            assert len(report.score_distribution) > 0


# ===================================================================
# get_results and reset
# ===================================================================


class TestGetResultsAndReset:
    def test_get_results(self, pipeline):
        pipeline.reset()
        pipeline.evaluate_single(_valid_conversation())
        results = pipeline.get_results()
        assert len(results) == 1
        assert isinstance(results[0], EvaluationResult)

    def test_reset_clears_results(self, pipeline):
        pipeline.reset()
        pipeline.evaluate_single(_valid_conversation())
        pipeline.reset()
        assert len(pipeline.get_results()) == 0

    def test_reset_clears_repair_stats(self, pipeline_with_repair, chain):
        pipeline_with_repair.evaluate_single(
            Conversation(messages=[]), chain=chain, seed=42,
        )
        pipeline_with_repair.reset()
        assert len(pipeline_with_repair.get_results()) == 0
        stats = pipeline_with_repair._repair_loop.get_stats()  # type: ignore[union-attr]
        assert stats.total_attempts == 0


# ===================================================================
# Integration
# ===================================================================


class TestIntegration:
    def test_pipeline_generated_conversation(self, orchestrator, chain, pipeline):
        pipeline.reset()
        conv = orchestrator.generate_conversation(chain, seed=42)
        result = pipeline.evaluate_single(conv)
        assert result.passed

    def test_batch_generated(self, orchestrator, chain, pipeline):
        pipeline.reset()
        convs = [orchestrator.generate_conversation(chain, seed=42 + i) for i in range(3)]
        report = pipeline.evaluate_batch(convs)
        assert report.total == 3
        assert report.passed >= 1

    def test_repair_structural(self, pipeline_with_repair, chain):
        pipeline_with_repair.reset()
        result = pipeline_with_repair.evaluate_single(
            Conversation(messages=[]), chain=chain, seed=42,
        )
        assert result.passed

    def test_report_with_repair_stats(self, pipeline_with_repair, chain):
        pipeline_with_repair.reset()
        pipeline_with_repair.evaluate_single(
            Conversation(messages=[]), chain=chain, seed=42,
        )
        report = pipeline_with_repair.generate_report()
        assert report.total == 1
        assert len(report.repair_stats) > 0
