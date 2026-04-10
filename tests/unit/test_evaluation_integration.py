"""Integration tests for the full Phase 7 evaluation pipeline (Task 48).

Exercises: generate conversations → validate → judge → repair → report.
All agents run in offline mode — no LLM calls.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tooluse_gen.agents import (
    AssistantAgent,
    Conversation,
    ConversationOrchestrator,
    OrchestratorConfig,
    ToolExecutor,
    UserSimulator,
)
from tooluse_gen.evaluation import (
    Accepted,
    ConversationValidator,
    EvaluationConfig,
    EvaluationPipeline,
    EvaluationReport,
    EvaluationResult,
    JudgeAgent,
    JudgeScores,
    RepairLoop,
    RepairNeeded,
    RepairStats,
    ValidationResult,
)
from tooluse_gen.graph.builder import GraphBuilder
from tooluse_gen.graph.chain_models import SamplingConstraints, ToolChain
from tooluse_gen.graph.embeddings import EmbeddingService
from tooluse_gen.graph.facade import ToolChainSampler
from tooluse_gen.graph.models import GraphConfig
from tooluse_gen.graph.sampler import SamplerConfig
from tooluse_gen.registry.models import Endpoint, HttpMethod, Parameter, ParameterType, Tool
from tooluse_gen.registry.registry import ToolRegistry

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Mock embedding (no real model)
# ---------------------------------------------------------------------------


class _MockEmb(EmbeddingService):
    def __init__(self) -> None:
        self._model = None
        self._cache_dir = None

    def embed_text(self, text: str) -> list[float]:
        h = hash(text)
        rng = np.random.default_rng(abs(h) % (2**31))
        vec = rng.standard_normal(384).tolist()
        n = sum(x * x for x in vec) ** 0.5
        return [x / n for x in vec]

    def embed_batch(
        self, texts: list[str], batch_size: int = 256, show_progress: bool = False
    ) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.add_tool(
        Tool(
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
        )
    )
    reg.add_tool(
        Tool(
            tool_id="weather", name="Weather API", domain="Weather",
            endpoints=[
                Endpoint(endpoint_id="weather/current", tool_id="weather", name="Current",
                         method=HttpMethod.GET, path="/w",
                         parameters=[Parameter(name="city", param_type=ParameterType.STRING, required=True)],
                         required_parameters=["city"]),
            ],
        )
    )
    return reg


@pytest.fixture(scope="module")
def graph(registry: ToolRegistry) -> object:
    return GraphBuilder(
        config=GraphConfig(include_semantic_edges=False, max_edges_per_node=20),
        embedding_service=_MockEmb(),
    ).build(registry)


@pytest.fixture(scope="module")
def sampler(graph: object) -> ToolChainSampler:
    return ToolChainSampler(graph, SamplerConfig(max_iterations=300, max_retries=30))  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def orchestrator(registry: ToolRegistry) -> ConversationOrchestrator:
    return ConversationOrchestrator(
        user_sim=UserSimulator(), assistant=AssistantAgent(registry=registry),
        executor=ToolExecutor(registry),
        config=OrchestratorConfig(require_disambiguation=False),
    )


@pytest.fixture(scope="module")
def conversations(
    orchestrator: ConversationOrchestrator, sampler: ToolChainSampler
) -> list[tuple[Conversation, ToolChain]]:
    rng = np.random.default_rng(42)
    results: list[tuple[Conversation, ToolChain]] = []
    for i in range(5):
        chain = sampler.sample_chain(
            SamplingConstraints(min_steps=2, max_steps=3, min_tools=1), rng
        )
        conv = orchestrator.generate_conversation(chain, seed=42 + i)
        results.append((conv, chain))
    return results


# ===================================================================
# 1. Import completeness
# ===================================================================


class TestImportCompleteness:
    def test_all_12_symbols(self):
        symbols = [
            Accepted, ConversationValidator, EvaluationConfig, EvaluationPipeline,
            EvaluationReport, EvaluationResult, JudgeAgent, JudgeScores,
            RepairLoop, RepairNeeded, RepairStats, ValidationResult,
        ]
        assert len(symbols) == 12

    def test_models_importable(self):
        assert JudgeScores is not None
        assert EvaluationResult is not None
        assert ValidationResult is not None
        assert RepairNeeded is not None
        assert Accepted is not None
        assert EvaluationConfig is not None
        assert EvaluationReport is not None

    def test_classes_importable(self):
        assert ConversationValidator is not None
        assert JudgeAgent is not None
        assert RepairLoop is not None
        assert EvaluationPipeline is not None
        assert RepairStats is not None


# ===================================================================
# 2. Validator integration
# ===================================================================


class TestValidatorIntegration:
    def test_generated_passes(self, conversations):
        v = ConversationValidator()
        for conv, _ in conversations:
            r = v.validate(conv)
            assert r.valid, f"Validation failed: {r.errors}"

    def test_empty_fails(self):
        r = ConversationValidator().validate(Conversation(messages=[]))
        assert not r.valid

    def test_with_registry(self, registry, conversations):
        v = ConversationValidator(registry=registry)
        conv, _ = conversations[0]
        r = v.validate(conv)
        # May pass or fail depending on args — just check it runs.
        assert isinstance(r, ValidationResult)

    def test_error_count(self):
        r = ConversationValidator().validate(Conversation(messages=[]))
        assert r.error_count > 0


# ===================================================================
# 3. Judge integration
# ===================================================================


class TestJudgeIntegration:
    def test_score_generated(self, conversations):
        conv, _ = conversations[0]
        s = JudgeAgent().score(conv)
        assert isinstance(s, JudgeScores)

    def test_scores_in_range(self, conversations):
        conv, _ = conversations[0]
        s = JudgeAgent().score(conv)
        for v in s.scores_dict.values():
            assert 1 <= v <= 5

    def test_has_reasoning(self, conversations):
        conv, _ = conversations[0]
        s = JudgeAgent().score(conv)
        assert len(s.reasoning) > 0

    def test_score_batch(self, conversations):
        convs = [c for c, _ in conversations]
        scores = JudgeAgent().score_batch(convs)
        assert len(scores) == len(convs)

    def test_aggregate(self, conversations):
        convs = [c for c, _ in conversations]
        scores = JudgeAgent().score_batch(convs)
        agg = JudgeAgent().aggregate_scores(scores)
        assert 1 <= agg.tool_correctness <= 5


# ===================================================================
# 4. RepairLoop integration
# ===================================================================


class TestRepairLoopIntegration:
    def test_valid_passes_first_try(self, orchestrator, conversations):
        conv, chain = conversations[0]
        loop = RepairLoop(
            orchestrator=orchestrator, validator=ConversationValidator(),
            judge=JudgeAgent(), config=EvaluationConfig(min_score=3.0),
        )
        _, result = loop.evaluate_and_repair(conv, chain, seed=42)
        assert result.passed
        assert result.attempt_number == 1

    def test_structural_repair(self, orchestrator, conversations):
        _, chain = conversations[0]
        loop = RepairLoop(
            orchestrator=orchestrator, validator=ConversationValidator(),
            judge=JudgeAgent(), config=EvaluationConfig(min_score=3.0, max_retries=2),
        )
        _, result = loop.evaluate_and_repair(Conversation(messages=[]), chain, seed=42)
        assert result.passed

    def test_quality_failure(self, orchestrator, conversations):
        conv, chain = conversations[0]
        loop = RepairLoop(
            orchestrator=orchestrator, validator=ConversationValidator(),
            judge=JudgeAgent(), config=EvaluationConfig(min_score=5.0, max_retries=1),
        )
        _, result = loop.evaluate_and_repair(conv, chain, seed=42)
        assert not result.passed

    def test_stats_tracked(self, orchestrator, conversations):
        conv, chain = conversations[0]
        loop = RepairLoop(
            orchestrator=orchestrator, validator=ConversationValidator(),
            judge=JudgeAgent(), config=EvaluationConfig(min_score=5.0, max_retries=1),
        )
        loop.evaluate_and_repair(conv, chain, seed=42)
        stats = loop.get_stats()
        assert stats.quality_repairs >= 1


# ===================================================================
# 5. EvaluationPipeline integration
# ===================================================================


class TestPipelineIntegration:
    def test_single_passes(self, conversations):
        conv, _ = conversations[0]
        pipe = EvaluationPipeline(
            validator=ConversationValidator(), judge=JudgeAgent(),
            config=EvaluationConfig(min_score=3.0),
        )
        result = pipe.evaluate_single(conv)
        assert result.passed

    def test_empty_without_repair_fails(self):
        pipe = EvaluationPipeline(
            validator=ConversationValidator(), judge=JudgeAgent(),
        )
        result = pipe.evaluate_single(Conversation(messages=[]))
        assert not result.passed

    def test_repair_fixes_structural(self, orchestrator, conversations):
        _, chain = conversations[0]
        repair = RepairLoop(
            orchestrator=orchestrator, validator=ConversationValidator(),
            judge=JudgeAgent(), config=EvaluationConfig(min_score=3.0),
        )
        pipe = EvaluationPipeline(
            validator=ConversationValidator(), judge=JudgeAgent(),
            repair_loop=repair, config=EvaluationConfig(min_score=3.0),
        )
        result = pipe.evaluate_single(Conversation(messages=[]), chain=chain, seed=42)
        assert result.passed

    def test_batch_totals(self, conversations):
        convs = [c for c, _ in conversations]
        pipe = EvaluationPipeline(
            validator=ConversationValidator(), judge=JudgeAgent(),
            config=EvaluationConfig(min_score=3.0),
        )
        report = pipe.evaluate_batch(convs)
        assert report.total == len(convs)

    def test_report_distribution(self, conversations):
        convs = [c for c, _ in conversations]
        pipe = EvaluationPipeline(
            validator=ConversationValidator(), judge=JudgeAgent(),
            config=EvaluationConfig(min_score=3.0),
        )
        report = pipe.evaluate_batch(convs)
        assert len(report.score_distribution) > 0

    def test_repair_stats_in_report(self, orchestrator, conversations):
        _, chain = conversations[0]
        repair = RepairLoop(
            orchestrator=orchestrator, validator=ConversationValidator(),
            judge=JudgeAgent(), config=EvaluationConfig(min_score=3.0),
        )
        pipe = EvaluationPipeline(
            validator=ConversationValidator(), judge=JudgeAgent(),
            repair_loop=repair, config=EvaluationConfig(min_score=3.0),
        )
        pipe.evaluate_single(Conversation(messages=[]), chain=chain, seed=42)
        report = pipe.generate_report()
        assert len(report.repair_stats) > 0


# ===================================================================
# 6. Full end-to-end flow
# ===================================================================


class TestFullFlow:
    def test_generate_validate_judge_report(self, conversations):
        convs = [c for c, _ in conversations]
        pipe = EvaluationPipeline(
            validator=ConversationValidator(), judge=JudgeAgent(),
            config=EvaluationConfig(min_score=3.0),
        )
        report = pipe.evaluate_batch(convs)
        assert report.total == len(convs)
        assert report.passed >= 1

    def test_pass_rate_in_range(self, conversations):
        convs = [c for c, _ in conversations]
        pipe = EvaluationPipeline(
            validator=ConversationValidator(), judge=JudgeAgent(),
            config=EvaluationConfig(min_score=3.0),
        )
        report = pipe.evaluate_batch(convs)
        assert 0.0 <= report.pass_rate <= 1.0

    def test_all_have_scores(self, conversations):
        convs = [c for c, _ in conversations]
        pipe = EvaluationPipeline(
            validator=ConversationValidator(), judge=JudgeAgent(),
            config=EvaluationConfig(min_score=3.0),
        )
        pipe.evaluate_batch(convs)
        for result in pipe.get_results():
            assert result.scores is not None

    def test_jsonl_with_evaluation(self, conversations):
        conv, _ = conversations[0]
        pipe = EvaluationPipeline(
            validator=ConversationValidator(), judge=JudgeAgent(),
            config=EvaluationConfig(min_score=3.0),
        )
        result = pipe.evaluate_single(conv)
        record = conv.to_jsonl_dict()
        record["eval_result"] = {"passed": result.passed}
        if result.scores:
            record["eval_result"]["scores"] = result.scores.scores_dict
        json_str = json.dumps(record, default=str)
        parsed = json.loads(json_str)
        assert "eval_result" in parsed


# ===================================================================
# 7. Determinism
# ===================================================================


class TestDeterminism:
    def test_same_scores(self, conversations):
        conv, _ = conversations[0]
        s1 = JudgeAgent().score(conv)
        s2 = JudgeAgent().score(conv)
        assert s1.scores_dict == s2.scores_dict

    def test_same_report(self, conversations):
        convs = [c for c, _ in conversations[:3]]
        pipe1 = EvaluationPipeline(
            validator=ConversationValidator(), judge=JudgeAgent(),
            config=EvaluationConfig(min_score=3.0),
        )
        pipe2 = EvaluationPipeline(
            validator=ConversationValidator(), judge=JudgeAgent(),
            config=EvaluationConfig(min_score=3.0),
        )
        r1 = pipe1.evaluate_batch(convs)
        r2 = pipe2.evaluate_batch(convs)
        assert r1.total == r2.total
        assert r1.passed == r2.passed


# ===================================================================
# 8. Edge cases
# ===================================================================


class TestEdgeCases:
    def test_single_batch(self, conversations):
        conv, _ = conversations[0]
        pipe = EvaluationPipeline(
            validator=ConversationValidator(), judge=JudgeAgent(),
            config=EvaluationConfig(min_score=3.0),
        )
        report = pipe.evaluate_batch([conv])
        assert report.total == 1

    def test_all_fail(self, conversations):
        convs = [c for c, _ in conversations]
        pipe = EvaluationPipeline(
            validator=ConversationValidator(), judge=JudgeAgent(),
            config=EvaluationConfig(min_score=5.0),
        )
        report = pipe.evaluate_batch(convs)
        assert report.total == len(convs)
        assert report.failed == len(convs)
        assert report.passed == 0

    def test_mixed_valid_invalid(self, conversations):
        conv, _ = conversations[0]
        pipe = EvaluationPipeline(
            validator=ConversationValidator(), judge=JudgeAgent(),
            config=EvaluationConfig(min_score=3.0),
        )
        mixed = [conv, Conversation(messages=[]), conv]
        report = pipe.evaluate_batch(mixed)
        assert report.total == 3
        assert report.passed >= 1
        assert report.failed >= 1
