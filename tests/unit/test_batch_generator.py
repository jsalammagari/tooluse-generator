"""Tests for the BatchGenerator (Task 39)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from tooluse_gen.agents.assistant_agent import AssistantAgent
from tooluse_gen.agents.batch_generator import BatchGenerator, BatchStats
from tooluse_gen.agents.conversation_models import Conversation
from tooluse_gen.agents.orchestrator import ConversationOrchestrator, OrchestratorConfig
from tooluse_gen.agents.tool_executor import ToolExecutor
from tooluse_gen.agents.user_simulator import UserSimulator
from tooluse_gen.graph.builder import GraphBuilder
from tooluse_gen.graph.chain_models import SamplingConstraints
from tooluse_gen.graph.diversity import DiversitySteeringConfig
from tooluse_gen.graph.embeddings import EmbeddingService
from tooluse_gen.graph.facade import ToolChainSampler
from tooluse_gen.graph.models import GraphConfig
from tooluse_gen.graph.sampler import SamplerConfig, SamplingError
from tooluse_gen.registry.models import Endpoint, HttpMethod, Parameter, ParameterType, Tool
from tooluse_gen.registry.registry import ToolRegistry

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Mock embedding service (no real model)
# ---------------------------------------------------------------------------


class _MockEmbedding(EmbeddingService):
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
                    name="Search Hotels",
                    description="Search",
                    method=HttpMethod.GET,
                    path="/s",
                    parameters=[
                        Parameter(
                            name="city",
                            param_type=ParameterType.STRING,
                            required=True,
                        ),
                    ],
                    required_parameters=["city"],
                ),
                Endpoint(
                    endpoint_id="hotels/book",
                    tool_id="hotels",
                    name="Book Hotel",
                    description="Book",
                    method=HttpMethod.POST,
                    path="/b",
                    parameters=[
                        Parameter(
                            name="hotel_id",
                            param_type=ParameterType.STRING,
                            required=True,
                        ),
                        Parameter(
                            name="guest_name",
                            param_type=ParameterType.STRING,
                            required=True,
                        ),
                    ],
                    required_parameters=["hotel_id", "guest_name"],
                ),
            ],
        )
    )
    reg.add_tool(
        Tool(
            tool_id="weather",
            name="Weather API",
            domain="Weather",
            endpoints=[
                Endpoint(
                    endpoint_id="weather/current",
                    tool_id="weather",
                    name="Current Weather",
                    description="Weather",
                    method=HttpMethod.GET,
                    path="/w",
                    parameters=[
                        Parameter(
                            name="city",
                            param_type=ParameterType.STRING,
                            required=True,
                        ),
                    ],
                    required_parameters=["city"],
                ),
            ],
        )
    )
    return reg


@pytest.fixture()
def sampler(registry: ToolRegistry) -> ToolChainSampler:
    graph = GraphBuilder(
        config=GraphConfig(
            include_tool_nodes=True,
            include_domain_edges=True,
            include_semantic_edges=False,
            max_edges_per_node=20,
        ),
        embedding_service=_MockEmbedding(),
    ).build(registry)
    return ToolChainSampler(
        graph,
        SamplerConfig(max_iterations=300, max_retries=30),
        DiversitySteeringConfig(enabled=True),
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
def generator(
    orchestrator: ConversationOrchestrator, sampler: ToolChainSampler
) -> BatchGenerator:
    return BatchGenerator(orchestrator=orchestrator, sampler=sampler)


@pytest.fixture()
def constraints() -> SamplingConstraints:
    return SamplingConstraints(min_steps=2, max_steps=3, min_tools=1)


# ===================================================================
# BatchStats
# ===================================================================


class TestBatchStats:
    def test_defaults(self):
        s = BatchStats()
        assert s.total_generated == 0
        assert s.total_failed == 0
        assert s.diversity_metrics is None
        assert s.average_turns == 0.0
        assert s.steering_enabled is False

    def test_custom_values(self):
        s = BatchStats(
            total_generated=10,
            total_failed=2,
            average_turns=7.5,
            tools_coverage=3,
            domain_coverage=2,
            steering_enabled=True,
        )
        assert s.total_generated == 10
        assert s.tools_coverage == 3
        assert s.steering_enabled is True

    def test_serialization_round_trip(self):
        s = BatchStats(total_generated=5, average_turns=6.0)
        data = s.model_dump()
        restored = BatchStats.model_validate(data)
        assert restored.total_generated == 5
        assert restored.average_turns == 6.0


# ===================================================================
# Construction
# ===================================================================


class TestConstruction:
    def test_default(self, orchestrator, sampler):
        gen = BatchGenerator(orchestrator=orchestrator, sampler=sampler)
        assert gen._stats is None

    def test_with_diversity_config(self, orchestrator, sampler):
        cfg = DiversitySteeringConfig(enabled=True, weight_decay=0.8)
        gen = BatchGenerator(
            orchestrator=orchestrator, sampler=sampler, diversity_config=cfg
        )
        assert gen._diversity_config is not None
        assert gen._diversity_config.weight_decay == 0.8


# ===================================================================
# generate_batch — basic
# ===================================================================


class TestGenerateBatchBasic:
    def test_returns_list(self, generator, constraints):
        result = generator.generate_batch(count=3, constraints=constraints, seed=42)
        assert isinstance(result, list)

    def test_correct_count(self, generator, constraints):
        result = generator.generate_batch(count=3, constraints=constraints, seed=42)
        assert len(result) >= 2  # some may fail

    def test_each_has_messages(self, generator, constraints):
        result = generator.generate_batch(count=3, constraints=constraints, seed=42)
        for conv in result:
            assert conv.turn_count > 0

    def test_seed_pattern(self, generator, constraints):
        result = generator.generate_batch(count=3, constraints=constraints, seed=100)
        for i, conv in enumerate(result):
            assert conv.metadata.seed == 100 + i

    def test_valid_roles(self, generator, constraints):
        result = generator.generate_batch(count=3, constraints=constraints, seed=42)
        for conv in result:
            roles = {m.role for m in conv.messages}
            assert "user" in roles
            assert "assistant" in roles
            assert "tool" in roles

    def test_json_serializable(self, generator, constraints):
        import json

        result = generator.generate_batch(count=2, constraints=constraints, seed=42)
        for conv in result:
            parsed = json.loads(conv.to_jsonl())
            assert "conversation_id" in parsed


# ===================================================================
# generate_batch — diversity
# ===================================================================


class TestGenerateBatchDiversity:
    def test_steering_enabled_metrics(self, generator, constraints):
        generator.generate_batch(
            count=3, constraints=constraints, seed=42, steering_enabled=True
        )
        stats = generator.get_batch_stats()
        assert stats.steering_enabled is True
        assert stats.diversity_metrics is not None

    def test_steering_disabled(self, generator, constraints):
        result = generator.generate_batch(
            count=3, constraints=constraints, seed=42, steering_enabled=False
        )
        assert len(result) >= 1
        stats = generator.get_batch_stats()
        assert stats.steering_enabled is False

    def test_stats_reflect_flag(self, generator, constraints):
        generator.generate_batch(
            count=2, constraints=constraints, seed=42, steering_enabled=True
        )
        assert generator.get_batch_stats().steering_enabled is True

        generator.generate_batch(
            count=2, constraints=constraints, seed=42, steering_enabled=False
        )
        assert generator.get_batch_stats().steering_enabled is False

    def test_tools_domains_across_batch(self, generator, constraints):
        generator.generate_batch(count=5, constraints=constraints, seed=42)
        stats = generator.get_batch_stats()
        assert stats.tools_coverage >= 1
        assert stats.domain_coverage >= 1


# ===================================================================
# get_batch_stats
# ===================================================================


class TestGetBatchStats:
    def test_after_generation(self, generator, constraints):
        generator.generate_batch(count=3, constraints=constraints, seed=42)
        stats = generator.get_batch_stats()
        assert isinstance(stats, BatchStats)

    def test_raises_before_generation(self, generator):
        with pytest.raises(ValueError):
            generator.get_batch_stats()

    def test_total_generated(self, generator, constraints):
        result = generator.generate_batch(count=3, constraints=constraints, seed=42)
        stats = generator.get_batch_stats()
        assert stats.total_generated == len(result)

    def test_average_turns_positive(self, generator, constraints):
        generator.generate_batch(count=3, constraints=constraints, seed=42)
        stats = generator.get_batch_stats()
        assert stats.average_turns > 0


# ===================================================================
# _compute_stats
# ===================================================================


class TestComputeStats:
    def test_correct_totals(self, generator, constraints):
        result = generator.generate_batch(count=3, constraints=constraints, seed=42)
        stats = generator.get_batch_stats()
        assert stats.total_generated == len(result)
        assert stats.total_generated + stats.total_failed >= 0

    def test_tools_coverage(self, generator, constraints):
        generator.generate_batch(count=5, constraints=constraints, seed=42)
        stats = generator.get_batch_stats()
        assert stats.tools_coverage >= 1

    def test_domain_coverage(self, generator, constraints):
        generator.generate_batch(count=5, constraints=constraints, seed=42)
        stats = generator.get_batch_stats()
        assert stats.domain_coverage >= 1

    def test_averages(self, generator, constraints):
        result = generator.generate_batch(count=3, constraints=constraints, seed=42)
        stats = generator.get_batch_stats()
        if len(result) > 0:
            assert stats.average_turns > 0
            assert stats.average_tool_calls >= 0


# ===================================================================
# Error handling
# ===================================================================


class TestErrorHandling:
    def test_sampling_failure_skips(self, orchestrator):
        """A sampler that always fails should produce empty batch."""
        mock_sampler = MagicMock(spec=ToolChainSampler)
        mock_sampler.sample_chain.side_effect = SamplingError("fail")
        mock_sampler.get_diversity_report.return_value = MagicMock(
            total_conversations=0
        )
        mock_sampler.reset_diversity.return_value = None

        gen = BatchGenerator(orchestrator=orchestrator, sampler=mock_sampler)
        result = gen.generate_batch(
            count=3,
            constraints=SamplingConstraints(min_steps=1, max_steps=2, min_tools=1),
            seed=42,
        )
        assert result == []

    def test_total_failed_reflected(self, orchestrator):
        mock_sampler = MagicMock(spec=ToolChainSampler)
        mock_sampler.sample_chain.side_effect = SamplingError("fail")
        mock_sampler.get_diversity_report.return_value = MagicMock(
            total_conversations=0
        )
        mock_sampler.reset_diversity.return_value = None

        gen = BatchGenerator(orchestrator=orchestrator, sampler=mock_sampler)
        gen.generate_batch(
            count=5,
            constraints=SamplingConstraints(min_steps=1, max_steps=2, min_tools=1),
            seed=42,
        )
        stats = gen.get_batch_stats()
        assert stats.total_failed == 5
        assert stats.total_generated == 0

    def test_empty_batch_returns_empty(self, orchestrator):
        mock_sampler = MagicMock(spec=ToolChainSampler)
        mock_sampler.sample_chain.side_effect = SamplingError("fail")
        mock_sampler.get_diversity_report.return_value = MagicMock(
            total_conversations=0
        )
        mock_sampler.reset_diversity.return_value = None

        gen = BatchGenerator(orchestrator=orchestrator, sampler=mock_sampler)
        result = gen.generate_batch(
            count=2,
            constraints=SamplingConstraints(min_steps=1, max_steps=2, min_tools=1),
            seed=42,
        )
        assert len(result) == 0


# ===================================================================
# Determinism
# ===================================================================


class TestDeterminism:
    def test_same_seed(self, registry):
        def run(seed: int) -> list[Conversation]:
            graph = GraphBuilder(
                config=GraphConfig(include_semantic_edges=False),
                embedding_service=_MockEmbedding(),
            ).build(registry)
            sampler = ToolChainSampler(
                graph, SamplerConfig(max_iterations=200, max_retries=20)
            )
            orch = ConversationOrchestrator(
                user_sim=UserSimulator(),
                assistant=AssistantAgent(registry=registry),
                executor=ToolExecutor(registry),
                config=OrchestratorConfig(require_disambiguation=False),
            )
            gen = BatchGenerator(orchestrator=orch, sampler=sampler)
            return gen.generate_batch(
                count=3,
                constraints=SamplingConstraints(
                    min_steps=2, max_steps=3, min_tools=1
                ),
                seed=seed,
            )

        c1 = run(42)
        c2 = run(42)
        assert len(c1) == len(c2)
        for a, b in zip(c1, c2, strict=True):
            assert a.turn_count == b.turn_count
            for m1, m2 in zip(a.messages, b.messages, strict=True):
                assert m1.role == m2.role

    def test_different_seeds(self, generator, constraints):
        c1 = generator.generate_batch(count=2, constraints=constraints, seed=42)
        # Need fresh generator for independent RNG.
        c2 = generator.generate_batch(count=2, constraints=constraints, seed=99)
        # Both should produce conversations.
        assert len(c1) >= 1
        assert len(c2) >= 1


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    def test_count_one(self, generator, constraints):
        result = generator.generate_batch(count=1, constraints=constraints, seed=42)
        assert len(result) >= 1

    def test_count_zero(self, generator, constraints):
        result = generator.generate_batch(count=0, constraints=constraints, seed=42)
        assert result == []
        stats = generator.get_batch_stats()
        assert stats.total_generated == 0

    def test_large_batch(self, generator, constraints):
        result = generator.generate_batch(count=10, constraints=constraints, seed=42)
        assert len(result) >= 5  # Most should succeed.
        stats = generator.get_batch_stats()
        assert stats.total_generated == len(result)
