"""Tests for batch generator interrupt handling (Task 80)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tooluse_gen.agents.assistant_agent import AssistantAgent
from tooluse_gen.agents.batch_generator import BatchGenerator
from tooluse_gen.agents.orchestrator import ConversationOrchestrator, OrchestratorConfig
from tooluse_gen.agents.tool_executor import ToolExecutor
from tooluse_gen.agents.user_simulator import UserSimulator
from tooluse_gen.graph.chain_models import SamplingConstraints
from tooluse_gen.graph.facade import ToolChainSampler
from tooluse_gen.graph.persistence import load_graph
from tooluse_gen.registry.serialization import load_registry

pytestmark = pytest.mark.unit

_CONSTRAINTS = SamplingConstraints(min_steps=1, max_steps=2, min_tools=1)


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
def batch_gen(build_artifacts: Path) -> BatchGenerator:
    registry, _ = load_registry(build_artifacts / "registry.json")
    graph, _ = load_graph(build_artifacts / "graph.pkl")
    orchestrator = ConversationOrchestrator(
        user_sim=UserSimulator(),
        assistant=AssistantAgent(registry=registry),
        executor=ToolExecutor(registry=registry),
        config=OrchestratorConfig(timeout_seconds=0.0),
    )
    sampler = ToolChainSampler(graph)
    return BatchGenerator(orchestrator=orchestrator, sampler=sampler)


# ===================================================================
# Tests
# ===================================================================


class TestInterruptCheck:
    def test_no_interrupt_runs_full(self, batch_gen: BatchGenerator) -> None:
        """Without interrupt_check, all conversations are generated."""
        convs = batch_gen.generate_batch(count=3, constraints=_CONSTRAINTS, seed=42)
        assert len(convs) == 3

    def test_interrupt_stops_early(self, batch_gen: BatchGenerator) -> None:
        """interrupt_check returning True stops the loop."""
        calls = [0]

        def check() -> bool:
            calls[0] += 1
            return calls[0] > 2  # allow 2 iterations, interrupt on 3rd

        convs = batch_gen.generate_batch(
            count=10, constraints=_CONSTRAINTS, seed=42, interrupt_check=check,
        )
        assert len(convs) <= 2

    def test_interrupt_returns_partial_results(
        self, batch_gen: BatchGenerator
    ) -> None:
        """Interrupted batch returns the conversations generated so far."""
        calls = [0]

        def check() -> bool:
            calls[0] += 1
            return calls[0] > 1

        convs = batch_gen.generate_batch(
            count=10, constraints=_CONSTRAINTS, seed=42, interrupt_check=check,
        )
        assert len(convs) >= 1
        for conv in convs:
            assert conv.conversation_id
            assert len(conv.messages) >= 1

    def test_interrupt_stats_correct(self, batch_gen: BatchGenerator) -> None:
        """BatchStats reflects actual generated count, not requested."""
        calls = [0]

        def check() -> bool:
            calls[0] += 1
            return calls[0] > 2

        convs = batch_gen.generate_batch(
            count=10, constraints=_CONSTRAINTS, seed=42, interrupt_check=check,
        )
        stats = batch_gen.get_batch_stats()
        assert stats.total_generated == len(convs)
        assert stats.total_generated <= 2

    def test_interrupt_none_default(self, batch_gen: BatchGenerator) -> None:
        """Default interrupt_check=None runs all iterations."""
        convs = batch_gen.generate_batch(
            count=3, constraints=_CONSTRAINTS, seed=42, interrupt_check=None,
        )
        assert len(convs) == 3

    def test_interrupt_check_called_per_iteration(self) -> None:
        """interrupt_check is called once per loop iteration."""
        from tooluse_gen.graph.diversity import DiversityMetrics

        mock_orch = MagicMock()
        mock_orch.generate_conversation.return_value = MagicMock(
            metadata=MagicMock(
                num_turns=2, num_tool_calls=1, generation_time_ms=10,
                tools_used=["t1"], domains=[],
            ),
        )
        mock_sampler = MagicMock()
        mock_sampler.sample_chain.return_value = MagicMock()
        mock_sampler.get_diversity_report.return_value = DiversityMetrics()

        gen = BatchGenerator(orchestrator=mock_orch, sampler=mock_sampler)

        check_mock = MagicMock(return_value=False)
        gen.generate_batch(
            count=5, constraints=_CONSTRAINTS, seed=42, interrupt_check=check_mock,
        )
        assert check_mock.call_count == 5

    def test_progress_bar_closed_on_interrupt(
        self, batch_gen: BatchGenerator
    ) -> None:
        """Progress bar is closed even when interrupted early."""
        calls = [0]

        def check() -> bool:
            calls[0] += 1
            return calls[0] > 1

        with patch("tooluse_gen.utils.progress.create_progress_bar") as mock_create:
            mock_bar = MagicMock()
            mock_create.return_value = mock_bar
            batch_gen.generate_batch(
                count=10, constraints=_CONSTRAINTS, seed=42,
                show_progress=True, interrupt_check=check,
            )
            mock_bar.close.assert_called_once()

    def test_immediate_interrupt(self, batch_gen: BatchGenerator) -> None:
        """Interrupt on first check returns empty list."""
        convs = batch_gen.generate_batch(
            count=10, constraints=_CONSTRAINTS, seed=42,
            interrupt_check=lambda: True,
        )
        assert len(convs) == 0
        stats = batch_gen.get_batch_stats()
        assert stats.total_generated == 0
