"""Tests for batch generation progress bar (Task 79)."""

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
    """Create a BatchGenerator from build artifacts."""
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


class TestBatchProgress:
    def test_show_progress_default_false(self, batch_gen: BatchGenerator) -> None:
        """show_progress defaults to False — no visible bar."""
        convs = batch_gen.generate_batch(count=2, constraints=_CONSTRAINTS, seed=42)
        assert len(convs) >= 1

    def test_show_progress_true_works(self, batch_gen: BatchGenerator) -> None:
        """show_progress=True completes without error."""
        convs = batch_gen.generate_batch(
            count=2, constraints=_CONSTRAINTS, seed=42, show_progress=True,
        )
        assert len(convs) >= 1

    def test_show_progress_false_no_bar(self, batch_gen: BatchGenerator) -> None:
        """show_progress=False creates a disabled tqdm (no output)."""
        with patch(
            "tooluse_gen.utils.progress.create_progress_bar"
        ) as mock_create:
            mock_bar = MagicMock()
            mock_create.return_value = mock_bar
            batch_gen.generate_batch(
                count=2, constraints=_CONSTRAINTS, seed=42, show_progress=False,
            )
            mock_create.assert_called_once()
            assert mock_create.call_args.kwargs["disable"] is True

    def test_progress_bar_advances(self, batch_gen: BatchGenerator) -> None:
        """Bar update() is called once per conversation attempt."""
        with patch(
            "tooluse_gen.utils.progress.create_progress_bar"
        ) as mock_create:
            mock_bar = MagicMock()
            mock_create.return_value = mock_bar
            batch_gen.generate_batch(
                count=3, constraints=_CONSTRAINTS, seed=42, show_progress=True,
            )
            assert mock_bar.update.call_count == 3
            mock_bar.close.assert_called_once()

    def test_progress_bar_advances_on_failure(self) -> None:
        """Bar advances even when sampling or generation fails."""
        mock_orch = MagicMock()
        mock_orch.generate_conversation.side_effect = RuntimeError("fail")
        mock_sampler = MagicMock()
        mock_sampler.sample_chain.return_value = MagicMock()
        mock_sampler.get_diversity_report.return_value = MagicMock()

        gen = BatchGenerator(orchestrator=mock_orch, sampler=mock_sampler)
        with patch(
            "tooluse_gen.utils.progress.create_progress_bar"
        ) as mock_create:
            mock_bar = MagicMock()
            mock_create.return_value = mock_bar
            convs = gen.generate_batch(
                count=3, constraints=_CONSTRAINTS, seed=42, show_progress=True,
            )
            assert len(convs) == 0  # all failed
            assert mock_bar.update.call_count == 3  # bar still advanced 3 times
            mock_bar.close.assert_called_once()

    def test_backward_compatible(self, batch_gen: BatchGenerator) -> None:
        """Calling without show_progress still works (default False)."""
        convs = batch_gen.generate_batch(
            count=2, constraints=_CONSTRAINTS, seed=42,
        )
        assert len(convs) >= 1
        stats = batch_gen.get_batch_stats()
        assert stats.total_generated >= 1
