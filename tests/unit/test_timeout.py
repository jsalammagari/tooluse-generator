"""Tests for per-conversation timeout handling (Task 78)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tooluse_gen.agents.assistant_agent import AssistantAgent
from tooluse_gen.agents.conversation_models import ConversationMetadata
from tooluse_gen.agents.orchestrator import ConversationOrchestrator, OrchestratorConfig
from tooluse_gen.agents.tool_executor import ToolExecutor
from tooluse_gen.agents.user_simulator import UserSimulator
from tooluse_gen.graph.chain_models import SamplingConstraints
from tooluse_gen.graph.facade import ToolChainSampler
from tooluse_gen.graph.persistence import load_graph
from tooluse_gen.registry.serialization import load_registry

pytestmark = pytest.mark.unit


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
def registry_and_graph(build_artifacts: Path):  # type: ignore[no-untyped-def]
    registry, _ = load_registry(build_artifacts / "registry.json")
    graph, _ = load_graph(build_artifacts / "graph.pkl")
    return registry, graph


@pytest.fixture()
def chain(registry_and_graph):  # type: ignore[no-untyped-def]
    _, graph = registry_and_graph
    sampler = ToolChainSampler(graph)
    rng = np.random.default_rng(42)
    return sampler.sample_chain(
        SamplingConstraints(min_steps=1, max_steps=2, min_tools=1), rng,
    )


# ===================================================================
# OrchestratorConfig
# ===================================================================


class TestTimeoutConfig:
    def test_default_timeout(self) -> None:
        cfg = OrchestratorConfig()
        assert cfg.timeout_seconds == 30.0

    def test_custom_timeout(self) -> None:
        cfg = OrchestratorConfig(timeout_seconds=10.0)
        assert cfg.timeout_seconds == 10.0

    def test_zero_timeout_means_no_limit(self) -> None:
        cfg = OrchestratorConfig(timeout_seconds=0.0)
        assert cfg.timeout_seconds == 0.0


# ===================================================================
# ConversationMetadata
# ===================================================================


class TestTimeoutMetadata:
    def test_timed_out_field_exists(self) -> None:
        meta = ConversationMetadata()
        assert hasattr(meta, "timed_out")

    def test_default_false(self) -> None:
        meta = ConversationMetadata()
        assert meta.timed_out is False

    def test_settable(self) -> None:
        meta = ConversationMetadata(timed_out=True)
        assert meta.timed_out is True


# ===================================================================
# Conversation Timeout
# ===================================================================


class TestConversationTimeout:
    def test_timeout_produces_partial_conversation(
        self, registry_and_graph, chain,  # type: ignore[no-untyped-def]
    ) -> None:
        """Very short timeout produces a partial conversation."""
        registry, _ = registry_and_graph
        config = OrchestratorConfig(timeout_seconds=0.001)
        orch = ConversationOrchestrator(
            user_sim=UserSimulator(),
            assistant=AssistantAgent(registry=registry),
            executor=ToolExecutor(registry=registry),
            config=config,
        )
        conv = orch.generate_conversation(chain, seed=42)
        # Should have at least the initial user message
        assert len(conv.messages) >= 1

    def test_timeout_sets_metadata_flag(
        self, registry_and_graph, chain,  # type: ignore[no-untyped-def]
    ) -> None:
        """Timed-out conversation has metadata.timed_out=True.

        We use max_turns=20 and a chain with many steps, combined with
        a very tight timeout and mock that makes time appear to pass
        faster than real time.
        """
        registry, _ = registry_and_graph
        # Use max_turns=20 so the loop would run many iterations,
        # giving the timeout check a chance to fire.
        config = OrchestratorConfig(timeout_seconds=0.5, max_turns=20)
        orch = ConversationOrchestrator(
            user_sim=UserSimulator(),
            assistant=AssistantAgent(registry=registry),
            executor=ToolExecutor(registry=registry),
            config=config,
        )

        # Sample a longer chain (3 steps) so the loop iterates more
        sampler = ToolChainSampler(registry_and_graph[1])
        long_chain = sampler.sample_chain(
            SamplingConstraints(min_steps=3, max_steps=5, min_tools=1),
            np.random.default_rng(99),
        )

        import time as _time
        original = _time.monotonic
        calls = [0]

        def fast_clock() -> float:
            calls[0] += 1
            # Each call advances by 0.2s so after 3 calls we're past 0.5s
            return original() + calls[0] * 0.2

        with patch("tooluse_gen.agents.orchestrator.time.monotonic", fast_clock):
            conv = orch.generate_conversation(long_chain, seed=42)
        assert conv.metadata.timed_out is True

    def test_no_timeout_flag_false(
        self, registry_and_graph, chain,  # type: ignore[no-untyped-def]
    ) -> None:
        """Normal conversation has metadata.timed_out=False."""
        registry, _ = registry_and_graph
        config = OrchestratorConfig(timeout_seconds=0.0)  # 0 = no limit
        orch = ConversationOrchestrator(
            user_sim=UserSimulator(),
            assistant=AssistantAgent(registry=registry),
            executor=ToolExecutor(registry=registry),
            config=config,
        )
        conv = orch.generate_conversation(chain, seed=42)
        assert conv.metadata.timed_out is False

    def test_timeout_conversation_has_valid_metadata(
        self, registry_and_graph, chain,  # type: ignore[no-untyped-def]
    ) -> None:
        """Timed-out conversations still have valid metadata."""
        registry, _ = registry_and_graph
        config = OrchestratorConfig(timeout_seconds=0.001)
        orch = ConversationOrchestrator(
            user_sim=UserSimulator(),
            assistant=AssistantAgent(registry=registry),
            executor=ToolExecutor(registry=registry),
            config=config,
        )
        conv = orch.generate_conversation(chain, seed=42)
        assert conv.metadata.seed == 42
        assert conv.metadata.generation_time_ms >= 0
        assert conv.conversation_id

    def test_zero_timeout_completes_normally(
        self, registry_and_graph, chain,  # type: ignore[no-untyped-def]
    ) -> None:
        """timeout_seconds=0 allows conversation to complete normally."""
        registry, _ = registry_and_graph
        config = OrchestratorConfig(timeout_seconds=0.0)
        orch = ConversationOrchestrator(
            user_sim=UserSimulator(),
            assistant=AssistantAgent(registry=registry),
            executor=ToolExecutor(registry=registry),
            config=config,
        )
        conv = orch.generate_conversation(chain, seed=42)
        assert len(conv.messages) >= 2
        assert conv.metadata.timed_out is False
