"""Tests for verbose mode and debug logging (Task 81)."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tooluse_gen.agents.assistant_agent import AssistantAgent
from tooluse_gen.agents.conversation_models import Conversation, Message
from tooluse_gen.agents.orchestrator import ConversationOrchestrator, OrchestratorConfig
from tooluse_gen.agents.tool_executor import ToolExecutor
from tooluse_gen.agents.user_simulator import UserSimulator
from tooluse_gen.graph.chain_models import SamplingConstraints
from tooluse_gen.graph.facade import ToolChainSampler
from tooluse_gen.graph.persistence import load_graph
from tooluse_gen.registry.serialization import load_registry
from tooluse_gen.utils.logging import setup_logging, trace_conversation

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
def orchestrator_and_chain(build_artifacts: Path):  # type: ignore[no-untyped-def]
    registry, _ = load_registry(build_artifacts / "registry.json")
    graph, _ = load_graph(build_artifacts / "graph.pkl")
    orch = ConversationOrchestrator(
        user_sim=UserSimulator(),
        assistant=AssistantAgent(registry=registry),
        executor=ToolExecutor(registry=registry),
        config=OrchestratorConfig(timeout_seconds=0.0),
    )
    sampler = ToolChainSampler(graph)
    chain = sampler.sample_chain(
        SamplingConstraints(min_steps=1, max_steps=2, min_tools=1),
        np.random.default_rng(42),
    )
    return orch, chain


# ===================================================================
# Verbosity levels
# ===================================================================


class TestVerbosityLevels:
    def test_verbosity_0_warning(self) -> None:
        root = setup_logging(verbosity=0)
        assert root.level <= logging.WARNING

    def test_verbosity_1_info(self) -> None:
        root = setup_logging(verbosity=1)
        assert root.level <= logging.INFO

    def test_verbosity_2_debug(self) -> None:
        root = setup_logging(verbosity=2)
        assert root.level <= logging.DEBUG


# ===================================================================
# Agent debug logging
# ===================================================================


class TestAgentDebugLogging:
    def test_orchestrator_logs_start(
        self, orchestrator_and_chain, caplog,  # type: ignore[no-untyped-def]
    ) -> None:
        orch, chain = orchestrator_and_chain
        with caplog.at_level(logging.DEBUG, logger="tooluse_gen"):
            orch.generate_conversation(chain, seed=42)
        assert any("Starting conversation" in r.message for r in caplog.records)

    def test_orchestrator_logs_tool_call(
        self, orchestrator_and_chain, caplog,  # type: ignore[no-untyped-def]
    ) -> None:
        orch, chain = orchestrator_and_chain
        with caplog.at_level(logging.DEBUG, logger="tooluse_gen"):
            orch.generate_conversation(chain, seed=42)
        assert any("Tool call:" in r.message for r in caplog.records)

    def test_orchestrator_logs_complete(
        self, orchestrator_and_chain, caplog,  # type: ignore[no-untyped-def]
    ) -> None:
        orch, chain = orchestrator_and_chain
        with caplog.at_level(logging.DEBUG, logger="tooluse_gen"):
            orch.generate_conversation(chain, seed=42)
        assert any("Conversation complete" in r.message for r in caplog.records)

    def test_user_simulator_logs_initial(
        self, orchestrator_and_chain, caplog,  # type: ignore[no-untyped-def]
    ) -> None:
        orch, chain = orchestrator_and_chain
        with caplog.at_level(logging.DEBUG, logger="tooluse_gen"):
            orch.generate_conversation(chain, seed=42)
        assert any("User initial msg" in r.message for r in caplog.records)

    def test_assistant_logs_response(
        self, orchestrator_and_chain, caplog,  # type: ignore[no-untyped-def]
    ) -> None:
        orch, chain = orchestrator_and_chain
        with caplog.at_level(logging.DEBUG, logger="tooluse_gen"):
            orch.generate_conversation(chain, seed=42)
        assert any("Assistant" in r.message for r in caplog.records)

    def test_tool_result_logged(
        self, orchestrator_and_chain, caplog,  # type: ignore[no-untyped-def]
    ) -> None:
        orch, chain = orchestrator_and_chain
        with caplog.at_level(logging.DEBUG, logger="tooluse_gen"):
            orch.generate_conversation(chain, seed=42)
        assert any("Tool result:" in r.message for r in caplog.records)


# ===================================================================
# trace_conversation
# ===================================================================


class TestTraceConversation:
    def test_trace_at_debug(self, caplog) -> None:  # type: ignore[no-untyped-def]
        conv = Conversation(messages=[
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
        ])
        with caplog.at_level(logging.DEBUG, logger="tooluse_gen"):
            trace_conversation(conv)
        assert any("Conversation" in r.message and "messages" in r.message for r in caplog.records)

    def test_trace_noop_at_warning(self, caplog) -> None:  # type: ignore[no-untyped-def]
        conv = Conversation(messages=[Message(role="user", content="hi")])
        with caplog.at_level(logging.WARNING, logger="tooluse_gen"):
            trace_conversation(conv)
        assert not any("Conversation" in r.message for r in caplog.records)

    def test_trace_shows_roles(self, caplog) -> None:  # type: ignore[no-untyped-def]
        conv = Conversation(messages=[
            Message(role="user", content="hi"),
            Message(role="assistant", content="ok"),
            Message(role="tool", content="data"),
        ])
        with caplog.at_level(logging.DEBUG, logger="tooluse_gen"):
            trace_conversation(conv)
        matching = [r for r in caplog.records if "roles=" in r.message]
        assert len(matching) >= 1
        assert "user" in matching[0].message
        assert "assistant" in matching[0].message
        assert "tool" in matching[0].message

    def test_trace_importable(self) -> None:
        from tooluse_gen.utils import trace_conversation as tc

        assert callable(tc)
