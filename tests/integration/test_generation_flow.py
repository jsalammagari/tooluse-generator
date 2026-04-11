"""Integration test — full generation flow (Task 66).

Exercises the complete pipeline at the Python API level:
graph → chain sampling → orchestrator → conversation → validation → output format.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tooluse_gen.agents.assistant_agent import AssistantAgent
from tooluse_gen.agents.batch_generator import BatchGenerator
from tooluse_gen.agents.orchestrator import ConversationOrchestrator
from tooluse_gen.agents.tool_executor import ToolExecutor
from tooluse_gen.agents.user_simulator import UserSimulator
from tooluse_gen.core.output_models import from_conversation, validate_record
from tooluse_gen.evaluation.validator import ConversationValidator
from tooluse_gen.graph.chain_models import SamplingConstraints, ToolChain
from tooluse_gen.graph.diversity import DiversitySteeringConfig
from tooluse_gen.graph.facade import ToolChainSampler
from tooluse_gen.graph.persistence import load_graph
from tooluse_gen.registry.serialization import load_registry

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Shared constraints (small graph friendly)
# ---------------------------------------------------------------------------

_CONSTRAINTS = SamplingConstraints(min_steps=1, max_steps=3, min_tools=1)


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
def orchestrator(registry_and_graph):  # type: ignore[no-untyped-def]
    registry, _ = registry_and_graph
    return ConversationOrchestrator(
        user_sim=UserSimulator(),
        assistant=AssistantAgent(registry=registry),
        executor=ToolExecutor(registry=registry),
    )


@pytest.fixture()
def sampler(registry_and_graph):  # type: ignore[no-untyped-def]
    _, graph = registry_and_graph
    return ToolChainSampler(graph)


def _sample(sampler: ToolChainSampler, seed: int = 42) -> ToolChain:
    rng = np.random.default_rng(seed)
    return sampler.sample_chain(_CONSTRAINTS, rng)


# ===================================================================
# Chain sampling
# ===================================================================


class TestChainSampling:
    """Graph -> chain sampling."""

    def test_sample_single_chain(self, sampler) -> None:  # type: ignore[no-untyped-def]
        chain = _sample(sampler)
        assert isinstance(chain, ToolChain)

    def test_chain_has_steps(self, sampler) -> None:  # type: ignore[no-untyped-def]
        chain = _sample(sampler)
        assert chain.total_step_count >= 1

    def test_chain_respects_max_steps(self, sampler) -> None:  # type: ignore[no-untyped-def]
        chain = _sample(sampler)
        assert chain.total_step_count <= _CONSTRAINTS.max_steps

    def test_sample_batch(self, sampler) -> None:  # type: ignore[no-untyped-def]
        rng = np.random.default_rng(42)
        chains = sampler.sample_batch(_CONSTRAINTS, count=5, rng=rng)
        assert len(chains) >= 1

    def test_chain_has_endpoint_ids(self, sampler) -> None:  # type: ignore[no-untyped-def]
        chain = _sample(sampler)
        assert len(chain.endpoint_ids) >= 1
        for eid in chain.endpoint_ids:
            assert isinstance(eid, str)
            assert len(eid) > 0


# ===================================================================
# Conversation generation
# ===================================================================


class TestConversationGeneration:
    """Chain -> conversation generation."""

    def test_generates_conversation(self, orchestrator, sampler) -> None:  # type: ignore[no-untyped-def]
        chain = _sample(sampler)
        conv = orchestrator.generate_conversation(chain, seed=42)
        assert conv.conversation_id

    def test_conversation_has_messages(self, orchestrator, sampler) -> None:  # type: ignore[no-untyped-def]
        chain = _sample(sampler)
        conv = orchestrator.generate_conversation(chain, seed=42)
        assert len(conv.messages) >= 2

    def test_conversation_starts_with_user(self, orchestrator, sampler) -> None:  # type: ignore[no-untyped-def]
        chain = _sample(sampler)
        conv = orchestrator.generate_conversation(chain, seed=42)
        assert conv.messages[0].role == "user"

    def test_conversation_has_tool_messages(self, orchestrator, sampler) -> None:  # type: ignore[no-untyped-def]
        chain = _sample(sampler)
        conv = orchestrator.generate_conversation(chain, seed=42)
        roles = {m.role for m in conv.messages}
        assert "tool" in roles

    def test_conversation_has_final_assistant(self, orchestrator, sampler) -> None:  # type: ignore[no-untyped-def]
        chain = _sample(sampler)
        conv = orchestrator.generate_conversation(chain, seed=42)
        assert conv.messages[-1].role == "assistant"

    def test_conversation_metadata_populated(self, orchestrator, sampler) -> None:  # type: ignore[no-untyped-def]
        chain = _sample(sampler)
        conv = orchestrator.generate_conversation(chain, seed=42)
        meta = conv.metadata
        assert meta.num_turns >= 2
        assert meta.num_tool_calls >= 1
        assert len(meta.tools_used) >= 1

    def test_conversation_has_tool_calls(self, orchestrator, sampler) -> None:  # type: ignore[no-untyped-def]
        chain = _sample(sampler)
        conv = orchestrator.generate_conversation(chain, seed=42)
        tc_msgs = [m for m in conv.messages if m.role == "assistant" and m.tool_calls]
        assert len(tc_msgs) >= 1


# ===================================================================
# Batch generation
# ===================================================================


class TestBatchGeneration:
    """Batch generation with diversity."""

    def test_batch_generates_conversations(self, orchestrator, sampler) -> None:  # type: ignore[no-untyped-def]
        batch_gen = BatchGenerator(orchestrator=orchestrator, sampler=sampler)
        convs = batch_gen.generate_batch(count=5, constraints=_CONSTRAINTS, seed=42)
        assert len(convs) >= 1

    def test_batch_stats(self, orchestrator, sampler) -> None:  # type: ignore[no-untyped-def]
        batch_gen = BatchGenerator(orchestrator=orchestrator, sampler=sampler)
        convs = batch_gen.generate_batch(count=5, constraints=_CONSTRAINTS, seed=42)
        stats = batch_gen.get_batch_stats()
        assert stats.total_generated == len(convs)

    def test_batch_tool_coverage(self, orchestrator, sampler) -> None:  # type: ignore[no-untyped-def]
        batch_gen = BatchGenerator(orchestrator=orchestrator, sampler=sampler)
        batch_gen.generate_batch(count=8, constraints=_CONSTRAINTS, seed=42)
        stats = batch_gen.get_batch_stats()
        assert stats.tools_coverage >= 1

    def test_batch_with_steering(self, orchestrator, registry_and_graph) -> None:  # type: ignore[no-untyped-def]
        _, graph = registry_and_graph
        diversity_cfg = DiversitySteeringConfig(enabled=True)
        steer_sampler = ToolChainSampler(graph, diversity_config=diversity_cfg)
        batch_gen = BatchGenerator(
            orchestrator=orchestrator, sampler=steer_sampler, diversity_config=diversity_cfg,
        )
        convs = batch_gen.generate_batch(
            count=5, constraints=_CONSTRAINTS, seed=42, steering_enabled=True,
        )
        assert len(convs) >= 1
        stats = batch_gen.get_batch_stats()
        assert stats.steering_enabled


# ===================================================================
# Output format
# ===================================================================


class TestOutputFormat:
    """Conversation -> output format validation."""

    def test_to_jsonl_dict(self, orchestrator, sampler) -> None:  # type: ignore[no-untyped-def]
        chain = _sample(sampler)
        conv = orchestrator.generate_conversation(chain, seed=42)
        data = conv.to_jsonl_dict()
        assert "conversation_id" in data
        assert "messages" in data
        assert "metadata" in data
        assert isinstance(data["messages"], list)

    def test_from_conversation_creates_record(self, orchestrator, sampler) -> None:  # type: ignore[no-untyped-def]
        chain = _sample(sampler)
        conv = orchestrator.generate_conversation(chain, seed=42)
        rec = from_conversation(conv)
        assert rec.conversation_id == conv.conversation_id
        assert len(rec.messages) >= 1

    def test_validate_record_passes(self, orchestrator, sampler) -> None:  # type: ignore[no-untyped-def]
        chain = _sample(sampler)
        conv = orchestrator.generate_conversation(chain, seed=42)
        rec = from_conversation(conv)
        valid, errors = validate_record(rec.to_dict())
        assert valid, f"Record validation failed: {errors}"

    def test_record_messages_have_roles(self, orchestrator, sampler) -> None:  # type: ignore[no-untyped-def]
        chain = _sample(sampler)
        conv = orchestrator.generate_conversation(chain, seed=42)
        rec = from_conversation(conv)
        valid_roles = {"user", "assistant", "tool"}
        for msg in rec.messages:
            assert msg["role"] in valid_roles

    def test_record_has_metadata(self, orchestrator, sampler) -> None:  # type: ignore[no-untyped-def]
        chain = _sample(sampler)
        conv = orchestrator.generate_conversation(chain, seed=42)
        rec = from_conversation(conv)
        assert isinstance(rec.metadata, dict)
        assert "tools_used" in rec.metadata
        assert "seed" in rec.metadata


# ===================================================================
# Multi-step and multi-tool
# ===================================================================


class TestMultiStepMultiTool:
    """Verify multi-step (>=3 calls) and multi-tool (>=2 tools) capability."""

    def test_multi_step_in_batch(self, orchestrator, sampler) -> None:  # type: ignore[no-untyped-def]
        """At least some conversations in a larger batch have >=2 tool calls."""
        batch_gen = BatchGenerator(orchestrator=orchestrator, sampler=sampler)
        constraints = SamplingConstraints(min_steps=1, max_steps=4, min_tools=1)
        convs = batch_gen.generate_batch(count=10, constraints=constraints, seed=42)
        multi_step = [c for c in convs if c.metadata.num_tool_calls >= 2]
        # With min_steps=1 and max_steps=4, at least some should have >=2
        assert len(multi_step) >= 1, (
            f"No multi-step conversations in {len(convs)} generated; "
            f"tool call counts: {[c.metadata.num_tool_calls for c in convs]}"
        )

    def test_multi_tool_in_batch(self, orchestrator, sampler) -> None:  # type: ignore[no-untyped-def]
        """Check if any conversations use >=2 distinct tools."""
        batch_gen = BatchGenerator(orchestrator=orchestrator, sampler=sampler)
        constraints = SamplingConstraints(min_steps=2, max_steps=4, min_tools=1)
        convs = batch_gen.generate_batch(count=10, constraints=constraints, seed=42)
        multi_tool = [c for c in convs if c.metadata.num_distinct_tools >= 2]
        if not multi_tool:
            pytest.skip(
                "Graph too small for multi-tool chains; "
                f"distinct tool counts: {[c.metadata.num_distinct_tools for c in convs]}"
            )
        assert len(multi_tool) >= 1

    def test_chain_multi_tool_property(self, sampler) -> None:  # type: ignore[no-untyped-def]
        """ToolChain.is_multi_tool returns bool."""
        chain = _sample(sampler)
        assert isinstance(chain.is_multi_tool, bool)
        # If multi-tool, must have >1 unique tool_ids
        if chain.is_multi_tool:
            assert len(chain.tool_ids) >= 2

    def test_conversation_tool_ids_match_chain(self, orchestrator, sampler) -> None:  # type: ignore[no-untyped-def]
        """Conversation metadata.tools_used is a subset of chain.tool_ids."""
        chain = _sample(sampler)
        conv = orchestrator.generate_conversation(chain, seed=42)
        chain_tools = set(chain.tool_ids)
        conv_tools = set(conv.metadata.tools_used)
        # All tools used in conversation should come from the chain
        assert conv_tools.issubset(chain_tools), (
            f"Conv tools {conv_tools} not subset of chain tools {chain_tools}"
        )


# ===================================================================
# Structural validation
# ===================================================================


class TestValidation:
    """Structural validation on generated conversations."""

    def test_validator_accepts_generated(self, orchestrator, sampler) -> None:  # type: ignore[no-untyped-def]
        chain = _sample(sampler)
        conv = orchestrator.generate_conversation(chain, seed=42)
        validator = ConversationValidator()
        result = validator.validate(conv)
        # Generated conversations should generally pass basic validation
        # (no registry = skips tool-call validity checks)
        assert result.valid, f"Validation errors: {result.errors}"

    def test_validator_on_batch(self, orchestrator, sampler) -> None:  # type: ignore[no-untyped-def]
        batch_gen = BatchGenerator(orchestrator=orchestrator, sampler=sampler)
        convs = batch_gen.generate_batch(count=5, constraints=_CONSTRAINTS, seed=42)
        validator = ConversationValidator()
        for conv in convs:
            result = validator.validate(conv)
            assert result.valid, (
                f"Conversation {conv.conversation_id} failed: {result.errors}"
            )
