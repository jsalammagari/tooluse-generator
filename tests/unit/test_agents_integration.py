"""Integration tests for the full Phase 6 multi-agent pipeline (Task 42).

Exercises: registry → graph → sampler → orchestrator → conversation → JSONL.
All agents run in offline mode — no LLM calls.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tooluse_gen.agents import (
    ArgumentGenerator,
    AssistantAgent,
    AssistantResponse,
    BatchGenerator,
    BatchStats,
    Conversation,
    ConversationContext,
    ConversationEvent,
    ConversationMetadata,
    ConversationOrchestrator,
    ConversationState,
    ConversationStateMachine,
    GenerationConfig,
    GroundingTracker,
    InvalidTransitionError,
    JudgeScores,
    Message,
    OrchestratorConfig,
    SchemaBasedGenerator,
    StateTransition,
    ToolCallRequest,
    ToolCallResponse,
    ToolExecutor,
    UserSimulator,
    ValuePool,
    ValueProvenance,
    format_available_values,
    format_grounding_context,
    format_value_for_prompt,
)
from tooluse_gen.graph.builder import GraphBuilder
from tooluse_gen.graph.chain_models import SamplingConstraints
from tooluse_gen.graph.diversity import DiversitySteeringConfig
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
        self,
        texts: list[str],
        batch_size: int = 256,
        show_progress: bool = False,
    ) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]


# ---------------------------------------------------------------------------
# Module-scoped fixtures (built once)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
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
                    description="Search for hotels in a city",
                    method=HttpMethod.GET,
                    path="/hotels/search",
                    parameters=[
                        Parameter(name="city", param_type=ParameterType.STRING, required=True),
                        Parameter(name="check_in", param_type=ParameterType.DATE, required=False),
                        Parameter(name="max_price", param_type=ParameterType.NUMBER, required=False),
                    ],
                    required_parameters=["city"],
                ),
                Endpoint(
                    endpoint_id="hotels/book",
                    tool_id="hotels",
                    name="Book Hotel",
                    description="Book a hotel room",
                    method=HttpMethod.POST,
                    path="/hotels/book",
                    parameters=[
                        Parameter(name="hotel_id", param_type=ParameterType.STRING, required=True),
                        Parameter(name="guest_name", param_type=ParameterType.STRING, required=True),
                    ],
                    required_parameters=["hotel_id", "guest_name"],
                ),
                Endpoint(
                    endpoint_id="hotels/reviews",
                    tool_id="hotels",
                    name="Hotel Reviews",
                    description="Get reviews",
                    method=HttpMethod.GET,
                    path="/hotels/{id}/reviews",
                    parameters=[
                        Parameter(name="hotel_id", param_type=ParameterType.STRING, required=True),
                    ],
                    required_parameters=["hotel_id"],
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
                    description="Get current weather",
                    method=HttpMethod.GET,
                    path="/weather",
                    parameters=[
                        Parameter(name="city", param_type=ParameterType.STRING, required=True),
                    ],
                    required_parameters=["city"],
                ),
                Endpoint(
                    endpoint_id="weather/forecast",
                    tool_id="weather",
                    name="Forecast",
                    description="Multi-day forecast",
                    method=HttpMethod.GET,
                    path="/forecast",
                    parameters=[
                        Parameter(name="city", param_type=ParameterType.STRING, required=True),
                    ],
                    required_parameters=["city"],
                ),
            ],
        )
    )
    reg.add_tool(
        Tool(
            tool_id="restaurants",
            name="Restaurant API",
            domain="Food",
            endpoints=[
                Endpoint(
                    endpoint_id="restaurants/search",
                    tool_id="restaurants",
                    name="Search Restaurants",
                    description="Find restaurants",
                    method=HttpMethod.GET,
                    path="/restaurants",
                    parameters=[
                        Parameter(name="city", param_type=ParameterType.STRING, required=True),
                    ],
                    required_parameters=["city"],
                ),
                Endpoint(
                    endpoint_id="restaurants/reserve",
                    tool_id="restaurants",
                    name="Make Reservation",
                    description="Reserve a table",
                    method=HttpMethod.POST,
                    path="/reservations",
                    parameters=[
                        Parameter(
                            name="restaurant_id",
                            param_type=ParameterType.STRING,
                            required=True,
                        ),
                        Parameter(
                            name="party_size",
                            param_type=ParameterType.INTEGER,
                            required=True,
                        ),
                    ],
                    required_parameters=["restaurant_id", "party_size"],
                ),
            ],
        )
    )
    return reg


@pytest.fixture(scope="module")
def graph(registry: ToolRegistry) -> object:  # nx.DiGraph
    config = GraphConfig(
        include_tool_nodes=True,
        include_domain_edges=True,
        include_semantic_edges=False,
        max_edges_per_node=20,
    )
    return GraphBuilder(config=config, embedding_service=_MockEmb()).build(registry)


@pytest.fixture(scope="module")
def sampler(graph: object) -> ToolChainSampler:
    return ToolChainSampler(
        graph,  # type: ignore[arg-type]
        SamplerConfig(max_iterations=300, max_retries=30),
        DiversitySteeringConfig(enabled=True),
    )


@pytest.fixture(scope="module")
def conversation(registry: ToolRegistry, sampler: ToolChainSampler) -> Conversation:
    """Generate a single conversation for many tests to inspect."""
    rng = np.random.default_rng(42)
    chain = sampler.sample_chain(
        SamplingConstraints(min_steps=2, max_steps=3, min_tools=1), rng
    )
    orch = ConversationOrchestrator(
        user_sim=UserSimulator(),
        assistant=AssistantAgent(registry=registry),
        executor=ToolExecutor(registry),
        config=OrchestratorConfig(require_disambiguation=False, require_final_answer=True),
    )
    return orch.generate_conversation(chain, seed=42)


@pytest.fixture(scope="module")
def jsonl_record(conversation: Conversation) -> dict:  # type: ignore[type-arg]
    return conversation.to_jsonl_dict()


# ===================================================================
# 1. Import completeness
# ===================================================================


class TestImportCompleteness:
    def test_all_agent_classes_importable(self):
        symbols = [
            ArgumentGenerator, AssistantAgent, AssistantResponse,
            BatchGenerator, BatchStats, Conversation, ConversationContext,
            ConversationEvent, ConversationMetadata, ConversationOrchestrator,
            ConversationState, ConversationStateMachine, GenerationConfig,
            GroundingTracker, InvalidTransitionError, JudgeScores, Message,
            OrchestratorConfig, SchemaBasedGenerator, StateTransition,
            ToolCallRequest, ToolCallResponse, ToolExecutor, UserSimulator,
            ValuePool, ValueProvenance, format_available_values,
            format_grounding_context, format_value_for_prompt,
        ]
        assert len(symbols) == 29

    def test_conversation_models_importable(self):
        assert Conversation is not None
        assert ConversationMetadata is not None
        assert Message is not None
        assert JudgeScores is not None
        assert GenerationConfig is not None

    def test_execution_models_importable(self):
        assert ConversationContext is not None
        assert ToolCallRequest is not None
        assert ToolCallResponse is not None

    def test_orchestrator_importable(self):
        assert ConversationOrchestrator is not None
        assert OrchestratorConfig is not None

    def test_state_machine_importable(self):
        assert ConversationState is not None
        assert ConversationEvent is not None
        assert ConversationStateMachine is not None
        assert InvalidTransitionError is not None
        assert StateTransition is not None


# ===================================================================
# 2. Single conversation pipeline
# ===================================================================


class TestSingleConversation:
    def test_returns_conversation(self, conversation):
        assert isinstance(conversation, Conversation)

    def test_has_messages(self, conversation):
        assert conversation.turn_count > 0

    def test_has_all_roles(self, conversation):
        roles = {m.role for m in conversation.messages}
        assert "user" in roles
        assert "assistant" in roles
        assert "tool" in roles

    def test_first_message_user(self, conversation):
        assert conversation.messages[0].role == "user"

    def test_last_message_assistant(self, conversation):
        last = conversation.messages[-1]
        assert last.role == "assistant"
        assert last.content is not None
        assert not last.tool_calls

    def test_tool_calls_present(self, conversation):
        tc_msgs = [m for m in conversation.messages if m.tool_calls]
        assert len(tc_msgs) >= 1

    def test_tool_responses_are_dicts(self, jsonl_record):
        for msg in jsonl_record["messages"]:
            if msg["role"] == "tool":
                assert isinstance(msg["content"], dict)

    def test_metadata_complete(self, conversation):
        m = conversation.metadata
        assert m.num_turns == conversation.turn_count
        assert m.num_tool_calls >= 1
        assert len(m.tools_used) >= 1
        assert len(m.domains) >= 1
        assert m.pattern != ""
        assert m.generation_time_ms >= 0
        assert isinstance(m.endpoints_called, list)
        assert len(m.endpoints_called) >= 1
        assert isinstance(m.disambiguation_count, int)
        assert isinstance(m.grounding_stats, dict)


# ===================================================================
# 3. JSONL format compliance
# ===================================================================


class TestJsonlFormat:
    def test_has_conversation_id(self, jsonl_record):
        assert isinstance(jsonl_record["conversation_id"], str)
        assert len(jsonl_record["conversation_id"]) > 0

    def test_has_messages_list(self, jsonl_record):
        assert isinstance(jsonl_record["messages"], list)
        assert len(jsonl_record["messages"]) > 0

    def test_messages_have_role_and_content(self, jsonl_record):
        for msg in jsonl_record["messages"]:
            assert "role" in msg
            assert "content" in msg

    def test_tool_calls_format(self, jsonl_record):
        for msg in jsonl_record["messages"]:
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    assert "endpoint" in tc
                    assert "arguments" in tc
                    assert "tool_name" in tc
                    assert "call_id" in tc

    def test_tool_content_is_dict(self, jsonl_record):
        for msg in jsonl_record["messages"]:
            if msg["role"] == "tool":
                assert isinstance(msg["content"], dict)

    def test_is_valid_json(self, conversation):
        json_str = conversation.to_jsonl()
        parsed = json.loads(json_str)
        assert parsed["conversation_id"] == conversation.conversation_id


# ===================================================================
# 4. Batch generation pipeline
# ===================================================================


class TestBatchGeneration:
    def test_batch_generation(self, registry, sampler):
        orch = ConversationOrchestrator(
            user_sim=UserSimulator(),
            assistant=AssistantAgent(registry=registry),
            executor=ToolExecutor(registry),
            config=OrchestratorConfig(require_disambiguation=False),
        )
        gen = BatchGenerator(orchestrator=orch, sampler=sampler)
        convs = gen.generate_batch(
            count=5,
            constraints=SamplingConstraints(min_steps=2, max_steps=3, min_tools=1),
            seed=42,
        )
        assert len(convs) >= 3

    def test_batch_all_valid(self, registry, sampler):
        orch = ConversationOrchestrator(
            user_sim=UserSimulator(),
            assistant=AssistantAgent(registry=registry),
            executor=ToolExecutor(registry),
            config=OrchestratorConfig(require_disambiguation=False),
        )
        gen = BatchGenerator(orchestrator=orch, sampler=sampler)
        convs = gen.generate_batch(
            count=3,
            constraints=SamplingConstraints(min_steps=2, max_steps=3, min_tools=1),
            seed=42,
        )
        for conv in convs:
            roles = {m.role for m in conv.messages}
            assert "user" in roles and "assistant" in roles and "tool" in roles

    def test_batch_metadata_seeds(self, registry, sampler):
        orch = ConversationOrchestrator(
            user_sim=UserSimulator(),
            assistant=AssistantAgent(registry=registry),
            executor=ToolExecutor(registry),
            config=OrchestratorConfig(require_disambiguation=False),
        )
        gen = BatchGenerator(orchestrator=orch, sampler=sampler)
        convs = gen.generate_batch(
            count=3,
            constraints=SamplingConstraints(min_steps=2, max_steps=3, min_tools=1),
            seed=100,
        )
        for i, conv in enumerate(convs):
            assert conv.metadata.seed == 100 + i

    def test_batch_stats(self, registry, sampler):
        orch = ConversationOrchestrator(
            user_sim=UserSimulator(),
            assistant=AssistantAgent(registry=registry),
            executor=ToolExecutor(registry),
            config=OrchestratorConfig(require_disambiguation=False),
        )
        gen = BatchGenerator(orchestrator=orch, sampler=sampler)
        gen.generate_batch(
            count=3,
            constraints=SamplingConstraints(min_steps=2, max_steps=3, min_tools=1),
            seed=42,
        )
        stats = gen.get_batch_stats()
        assert stats.total_generated >= 1
        assert stats.average_turns > 0

    def test_batch_diversity_metrics(self, registry, sampler):
        orch = ConversationOrchestrator(
            user_sim=UserSimulator(),
            assistant=AssistantAgent(registry=registry),
            executor=ToolExecutor(registry),
            config=OrchestratorConfig(require_disambiguation=False),
        )
        gen = BatchGenerator(orchestrator=orch, sampler=sampler)
        gen.generate_batch(
            count=5,
            constraints=SamplingConstraints(min_steps=2, max_steps=3, min_tools=1),
            seed=42,
        )
        stats = gen.get_batch_stats()
        assert stats.diversity_metrics is not None


# ===================================================================
# 5. Metadata completeness
# ===================================================================


class TestMetadataCompleteness:
    def test_endpoints_called(self, conversation):
        m = conversation.metadata
        assert isinstance(m.endpoints_called, list)
        assert m.num_tool_calls == len(m.endpoints_called)

    def test_grounding_stats(self, conversation):
        gs = conversation.metadata.grounding_stats
        assert "grounded_args" in gs
        assert "fresh_args" in gs
        assert "total_args" in gs
        assert gs["total_args"] == gs["grounded_args"] + gs["fresh_args"]

    def test_disambiguation_count(self, conversation):
        assert isinstance(conversation.metadata.disambiguation_count, int)
        assert conversation.metadata.disambiguation_count >= 0

    def test_pattern(self, conversation):
        assert conversation.metadata.pattern in (
            "sequential",
            "parallel",
            "branch_and_merge",
            "iterative",
        )

    def test_generation_time(self, conversation):
        assert conversation.metadata.generation_time_ms >= 0


# ===================================================================
# 6. Disambiguation flow
# ===================================================================


class TestDisambiguationFlow:
    def test_disambiguation_increases_turns(self, registry, sampler):
        rng = np.random.default_rng(42)
        chain = sampler.sample_chain(
            SamplingConstraints(min_steps=2, max_steps=3, min_tools=1), rng
        )

        orch_no = ConversationOrchestrator(
            user_sim=UserSimulator(),
            assistant=AssistantAgent(registry=registry),
            executor=ToolExecutor(registry),
            config=OrchestratorConfig(require_disambiguation=False),
        )
        conv_no = orch_no.generate_conversation(chain, seed=42)

        orch_yes = ConversationOrchestrator(
            user_sim=UserSimulator(),
            assistant=AssistantAgent(registry=registry),
            executor=ToolExecutor(registry),
            config=OrchestratorConfig(
                require_disambiguation=True, disambiguation_probability=1.0
            ),
        )
        conv_yes = orch_yes.generate_conversation(chain, seed=42)

        assert conv_yes.turn_count >= conv_no.turn_count

    def test_disambiguation_metadata_tracked(self, registry, sampler):
        rng = np.random.default_rng(42)
        chain = sampler.sample_chain(
            SamplingConstraints(min_steps=2, max_steps=3, min_tools=1), rng
        )
        orch = ConversationOrchestrator(
            user_sim=UserSimulator(),
            assistant=AssistantAgent(registry=registry),
            executor=ToolExecutor(registry),
            config=OrchestratorConfig(
                require_disambiguation=True, disambiguation_probability=1.0
            ),
        )
        conv = orch.generate_conversation(chain, seed=42)
        assert conv.metadata.disambiguation_count >= 1


# ===================================================================
# 7. Determinism
# ===================================================================


class TestDeterminism:
    def test_deterministic_single(self, registry, sampler):
        rng1 = np.random.default_rng(42)
        chain = sampler.sample_chain(
            SamplingConstraints(min_steps=2, max_steps=3, min_tools=1), rng1
        )

        def run() -> Conversation:
            orch = ConversationOrchestrator(
                user_sim=UserSimulator(),
                assistant=AssistantAgent(registry=registry),
                executor=ToolExecutor(registry),
                config=OrchestratorConfig(require_disambiguation=False),
            )
            return orch.generate_conversation(chain, seed=42)

        c1 = run()
        c2 = run()
        assert c1.turn_count == c2.turn_count
        for m1, m2 in zip(c1.messages, c2.messages, strict=True):
            assert m1.role == m2.role
            assert m1.content == m2.content

    def test_deterministic_batch(self, registry, sampler):
        def run() -> list[Conversation]:
            orch = ConversationOrchestrator(
                user_sim=UserSimulator(),
                assistant=AssistantAgent(registry=registry),
                executor=ToolExecutor(registry),
                config=OrchestratorConfig(require_disambiguation=False),
            )
            gen = BatchGenerator(
                orchestrator=orch,
                sampler=ToolChainSampler(
                    sampler._graph,  # type: ignore[attr-defined]
                    SamplerConfig(max_iterations=300, max_retries=30),
                ),
            )
            return gen.generate_batch(
                count=3,
                constraints=SamplingConstraints(min_steps=2, max_steps=3, min_tools=1),
                seed=42,
            )

        b1 = run()
        b2 = run()
        assert len(b1) == len(b2)
        for c1, c2 in zip(b1, b2, strict=True):
            assert c1.turn_count == c2.turn_count


# ===================================================================
# 8. State machine integration
# ===================================================================


class TestStateMachineIntegration:
    def test_state_machine_standalone(self):
        sm = ConversationStateMachine()
        sm.transition(ConversationEvent.START)
        sm.transition(ConversationEvent.USER_MESSAGE)
        sm.transition(ConversationEvent.ASSISTANT_TOOL_CALL)
        sm.transition(ConversationEvent.TOOL_RESULT)
        sm.transition(ConversationEvent.USER_MESSAGE)
        sm.transition(ConversationEvent.ASSISTANT_FINAL)
        assert sm.state == ConversationState.COMPLETE
        assert sm.is_terminal
        assert len(sm.history) == 6

    def test_orchestrator_works_with_state_machine(self, registry, sampler):
        rng = np.random.default_rng(42)
        chain = sampler.sample_chain(
            SamplingConstraints(min_steps=2, max_steps=3, min_tools=1), rng
        )
        orch = ConversationOrchestrator(
            user_sim=UserSimulator(),
            assistant=AssistantAgent(registry=registry),
            executor=ToolExecutor(registry),
            config=OrchestratorConfig(require_disambiguation=False),
        )
        conv = orch.generate_conversation(chain, seed=42)
        assert conv.turn_count >= 4
        roles = {m.role for m in conv.messages}
        assert "user" in roles and "assistant" in roles and "tool" in roles
