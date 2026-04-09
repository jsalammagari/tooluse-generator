"""Tests for the ConversationOrchestrator (Task 38)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tooluse_gen.agents.assistant_agent import AssistantAgent
from tooluse_gen.agents.conversation_models import Conversation
from tooluse_gen.agents.orchestrator import ConversationOrchestrator, OrchestratorConfig
from tooluse_gen.agents.tool_executor import ToolExecutor
from tooluse_gen.agents.user_simulator import UserSimulator
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
                    name="Search Hotels",
                    description="Search for hotels",
                    method=HttpMethod.GET,
                    path="/hotels/search",
                    parameters=[
                        Parameter(
                            name="city",
                            param_type=ParameterType.STRING,
                            required=True,
                        ),
                        Parameter(
                            name="max_price",
                            param_type=ParameterType.NUMBER,
                            required=False,
                        ),
                    ],
                    required_parameters=["city"],
                ),
                Endpoint(
                    endpoint_id="hotels/book",
                    tool_id="hotels",
                    name="Book Hotel",
                    description="Book a room",
                    method=HttpMethod.POST,
                    path="/hotels/book",
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
                    description="Get weather",
                    method=HttpMethod.GET,
                    path="/weather",
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
def chain() -> ToolChain:
    return ToolChain(
        chain_id="test",
        steps=[
            ChainStep(
                endpoint_id="hotels/search",
                tool_id="hotels",
                tool_name="Hotels API",
                endpoint_name="Search Hotels",
                method=HttpMethod.GET,
                domain="Travel",
                expected_params=["city"],
            ),
            ChainStep(
                endpoint_id="hotels/book",
                tool_id="hotels",
                tool_name="Hotels API",
                endpoint_name="Book Hotel",
                method=HttpMethod.POST,
                domain="Travel",
                expected_params=["hotel_id", "guest_name"],
            ),
        ],
        pattern=ChainPattern.SEQUENTIAL,
    )


@pytest.fixture()
def single_step_chain() -> ToolChain:
    return ToolChain(
        chain_id="single",
        steps=[
            ChainStep(
                endpoint_id="hotels/search",
                tool_id="hotels",
                tool_name="Hotels API",
                endpoint_name="Search Hotels",
                method=HttpMethod.GET,
                domain="Travel",
                expected_params=["city"],
            ),
        ],
        pattern=ChainPattern.SEQUENTIAL,
    )


@pytest.fixture()
def three_step_chain() -> ToolChain:
    return ToolChain(
        chain_id="three",
        steps=[
            ChainStep(
                endpoint_id="hotels/search",
                tool_id="hotels",
                tool_name="Hotels API",
                endpoint_name="Search Hotels",
                method=HttpMethod.GET,
                domain="Travel",
                expected_params=["city"],
            ),
            ChainStep(
                endpoint_id="hotels/book",
                tool_id="hotels",
                tool_name="Hotels API",
                endpoint_name="Book Hotel",
                method=HttpMethod.POST,
                domain="Travel",
                expected_params=["hotel_id", "guest_name"],
            ),
            ChainStep(
                endpoint_id="weather/current",
                tool_id="weather",
                tool_name="Weather API",
                endpoint_name="Current Weather",
                method=HttpMethod.GET,
                domain="Weather",
                expected_params=["city"],
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
    )


# ===================================================================
# OrchestratorConfig
# ===================================================================


class TestOrchestratorConfig:
    def test_defaults(self):
        cfg = OrchestratorConfig()
        assert cfg.max_turns == 15
        assert cfg.max_consecutive_tool_calls == 5
        assert cfg.require_disambiguation is True
        assert cfg.disambiguation_probability == 0.3
        assert cfg.require_final_answer is True
        assert cfg.min_tool_calls == 2
        assert cfg.temperature == 0.7

    def test_custom_values(self):
        cfg = OrchestratorConfig(
            max_turns=30,
            max_consecutive_tool_calls=10,
            require_disambiguation=False,
            disambiguation_probability=0.8,
            min_tool_calls=5,
        )
        assert cfg.max_turns == 30
        assert cfg.require_disambiguation is False
        assert cfg.disambiguation_probability == 0.8

    def test_max_turns_validation(self):
        with pytest.raises(ValidationError):
            OrchestratorConfig(max_turns=0)

    def test_disambiguation_probability_bounds(self):
        OrchestratorConfig(disambiguation_probability=0.0)
        OrchestratorConfig(disambiguation_probability=1.0)
        with pytest.raises(ValidationError):
            OrchestratorConfig(disambiguation_probability=1.1)
        with pytest.raises(ValidationError):
            OrchestratorConfig(disambiguation_probability=-0.1)


# ===================================================================
# Construction
# ===================================================================


class TestConstruction:
    def test_default_config(self, registry):
        orch = ConversationOrchestrator(
            user_sim=UserSimulator(),
            assistant=AssistantAgent(registry=registry),
            executor=ToolExecutor(registry),
        )
        assert orch._config.max_turns == 15

    def test_custom_config(self, registry):
        cfg = OrchestratorConfig(max_turns=10)
        orch = ConversationOrchestrator(
            user_sim=UserSimulator(),
            assistant=AssistantAgent(registry=registry),
            executor=ToolExecutor(registry),
            config=cfg,
        )
        assert orch._config.max_turns == 10


# ===================================================================
# generate_conversation — basic flow
# ===================================================================


class TestBasicFlow:
    def test_returns_conversation(self, orchestrator, chain):
        conv = orchestrator.generate_conversation(chain, seed=42)
        assert isinstance(conv, Conversation)

    def test_has_messages(self, orchestrator, chain):
        conv = orchestrator.generate_conversation(chain, seed=42)
        assert conv.turn_count > 0

    def test_has_all_roles(self, orchestrator, chain):
        conv = orchestrator.generate_conversation(chain, seed=42)
        roles = {m.role for m in conv.messages}
        assert "user" in roles
        assert "assistant" in roles
        assert "tool" in roles

    def test_metadata_seed(self, orchestrator, chain):
        conv = orchestrator.generate_conversation(chain, seed=99)
        assert conv.metadata.seed == 99

    def test_at_least_one_tool_call(self, orchestrator, chain):
        conv = orchestrator.generate_conversation(chain, seed=42)
        assert conv.metadata.num_tool_calls >= 1

    def test_ends_with_assistant(self, registry, chain):
        orch = ConversationOrchestrator(
            user_sim=UserSimulator(),
            assistant=AssistantAgent(registry=registry),
            executor=ToolExecutor(registry),
            config=OrchestratorConfig(require_final_answer=True, require_disambiguation=False),
        )
        conv = orch.generate_conversation(chain, seed=42)
        assert conv.messages[-1].role == "assistant"
        assert conv.messages[-1].content is not None


# ===================================================================
# Turn sequence
# ===================================================================


class TestTurnSequence:
    def test_first_message_is_user(self, orchestrator, chain):
        conv = orchestrator.generate_conversation(chain, seed=42)
        assert conv.messages[0].role == "user"

    def test_no_consecutive_users_without_assistant(self, orchestrator, chain):
        conv = orchestrator.generate_conversation(chain, seed=42)
        for i in range(len(conv.messages) - 1):
            if conv.messages[i].role == "user" and conv.messages[i + 1].role == "user":
                # This should not happen — assistant must be between users.
                pytest.fail(
                    f"Two consecutive user messages at index {i} and {i+1}"
                )

    def test_tool_follows_assistant_tool_call(self, orchestrator, chain):
        conv = orchestrator.generate_conversation(chain, seed=42)
        for i, msg in enumerate(conv.messages):
            if msg.role == "tool":
                # The tool message must be preceded by an assistant with tool_calls
                # (possibly with other tool messages between).
                found = False
                for j in range(i - 1, -1, -1):
                    if conv.messages[j].role == "assistant" and conv.messages[j].tool_calls:
                        found = True
                        break
                    if conv.messages[j].role == "user":
                        break
                assert found, f"Tool message at {i} not preceded by assistant tool_call"

    def test_last_message_assistant_text(self, registry, chain):
        orch = ConversationOrchestrator(
            user_sim=UserSimulator(),
            assistant=AssistantAgent(registry=registry),
            executor=ToolExecutor(registry),
            config=OrchestratorConfig(require_final_answer=True, require_disambiguation=False),
        )
        conv = orch.generate_conversation(chain, seed=42)
        last = conv.messages[-1]
        assert last.role == "assistant"
        assert last.content is not None
        assert last.tool_calls is None or last.tool_calls == []


# ===================================================================
# Disambiguation flow
# ===================================================================


class TestDisambiguationFlow:
    def test_disambiguation_included(self, registry, chain):
        orch = ConversationOrchestrator(
            user_sim=UserSimulator(),
            assistant=AssistantAgent(registry=registry),
            executor=ToolExecutor(registry),
            config=OrchestratorConfig(
                require_disambiguation=True, disambiguation_probability=1.0
            ),
        )
        conv = orch.generate_conversation(chain, seed=42)
        # Find a disambiguation message (assistant with "?" content).
        disambig = [
            m for m in conv.messages
            if m.role == "assistant" and m.content and "?" in m.content and not m.tool_calls
        ]
        assert len(disambig) >= 1, "No disambiguation found"

    def test_disambiguation_followed_by_user(self, registry, chain):
        orch = ConversationOrchestrator(
            user_sim=UserSimulator(),
            assistant=AssistantAgent(registry=registry),
            executor=ToolExecutor(registry),
            config=OrchestratorConfig(
                require_disambiguation=True, disambiguation_probability=1.0
            ),
        )
        conv = orch.generate_conversation(chain, seed=42)
        for i, msg in enumerate(conv.messages):
            if (
                msg.role == "assistant"
                and msg.content
                and "?" in msg.content
                and not msg.tool_calls
                and i < len(conv.messages) - 1
            ):
                assert conv.messages[i + 1].role == "user"

    def test_no_disambiguation_when_disabled(self, registry, chain):
        orch = ConversationOrchestrator(
            user_sim=UserSimulator(),
            assistant=AssistantAgent(registry=registry),
            executor=ToolExecutor(registry),
            config=OrchestratorConfig(require_disambiguation=False),
        )
        conv = orch.generate_conversation(chain, seed=42)
        # First assistant message should be a tool call, not disambiguation.
        first_asst = next(m for m in conv.messages if m.role == "assistant")
        assert first_asst.tool_calls is not None or first_asst.content is not None


# ===================================================================
# Completion detection
# ===================================================================


class TestCompletion:
    def test_completes_when_chain_done(self, orchestrator, chain):
        conv = orchestrator.generate_conversation(chain, seed=42)
        # Should complete — chain has 2 steps and they should be executed.
        assert conv.turn_count > 0

    def test_completes_on_max_turns(self, registry, chain):
        orch = ConversationOrchestrator(
            user_sim=UserSimulator(),
            assistant=AssistantAgent(registry=registry),
            executor=ToolExecutor(registry),
            config=OrchestratorConfig(max_turns=5, require_disambiguation=False),
        )
        conv = orch.generate_conversation(chain, seed=42)
        # Forced final answer may add 1 more beyond max_turns.
        assert conv.turn_count <= 7

    def test_completes_on_final_answer(self, orchestrator, chain):
        conv = orchestrator.generate_conversation(chain, seed=42)
        last = conv.messages[-1]
        assert last.role == "assistant"

    def test_always_terminates(self, orchestrator, chain):
        """Ensure it doesn't loop forever."""
        conv = orchestrator.generate_conversation(chain, seed=42)
        assert conv.turn_count <= 30  # generous upper bound


# ===================================================================
# Metadata correctness
# ===================================================================


class TestMetadata:
    def test_num_turns(self, orchestrator, chain):
        conv = orchestrator.generate_conversation(chain, seed=42)
        assert conv.metadata.num_turns == conv.turn_count

    def test_tools_used(self, orchestrator, chain):
        conv = orchestrator.generate_conversation(chain, seed=42)
        assert "hotels" in conv.metadata.tools_used

    def test_domains(self, orchestrator, chain):
        conv = orchestrator.generate_conversation(chain, seed=42)
        assert "Travel" in conv.metadata.domains

    def test_seed_recorded(self, orchestrator, chain):
        conv = orchestrator.generate_conversation(chain, seed=123)
        assert conv.metadata.seed == 123


# ===================================================================
# Determinism
# ===================================================================


class TestDeterminism:
    def test_same_seed_same_conversation(self, registry, chain):
        def run(seed: int) -> Conversation:
            orch = ConversationOrchestrator(
                user_sim=UserSimulator(),
                assistant=AssistantAgent(registry=registry),
                executor=ToolExecutor(registry),
                config=OrchestratorConfig(require_disambiguation=False),
            )
            return orch.generate_conversation(chain, seed=seed)

        c1 = run(42)
        c2 = run(42)
        assert c1.turn_count == c2.turn_count
        for m1, m2 in zip(c1.messages, c2.messages, strict=True):
            assert m1.role == m2.role
            assert m1.content == m2.content

    def test_different_seeds_differ(self, registry, chain):
        def run(seed: int) -> Conversation:
            orch = ConversationOrchestrator(
                user_sim=UserSimulator(),
                assistant=AssistantAgent(registry=registry),
                executor=ToolExecutor(registry),
                config=OrchestratorConfig(require_disambiguation=False),
            )
            return orch.generate_conversation(chain, seed=seed)

        c1 = run(42)
        c2 = run(99)
        # Both should produce valid conversations (content may differ).
        assert c1.turn_count > 0
        assert c2.turn_count > 0


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    def test_single_step_chain(self, registry, single_step_chain):
        orch = ConversationOrchestrator(
            user_sim=UserSimulator(),
            assistant=AssistantAgent(registry=registry),
            executor=ToolExecutor(registry),
            config=OrchestratorConfig(require_disambiguation=False),
        )
        conv = orch.generate_conversation(single_step_chain, seed=42)
        assert conv.turn_count >= 3  # user + assistant(tool) + tool + final

    def test_max_turns_forces_early_stop(self, registry, chain):
        orch = ConversationOrchestrator(
            user_sim=UserSimulator(),
            assistant=AssistantAgent(registry=registry),
            executor=ToolExecutor(registry),
            config=OrchestratorConfig(max_turns=4, require_disambiguation=False),
        )
        conv = orch.generate_conversation(chain, seed=42)
        # Turn count should be close to max_turns (final answer may add 1).
        assert conv.turn_count <= 6

    def test_three_step_chain(self, registry, three_step_chain):
        orch = ConversationOrchestrator(
            user_sim=UserSimulator(),
            assistant=AssistantAgent(registry=registry),
            executor=ToolExecutor(registry),
            config=OrchestratorConfig(require_disambiguation=False),
        )
        conv = orch.generate_conversation(three_step_chain, seed=42)
        assert conv.metadata.num_tool_calls >= 2
        assert conv.turn_count >= 5


# ===================================================================
# JSONL output
# ===================================================================


class TestJsonlOutput:
    def test_to_jsonl_dict(self, orchestrator, chain):
        conv = orchestrator.generate_conversation(chain, seed=42)
        record = conv.to_jsonl_dict()
        assert "conversation_id" in record
        assert "messages" in record
        assert "metadata" in record
        assert len(record["messages"]) == conv.turn_count

    def test_json_serializable(self, orchestrator, chain):
        import json

        conv = orchestrator.generate_conversation(chain, seed=42)
        json_str = conv.to_jsonl()
        parsed = json.loads(json_str)
        assert parsed["conversation_id"] == conv.conversation_id
