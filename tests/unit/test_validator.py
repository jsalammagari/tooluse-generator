"""Tests for the ConversationValidator (Task 44)."""

from __future__ import annotations

import pytest

from tooluse_gen.agents.conversation_models import Conversation, Message
from tooluse_gen.agents.execution_models import ToolCallRequest
from tooluse_gen.evaluation.validator import ConversationValidator
from tooluse_gen.registry.models import Endpoint, HttpMethod, Parameter, ParameterType, Tool
from tooluse_gen.registry.registry import ToolRegistry

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _conv(messages: list[Message]) -> Conversation:
    return Conversation(messages=messages)


def _user(content: str = "Hello") -> Message:
    return Message(role="user", content=content)


def _assistant(content: str = "Sure!") -> Message:
    return Message(role="assistant", content=content)


def _assistant_tc(
    endpoint_id: str = "hotels/search",
    tool_id: str = "hotels",
    arguments: dict | None = None,
) -> Message:
    tc = ToolCallRequest(
        endpoint_id=endpoint_id,
        tool_id=tool_id,
        tool_name="Hotels API",
        endpoint_name="Search",
        arguments=arguments if arguments is not None else {"city": "Paris"},
    )
    return Message(role="assistant", tool_calls=[tc])


def _tool(output: dict | None = None, call_id: str = "c1") -> Message:
    return Message(
        role="tool", tool_call_id=call_id, tool_output=output or {"id": "htl_1"}
    )


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
                    name="Search",
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
                    name="Book",
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
    return reg


# ===================================================================
# validate — overall
# ===================================================================


class TestValidateOverall:
    def test_valid_conversation(self):
        conv = _conv([_user(), _assistant_tc(), _tool(), _assistant()])
        r = ConversationValidator().validate(conv)
        assert r.valid
        assert r.errors == []

    def test_invalid_conversation(self):
        conv = _conv([])
        r = ConversationValidator().validate(conv)
        assert not r.valid
        assert r.error_count > 0

    def test_empty_conversation(self):
        conv = _conv([])
        r = ConversationValidator().validate(conv)
        assert not r.valid
        assert "Conversation has no messages" in r.errors


# ===================================================================
# _check_message_structure
# ===================================================================


class TestMessageStructure:
    def test_valid_structure(self):
        conv = _conv([_user(), _assistant_tc(), _tool(), _assistant()])
        errs = ConversationValidator()._check_message_structure(conv)
        assert errs == []

    def test_no_messages(self):
        conv = _conv([])
        errs = ConversationValidator()._check_message_structure(conv)
        assert "Conversation has no messages" in errs

    def test_consecutive_users(self):
        conv = _conv([_user("a"), _user("b"), _assistant()])
        errs = ConversationValidator()._check_message_structure(conv)
        assert any("Consecutive user" in e for e in errs)

    def test_tool_without_preceding_tc(self):
        conv = _conv([_user(), _assistant(), _tool()])
        errs = ConversationValidator()._check_message_structure(conv)
        assert any("no preceding assistant tool call" in e for e in errs)

    def test_empty_tool_calls_list(self):
        msg = Message(role="assistant", tool_calls=[])
        conv = _conv([_user(), msg, _assistant()])
        errs = ConversationValidator()._check_message_structure(conv)
        assert any("empty tool_calls list" in e for e in errs)

    def test_user_no_content(self):
        msg = Message(role="user", content=None)
        conv = _conv([msg, _assistant()])
        errs = ConversationValidator()._check_message_structure(conv)
        assert any("has no content" in e for e in errs)

    def test_disambiguation_flow(self):
        conv = _conv([
            _user(),
            _assistant("What city?"),
            _user("Paris"),
            _assistant_tc(),
            _tool(),
            _assistant(),
        ])
        errs = ConversationValidator()._check_message_structure(conv)
        assert errs == []


# ===================================================================
# _check_tool_call_validity
# ===================================================================


class TestToolCallValidity:
    def test_valid_tool_call(self, registry):
        conv = _conv([
            _user(),
            _assistant_tc("hotels/search", arguments={"city": "Paris"}),
            _tool(),
            _assistant(),
        ])
        errs = ConversationValidator(registry=registry)._check_tool_call_validity(conv)
        assert errs == []

    def test_missing_endpoint(self, registry):
        conv = _conv([
            _user(),
            _assistant_tc("nonexistent/ep"),
            _tool(),
            _assistant(),
        ])
        errs = ConversationValidator(registry=registry)._check_tool_call_validity(conv)
        assert any("not found in registry" in e for e in errs)

    def test_missing_required_param(self, registry):
        conv = _conv([
            _user(),
            _assistant_tc("hotels/search", arguments={}),
            _tool(),
            _assistant(),
        ])
        errs = ConversationValidator(registry=registry)._check_tool_call_validity(conv)
        assert any("city" in e for e in errs)

    def test_no_registry_skips(self):
        conv = _conv([_user(), _assistant_tc("anything"), _tool(), _assistant()])
        errs = ConversationValidator(registry=None)._check_tool_call_validity(conv)
        assert errs == []

    def test_multiple_calls_one_invalid(self, registry):
        conv = _conv([
            _user(),
            _assistant_tc("hotels/search", arguments={"city": "Rome"}),
            _tool(),
            _assistant_tc("hotels/book", arguments={}),  # missing hotel_id, guest_name
            _tool(),
            _assistant(),
        ])
        errs = ConversationValidator(registry=registry)._check_tool_call_validity(conv)
        assert any("hotel_id" in e for e in errs)
        assert any("guest_name" in e for e in errs)
        # First call should be fine
        assert not any("city" in e for e in errs)


# ===================================================================
# _check_grounding_consistency
# ===================================================================


class TestGroundingConsistency:
    def test_valid_sequence(self):
        conv = _conv([
            _user(),
            _assistant_tc(),
            _tool(),
            _assistant_tc("hotels/book"),
            _tool(),
            _assistant(),
        ])
        errs = ConversationValidator()._check_grounding_consistency(conv)
        assert errs == []

    def test_back_to_back_tool_calls(self):
        conv = _conv([
            _user(),
            _assistant_tc(),
            _assistant_tc("hotels/book"),  # no tool response between
            _tool(),
            _assistant(),
        ])
        errs = ConversationValidator()._check_grounding_consistency(conv)
        assert any("no preceding tool response" in e for e in errs)

    def test_single_tool_call(self):
        conv = _conv([_user(), _assistant_tc(), _tool(), _assistant()])
        errs = ConversationValidator()._check_grounding_consistency(conv)
        assert errs == []


# ===================================================================
# _check_minimum_requirements
# ===================================================================


class TestMinimumRequirements:
    def test_has_tool_calls(self):
        conv = _conv([_user(), _assistant_tc(), _tool(), _assistant()])
        errs = ConversationValidator()._check_minimum_requirements(conv)
        assert errs == []

    def test_no_tool_calls(self):
        conv = _conv([_user(), _assistant()])
        errs = ConversationValidator()._check_minimum_requirements(conv)
        assert "No tool calls in conversation" in errs

    def test_multiple_tool_calls(self):
        conv = _conv([
            _user(),
            _assistant_tc(),
            _tool(),
            _assistant_tc("hotels/book"),
            _tool(),
            _assistant(),
        ])
        errs = ConversationValidator()._check_minimum_requirements(conv)
        assert errs == []


# ===================================================================
# _check_conversation_completeness
# ===================================================================


class TestConversationCompleteness:
    def test_complete(self):
        conv = _conv([_user(), _assistant_tc(), _tool(), _assistant()])
        errs = ConversationValidator()._check_conversation_completeness(conv)
        assert errs == []

    def test_not_start_with_user(self):
        conv = _conv([_assistant(), _user()])
        errs = ConversationValidator()._check_conversation_completeness(conv)
        assert any("does not start with a user" in e for e in errs)

    def test_no_assistant(self):
        conv = _conv([_user()])
        errs = ConversationValidator()._check_conversation_completeness(conv)
        assert any("no assistant messages" in e for e in errs)

    def test_ends_with_tool(self):
        conv = _conv([_user(), _assistant_tc(), _tool()])
        errs = ConversationValidator()._check_conversation_completeness(conv)
        assert any("does not end with an assistant" in e for e in errs)

    def test_ends_with_assistant_tc(self):
        # Assistant tool call at end — not ideal but the orchestrator
        # forces a final text answer, so this may occur in edge cases.
        conv = _conv([_user(), _assistant_tc()])
        errs = ConversationValidator()._check_conversation_completeness(conv)
        # The message role IS assistant, so completeness check passes.
        assert not any("does not end with an assistant" in e for e in errs)


# ===================================================================
# Integration with real pipeline
# ===================================================================


class TestIntegration:
    def test_orchestrator_conversation(self):
        """Validate a conversation generated by the orchestrator."""
        import numpy as np

        from tooluse_gen.agents import (
            AssistantAgent,
            ConversationOrchestrator,
            OrchestratorConfig,
            ToolExecutor,
            UserSimulator,
        )
        from tooluse_gen.graph.builder import GraphBuilder
        from tooluse_gen.graph.chain_models import SamplingConstraints
        from tooluse_gen.graph.embeddings import EmbeddingService
        from tooluse_gen.graph.facade import ToolChainSampler
        from tooluse_gen.graph.models import GraphConfig
        from tooluse_gen.graph.sampler import SamplerConfig

        reg = ToolRegistry()
        reg.add_tool(
            Tool(
                tool_id="h",
                name="H",
                domain="Travel",
                endpoints=[
                    Endpoint(
                        endpoint_id="h/s",
                        tool_id="h",
                        name="Search",
                        method=HttpMethod.GET,
                        path="/s",
                        parameters=[
                            Parameter(
                                name="city",
                                param_type=ParameterType.STRING,
                                required=True,
                            )
                        ],
                        required_parameters=["city"],
                    ),
                    Endpoint(
                        endpoint_id="h/b",
                        tool_id="h",
                        name="Book",
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

        class _MockEmb(EmbeddingService):
            def __init__(self) -> None:
                self._model = None
                self._cache_dir = None

            def embed_text(self, text: str) -> list[float]:
                rng = np.random.default_rng(abs(hash(text)) % (2**31))
                v = rng.standard_normal(384).tolist()
                n = sum(x * x for x in v) ** 0.5
                return [x / n for x in v]

            def embed_batch(self, texts: list[str], **kw: object) -> list[list[float]]:
                return [self.embed_text(t) for t in texts]

        graph = GraphBuilder(
            config=GraphConfig(include_semantic_edges=False),
            embedding_service=_MockEmb(),
        ).build(reg)
        sampler = ToolChainSampler(
            graph, SamplerConfig(max_iterations=200, max_retries=20)
        )
        rng = np.random.default_rng(42)
        chain = sampler.sample_chain(
            SamplingConstraints(min_steps=2, max_steps=3, min_tools=1), rng
        )

        orch = ConversationOrchestrator(
            user_sim=UserSimulator(),
            assistant=AssistantAgent(registry=reg),
            executor=ToolExecutor(reg),
            config=OrchestratorConfig(require_disambiguation=False),
        )
        conv = orch.generate_conversation(chain, seed=42)
        result = ConversationValidator().validate(conv)
        assert result.valid, f"Pipeline conv should be valid: {result.errors}"

    def test_with_registry(self, registry):
        conv = _conv([
            _user(),
            _assistant_tc("hotels/search", arguments={"city": "London"}),
            _tool(),
            _assistant(),
        ])
        r = ConversationValidator(registry=registry).validate(conv)
        assert r.valid

    def test_without_registry(self):
        conv = _conv([
            _user(),
            _assistant_tc("anything/endpoint"),
            _tool(),
            _assistant(),
        ])
        r = ConversationValidator(registry=None).validate(conv)
        assert r.valid


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    def test_single_user_message(self):
        conv = _conv([_user()])
        r = ConversationValidator().validate(conv)
        assert not r.valid
        assert any("no assistant" in e for e in r.errors)
        assert any("No tool calls" in e for e in r.errors)

    def test_long_conversation(self):
        msgs: list[Message] = [_user()]
        for _ in range(10):
            msgs.append(_assistant_tc())
            msgs.append(_tool())
            msgs.append(_user("more"))
        msgs.append(_assistant())
        conv = _conv(msgs)
        r = ConversationValidator().validate(conv)
        assert r.valid

    def test_multiple_tool_calls_one_message(self):
        tc1 = ToolCallRequest(
            endpoint_id="hotels/search",
            tool_id="hotels",
            tool_name="H",
            endpoint_name="S",
            arguments={"city": "Paris"},
        )
        tc2 = ToolCallRequest(
            endpoint_id="hotels/book",
            tool_id="hotels",
            tool_name="H",
            endpoint_name="B",
            arguments={"hotel_id": "h1", "guest_name": "Alice"},
        )
        msg = Message(role="assistant", tool_calls=[tc1, tc2])
        conv = _conv([_user(), msg, _tool(), _tool(), _assistant()])
        r = ConversationValidator().validate(conv)
        assert r.valid
