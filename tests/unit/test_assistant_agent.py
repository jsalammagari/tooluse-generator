"""Tests for the AssistantAgent (Task 37)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from tooluse_gen.agents.assistant_agent import (
    AssistantAgent,
    AssistantResponse,
    _flatten_steps,
    _ptype_to_json,
)
from tooluse_gen.agents.conversation_models import GenerationConfig
from tooluse_gen.agents.execution_models import (
    ConversationContext,
    ToolCallRequest,
    ToolCallResponse,
)
from tooluse_gen.graph.chain_models import ChainPattern, ChainStep, ParallelGroup, ToolChain
from tooluse_gen.registry.models import Endpoint, HttpMethod, Parameter, ParameterType, Tool
from tooluse_gen.registry.registry import ToolRegistry

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------


def _make_mock_client(
    content: str = "I'll help you with that.",
    tool_calls: list[MagicMock] | None = None,
) -> MagicMock:
    mock = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_choice.message.tool_calls = tool_calls
    mock_response.choices = [mock_choice]
    mock.chat.completions.create.return_value = mock_response
    return mock


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
                            description="City name",
                        ),
                        Parameter(
                            name="max_price",
                            param_type=ParameterType.NUMBER,
                            required=False,
                            description="Max price",
                        ),
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
                        Parameter(
                            name="hotel_id",
                            param_type=ParameterType.STRING,
                            required=True,
                            description="Hotel ID",
                        ),
                        Parameter(
                            name="guest_name",
                            param_type=ParameterType.STRING,
                            required=True,
                            description="Guest name",
                        ),
                    ],
                    required_parameters=["hotel_id", "guest_name"],
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
def context(chain: ToolChain) -> ConversationContext:
    ctx = ConversationContext(chain=chain)
    ctx.add_message("user", "I want to find a hotel in Paris")
    return ctx


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ===================================================================
# AssistantResponse model
# ===================================================================


class TestAssistantResponse:
    def test_default(self):
        r = AssistantResponse()
        assert r.content is None
        assert r.tool_calls is None
        assert r.is_disambiguation is False
        assert r.is_final_answer is False

    def test_tool_call_response(self):
        tc = ToolCallRequest(
            endpoint_id="e", tool_id="t", tool_name="T", endpoint_name="E"
        )
        r = AssistantResponse(tool_calls=[tc])
        assert r.tool_calls is not None
        assert len(r.tool_calls) == 1
        assert r.content is None

    def test_disambiguation_response(self):
        r = AssistantResponse(content="What city?", is_disambiguation=True)
        assert r.is_disambiguation is True
        assert r.content == "What city?"

    def test_final_answer_response(self):
        r = AssistantResponse(content="All done!", is_final_answer=True)
        assert r.is_final_answer is True


# ===================================================================
# Construction
# ===================================================================


class TestConstruction:
    def test_offline_mode(self, registry):
        agent = AssistantAgent(registry=registry)
        assert agent._client is None

    def test_with_mock_client(self, registry):
        mock = _make_mock_client()
        agent = AssistantAgent(registry=registry, llm_client=mock)
        assert agent._client is mock

    def test_custom_model_temp(self, registry):
        agent = AssistantAgent(
            registry=registry, model="gpt-4o-mini", temperature=1.0
        )
        assert agent._model == "gpt-4o-mini"
        assert agent._temperature == 1.0


# ===================================================================
# generate_response — tool call
# ===================================================================


class TestToolCall:
    def test_returns_tool_call(self, registry, context, rng):
        agent = AssistantAgent(registry=registry)
        config = GenerationConfig(include_disambiguation=False)
        resp = agent.generate_response(context, rng, config)
        assert resp.tool_calls is not None
        assert len(resp.tool_calls) == 1

    def test_targets_correct_endpoint(self, registry, context, rng):
        agent = AssistantAgent(registry=registry)
        config = GenerationConfig(include_disambiguation=False)
        resp = agent.generate_response(context, rng, config)
        assert resp.tool_calls is not None
        assert resp.tool_calls[0].endpoint_id == "hotels/search"

    def test_required_params_filled(self, registry, context, rng):
        agent = AssistantAgent(registry=registry)
        config = GenerationConfig(include_disambiguation=False)
        resp = agent.generate_response(context, rng, config)
        assert resp.tool_calls is not None
        assert "city" in resp.tool_calls[0].arguments

    def test_grounding_injected(self, registry, chain, rng):
        agent = AssistantAgent(registry=registry)
        config = GenerationConfig(include_disambiguation=False)
        ctx = ConversationContext(chain=chain)
        ctx.add_message("user", "hi")

        # Simulate step 0 output with grounding values.
        resp0 = agent.generate_response(ctx, rng, config)
        assert resp0.tool_calls is not None
        tool_resp = ToolCallResponse(
            call_id=resp0.tool_calls[0].call_id,
            status_code=200,
            data={"results": [{"id": "htl_1"}]},
            extractable_values={"hotel_id": "htl_1", "name": "Grand"},
            generated_ids={"hotel_id": "htl_1"},
        )
        ctx.add_tool_output(tool_resp)
        ctx.advance_step()

        # Step 1 should pick up hotel_id from grounding.
        resp1 = agent.generate_response(ctx, rng, config)
        assert resp1.tool_calls is not None
        assert resp1.tool_calls[0].endpoint_id == "hotels/book"
        assert resp1.tool_calls[0].arguments.get("hotel_id") == "htl_1"

    def test_second_step_endpoint(self, registry, chain, rng):
        agent = AssistantAgent(registry=registry)
        config = GenerationConfig(include_disambiguation=False)
        ctx = ConversationContext(chain=chain)
        ctx.add_message("user", "hi")

        resp0 = agent.generate_response(ctx, rng, config)
        assert resp0.tool_calls is not None
        tool_resp = ToolCallResponse(
            call_id=resp0.tool_calls[0].call_id,
            status_code=200,
            data={},
        )
        ctx.add_tool_output(tool_resp)
        ctx.advance_step()

        resp1 = agent.generate_response(ctx, rng, config)
        assert resp1.tool_calls is not None
        assert resp1.tool_calls[0].endpoint_id == "hotels/book"


# ===================================================================
# generate_response — disambiguation
# ===================================================================


class TestDisambiguation:
    def test_disambiguation_triggers(self, registry, context, rng):
        agent = AssistantAgent(registry=registry)
        config = GenerationConfig(
            include_disambiguation=True, disambiguation_probability=1.0
        )
        resp = agent.generate_response(context, rng, config)
        assert resp.is_disambiguation is True

    def test_disambiguation_is_question(self, registry, context, rng):
        agent = AssistantAgent(registry=registry)
        config = GenerationConfig(
            include_disambiguation=True, disambiguation_probability=1.0
        )
        resp = agent.generate_response(context, rng, config)
        assert resp.content is not None
        assert "?" in resp.content

    def test_no_disambiguation_when_disabled(self, registry, context, rng):
        agent = AssistantAgent(registry=registry)
        config = GenerationConfig(include_disambiguation=False)
        resp = agent.generate_response(context, rng, config)
        assert resp.is_disambiguation is False

    def test_no_disambiguation_after_first_step(self, registry, chain, rng):
        agent = AssistantAgent(registry=registry)
        config = GenerationConfig(
            include_disambiguation=True, disambiguation_probability=1.0
        )
        ctx = ConversationContext(chain=chain)
        ctx.add_message("user", "hi")
        ctx.advance_step()  # Move past step 0.
        resp = agent.generate_response(ctx, rng, config)
        assert resp.is_disambiguation is False


# ===================================================================
# generate_response — final answer
# ===================================================================


class TestFinalAnswer:
    def test_final_when_all_steps_done(self, registry, chain, rng):
        agent = AssistantAgent(registry=registry)
        config = GenerationConfig(include_disambiguation=False)
        ctx = ConversationContext(chain=chain)
        ctx.add_message("user", "hi")

        # Advance past all steps.
        flat = _flatten_steps(chain)
        for _ in flat:
            ctx.advance_step()

        resp = agent.generate_response(ctx, rng, config)
        assert resp.is_final_answer is True

    def test_final_has_flag(self, registry, chain, rng):
        agent = AssistantAgent(registry=registry)
        ctx = ConversationContext(chain=chain)
        for _ in _flatten_steps(chain):
            ctx.advance_step()
        resp = agent.generate_response(ctx, rng)
        assert resp.is_final_answer is True
        assert resp.content is not None

    def test_final_content_nonempty(self, registry, chain, rng):
        agent = AssistantAgent(registry=registry)
        ctx = ConversationContext(chain=chain)
        for _ in _flatten_steps(chain):
            ctx.advance_step()
        resp = agent.generate_response(ctx, rng)
        assert resp.content is not None
        assert len(resp.content) > 5

    def test_final_when_no_chain(self, registry, rng):
        agent = AssistantAgent(registry=registry)
        ctx = ConversationContext()
        ctx.add_message("user", "hi")
        resp = agent.generate_response(ctx, rng)
        assert resp.is_final_answer is True


# ===================================================================
# _build_tools_schema
# ===================================================================


class TestBuildToolsSchema:
    def test_returns_list(self, registry, chain):
        agent = AssistantAgent(registry=registry)
        schema = agent._build_tools_schema(chain)
        assert isinstance(schema, list)
        assert len(schema) == 2  # search + book

    def test_function_type(self, registry, chain):
        agent = AssistantAgent(registry=registry)
        schema = agent._build_tools_schema(chain)
        for s in schema:
            assert s["type"] == "function"

    def test_required_params(self, registry, chain):
        agent = AssistantAgent(registry=registry)
        schema = agent._build_tools_schema(chain)
        search = next(
            s for s in schema if s["function"]["name"] == "hotels/search"
        )
        assert search["function"]["parameters"]["required"] == ["city"]

    def test_properties_mapped(self, registry, chain):
        agent = AssistantAgent(registry=registry)
        schema = agent._build_tools_schema(chain)
        search = next(
            s for s in schema if s["function"]["name"] == "hotels/search"
        )
        props = search["function"]["parameters"]["properties"]
        assert "city" in props
        assert props["city"]["type"] == "string"
        assert "max_price" in props
        assert props["max_price"]["type"] == "number"


# ===================================================================
# _build_system_prompt
# ===================================================================


class TestBuildSystemPrompt:
    def test_includes_tool_desc(self, registry, context):
        agent = AssistantAgent(registry=registry)
        prompt = agent._build_system_prompt(context)
        assert "Search Hotels" in prompt
        assert "Book Hotel" in prompt

    def test_includes_grounding(self, registry, chain):
        agent = AssistantAgent(registry=registry)
        ctx = ConversationContext(chain=chain)
        ctx.grounding_values["city"] = "Paris"
        prompt = agent._build_system_prompt(ctx)
        assert "Paris" in prompt or "city" in prompt


# ===================================================================
# _should_disambiguate
# ===================================================================


class TestShouldDisambiguate:
    def test_true_on_first_step(self, registry, context):
        agent = AssistantAgent(registry=registry)
        config = GenerationConfig(disambiguation_probability=1.0)
        assert agent._should_disambiguate(
            context, np.random.default_rng(42), config
        )

    def test_false_after_step_0(self, registry, chain):
        agent = AssistantAgent(registry=registry)
        config = GenerationConfig(disambiguation_probability=1.0)
        ctx = ConversationContext(chain=chain)
        ctx.add_message("user", "hi")
        ctx.advance_step()
        assert not agent._should_disambiguate(ctx, np.random.default_rng(42), config)

    def test_false_when_outputs_exist(self, registry, context):
        agent = AssistantAgent(registry=registry)
        config = GenerationConfig(disambiguation_probability=1.0)
        context.tool_outputs.append(
            ToolCallResponse(call_id="x", data={})
        )
        assert not agent._should_disambiguate(
            context, np.random.default_rng(42), config
        )

    def test_false_when_already_asked(self, registry, chain):
        agent = AssistantAgent(registry=registry)
        config = GenerationConfig(disambiguation_probability=1.0)
        ctx = ConversationContext(chain=chain)
        ctx.add_message("user", "hi")
        ctx.add_message("assistant", "What city?")
        assert not agent._should_disambiguate(ctx, np.random.default_rng(42), config)

    def test_deterministic(self, registry, context):
        agent = AssistantAgent(registry=registry)
        config = GenerationConfig(disambiguation_probability=0.5)
        r1 = agent._should_disambiguate(
            context, np.random.default_rng(42), config
        )
        r2 = agent._should_disambiguate(
            context, np.random.default_rng(42), config
        )
        assert r1 == r2


# ===================================================================
# _resolve_argument
# ===================================================================


class TestResolveArgument:
    def test_exact_match(self, registry, chain):
        agent = AssistantAgent(registry=registry)
        ctx = ConversationContext(chain=chain)
        ctx.grounding_values["city"] = "Tokyo"
        ep = registry.get_endpoint("hotels/search")
        assert ep is not None
        param = ep.parameters[0]  # city
        val = agent._resolve_argument(param, ctx, np.random.default_rng(1))
        assert val == "Tokyo"

    def test_substring_match(self, registry, chain):
        agent = AssistantAgent(registry=registry)
        ctx = ConversationContext(chain=chain)
        ctx.grounding_values["hotel_id"] = "htl_99"
        ep = registry.get_endpoint("hotels/book")
        assert ep is not None
        param = next(p for p in ep.parameters if p.name == "hotel_id")
        val = agent._resolve_argument(param, ctx, np.random.default_rng(1))
        assert val == "htl_99"

    def test_enum_value(self, registry, chain):
        agent = AssistantAgent(registry=registry)
        ctx = ConversationContext(chain=chain)
        param = Parameter(
            name="color", param_type=ParameterType.STRING, enum_values=["red", "blue"]
        )
        val = agent._resolve_argument(param, ctx, np.random.default_rng(42))
        assert val in ("red", "blue")

    def test_default_value(self, registry, chain):
        agent = AssistantAgent(registry=registry)
        ctx = ConversationContext(chain=chain)
        param = Parameter(
            name="limit", param_type=ParameterType.INTEGER, default=10
        )
        val = agent._resolve_argument(param, ctx, np.random.default_rng(1))
        assert val == 10

    def test_placeholder_string(self, registry, chain):
        agent = AssistantAgent(registry=registry)
        ctx = ConversationContext(chain=chain)
        param = Parameter(name="nickname", param_type=ParameterType.STRING)
        val = agent._resolve_argument(param, ctx, np.random.default_rng(1))
        assert val == "nickname_value"

    def test_placeholder_integer(self, registry, chain):
        agent = AssistantAgent(registry=registry)
        ctx = ConversationContext(chain=chain)
        param = Parameter(name="count", param_type=ParameterType.INTEGER)
        val = agent._resolve_argument(param, ctx, np.random.default_rng(1))
        assert isinstance(val, int)

    def test_placeholder_number(self, registry, chain):
        agent = AssistantAgent(registry=registry)
        ctx = ConversationContext(chain=chain)
        param = Parameter(name="amount", param_type=ParameterType.NUMBER)
        val = agent._resolve_argument(param, ctx, np.random.default_rng(1))
        assert isinstance(val, float)

    def test_placeholder_boolean(self, registry, chain):
        agent = AssistantAgent(registry=registry)
        ctx = ConversationContext(chain=chain)
        param = Parameter(name="active", param_type=ParameterType.BOOLEAN)
        val = agent._resolve_argument(param, ctx, np.random.default_rng(1))
        assert isinstance(val, bool)


# ===================================================================
# _ptype_to_json
# ===================================================================


class TestPtypeToJson:
    def test_known_types(self):
        assert _ptype_to_json("string") == "string"
        assert _ptype_to_json("integer") == "integer"
        assert _ptype_to_json("number") == "number"
        assert _ptype_to_json("boolean") == "boolean"
        assert _ptype_to_json("array") == "array"
        assert _ptype_to_json("object") == "object"

    def test_unknown_defaults_string(self):
        assert _ptype_to_json("foobar") == "string"
        assert _ptype_to_json("") == "string"


# ===================================================================
# _flatten_steps
# ===================================================================


class TestFlattenSteps:
    def test_sequential(self, chain):
        flat = _flatten_steps(chain)
        assert len(flat) == 2
        assert all(isinstance(s, ChainStep) for s in flat)

    def test_parallel_group(self):
        c = ToolChain(
            steps=[
                ParallelGroup(
                    steps=[
                        ChainStep(
                            endpoint_id="a",
                            tool_id="a",
                            tool_name="A",
                            endpoint_name="A",
                        ),
                        ChainStep(
                            endpoint_id="b",
                            tool_id="b",
                            tool_name="B",
                            endpoint_name="B",
                        ),
                    ]
                ),
                ChainStep(
                    endpoint_id="c",
                    tool_id="c",
                    tool_name="C",
                    endpoint_name="C",
                ),
            ],
            pattern=ChainPattern.PARALLEL,
        )
        flat = _flatten_steps(c)
        assert len(flat) == 3
        assert flat[0].endpoint_id == "a"
        assert flat[2].endpoint_id == "c"


# ===================================================================
# Determinism
# ===================================================================


class TestDeterminism:
    def test_same_seed_same_args(self, registry, chain):
        agent = AssistantAgent(registry=registry)
        config = GenerationConfig(include_disambiguation=False)

        ctx1 = ConversationContext(chain=chain)
        ctx1.add_message("user", "hi")
        r1 = agent.generate_response(ctx1, np.random.default_rng(42), config)

        ctx2 = ConversationContext(chain=chain)
        ctx2.add_message("user", "hi")
        r2 = agent.generate_response(ctx2, np.random.default_rng(42), config)

        assert r1.tool_calls is not None
        assert r2.tool_calls is not None
        assert r1.tool_calls[0].arguments == r2.tool_calls[0].arguments
