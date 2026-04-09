"""Integration tests for Task 34 — full execution pipeline.

Exercises: create context → generate arguments → execute tool →
verify grounding propagation → advance step → repeat.
"""

from __future__ import annotations

import numpy as np
import pytest

from tooluse_gen.agents import (
    ArgumentGenerator,
    ConversationContext,
    GroundingTracker,
    SchemaBasedGenerator,
    ToolCallRequest,
    ToolCallResponse,
    ToolExecutor,
    ValuePool,
    ValueProvenance,
    format_available_values,
    format_grounding_context,
    format_value_for_prompt,
)
from tooluse_gen.graph.chain_models import ChainPattern, ChainStep, ToolChain
from tooluse_gen.registry.models import (
    Endpoint,
    HttpMethod,
    Parameter,
    ParameterType,
    Tool,
)
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
            tool_id="hotel",
            name="Hotel API",
            domain="Travel",
            completeness_score=0.8,
            endpoints=[
                Endpoint(
                    endpoint_id="hotel/GET/search",
                    tool_id="hotel",
                    name="Search Hotels",
                    method=HttpMethod.GET,
                    path="/hotels",
                    parameters=[
                        Parameter(name="city", param_type=ParameterType.STRING, required=True),
                        Parameter(name="check_in", param_type=ParameterType.DATE, required=True),
                        Parameter(
                            name="max_price", param_type=ParameterType.NUMBER, required=False
                        ),
                    ],
                    required_parameters=["city", "check_in"],
                ),
                Endpoint(
                    endpoint_id="hotel/POST/book",
                    tool_id="hotel",
                    name="Book Hotel",
                    method=HttpMethod.POST,
                    path="/bookings",
                    parameters=[
                        Parameter(
                            name="hotel_id", param_type=ParameterType.STRING, required=True
                        ),
                        Parameter(
                            name="guest_name", param_type=ParameterType.STRING, required=True
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
            completeness_score=0.7,
            endpoints=[
                Endpoint(
                    endpoint_id="weather/GET/current",
                    tool_id="weather",
                    name="Current Weather",
                    method=HttpMethod.GET,
                    path="/weather/current",
                    parameters=[
                        Parameter(name="city", param_type=ParameterType.STRING, required=True),
                    ],
                    required_parameters=["city"],
                ),
                Endpoint(
                    endpoint_id="weather/DELETE/alert",
                    tool_id="weather",
                    name="Delete Alert",
                    method=HttpMethod.DELETE,
                    path="/alerts/{id}",
                    parameters=[
                        Parameter(
                            name="alert_id", param_type=ParameterType.STRING, required=True
                        ),
                    ],
                    required_parameters=["alert_id"],
                ),
            ],
        )
    )
    return reg


@pytest.fixture()
def chain() -> ToolChain:
    steps: list[ChainStep] = [
        ChainStep(
            endpoint_id="hotel/GET/search",
            tool_id="hotel",
            tool_name="Hotel API",
            endpoint_name="Search Hotels",
            method=HttpMethod.GET,
            path="/hotels",
            expected_params=["city", "check_in"],
            domain="Travel",
        ),
        ChainStep(
            endpoint_id="hotel/POST/book",
            tool_id="hotel",
            tool_name="Hotel API",
            endpoint_name="Book Hotel",
            method=HttpMethod.POST,
            path="/bookings",
            expected_params=["hotel_id", "guest_name"],
            domain="Travel",
        ),
        ChainStep(
            endpoint_id="weather/GET/current",
            tool_id="weather",
            tool_name="Weather API",
            endpoint_name="Current Weather",
            method=HttpMethod.GET,
            path="/weather/current",
            expected_params=["city"],
            domain="Weather",
        ),
    ]
    return ToolChain(chain_id="integration_test", steps=steps, pattern=ChainPattern.SEQUENTIAL)


def _run_pipeline(
    registry: ToolRegistry,
    chain: ToolChain,
    rng: np.random.Generator,
) -> tuple[ConversationContext, GroundingTracker]:
    """Execute the full 3-step pipeline, returning context and tracker."""
    ctx = ConversationContext(chain=chain)
    executor = ToolExecutor(registry)
    arg_gen = ArgumentGenerator()
    tracker = GroundingTracker()

    for i, step in enumerate(chain.steps):
        if not isinstance(step, ChainStep):
            continue
        ep = registry.get_endpoint(step.endpoint_id)
        assert ep is not None
        args = arg_gen.generate_arguments(ep, ctx, rng)
        req = ToolCallRequest.from_chain_step(step, args)
        resp = executor.execute(req, ctx, rng)
        ctx.add_tool_output(resp)
        tracker.track_from_response(resp, step.endpoint_id, i)
        if i < len(chain.steps) - 1:
            ctx.advance_step()

    return ctx, tracker


# ===========================================================================
# 1. Import completeness
# ===========================================================================


class TestImports:
    def test_all_symbols_importable(self) -> None:
        assert ArgumentGenerator is not None
        assert ConversationContext is not None
        assert GroundingTracker is not None
        assert SchemaBasedGenerator is not None
        assert ToolCallRequest is not None
        assert ToolCallResponse is not None
        assert ToolExecutor is not None
        assert ValuePool is not None
        assert ValueProvenance is not None
        assert callable(format_available_values)
        assert callable(format_grounding_context)
        assert callable(format_value_for_prompt)

    def test_correct_types(self) -> None:
        assert isinstance(ArgumentGenerator, type)
        assert isinstance(ConversationContext, type)
        assert isinstance(ToolExecutor, type)
        assert isinstance(GroundingTracker, type)


# ===========================================================================
# 2. Full pipeline
# ===========================================================================


class TestFullPipeline:
    def test_three_step_pipeline(
        self, registry: ToolRegistry, chain: ToolChain
    ) -> None:
        ctx, tracker = _run_pipeline(registry, chain, np.random.default_rng(42))

        assert ctx.current_step == 2
        assert len(ctx.tool_outputs) == 3
        assert len(ctx.grounding_values) > 0
        available = ctx.get_available_values()
        assert len(available) > 5

    def test_all_responses_success(
        self, registry: ToolRegistry, chain: ToolChain
    ) -> None:
        ctx, _ = _run_pipeline(registry, chain, np.random.default_rng(42))
        for resp in ctx.tool_outputs:
            assert resp.is_success

    def test_chain_attached(
        self, registry: ToolRegistry, chain: ToolChain
    ) -> None:
        ctx, _ = _run_pipeline(registry, chain, np.random.default_rng(42))
        assert ctx.chain is not None
        assert ctx.chain.chain_id == "integration_test"


# ===========================================================================
# 3. Grounding propagation
# ===========================================================================


class TestGroundingPropagation:
    def test_step0_values_available_in_step1(
        self, registry: ToolRegistry, chain: ToolChain
    ) -> None:
        ctx = ConversationContext(chain=chain)
        executor = ToolExecutor(registry)
        arg_gen = ArgumentGenerator()
        rng = np.random.default_rng(42)

        # Step 0
        ep0 = registry.get_endpoint("hotel/GET/search")
        assert ep0 is not None
        args0 = arg_gen.generate_arguments(ep0, ctx, rng)
        req0 = ToolCallRequest.from_chain_step(chain.steps[0], args0)
        resp0 = executor.execute(req0, ctx, rng)
        ctx.add_tool_output(resp0)
        ctx.advance_step()

        # Step 1 should see values from step 0
        available = ctx.get_available_values()
        assert len(available) > 0

    def test_generated_ids_reused(
        self, registry: ToolRegistry, chain: ToolChain
    ) -> None:
        ctx, _ = _run_pipeline(registry, chain, np.random.default_rng(42))
        assert len(ctx.generated_ids) > 0

    def test_values_accumulate(
        self, registry: ToolRegistry, chain: ToolChain
    ) -> None:
        ctx, _ = _run_pipeline(registry, chain, np.random.default_rng(42))
        # Should have values from all 3 steps
        available = ctx.get_available_values()
        assert len(available) >= 6

    def test_step_prefixed_keys_present(
        self, registry: ToolRegistry, chain: ToolChain
    ) -> None:
        ctx, _ = _run_pipeline(registry, chain, np.random.default_rng(42))
        step_keys = [k for k in ctx.grounding_values if k.startswith("step_")]
        assert len(step_keys) >= 3  # at least one per step

    def test_get_available_values_comprehensive(
        self, registry: ToolRegistry, chain: ToolChain
    ) -> None:
        ctx, _ = _run_pipeline(registry, chain, np.random.default_rng(42))
        vals = ctx.get_available_values()
        # Should include both grounding_values and generated_ids
        for gid_key in ctx.generated_ids:
            assert gid_key in vals


# ===========================================================================
# 4. Formatting integration
# ===========================================================================


class TestFormattingIntegration:
    def test_format_with_provenance(
        self, registry: ToolRegistry, chain: ToolChain
    ) -> None:
        ctx, tracker = _run_pipeline(registry, chain, np.random.default_rng(42))
        text = format_available_values(ctx, tracker)
        assert "Available values" in text
        # Should contain at least one endpoint reference
        has_endpoint = any(
            ep in text
            for ep in ["hotel/GET/search", "hotel/POST/book", "weather/GET/current"]
        )
        assert has_endpoint

    def test_format_grounding_context_count(
        self, registry: ToolRegistry, chain: ToolChain
    ) -> None:
        ctx, _ = _run_pipeline(registry, chain, np.random.default_rng(42))
        struct = format_grounding_context(ctx)
        assert struct["prior_tool_calls"] == 3
        assert struct["current_step"] == 2

    def test_format_value_on_real_data(
        self, registry: ToolRegistry, chain: ToolChain
    ) -> None:
        ctx, _ = _run_pipeline(registry, chain, np.random.default_rng(42))
        for val in ctx.grounding_values.values():
            result = format_value_for_prompt(val)
            assert isinstance(result, str) and len(result) > 0

    def test_prompt_no_step_prefix(
        self, registry: ToolRegistry, chain: ToolChain
    ) -> None:
        ctx, tracker = _run_pipeline(registry, chain, np.random.default_rng(42))
        text = format_available_values(ctx, tracker)
        for line in text.split("\n"):
            if line.startswith("- "):
                key = line.split(":")[0].strip("- ")
                assert "." not in key


# ===========================================================================
# 5. Message history
# ===========================================================================


class TestMessageHistory:
    def test_user_and_assistant_messages(
        self, registry: ToolRegistry, chain: ToolChain
    ) -> None:
        ctx, _ = _run_pipeline(registry, chain, np.random.default_rng(42))
        ctx.add_message("user", "Plan my trip")
        ctx.add_message("assistant", "I'll help you.")
        user_msgs = [m for m in ctx.messages if m["role"] == "user"]
        asst_msgs = [m for m in ctx.messages if m["role"] == "assistant"]
        assert len(user_msgs) >= 1
        assert len(asst_msgs) >= 1

    def test_tool_messages_present(
        self, registry: ToolRegistry, chain: ToolChain
    ) -> None:
        ctx, _ = _run_pipeline(registry, chain, np.random.default_rng(42))
        tool_msgs = [m for m in ctx.messages if m["role"] == "tool"]
        assert len(tool_msgs) == 3  # one per step

    def test_history_order(
        self, registry: ToolRegistry, chain: ToolChain
    ) -> None:
        ctx, _ = _run_pipeline(registry, chain, np.random.default_rng(42))
        history = ctx.get_history_for_prompt()
        assert len(history) == 3  # 3 tool messages
        assert all(m["role"] == "tool" for m in history)


# ===========================================================================
# 6. Context state
# ===========================================================================


class TestContextState:
    def test_step_increments(
        self, registry: ToolRegistry, chain: ToolChain
    ) -> None:
        ctx, _ = _run_pipeline(registry, chain, np.random.default_rng(42))
        assert ctx.current_step == 2

    def test_chain_attached(
        self, registry: ToolRegistry, chain: ToolChain
    ) -> None:
        ctx, _ = _run_pipeline(registry, chain, np.random.default_rng(42))
        assert ctx.chain is not None
        assert ctx.chain.total_step_count == 3

    def test_conversation_id_set(
        self, registry: ToolRegistry, chain: ToolChain
    ) -> None:
        ctx, _ = _run_pipeline(registry, chain, np.random.default_rng(42))
        assert len(ctx.conversation_id) > 0


# ===========================================================================
# 7. Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_endpoint_not_found(self, registry: ToolRegistry) -> None:
        executor = ToolExecutor(registry)
        req = ToolCallRequest(
            endpoint_id="nonexistent",
            tool_id="x",
            tool_name="X",
            endpoint_name="Y",
        )
        resp = executor.execute(req, ConversationContext(), np.random.default_rng(42))
        assert resp.status_code == 404
        assert not resp.is_success

    def test_empty_arguments(self, registry: ToolRegistry) -> None:
        executor = ToolExecutor(registry)
        req = ToolCallRequest(
            endpoint_id="hotel/GET/search",
            tool_id="hotel",
            tool_name="Hotel API",
            endpoint_name="Search Hotels",
            arguments={},
        )
        resp = executor.execute(req, ConversationContext(), np.random.default_rng(42))
        assert resp.is_success

    def test_deterministic(
        self, registry: ToolRegistry, chain: ToolChain
    ) -> None:
        ctx_a, _ = _run_pipeline(registry, chain, np.random.default_rng(99))
        ctx_b, _ = _run_pipeline(registry, chain, np.random.default_rng(99))
        for ra, rb in zip(ctx_a.tool_outputs, ctx_b.tool_outputs, strict=True):
            assert ra.data == rb.data


# ===========================================================================
# 8. ToolCallRequest integration
# ===========================================================================


class TestToolCallRequestIntegration:
    def test_from_chain_step(self, chain: ToolChain) -> None:
        step = chain.steps[0]
        assert isinstance(step, ChainStep)
        req = ToolCallRequest.from_chain_step(step, {"city": "Paris"})
        assert req.endpoint_id == "hotel/GET/search"
        assert req.tool_name == "Hotel API"
        assert req.arguments["city"] == "Paris"

    def test_request_matches_generated_args(
        self, registry: ToolRegistry, chain: ToolChain
    ) -> None:
        arg_gen = ArgumentGenerator()
        ep = registry.get_endpoint("hotel/GET/search")
        assert ep is not None
        args = arg_gen.generate_arguments(ep, ConversationContext(), np.random.default_rng(42))
        step = chain.steps[0]
        assert isinstance(step, ChainStep)
        req = ToolCallRequest.from_chain_step(step, args)
        assert req.arguments == args
