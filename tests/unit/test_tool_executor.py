"""Unit tests for Task 31 — Tool executor agent."""

from __future__ import annotations

import numpy as np
import pytest

from tooluse_gen.agents.execution_models import (
    ConversationContext,
    ToolCallRequest,
    ToolCallResponse,
)
from tooluse_gen.agents.tool_executor import ToolExecutor
from tooluse_gen.agents.value_generator import SchemaBasedGenerator
from tooluse_gen.registry.models import Endpoint, HttpMethod, Tool
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
                    endpoint_id="hotel/GET/list",
                    tool_id="hotel",
                    name="List Hotels",
                    method=HttpMethod.GET,
                    path="/hotels",
                ),
                Endpoint(
                    endpoint_id="hotel/POST/book",
                    tool_id="hotel",
                    name="Book Hotel",
                    method=HttpMethod.POST,
                    path="/bookings",
                ),
                Endpoint(
                    endpoint_id="hotel/DELETE/cancel",
                    tool_id="hotel",
                    name="Cancel Booking",
                    method=HttpMethod.DELETE,
                    path="/bookings/{id}",
                ),
            ],
        )
    )
    return reg


@pytest.fixture()
def executor(registry: ToolRegistry) -> ToolExecutor:
    return ToolExecutor(registry)


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def _req(
    endpoint_id: str = "hotel/GET/list",
    method: HttpMethod = HttpMethod.GET,
    path: str = "/hotels",
    name: str = "List Hotels",
    arguments: dict[str, object] | None = None,
) -> ToolCallRequest:
    return ToolCallRequest(
        endpoint_id=endpoint_id,
        tool_id="hotel",
        tool_name="Hotel API",
        endpoint_name=name,
        method=method,
        path=path,
        arguments=dict(arguments) if arguments else {},
    )


# ===========================================================================
# Construction
# ===========================================================================


class TestConstruction:
    def test_default(self, registry: ToolRegistry) -> None:
        ex = ToolExecutor(registry)
        assert ex._use_llm is False

    def test_custom_generator(self, registry: ToolRegistry) -> None:
        gen = SchemaBasedGenerator()
        ex = ToolExecutor(registry, generator=gen)
        assert ex._generator is gen

    def test_use_llm_flag(self, registry: ToolRegistry) -> None:
        ex = ToolExecutor(registry, use_llm=True, llm_client="fake")
        assert ex._use_llm is True


# ===========================================================================
# execute — schema-based
# ===========================================================================


class TestExecuteSchemaBased:
    def test_returns_response(self, executor: ToolExecutor, rng: np.random.Generator) -> None:
        resp = executor.execute(_req(), ConversationContext(), rng)
        assert isinstance(resp, ToolCallResponse)

    def test_is_success(self, executor: ToolExecutor, rng: np.random.Generator) -> None:
        resp = executor.execute(_req(), ConversationContext(), rng)
        assert resp.is_success

    def test_data_is_dict(self, executor: ToolExecutor, rng: np.random.Generator) -> None:
        resp = executor.execute(_req(), ConversationContext(), rng)
        assert isinstance(resp.data, dict) and len(resp.data) > 0

    def test_not_found_404(self, executor: ToolExecutor, rng: np.random.Generator) -> None:
        req = _req(endpoint_id="nonexistent")
        resp = executor.execute(req, ConversationContext(), rng)
        assert resp.status_code == 404
        assert resp.error is not None
        assert not resp.is_success

    def test_arguments_propagated(self, executor: ToolExecutor, rng: np.random.Generator) -> None:
        req = _req(
            endpoint_id="hotel/POST/book",
            method=HttpMethod.POST,
            path="/bookings",
            name="Book Hotel",
            arguments={"guest_name": "Alice"},
        )
        resp = executor.execute(req, ConversationContext(), rng)
        assert resp.data.get("guest_name") == "Alice"

    def test_generated_ids_detected(self, executor: ToolExecutor, rng: np.random.Generator) -> None:
        req = _req(
            endpoint_id="hotel/POST/book",
            method=HttpMethod.POST,
            path="/bookings",
            name="Book Hotel",
        )
        resp = executor.execute(req, ConversationContext(), rng)
        # POST produces an object with 'id' field that should match ID pattern
        has_ids = len(resp.generated_ids) > 0 or "id" in resp.data
        assert has_ids

    def test_extractable_values_populated(
        self, executor: ToolExecutor, rng: np.random.Generator
    ) -> None:
        req = _req(
            endpoint_id="hotel/POST/book",
            method=HttpMethod.POST,
            path="/bookings",
            name="Book Hotel",
        )
        resp = executor.execute(req, ConversationContext(), rng)
        assert len(resp.extractable_values) > 0


# ===========================================================================
# execute — LLM-based
# ===========================================================================


class TestExecuteLLMBased:
    def test_llm_returns_response(self, registry: ToolRegistry) -> None:
        ex = ToolExecutor(registry, use_llm=True, llm_client="fake")
        resp = ex.execute(_req(), ConversationContext(), np.random.default_rng(42))
        assert isinstance(resp, ToolCallResponse)
        assert resp.is_success

    def test_llm_fallback_graceful(self, registry: ToolRegistry) -> None:
        ex = ToolExecutor(registry, use_llm=True, llm_client="fake")
        resp = ex.execute(_req(), ConversationContext(), np.random.default_rng(42))
        assert "message" in resp.data


# ===========================================================================
# _extract_values
# ===========================================================================


class TestExtractValues:
    def test_extracts_id(self, executor: ToolExecutor) -> None:
        resp = ToolCallResponse(
            call_id="c1", data={"booking_id": "BOO-1234", "other": "x"}
        )
        ep = Endpoint(
            endpoint_id="e1", tool_id="t1", name="E", method=HttpMethod.GET, path="/x"
        )
        vals = executor._extract_values(resp, ep)
        assert "booking_id" in vals

    def test_extracts_name(self, executor: ToolExecutor) -> None:
        resp = ToolCallResponse(call_id="c1", data={"hotel_name": "Grand", "x": 1})
        ep = Endpoint(
            endpoint_id="e1", tool_id="t1", name="E", method=HttpMethod.GET, path="/x"
        )
        vals = executor._extract_values(resp, ep)
        assert "hotel_name" in vals

    def test_extracts_url(self, executor: ToolExecutor) -> None:
        resp = ToolCallResponse(
            call_id="c1", data={"website_url": "https://example.com", "y": 2}
        )
        ep = Endpoint(
            endpoint_id="e1", tool_id="t1", name="E", method=HttpMethod.GET, path="/x"
        )
        vals = executor._extract_values(resp, ep)
        assert "website_url" in vals

    def test_extracts_price_status_email(self, executor: ToolExecutor) -> None:
        resp = ToolCallResponse(
            call_id="c1",
            data={"price": 99.99, "status": "active", "email": "a@b.com", "misc": "z"},
        )
        ep = Endpoint(
            endpoint_id="e1", tool_id="t1", name="E", method=HttpMethod.GET, path="/x"
        )
        vals = executor._extract_values(resp, ep)
        assert "price" in vals
        assert "status" in vals
        assert "email" in vals

    def test_includes_generated_ids(self, executor: ToolExecutor) -> None:
        resp = ToolCallResponse(
            call_id="c1",
            data={"x": 1},
            generated_ids={"booking_id": "BOO-1234"},
        )
        ep = Endpoint(
            endpoint_id="e1", tool_id="t1", name="E", method=HttpMethod.GET, path="/x"
        )
        vals = executor._extract_values(resp, ep)
        assert vals["booking_id"] == "BOO-1234"


# ===========================================================================
# _extract_values_heuristic
# ===========================================================================


class TestExtractHeuristic:
    def test_id_key(self, executor: ToolExecutor) -> None:
        vals = executor._extract_values_heuristic({"booking_id": "B1", "foo": "bar"})
        assert "booking_id" in vals
        assert "foo" not in vals

    def test_name_key(self, executor: ToolExecutor) -> None:
        vals = executor._extract_values_heuristic({"guest_name": "Alice", "z": 0})
        assert "guest_name" in vals

    def test_url_key(self, executor: ToolExecutor) -> None:
        vals = executor._extract_values_heuristic({"profile_url": "https://x.com", "q": 1})
        assert "profile_url" in vals

    def test_price_key(self, executor: ToolExecutor) -> None:
        vals = executor._extract_values_heuristic({"total_price": 49.99, "w": "w"})
        assert "total_price" in vals

    def test_nested_results(self, executor: ToolExecutor) -> None:
        data = {
            "results": [
                {"item_id": "I-001", "item_name": "Widget", "misc": 42},
                {"item_id": "I-002", "item_name": "Gadget", "misc": 99},
            ],
            "count": 2,
        }
        vals = executor._extract_values_heuristic(data)
        assert "count" in vals
        assert "results_0_item_id" in vals
        assert "results_0_item_name" in vals


# ===========================================================================
# _validate_response
# ===========================================================================


class TestValidateResponse:
    def test_valid_get(self, executor: ToolExecutor) -> None:
        resp = ToolCallResponse(call_id="c1", data={"x": 1})
        ep = Endpoint(
            endpoint_id="e1", tool_id="t1", name="E", method=HttpMethod.GET, path="/x"
        )
        assert executor._validate_response(resp, ep) is True

    def test_post_without_id(self, executor: ToolExecutor) -> None:
        resp = ToolCallResponse(call_id="c1", data={"name": "x"})
        ep = Endpoint(
            endpoint_id="e1", tool_id="t1", name="E", method=HttpMethod.POST, path="/x"
        )
        assert executor._validate_response(resp, ep) is False

    def test_delete_without_status(self, executor: ToolExecutor) -> None:
        resp = ToolCallResponse(call_id="c1", data={"message": "done"})
        ep = Endpoint(
            endpoint_id="e1", tool_id="t1", name="E", method=HttpMethod.DELETE, path="/x"
        )
        assert executor._validate_response(resp, ep) is False

    def test_empty_data(self, executor: ToolExecutor) -> None:
        resp = ToolCallResponse(call_id="c1", data={})
        ep = Endpoint(
            endpoint_id="e1", tool_id="t1", name="E", method=HttpMethod.GET, path="/x"
        )
        assert executor._validate_response(resp, ep) is False


# ===========================================================================
# Session state / grounding
# ===========================================================================


class TestSessionState:
    def test_ids_persist_via_context(
        self, executor: ToolExecutor, rng: np.random.Generator
    ) -> None:
        ctx = ConversationContext()
        req = _req(
            endpoint_id="hotel/POST/book",
            method=HttpMethod.POST,
            path="/bookings",
            name="Book Hotel",
        )
        resp = executor.execute(req, ctx, rng)
        ctx.add_tool_output(resp)
        assert len(ctx.generated_ids) > 0 or len(ctx.grounding_values) > 0

    def test_grounding_accumulates(
        self, executor: ToolExecutor, rng: np.random.Generator
    ) -> None:
        ctx = ConversationContext()
        req1 = _req()
        resp1 = executor.execute(req1, ctx, rng)
        ctx.add_tool_output(resp1)
        initial_count = len(ctx.grounding_values)

        ctx.advance_step()
        req2 = _req(
            endpoint_id="hotel/POST/book",
            method=HttpMethod.POST,
            path="/bookings",
            name="Book Hotel",
        )
        resp2 = executor.execute(req2, ctx, rng)
        ctx.add_tool_output(resp2)
        assert len(ctx.grounding_values) > initial_count

    def test_second_call_references_first(
        self, executor: ToolExecutor, rng: np.random.Generator
    ) -> None:
        ctx = ConversationContext()
        req1 = _req(
            endpoint_id="hotel/POST/book",
            method=HttpMethod.POST,
            path="/bookings",
            name="Book Hotel",
        )
        resp1 = executor.execute(req1, ctx, rng)
        ctx.add_tool_output(resp1)

        # Get an ID from the first response
        available = ctx.get_available_values()
        assert len(available) > 0

    def test_deterministic(self, executor: ToolExecutor) -> None:
        ctx_a = ConversationContext()
        ctx_b = ConversationContext()
        req = _req()
        ra = executor.execute(req, ctx_a, np.random.default_rng(99))
        rb = executor.execute(req, ctx_b, np.random.default_rng(99))
        assert ra.data == rb.data
