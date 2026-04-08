"""Unit tests for Task 29 — Execution models."""

from __future__ import annotations

import json
import uuid
from datetime import datetime

import pytest

from tooluse_gen.agents.execution_models import (
    ConversationContext,
    ToolCallRequest,
    ToolCallResponse,
)
from tooluse_gen.graph.chain_models import ChainPattern, ChainStep, ToolChain
from tooluse_gen.registry.models import HttpMethod

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_step() -> ChainStep:
    return ChainStep(
        endpoint_id="weather/GET/cur",
        tool_id="weather",
        tool_name="Weather API",
        endpoint_name="Current Weather",
        method=HttpMethod.GET,
        path="/current",
        expected_params=["city"],
        domain="Weather",
    )


def _make_response(call_id: str = "test-call") -> ToolCallResponse:
    return ToolCallResponse(
        call_id=call_id,
        status_code=200,
        data={"temperature": 72.5},
        generated_ids={"session_id": "S-001"},
        extractable_values={"temperature": 72.5, "city": "NYC"},
    )


# ===========================================================================
# ToolCallRequest
# ===========================================================================


class TestToolCallRequest:
    def test_defaults(self) -> None:
        req = ToolCallRequest(
            endpoint_id="e1", tool_id="t1", tool_name="T1", endpoint_name="E1"
        )
        assert req.method == "GET"
        assert req.path == ""
        assert req.arguments == {}
        assert len(req.call_id) > 0

    def test_call_id_is_uuid(self) -> None:
        req = ToolCallRequest(
            endpoint_id="e1", tool_id="t1", tool_name="T1", endpoint_name="E1"
        )
        uuid.UUID(req.call_id)  # should not raise

    def test_custom_arguments(self) -> None:
        req = ToolCallRequest(
            endpoint_id="e1", tool_id="t1", tool_name="T1", endpoint_name="E1",
            arguments={"city": "NYC", "units": "metric"},
        )
        assert req.arguments == {"city": "NYC", "units": "metric"}

    def test_from_chain_step(self) -> None:
        step = _make_step()
        req = ToolCallRequest.from_chain_step(step)
        assert req.endpoint_id == "weather/GET/cur"
        assert req.tool_id == "weather"
        assert req.tool_name == "Weather API"
        assert req.endpoint_name == "Current Weather"
        assert req.method == "GET"
        assert req.path == "/current"
        assert req.arguments == {}

    def test_from_chain_step_with_args(self) -> None:
        step = _make_step()
        req = ToolCallRequest.from_chain_step(step, arguments={"city": "LA"})
        assert req.arguments == {"city": "LA"}

    def test_serialization_round_trip(self) -> None:
        req = ToolCallRequest(
            endpoint_id="e1", tool_id="t1", tool_name="T1", endpoint_name="E1",
            arguments={"k": "v"},
        )
        data = req.model_dump()
        restored = ToolCallRequest.model_validate(data)
        assert restored.endpoint_id == req.endpoint_id
        assert restored.arguments == req.arguments


# ===========================================================================
# ToolCallResponse
# ===========================================================================


class TestToolCallResponse:
    def test_construction(self) -> None:
        resp = ToolCallResponse(call_id="c1")
        assert resp.status_code == 200
        assert resp.data == {}
        assert resp.error is None

    def test_is_success_200(self) -> None:
        resp = ToolCallResponse(call_id="c1", status_code=200)
        assert resp.is_success is True

    def test_is_success_201(self) -> None:
        resp = ToolCallResponse(call_id="c1", status_code=201)
        assert resp.is_success is True

    def test_is_success_false_404(self) -> None:
        resp = ToolCallResponse(call_id="c1", status_code=404)
        assert resp.is_success is False

    def test_is_success_false_error(self) -> None:
        resp = ToolCallResponse(call_id="c1", status_code=200, error="boom")
        assert resp.is_success is False

    def test_generated_ids_and_extractable(self) -> None:
        resp = _make_response()
        assert resp.generated_ids == {"session_id": "S-001"}
        assert resp.extractable_values["temperature"] == 72.5


# ===========================================================================
# ConversationContext — construction
# ===========================================================================


class TestContextConstruction:
    def test_defaults(self) -> None:
        ctx = ConversationContext()
        assert len(ctx.conversation_id) > 0
        assert ctx.messages == []
        assert ctx.current_step == 0
        assert ctx.chain is None
        assert ctx.tool_outputs == []

    def test_with_chain(self) -> None:
        step = _make_step()
        chain = ToolChain(steps=[step], pattern=ChainPattern.SEQUENTIAL)
        ctx = ConversationContext(chain=chain)
        assert ctx.chain is not None
        assert ctx.chain.total_step_count == 1

    def test_created_at_parseable(self) -> None:
        ctx = ConversationContext()
        dt = datetime.fromisoformat(ctx.created_at)
        assert dt.year >= 2024

    def test_custom_id(self) -> None:
        ctx = ConversationContext(conversation_id="my-conv-001")
        assert ctx.conversation_id == "my-conv-001"


# ===========================================================================
# ConversationContext — add_message
# ===========================================================================


class TestContextAddMessage:
    def test_user_message(self) -> None:
        ctx = ConversationContext()
        ctx.add_message("user", "Hello")
        assert len(ctx.messages) == 1
        assert ctx.messages[0] == {"role": "user", "content": "Hello"}

    def test_assistant_message(self) -> None:
        ctx = ConversationContext()
        ctx.add_message("assistant", "Sure, let me help.")
        assert ctx.messages[0]["role"] == "assistant"

    def test_with_tool_calls(self) -> None:
        ctx = ConversationContext()
        ctx.add_message("assistant", "Calling tool.", tool_calls=[{"call_id": "c1"}])
        assert "tool_calls" in ctx.messages[0]
        assert ctx.messages[0]["tool_calls"] == [{"call_id": "c1"}]

    def test_messages_accumulate(self) -> None:
        ctx = ConversationContext()
        ctx.add_message("user", "Q1")
        ctx.add_message("assistant", "A1")
        ctx.add_message("user", "Q2")
        assert len(ctx.messages) == 3
        assert ctx.messages[0]["role"] == "user"
        assert ctx.messages[1]["role"] == "assistant"
        assert ctx.messages[2]["role"] == "user"


# ===========================================================================
# ConversationContext — add_tool_output
# ===========================================================================


class TestContextAddToolOutput:
    def test_appends_to_outputs(self) -> None:
        ctx = ConversationContext()
        resp = _make_response()
        ctx.add_tool_output(resp)
        assert len(ctx.tool_outputs) == 1
        assert ctx.tool_outputs[0].call_id == "test-call"

    def test_merges_generated_ids(self) -> None:
        ctx = ConversationContext()
        ctx.add_tool_output(_make_response())
        assert ctx.generated_ids["session_id"] == "S-001"

    def test_merges_values_with_step_prefix(self) -> None:
        ctx = ConversationContext()
        ctx.add_tool_output(_make_response())
        assert ctx.grounding_values["step_0.temperature"] == 72.5
        assert ctx.grounding_values["step_0.city"] == "NYC"

    def test_merges_values_raw_keys(self) -> None:
        ctx = ConversationContext()
        ctx.add_tool_output(_make_response())
        assert ctx.grounding_values["temperature"] == 72.5
        assert ctx.grounding_values["city"] == "NYC"

    def test_adds_tool_message(self) -> None:
        ctx = ConversationContext()
        ctx.add_tool_output(_make_response())
        assert len(ctx.messages) == 1
        assert ctx.messages[0]["role"] == "tool"
        parsed = json.loads(ctx.messages[0]["content"])
        assert parsed["temperature"] == 72.5

    def test_multiple_outputs_accumulate(self) -> None:
        ctx = ConversationContext()
        ctx.add_tool_output(_make_response("c1"))
        ctx.advance_step()
        resp2 = ToolCallResponse(
            call_id="c2", status_code=200, data={"forecast": "rain"},
            extractable_values={"forecast": "rain"},
        )
        ctx.add_tool_output(resp2)
        assert len(ctx.tool_outputs) == 2
        assert ctx.grounding_values["step_0.temperature"] == 72.5
        assert ctx.grounding_values["step_1.forecast"] == "rain"
        assert ctx.grounding_values["forecast"] == "rain"


# ===========================================================================
# ConversationContext — get_available_values
# ===========================================================================


class TestContextAvailableValues:
    def test_empty(self) -> None:
        ctx = ConversationContext()
        assert ctx.get_available_values() == {}

    def test_after_output(self) -> None:
        ctx = ConversationContext()
        ctx.add_tool_output(_make_response())
        vals = ctx.get_available_values()
        assert "temperature" in vals
        assert "session_id" in vals  # from generated_ids

    def test_multiple_steps(self) -> None:
        ctx = ConversationContext()
        ctx.add_tool_output(_make_response("c1"))
        ctx.advance_step()
        resp2 = ToolCallResponse(
            call_id="c2", data={"x": 1},
            extractable_values={"wind": 10},
            generated_ids={"order_id": "O-99"},
        )
        ctx.add_tool_output(resp2)
        vals = ctx.get_available_values()
        assert vals["temperature"] == 72.5
        assert vals["wind"] == 10
        assert vals["session_id"] == "S-001"
        assert vals["order_id"] == "O-99"


# ===========================================================================
# ConversationContext — other methods
# ===========================================================================


class TestContextOtherMethods:
    def test_history_returns_copy(self) -> None:
        ctx = ConversationContext()
        ctx.add_message("user", "Hi")
        history = ctx.get_history_for_prompt()
        assert len(history) == 1
        # Mutating the copy should not affect original
        history.append({"role": "test", "content": "injected"})
        assert len(ctx.messages) == 1

    def test_last_output_none(self) -> None:
        ctx = ConversationContext()
        assert ctx.get_last_tool_output() is None

    def test_last_output(self) -> None:
        ctx = ConversationContext()
        ctx.add_tool_output(_make_response("c1"))
        ctx.add_tool_output(ToolCallResponse(call_id="c2", data={"x": 1}))
        last = ctx.get_last_tool_output()
        assert last is not None
        assert last.call_id == "c2"

    def test_advance_step(self) -> None:
        ctx = ConversationContext()
        assert ctx.current_step == 0
        ctx.advance_step()
        assert ctx.current_step == 1
        ctx.advance_step()
        assert ctx.current_step == 2
