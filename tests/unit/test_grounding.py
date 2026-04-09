"""Unit tests for Task 33 — Grounding value injection formatter."""

from __future__ import annotations

import pytest

from tooluse_gen.agents.execution_models import ConversationContext, ToolCallResponse
from tooluse_gen.agents.grounding import (
    GroundingTracker,
    ValueProvenance,
    format_available_values,
    format_grounding_context,
    format_value_for_prompt,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resp(
    extractable: dict[str, object] | None = None,
    gen_ids: dict[str, str] | None = None,
) -> ToolCallResponse:
    return ToolCallResponse(
        call_id="c1",
        data={"x": 1},
        extractable_values=dict(extractable or {}),
        generated_ids=dict(gen_ids or {}),
    )


def _ctx_with_values() -> ConversationContext:
    ctx = ConversationContext()
    resp = ToolCallResponse(
        call_id="c1",
        data={"temp": 72.5},
        extractable_values={"temperature": 72.5, "city": "Paris"},
        generated_ids={"hotel_id": "HOT-1234"},
    )
    ctx.add_tool_output(resp)
    return ctx


# ===========================================================================
# ValueProvenance
# ===========================================================================


class TestValueProvenance:
    def test_construction(self) -> None:
        p = ValueProvenance(
            value_key="hotel_id",
            value="HOT-1234",
            source_endpoint="hotel/POST/book",
            step_index=1,
            value_type="id",
        )
        assert p.value_key == "hotel_id"
        assert p.value == "HOT-1234"
        assert p.source_endpoint == "hotel/POST/book"
        assert p.step_index == 1
        assert p.value_type == "id"

    def test_default_value_type(self) -> None:
        p = ValueProvenance(
            value_key="city", value="NYC", source_endpoint="ep", step_index=0
        )
        assert p.value_type == ""

    def test_round_trip(self) -> None:
        p = ValueProvenance(
            value_key="k", value=42, source_endpoint="ep", step_index=2, value_type="count"
        )
        restored = ValueProvenance.model_validate(p.model_dump())
        assert restored.value_key == "k"
        assert restored.value == 42


# ===========================================================================
# GroundingTracker — track_value
# ===========================================================================


class TestTrackValue:
    def test_tracks_key(self) -> None:
        t = GroundingTracker()
        t.track_value("city", "NYC", "ep", 0)
        assert t.get_provenance("city") is not None

    def test_tracks_step_prefix(self) -> None:
        t = GroundingTracker()
        t.track_value("city", "NYC", "ep", 0)
        assert t.get_provenance("step_0.city") is not None

    def test_multiple_accumulated(self) -> None:
        t = GroundingTracker()
        t.track_value("city", "NYC", "ep1", 0)
        t.track_value("price", 99.99, "ep2", 1)
        assert t.get_provenance("city") is not None
        assert t.get_provenance("price") is not None

    def test_overwrite(self) -> None:
        t = GroundingTracker()
        t.track_value("city", "NYC", "ep1", 0)
        t.track_value("city", "London", "ep2", 1)
        prov = t.get_provenance("city")
        assert prov is not None and prov.value == "London"


# ===========================================================================
# GroundingTracker — track_from_response
# ===========================================================================


class TestTrackFromResponse:
    def test_tracks_extractable(self) -> None:
        t = GroundingTracker()
        resp = _resp(extractable={"temp": 72.5, "city": "Paris"})
        t.track_from_response(resp, "weather/GET/cur", 0)
        assert t.get_provenance("temp") is not None
        assert t.get_provenance("city") is not None

    def test_tracks_generated_ids(self) -> None:
        t = GroundingTracker()
        resp = _resp(gen_ids={"hotel_id": "HOT-1234"})
        t.track_from_response(resp, "hotel/POST/book", 1)
        prov = t.get_provenance("hotel_id")
        assert prov is not None
        assert prov.value_type == "id"

    def test_empty_response(self) -> None:
        t = GroundingTracker()
        t.track_from_response(_resp(), "ep", 0)
        assert t.get_all_provenance() == {}


# ===========================================================================
# GroundingTracker — get_provenance
# ===========================================================================


class TestGetProvenance:
    def test_returns_tracked(self) -> None:
        t = GroundingTracker()
        t.track_value("city", "NYC", "ep", 0)
        p = t.get_provenance("city")
        assert p is not None and p.value == "NYC"

    def test_returns_none_unknown(self) -> None:
        t = GroundingTracker()
        assert t.get_provenance("nonexistent") is None

    def test_step_prefixed(self) -> None:
        t = GroundingTracker()
        t.track_value("city", "NYC", "ep", 2)
        p = t.get_provenance("step_2.city")
        assert p is not None and p.value == "NYC"


# ===========================================================================
# GroundingTracker — reset
# ===========================================================================


class TestTrackerReset:
    def test_clears(self) -> None:
        t = GroundingTracker()
        t.track_value("city", "NYC", "ep", 0)
        t.reset()
        assert t.get_provenance("city") is None
        assert t.get_all_provenance() == {}


# ===========================================================================
# format_available_values
# ===========================================================================


class TestFormatAvailableValues:
    def test_empty_context(self) -> None:
        ctx = ConversationContext()
        assert format_available_values(ctx) == "No values available from prior tool calls."

    def test_with_values_no_tracker(self) -> None:
        ctx = _ctx_with_values()
        text = format_available_values(ctx)
        assert "city: Paris" in text
        assert "temperature: 72.50" in text
        assert "hotel_id: HOT-1234" in text

    def test_with_tracker(self) -> None:
        ctx = _ctx_with_values()
        tracker = GroundingTracker()
        resp = _resp(
            extractable={"temperature": 72.5, "city": "Paris"},
            gen_ids={"hotel_id": "HOT-1234"},
        )
        tracker.track_from_response(resp, "weather/GET/cur", 0)
        text = format_available_values(ctx, tracker)
        assert "weather/GET/cur" in text
        assert "step 0" in text

    def test_skips_step_prefixed(self) -> None:
        ctx = _ctx_with_values()
        text = format_available_values(ctx)
        assert "step_0." not in text

    def test_long_values_truncated(self) -> None:
        ctx = ConversationContext(grounding_values={"desc": "x" * 100})
        text = format_available_values(ctx)
        assert "..." in text

    def test_sorted_alphabetically(self) -> None:
        ctx = ConversationContext(grounding_values={"zebra": 1, "apple": 2, "mango": 3})
        text = format_available_values(ctx)
        lines = text.strip().split("\n")[1:]  # skip header
        keys = [line.split(":")[0].strip("- ") for line in lines]
        assert keys == sorted(keys)


# ===========================================================================
# format_grounding_context
# ===========================================================================


class TestFormatGroundingContext:
    def test_empty_context(self) -> None:
        ctx = ConversationContext()
        result = format_grounding_context(ctx)
        assert result["available_values"] == {}
        assert result["generated_ids"] == {}
        assert result["current_step"] == 0
        assert result["prior_tool_calls"] == 0

    def test_with_values(self) -> None:
        ctx = _ctx_with_values()
        result = format_grounding_context(ctx)
        assert "temperature" in result["available_values"]
        assert "city" in result["available_values"]

    def test_skips_step_prefixed(self) -> None:
        ctx = _ctx_with_values()
        result = format_grounding_context(ctx)
        for key in result["available_values"]:
            assert "." not in key

    def test_includes_metadata(self) -> None:
        ctx = _ctx_with_values()
        result = format_grounding_context(ctx)
        assert result["generated_ids"] == {"hotel_id": "HOT-1234"}
        assert result["current_step"] == 0
        assert result["prior_tool_calls"] == 1


# ===========================================================================
# format_value_for_prompt
# ===========================================================================


class TestFormatValueForPrompt:
    def test_string(self) -> None:
        assert format_value_for_prompt("hello") == "hello"

    def test_long_string(self) -> None:
        result = format_value_for_prompt("x" * 100)
        assert len(result) <= 54  # 50 + "..."
        assert result.endswith("...")

    def test_float(self) -> None:
        assert format_value_for_prompt(72.5) == "72.50"

    def test_integer(self) -> None:
        assert format_value_for_prompt(42) == "42"

    def test_bool_true(self) -> None:
        assert format_value_for_prompt(True) == "true"

    def test_bool_false(self) -> None:
        assert format_value_for_prompt(False) == "false"

    def test_list_short(self) -> None:
        assert format_value_for_prompt([1, 2, 3]) == "[1, 2, 3]"

    def test_list_long(self) -> None:
        assert format_value_for_prompt([1, 2, 3, 4, 5]) == "[1, 2, 3, ...]"

    def test_dict(self) -> None:
        assert format_value_for_prompt({"a": 1, "b": 2}) == "{a, b}"

    def test_none(self) -> None:
        assert format_value_for_prompt(None) == "null"
