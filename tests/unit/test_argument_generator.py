"""Unit tests for Task 32 — Argument generator."""

from __future__ import annotations

import numpy as np
import pytest

from tooluse_gen.agents.argument_generator import ArgumentGenerator
from tooluse_gen.agents.execution_models import ConversationContext
from tooluse_gen.agents.value_generator import ValuePool
from tooluse_gen.registry.models import Endpoint, HttpMethod, Parameter, ParameterType

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _param(
    name: str = "city",
    param_type: ParameterType = ParameterType.STRING,
    required: bool = True,
    **kw: object,
) -> Parameter:
    return Parameter(name=name, param_type=param_type, required=required, **kw)  # type: ignore[arg-type]


def _ep(
    params: list[Parameter] | None = None,
    required: list[str] | None = None,
) -> Endpoint:
    return Endpoint(
        endpoint_id="e1",
        tool_id="t1",
        name="Test",
        method=HttpMethod.GET,
        path="/test",
        parameters=params or [],
        required_parameters=required or [],
    )


# ===========================================================================
# Construction
# ===========================================================================


class TestConstruction:
    def test_default_pool(self) -> None:
        gen = ArgumentGenerator()
        assert gen._pool is not None

    def test_custom_pool(self) -> None:
        pool = ValuePool(seed_data={"custom": ["a", "b"]})
        gen = ArgumentGenerator(pool=pool)
        assert gen._pool is pool


# ===========================================================================
# generate_arguments
# ===========================================================================


class TestGenerateArguments:
    def test_returns_dict(self) -> None:
        gen = ArgumentGenerator()
        ep = _ep([_param("city", required=True)], ["city"])
        result = gen.generate_arguments(ep, ConversationContext(), np.random.default_rng(42))
        assert isinstance(result, dict)

    def test_required_present(self) -> None:
        gen = ArgumentGenerator()
        ep = _ep(
            [_param("city", required=True), _param("date", ParameterType.DATE, required=True)],
            ["city", "date"],
        )
        result = gen.generate_arguments(ep, ConversationContext(), np.random.default_rng(42))
        assert "city" in result
        assert "date" in result

    def test_optional_sometimes_included(self) -> None:
        gen = ArgumentGenerator()
        ep = _ep(
            [_param("opt1", required=False), _param("opt2", required=False),
             _param("opt3", required=False), _param("opt4", required=False)],
        )
        included_counts = 0
        for seed in range(20):
            result = gen.generate_arguments(ep, ConversationContext(), np.random.default_rng(seed))
            included_counts += len(result)
        # With p=0.5 and 4 optional params over 20 runs, expect some but not all
        assert 0 < included_counts < 80

    def test_empty_endpoint(self) -> None:
        gen = ArgumentGenerator()
        ep = _ep()
        result = gen.generate_arguments(ep, ConversationContext(), np.random.default_rng(42))
        assert result == {}

    def test_only_optional_can_be_empty(self) -> None:
        gen = ArgumentGenerator()
        ep = _ep([_param("opt", required=False)])
        # With enough seeds, at least one run should produce empty
        empties = sum(
            1
            for seed in range(20)
            if not gen.generate_arguments(ep, ConversationContext(), np.random.default_rng(seed))
        )
        assert empties > 0


# ===========================================================================
# _fill_required_params
# ===========================================================================


class TestFillRequired:
    def test_always_has_value(self) -> None:
        gen = ArgumentGenerator()
        ep = _ep([_param("city", required=True)], ["city"])
        result = gen._fill_required_params(ep, ConversationContext(), np.random.default_rng(42))
        assert "city" in result and result["city"] is not None

    def test_enum_picks_from_values(self) -> None:
        gen = ArgumentGenerator()
        ep = _ep(
            [_param("status", required=True, enum_values=["active", "inactive"])],
            ["status"],
        )
        result = gen._fill_required_params(ep, ConversationContext(), np.random.default_rng(42))
        assert result["status"] in ("active", "inactive")

    def test_default_used_no_grounding(self) -> None:
        gen = ArgumentGenerator()
        ep = _ep([_param("guests", ParameterType.INTEGER, required=True, default=1)], ["guests"])
        result = gen._fill_required_params(ep, ConversationContext(), np.random.default_rng(42))
        assert result["guests"] == 1

    def test_grounded_value_preferred(self) -> None:
        gen = ArgumentGenerator()
        ep = _ep([_param("city", required=True)], ["city"])
        ctx = ConversationContext(grounding_values={"city": "Paris"})
        result = gen._fill_required_params(ep, ctx, np.random.default_rng(42))
        assert result["city"] == "Paris"

    def test_multiple_required(self) -> None:
        gen = ArgumentGenerator()
        ep = _ep(
            [_param("a", required=True), _param("b", required=True), _param("c", required=True)],
            ["a", "b", "c"],
        )
        result = gen._fill_required_params(ep, ConversationContext(), np.random.default_rng(42))
        assert len(result) == 3


# ===========================================================================
# _fill_optional_params
# ===========================================================================


class TestFillOptional:
    def test_all_included_probability_one(self) -> None:
        gen = ArgumentGenerator()
        ep = _ep([_param("a", required=False), _param("b", required=False)])
        result = gen._fill_optional_params(
            ep, ConversationContext(), np.random.default_rng(42), include_probability=1.0
        )
        assert "a" in result and "b" in result

    def test_none_included_probability_zero(self) -> None:
        gen = ArgumentGenerator()
        ep = _ep([_param("a", required=False), _param("b", required=False)])
        result = gen._fill_optional_params(
            ep, ConversationContext(), np.random.default_rng(42), include_probability=0.0
        )
        assert result == {}

    def test_enum_respected(self) -> None:
        gen = ArgumentGenerator()
        ep = _ep([_param("cur", required=False, enum_values=["USD", "EUR"])])
        result = gen._fill_optional_params(
            ep, ConversationContext(), np.random.default_rng(42), include_probability=1.0
        )
        if "cur" in result:
            assert result["cur"] in ("USD", "EUR")

    def test_default_used(self) -> None:
        gen = ArgumentGenerator()
        ep = _ep([_param("limit", ParameterType.INTEGER, required=False, default=10)])
        result = gen._fill_optional_params(
            ep, ConversationContext(), np.random.default_rng(42), include_probability=1.0
        )
        assert result.get("limit") == 10


# ===========================================================================
# _resolve_grounded_value
# ===========================================================================


class TestResolveGrounded:
    def test_exact_match(self) -> None:
        gen = ArgumentGenerator()
        ctx = ConversationContext(grounding_values={"city": "NYC"})
        result = gen._resolve_grounded_value(_param("city"), ctx)
        assert result == "NYC"

    def test_step_prefix_match(self) -> None:
        gen = ArgumentGenerator()
        ctx = ConversationContext(grounding_values={"step_0.city": "Tokyo"})
        result = gen._resolve_grounded_value(_param("city"), ctx)
        assert result == "Tokyo"

    def test_substring_match(self) -> None:
        gen = ArgumentGenerator()
        ctx = ConversationContext(grounding_values={"hotel_id": "HOT-1234"})
        result = gen._resolve_grounded_value(_param("hotel_id"), ctx)
        assert result == "HOT-1234"

    def test_no_match(self) -> None:
        gen = ArgumentGenerator()
        ctx = ConversationContext(grounding_values={"temperature": 72.5})
        result = gen._resolve_grounded_value(_param("city"), ctx)
        assert result is None

    def test_empty_context(self) -> None:
        gen = ArgumentGenerator()
        result = gen._resolve_grounded_value(_param("city"), ConversationContext())
        assert result is None


# ===========================================================================
# _generate_fresh_value
# ===========================================================================


class TestGenerateFresh:
    def test_string(self) -> None:
        gen = ArgumentGenerator()
        result = gen._generate_fresh_value(_param("city", ParameterType.STRING), np.random.default_rng(42))
        assert isinstance(result, str)

    def test_integer(self) -> None:
        gen = ArgumentGenerator()
        result = gen._generate_fresh_value(_param("count", ParameterType.INTEGER), np.random.default_rng(42))
        assert isinstance(result, int)

    def test_number(self) -> None:
        gen = ArgumentGenerator()
        result = gen._generate_fresh_value(_param("price", ParameterType.NUMBER), np.random.default_rng(42))
        assert isinstance(result, (int, float))

    def test_boolean(self) -> None:
        gen = ArgumentGenerator()
        result = gen._generate_fresh_value(_param("active", ParameterType.BOOLEAN), np.random.default_rng(42))
        assert isinstance(result, bool)

    def test_date(self) -> None:
        gen = ArgumentGenerator()
        result = gen._generate_fresh_value(_param("check_in", ParameterType.DATE), np.random.default_rng(42))
        assert isinstance(result, str) and "-" in result

    def test_fuzzy_city(self) -> None:
        gen = ArgumentGenerator()
        result = gen._generate_fresh_value(_param("city_name", ParameterType.STRING), np.random.default_rng(42))
        assert isinstance(result, str) and len(result) > 2


# ===========================================================================
# _match_param_to_grounding
# ===========================================================================


class TestMatchGrounding:
    def test_exact(self) -> None:
        gen = ArgumentGenerator()
        result = gen._match_param_to_grounding("city", "string", {"city": "NYC", "other": 1})
        assert result == "city"

    def test_step_prefix(self) -> None:
        gen = ArgumentGenerator()
        result = gen._match_param_to_grounding("city", "string", {"step_0.city": "NYC"})
        assert result == "step_0.city"

    def test_substring(self) -> None:
        gen = ArgumentGenerator()
        result = gen._match_param_to_grounding("hotel_id", "string", {"hotel_id": "H-1"})
        assert result == "hotel_id"

    def test_no_match(self) -> None:
        gen = ArgumentGenerator()
        result = gen._match_param_to_grounding("zzz", "string", {"abc": 1})
        assert result is None


# ===========================================================================
# Integration
# ===========================================================================


class TestIntegration:
    def test_grounding_produces_grounded(self) -> None:
        gen = ArgumentGenerator()
        ep = _ep(
            [_param("city", required=True), _param("hotel_id", required=True)],
            ["city", "hotel_id"],
        )
        ctx = ConversationContext(
            grounding_values={"city": "London", "hotel_id": "HOT-999"}
        )
        result = gen.generate_arguments(ep, ctx, np.random.default_rng(42))
        assert result["city"] == "London"
        assert result["hotel_id"] == "HOT-999"

    def test_deterministic(self) -> None:
        gen = ArgumentGenerator()
        ep = _ep(
            [_param("city", required=True), _param("price", ParameterType.NUMBER, required=True)],
            ["city", "price"],
        )
        a1 = gen.generate_arguments(ep, ConversationContext(), np.random.default_rng(99))
        a2 = gen.generate_arguments(ep, ConversationContext(), np.random.default_rng(99))
        assert a1 == a2
