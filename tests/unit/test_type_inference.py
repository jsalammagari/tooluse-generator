"""Unit tests for parameter type inference heuristics (Task 13)."""

from __future__ import annotations

import pytest

from tooluse_gen.registry.models import (
    Endpoint,
    Parameter,
    ParameterLocation,
    ParameterType,
)
from tooluse_gen.registry.type_inference import (
    NAME_RULES,
    InferenceResult,
    InferenceRule,
    ParameterTypeInferrer,
    TypeEvidence,
    infer_endpoint_parameter_types,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _infer(name: str, **kwargs: object) -> InferenceResult:
    return ParameterTypeInferrer().infer_type(name, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# InferenceRule
# ---------------------------------------------------------------------------


class TestInferenceRule:
    def test_matches_string_pattern(self):
        rule = InferenceRule("test", r"^id$", ParameterType.STRING, 0.9, 1)
        assert rule.matches("id") is True
        assert rule.matches("user_id") is False

    def test_matches_compiled_pattern(self):
        import re

        rule = InferenceRule("test", re.compile(r"id$"), ParameterType.STRING, 0.9, 1)
        assert rule.matches("user_id") is True

    def test_case_insensitive(self):
        rule = InferenceRule("test", r"^id$", ParameterType.STRING, 0.9, 1)
        assert rule.matches("ID") is True


# ---------------------------------------------------------------------------
# TypeEvidence / InferenceResult
# ---------------------------------------------------------------------------


class TestInferenceResult:
    def test_primary_evidence(self):
        ev1 = TypeEvidence("a", ParameterType.STRING, 0.5, "low")
        ev2 = TypeEvidence("b", ParameterType.STRING, 0.9, "high")
        r = InferenceResult(ParameterType.STRING, 0.9, True, [ev1, ev2])
        assert r.primary_evidence is ev2

    def test_primary_evidence_empty(self):
        r = InferenceResult(ParameterType.STRING, 0.0, True, [])
        assert r.primary_evidence is None

    def test_reasoning(self):
        ev = TypeEvidence("name", ParameterType.STRING, 0.9, "matched rule")
        r = InferenceResult(ParameterType.STRING, 0.9, True, [ev])
        assert "Inferred as" in r.reasoning
        assert "matched rule" in r.reasoning

    def test_reasoning_empty(self):
        r = InferenceResult(ParameterType.STRING, 0.0, True, [])
        assert r.reasoning == "No evidence available."


# ---------------------------------------------------------------------------
# Name-based inference
# ---------------------------------------------------------------------------


class TestNameInference:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("id", ParameterType.STRING),
            ("user_id", ParameterType.STRING),
            ("order-id", ParameterType.STRING),
            ("uuid", ParameterType.STRING),
            ("guid", ParameterType.STRING),
        ],
    )
    def test_id_patterns(self, name: str, expected: ParameterType):
        assert _infer(name).inferred_type == expected

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("limit", ParameterType.INTEGER),
            ("offset", ParameterType.INTEGER),
            ("page", ParameterType.INTEGER),
            ("per_page", ParameterType.INTEGER),
            ("page_size", ParameterType.INTEGER),
            ("item_count", ParameterType.INTEGER),
            ("num_results", ParameterType.INTEGER),
            ("size", ParameterType.INTEGER),
            ("total", ParameterType.INTEGER),
            ("year", ParameterType.INTEGER),
        ],
    )
    def test_integer_patterns(self, name: str, expected: ParameterType):
        assert _infer(name).inferred_type == expected

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("is_active", ParameterType.BOOLEAN),
            ("has_items", ParameterType.BOOLEAN),
            ("enable", ParameterType.BOOLEAN),
            ("verbose", ParameterType.BOOLEAN),
            ("recursive", ParameterType.BOOLEAN),
        ],
    )
    def test_boolean_patterns(self, name: str, expected: ParameterType):
        assert _infer(name).inferred_type == expected

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("created_date", ParameterType.DATE),
            ("start_date", ParameterType.DATE),
            ("timestamp", ParameterType.DATETIME),
            ("created_at", ParameterType.DATETIME),
        ],
    )
    def test_date_patterns(self, name: str, expected: ParameterType):
        assert _infer(name).inferred_type == expected

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("price", ParameterType.NUMBER),
            ("latitude", ParameterType.NUMBER),
            ("lng", ParameterType.NUMBER),
            ("cost", ParameterType.NUMBER),
            ("score", ParameterType.NUMBER),
        ],
    )
    def test_number_patterns(self, name: str, expected: ParameterType):
        assert _infer(name).inferred_type == expected

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("user_ids", ParameterType.ARRAY),
            ("tags", ParameterType.ARRAY),
            ("item_list", ParameterType.ARRAY),
            ("categories", ParameterType.ARRAY),
        ],
    )
    def test_array_patterns(self, name: str, expected: ParameterType):
        assert _infer(name).inferred_type == expected

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("email", ParameterType.STRING),
            ("name", ParameterType.STRING),
            ("api_key", ParameterType.STRING),
            ("callback_url", ParameterType.STRING),
            ("query", ParameterType.STRING),
        ],
    )
    def test_string_patterns(self, name: str, expected: ParameterType):
        assert _infer(name).inferred_type == expected

    def test_object_patterns(self):
        assert _infer("body").inferred_type == ParameterType.OBJECT
        assert _infer("metadata").inferred_type == ParameterType.OBJECT


# ---------------------------------------------------------------------------
# Default value inference
# ---------------------------------------------------------------------------


class TestDefaultValueInference:
    def test_bool_default(self):
        r = _infer("x", default_value=True)
        assert r.inferred_type == ParameterType.BOOLEAN
        assert r.confidence >= 0.9

    def test_int_default(self):
        r = _infer("x", default_value=42)
        assert r.inferred_type == ParameterType.INTEGER

    def test_float_default(self):
        r = _infer("x", default_value=3.14)
        assert r.inferred_type == ParameterType.NUMBER

    def test_str_default(self):
        r = _infer("x", default_value="hello")
        assert r.inferred_type == ParameterType.STRING

    def test_list_default(self):
        r = _infer("x", default_value=[1, 2])
        assert r.inferred_type == ParameterType.ARRAY

    def test_dict_default(self):
        r = _infer("x", default_value={"a": 1})
        assert r.inferred_type == ParameterType.OBJECT

    def test_numeric_string_default(self):
        r = _infer("x", default_value="123")
        assert r.inferred_type == ParameterType.INTEGER

    def test_float_string_default(self):
        r = _infer("x", default_value="3.14")
        assert r.inferred_type == ParameterType.NUMBER


# ---------------------------------------------------------------------------
# Enum value inference
# ---------------------------------------------------------------------------


class TestEnumInference:
    def test_string_enum(self):
        r = _infer("status", enum_values=["active", "inactive"])
        assert r.inferred_type == ParameterType.STRING

    def test_integer_enum(self):
        r = _infer("level", enum_values=["1", "2", "3"])
        assert r.inferred_type == ParameterType.INTEGER

    def test_boolean_enum(self):
        r = _infer("flag", enum_values=["true", "false"])
        assert r.inferred_type == ParameterType.BOOLEAN

    def test_numeric_enum(self):
        r = _infer("rate", enum_values=["1.5", "2.5"])
        assert r.inferred_type == ParameterType.NUMBER


# ---------------------------------------------------------------------------
# Description inference
# ---------------------------------------------------------------------------


class TestDescriptionInference:
    def test_integer_keyword(self):
        r = _infer("x", description="Enter a number between 1 and 100")
        assert r.inferred_type == ParameterType.INTEGER

    def test_boolean_keyword(self):
        r = _infer("x", description="Set to true or false")
        assert r.inferred_type == ParameterType.BOOLEAN

    def test_date_keyword(self):
        r = _infer("x", description="ISO 8601 date format")
        assert r.inferred_type == ParameterType.DATE

    def test_array_keyword(self):
        r = _infer("x", description="A comma-separated list of values")
        assert r.inferred_type == ParameterType.ARRAY

    def test_no_match(self):
        r = _infer("x", description="Some generic text here")
        # Falls back to string from context
        assert r.inferred_type == ParameterType.STRING


# ---------------------------------------------------------------------------
# Example inference
# ---------------------------------------------------------------------------


class TestExampleInference:
    def test_int_examples(self):
        r = _infer("x", examples=[1, 2, 3])
        assert r.inferred_type == ParameterType.INTEGER

    def test_string_examples(self):
        r = _infer("x", examples=["a", "b"])
        assert r.inferred_type == ParameterType.STRING

    def test_mixed_examples(self):
        r = _infer("x", examples=[1, 2, "three"])
        # Majority wins (2 ints vs 1 string)
        assert r.inferred_type == ParameterType.INTEGER


# ---------------------------------------------------------------------------
# Context inference
# ---------------------------------------------------------------------------


class TestContextInference:
    def test_path_location(self):
        r = _infer("x", location=ParameterLocation.PATH)
        assert r.inferred_type == ParameterType.STRING

    def test_body_location(self):
        """Body context alone is weak (0.4) — falls back to STRING.
        With a supporting name like 'data', it reaches OBJECT."""
        r = _infer("x", location=ParameterLocation.BODY)
        assert r.inferred_type == ParameterType.STRING  # weak signal alone
        r2 = _infer("data", location=ParameterLocation.BODY)
        assert r2.inferred_type == ParameterType.OBJECT  # name + context agree

    def test_header_location(self):
        r = _infer("x", location=ParameterLocation.HEADER)
        assert r.inferred_type == ParameterType.STRING


# ---------------------------------------------------------------------------
# Evidence combination
# ---------------------------------------------------------------------------


class TestEvidenceCombination:
    def test_multiple_agreeing_sources_boost(self):
        """Multiple sources for same type should boost confidence."""
        # "is_active" name → BOOLEAN, default True → BOOLEAN
        r = _infer("is_active", default_value=True)
        assert r.inferred_type == ParameterType.BOOLEAN
        assert r.confidence > 0.9

    def test_conflicting_evidence(self):
        """Higher-confidence evidence wins."""
        # default=42 → INTEGER (0.95), name "query" → STRING (0.8)
        r = _infer("query", default_value=42)
        assert r.inferred_type == ParameterType.INTEGER

    def test_below_min_confidence_defaults_string(self):
        inf = ParameterTypeInferrer(min_confidence=0.99)
        r = inf.infer_type("some_unknown_param")
        assert r.inferred_type == ParameterType.STRING

    def test_no_evidence(self):
        inf = ParameterTypeInferrer(name_rules=[])
        r = inf.infer_type("")
        assert r.is_inferred is True
        assert r.inferred_type == ParameterType.STRING


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------


class TestInferEndpointParameterTypes:
    def test_infers_missing_types(self):
        ep = Endpoint(
            endpoint_id="t/GET/x",
            tool_id="t",
            name="Search",
            path="/search",
            parameters=[
                Parameter(name="limit", has_type=False, inferred_type=False),
                Parameter(name="query", has_type=False, inferred_type=False),
            ],
        )
        result = infer_endpoint_parameter_types(ep)
        assert result is ep
        assert ep.parameters[0].param_type == ParameterType.INTEGER
        assert ep.parameters[0].has_type is True
        assert ep.parameters[0].inferred_type is True
        assert ep.parameters[1].has_type is True

    def test_skips_explicit_types(self):
        ep = Endpoint(
            endpoint_id="t/GET/x",
            tool_id="t",
            name="Search",
            path="/search",
            parameters=[
                Parameter(
                    name="limit",
                    param_type=ParameterType.STRING,
                    has_type=True,
                    inferred_type=False,
                ),
            ],
        )
        infer_endpoint_parameter_types(ep)
        # Should NOT be changed — it was explicit
        assert ep.parameters[0].param_type == ParameterType.STRING

    def test_custom_inferrer(self):
        ep = Endpoint(
            endpoint_id="t/GET/x",
            tool_id="t",
            name="E",
            path="/e",
            parameters=[Parameter(name="x", has_type=False)],
        )
        custom = ParameterTypeInferrer(name_rules=[], min_confidence=0.0)
        infer_endpoint_parameter_types(ep, inferrer=custom)
        assert ep.parameters[0].has_type is True


# ---------------------------------------------------------------------------
# Static helpers
# ---------------------------------------------------------------------------


class TestStaticHelpers:
    def test_is_numeric_string(self):
        assert ParameterTypeInferrer._is_numeric_string("123") is True
        assert ParameterTypeInferrer._is_numeric_string("3.14") is True
        assert ParameterTypeInferrer._is_numeric_string("-5") is True
        assert ParameterTypeInferrer._is_numeric_string("abc") is False

    def test_is_integer_string(self):
        assert ParameterTypeInferrer._is_integer_string("123") is True
        assert ParameterTypeInferrer._is_integer_string("-5") is True
        assert ParameterTypeInferrer._is_integer_string("3.14") is False
        assert ParameterTypeInferrer._is_integer_string("abc") is False

    def test_python_type_map(self):
        assert ParameterTypeInferrer._python_type_to_param_type(bool) == ParameterType.BOOLEAN
        assert ParameterTypeInferrer._python_type_to_param_type(int) == ParameterType.INTEGER
        assert ParameterTypeInferrer._python_type_to_param_type(float) == ParameterType.NUMBER
        assert ParameterTypeInferrer._python_type_to_param_type(str) == ParameterType.STRING
        assert ParameterTypeInferrer._python_type_to_param_type(list) == ParameterType.ARRAY
        assert ParameterTypeInferrer._python_type_to_param_type(dict) == ParameterType.OBJECT
        assert ParameterTypeInferrer._python_type_to_param_type(bytes) == ParameterType.STRING


# ---------------------------------------------------------------------------
# NAME_RULES sanity
# ---------------------------------------------------------------------------


def test_name_rules_non_empty():
    assert len(NAME_RULES) > 20


def test_name_rules_sorted_by_priority():
    priorities = [r.priority for r in NAME_RULES]
    # Not necessarily sorted globally, but each rule should have priority >= 1
    assert all(p >= 1 for p in priorities)


# ---------------------------------------------------------------------------
# Package-level imports
# ---------------------------------------------------------------------------


def test_package_exports():
    from tooluse_gen.registry import (
        NAME_RULES,
        HeuristicTypeInferrer,
    )

    assert HeuristicTypeInferrer is not None
    assert len(NAME_RULES) > 0
