"""Unit tests for registry Pydantic models (Task 8)."""

from __future__ import annotations

import json

import pytest

from tooluse_gen.registry.models import (
    Endpoint,
    HttpMethod,
    Parameter,
    ParameterLocation,
    ParameterType,
    ResponseSchema,
    Tool,
    generate_endpoint_id,
    normalize_parameter_name,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_param(**kwargs: object) -> Parameter:
    return Parameter(name="q", **kwargs)  # type: ignore[arg-type]


def _make_endpoint(**kwargs: object) -> Endpoint:
    defaults: dict[str, object] = {
        "endpoint_id": "tool1/GET/abcd1234",
        "tool_id": "tool1",
        "name": "Search",
        "path": "/search",
    }
    defaults.update(kwargs)
    return Endpoint(**defaults)  # type: ignore[arg-type]


def _make_tool(**kwargs: object) -> Tool:
    defaults: dict[str, object] = {"tool_id": "t1", "name": "My Tool"}
    defaults.update(kwargs)
    return Tool(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


def test_parameter_type_values():
    assert ParameterType.STRING == "string"
    assert ParameterType.INTEGER == "integer"
    assert ParameterType.UNKNOWN == "unknown"


def test_http_method_values():
    assert HttpMethod.GET == "GET"
    assert HttpMethod.DELETE == "DELETE"


def test_parameter_location_values():
    assert ParameterLocation.QUERY == "query"
    assert ParameterLocation.BODY == "body"


# ---------------------------------------------------------------------------
# normalize_parameter_name
# ---------------------------------------------------------------------------


def test_normalize_strips_whitespace():
    assert normalize_parameter_name("  foo  ") == "foo"


def test_normalize_lowercases():
    assert normalize_parameter_name("ContentType") == "contenttype"


def test_normalize_replaces_special_chars():
    assert normalize_parameter_name("Content-Type") == "content_type"


def test_normalize_strips_leading_dollar():
    assert normalize_parameter_name("$filter") == "filter"


def test_normalize_empty_becomes_unknown():
    assert normalize_parameter_name("   ") == "unknown"


def test_normalize_collapses_multiple_unsafe():
    assert normalize_parameter_name("a--b") == "a__b"


# ---------------------------------------------------------------------------
# generate_endpoint_id
# ---------------------------------------------------------------------------


def test_generate_endpoint_id_format():
    eid = generate_endpoint_id("tool1", "get", "/search")
    parts = eid.split("/")
    assert parts[0] == "tool1"
    assert parts[1] == "GET"
    assert len(parts[2]) == 8


def test_generate_endpoint_id_stable():
    id1 = generate_endpoint_id("tool1", "GET", "/users/{id}")
    id2 = generate_endpoint_id("tool1", "GET", "/users/{id}")
    assert id1 == id2


def test_generate_endpoint_id_path_param_normalised():
    # Different path param styles should yield same ID
    id1 = generate_endpoint_id("t", "GET", "/users/{id}/posts")
    id2 = generate_endpoint_id("t", "GET", "/users/:id/posts")
    assert id1 == id2


def test_generate_endpoint_id_different_methods_differ():
    id_get = generate_endpoint_id("t", "GET", "/items")
    id_post = generate_endpoint_id("t", "POST", "/items")
    assert id_get != id_post


def test_generate_endpoint_id_case_insensitive_method():
    assert generate_endpoint_id("t", "get", "/x") == generate_endpoint_id("t", "GET", "/x")


# ---------------------------------------------------------------------------
# Parameter
# ---------------------------------------------------------------------------


def test_parameter_defaults():
    p = Parameter(name="q")
    assert p.description == ""
    assert p.param_type == ParameterType.STRING
    assert p.location == ParameterLocation.QUERY
    assert p.required is False
    assert p.default is None
    assert p.enum_values is None
    assert p.has_description is False
    assert p.has_type is False
    assert p.inferred_type is False


def test_parameter_name_normalised():
    p = Parameter(name="Content-Type")
    assert p.name == "content_type"


def test_parameter_description_stripped():
    p = Parameter(name="q", description="  hello  ")
    assert p.description == "hello"


def test_parameter_raw_definition_excluded_from_serialisation():
    p = Parameter(name="q", raw_definition={"type": "string"})
    data = p.model_dump()
    assert "raw_definition" not in data


def test_parameter_enum_values():
    p = Parameter(name="sort", enum_values=["asc", "desc"])
    assert p.enum_values == ["asc", "desc"]


def test_parameter_array_with_items_type():
    p = Parameter(name="ids", param_type=ParameterType.ARRAY, items_type=ParameterType.INTEGER)
    assert p.items_type == ParameterType.INTEGER


def test_parameter_object_with_properties():
    child = Parameter(name="city")
    p = Parameter(name="location", param_type=ParameterType.OBJECT, properties={"city": child})
    assert p.properties is not None
    assert "city" in p.properties


def test_parameter_serialisation_uses_string_values():
    p = Parameter(name="q", param_type=ParameterType.INTEGER, location=ParameterLocation.PATH)
    data = p.model_dump()
    assert data["param_type"] == "integer"
    assert data["location"] == "path"


def test_parameter_quality_indicators_set():
    p = Parameter(name="q", has_description=True, has_type=True, inferred_type=True)
    assert p.has_description
    assert p.has_type
    assert p.inferred_type


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


def test_endpoint_defaults():
    ep = _make_endpoint()
    assert ep.description == ""
    assert ep.method == HttpMethod.GET
    assert ep.parameters == []
    assert ep.required_parameters == []
    assert ep.response_schema is None
    assert ep.completeness_score == 0.0


def test_endpoint_has_required_params_false():
    ep = _make_endpoint()
    assert ep.has_required_params is False


def test_endpoint_has_required_params_true():
    ep = _make_endpoint(required_parameters=["q"])
    assert ep.has_required_params is True


def test_endpoint_param_count():
    params = [Parameter(name="a"), Parameter(name="b")]
    ep = _make_endpoint(parameters=params)
    assert ep.param_count == 2


def test_endpoint_get_parameter_found():
    params = [Parameter(name="q"), Parameter(name="limit")]
    ep = _make_endpoint(parameters=params)
    result = ep.get_parameter("q")
    assert result is not None
    assert result.name == "q"


def test_endpoint_get_parameter_not_found():
    ep = _make_endpoint()
    assert ep.get_parameter("missing") is None


def test_endpoint_raw_definition_excluded():
    ep = _make_endpoint(raw_definition={"x": 1})
    data = ep.model_dump()
    assert "raw_definition" not in data


def test_endpoint_completeness_bounds():
    ep = _make_endpoint(completeness_score=0.75)
    assert ep.completeness_score == 0.75
    with pytest.raises(ValueError):
        _make_endpoint(completeness_score=1.5)


def test_endpoint_method_serialised_as_string():
    ep = _make_endpoint(method=HttpMethod.POST)
    data = ep.model_dump()
    assert data["method"] == "POST"


def test_endpoint_with_response_schema():
    rs = ResponseSchema(status_code=200, description="OK")
    ep = _make_endpoint(response_schema=rs)
    assert ep.response_schema is not None
    assert ep.response_schema.status_code == 200


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


def test_tool_defaults():
    t = _make_tool()
    assert t.description == ""
    assert t.domain == ""
    assert t.base_url == ""
    assert t.auth_type == "none"
    assert t.endpoints == []
    assert t.completeness_score == 0.0
    assert t.created_at is None
    assert t.raw_schema is None


def test_tool_endpoint_count_computed():
    ep1 = _make_endpoint(endpoint_id="t1/GET/aaa00001", tool_id="t1")
    ep2 = _make_endpoint(endpoint_id="t1/POST/bbb00002", tool_id="t1")
    t = _make_tool(endpoints=[ep1, ep2])
    assert t.endpoint_count == 2


def test_tool_endpoint_count_in_serialisation():
    t = _make_tool()
    data = t.model_dump()
    assert "endpoint_count" in data
    assert data["endpoint_count"] == 0


def test_tool_is_complete_false():
    t = _make_tool(completeness_score=0.4)
    assert t.is_complete is False


def test_tool_is_complete_true():
    t = _make_tool(completeness_score=0.5)
    assert t.is_complete is True


def test_tool_endpoint_ids():
    ep1 = _make_endpoint(endpoint_id="t1/GET/aaa00001", tool_id="t1")
    ep2 = _make_endpoint(endpoint_id="t1/POST/bbb00002", tool_id="t1")
    t = _make_tool(endpoints=[ep1, ep2])
    assert t.endpoint_ids == ["t1/GET/aaa00001", "t1/POST/bbb00002"]


def test_tool_get_endpoint_found():
    ep = _make_endpoint(endpoint_id="t1/GET/aaa00001", tool_id="t1")
    t = _make_tool(endpoints=[ep])
    assert t.get_endpoint("t1/GET/aaa00001") is ep


def test_tool_get_endpoint_not_found():
    t = _make_tool()
    assert t.get_endpoint("missing") is None


def test_tool_get_endpoints_by_method():
    ep_get = _make_endpoint(endpoint_id="t1/GET/aaa00001", tool_id="t1", method=HttpMethod.GET)
    ep_post = _make_endpoint(
        endpoint_id="t1/POST/bbb00002", tool_id="t1", method=HttpMethod.POST
    )
    t = _make_tool(endpoints=[ep_get, ep_post])
    gets = t.get_endpoints_by_method(HttpMethod.GET)
    assert len(gets) == 1
    assert gets[0].endpoint_id == "t1/GET/aaa00001"


def test_tool_raw_schema_excluded():
    t = _make_tool(raw_schema={"api": "data"})
    data = t.model_dump()
    assert "raw_schema" not in data


def test_tool_json_round_trip():
    ep = _make_endpoint(endpoint_id="t1/GET/aaa00001", tool_id="t1")
    t = _make_tool(
        tool_id="t1",
        name="WeatherAPI",
        domain="Weather",
        endpoints=[ep],
        completeness_score=0.8,
    )
    json_str = t.model_dump_json()
    data = json.loads(json_str)
    assert data["tool_id"] == "t1"
    assert data["endpoint_count"] == 1
    assert data["completeness_score"] == 0.8
    assert "raw_schema" not in data


# ---------------------------------------------------------------------------
# ResponseSchema
# ---------------------------------------------------------------------------


def test_response_schema_defaults():
    rs = ResponseSchema()
    assert rs.status_code == 200
    assert rs.description == ""
    assert rs.schema_type == ParameterType.OBJECT
    assert rs.properties == {}
    assert rs.example is None


def test_response_schema_custom():
    rs = ResponseSchema(
        status_code=201,
        description="Created",
        schema_type=ParameterType.ARRAY,
        example=[{"id": 1}],
    )
    assert rs.status_code == 201
    assert rs.schema_type == "array"
