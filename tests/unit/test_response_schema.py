"""Unit tests for the rich response schema models (Task 9)."""

from __future__ import annotations

import json

import pytest

from tooluse_gen.registry.response_schema import (
    DEFAULT_LIST_RESPONSE,
    DEFAULT_OBJECT_RESPONSE,
    ExtractableType,
    FieldType,
    ResponseField,
    ResponseSchema,
    calculate_schema_completeness,
    flatten_response_fields,
    identify_extractable_type,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _field(name: str = "f", **kwargs: object) -> ResponseField:
    return ResponseField(name=name, **kwargs)  # type: ignore[arg-type]


def _schema(**kwargs: object) -> ResponseSchema:
    return ResponseSchema(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# FieldType enum
# ---------------------------------------------------------------------------


def test_field_type_values():
    assert FieldType.STRING == "string"
    assert FieldType.ANY == "any"
    assert FieldType.NULL == "null"
    assert FieldType.ARRAY == "array"
    assert FieldType.OBJECT == "object"


# ---------------------------------------------------------------------------
# ExtractableType enum
# ---------------------------------------------------------------------------


def test_extractable_type_values():
    assert ExtractableType.ID == "id"
    assert ExtractableType.NONE == "none"
    assert ExtractableType.URL == "url"
    assert ExtractableType.DATE == "date"
    assert ExtractableType.COUNT == "count"
    assert ExtractableType.STATUS == "status"
    assert ExtractableType.REFERENCE == "reference"
    assert ExtractableType.NAME == "name"


# ---------------------------------------------------------------------------
# ResponseField — defaults and basic behaviour
# ---------------------------------------------------------------------------


def test_response_field_defaults():
    f = _field()
    assert f.name == "f"
    assert f.field_type == FieldType.ANY
    assert f.description == ""
    assert f.nullable is True
    assert f.items_type is None
    assert f.items_schema is None
    assert f.properties == {}
    assert f.extractable_type == ExtractableType.NONE
    assert f.extractable_confidence == 0.0
    assert f.example is None


def test_response_field_custom():
    f = _field(
        name="user_id",
        field_type=FieldType.STRING,
        description="  The user identifier  ",
        nullable=False,
        extractable_type=ExtractableType.ID,
        extractable_confidence=0.9,
        example="usr_123",
    )
    assert f.name == "user_id"
    assert f.field_type == "string"
    assert f.description == "The user identifier"  # stripped
    assert f.nullable is False
    assert f.extractable_confidence == 0.9
    assert f.example == "usr_123"


def test_response_field_is_extractable():
    assert _field(extractable_type=ExtractableType.ID).is_extractable is True
    assert _field(extractable_type=ExtractableType.NONE).is_extractable is False
    assert _field().is_extractable is False


def test_response_field_is_collection():
    assert _field(field_type=FieldType.ARRAY).is_collection is True
    assert _field(field_type=FieldType.STRING).is_collection is False


def test_response_field_is_nested():
    child = _field(name="city")
    f = _field(field_type=FieldType.OBJECT, properties={"city": child})
    assert f.is_nested is True


def test_response_field_is_nested_empty_properties():
    f = _field(field_type=FieldType.OBJECT)
    assert f.is_nested is False


def test_response_field_is_nested_non_object():
    f = _field(field_type=FieldType.STRING)
    assert f.is_nested is False


def test_response_field_array_with_items_schema():
    item = _field(name="item", field_type=FieldType.OBJECT)
    f = _field(field_type=FieldType.ARRAY, items_type=FieldType.OBJECT, items_schema=item)
    assert f.is_collection is True
    assert f.items_schema is not None
    assert f.items_type == "object"


def test_response_field_extractable_confidence_bounds():
    _field(extractable_confidence=0.0)
    _field(extractable_confidence=1.0)
    with pytest.raises(ValueError):
        _field(extractable_confidence=-0.1)
    with pytest.raises(ValueError):
        _field(extractable_confidence=1.1)


def test_response_field_json_round_trip():
    f = _field(
        name="order_id",
        field_type=FieldType.STRING,
        extractable_type=ExtractableType.ID,
        extractable_confidence=0.8,
    )
    data = json.loads(f.model_dump_json())
    assert data["name"] == "order_id"
    assert data["field_type"] == "string"
    assert data["extractable_type"] == "id"


# ---------------------------------------------------------------------------
# ResponseSchema — defaults and basic behaviour
# ---------------------------------------------------------------------------


def test_response_schema_defaults():
    s = _schema()
    assert s.status_code == 200
    assert s.content_type == "application/json"
    assert s.body_type == FieldType.OBJECT
    assert s.fields == {}
    assert s.items_schema is None
    assert s.extractable_fields == []
    assert s.has_schema is True
    assert s.schema_completeness == 0.0
    assert s.raw_schema is None


def test_response_schema_custom():
    id_field = _field(name="id", extractable_type=ExtractableType.ID)
    s = _schema(
        status_code=201,
        content_type="text/plain",
        body_type=FieldType.OBJECT,
        fields={"id": id_field},
        extractable_fields=["id"],
        has_schema=True,
        schema_completeness=0.8,
    )
    assert s.status_code == 201
    assert s.content_type == "text/plain"
    assert s.has_extractable_fields is True


def test_response_schema_is_list_response():
    assert _schema(body_type=FieldType.ARRAY).is_list_response is True
    assert _schema(body_type=FieldType.OBJECT).is_list_response is False


def test_response_schema_has_extractable_fields():
    assert _schema(extractable_fields=["id"]).has_extractable_fields is True
    assert _schema().has_extractable_fields is False


def test_response_schema_raw_schema_excluded():
    s = _schema(raw_schema={"type": "object"})
    data = s.model_dump()
    assert "raw_schema" not in data


def test_response_schema_completeness_bounds():
    _schema(schema_completeness=0.0)
    _schema(schema_completeness=1.0)
    with pytest.raises(ValueError):
        _schema(schema_completeness=-0.1)
    with pytest.raises(ValueError):
        _schema(schema_completeness=1.1)


# ---------------------------------------------------------------------------
# ResponseSchema.get_field
# ---------------------------------------------------------------------------


def test_get_field_simple():
    f = _field(name="id", field_type=FieldType.STRING)
    s = _schema(fields={"id": f})
    assert s.get_field("id") is f


def test_get_field_missing():
    s = _schema()
    assert s.get_field("missing") is None


def test_get_field_nested():
    city = _field(name="city", field_type=FieldType.STRING)
    address = _field(name="address", field_type=FieldType.OBJECT, properties={"city": city})
    s = _schema(fields={"address": address})
    assert s.get_field("address.city") is city


def test_get_field_array_index():
    item_id = _field(name="id", field_type=FieldType.STRING)
    item = _field(
        name="item",
        field_type=FieldType.OBJECT,
        properties={"id": item_id},
    )
    results = _field(name="results", field_type=FieldType.ARRAY, items_schema=item)
    s = _schema(fields={"results": results})
    assert s.get_field("results.0.id") is item_id
    assert s.get_field("results.0") is item


def test_get_field_empty_path_segment():
    s = _schema()
    assert s.get_field("") is None


def test_get_field_deeply_nested():
    leaf = _field(name="zip", field_type=FieldType.STRING)
    loc = _field(name="location", field_type=FieldType.OBJECT, properties={"zip": leaf})
    addr = _field(name="address", field_type=FieldType.OBJECT, properties={"location": loc})
    s = _schema(fields={"address": addr})
    assert s.get_field("address.location.zip") is leaf


# ---------------------------------------------------------------------------
# ResponseSchema.get_all_extractable
# ---------------------------------------------------------------------------


def test_get_all_extractable_empty():
    s = _schema()
    assert s.get_all_extractable() == []


def test_get_all_extractable_flat():
    f1 = _field(name="id", extractable_type=ExtractableType.ID)
    f2 = _field(name="data")
    s = _schema(fields={"id": f1, "data": f2})
    result = s.get_all_extractable()
    assert len(result) == 1
    assert result[0] == ("id", f1)


def test_get_all_extractable_nested():
    inner = _field(name="user_id", extractable_type=ExtractableType.ID)
    outer = _field(name="user", field_type=FieldType.OBJECT, properties={"user_id": inner})
    s = _schema(fields={"user": outer})
    result = s.get_all_extractable()
    assert len(result) == 1
    path, field = result[0]
    assert path == "user.user_id"
    assert field is inner


def test_get_all_extractable_array_items():
    item_id = _field(name="id", extractable_type=ExtractableType.ID)
    item = _field(name="item", field_type=FieldType.OBJECT, properties={"id": item_id})
    results = _field(name="results", field_type=FieldType.ARRAY, items_schema=item)
    s = _schema(fields={"results": results})
    result = s.get_all_extractable()
    paths = [p for p, _ in result]
    assert "results.[].id" in paths


# ---------------------------------------------------------------------------
# identify_extractable_type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("id", ExtractableType.ID),
        ("user_id", ExtractableType.ID),
        ("userId", ExtractableType.ID),
        ("uuid", ExtractableType.ID),
        ("guid", ExtractableType.ID),
        ("orderId", ExtractableType.ID),
        ("name", ExtractableType.NAME),
        ("title", ExtractableType.NAME),
        ("label", ExtractableType.NAME),
        ("display_name", ExtractableType.NAME),
        ("url", ExtractableType.URL),
        ("image_url", ExtractableType.URL),
        ("permalink", ExtractableType.URL),
        ("href", ExtractableType.URL),
        ("created_at", ExtractableType.DATE),
        ("updated_at", ExtractableType.DATE),
        ("timestamp", ExtractableType.DATE),
        ("date", ExtractableType.DATE),
        ("datetime", ExtractableType.DATE),
        ("count", ExtractableType.COUNT),
        ("total", ExtractableType.COUNT),
        ("size", ExtractableType.COUNT),
        ("length", ExtractableType.COUNT),
        ("status", ExtractableType.STATUS),
        ("state", ExtractableType.STATUS),
        ("random_field", ExtractableType.NONE),
        ("description", ExtractableType.NONE),
        ("", ExtractableType.NONE),
    ],
)
def test_identify_extractable_type(name: str, expected: ExtractableType):
    assert identify_extractable_type(name) == expected


def test_identify_extractable_type_whitespace():
    assert identify_extractable_type("  user_id  ") == ExtractableType.ID


# ---------------------------------------------------------------------------
# calculate_schema_completeness
# ---------------------------------------------------------------------------


def test_completeness_empty_schema():
    s = _schema()
    assert calculate_schema_completeness(s) == 0.0


def test_completeness_fields_only():
    s = _schema(fields={"x": _field(name="x")})
    assert calculate_schema_completeness(s) == 0.3


def test_completeness_with_types():
    s = _schema(fields={"x": _field(name="x", field_type=FieldType.STRING)})
    assert calculate_schema_completeness(s) == 0.5  # 0.3 + 0.2


def test_completeness_with_descriptions():
    s = _schema(
        fields={
            "x": _field(name="x", field_type=FieldType.STRING, description="A field"),
        }
    )
    assert calculate_schema_completeness(s) == 0.7  # 0.3 + 0.2 + 0.2


def test_completeness_with_extractable():
    s = _schema(
        fields={
            "x": _field(
                name="x",
                field_type=FieldType.STRING,
                description="desc",
                extractable_type=ExtractableType.ID,
            ),
        }
    )
    assert calculate_schema_completeness(s) == 0.9  # 0.3 + 0.2 + 0.2 + 0.2


def test_completeness_full():
    s = _schema(
        fields={
            "x": _field(
                name="x",
                field_type=FieldType.STRING,
                description="desc",
                extractable_type=ExtractableType.ID,
                example="abc",
            ),
        }
    )
    assert calculate_schema_completeness(s) == 1.0


# ---------------------------------------------------------------------------
# flatten_response_fields
# ---------------------------------------------------------------------------


def test_flatten_empty():
    s = _schema()
    assert flatten_response_fields(s) == {}


def test_flatten_simple():
    f = _field(name="id", field_type=FieldType.STRING)
    s = _schema(fields={"id": f})
    flat = flatten_response_fields(s)
    assert "id" in flat
    assert flat["id"] is f


def test_flatten_nested():
    city = _field(name="city", field_type=FieldType.STRING)
    address = _field(name="address", field_type=FieldType.OBJECT, properties={"city": city})
    s = _schema(fields={"address": address})
    flat = flatten_response_fields(s)
    assert "address" in flat
    assert "address.city" in flat
    assert flat["address.city"] is city


def test_flatten_array():
    item_id = _field(name="id", field_type=FieldType.STRING)
    item = _field(name="item", field_type=FieldType.OBJECT, properties={"id": item_id})
    results = _field(name="results", field_type=FieldType.ARRAY, items_schema=item)
    s = _schema(fields={"results": results})
    flat = flatten_response_fields(s)
    assert "results" in flat
    assert "results.[]" in flat
    assert "results.[].id" in flat


def test_flatten_top_level_array():
    item = _field(
        name="item",
        field_type=FieldType.OBJECT,
        properties={
            "id": _field(name="id", field_type=FieldType.STRING),
        },
    )
    s = _schema(body_type=FieldType.ARRAY, items_schema=item)
    flat = flatten_response_fields(s)
    assert "[]" in flat
    assert "[].id" in flat


# ---------------------------------------------------------------------------
# Default schemas
# ---------------------------------------------------------------------------


def test_default_list_response():
    s = DEFAULT_LIST_RESPONSE
    assert s.is_list_response is True
    assert s.has_schema is False
    assert s.schema_completeness == 0.1
    assert s.items_schema is not None
    assert "id" in s.items_schema.properties


def test_default_object_response():
    s = DEFAULT_OBJECT_RESPONSE
    assert s.is_list_response is False
    assert s.has_schema is False
    assert s.schema_completeness == 0.1
    assert "id" in s.fields
    assert "status" in s.fields
    assert s.fields["id"].extractable_type == "id"
    assert s.fields["status"].extractable_type == "status"


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


def test_response_schema_json_round_trip():
    inner = _field(name="user_id", field_type=FieldType.STRING, extractable_type=ExtractableType.ID)
    outer = _field(name="user", field_type=FieldType.OBJECT, properties={"user_id": inner})
    s = _schema(
        status_code=200,
        fields={"user": outer},
        extractable_fields=["user.user_id"],
        raw_schema={"type": "object"},
    )
    data = json.loads(s.model_dump_json())
    assert data["status_code"] == 200
    assert "user" in data["fields"]
    assert "raw_schema" not in data
    assert data["extractable_fields"] == ["user.user_id"]


def test_response_field_recursive_nesting():
    """Verify deeply nested field structures serialise correctly."""
    leaf = _field(name="zip", field_type=FieldType.STRING)
    mid = _field(name="location", field_type=FieldType.OBJECT, properties={"zip": leaf})
    top = _field(name="address", field_type=FieldType.OBJECT, properties={"location": mid})
    s = _schema(fields={"address": top})
    data = json.loads(s.model_dump_json())
    addr = data["fields"]["address"]
    assert addr["properties"]["location"]["properties"]["zip"]["field_type"] == "string"
