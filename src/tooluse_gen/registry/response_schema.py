"""Rich response schema models for API endpoints.

Response schemas in ToolBench are frequently missing or incomplete.
These models handle partial data gracefully and identify *extractable*
fields — values (IDs, names, URLs …) that downstream tool calls can
reference for realistic chaining.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FieldType(str, Enum):
    """Types of fields in a response schema."""

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"
    ANY = "any"


class ExtractableType(str, Enum):
    """Types of extractable values for grounding.

    These values can be referenced in subsequent tool calls to build
    realistic multi-step chains.
    """

    ID = "id"
    NAME = "name"
    URL = "url"
    DATE = "date"
    COUNT = "count"
    STATUS = "status"
    REFERENCE = "reference"
    NONE = "none"


# ---------------------------------------------------------------------------
# ResponseField
# ---------------------------------------------------------------------------


class ResponseField(BaseModel):
    """A single field within an API response body.

    Supports nested structures (objects with ``properties``, arrays with
    ``items_schema``) and tracks whether the field carries an
    *extractable* value useful for chaining.
    """

    model_config = ConfigDict(use_enum_values=True)

    name: str = Field(..., description="Field name.")
    field_type: FieldType = Field(
        default=FieldType.ANY,
        description="Data type of this field.",
    )
    description: str = Field(default="", description="Human-readable description.")
    nullable: bool = Field(default=True, description="Whether the field may be null.")

    # Nested type info
    items_type: FieldType | None = Field(
        default=None,
        description="Element type for array fields.",
    )
    items_schema: ResponseField | None = Field(
        default=None,
        description="Full schema for array elements.",
    )
    properties: dict[str, ResponseField] = Field(
        default_factory=dict,
        description="Child fields for object types.",
    )

    # Extractable metadata
    extractable_type: ExtractableType = Field(
        default=ExtractableType.NONE,
        description="What kind of extractable value this field carries.",
    )
    extractable_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence that the extractable classification is correct.",
    )
    example: Any | None = Field(default=None, description="Example value.")

    @field_validator("description")
    @classmethod
    def strip_description(cls, v: str) -> str:
        return v.strip()

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def is_extractable(self) -> bool:
        """True when this field carries an extractable value."""
        return self.extractable_type != ExtractableType.NONE

    @property
    def is_collection(self) -> bool:
        """True when this field is an array."""
        return self.field_type == FieldType.ARRAY

    @property
    def is_nested(self) -> bool:
        """True when this field is an object with child properties."""
        return self.field_type == FieldType.OBJECT and bool(self.properties)


# ---------------------------------------------------------------------------
# ResponseSchema
# ---------------------------------------------------------------------------


class ResponseSchema(BaseModel):
    """Full response schema for an API endpoint.

    When ToolBench provides no schema, ``has_schema`` is ``False`` and
    default placeholder fields are used instead.
    """

    model_config = ConfigDict(use_enum_values=True)

    status_code: int = Field(default=200, description="HTTP status code.")
    content_type: str = Field(
        default="application/json",
        description="Response content type.",
    )
    body_type: FieldType = Field(
        default=FieldType.OBJECT,
        description="Top-level body type.",
    )
    fields: dict[str, ResponseField] = Field(
        default_factory=dict,
        description="Top-level response fields (for object bodies).",
    )
    items_schema: ResponseField | None = Field(
        default=None,
        description="Schema for array elements (for array bodies).",
    )
    extractable_fields: list[str] = Field(
        default_factory=list,
        description="Names of fields whose values are extractable.",
    )
    has_schema: bool = Field(
        default=True,
        description="True when a schema was provided, not inferred.",
    )
    schema_completeness: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How complete the schema definition is.",
    )

    # Raw source — excluded from serialisation
    raw_schema: dict[str, Any] | None = Field(
        default=None,
        exclude=True,
        description="Original ToolBench response schema (not serialised).",
    )

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def is_list_response(self) -> bool:
        """True when the top-level body is an array."""
        return self.body_type == FieldType.ARRAY

    @property
    def has_extractable_fields(self) -> bool:
        """True when at least one extractable field is declared."""
        return bool(self.extractable_fields)

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get_field(self, path: str) -> ResponseField | None:
        """Retrieve a field by dot-notation *path*.

        Supports:
        - Simple names: ``"id"``
        - Nested paths: ``"user.name"``
        - Array indices (navigates into ``items_schema``): ``"results.0.id"``

        Returns ``None`` when any segment is unresolvable.
        """
        parts = path.split(".")
        return _resolve_path(parts, self.fields, self.items_schema)

    def get_all_extractable(self) -> list[tuple[str, ResponseField]]:
        """Return ``(dot_path, field)`` tuples for every extractable field."""
        results: list[tuple[str, ResponseField]] = []
        _collect_extractable("", self.fields, self.items_schema, results)
        return results


# ---------------------------------------------------------------------------
# Private traversal helpers
# ---------------------------------------------------------------------------


def _resolve_path(
    parts: list[str],
    fields: dict[str, ResponseField],
    items_schema: ResponseField | None,
) -> ResponseField | None:
    """Walk *parts* through nested fields/items_schema."""
    if not parts:
        return None

    head, rest = parts[0], parts[1:]
    # Numeric index → dive into items_schema
    if head.isdigit():
        if items_schema is None:
            return None
        if not rest:
            return items_schema
        return _resolve_path(rest, items_schema.properties, items_schema.items_schema)

    field = fields.get(head)
    if field is None:
        return None
    if not rest:
        return field
    return _resolve_path(rest, field.properties, field.items_schema)


def _collect_extractable(
    prefix: str,
    fields: dict[str, ResponseField],
    items_schema: ResponseField | None,
    out: list[tuple[str, ResponseField]],
) -> None:
    """Recursively collect extractable fields with their dot-paths."""
    for name, field in fields.items():
        path = f"{prefix}{name}" if not prefix else f"{prefix}.{name}"
        if field.is_extractable:
            out.append((path, field))
        # Recurse into nested objects
        if field.properties:
            _collect_extractable(path, field.properties, field.items_schema, out)
        # Recurse into array element schema
        if field.items_schema is not None:
            item_prefix = f"{path}.[]"
            if field.items_schema.is_extractable:
                out.append((item_prefix, field.items_schema))
            if field.items_schema.properties:
                _collect_extractable(
                    item_prefix, field.items_schema.properties, field.items_schema.items_schema, out
                )

    # Top-level array body
    if items_schema is not None and not fields:
        item_prefix = f"{prefix}.[]" if prefix else "[]"
        if items_schema.is_extractable:
            out.append((item_prefix, items_schema))
        if items_schema.properties:
            _collect_extractable(
                item_prefix, items_schema.properties, items_schema.items_schema, out
            )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

# Patterns for heuristic extractable-type detection
_ID_PATTERNS = {"id", "uuid", "guid"}
_ID_SUFFIXES = ("_id", "id")
_NAME_PATTERNS = {"name", "title", "label"}
_NAME_SUFFIXES = ("_name",)
_URL_SUBSTRINGS = ("url", "link", "href")
_DATE_SUBSTRINGS = ("date", "time")
_DATE_PREFIXES = ("created", "updated")
_COUNT_PATTERNS = {"count", "total", "size", "length"}
_STATUS_PATTERNS = {"status", "state"}


def identify_extractable_type(
    field_name: str,
    field_type: FieldType = FieldType.ANY,
) -> ExtractableType:
    """Heuristically identify the extractable type from a field name.

    Rules (checked in order):
    - ``*_id``, ``*Id``, ``id``, ``uuid``, ``guid`` → ID
    - ``*_name``, ``name``, ``title``, ``label`` → NAME
    - ``*url*``, ``*link*``, ``*href*`` → URL
    - ``*date*``, ``*time*``, ``created*``, ``updated*`` → DATE
    - ``count``, ``total``, ``size``, ``length`` → COUNT
    - ``status``, ``state`` → STATUS

    Returns :attr:`ExtractableType.NONE` when no pattern matches.
    """
    lower = field_name.lower().strip()
    if not lower:
        return ExtractableType.NONE

    # ID
    if lower in _ID_PATTERNS or lower.endswith(_ID_SUFFIXES):
        return ExtractableType.ID

    # NAME
    if lower in _NAME_PATTERNS or lower.endswith(_NAME_SUFFIXES):
        return ExtractableType.NAME

    # URL
    if any(sub in lower for sub in _URL_SUBSTRINGS):
        return ExtractableType.URL

    # DATE
    if any(sub in lower for sub in _DATE_SUBSTRINGS) or lower.startswith(_DATE_PREFIXES):
        return ExtractableType.DATE

    # COUNT
    if lower in _COUNT_PATTERNS:
        return ExtractableType.COUNT

    # STATUS
    if lower in _STATUS_PATTERNS:
        return ExtractableType.STATUS

    return ExtractableType.NONE


def calculate_schema_completeness(schema: ResponseSchema) -> float:
    """Calculate a completeness score for *schema*.

    Factors (weighted):
    - Has any fields defined: +0.3
    - Has field types (not all ``ANY``): +0.2
    - Has field descriptions: +0.2
    - Has extractable fields: +0.2
    - Has example values: +0.1
    """
    score = 0.0
    all_fields = _all_leaf_fields(schema)

    if all_fields:
        score += 0.3
        if any(f.field_type != FieldType.ANY for f in all_fields):
            score += 0.2
        if any(f.description for f in all_fields):
            score += 0.2
        if any(f.is_extractable for f in all_fields):
            score += 0.2
        if any(f.example is not None for f in all_fields):
            score += 0.1

    return round(score, 2)


def _all_leaf_fields(schema: ResponseSchema) -> list[ResponseField]:
    """Collect all ResponseField instances from *schema* (flat list)."""
    result: list[ResponseField] = []
    _gather_fields(schema.fields, schema.items_schema, result)
    return result


def _gather_fields(
    fields: dict[str, ResponseField],
    items_schema: ResponseField | None,
    out: list[ResponseField],
) -> None:
    for field in fields.values():
        out.append(field)
        if field.properties:
            _gather_fields(field.properties, field.items_schema, out)
        if field.items_schema is not None:
            out.append(field.items_schema)
            if field.items_schema.properties:
                _gather_fields(field.items_schema.properties, field.items_schema.items_schema, out)
    if items_schema is not None:
        out.append(items_schema)
        if items_schema.properties:
            _gather_fields(items_schema.properties, items_schema.items_schema, out)


def flatten_response_fields(schema: ResponseSchema) -> dict[str, ResponseField]:
    """Flatten a nested response schema to dot-notation paths.

    Example::

        {"user": ResponseField(properties={"id": ..., "name": ...})}
        → {"user.id": ..., "user.name": ...}

    Top-level fields without nesting are included as-is.
    """
    result: dict[str, ResponseField] = {}
    _flatten("", schema.fields, schema.items_schema, result)
    return result


def _flatten(
    prefix: str,
    fields: dict[str, ResponseField],
    items_schema: ResponseField | None,
    out: dict[str, ResponseField],
) -> None:
    for name, field in fields.items():
        path = f"{prefix}{name}" if not prefix else f"{prefix}.{name}"
        # Leaf or scalar → add directly
        if not field.properties and field.items_schema is None:
            out[path] = field
        else:
            # Still include the parent node
            out[path] = field
            if field.properties:
                _flatten(path, field.properties, field.items_schema, out)
            if field.items_schema is not None:
                item_prefix = f"{path}.[]"
                out[item_prefix] = field.items_schema
                if field.items_schema.properties:
                    _flatten(
                        item_prefix,
                        field.items_schema.properties,
                        field.items_schema.items_schema,
                        out,
                    )

    if items_schema is not None and not fields:
        item_prefix = f"{prefix}.[]" if prefix else "[]"
        out[item_prefix] = items_schema
        if items_schema.properties:
            _flatten(item_prefix, items_schema.properties, items_schema.items_schema, out)


# ---------------------------------------------------------------------------
# Default schemas (for endpoints with no schema provided)
# ---------------------------------------------------------------------------

DEFAULT_LIST_RESPONSE = ResponseSchema(
    body_type=FieldType.ARRAY,
    items_schema=ResponseField(
        name="item",
        field_type=FieldType.OBJECT,
        properties={
            "id": ResponseField(
                name="id",
                field_type=FieldType.STRING,
                extractable_type=ExtractableType.ID,
                extractable_confidence=0.5,
            ),
        },
    ),
    has_schema=False,
    schema_completeness=0.1,
)

DEFAULT_OBJECT_RESPONSE = ResponseSchema(
    body_type=FieldType.OBJECT,
    fields={
        "id": ResponseField(
            name="id",
            field_type=FieldType.STRING,
            extractable_type=ExtractableType.ID,
            extractable_confidence=0.5,
        ),
        "status": ResponseField(
            name="status",
            field_type=FieldType.STRING,
            extractable_type=ExtractableType.STATUS,
            extractable_confidence=0.5,
        ),
    },
    has_schema=False,
    schema_completeness=0.1,
)
