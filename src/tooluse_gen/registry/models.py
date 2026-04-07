"""Pydantic data models for the Tool Registry.

Hierarchy:
    Tool
    └── Endpoint (one-to-many)
        └── Parameter (one-to-many)

All models use ``use_enum_values=True`` so that serialized JSON contains
plain strings rather than enum wrapper objects.  Fields that carry raw
ToolBench source data are excluded from serialisation to avoid bloat.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ParameterType(str, Enum):
    """Normalised parameter types across ToolBench schemas."""

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    FILE = "file"
    DATE = "date"
    DATETIME = "datetime"
    UNKNOWN = "unknown"


class ParameterLocation(str, Enum):
    """Where the parameter appears in the HTTP request."""

    QUERY = "query"
    PATH = "path"
    HEADER = "header"
    BODY = "body"
    FORM = "form"


class HttpMethod(str, Enum):
    """Supported HTTP methods."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


# ---------------------------------------------------------------------------
# ResponseSchema (referenced by Endpoint)
# ---------------------------------------------------------------------------


class ResponseSchema(BaseModel):
    """Lightweight representation of an endpoint's response structure."""

    model_config = ConfigDict(use_enum_values=True)

    status_code: int = 200
    description: str = ""
    schema_type: ParameterType = ParameterType.OBJECT
    properties: dict[str, Any] = Field(default_factory=dict)
    example: Any | None = None


# ---------------------------------------------------------------------------
# Parameter
# ---------------------------------------------------------------------------


class Parameter(BaseModel):
    """A single parameter in an API endpoint.

    Quality indicators (``has_description``, ``has_type``,
    ``inferred_type``) are computed from the raw definition so callers can
    filter or score endpoints by data completeness.
    """

    model_config = ConfigDict(use_enum_values=True)

    # Core fields
    name: str = Field(..., description="Parameter name (required).")
    description: str = Field(default="", description="Human-readable description.")
    param_type: ParameterType = Field(
        default=ParameterType.STRING,
        description="Normalised parameter type.",
    )
    location: ParameterLocation = Field(
        default=ParameterLocation.QUERY,
        description="Where the parameter appears in the request.",
    )
    required: bool = Field(default=False, description="Whether the parameter is required.")
    default: Any | None = Field(default=None, description="Default value if not provided.")
    enum_values: list[str] | None = Field(
        default=None, description="Allowed values when the parameter is an enum."
    )

    # Nested type info
    items_type: ParameterType | None = Field(
        default=None,
        description="Element type for array parameters.",
    )
    properties: dict[str, Parameter] | None = Field(
        default=None,
        description="Child parameters for object types.",
    )

    # Raw source — excluded from serialisation
    raw_definition: dict[str, Any] | None = Field(
        default=None,
        exclude=True,
        description="Original ToolBench parameter definition (not serialised).",
    )

    # Quality indicators
    has_description: bool = Field(
        default=False,
        description="True when a non-empty description was present in the source.",
    )
    has_type: bool = Field(
        default=False,
        description="True when an explicit type was present in the source.",
    )
    inferred_type: bool = Field(
        default=False,
        description="True when the type was inferred rather than declared.",
    )

    @field_validator("name")
    @classmethod
    def normalise_name(cls, v: str) -> str:
        return normalize_parameter_name(v)

    @field_validator("description")
    @classmethod
    def strip_description(cls, v: str) -> str:
        return v.strip()


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


class Endpoint(BaseModel):
    """A single API endpoint belonging to a :class:`Tool`.

    ``completeness_score`` is a float in ``[0, 1]`` summarising how much
    useful metadata the endpoint carries (description, typed parameters,
    response schema, etc.).
    """

    model_config = ConfigDict(use_enum_values=True)

    endpoint_id: str = Field(..., description="Unique identifier: {tool_id}/{METHOD}/{hash}.")
    tool_id: str = Field(..., description="Parent tool identifier.")
    name: str = Field(..., description="Human-readable endpoint name.")
    description: str = Field(default="", description="Endpoint description.")
    method: HttpMethod = Field(default=HttpMethod.GET, description="HTTP method.")
    path: str = Field(..., description="URL path template.")
    parameters: list[Parameter] = Field(default_factory=list)
    required_parameters: list[str] = Field(
        default_factory=list,
        description="Names of required parameters.",
    )
    response_schema: ResponseSchema | None = Field(
        default=None,
        description="Expected response structure.",
    )
    category: str = Field(
        default="",
        description="Semantic category, e.g. 'search', 'create', 'update', 'delete'.",
    )
    tags: list[str] = Field(default_factory=list)
    completeness_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Data quality score in [0, 1].",
    )

    # Raw source — excluded from serialisation
    raw_definition: dict[str, Any] | None = Field(
        default=None,
        exclude=True,
        description="Original ToolBench endpoint definition (not serialised).",
    )

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def has_required_params(self) -> bool:
        """True when at least one required parameter exists."""
        return bool(self.required_parameters)

    @property
    def param_count(self) -> int:
        """Total number of parameters."""
        return len(self.parameters)

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get_parameter(self, name: str) -> Parameter | None:
        """Return the parameter with *name*, or ``None`` if not found."""
        for param in self.parameters:
            if param.name == name:
                return param
        return None


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class Tool(BaseModel):
    """Top-level representation of an API tool from ToolBench.

    A tool groups one or more :class:`Endpoint` objects under a shared
    base URL and provides metadata (domain, auth type, etc.) used for
    sampling and diversity steering during generation.
    """

    model_config = ConfigDict(use_enum_values=True)

    tool_id: str = Field(..., description="Unique tool identifier.")
    name: str = Field(..., description="Human-readable tool name.")
    description: str = Field(default="", description="Tool description.")
    domain: str = Field(default="", description="Primary domain/category, e.g. 'Finance'.")
    categories: list[str] = Field(default_factory=list, description="All applicable categories.")
    base_url: str = Field(default="", description="API base URL.")
    api_version: str = Field(default="", description="API version string.")
    auth_type: str = Field(
        default="none",
        description="Authentication type: 'apikey', 'oauth', 'none', etc.",
    )
    endpoints: list[Endpoint] = Field(default_factory=list)
    completeness_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Aggregate data quality score in [0, 1].",
    )
    created_at: datetime | None = Field(default=None)
    updated_at: datetime | None = Field(default=None)

    # Raw source — excluded from serialisation
    raw_schema: dict[str, Any] | None = Field(
        default=None,
        exclude=True,
        description="Original ToolBench schema (not serialised).",
    )
    source_file: str | None = Field(
        default=None,
        description="Path to the source JSON file.",
    )

    # ------------------------------------------------------------------
    # Computed fields
    # ------------------------------------------------------------------

    @computed_field
    @property
    def endpoint_count(self) -> int:
        """Number of endpoints attached to this tool."""
        return len(self.endpoints)

    # ------------------------------------------------------------------
    # Computed properties (non-serialised)
    # ------------------------------------------------------------------

    @property
    def is_complete(self) -> bool:
        """True when ``completeness_score >= 0.5``."""
        return self.completeness_score >= 0.5

    @property
    def endpoint_ids(self) -> list[str]:
        """Ordered list of all endpoint IDs."""
        return [ep.endpoint_id for ep in self.endpoints]

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get_endpoint(self, endpoint_id: str) -> Endpoint | None:
        """Return the endpoint with *endpoint_id*, or ``None``."""
        for ep in self.endpoints:
            if ep.endpoint_id == endpoint_id:
                return ep
        return None

    def get_endpoints_by_method(self, method: HttpMethod) -> list[Endpoint]:
        """Return all endpoints whose HTTP method matches *method*."""
        target = method.value if isinstance(method, HttpMethod) else str(method).upper()
        return [ep for ep in self.endpoints if ep.method == target]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

# Regex matching URL path parameters: {id}, {user_id}, :id, <id>
_PATH_PARAM_RE = re.compile(r"\{[^}]+\}|:[a-zA-Z_]\w*|<[^>]+>")

# Characters that are not safe in parameter names
_UNSAFE_CHARS_RE = re.compile(r"[^\w]")


def generate_endpoint_id(tool_id: str, method: str, path: str) -> str:
    """Generate a stable, unique endpoint identifier.

    Algorithm:
    1. Normalise the path by replacing all path-parameter placeholders
       (``{id}``, ``:id``, ``<id>``) with the literal ``_param_``.
    2. SHA-256 hash the normalised ``METHOD:path`` string.
    3. Return ``{tool_id}/{METHOD}/{hash[:8]}``.

    Args:
        tool_id: The parent tool identifier.
        method:  HTTP method string (case-insensitive).
        path:    URL path template, e.g. ``/users/{id}/posts``.

    Returns:
        A stable string suitable for use as a dict key or graph node ID.

    Example::

        generate_endpoint_id("rapidapi.weather", "get", "/forecast/{city}")
        # → "rapidapi.weather/GET/a3f2c1b0"
    """
    normalised_path = _PATH_PARAM_RE.sub("_param_", path)
    method_upper = method.upper()
    raw = f"{method_upper}:{normalised_path}"
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return f"{tool_id}/{method_upper}/{digest[:8]}"


def normalize_parameter_name(name: str) -> str:
    """Return a clean, consistently-cased parameter name.

    Steps:
    1. Strip leading/trailing whitespace.
    2. Replace runs of unsafe characters (anything non-alphanumeric or
       non-underscore) with a single underscore.
    3. Strip leading/trailing underscores from the result.
    4. Lower-case the whole name.

    Args:
        name: Raw parameter name from the ToolBench definition.

    Returns:
        Normalised parameter name.  Falls back to ``"unknown"`` for
        names that become empty after normalisation.

    Example::

        normalize_parameter_name("  Content-Type  ")  # → "content_type"
        normalize_parameter_name("$filter")           # → "filter"
    """
    stripped = name.strip()
    normalised = _UNSAFE_CHARS_RE.sub("_", stripped)
    normalised = normalised.strip("_").lower()
    return normalised or "unknown"
