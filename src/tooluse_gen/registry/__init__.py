"""Tool Registry package — models, loaders, and graph construction."""

from tooluse_gen.registry.models import (
    Endpoint,
    HttpMethod,
    Parameter,
    ParameterLocation,
    ParameterType,
    Tool,
    generate_endpoint_id,
    normalize_parameter_name,
)
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

__all__ = [
    # Enums
    "ParameterType",
    "ParameterLocation",
    "HttpMethod",
    "FieldType",
    "ExtractableType",
    # Models (legacy)
    "Parameter",
    "Endpoint",
    "Tool",
    # Response schema models
    "ResponseField",
    "ResponseSchema",
    # Default schemas
    "DEFAULT_LIST_RESPONSE",
    "DEFAULT_OBJECT_RESPONSE",
    # Helpers
    "generate_endpoint_id",
    "normalize_parameter_name",
    "identify_extractable_type",
    "calculate_schema_completeness",
    "flatten_response_fields",
]
