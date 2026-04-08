"""Tool Registry package — models, loaders, and graph construction."""

from tooluse_gen.registry.completeness import (
    DEFAULT_WEIGHTS,
    CompletenessCalculator,
    CompletenessWeights,
    QualityTier,
    ScoreBreakdown,
    count_documented_params,
    filter_by_quality,
    generate_quality_report,
    get_quality_tier,
    get_score_breakdown,
    is_explicit_type,
    is_meaningful_description,
)
from tooluse_gen.registry.loader import (
    InvalidToolError,
    LoaderConfig,
    LoaderStats,
    MissingRequiredFieldError,
    ParameterTypeInferrer,
    RawToolParser,
    ToolBenchLoader,
    ToolBenchLoaderError,
    ToolNormalizer,
)
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
from tooluse_gen.registry.normalizers import (
    FieldNormalizer,
    NormalizationStats,
    PathNormalizer,
    TextNormalizer,
    TypeNormalizer,
    ValueNormalizer,
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
from tooluse_gen.registry.type_inference import (
    NAME_RULES,
    InferenceResult,
    InferenceRule,
    TypeEvidence,
    infer_endpoint_parameter_types,
)
from tooluse_gen.registry.type_inference import (
    ParameterTypeInferrer as HeuristicTypeInferrer,
)

__all__ = [
    # Enums
    "ParameterType",
    "ParameterLocation",
    "HttpMethod",
    "FieldType",
    "ExtractableType",
    "QualityTier",
    # Models
    "Parameter",
    "Endpoint",
    "Tool",
    # Response schema models
    "ResponseField",
    "ResponseSchema",
    # Completeness
    "CompletenessWeights",
    "DEFAULT_WEIGHTS",
    "CompletenessCalculator",
    "ScoreBreakdown",
    # Loader
    "LoaderConfig",
    "LoaderStats",
    "RawToolParser",
    "ToolNormalizer",
    "ParameterTypeInferrer",
    "ToolBenchLoader",
    "ToolBenchLoaderError",
    "InvalidToolError",
    "MissingRequiredFieldError",
    # Normalizers
    "TextNormalizer",
    "TypeNormalizer",
    "PathNormalizer",
    "ValueNormalizer",
    "FieldNormalizer",
    "NormalizationStats",
    # Type inference
    "InferenceRule",
    "InferenceResult",
    "TypeEvidence",
    "NAME_RULES",
    "HeuristicTypeInferrer",
    "infer_endpoint_parameter_types",
    # Default schemas
    "DEFAULT_LIST_RESPONSE",
    "DEFAULT_OBJECT_RESPONSE",
    # Helpers
    "generate_endpoint_id",
    "normalize_parameter_name",
    "identify_extractable_type",
    "calculate_schema_completeness",
    "flatten_response_fields",
    "get_quality_tier",
    "filter_by_quality",
    "get_score_breakdown",
    "generate_quality_report",
    "is_meaningful_description",
    "is_explicit_type",
    "count_documented_params",
]
