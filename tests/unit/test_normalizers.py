"""Unit tests for field normalisation utilities (Task 12)."""

from __future__ import annotations

import pytest

from tooluse_gen.registry.models import (
    HttpMethod,
    ParameterLocation,
    ParameterType,
)
from tooluse_gen.registry.normalizers import (
    FieldNormalizer,
    NormalizationStats,
    PathNormalizer,
    TextNormalizer,
    TypeNormalizer,
    ValueNormalizer,
)

pytestmark = pytest.mark.unit


# =========================================================================
# TextNormalizer
# =========================================================================


class TestTextNormalizeName:
    def test_strips_whitespace(self):
        assert TextNormalizer.normalize_name("  Hello  ") == "Hello"

    def test_removes_invalid_chars(self):
        assert TextNormalizer.normalize_name("Hello@World!") == "HelloWorld"

    def test_collapses_whitespace(self):
        assert TextNormalizer.normalize_name("Hello   World") == "Hello World"

    def test_truncates_at_200(self):
        result = TextNormalizer.normalize_name("A" * 250)
        assert len(result) == 200

    def test_none_returns_empty(self):
        assert TextNormalizer.normalize_name(None) == ""

    def test_empty_returns_empty(self):
        assert TextNormalizer.normalize_name("") == ""

    def test_keeps_hyphens_underscores(self):
        assert TextNormalizer.normalize_name("my-tool_v2") == "my-tool_v2"

    def test_unicode_preserved(self):
        assert TextNormalizer.normalize_name("café") == "café"


class TestTextNormalizeDescription:
    def test_strips_whitespace(self):
        assert TextNormalizer.normalize_description("  hello  ") == "hello"

    def test_collapses_whitespace(self):
        assert TextNormalizer.normalize_description("a   b   c") == "a b c"

    def test_none_returns_empty(self):
        assert TextNormalizer.normalize_description(None) == ""

    def test_empty_returns_empty(self):
        assert TextNormalizer.normalize_description("") == ""

    def test_truncates_at_2000(self):
        result = TextNormalizer.normalize_description("A" * 2500)
        assert len(result) == 2000

    def test_removes_name_echo(self):
        assert TextNormalizer.normalize_description("Search", name="Search") == ""

    def test_removes_trivial_prefix_echo(self):
        assert TextNormalizer.normalize_description("The Search", name="Search") == ""
        assert TextNormalizer.normalize_description("A Search", name="Search") == ""

    def test_keeps_meaningful_description(self):
        result = TextNormalizer.normalize_description("Search for items by keyword", name="Search")
        assert result == "Search for items by keyword"

    def test_decodes_html_entities(self):
        assert TextNormalizer.normalize_description("a &amp; b") == "a & b"

    def test_fixes_escaped_unicode(self):
        assert TextNormalizer.normalize_description("caf\\u00e9") == "café"


class TestTextNormalizeIdentifier:
    def test_basic(self):
        assert TextNormalizer.normalize_identifier("My Tool") == "my_tool"

    def test_strips_special_chars(self):
        assert TextNormalizer.normalize_identifier("tool@v2!") == "toolv2"

    def test_lowercase(self):
        assert TextNormalizer.normalize_identifier("MyTool") == "mytool"

    def test_none_empty(self):
        assert TextNormalizer.normalize_identifier(None) == ""

    def test_fallback_source(self):
        assert TextNormalizer.normalize_identifier(None, fallback_source="My Tool") == "my_tool"

    def test_keeps_hyphens_underscores(self):
        assert TextNormalizer.normalize_identifier("my-tool_v2") == "my-tool_v2"

    def test_strips_leading_trailing_separators(self):
        assert TextNormalizer.normalize_identifier("__my_tool__") == "my_tool"


class TestTextFixEncoding:
    def test_mojibake_repair(self):
        # "é" encoded as UTF-8 bytes then decoded as Latin-1 gives "Ã©"
        broken = "é".encode().decode("latin-1")
        assert TextNormalizer.fix_encoding(broken) == "é"

    def test_escaped_unicode(self):
        assert TextNormalizer.fix_encoding("caf\\u00e9") == "café"

    def test_html_entities(self):
        assert TextNormalizer.fix_encoding("a &amp; b") == "a & b"

    def test_clean_text_unchanged(self):
        assert TextNormalizer.fix_encoding("hello world") == "hello world"


# =========================================================================
# TypeNormalizer
# =========================================================================


class TestTypeNormalizerParameterType:
    @pytest.mark.parametrize(
        ("raw", "expected_type"),
        [
            ("string", ParameterType.STRING),
            ("str", ParameterType.STRING),
            ("text", ParameterType.STRING),
            ("integer", ParameterType.INTEGER),
            ("int", ParameterType.INTEGER),
            ("int32", ParameterType.INTEGER),
            ("int64", ParameterType.INTEGER),
            ("long", ParameterType.INTEGER),
            ("number", ParameterType.NUMBER),
            ("float", ParameterType.NUMBER),
            ("double", ParameterType.NUMBER),
            ("boolean", ParameterType.BOOLEAN),
            ("bool", ParameterType.BOOLEAN),
            ("array", ParameterType.ARRAY),
            ("list", ParameterType.ARRAY),
            ("object", ParameterType.OBJECT),
            ("dict", ParameterType.OBJECT),
            ("json", ParameterType.OBJECT),
            ("file", ParameterType.FILE),
            ("binary", ParameterType.FILE),
            ("date", ParameterType.DATE),
            ("datetime", ParameterType.DATETIME),
            ("date-time", ParameterType.DATETIME),
        ],
    )
    def test_known_types(self, raw: str, expected_type: ParameterType):
        pt, explicit = TypeNormalizer.normalize_parameter_type(raw)
        assert pt == expected_type
        assert explicit is True

    def test_case_insensitive(self):
        pt, _ = TypeNormalizer.normalize_parameter_type("STRING")
        assert pt == ParameterType.STRING

    def test_with_whitespace(self):
        pt, _ = TypeNormalizer.normalize_parameter_type("  integer  ")
        assert pt == ParameterType.INTEGER

    def test_none(self):
        pt, explicit = TypeNormalizer.normalize_parameter_type(None)
        assert pt == ParameterType.UNKNOWN
        assert explicit is False

    def test_empty(self):
        pt, explicit = TypeNormalizer.normalize_parameter_type("")
        assert pt == ParameterType.UNKNOWN
        assert explicit is False

    def test_unknown(self):
        pt, explicit = TypeNormalizer.normalize_parameter_type("custom_type")
        assert pt == ParameterType.UNKNOWN
        assert explicit is False


class TestTypeNormalizerHttpMethod:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("get", HttpMethod.GET),
            ("GET", HttpMethod.GET),
            ("post", HttpMethod.POST),
            ("PUT", HttpMethod.PUT),
            ("delete", HttpMethod.DELETE),
            ("patch", HttpMethod.PATCH),
            ("head", HttpMethod.HEAD),
            ("options", HttpMethod.OPTIONS),
        ],
    )
    def test_known_methods(self, raw: str, expected: HttpMethod):
        assert TypeNormalizer.normalize_http_method(raw) == expected

    def test_none_defaults_get(self):
        assert TypeNormalizer.normalize_http_method(None) == HttpMethod.GET

    def test_empty_defaults_get(self):
        assert TypeNormalizer.normalize_http_method("") == HttpMethod.GET

    def test_unknown_defaults_get(self):
        assert TypeNormalizer.normalize_http_method("FOOBAR") == HttpMethod.GET

    def test_whitespace(self):
        assert TypeNormalizer.normalize_http_method("  post  ") == HttpMethod.POST


class TestTypeNormalizerLocation:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("query", ParameterLocation.QUERY),
            ("path", ParameterLocation.PATH),
            ("header", ParameterLocation.HEADER),
            ("body", ParameterLocation.BODY),
            ("formData", ParameterLocation.FORM),
            ("form", ParameterLocation.FORM),
        ],
    )
    def test_known_locations(self, raw: str, expected: ParameterLocation):
        assert TypeNormalizer.normalize_location(raw) == expected

    def test_none_defaults_query(self):
        assert TypeNormalizer.normalize_location(None) == ParameterLocation.QUERY

    def test_unknown_defaults_query(self):
        assert TypeNormalizer.normalize_location("cookie") == ParameterLocation.QUERY


# =========================================================================
# PathNormalizer
# =========================================================================


class TestPathNormalizerBaseUrl:
    def test_basic(self):
        assert (
            PathNormalizer.normalize_base_url("https://api.example.com/")
            == "https://api.example.com"
        )

    def test_adds_https(self):
        assert PathNormalizer.normalize_base_url("api.example.com") == "https://api.example.com"

    def test_protocol_relative(self):
        assert PathNormalizer.normalize_base_url("//api.example.com") == "https://api.example.com"

    def test_strips_trailing_slashes(self):
        assert (
            PathNormalizer.normalize_base_url("https://api.example.com///")
            == "https://api.example.com"
        )

    def test_none_returns_empty(self):
        assert PathNormalizer.normalize_base_url(None) == ""

    def test_empty_returns_empty(self):
        assert PathNormalizer.normalize_base_url("") == ""

    def test_preserves_http(self):
        assert (
            PathNormalizer.normalize_base_url("http://api.example.com") == "http://api.example.com"
        )


class TestPathNormalizerEndpointPath:
    def test_basic(self):
        assert PathNormalizer.normalize_endpoint_path("/users") == "/users"

    def test_adds_leading_slash(self):
        assert PathNormalizer.normalize_endpoint_path("users") == "/users"

    def test_removes_trailing_slash(self):
        assert PathNormalizer.normalize_endpoint_path("/users/") == "/users"

    def test_root_preserved(self):
        assert PathNormalizer.normalize_endpoint_path("/") == "/"

    def test_none_returns_root(self):
        assert PathNormalizer.normalize_endpoint_path(None) == "/"

    def test_strips_query_string(self):
        assert PathNormalizer.normalize_endpoint_path("/search?q=test") == "/search"

    def test_unifies_colon_params(self):
        assert PathNormalizer.normalize_endpoint_path("/users/:id") == "/users/{id}"

    def test_unifies_angle_params(self):
        assert PathNormalizer.normalize_endpoint_path("/users/<id>") == "/users/{id}"

    def test_preserves_brace_params(self):
        assert PathNormalizer.normalize_endpoint_path("/users/{id}") == "/users/{id}"

    def test_decodes_percent_encoding(self):
        assert PathNormalizer.normalize_endpoint_path("/search%20items") == "/search items"

    def test_mixed_params(self):
        result = PathNormalizer.normalize_endpoint_path("/users/:user_id/posts/<post_id>")
        assert result == "/users/{user_id}/posts/{post_id}"


class TestPathNormalizerExtractParams:
    def test_brace_params(self):
        assert PathNormalizer.extract_path_parameters("/users/{user_id}/posts/{post_id}") == [
            "user_id",
            "post_id",
        ]

    def test_colon_params(self):
        assert PathNormalizer.extract_path_parameters("/users/:id") == ["id"]

    def test_angle_params(self):
        assert PathNormalizer.extract_path_parameters("/users/<id>") == ["id"]

    def test_no_params(self):
        assert PathNormalizer.extract_path_parameters("/users") == []

    def test_deduplication(self):
        assert PathNormalizer.extract_path_parameters("/users/{id}/items/{id}") == ["id"]

    def test_mixed_styles(self):
        result = PathNormalizer.extract_path_parameters("/a/{x}/b/:y/c/<z>")
        assert result == ["x", "y", "z"]


# =========================================================================
# ValueNormalizer
# =========================================================================


class TestValueNormalizerDefault:
    def test_int_coercion(self):
        assert ValueNormalizer.normalize_default_value("123", ParameterType.INTEGER) == 123

    def test_float_coercion(self):
        assert ValueNormalizer.normalize_default_value(
            "3.14", ParameterType.NUMBER
        ) == pytest.approx(3.14)

    def test_bool_true(self):
        assert ValueNormalizer.normalize_default_value("true", ParameterType.BOOLEAN) is True

    def test_bool_false(self):
        assert ValueNormalizer.normalize_default_value("false", ParameterType.BOOLEAN) is False

    def test_bool_passthrough(self):
        assert ValueNormalizer.normalize_default_value(True, ParameterType.BOOLEAN) is True

    def test_string_coercion(self):
        assert ValueNormalizer.normalize_default_value(123, ParameterType.STRING) == "123"

    def test_none_returns_none(self):
        assert ValueNormalizer.normalize_default_value(None, ParameterType.INTEGER) is None

    def test_failed_coercion_returns_none(self):
        assert ValueNormalizer.normalize_default_value("abc", ParameterType.INTEGER) is None

    def test_passthrough_for_unknown_type(self):
        assert ValueNormalizer.normalize_default_value("val", ParameterType.UNKNOWN) == "val"


class TestValueNormalizerEnum:
    def test_list(self):
        assert ValueNormalizer.normalize_enum_values(["a", "b"]) == ["a", "b"]

    def test_list_with_int(self):
        assert ValueNormalizer.normalize_enum_values([1, 2]) == ["1", "2"]

    def test_comma_string(self):
        assert ValueNormalizer.normalize_enum_values("a, b, c") == ["a", "b", "c"]

    def test_single_string(self):
        assert ValueNormalizer.normalize_enum_values("active") == ["active"]

    def test_single_scalar(self):
        assert ValueNormalizer.normalize_enum_values(42) == ["42"]

    def test_none(self):
        assert ValueNormalizer.normalize_enum_values(None) is None

    def test_empty_list(self):
        assert ValueNormalizer.normalize_enum_values([]) is None

    def test_empty_string(self):
        assert ValueNormalizer.normalize_enum_values("") is None

    def test_list_with_none(self):
        assert ValueNormalizer.normalize_enum_values(["a", None, "b"]) == ["a", "b"]


class TestValueNormalizerRequired:
    def test_bool_true(self):
        assert ValueNormalizer.normalize_required_flag(True) is True

    def test_bool_false(self):
        assert ValueNormalizer.normalize_required_flag(False) is False

    def test_string_true(self):
        assert ValueNormalizer.normalize_required_flag("true") is True

    def test_string_false(self):
        assert ValueNormalizer.normalize_required_flag("false") is False

    def test_string_yes(self):
        assert ValueNormalizer.normalize_required_flag("yes") is True

    def test_string_no(self):
        assert ValueNormalizer.normalize_required_flag("no") is False

    def test_string_optional(self):
        assert ValueNormalizer.normalize_required_flag("optional") is False

    def test_int_1(self):
        assert ValueNormalizer.normalize_required_flag(1) is True

    def test_int_0(self):
        assert ValueNormalizer.normalize_required_flag(0) is False

    def test_string_1(self):
        assert ValueNormalizer.normalize_required_flag("1") is True


# =========================================================================
# NormalizationStats
# =========================================================================


class TestNormalizationStats:
    def test_defaults(self):
        s = NormalizationStats()
        assert s.text_cleaned == 0
        assert s.encoding_fixed == 0

    def test_to_dict(self):
        s = NormalizationStats(text_cleaned=5, types_normalized=3)
        d = s.to_dict()
        assert d["text_cleaned"] == 5
        assert d["types_normalized"] == 3
        assert "values_coerced" in d


# =========================================================================
# FieldNormalizer (composite)
# =========================================================================


class TestFieldNormalizerTool:
    def test_normalize_tool_fields(self):
        fn = FieldNormalizer()
        raw = {
            "name": "  WeatherAPI  ",
            "description": "Provides weather data &amp; forecasts worldwide.",
            "category": "Weather",
            "api_host": "api.weather.com",
            "id": "weather_api",
        }
        result = fn.normalize_tool_fields(raw)
        assert result["name"] == "WeatherAPI"
        assert result["description"] == "Provides weather data & forecasts worldwide."
        assert result["domain"] == "Weather"
        assert result["base_url"] == "https://api.weather.com"
        assert result["tool_id"] == "weather_api"
        assert result["raw_schema"] is raw
        assert fn.stats.text_cleaned >= 3
        assert fn.stats.paths_normalized >= 1


class TestFieldNormalizerEndpoint:
    def test_normalize_endpoint_fields(self):
        fn = FieldNormalizer()
        raw = {
            "name": "GetForecast",
            "api_description": "Returns a detailed weather forecast report.",
            "method": "get",
            "url": "/forecast/:city?days=7",
        }
        result = fn.normalize_endpoint_fields(raw)
        assert result["name"] == "GetForecast"
        assert result["description"] == "Returns a detailed weather forecast report."
        assert result["method"] == HttpMethod.GET
        assert result["path"] == "/forecast/{city}"
        assert result["raw_definition"] is raw


class TestFieldNormalizerParameter:
    def test_normalize_parameter_fields(self):
        fn = FieldNormalizer()
        raw = {
            "name": "city",
            "description": "The target city to look up weather data for.",
            "type": "STRING",
            "required": "true",
            "default": "London",
            "in": "query",
        }
        result = fn.normalize_parameter_fields(raw)
        assert result["name"] == "city"
        assert result["description"] == "The target city to look up weather data for."
        assert result["param_type"] == ParameterType.STRING
        assert result["required"] is True
        assert result["default"] == "London"
        assert result["location"] == ParameterLocation.QUERY
        assert result["has_type"] is True
        assert result["raw_definition"] is raw

    def test_type_defaults_to_unknown(self):
        fn = FieldNormalizer()
        result = fn.normalize_parameter_fields({"name": "q"})
        assert result["param_type"] == ParameterType.UNKNOWN
        assert result["has_type"] is False
        assert fn.stats.types_defaulted >= 1


# =========================================================================
# Package-level imports
# =========================================================================


def test_package_exports():
    from tooluse_gen.registry import (
        FieldNormalizer,
        NormalizationStats,
    )

    assert FieldNormalizer is not None
    assert NormalizationStats is not None
