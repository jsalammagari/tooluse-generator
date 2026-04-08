"""Unit tests for the ToolBench JSON loader (Task 11)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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
    ParameterLocation,
    ParameterType,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures — write temp JSON files
# ---------------------------------------------------------------------------


def _write_json(tmp_path: Path, name: str, data: object) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


SINGLE_TOOL: dict = {
    "name": "WeatherAPI",
    "description": "Provides weather data for any location worldwide.",
    "category": "Weather",
    "api_host": "https://api.weather.com",
    "api_list": [
        {
            "name": "GetForecast",
            "api_description": "Returns a 7-day weather forecast for the given city.",
            "method": "GET",
            "url": "/forecast",
            "parameters": [
                {
                    "name": "city",
                    "description": "Target city name for the weather forecast lookup.",
                    "type": "string",
                    "required": True,
                },
                {
                    "name": "days",
                    "type": "integer",
                    "required": False,
                    "default": 7,
                },
            ],
            "response": {"status_code": 200, "description": "Forecast data"},
        }
    ],
}


TOOL_LIST: list = [
    {"name": "ToolA", "api_list": [{"name": "ep1", "url": "/a"}]},
    {"name": "ToolB", "api_list": [{"name": "ep2", "url": "/b"}]},
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TestExceptions:
    def test_invalid_tool_error_carries_raw_data(self):
        err = InvalidToolError("bad", raw_data={"x": 1})
        assert err.raw_data == {"x": 1}
        assert "bad" in str(err)

    def test_missing_required_field_error(self):
        err = MissingRequiredFieldError("field missing")
        assert isinstance(err, ToolBenchLoaderError)


# ---------------------------------------------------------------------------
# LoaderConfig
# ---------------------------------------------------------------------------


class TestLoaderConfig:
    def test_defaults(self):
        c = LoaderConfig()
        assert c.strict_mode is False
        assert c.infer_types is True
        assert c.min_endpoints == 1
        assert c.max_endpoints == 100
        assert c.calculate_completeness is True
        assert c.min_completeness == 0.0
        assert c.max_tools is None


# ---------------------------------------------------------------------------
# LoaderStats
# ---------------------------------------------------------------------------


class TestLoaderStats:
    def test_defaults(self):
        s = LoaderStats()
        assert s.files_processed == 0
        assert s.errors == []
        assert s.quality_distribution == {}

    def test_to_dict(self):
        s = LoaderStats(files_processed=1, tools_loaded=2)
        d = s.to_dict()
        assert d["files_processed"] == 1
        assert d["tools_loaded"] == 2
        assert "errors" in d


# ---------------------------------------------------------------------------
# ParameterTypeInferrer
# ---------------------------------------------------------------------------


class TestParameterTypeInferrer:
    def setup_method(self):
        self.inf = ParameterTypeInferrer()

    def test_infer_from_bool_default(self):
        assert self.inf.infer("flag", True) == ParameterType.BOOLEAN

    def test_infer_from_int_default(self):
        assert self.inf.infer("x", 42) == ParameterType.INTEGER

    def test_infer_from_float_default(self):
        assert self.inf.infer("x", 3.14) == ParameterType.NUMBER

    def test_infer_from_list_default(self):
        assert self.inf.infer("x", [1, 2]) == ParameterType.ARRAY

    def test_infer_from_dict_default(self):
        assert self.inf.infer("x", {"a": 1}) == ParameterType.OBJECT

    def test_infer_bool_name(self):
        assert self.inf.infer("is_active") == ParameterType.BOOLEAN
        assert self.inf.infer("has_items") == ParameterType.BOOLEAN
        assert self.inf.infer("enable_cache") == ParameterType.BOOLEAN

    def test_infer_number_name(self):
        assert self.inf.infer("count") == ParameterType.NUMBER
        assert self.inf.infer("page_limit") == ParameterType.NUMBER
        assert self.inf.infer("offset") == ParameterType.NUMBER

    def test_infer_id_name(self):
        assert self.inf.infer("user_id") == ParameterType.STRING
        assert self.inf.infer("uuid") == ParameterType.STRING

    def test_infer_fallback(self):
        assert self.inf.infer("query") == ParameterType.STRING


# ---------------------------------------------------------------------------
# RawToolParser
# ---------------------------------------------------------------------------


class TestRawToolParser:
    def setup_method(self):
        self.parser = RawToolParser(LoaderConfig())

    # -- detect_format ------------------------------------------------------

    def test_detect_single_tool(self):
        assert self.parser.detect_format({"name": "T"}) == "single_tool"

    def test_detect_tool_list(self):
        assert self.parser.detect_format([{"name": "T"}]) == "tool_list"

    def test_detect_toolbench_v1(self):
        assert self.parser.detect_format({"tools": []}) == "toolbench_v1"

    def test_detect_toolbench_v2(self):
        assert self.parser.detect_format({"api_list": []}) == "toolbench_v2"

    def test_detect_openapi(self):
        assert self.parser.detect_format({"openapi": "3.0"}) == "openapi"
        assert self.parser.detect_format({"swagger": "2.0"}) == "openapi"

    # -- parse_file ---------------------------------------------------------

    def test_parse_single_tool(self, tmp_path: Path):
        f = _write_json(tmp_path, "tool.json", SINGLE_TOOL)
        result = self.parser.parse_file(f)
        assert len(result) == 1
        assert result[0]["name"] == "WeatherAPI"

    def test_parse_tool_list(self, tmp_path: Path):
        f = _write_json(tmp_path, "tools.json", TOOL_LIST)
        result = self.parser.parse_file(f)
        assert len(result) == 2

    def test_parse_toolbench_v1(self, tmp_path: Path):
        f = _write_json(tmp_path, "v1.json", {"tools": TOOL_LIST})
        result = self.parser.parse_file(f)
        assert len(result) == 2

    def test_parse_toolbench_v2(self, tmp_path: Path):
        f = _write_json(tmp_path, "v2.json", {"api_list": TOOL_LIST})
        result = self.parser.parse_file(f)
        assert len(result) == 2

    def test_parse_openapi(self, tmp_path: Path):
        f = _write_json(tmp_path, "oas.json", {"openapi": "3.0", "info": {}})
        result = self.parser.parse_file(f)
        assert len(result) == 1  # whole spec treated as one tool

    # -- parse_directory ----------------------------------------------------

    def test_parse_directory(self, tmp_path: Path):
        _write_json(tmp_path, "a.json", SINGLE_TOOL)
        _write_json(tmp_path, "b.json", TOOL_LIST)
        results = list(self.parser.parse_directory(tmp_path))
        assert len(results) == 3  # 1 + 2

    def test_parse_directory_skips_bad_json(self, tmp_path: Path):
        _write_json(tmp_path, "good.json", SINGLE_TOOL)
        (tmp_path / "bad.json").write_text("not json!")
        results = list(self.parser.parse_directory(tmp_path))
        assert len(results) == 1

    def test_parse_directory_recursive(self, tmp_path: Path):
        sub = tmp_path / "sub"
        sub.mkdir()
        _write_json(sub, "nested.json", SINGLE_TOOL)
        results = list(self.parser.parse_directory(tmp_path, recursive=True))
        assert len(results) == 1

    def test_parse_directory_non_recursive(self, tmp_path: Path):
        sub = tmp_path / "sub"
        sub.mkdir()
        _write_json(sub, "nested.json", SINGLE_TOOL)
        _write_json(tmp_path, "top.json", SINGLE_TOOL)
        results = list(self.parser.parse_directory(tmp_path, recursive=False))
        assert len(results) == 1  # only top-level


# ---------------------------------------------------------------------------
# ToolNormalizer
# ---------------------------------------------------------------------------


class TestToolNormalizer:
    def setup_method(self):
        self.norm = ToolNormalizer(LoaderConfig())

    # -- normalize_tool -----------------------------------------------------

    def test_normalize_single_tool(self):
        tool = self.norm.normalize_tool(SINGLE_TOOL)
        assert tool is not None
        assert tool.name == "WeatherAPI"
        assert tool.domain == "Weather"
        assert tool.base_url == "https://api.weather.com"
        assert len(tool.endpoints) == 1
        assert tool.raw_schema is not None

    def test_normalize_tool_derives_id_from_name(self):
        tool = self.norm.normalize_tool({"name": "My Cool API", "api_list": [{"name": "ep", "url": "/x"}]})
        assert tool is not None
        assert tool.tool_id == "my_cool_api"

    def test_normalize_tool_uses_explicit_id(self):
        tool = self.norm.normalize_tool({"id": "custom_id", "name": "T", "api_list": [{"name": "e", "url": "/"}]})
        assert tool is not None
        assert tool.tool_id == "custom_id"

    def test_normalize_tool_no_name_no_id_returns_none(self):
        tool = self.norm.normalize_tool({})
        assert tool is None

    def test_normalize_tool_strict_raises(self):
        norm = ToolNormalizer(LoaderConfig(strict_mode=True))
        with pytest.raises(InvalidToolError):
            norm.normalize_tool({})

    def test_normalize_tool_source_file(self):
        tool = self.norm.normalize_tool(SINGLE_TOOL, source_file="data/weather.json")
        assert tool is not None
        assert tool.source_file == "data/weather.json"

    def test_normalize_tool_alternative_field_names(self):
        raw = {
            "tool_name": "AltTool",
            "desc": "An alternative tool for testing field name mappings.",
            "category_name": "Testing",
            "host": "https://alt.api",
            "apis": [{"operation_id": "do_thing", "endpoint": "/do"}],
        }
        tool = self.norm.normalize_tool(raw)
        assert tool is not None
        assert tool.name == "AltTool"
        assert tool.description == "An alternative tool for testing field name mappings."
        assert tool.domain == "Testing"
        assert tool.base_url == "https://alt.api"

    # -- normalize_endpoint -------------------------------------------------

    def test_normalize_endpoint(self):
        raw = {
            "name": "GetForecast",
            "description": "Returns weather forecast data for a specified location.",
            "method": "POST",
            "path": "/forecast",
            "parameters": [{"name": "city", "type": "string", "required": True}],
            "response": {"status_code": 200},
        }
        ep = self.norm.normalize_endpoint(raw, "weather")
        assert ep is not None
        assert ep.name == "GetForecast"
        assert ep.method == "POST"
        assert ep.path == "/forecast"
        assert len(ep.parameters) == 1
        assert ep.response_schema is not None
        assert ep.response_schema.status_code == 200

    def test_normalize_endpoint_defaults(self):
        ep = self.norm.normalize_endpoint({"name": "ep"}, "t")
        assert ep is not None
        assert ep.method == "GET"
        assert ep.path == "/"

    def test_normalize_endpoint_string_params(self):
        """Some formats list param names as bare strings."""
        ep = self.norm.normalize_endpoint(
            {"name": "ep", "url": "/x", "required_parameters": ["city", "date"]}, "t"
        )
        assert ep is not None
        assert len(ep.parameters) == 2
        assert all(p.required for p in ep.parameters)

    def test_normalize_endpoint_no_response_schema(self):
        ep = self.norm.normalize_endpoint({"name": "ep", "url": "/x"}, "t")
        assert ep is not None
        assert ep.response_schema is None

    # -- normalize_parameter ------------------------------------------------

    def test_normalize_parameter_full(self):
        raw = {
            "name": "city",
            "description": "Target city name for the weather forecast lookup.",
            "type": "string",
            "required": True,
            "default": "London",
            "in": "query",
        }
        p = self.norm.normalize_parameter(raw)
        assert p.name == "city"
        assert p.description == "Target city name for the weather forecast lookup."
        assert p.param_type == ParameterType.STRING
        assert p.required is True
        assert p.default == "London"
        assert p.location == ParameterLocation.QUERY
        assert p.has_description is True
        assert p.has_type is True
        assert p.inferred_type is False

    def test_normalize_parameter_infers_type(self):
        p = self.norm.normalize_parameter({"name": "is_active"})
        assert p.param_type == ParameterType.BOOLEAN
        assert p.inferred_type is True
        assert p.has_type is True

    def test_normalize_parameter_no_infer(self):
        norm = ToolNormalizer(LoaderConfig(infer_types=False))
        p = norm.normalize_parameter({"name": "is_active"})
        assert p.param_type == ParameterType.UNKNOWN
        assert p.inferred_type is False

    def test_normalize_parameter_enum(self):
        p = self.norm.normalize_parameter({"name": "color", "enum": ["red", "blue"]})
        assert p.enum_values == ["red", "blue"]

    def test_normalize_parameter_optional_inverted(self):
        """optional=True means param is optional → required=False."""
        p_opt = self.norm.normalize_parameter({"name": "q", "optional": True})
        assert p_opt.required is False
        p_req = self.norm.normalize_parameter({"name": "q", "optional": False})
        assert p_req.required is True

    def test_normalize_parameter_location_mapping(self):
        p = self.norm.normalize_parameter({"name": "id", "in": "path"})
        assert p.location == ParameterLocation.PATH

    def test_normalize_parameter_nested_type(self):
        """Dot-notation field lookup for schema.type."""
        p = self.norm.normalize_parameter({"name": "x", "schema": {"type": "integer"}})
        assert p.param_type == ParameterType.INTEGER

    def test_normalize_parameter_unknown_type(self):
        p = self.norm.normalize_parameter({"name": "x", "type": "weird_custom_type"})
        assert p.param_type == ParameterType.UNKNOWN

    def test_normalize_parameter_string_required_true(self):
        p = self.norm.normalize_parameter({"name": "q", "required": "true"})
        assert p.required is True

    def test_normalize_parameter_string_required_false(self):
        p = self.norm.normalize_parameter({"name": "q", "required": "false"})
        assert p.required is False

    # -- _get_field / _get_nested -------------------------------------------

    def test_get_field_first_match(self):
        raw = {"desc": "hello", "description": "world"}
        val = self.norm._get_field(raw, "description", self.norm.TOOL_FIELD_MAPPINGS)
        assert val == "world"  # "description" is first in mapping

    def test_get_field_fallback(self):
        raw = {"summary": "hi"}
        val = self.norm._get_field(raw, "description", self.norm.TOOL_FIELD_MAPPINGS)
        assert val == "hi"

    def test_get_field_default(self):
        val = self.norm._get_field({}, "description", self.norm.TOOL_FIELD_MAPPINGS, "def")
        assert val == "def"

    def test_get_nested(self):
        raw = {"schema": {"type": "integer"}}
        assert ToolNormalizer._get_nested(raw, "schema.type") == "integer"

    def test_get_nested_missing(self):
        assert ToolNormalizer._get_nested({}, "a.b.c") is None


# ---------------------------------------------------------------------------
# ToolBenchLoader
# ---------------------------------------------------------------------------


class TestToolBenchLoader:
    def test_load_file_single(self, tmp_path: Path):
        f = _write_json(tmp_path, "tool.json", SINGLE_TOOL)
        loader = ToolBenchLoader()
        tools = loader.load_file(f)
        assert len(tools) == 1
        assert tools[0].name == "WeatherAPI"
        assert tools[0].completeness_score > 0.0

        stats = loader.get_stats()
        assert stats.files_processed == 1
        assert stats.files_failed == 0
        assert stats.tools_loaded == 1
        assert stats.endpoints_loaded == 1
        assert stats.parameters_loaded == 2

    def test_load_file_bad_json(self, tmp_path: Path):
        f = tmp_path / "bad.json"
        f.write_text("not json!")
        loader = ToolBenchLoader()
        tools = loader.load_file(f)
        assert tools == []
        assert loader.get_stats().files_failed == 1

    def test_load_file_strict_raises(self, tmp_path: Path):
        f = tmp_path / "bad.json"
        f.write_text("not json!")
        loader = ToolBenchLoader(LoaderConfig(strict_mode=True))
        with pytest.raises(ToolBenchLoaderError):
            loader.load_file(f)

    def test_load_directory(self, tmp_path: Path):
        _write_json(tmp_path, "a.json", SINGLE_TOOL)
        _write_json(tmp_path, "b.json", TOOL_LIST)
        loader = ToolBenchLoader()
        tools = loader.load_directory(tmp_path)
        assert len(tools) == 3
        stats = loader.get_stats()
        assert stats.files_processed == 2
        assert stats.tools_loaded == 3

    def test_load_directory_not_a_dir(self, tmp_path: Path):
        loader = ToolBenchLoader()
        with pytest.raises(ToolBenchLoaderError):
            loader.load_directory(tmp_path / "nonexistent")

    def test_load_directory_with_bad_file(self, tmp_path: Path):
        _write_json(tmp_path, "good.json", SINGLE_TOOL)
        (tmp_path / "bad.json").write_text("bad!")
        loader = ToolBenchLoader()
        tools = loader.load_directory(tmp_path)
        assert len(tools) == 1
        assert loader.get_stats().files_failed == 1

    def test_load_directory_max_tools(self, tmp_path: Path):
        for i in range(5):
            _write_json(tmp_path, f"t{i}.json", SINGLE_TOOL)
        loader = ToolBenchLoader(LoaderConfig(max_tools=2))
        tools = loader.load_directory(tmp_path)
        assert len(tools) == 2

    def test_load_directory_min_endpoints_filter(self, tmp_path: Path):
        no_ep = {"name": "NoEp"}  # no endpoints → filtered
        _write_json(tmp_path, "no_ep.json", no_ep)
        _write_json(tmp_path, "good.json", SINGLE_TOOL)
        loader = ToolBenchLoader(LoaderConfig(min_endpoints=1))
        tools = loader.load_directory(tmp_path)
        assert len(tools) == 1
        assert loader.get_stats().tools_skipped >= 1

    def test_load_directory_max_endpoints_filter(self, tmp_path: Path):
        loader = ToolBenchLoader(LoaderConfig(max_endpoints=0))
        _write_json(tmp_path, "tool.json", SINGLE_TOOL)
        tools = loader.load_directory(tmp_path)
        assert len(tools) == 0

    def test_load_directory_min_completeness_filter(self, tmp_path: Path):
        loader = ToolBenchLoader(LoaderConfig(min_completeness=0.99))
        _write_json(tmp_path, "tool.json", SINGLE_TOOL)
        tools = loader.load_directory(tmp_path)
        assert len(tools) == 0

    def test_completeness_scoring_applied(self, tmp_path: Path):
        f = _write_json(tmp_path, "tool.json", SINGLE_TOOL)
        loader = ToolBenchLoader()
        tools = loader.load_file(f)
        assert tools[0].completeness_score > 0.0
        assert tools[0].endpoints[0].completeness_score > 0.0

    def test_completeness_scoring_disabled(self, tmp_path: Path):
        f = _write_json(tmp_path, "tool.json", SINGLE_TOOL)
        loader = ToolBenchLoader(LoaderConfig(calculate_completeness=False))
        tools = loader.load_file(f)
        assert tools[0].completeness_score == 0.0

    def test_stats_quality_distribution(self, tmp_path: Path):
        _write_json(tmp_path, "tool.json", SINGLE_TOOL)
        loader = ToolBenchLoader()
        loader.load_directory(tmp_path)
        dist = loader.get_stats().quality_distribution
        assert sum(dist.values()) == 1

    def test_stats_missing_counts(self, tmp_path: Path):
        tool_sparse = {
            "name": "Sparse",
            "api_list": [
                {"name": "ep", "url": "/x", "parameters": [{"name": "q"}]},
            ],
        }
        _write_json(tmp_path, "sparse.json", tool_sparse)
        loader = ToolBenchLoader()
        loader.load_directory(tmp_path)
        stats = loader.get_stats()
        assert stats.missing_response_schemas >= 1

    def test_load_from_glob(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        _write_json(tmp_path, "tool.json", SINGLE_TOOL)
        monkeypatch.chdir(tmp_path)
        loader = ToolBenchLoader()
        tools = loader.load_from_glob("*.json")
        assert len(tools) == 1

    def test_stats_reset_between_loads(self, tmp_path: Path):
        _write_json(tmp_path, "tool.json", SINGLE_TOOL)
        loader = ToolBenchLoader()
        loader.load_file(tmp_path / "tool.json")
        assert loader.get_stats().tools_loaded == 1
        loader.load_file(tmp_path / "tool.json")
        assert loader.get_stats().tools_loaded == 1  # reset, not accumulated


# ---------------------------------------------------------------------------
# Package-level imports
# ---------------------------------------------------------------------------


def test_package_exports():
    from tooluse_gen.registry import (
        LoaderConfig,
        ToolBenchLoader,
    )

    assert ToolBenchLoader is not None
    assert LoaderConfig is not None
