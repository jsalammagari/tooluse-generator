"""Shared fixtures for registry unit tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tooluse_gen.registry.loader import LoaderConfig, ToolBenchLoader

# ---------------------------------------------------------------------------
# Sample data with various completeness levels
# ---------------------------------------------------------------------------

COMPLETE_TOOL: dict = {
    "tool_id": "weather_api",
    "name": "Weather API",
    "description": "Get weather information for any location worldwide.",
    "domain": "Weather",
    "base_url": "https://api.weather.com",
    "endpoints": [
        {
            "name": "Get Current Weather",
            "description": "Get current weather conditions for a given location.",
            "method": "GET",
            "path": "/weather/current",
            "parameters": [
                {
                    "name": "location",
                    "description": "City name or coordinates to look up.",
                    "type": "string",
                    "required": True,
                },
                {
                    "name": "units",
                    "type": "string",
                    "default": "celsius",
                    "enum": ["celsius", "fahrenheit"],
                },
            ],
            "response": {"status_code": 200, "description": "Weather data"},
        },
    ],
}

MINIMAL_TOOL: dict = {
    "name": "Minimal API",
    "endpoints": [{"path": "/data"}],
}

EMPTY_TOOL: dict = {}

MALFORMED_TOOL: dict = {
    "name": "Malformed",
    "endpoints": "not_an_array",
}

UNICODE_TOOL: dict = {
    "name": "日本語 API",
    "description": "émojis 🎉 and spëcial çharacters.",
    "endpoints": [{"name": "获取数据", "path": "/données"}],
}

LARGE_TOOL: dict = {
    "name": "Large API",
    "endpoints": [{"name": f"ep_{i}", "path": f"/p/{i}"} for i in range(150)],
}

NULL_VALUES_TOOL: dict = {
    "name": "NullTool",
    "description": None,
    "domain": None,
    "endpoints": [
        {
            "name": "ep",
            "path": "/x",
            "method": None,
            "parameters": [
                {"name": "q", "type": None, "description": None, "required": None},
            ],
        },
    ],
}

DUPLICATE_PARAMS_TOOL: dict = {
    "name": "DupParams",
    "endpoints": [
        {
            "name": "ep",
            "path": "/x",
            "parameters": [
                {"name": "q", "type": "string"},
                {"name": "q", "type": "integer"},
            ],
        },
    ],
}

ALT_FORMAT_TOOL: dict = {
    "tool_name": "AltFormat",
    "desc": "Uses alternative field names for all the fields.",
    "category_name": "Testing",
    "host": "alt.api.com",
    "apis": [
        {
            "api_name": "DoThing",
            "api_description": "Performs an action on the server with given params.",
            "http_method": "post",
            "api_url": "/do",
            "params": [
                {"param_name": "input", "data_type": "string", "is_required": True},
            ],
        },
    ],
}

METHOD_VARIANTS_TOOL: dict = {
    "name": "MethodVariants",
    "endpoints": [
        {"name": "get_ep", "path": "/a", "method": "GET"},
        {"name": "post_ep", "path": "/b", "method": "post"},
        {"name": "put_ep", "path": "/c", "method": "PUT"},
        {"name": "del_ep", "path": "/d", "method": "delete"},
        {"name": "no_method", "path": "/e"},
    ],
}

PATH_VARIANTS_TOOL: dict = {
    "name": "PathVariants",
    "endpoints": [
        {"name": "brace", "path": "/users/{id}"},
        {"name": "colon", "path": "/users/:id"},
        {"name": "angle", "path": "/users/<id>"},
        {"name": "query", "path": "/search?q=test"},
        {"name": "trailing", "path": "/items/"},
        {"name": "no_leading", "path": "items"},
    ],
}

TYPE_VARIANTS_TOOL: dict = {
    "name": "TypeVariants",
    "endpoints": [
        {
            "name": "ep",
            "path": "/x",
            "parameters": [
                {"name": "a", "type": "string"},
                {"name": "b", "type": "int"},
                {"name": "c", "type": "boolean"},
                {"name": "d", "type": "float"},
                {"name": "e", "type": "array"},
                {"name": "f", "type": "OBJECT"},
                {"name": "g", "type": "date-time"},
                {"name": "h", "type": "unknown_type"},
                {"name": "i"},  # no type
            ],
        },
    ],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def temp_data_dir(tmp_path: Path) -> Path:
    """Create a temp directory populated with test JSON files."""
    (tmp_path / "complete.json").write_text(json.dumps(COMPLETE_TOOL))
    (tmp_path / "minimal.json").write_text(json.dumps(MINIMAL_TOOL))
    (tmp_path / "empty.json").write_text(json.dumps(EMPTY_TOOL))
    (tmp_path / "unicode.json").write_text(
        json.dumps(UNICODE_TOOL, ensure_ascii=False), encoding="utf-8"
    )
    (tmp_path / "multiple.json").write_text(json.dumps([COMPLETE_TOOL, MINIMAL_TOOL]))
    (tmp_path / "malformed_ep.json").write_text(json.dumps(MALFORMED_TOOL))
    (tmp_path / "null_values.json").write_text(json.dumps(NULL_VALUES_TOOL))
    (tmp_path / "dup_params.json").write_text(json.dumps(DUPLICATE_PARAMS_TOOL))
    (tmp_path / "alt_format.json").write_text(json.dumps(ALT_FORMAT_TOOL))
    (tmp_path / "method_variants.json").write_text(json.dumps(METHOD_VARIANTS_TOOL))
    (tmp_path / "path_variants.json").write_text(json.dumps(PATH_VARIANTS_TOOL))
    (tmp_path / "type_variants.json").write_text(json.dumps(TYPE_VARIANTS_TOOL))
    (tmp_path / "invalid.json").write_text("NOT VALID JSON {{{")
    (tmp_path / "large.json").write_text(json.dumps(LARGE_TOOL))

    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "nested.json").write_text(json.dumps(COMPLETE_TOOL))

    return tmp_path


@pytest.fixture()
def loader() -> ToolBenchLoader:
    return ToolBenchLoader(LoaderConfig())


@pytest.fixture()
def strict_loader() -> ToolBenchLoader:
    return ToolBenchLoader(LoaderConfig(strict_mode=True))
