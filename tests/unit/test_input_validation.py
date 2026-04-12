"""Tests for CLI input validation (Task 82)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from tooluse_gen.cli.main import app

pytestmark = pytest.mark.unit

runner = CliRunner()


# ---------------------------------------------------------------------------
# Mock embedding service (for build edge-case tests)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_embeddings(mock_embedding_service: type) -> object:
    with (
        patch("tooluse_gen.graph.builder.EmbeddingService", mock_embedding_service),
        patch("tooluse_gen.graph.embeddings.EmbeddingService", mock_embedding_service),
    ):
        yield


# ===================================================================
# Build validation
# ===================================================================


class TestBuildValidation:
    def test_missing_input_dir(self) -> None:
        result = runner.invoke(app, ["build", "--input-dir", "/tmp/nope_xyz_82"])
        assert result.exit_code != 0
        assert "does not exist" in result.output

    def test_empty_input_dir_no_json(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty_dir"
        empty.mkdir()
        result = runner.invoke(app, [
            "build", "--input-dir", str(empty), "--output-dir", str(tmp_path / "out"),
        ])
        assert result.exit_code != 0
        assert "No JSON files" in result.output

    def test_no_force_existing_artifacts(self, tmp_path: Path) -> None:
        out = tmp_path / "build"
        out.mkdir()
        (out / "registry.json").write_text("{}")
        result = runner.invoke(app, [
            "build", "--input-dir", str(tmp_path), "--output-dir", str(out),
        ])
        assert result.exit_code != 0
        assert "force" in result.output.lower() or "artifact" in result.output.lower()

    def test_input_dir_with_json_succeeds(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        data.mkdir()
        # Rich enough tool to pass FAIR quality filter
        (data / "tool.json").write_text(json.dumps({
            "tool_name": "Weather API", "tool_description": "Get weather forecasts and conditions",
            "home_url": "https://api.weather.example.com",
            "api_list": [
                {"name": "current", "url": "/current",
                 "description": "Get current weather for a city",
                 "method": "GET",
                 "required_parameters": [
                     {"name": "city", "type": "STRING", "description": "City name", "default": ""}
                 ],
                 "optional_parameters": [
                     {"name": "units", "type": "STRING", "description": "Units", "default": "metric"}
                 ]},
                {"name": "forecast", "url": "/forecast",
                 "description": "Get weather forecast",
                 "method": "GET",
                 "required_parameters": [
                     {"name": "city", "type": "STRING", "description": "City", "default": ""}
                 ],
                 "optional_parameters": []},
            ],
        }))
        out = tmp_path / "out"
        result = runner.invoke(app, [
            "build", "--input-dir", str(data), "--output-dir", str(out), "--force",
        ])
        assert result.exit_code == 0


# ===================================================================
# Generate validation
# ===================================================================


class TestGenerateValidation:
    def test_missing_build_dir(self) -> None:
        result = runner.invoke(app, [
            "generate", "--output", "x.jsonl", "--build-dir", "/tmp/nope_xyz_82",
        ])
        assert result.exit_code != 0
        assert "does not exist" in result.output

    def test_missing_registry(self, tmp_path: Path) -> None:
        bd = tmp_path / "bad_build"
        bd.mkdir()
        (bd / "graph.pkl").write_bytes(b"x")
        result = runner.invoke(app, [
            "generate", "--output", str(tmp_path / "x.jsonl"), "--build-dir", str(bd),
        ])
        assert result.exit_code != 0
        assert "registry" in result.output.lower()

    def test_missing_graph(self, tmp_path: Path) -> None:
        bd = tmp_path / "bad_build"
        bd.mkdir()
        (bd / "registry.json").write_text("{}")
        result = runner.invoke(app, [
            "generate", "--output", str(tmp_path / "x.jsonl"), "--build-dir", str(bd),
        ])
        assert result.exit_code != 0
        assert "graph" in result.output.lower()

    def test_min_greater_than_max(self) -> None:
        result = runner.invoke(app, [
            "generate", "--output", "x.jsonl",
            "--min-steps", "5", "--max-steps", "2",
        ])
        assert result.exit_code == 1


# ===================================================================
# Evaluate validation
# ===================================================================


class TestEvaluateValidation:
    def test_missing_input_file(self) -> None:
        result = runner.invoke(app, ["evaluate", "/tmp/nope_xyz_82.jsonl"])
        assert result.exit_code != 0

    def test_invalid_format(self) -> None:
        result = runner.invoke(app, ["evaluate", "x.jsonl", "--format", "xml"])
        assert result.exit_code == 1

    def test_empty_input(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        result = runner.invoke(app, ["evaluate", str(empty)])
        assert result.exit_code != 0

    def test_directory_as_input(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["evaluate", str(tmp_path)])
        assert result.exit_code != 0
        assert "file, not a directory" in result.output


# ===================================================================
# ToolBench edge cases
# ===================================================================


class TestEdgeCases:
    def test_build_handles_malformed_json(self, tmp_path: Path) -> None:
        """Build doesn't crash on malformed JSON (loader skips them)."""
        data = tmp_path / "data"
        data.mkdir()
        (data / "bad.json").write_text("{{{not json at all")
        # Good tool must be rich enough to pass FAIR quality filter
        (data / "good.json").write_text(json.dumps({
            "tool_name": "Weather API", "tool_description": "Get weather data",
            "home_url": "https://api.weather.example.com",
            "api_list": [
                {"name": "current", "url": "/current", "description": "Current weather",
                 "method": "GET",
                 "required_parameters": [
                     {"name": "city", "type": "STRING", "description": "City", "default": ""}
                 ],
                 "optional_parameters": []},
                {"name": "forecast", "url": "/forecast", "description": "Forecast",
                 "method": "GET",
                 "required_parameters": [
                     {"name": "city", "type": "STRING", "description": "City", "default": ""}
                 ],
                 "optional_parameters": []},
            ],
        }))
        out = tmp_path / "out"
        result = runner.invoke(app, [
            "build", "--input-dir", str(data), "--output-dir", str(out), "--force",
        ])
        # Should succeed — loader skips malformed files
        assert result.exit_code == 0

    def test_build_handles_empty_descriptions(self, tmp_path: Path) -> None:
        """Build doesn't crash on tools with empty descriptions (may be filtered)."""
        data = tmp_path / "data"
        data.mkdir()
        (data / "tool.json").write_text(json.dumps({
            "tool_name": "NoDesc", "tool_description": "",
            "api_list": [{"name": "get", "url": "/get", "description": "",
                          "method": "GET", "required_parameters": [], "optional_parameters": []}],
        }))
        out = tmp_path / "out"
        result = runner.invoke(app, [
            "build", "--input-dir", str(data), "--output-dir", str(out), "--force",
        ])
        # Tool with empty descriptions likely fails quality filter
        # The key test is that it doesn't crash with an unhandled exception
        assert result.exit_code in (0, 1)

    def test_build_handles_zero_endpoint_tools(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        data.mkdir()
        (data / "empty_tool.json").write_text(json.dumps({
            "tool_name": "Empty", "tool_description": "No endpoints",
            "api_list": [],
        }))
        out = tmp_path / "out"
        result = runner.invoke(app, [
            "build", "--input-dir", str(data), "--output-dir", str(out), "--force",
        ])
        # Should fail because no tools pass quality filter (0 endpoints)
        assert result.exit_code != 0
