"""Tests for CLI flag wiring (Task 61)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tooluse_gen.cli.main import _state, app

pytestmark = pytest.mark.unit

runner = CliRunner()


# ===================================================================
# --verbose / -v
# ===================================================================


class TestVerboseFlag:
    def test_default_verbosity_zero(self) -> None:
        runner.invoke(app, ["build", "--help"])
        assert _state["verbose"] == 0

    def test_single_v(self) -> None:
        runner.invoke(app, ["-v", "build", "--help"])
        assert _state["verbose"] == 1

    def test_double_v(self) -> None:
        runner.invoke(app, ["-vv", "build", "--help"])
        assert _state["verbose"] == 2

    def test_verbose_sets_debug_logging(self) -> None:
        runner.invoke(app, ["-vv", "build", "--help"])
        root = logging.getLogger("tooluse_gen")
        assert root.level <= logging.DEBUG


# ===================================================================
# --quiet / -q
# ===================================================================


class TestQuietFlag:
    def test_default_not_quiet(self) -> None:
        runner.invoke(app, ["build", "--help"])
        assert _state["quiet"] is False

    def test_quiet_sets_state(self) -> None:
        runner.invoke(app, ["--quiet", "build", "--help"])
        assert _state["quiet"] is True

    def test_quiet_suppresses_build_panel(self) -> None:
        result = runner.invoke(
            app, ["--quiet", "build", "--input-dir", "/tmp/nope_xyz"]
        )
        assert "tooluse build" not in result.output


# ===================================================================
# --config / -c
# ===================================================================


class TestConfigFlag:
    def test_default_config_path(self) -> None:
        runner.invoke(app, ["build", "--help"])
        assert str(_state["config"]) == "config/default.yaml"

    def test_custom_config_path(self) -> None:
        runner.invoke(app, ["--config", "my_config.yaml", "build", "--help"])
        assert str(_state["config"]) == "my_config.yaml"

    def test_config_shown_in_build_output(self) -> None:
        result = runner.invoke(
            app, ["--config", "custom.yaml", "build", "--input-dir", "/tmp/nope_xyz"]
        )
        assert "custom.yaml" in result.output

    def test_config_shown_in_generate_output(self) -> None:
        result = runner.invoke(
            app,
            ["--config", "custom.yaml", "generate", "--output", "x.jsonl",
             "--build-dir", "/tmp/nope_xyz"],
        )
        assert "custom.yaml" in result.output


# ===================================================================
# --config-from
# ===================================================================


class TestConfigFromFlag:
    def test_config_from_stored_in_state(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "prev.jsonl"
        header = {"__metadata__": True, "config": {}, "seed": 42}
        jsonl.write_text(json.dumps(header) + "\n")
        runner.invoke(app, ["--config-from", str(jsonl), "build", "--help"])
        assert _state.get("config_from") is not None

    def test_config_from_not_set_by_default(self) -> None:
        runner.invoke(app, ["build", "--help"])
        assert _state.get("config_from") is None

    def test_config_from_shown_in_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert "config-from" in result.output


# ===================================================================
# --no-cross-conversation-steering
# ===================================================================


class TestSteeringFlag:
    def test_steering_enabled_by_default(self) -> None:
        result = runner.invoke(
            app,
            ["generate", "--output", "out.jsonl", "--build-dir", "/tmp/nope_xyz"],
        )
        assert "True" in result.output

    def test_no_steering_flag(self) -> None:
        result = runner.invoke(
            app,
            ["generate", "--output", "out.jsonl",
             "--no-cross-conversation-steering", "--build-dir", "/tmp/nope_xyz"],
        )
        assert "False" in result.output

    def test_steering_flag_in_help(self) -> None:
        result = runner.invoke(app, ["generate", "--help"])
        assert "no-cross-conversa" in result.output


# ===================================================================
# --no-cache
# ===================================================================


class TestNoCacheFlag:
    def test_no_cache_shown_in_output(self) -> None:
        result = runner.invoke(
            app,
            ["generate", "--output", "out.jsonl",
             "--no-cache", "--build-dir", "/tmp/nope_xyz"],
        )
        assert "disabled" in result.output.lower()

    def test_cache_enabled_by_default(self) -> None:
        result = runner.invoke(
            app,
            ["generate", "--output", "out.jsonl", "--build-dir", "/tmp/nope_xyz"],
        )
        assert "enabled" in result.output.lower()

    def test_no_cache_in_help(self) -> None:
        result = runner.invoke(app, ["generate", "--help"])
        assert "no-cache" in result.output


# ===================================================================
# Input validation
# ===================================================================


class TestInputValidation:
    def test_build_rejects_nonexistent_input_dir(self) -> None:
        result = runner.invoke(app, ["build", "--input-dir", "/tmp/nope_xyz"])
        assert result.exit_code != 0

    def test_generate_rejects_nonexistent_build_dir(self) -> None:
        result = runner.invoke(
            app,
            ["generate", "--output", "out.jsonl", "--build-dir", "/tmp/nope_xyz"],
        )
        assert result.exit_code != 0

    def test_evaluate_rejects_nonexistent_input(self) -> None:
        result = runner.invoke(app, ["evaluate", "/tmp/nope_xyz.jsonl"])
        assert result.exit_code != 0

    def test_generate_rejects_min_gt_max_steps(self) -> None:
        result = runner.invoke(
            app,
            ["generate", "--output", "out.jsonl",
             "--min-steps", "5", "--max-steps", "2"],
        )
        assert result.exit_code == 1

    def test_evaluate_rejects_invalid_format(self) -> None:
        result = runner.invoke(app, ["evaluate", "f.jsonl", "--format", "xml"])
        assert result.exit_code == 1

    def test_build_no_force_rejects_existing(self, tmp_path: Path) -> None:
        out = tmp_path / "build_out"
        out.mkdir()
        (out / "registry.json").write_text("{}")
        result = runner.invoke(
            app,
            ["build", "--input-dir", str(tmp_path), "--output-dir", str(out)],
        )
        assert result.exit_code != 0
