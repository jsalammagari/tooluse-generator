"""Unit tests for the Typer CLI skeleton (Task 3)."""

import pytest
from typer.testing import CliRunner

from tooluse_gen.cli import app

pytestmark = pytest.mark.unit

RUNNER = CliRunner()


# ---------------------------------------------------------------------------
# Top-level help
# ---------------------------------------------------------------------------
def test_root_help():
    result = RUNNER.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("build", "generate", "evaluate"):
        assert cmd in result.output


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def test_build_help():
    result = RUNNER.invoke(app, ["build", "--help"])
    assert result.exit_code == 0
    for flag in (
        "--input-dir",
        "--output-dir",
        "--embedding-model",
        "--similarity-thresh",
        "--force",
    ):
        # Rich may truncate long flag names with '…' in narrow terminals; check prefix
        assert flag in result.output


def test_build_rejects_missing_input_dir():
    result = RUNNER.invoke(app, ["build", "--input-dir", "/tmp/nonexistent_dir_xyz"])
    # Build now validates input_dir exists — non-existent path should fail
    assert result.exit_code != 0


def test_build_shows_options_in_panel():
    result = RUNNER.invoke(
        app,
        ["build", "--input-dir", "/tmp/nonexistent_dir_xyz"],
    )
    # Panel is still printed even when the build later fails
    assert "tooluse build" in result.output


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------
def test_generate_help():
    result = RUNNER.invoke(app, ["generate", "--help"])
    assert result.exit_code == 0
    for flag in (
        "--output",
        "--build-dir",
        "--count",
        "--seed",
        "--min-steps",
        "--max-steps",
        "--domains",
        "--no-cross-conversa",  # Rich truncates long flag names in narrow terminals
        "--max-retries",
        "--quality-threshold",
    ):
        assert flag in result.output


def test_generate_rejects_missing_build_dir():
    result = RUNNER.invoke(app, ["generate", "--output", "out.jsonl", "--build-dir", "/tmp/nope"])
    # Generate now validates build_dir — non-existent path should fail
    assert result.exit_code != 0


def test_generate_steering_flag_shown():
    result = RUNNER.invoke(
        app,
        ["generate", "--output", "out.jsonl", "--no-cross-conversation-steering",
         "--build-dir", "/tmp/nope"],
    )
    # Table is still printed showing steering=False before the build-dir error
    assert "False" in result.output


def test_generate_steering_on_by_default_shown():
    result = RUNNER.invoke(app, ["generate", "--output", "out.jsonl", "--build-dir", "/tmp/nope"])
    # Table is still printed showing steering=True before the build-dir error
    assert "True" in result.output


def test_generate_invalid_steps_order():
    result = RUNNER.invoke(
        app, ["generate", "--output", "out.jsonl", "--min-steps", "5", "--max-steps", "2"]
    )
    assert result.exit_code == 1


def test_generate_domains_parsed():
    result = RUNNER.invoke(
        app, ["generate", "--output", "out.jsonl", "--domains", "Travel,Finance",
              "--build-dir", "/tmp/nope"]
    )
    # Table shows domains even though build_dir doesn't exist
    assert "Travel" in result.output


def test_generate_defaults_shown():
    result = RUNNER.invoke(
        app, ["generate", "--output", "out.jsonl", "--build-dir", "/tmp/nope"]
    )
    # Table shows defaults even though build_dir doesn't exist
    assert "100" in result.output  # default count
    assert "42" in result.output  # default seed
    assert "3.5" in result.output  # default quality-threshold


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------
def test_evaluate_help():
    result = RUNNER.invoke(app, ["evaluate", "--help"])
    assert result.exit_code == 0
    for flag in ("--output", "--format", "--rescore"):
        assert flag in result.output


def test_evaluate_rejects_missing_input():
    result = RUNNER.invoke(app, ["evaluate", "/tmp/nonexistent_xyz.jsonl"])
    # Evaluate now validates input_path — non-existent file should fail
    assert result.exit_code != 0


def test_evaluate_shows_input_path():
    result = RUNNER.invoke(app, ["evaluate", "/tmp/nonexistent_xyz.jsonl"])
    # Panel is still printed showing path before the validation error
    assert "nonexistent_xyz.jsonl" in result.output


def test_evaluate_invalid_format():
    result = RUNNER.invoke(app, ["evaluate", "f.jsonl", "--format", "xml"])
    assert result.exit_code == 1


def test_evaluate_rescore_flag_shown():
    result = RUNNER.invoke(app, ["evaluate", "/tmp/nonexistent_xyz.jsonl", "--rescore"])
    # Panel shows rescore=True before validation error
    assert "True" in result.output


# ---------------------------------------------------------------------------
# Global options
# ---------------------------------------------------------------------------
def test_global_quiet_suppresses_panel():
    result = RUNNER.invoke(app, ["--quiet", "build", "--input-dir", "/tmp/nonexistent_dir_xyz"])
    # In quiet mode the panel should NOT appear
    assert "tooluse build" not in result.output


def test_global_config_option():
    result = RUNNER.invoke(
        app, ["--config", "my_config.yaml", "build", "--input-dir", "/tmp/nonexistent_dir_xyz"]
    )
    # Config option should be visible in the panel even though build will fail
    assert "my_config.yaml" in result.output


def test_cli_app_exported_from_package():
    from tooluse_gen.cli import app as exported_app

    assert exported_app is not None
