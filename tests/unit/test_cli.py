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
    for flag in ("--input-dir", "--output-dir", "--embedding-model", "--similarity-thresh", "--force"):
        # Rich may truncate long flag names with '…' in narrow terminals; check prefix
        assert flag in result.output


def test_build_stub_output():
    result = RUNNER.invoke(app, ["build", "--input-dir", "data/toolbench"])
    assert result.exit_code == 0
    assert "Not implemented yet" in result.output


def test_build_shows_options_in_output():
    result = RUNNER.invoke(
        app,
        ["build", "--input-dir", "data/toolbench", "--output-dir", "out/build", "--force"],
    )
    assert result.exit_code == 0
    assert "data/toolbench" in result.output
    # Path normalises "out/build" — check the directory name appears
    assert "out" in result.output


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


def test_generate_stub_output():
    result = RUNNER.invoke(app, ["generate", "--output", "out.jsonl"])
    assert result.exit_code == 0
    assert "Not implemented yet" in result.output


def test_generate_steering_flag():
    result = RUNNER.invoke(
        app, ["generate", "--output", "out.jsonl", "--no-cross-conversation-steering"]
    )
    assert result.exit_code == 0
    assert "False" in result.output  # steering disabled


def test_generate_steering_on_by_default():
    result = RUNNER.invoke(app, ["generate", "--output", "out.jsonl"])
    assert result.exit_code == 0
    assert "True" in result.output  # steering enabled by default


def test_generate_invalid_steps_order():
    result = RUNNER.invoke(
        app, ["generate", "--output", "out.jsonl", "--min-steps", "5", "--max-steps", "2"]
    )
    assert result.exit_code == 1


def test_generate_domains_parsed():
    result = RUNNER.invoke(
        app, ["generate", "--output", "out.jsonl", "--domains", "Travel,Finance"]
    )
    assert result.exit_code == 0
    assert "Travel" in result.output


def test_generate_defaults():
    result = RUNNER.invoke(app, ["generate", "--output", "out.jsonl"])
    assert result.exit_code == 0
    assert "100" in result.output   # default count
    assert "42" in result.output    # default seed
    assert "3.5" in result.output   # default quality-threshold


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------
def test_evaluate_help():
    result = RUNNER.invoke(app, ["evaluate", "--help"])
    assert result.exit_code == 0
    for flag in ("--output", "--format", "--rescore"):
        assert flag in result.output


def test_evaluate_stub_output():
    result = RUNNER.invoke(app, ["evaluate", "conversations.jsonl"])
    assert result.exit_code == 0
    assert "Not implemented yet" in result.output


def test_evaluate_shows_input_path():
    result = RUNNER.invoke(app, ["evaluate", "my_data.jsonl"])
    assert result.exit_code == 0
    assert "my_data.jsonl" in result.output


def test_evaluate_invalid_format():
    result = RUNNER.invoke(app, ["evaluate", "f.jsonl", "--format", "xml"])
    assert result.exit_code == 1


def test_evaluate_valid_formats():
    for fmt in ("json", "table", "markdown"):
        result = RUNNER.invoke(app, ["evaluate", "f.jsonl", "--format", fmt])
        assert result.exit_code == 0, f"format={fmt} failed"


def test_evaluate_rescore_flag():
    result = RUNNER.invoke(app, ["evaluate", "f.jsonl", "--rescore"])
    assert result.exit_code == 0
    assert "True" in result.output


# ---------------------------------------------------------------------------
# Global options
# ---------------------------------------------------------------------------
def test_global_quiet_suppresses_panel():
    result = RUNNER.invoke(app, ["--quiet", "build", "--input-dir", "data/"])
    assert result.exit_code == 0
    # Panel/table not rendered; only stub line
    assert "Not implemented yet" in result.output
    assert "input-dir" not in result.output


def test_global_config_option():
    result = RUNNER.invoke(
        app, ["--config", "my_config.yaml", "build", "--input-dir", "data/"]
    )
    assert result.exit_code == 0
    assert "my_config.yaml" in result.output


def test_cli_app_exported_from_package():
    from tooluse_gen.cli import app as exported_app

    assert exported_app is not None
