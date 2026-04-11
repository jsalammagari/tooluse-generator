"""Smoke tests for Task 1 — package structure and CLI entry point."""

import importlib
import pathlib

import pytest
from typer.testing import CliRunner

import tooluse_gen
from tooluse_gen.cli.main import app

pytestmark = pytest.mark.unit

RUNNER = CliRunner()


def test_version_defined():
    assert tooluse_gen.__version__ == "0.1.0"


@pytest.mark.parametrize(
    "submodule",
    [
        "tooluse_gen.cli.main",
        "tooluse_gen.core",
        "tooluse_gen.agents",
        "tooluse_gen.graph",
        "tooluse_gen.registry",
        "tooluse_gen.evaluation",
        "tooluse_gen.utils",
    ],
)
def test_submodule_imports(submodule):
    mod = importlib.import_module(submodule)
    assert mod is not None


def test_cli_help():
    result = RUNNER.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "build" in result.output
    assert "generate" in result.output
    assert "evaluate" in result.output


def test_cli_build_invokes():
    # Build with non-existent input dir should fail gracefully (exit 1)
    result = RUNNER.invoke(app, ["build", "--input-dir", "/tmp/nonexistent_dir_xyz"])
    assert result.exit_code == 1


def test_cli_generate_defaults():
    # Generate with default build-dir that doesn't exist should fail gracefully
    result = RUNNER.invoke(
        app, ["generate", "--output", "out.jsonl", "--build-dir", "/tmp/nope"]
    )
    assert result.exit_code == 1
    assert "42" in result.output  # default seed shown in table before error


def test_cli_generate_no_steering():
    result = RUNNER.invoke(
        app,
        ["generate", "--output", "out.jsonl", "--no-cross-conversation-steering",
         "--build-dir", "/tmp/nope"],
    )
    assert result.exit_code == 1
    assert "False" in result.output  # steering disabled shown in table


def test_cli_evaluate_invokes():
    # INPUT is now a required positional argument
    result = RUNNER.invoke(app, ["evaluate", "conversations.jsonl"])
    assert result.exit_code == 0
    assert "Not implemented yet" in result.output


@pytest.mark.parametrize(
    "required",
    [
        "src/tooluse_gen/__init__.py",
        "src/tooluse_gen/cli/main.py",
        "config/default.yaml",
        "pyproject.toml",
        "README.md",
        "DESIGN.md",
        ".gitignore",
        ".env.example",
        "data/toolenv/tools",
        "output/.gitkeep",
        "tests/unit/__init__.py",
        "tests/integration/__init__.py",
        "tests/e2e/__init__.py",
    ],
)
def test_required_file_exists(required):
    base = pathlib.Path(__file__).parents[2]
    assert (base / required).exists(), f"Missing: {required}"


def test_config_yaml_loads():
    import yaml

    base = pathlib.Path(__file__).parents[2]
    cfg = yaml.safe_load((base / "config/default.yaml").read_text())
    assert "models" in cfg
    assert "quality" in cfg
    assert "sampling" in cfg
    assert "diversity" in cfg
    assert "paths" in cfg
    assert cfg["seed"] == 42
