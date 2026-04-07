"""Smoke tests for Task 1 — package structure and CLI entry point."""

import importlib
import pathlib

import pytest
from typer.testing import CliRunner

from tooluse_gen.cli.main import app
import tooluse_gen


RUNNER = CliRunner()


def test_version_defined():
    assert tooluse_gen.__version__ == "0.1.0"


@pytest.mark.parametrize("submodule", [
    "tooluse_gen.cli.main",
    "tooluse_gen.core",
    "tooluse_gen.agents",
    "tooluse_gen.graph",
    "tooluse_gen.registry",
    "tooluse_gen.evaluation",
    "tooluse_gen.utils",
])
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
    result = RUNNER.invoke(app, ["build"])
    assert result.exit_code == 0
    assert "not yet implemented" in result.output


def test_cli_generate_defaults():
    result = RUNNER.invoke(app, ["generate"])
    assert result.exit_code == 0
    assert "seed=42" in result.output
    assert "steering=True" in result.output


def test_cli_generate_no_steering():
    result = RUNNER.invoke(app, ["generate", "--no-cross-conversation-steering"])
    assert result.exit_code == 0
    assert "steering=False" in result.output


def test_cli_evaluate_invokes():
    result = RUNNER.invoke(app, ["evaluate"])
    assert result.exit_code == 0
    assert "not yet implemented" in result.output


@pytest.mark.parametrize("required", [
    "src/tooluse_gen/__init__.py",
    "src/tooluse_gen/cli/main.py",
    "config/default.yaml",
    "pyproject.toml",
    "README.md",
    "DESIGN.md",
    ".gitignore",
    ".env.example",
    "data/.gitkeep",
    "output/.gitkeep",
    "tests/unit/__init__.py",
    "tests/integration/__init__.py",
    "tests/e2e/__init__.py",
])
def test_required_file_exists(required):
    base = pathlib.Path(__file__).parents[2]
    assert (base / required).exists(), f"Missing: {required}"


def test_config_yaml_loads():
    import yaml
    base = pathlib.Path(__file__).parents[2]
    cfg = yaml.safe_load((base / "config/default.yaml").read_text())
    assert "models" in cfg
    assert "generation" in cfg
    assert "steering" in cfg
    assert cfg["generation"]["seed"] == 42
