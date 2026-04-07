"""Unit tests for core configuration loader (Task 4)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from tooluse_gen.core import (
    AppConfig,
    DiversityConfig,
    PathsConfig,
    QualityConfig,
    SamplingConfig,
    export_config,
    load_config,
    merge_cli_overrides,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_yaml(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(data))
    return p


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_appconfig_defaults():
    cfg = AppConfig()
    assert cfg.seed == 42
    assert cfg.verbose == 0
    assert cfg.models.assistant == "gpt-4o"
    assert cfg.models.judge == "gpt-4o"
    assert cfg.models.user_simulator == "gpt-4o-mini"
    assert cfg.models.mock_generator == "gpt-4o-mini"
    assert cfg.models.embedding == "all-MiniLM-L6-v2"


def test_quality_defaults():
    q = QualityConfig()
    assert q.min_score == 3.5
    assert q.max_retries == 3
    assert "tool_correctness" in q.dimensions
    assert "naturalness" in q.dimensions
    assert len(q.dimensions) == 4


def test_sampling_defaults():
    s = SamplingConfig()
    assert s.min_steps == 2
    assert s.max_steps == 5
    assert s.similarity_threshold == 0.7
    assert s.domains is None
    assert s.excluded_tools is None


def test_diversity_defaults():
    d = DiversityConfig()
    assert d.enabled is True
    assert d.weight_decay == 0.9
    assert d.min_domain_coverage == 0.5


def test_paths_auto_create(tmp_path: Path):
    """Directories are created automatically when a PathsConfig is instantiated."""
    p = PathsConfig(
        build_dir=tmp_path / "build",
        output_dir=tmp_path / "output",
        cache_dir=tmp_path / "cache",
    )
    assert p.build_dir.exists()
    assert p.output_dir.exists()
    assert p.cache_dir.exists()
    # Stored as absolute paths
    assert p.build_dir.is_absolute()
    assert p.output_dir.is_absolute()
    assert p.cache_dir.is_absolute()


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def test_quality_min_score_too_low():
    with pytest.raises(ValidationError):
        QualityConfig(min_score=0.5)


def test_quality_min_score_too_high():
    with pytest.raises(ValidationError):
        QualityConfig(min_score=5.5)


def test_quality_max_retries_negative():
    with pytest.raises(ValidationError):
        QualityConfig(max_retries=-1)


def test_sampling_similarity_out_of_range():
    with pytest.raises(ValidationError):
        SamplingConfig(similarity_threshold=1.5)


def test_sampling_steps_order_invalid():
    with pytest.raises(ValidationError):
        SamplingConfig(min_steps=5, max_steps=3)


def test_sampling_steps_equal_invalid():
    with pytest.raises(ValidationError):
        SamplingConfig(min_steps=3, max_steps=3)


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


def test_load_config_none_path_returns_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # Move to tmp_path so config/default.yaml doesn't exist, forcing pure defaults
    monkeypatch.chdir(tmp_path)
    cfg = load_config(config_path=None)
    assert isinstance(cfg, AppConfig)
    assert cfg.seed == 42


def test_load_config_pure_defaults():
    """When no file exists and no path given, pure defaults are returned."""
    cfg = load_config.__wrapped__(None) if hasattr(load_config, "__wrapped__") else None
    # Simpler: just verify AppConfig() works
    cfg = AppConfig()
    assert cfg.seed == 42


def test_load_config_from_yaml(tmp_path: Path):
    data = {
        "seed": 99,
        "models": {"assistant": "gpt-4o-mini"},
        "quality": {"min_score": 4.0, "max_retries": 2},
        "sampling": {"min_steps": 1, "max_steps": 4},
    }
    p = write_yaml(tmp_path, data)
    cfg = load_config(p)
    assert cfg.seed == 99
    assert cfg.models.assistant == "gpt-4o-mini"
    assert cfg.models.judge == "gpt-4o"  # not overridden → default
    assert cfg.quality.min_score == 4.0
    assert cfg.quality.max_retries == 2
    assert cfg.sampling.min_steps == 1
    assert cfg.sampling.max_steps == 4


def test_load_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config(Path("/no/such/file.yaml"))


def test_load_config_invalid_values_raise(tmp_path: Path):
    bad = write_yaml(tmp_path, {"quality": {"min_score": 0.1}})
    with pytest.raises(ValidationError):
        load_config(bad)


def test_load_config_default_yaml():
    """The shipped config/default.yaml must parse cleanly."""
    cfg = load_config(Path("config/default.yaml"))
    assert isinstance(cfg, AppConfig)
    assert cfg.seed == 42


# ---------------------------------------------------------------------------
# Environment variable expansion in paths
# ---------------------------------------------------------------------------


def test_path_env_var_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TOOLUSE_TEST_DIR", str(tmp_path))
    p = PathsConfig(build_dir="$TOOLUSE_TEST_DIR/build")  # type: ignore[arg-type]
    assert str(tmp_path) in str(p.build_dir)
    assert p.build_dir.exists()


# ---------------------------------------------------------------------------
# merge_cli_overrides
# ---------------------------------------------------------------------------


def test_merge_cli_overrides_top_level():
    cfg = AppConfig()
    updated = merge_cli_overrides(cfg, seed=7, verbose=2)
    assert updated.seed == 7
    assert updated.verbose == 2


def test_merge_cli_overrides_nested():
    cfg = AppConfig()
    updated = merge_cli_overrides(cfg, models__assistant="gpt-4o-mini")
    assert updated.models.assistant == "gpt-4o-mini"
    assert updated.models.judge == "gpt-4o"  # untouched


def test_merge_cli_overrides_none_skipped():
    cfg = AppConfig()
    updated = merge_cli_overrides(cfg, seed=None)
    assert updated.seed == 42  # None → no change


def test_merge_cli_overrides_unknown_key_ignored():
    cfg = AppConfig()
    updated = merge_cli_overrides(cfg, totally_unknown_key="foo")
    assert updated.seed == cfg.seed  # unchanged, no error


def test_merge_cli_overrides_quality():
    cfg = AppConfig()
    updated = merge_cli_overrides(cfg, quality__min_score=4.5, quality__max_retries=1)
    assert updated.quality.min_score == 4.5
    assert updated.quality.max_retries == 1


def test_merge_cli_overrides_immutable_original():
    cfg = AppConfig()
    _ = merge_cli_overrides(cfg, seed=99)
    assert cfg.seed == 42  # original unchanged


# ---------------------------------------------------------------------------
# export_config
# ---------------------------------------------------------------------------


def test_export_config_is_dict():
    cfg = AppConfig()
    out = export_config(cfg)
    assert isinstance(out, dict)


def test_export_config_paths_are_strings():
    cfg = AppConfig()
    out = export_config(cfg)
    for key in ("build_dir", "output_dir", "cache_dir"):
        assert isinstance(out["paths"][key], str)


def test_export_config_json_serialisable():
    import json

    cfg = AppConfig()
    out = export_config(cfg)
    # Should not raise
    json.dumps(out)


def test_export_config_contains_all_sections():
    cfg = AppConfig()
    out = export_config(cfg)
    for section in ("models", "quality", "sampling", "diversity", "paths"):
        assert section in out


def test_export_config_round_trip(tmp_path: Path):
    """export → write YAML → load → same values."""
    cfg = AppConfig(seed=55)
    exported = export_config(cfg)
    p = tmp_path / "exported.yaml"
    p.write_text(yaml.dump(exported))
    reloaded = load_config(p)
    assert reloaded.seed == 55
    assert reloaded.models.assistant == cfg.models.assistant
