"""Tests for reproducibility utilities (Task 51)."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from tooluse_gen.core.config import load_config
from tooluse_gen.core.jsonl_io import JSONLReader, JSONLWriter
from tooluse_gen.core.output_models import ConversationRecord
from tooluse_gen.core.reproducibility import (
    compare_configs,
    embed_config_in_output,
    ensure_reproducibility,
    load_config_from_output,
    serialize_run_config,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(cid: str = "c1") -> ConversationRecord:
    return ConversationRecord(
        conversation_id=cid,
        messages=[{"role": "user", "content": "hi"}],
        metadata={"seed": 42},
    )


# ===================================================================
# serialize_run_config
# ===================================================================


class TestSerializeRunConfig:
    def test_returns_dict(self):
        cfg = load_config()
        result = serialize_run_config(cfg, seed=42)
        assert isinstance(result, dict)

    def test_has_config_key(self):
        cfg = load_config()
        result = serialize_run_config(cfg, seed=42)
        assert "config" in result
        assert isinstance(result["config"], dict)
        assert "sampling" in result["config"]

    def test_has_seed(self):
        cfg = load_config()
        result = serialize_run_config(cfg, seed=99)
        assert result["seed"] == 99

    def test_has_timestamp(self):
        cfg = load_config()
        result = serialize_run_config(cfg, seed=42)
        assert isinstance(result["timestamp"], str)
        assert "T" in result["timestamp"]  # ISO-8601

    def test_cli_args_default_empty(self):
        cfg = load_config()
        result = serialize_run_config(cfg, seed=42)
        assert result["cli_args"] == {}

    def test_cli_args_passed(self):
        cfg = load_config()
        result = serialize_run_config(cfg, seed=42, cli_args={"count": 10})
        assert result["cli_args"] == {"count": 10}

    def test_has_version(self):
        cfg = load_config()
        result = serialize_run_config(cfg, seed=42)
        assert isinstance(result["version"], str)


# ===================================================================
# embed_config_in_output
# ===================================================================


class TestEmbedConfigInOutput:
    def test_returns_list(self):
        records = [_make_record("a"), _make_record("b")]
        result = embed_config_in_output(records, {"seed": 42})
        assert isinstance(result, list)

    def test_same_length(self):
        records = [_make_record(f"c{i}") for i in range(3)]
        result = embed_config_in_output(records, {"seed": 1})
        assert len(result) == 3

    def test_has_run_config(self):
        records = [_make_record()]
        result = embed_config_in_output(records, {"seed": 42, "model": "x"})
        assert "run_config" in result[0].metadata
        assert result[0].metadata["run_config"]["seed"] == 42

    def test_does_not_mutate_original(self):
        records = [_make_record()]
        embed_config_in_output(records, {"seed": 42})
        assert "run_config" not in records[0].metadata

    def test_config_matches(self):
        rc = {"seed": 7, "note": "test"}
        result = embed_config_in_output([_make_record()], rc)
        assert result[0].metadata["run_config"] == rc


# ===================================================================
# load_config_from_output
# ===================================================================


class TestLoadConfigFromOutput:
    def test_from_header(self, tmp_path: Path):
        p = tmp_path / "out.jsonl"
        w = JSONLWriter(p)
        w.write_header({"run_config": {"seed": 42, "model": "gpt-4o"}})
        w.write_record(_make_record())
        loaded = load_config_from_output(p)
        assert loaded["seed"] == 42

    def test_from_record_metadata(self, tmp_path: Path):
        p = tmp_path / "out.jsonl"
        rec = _make_record()
        rec.metadata["run_config"] = {"seed": 99}
        JSONLWriter(p).write_record(rec)
        loaded = load_config_from_output(p)
        assert loaded["seed"] == 99

    def test_raises_when_no_config(self, tmp_path: Path):
        p = tmp_path / "out.jsonl"
        JSONLWriter(p).write_record(_make_record())
        with pytest.raises(ValueError, match="No run config"):
            load_config_from_output(p)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config_from_output("/tmp/nonexistent_xyz.jsonl")

    def test_returns_correct_dict(self, tmp_path: Path):
        p = tmp_path / "out.jsonl"
        rc = {"seed": 55, "version": "0.1.0"}
        w = JSONLWriter(p)
        w.write_header({"run_config": rc})
        w.write_record(_make_record())
        loaded = load_config_from_output(p)
        assert loaded == rc


# ===================================================================
# ensure_reproducibility
# ===================================================================


class TestEnsureReproducibility:
    def test_sets_seed(self):
        ensure_reproducibility(42)
        # No exception means success

    def test_same_seed_same_values(self):
        ensure_reproducibility(42)
        a = [random.random() for _ in range(5)]
        na = np.random.default_rng(42).random(5).tolist()

        ensure_reproducibility(42)
        b = [random.random() for _ in range(5)]
        nb = np.random.default_rng(42).random(5).tolist()

        assert a == b
        assert na == nb

    def test_different_seeds(self):
        ensure_reproducibility(42)
        a = random.random()
        ensure_reproducibility(99)
        b = random.random()
        assert a != b


# ===================================================================
# compare_configs
# ===================================================================


class TestCompareConfigs:
    def test_identical(self):
        a = {"seed": 42, "config": {"model": "gpt-4o"}}
        b = {"seed": 42, "config": {"model": "gpt-4o"}}
        assert compare_configs(a, b) == []

    def test_different_seed(self):
        a = {"seed": 42}
        b = {"seed": 99}
        diffs = compare_configs(a, b)
        assert len(diffs) == 1
        assert "seed" in diffs[0]

    def test_different_config(self):
        a = {"config": {"model": "gpt-4o"}}
        b = {"config": {"model": "gpt-3.5"}}
        diffs = compare_configs(a, b)
        assert len(diffs) == 1

    def test_timestamp_skipped(self):
        a = {"seed": 42, "timestamp": "2024-01-01"}
        b = {"seed": 42, "timestamp": "2025-12-31"}
        assert compare_configs(a, b) == []

    def test_multiple_diffs(self):
        a = {"seed": 42, "config": "x", "version": "1"}
        b = {"seed": 99, "config": "y", "version": "2"}
        diffs = compare_configs(a, b)
        assert len(diffs) == 3


# ===================================================================
# Round-trip integration
# ===================================================================


class TestRoundTrip:
    def test_full_round_trip(self, tmp_path: Path):
        cfg = load_config()
        rc = serialize_run_config(cfg, seed=42, cli_args={"count": 5})
        records = [_make_record(f"c{i}") for i in range(3)]
        embedded = embed_config_in_output(records, rc)

        p = tmp_path / "rt.jsonl"
        w = JSONLWriter(p)
        w.write_header({"run_config": rc})
        w.write_batch(embedded)

        loaded = load_config_from_output(p)
        assert loaded["seed"] == 42
        assert loaded["cli_args"] == {"count": 5}

    def test_config_survives(self, tmp_path: Path):
        cfg = load_config()
        rc = serialize_run_config(cfg, seed=77)
        p = tmp_path / "rt.jsonl"
        w = JSONLWriter(p)
        w.write_header({"run_config": rc})
        w.write_record(_make_record())

        loaded = load_config_from_output(p)
        assert loaded["seed"] == 77
        assert "config" in loaded

    def test_header_has_run_config(self, tmp_path: Path):
        p = tmp_path / "rt.jsonl"
        w = JSONLWriter(p)
        w.write_header({"run_config": {"seed": 1}})
        w.write_record(_make_record())
        meta = JSONLReader(p).read_metadata()
        assert meta is not None
        assert "run_config" in meta

    def test_record_has_run_config(self, tmp_path: Path):
        rc = {"seed": 42}
        records = embed_config_in_output([_make_record()], rc)
        p = tmp_path / "rt.jsonl"
        JSONLWriter(p).write_batch(records)
        loaded = JSONLReader(p).read_all()
        assert loaded[0].metadata.get("run_config") == rc


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    def test_empty_cli_args(self):
        cfg = load_config()
        rc = serialize_run_config(cfg, seed=42, cli_args=None)
        assert rc["cli_args"] == {}

    def test_custom_config(self):
        cfg = load_config()
        rc = serialize_run_config(cfg, seed=123, cli_args={"model": "custom", "verbose": True})
        assert rc["seed"] == 123
        assert rc["cli_args"]["model"] == "custom"

    def test_large_config(self):
        cfg = load_config()
        big_args = {f"key_{i}": f"val_{i}" for i in range(100)}
        rc = serialize_run_config(cfg, seed=42, cli_args=big_args)
        assert len(rc["cli_args"]) == 100
