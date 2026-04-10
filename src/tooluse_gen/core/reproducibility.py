"""Reproducibility utilities.

Serialize run configuration, embed it in output records, and extract it
from previous outputs so runs can be exactly reproduced.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tooluse_gen
from tooluse_gen.core.config import AppConfig
from tooluse_gen.core.jsonl_io import JSONLReader
from tooluse_gen.core.output_models import ConversationRecord
from tooluse_gen.utils.logging import get_logger
from tooluse_gen.utils.seeding import set_global_seed

logger = get_logger("core.reproducibility")


def serialize_run_config(
    config: AppConfig,
    seed: int,
    cli_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture the full run configuration as a plain dict."""
    return {
        "config": config.model_dump(),
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cli_args": cli_args or {},
        "version": tooluse_gen.__version__,
    }


def embed_config_in_output(
    records: list[ConversationRecord],
    run_config: dict[str, Any],
) -> list[ConversationRecord]:
    """Return new records with *run_config* embedded in each metadata."""
    result: list[ConversationRecord] = []
    for rec in records:
        copy = rec.model_copy(deep=True)
        copy.metadata["run_config"] = run_config
        result.append(copy)
    return result


def load_config_from_output(jsonl_path: Path | str) -> dict[str, Any]:
    """Extract the run config from a previously written JSONL file."""
    reader = JSONLReader(jsonl_path)

    # Try header first.
    meta = reader.read_metadata()
    if meta is not None and "run_config" in meta:
        return meta["run_config"]  # type: ignore[no-any-return]

    # Fall back to first record's metadata.
    records = reader.read_all()
    if records:
        rc = records[0].metadata.get("run_config")
        if rc is not None:
            return rc  # type: ignore[no-any-return]

    raise ValueError("No run config found in output file")


def ensure_reproducibility(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    set_global_seed(seed)
    logger.info("Seed set to %d for reproducibility", seed)


def compare_configs(
    config_a: dict[str, Any],
    config_b: dict[str, Any],
) -> list[str]:
    """Return human-readable differences between two run configs.

    The ``timestamp`` key is always skipped.
    """
    diffs: list[str] = []
    all_keys = sorted(set(config_a) | set(config_b))
    for key in all_keys:
        if key == "timestamp":
            continue
        val_a = config_a.get(key)
        val_b = config_b.get(key)
        if val_a != val_b:
            diffs.append(f"{key}: {val_a} != {val_b}")
    return diffs
