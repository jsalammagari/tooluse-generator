"""Streaming JSONL I/O for conversation records.

:class:`JSONLWriter` appends :class:`ConversationRecord` objects to a
file one line at a time.  :class:`JSONLReader` reads them back, either
all at once or via a memory-efficient generator.

An optional metadata header (first line with ``__metadata__``) embeds
the run configuration for reproducibility.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tooluse_gen.core.output_models import ConversationRecord
from tooluse_gen.utils.logging import get_logger

logger = get_logger("core.jsonl_io")


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class JSONLWriter:
    """Streaming JSONL writer for conversation records."""

    def __init__(self, output_path: Path | str) -> None:
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._has_header: bool = False
        self._count: int = 0
        self._logger = logger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_record(self, record: ConversationRecord) -> None:
        """Append a single record."""
        with open(self._path, "a") as fh:
            fh.write(record.to_jsonl() + "\n")
        self._count += 1
        self._logger.debug("Wrote record %s", record.conversation_id)

    def write_batch(self, records: list[ConversationRecord]) -> None:
        """Append multiple records in one open/close cycle."""
        with open(self._path, "a") as fh:
            for rec in records:
                fh.write(rec.to_jsonl() + "\n")
        self._count += len(records)
        self._logger.info("Wrote batch of %d records", len(records))

    def write_header(self, metadata: dict[str, Any]) -> None:
        """Write a metadata header as the first line.

        Must be called *before* any records are written.
        """
        if self._count > 0:
            self._logger.warning(
                "Header skipped — %d records already written", self._count
            )
            return
        if self._has_header:
            self._logger.warning("Header already written, skipping")
            return

        line = json.dumps({"__metadata__": True, **metadata}, default=str)
        with open(self._path, "a") as fh:
            fh.write(line + "\n")
        self._has_header = True
        self._logger.debug("Wrote metadata header")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Number of records written (excluding header)."""
        return self._count

    @property
    def path(self) -> Path:
        """Output file path."""
        return self._path


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class JSONLReader:
    """Streaming JSONL reader for conversation records."""

    def __init__(self, input_path: Path | str) -> None:
        self._path = Path(input_path)
        if not self._path.exists():
            raise FileNotFoundError(f"File not found: {self._path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_all(self) -> list[ConversationRecord]:
        """Read all records, skipping the metadata header if present."""
        return list(self.read_iterator())

    def read_iterator(self) -> Iterator[ConversationRecord]:
        """Yield records one at a time (memory-efficient)."""
        with open(self._path) as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                data = json.loads(stripped)
                if data.get("__metadata__"):
                    continue
                yield ConversationRecord.model_validate(data)

    def read_metadata(self) -> dict[str, Any] | None:
        """Return the metadata header, or ``None`` if absent."""
        with open(self._path) as fh:
            first = fh.readline().strip()
        if not first:
            return None
        try:
            data = json.loads(first)
        except json.JSONDecodeError:
            return None
        if not data.get("__metadata__"):
            return None
        result = dict(data)
        result.pop("__metadata__", None)
        return result

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        """Input file path."""
        return self._path

    @property
    def record_count(self) -> int:
        """Count records (excluding metadata header)."""
        count = 0
        with open(self._path) as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if not data.get("__metadata__"):
                    count += 1
        return count
