"""Tests for JSONL Writer & Reader (Task 50)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tooluse_gen.core.jsonl_io import JSONLReader, JSONLWriter
from tooluse_gen.core.output_models import ConversationRecord

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(cid: str = "c1") -> ConversationRecord:
    return ConversationRecord(
        conversation_id=cid,
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        judge_scores={
            "tool_correctness": 4,
            "argument_grounding": 3,
            "task_completion": 5,
            "naturalness": 4,
        },
        metadata={"seed": 42},
    )


# ===================================================================
# JSONLWriter — basic
# ===================================================================


class TestWriterBasic:
    def test_write_record_single_line(self, tmp_path: Path):
        p = tmp_path / "out.jsonl"
        w = JSONLWriter(p)
        w.write_record(_make_record())
        assert p.read_text().count("\n") == 1

    def test_write_record_appends(self, tmp_path: Path):
        p = tmp_path / "out.jsonl"
        w = JSONLWriter(p)
        w.write_record(_make_record("a"))
        w.write_record(_make_record("b"))
        assert p.read_text().count("\n") == 2

    def test_write_batch(self, tmp_path: Path):
        p = tmp_path / "out.jsonl"
        w = JSONLWriter(p)
        w.write_batch([_make_record(f"c{i}") for i in range(5)])
        assert p.read_text().count("\n") == 5

    def test_count_tracks(self, tmp_path: Path):
        p = tmp_path / "out.jsonl"
        w = JSONLWriter(p)
        assert w.count == 0
        w.write_record(_make_record())
        assert w.count == 1
        w.write_batch([_make_record(), _make_record()])
        assert w.count == 3

    def test_creates_parent_dirs(self, tmp_path: Path):
        p = tmp_path / "sub" / "dir" / "out.jsonl"
        w = JSONLWriter(p)
        w.write_record(_make_record())
        assert p.exists()

    def test_path_property(self, tmp_path: Path):
        p = tmp_path / "out.jsonl"
        w = JSONLWriter(p)
        assert w.path == p


# ===================================================================
# JSONLWriter — header
# ===================================================================


class TestWriterHeader:
    def test_write_header(self, tmp_path: Path):
        p = tmp_path / "out.jsonl"
        w = JSONLWriter(p)
        w.write_header({"seed": 42})
        first_line = p.read_text().splitlines()[0]
        data = json.loads(first_line)
        assert data["__metadata__"] is True
        assert data["seed"] == 42

    def test_header_has_metadata_key(self, tmp_path: Path):
        p = tmp_path / "out.jsonl"
        w = JSONLWriter(p)
        w.write_header({"x": 1})
        data = json.loads(p.read_text().splitlines()[0])
        assert "__metadata__" in data

    def test_header_before_records(self, tmp_path: Path):
        p = tmp_path / "out.jsonl"
        w = JSONLWriter(p)
        w.write_header({"seed": 1})
        w.write_record(_make_record())
        lines = p.read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]).get("__metadata__") is True
        assert json.loads(lines[1]).get("conversation_id") == "c1"

    def test_header_after_records_skips(self, tmp_path: Path):
        p = tmp_path / "out.jsonl"
        w = JSONLWriter(p)
        w.write_record(_make_record())
        w.write_header({"seed": 1})  # should be skipped
        lines = p.read_text().splitlines()
        assert len(lines) == 1  # only the record


# ===================================================================
# JSONLReader — basic
# ===================================================================


class TestReaderBasic:
    def test_read_all(self, tmp_path: Path):
        p = tmp_path / "out.jsonl"
        w = JSONLWriter(p)
        w.write_batch([_make_record(f"c{i}") for i in range(3)])
        records = JSONLReader(p).read_all()
        assert isinstance(records, list)
        assert all(isinstance(r, ConversationRecord) for r in records)

    def test_read_all_length(self, tmp_path: Path):
        p = tmp_path / "out.jsonl"
        w = JSONLWriter(p)
        w.write_batch([_make_record(f"c{i}") for i in range(4)])
        assert len(JSONLReader(p).read_all()) == 4

    def test_conversation_ids(self, tmp_path: Path):
        p = tmp_path / "out.jsonl"
        w = JSONLWriter(p)
        w.write_batch([_make_record(f"c{i}") for i in range(3)])
        records = JSONLReader(p).read_all()
        assert [r.conversation_id for r in records] == ["c0", "c1", "c2"]

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            JSONLReader("/tmp/nonexistent_xyz.jsonl")

    def test_path_property(self, tmp_path: Path):
        p = tmp_path / "out.jsonl"
        p.touch()
        assert JSONLReader(p).path == p


# ===================================================================
# JSONLReader — streaming
# ===================================================================


class TestReaderStreaming:
    def test_read_iterator_yields(self, tmp_path: Path):
        p = tmp_path / "out.jsonl"
        w = JSONLWriter(p)
        w.write_batch([_make_record(f"c{i}") for i in range(3)])
        for rec in JSONLReader(p).read_iterator():
            assert isinstance(rec, ConversationRecord)

    def test_read_iterator_count(self, tmp_path: Path):
        p = tmp_path / "out.jsonl"
        w = JSONLWriter(p)
        w.write_batch([_make_record(f"c{i}") for i in range(5)])
        assert sum(1 for _ in JSONLReader(p).read_iterator()) == 5

    def test_iterator_is_generator(self, tmp_path: Path):
        p = tmp_path / "out.jsonl"
        p.write_text('{"conversation_id":"c","messages":[{"role":"user","content":"x"}]}\n')
        it = JSONLReader(p).read_iterator()
        import types

        assert isinstance(it, types.GeneratorType)

    def test_large_batch(self, tmp_path: Path):
        p = tmp_path / "out.jsonl"
        w = JSONLWriter(p)
        w.write_batch([_make_record(f"c{i}") for i in range(60)])
        assert sum(1 for _ in JSONLReader(p).read_iterator()) == 60


# ===================================================================
# JSONLReader — metadata
# ===================================================================


class TestReaderMetadata:
    def test_read_metadata(self, tmp_path: Path):
        p = tmp_path / "out.jsonl"
        w = JSONLWriter(p)
        w.write_header({"seed": 42, "model": "gpt-4o"})
        w.write_record(_make_record())
        meta = JSONLReader(p).read_metadata()
        assert meta is not None
        assert meta["seed"] == 42

    def test_read_metadata_none(self, tmp_path: Path):
        p = tmp_path / "out.jsonl"
        w = JSONLWriter(p)
        w.write_record(_make_record())
        assert JSONLReader(p).read_metadata() is None

    def test_excludes_metadata_key(self, tmp_path: Path):
        p = tmp_path / "out.jsonl"
        w = JSONLWriter(p)
        w.write_header({"seed": 1})
        meta = JSONLReader(p).read_metadata()
        assert meta is not None
        assert "__metadata__" not in meta

    def test_read_all_skips_header(self, tmp_path: Path):
        p = tmp_path / "out.jsonl"
        w = JSONLWriter(p)
        w.write_header({"seed": 1})
        w.write_batch([_make_record(f"c{i}") for i in range(3)])
        records = JSONLReader(p).read_all()
        assert len(records) == 3
        assert all(r.conversation_id.startswith("c") for r in records)


# ===================================================================
# Round-trip
# ===================================================================


class TestRoundTrip:
    def test_write_read_match(self, tmp_path: Path):
        p = tmp_path / "rt.jsonl"
        orig = _make_record("rt1")
        JSONLWriter(p).write_record(orig)
        loaded = JSONLReader(p).read_all()
        assert len(loaded) == 1
        assert loaded[0].conversation_id == "rt1"
        assert loaded[0].judge_scores == orig.judge_scores

    def test_header_preserved(self, tmp_path: Path):
        p = tmp_path / "rt.jsonl"
        w = JSONLWriter(p)
        w.write_header({"seed": 99, "note": "test"})
        w.write_record(_make_record())
        reader = JSONLReader(p)
        meta = reader.read_metadata()
        assert meta is not None and meta["seed"] == 99
        assert len(reader.read_all()) == 1

    def test_batch_round_trip(self, tmp_path: Path):
        p = tmp_path / "rt.jsonl"
        batch = [_make_record(f"b{i}") for i in range(10)]
        JSONLWriter(p).write_batch(batch)
        loaded = JSONLReader(p).read_all()
        assert len(loaded) == 10

    def test_iterator_round_trip(self, tmp_path: Path):
        p = tmp_path / "rt.jsonl"
        JSONLWriter(p).write_batch([_make_record(f"i{i}") for i in range(5)])
        ids = [r.conversation_id for r in JSONLReader(p).read_iterator()]
        assert ids == [f"i{i}" for i in range(5)]

    def test_record_count(self, tmp_path: Path):
        p = tmp_path / "rt.jsonl"
        w = JSONLWriter(p)
        w.write_header({"x": 1})
        w.write_batch([_make_record(f"c{i}") for i in range(7)])
        assert JSONLReader(p).record_count == 7


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    def test_empty_file(self, tmp_path: Path):
        p = tmp_path / "empty.jsonl"
        p.touch()
        reader = JSONLReader(p)
        assert reader.read_all() == []
        assert reader.record_count == 0
        assert reader.read_metadata() is None

    def test_single_record(self, tmp_path: Path):
        p = tmp_path / "single.jsonl"
        JSONLWriter(p).write_record(_make_record("only"))
        records = JSONLReader(p).read_all()
        assert len(records) == 1
        assert records[0].conversation_id == "only"

    def test_header_only(self, tmp_path: Path):
        p = tmp_path / "header.jsonl"
        w = JSONLWriter(p)
        w.write_header({"seed": 1})
        reader = JSONLReader(p)
        assert reader.read_all() == []
        assert reader.record_count == 0
        assert reader.read_metadata() is not None

    def test_large_round_trip(self, tmp_path: Path):
        p = tmp_path / "large.jsonl"
        n = 100
        batch = [_make_record(f"lg{i}") for i in range(n)]
        JSONLWriter(p).write_batch(batch)
        loaded = JSONLReader(p).read_all()
        assert len(loaded) == n
        assert loaded[-1].conversation_id == f"lg{n - 1}"
