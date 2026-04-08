"""Unit tests for registry serialization (Task 15)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tooluse_gen.registry.models import Endpoint, HttpMethod, Parameter, Tool
from tooluse_gen.registry.registry import ToolRegistry
from tooluse_gen.registry.serialization import (
    ChecksumError,
    RegistryJSONSerializer,
    RegistryPickleSerializer,
    SerializationError,
    SerializationFormat,
    SerializationMetadata,
    VersionIncompatibleError,
    get_registry_info,
    load_registry,
    save_registry,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _ep(eid: str = "t/GET/x", tool_id: str = "t") -> Endpoint:
    return Endpoint(
        endpoint_id=eid,
        tool_id=tool_id,
        name=eid,
        path="/x",
        method=HttpMethod.GET,
        parameters=[Parameter(name="q", has_type=True)],
    )


def _tool(tid: str = "t", domain: str = "Weather") -> Tool:
    return Tool(
        tool_id=tid,
        name=tid,
        domain=domain,
        endpoints=[_ep(eid=f"{tid}/GET/x", tool_id=tid)],
        completeness_score=0.7,
    )


def _registry(*tids: str) -> ToolRegistry:
    reg = ToolRegistry()
    for tid in tids or ("t",):
        reg.add_tool(_tool(tid))
    return reg


# ---------------------------------------------------------------------------
# SerializationMetadata
# ---------------------------------------------------------------------------


class TestSerializationMetadata:
    def test_to_dict(self):
        meta = SerializationMetadata(
            version="1.0.0",
            created_at="2026-01-01",
            tool_count=1,
            endpoint_count=2,
            source_info={},
            checksum="abc",
        )
        d = meta.to_dict()
        assert d["version"] == "1.0.0"
        assert d["tool_count"] == 1


# ---------------------------------------------------------------------------
# JSON Serializer
# ---------------------------------------------------------------------------


class TestJSONSerializer:
    def test_serialize_deserialize_round_trip(self, tmp_path: Path):
        reg = _registry("a", "b")
        ser = RegistryJSONSerializer()
        meta = ser.serialize(reg, tmp_path / "reg.json")
        assert meta.tool_count == 2
        assert meta.endpoint_count == 2
        assert (tmp_path / "reg.json").exists()

        reg2, meta2 = ser.deserialize(tmp_path / "reg.json")
        assert len(reg2) == 2
        assert reg2.has_tool("a")
        assert reg2.has_tool("b")
        assert meta2.version == "1.0.0"

    def test_pretty_vs_compact(self, tmp_path: Path):
        reg = _registry()
        ser = RegistryJSONSerializer()
        ser.serialize(reg, tmp_path / "pretty.json", pretty=True)
        ser.serialize(reg, tmp_path / "compact.json", pretty=False)
        pretty_size = (tmp_path / "pretty.json").stat().st_size
        compact_size = (tmp_path / "compact.json").stat().st_size
        assert pretty_size > compact_size

    def test_include_raw_schemas(self, tmp_path: Path):
        t = _tool()
        t.raw_schema = {"original": True}
        reg = ToolRegistry()
        reg.add_tool(t)
        ser = RegistryJSONSerializer()

        ser.serialize(reg, tmp_path / "no_raw.json", include_raw_schemas=False)
        data_no = json.loads((tmp_path / "no_raw.json").read_text())
        assert "raw_schema" not in data_no["registry"]["tools"][0]

        ser.serialize(reg, tmp_path / "with_raw.json", include_raw_schemas=True)
        data_yes = json.loads((tmp_path / "with_raw.json").read_text())
        assert data_yes["registry"]["tools"][0].get("raw_schema") == {"original": True}

    def test_checksum_verification(self, tmp_path: Path):
        reg = _registry()
        ser = RegistryJSONSerializer()
        ser.serialize(reg, tmp_path / "reg.json")

        # Tamper with content
        data = json.loads((tmp_path / "reg.json").read_text())
        data["registry"]["tools"][0]["name"] = "TAMPERED"
        (tmp_path / "reg.json").write_text(json.dumps(data))

        with pytest.raises(ChecksumError):
            ser.deserialize(tmp_path / "reg.json")

    def test_version_incompatible(self, tmp_path: Path):
        reg = _registry()
        ser = RegistryJSONSerializer()
        ser.serialize(reg, tmp_path / "reg.json")

        data = json.loads((tmp_path / "reg.json").read_text())
        data["metadata"]["version"] = "2.0.0"
        (tmp_path / "reg.json").write_text(json.dumps(data))

        with pytest.raises(VersionIncompatibleError):
            ser.deserialize(tmp_path / "reg.json")

    def test_missing_file(self):
        ser = RegistryJSONSerializer()
        with pytest.raises(SerializationError):
            ser.deserialize(Path("/nonexistent.json"))

    def test_invalid_json(self, tmp_path: Path):
        (tmp_path / "bad.json").write_text("not json!")
        ser = RegistryJSONSerializer()
        with pytest.raises(SerializationError):
            ser.deserialize(tmp_path / "bad.json")

    def test_missing_keys(self, tmp_path: Path):
        (tmp_path / "empty.json").write_text("{}")
        ser = RegistryJSONSerializer()
        with pytest.raises(SerializationError):
            ser.deserialize(tmp_path / "empty.json")

    def test_creates_parent_dirs(self, tmp_path: Path):
        reg = _registry()
        ser = RegistryJSONSerializer()
        out = tmp_path / "sub" / "dir" / "reg.json"
        ser.serialize(reg, out)
        assert out.exists()


# ---------------------------------------------------------------------------
# Pickle Serializer
# ---------------------------------------------------------------------------


class TestPickleSerializer:
    def test_serialize_deserialize_compressed(self, tmp_path: Path):
        reg = _registry("a", "b")
        ser = RegistryPickleSerializer()
        meta = ser.serialize(reg, tmp_path / "reg.pkl", compress=True)
        assert meta.tool_count == 2
        assert (tmp_path / "reg.pkl").exists()

        reg2, meta2 = ser.deserialize(tmp_path / "reg.pkl")
        assert len(reg2) == 2
        assert reg2.has_tool("a")

    def test_serialize_deserialize_uncompressed(self, tmp_path: Path):
        reg = _registry()
        ser = RegistryPickleSerializer()
        ser.serialize(reg, tmp_path / "reg.pkl", compress=False)
        reg2, _ = ser.deserialize(tmp_path / "reg.pkl")
        assert len(reg2) == 1

    def test_compressed_smaller(self, tmp_path: Path):
        reg = _registry("a", "b", "c")
        ser = RegistryPickleSerializer()
        ser.serialize(reg, tmp_path / "comp.pkl", compress=True)
        ser.serialize(reg, tmp_path / "raw.pkl", compress=False)
        # Compressed should generally be smaller (or equal for tiny data)
        assert (tmp_path / "comp.pkl").stat().st_size <= (tmp_path / "raw.pkl").stat().st_size + 50

    def test_magic_header(self, tmp_path: Path):
        reg = _registry()
        ser = RegistryPickleSerializer()
        ser.serialize(reg, tmp_path / "reg.pkl")
        raw = (tmp_path / "reg.pkl").read_bytes()
        assert raw[:8] == b"TOOLREG\x00"

    def test_checksum_verification(self, tmp_path: Path):
        reg = _registry()
        ser = RegistryPickleSerializer()
        ser.serialize(reg, tmp_path / "reg.pkl", compress=False)

        # Tamper with bytes
        raw = bytearray((tmp_path / "reg.pkl").read_bytes())
        raw[-1] = (raw[-1] + 1) % 256
        (tmp_path / "reg.pkl").write_bytes(bytes(raw))

        with pytest.raises((ChecksumError, SerializationError)):
            ser.deserialize(tmp_path / "reg.pkl")

    def test_missing_file(self):
        ser = RegistryPickleSerializer()
        with pytest.raises(SerializationError):
            ser.deserialize(Path("/nonexistent.pkl"))

    def test_invalid_header(self, tmp_path: Path):
        (tmp_path / "bad.pkl").write_bytes(b"NOT_TOOLREG")
        ser = RegistryPickleSerializer()
        with pytest.raises(SerializationError):
            ser.deserialize(tmp_path / "bad.pkl")

    def test_creates_parent_dirs(self, tmp_path: Path):
        reg = _registry()
        ser = RegistryPickleSerializer()
        out = tmp_path / "deep" / "nested" / "reg.pkl"
        ser.serialize(reg, out)
        assert out.exists()


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------


class TestUnifiedInterface:
    def test_save_load_json(self, tmp_path: Path):
        reg = _registry("a")
        meta = save_registry(reg, tmp_path / "reg.json")
        assert meta.tool_count == 1

        reg2, meta2 = load_registry(tmp_path / "reg.json")
        assert len(reg2) == 1

    def test_save_load_pickle(self, tmp_path: Path):
        reg = _registry("a")
        meta = save_registry(reg, tmp_path / "reg.pkl")
        assert meta.tool_count == 1

        reg2, meta2 = load_registry(tmp_path / "reg.pkl")
        assert len(reg2) == 1

    def test_save_load_pickle_extension(self, tmp_path: Path):
        reg = _registry()
        save_registry(reg, tmp_path / "reg.pickle")
        reg2, _ = load_registry(tmp_path / "reg.pickle")
        assert len(reg2) == 1

    def test_auto_detect_format(self, tmp_path: Path):
        reg = _registry()
        save_registry(reg, tmp_path / "reg.json")
        save_registry(reg, tmp_path / "reg.pkl")

        # load_registry auto-detects by content
        r1, _ = load_registry(tmp_path / "reg.json")
        r2, _ = load_registry(tmp_path / "reg.pkl")
        assert len(r1) == len(r2) == 1

    def test_explicit_format(self, tmp_path: Path):
        reg = _registry()
        save_registry(reg, tmp_path / "data.bin", fmt=SerializationFormat.PICKLE)
        reg2, _ = load_registry(tmp_path / "data.bin")
        assert len(reg2) == 1

    def test_unknown_extension(self, tmp_path: Path):
        with pytest.raises(SerializationError):
            save_registry(_registry(), tmp_path / "reg.xyz")

    def test_load_missing_file(self):
        with pytest.raises(SerializationError):
            load_registry(Path("/nonexistent.file"))


# ---------------------------------------------------------------------------
# get_registry_info
# ---------------------------------------------------------------------------


class TestGetRegistryInfo:
    def test_json(self, tmp_path: Path):
        reg = _registry("a", "b")
        save_registry(reg, tmp_path / "reg.json")
        meta = get_registry_info(tmp_path / "reg.json")
        assert meta.tool_count == 2

    def test_pickle(self, tmp_path: Path):
        reg = _registry("a")
        save_registry(reg, tmp_path / "reg.pkl")
        meta = get_registry_info(tmp_path / "reg.pkl")
        assert meta.tool_count == 1

    def test_missing_file(self):
        with pytest.raises(SerializationError):
            get_registry_info(Path("/nope"))


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TestExceptions:
    def test_hierarchy(self):
        assert issubclass(VersionIncompatibleError, SerializationError)
        assert issubclass(ChecksumError, SerializationError)


# ---------------------------------------------------------------------------
# Package exports
# ---------------------------------------------------------------------------


def test_package_exports():
    from tooluse_gen.registry import (
        SerializationFormat,
    )

    assert SerializationFormat.JSON == "json"
    assert SerializationFormat.PICKLE == "pickle"
