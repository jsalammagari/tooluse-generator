"""Registry serialization and deserialization.

Two formats are supported:

* **JSON** — human-readable, useful for debugging and inspection.
* **Pickle** — fast binary format with optional gzip compression.

Both formats embed :class:`SerializationMetadata` (version, timestamp,
counts, checksum) so the loader can verify integrity and compatibility
before fully deserializing.

Unified entry points :func:`save_registry` and :func:`load_registry`
auto-detect format from the file extension.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from tooluse_gen.registry.models import Tool
from tooluse_gen.registry.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SerializationError(Exception):
    """Base serialization exception."""


class VersionIncompatibleError(SerializationError):
    """File version is incompatible with this code."""


class ChecksumError(SerializationError):
    """Checksum verification failed."""


# ---------------------------------------------------------------------------
# Format enum & metadata
# ---------------------------------------------------------------------------


class SerializationFormat(str, Enum):
    JSON = "json"
    PICKLE = "pickle"


@dataclass
class SerializationMetadata:
    """Metadata stored alongside the serialized registry."""

    version: str
    created_at: str  # ISO-8601
    tool_count: int
    endpoint_count: int
    source_info: dict[str, Any]
    checksum: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# JSON serializer
# ---------------------------------------------------------------------------

_COMPATIBLE_MAJOR = 1  # accept any 1.x.x


class RegistryJSONSerializer:
    """JSON serialization for debugging and inspection."""

    CURRENT_VERSION = "1.0.0"

    def serialize(
        self,
        registry: ToolRegistry,
        output_path: Path,
        pretty: bool = True,
        include_raw_schemas: bool = False,
    ) -> SerializationMetadata:
        """Serialize *registry* to a JSON file at *output_path*."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = self._registry_to_dict(registry, include_raw=include_raw_schemas)
        checksum = self._compute_checksum(payload)

        meta = SerializationMetadata(
            version=self.CURRENT_VERSION,
            created_at=datetime.now(timezone.utc).isoformat(),
            tool_count=len(registry),
            endpoint_count=sum(len(t.endpoints) for t in registry.tools()),
            source_info=registry._source_info,
            checksum=checksum,
        )

        envelope: dict[str, Any] = {
            "metadata": meta.to_dict(),
            "registry": payload,
        }

        indent = 2 if pretty else None
        output_path.write_text(json.dumps(envelope, indent=indent, default=str), encoding="utf-8")
        return meta

    def deserialize(self, input_path: Path) -> tuple[ToolRegistry, SerializationMetadata]:
        """Deserialize a registry from a JSON file."""
        input_path = Path(input_path)
        if not input_path.exists():
            raise SerializationError(f"File not found: {input_path}")

        try:
            raw = json.loads(input_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SerializationError(f"Invalid JSON: {exc}") from exc

        if "metadata" not in raw or "registry" not in raw:
            raise SerializationError("Missing 'metadata' or 'registry' keys.")

        meta_dict = raw["metadata"]
        meta = SerializationMetadata(**meta_dict)
        self._check_version_compatibility(meta.version)

        payload = raw["registry"]
        if not self._verify_checksum(payload, meta.checksum):
            raise ChecksumError("JSON checksum verification failed.")

        registry = self._dict_to_registry(payload)
        registry._source_info = meta.source_info
        return registry, meta

    # -- helpers ------------------------------------------------------------

    def _registry_to_dict(self, registry: ToolRegistry, include_raw: bool) -> dict[str, Any]:
        tools_list = []
        for tool in registry.tools():
            d = tool.model_dump()
            if include_raw:
                # raw_schema is exclude=True on the model, so add it back
                if tool.raw_schema is not None:
                    d["raw_schema"] = tool.raw_schema
                for ep_dict, ep_obj in zip(d.get("endpoints", []), tool.endpoints, strict=False):
                    if ep_obj.raw_definition is not None:
                        ep_dict["raw_definition"] = ep_obj.raw_definition
                    for p_dict, p_obj in zip(ep_dict.get("parameters", []), ep_obj.parameters, strict=False):
                        if p_obj.raw_definition is not None:
                            p_dict["raw_definition"] = p_obj.raw_definition
            tools_list.append(d)
        return {"tools": tools_list}

    def _dict_to_registry(self, data: dict[str, Any]) -> ToolRegistry:
        registry = ToolRegistry()
        for raw_tool in data.get("tools", []):
            tool = Tool.model_validate(raw_tool)
            registry.add_tools([tool])
        return registry

    def _compute_checksum(self, data: dict[str, Any]) -> str:
        serialized = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()[:16]

    def _verify_checksum(self, data: dict[str, Any], expected: str) -> bool:
        return self._compute_checksum(data) == expected

    def _check_version_compatibility(self, version: str) -> None:
        try:
            major = int(version.split(".")[0])
        except (ValueError, IndexError) as exc:
            raise VersionIncompatibleError(f"Invalid version: {version}") from exc
        if major != _COMPATIBLE_MAJOR:
            raise VersionIncompatibleError(
                f"Version {version} not compatible (expected major={_COMPATIBLE_MAJOR})."
            )


# ---------------------------------------------------------------------------
# Pickle serializer
# ---------------------------------------------------------------------------


class RegistryPickleSerializer:
    """Fast binary serialization with optional gzip compression."""

    CURRENT_VERSION = "1.0.0"
    MAGIC_HEADER = b"TOOLREG\x00"

    def serialize(
        self,
        registry: ToolRegistry,
        output_path: Path,
        compress: bool = True,
    ) -> SerializationMetadata:
        """Serialize *registry* to a binary pickle file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = registry.to_dict()
        pickled = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        checksum = hashlib.sha256(pickled).hexdigest()[:16]

        meta = SerializationMetadata(
            version=self.CURRENT_VERSION,
            created_at=datetime.now(timezone.utc).isoformat(),
            tool_count=len(registry),
            endpoint_count=sum(len(t.endpoints) for t in registry.tools()),
            source_info=registry._source_info,
            checksum=checksum,
        )

        meta_bytes = json.dumps(meta.to_dict()).encode("utf-8")
        meta_len = len(meta_bytes).to_bytes(4, "big")

        blob = self.MAGIC_HEADER + meta_len + meta_bytes + pickled

        if compress:
            blob = self.MAGIC_HEADER + b"\x01" + gzip.compress(blob[len(self.MAGIC_HEADER) :])
        else:
            blob = self.MAGIC_HEADER + b"\x00" + blob[len(self.MAGIC_HEADER) :]

        output_path.write_bytes(blob)
        return meta

    def deserialize(self, input_path: Path) -> tuple[ToolRegistry, SerializationMetadata]:
        """Deserialize a registry from a binary pickle file."""
        input_path = Path(input_path)
        if not input_path.exists():
            raise SerializationError(f"File not found: {input_path}")

        raw = input_path.read_bytes()
        header_len = len(self.MAGIC_HEADER)

        if len(raw) < header_len + 1 or raw[:header_len] != self.MAGIC_HEADER:
            raise SerializationError("Invalid file: missing TOOLREG header.")

        compressed = raw[header_len] == 1
        body = raw[header_len + 1 :]

        if compressed:
            try:
                body = gzip.decompress(body)
            except Exception as exc:
                raise SerializationError(f"Decompression failed: {exc}") from exc

        # body now has: meta_len(4) + meta_bytes + pickled
        if len(body) < 4:
            raise SerializationError("Truncated file.")

        meta_len = int.from_bytes(body[:4], "big")
        if len(body) < 4 + meta_len:
            raise SerializationError("Truncated metadata.")

        meta_bytes = body[4 : 4 + meta_len]
        pickled = body[4 + meta_len :]

        meta_dict = json.loads(meta_bytes.decode("utf-8"))
        meta = SerializationMetadata(**meta_dict)

        try:
            major = int(meta.version.split(".")[0])
        except (ValueError, IndexError) as exc:
            raise VersionIncompatibleError(f"Invalid version: {meta.version}") from exc
        if major != _COMPATIBLE_MAJOR:
            raise VersionIncompatibleError(
                f"Version {meta.version} not compatible (expected major={_COMPATIBLE_MAJOR})."
            )

        checksum = hashlib.sha256(pickled).hexdigest()[:16]
        if checksum != meta.checksum:
            raise ChecksumError("Pickle checksum verification failed.")

        payload = pickle.loads(pickled)  # noqa: S301
        registry = ToolRegistry.from_dict(payload)
        registry._source_info = meta.source_info
        return registry, meta


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

_EXT_TO_FORMAT: dict[str, SerializationFormat] = {
    ".json": SerializationFormat.JSON,
    ".pkl": SerializationFormat.PICKLE,
    ".pickle": SerializationFormat.PICKLE,
}


def save_registry(
    registry: ToolRegistry,
    output_path: Path | str,
    fmt: SerializationFormat | None = None,
    **kwargs: Any,
) -> SerializationMetadata:
    """Save *registry* to *output_path*.

    *fmt* is auto-detected from the file extension when ``None``.
    """
    path = Path(output_path)
    if fmt is None:
        fmt = _EXT_TO_FORMAT.get(path.suffix.lower())
        if fmt is None:
            raise SerializationError(f"Cannot detect format from extension '{path.suffix}'.")

    if fmt == SerializationFormat.JSON:
        return RegistryJSONSerializer().serialize(registry, path, **kwargs)
    return RegistryPickleSerializer().serialize(registry, path, **kwargs)


def load_registry(input_path: Path | str) -> tuple[ToolRegistry, SerializationMetadata]:
    """Load a registry from *input_path* (auto-detects format)."""
    path = Path(input_path)
    if not path.exists():
        raise SerializationError(f"File not found: {path}")

    # Try magic header first
    raw = path.read_bytes()
    if raw[: len(RegistryPickleSerializer.MAGIC_HEADER)] == RegistryPickleSerializer.MAGIC_HEADER:
        return RegistryPickleSerializer().deserialize(path)

    # Fall back to JSON
    return RegistryJSONSerializer().deserialize(path)


def get_registry_info(input_path: Path | str) -> SerializationMetadata:
    """Read metadata from a serialized registry without loading tools."""
    path = Path(input_path)
    if not path.exists():
        raise SerializationError(f"File not found: {path}")

    raw = path.read_bytes()
    header = RegistryPickleSerializer.MAGIC_HEADER

    if raw[: len(header)] == header:
        # Pickle format
        compressed = raw[len(header)] == 1
        body = raw[len(header) + 1 :]
        if compressed:
            body = gzip.decompress(body)
        meta_len = int.from_bytes(body[:4], "big")
        meta_bytes = body[4 : 4 + meta_len]
        return SerializationMetadata(**json.loads(meta_bytes.decode("utf-8")))

    # JSON format
    data = json.loads(raw.decode("utf-8"))
    if "metadata" not in data:
        raise SerializationError("Missing 'metadata' key in JSON file.")
    return SerializationMetadata(**data["metadata"])
