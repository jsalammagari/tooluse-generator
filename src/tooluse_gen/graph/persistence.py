"""Graph serialization and persistence.

Provides standalone functions for saving/loading NetworkX graphs and
embedding caches to disk.  The graph binary format mirrors the registry
serializer: a magic header, optional gzip compression, JSON metadata,
and a pickled payload.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import networkx as nx

from tooluse_gen.graph.models import GraphStats
from tooluse_gen.utils.logging import get_logger

logger = get_logger("graph.persistence")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CURRENT_VERSION = "1.0.0"
_COMPATIBLE_MAJOR = 1
MAGIC_HEADER = b"TOOLGRAPH\x00"

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GraphSerializationError(Exception):
    """Base exception for graph serialization errors."""


class GraphVersionError(GraphSerializationError):
    """Incompatible file version."""


class GraphChecksumError(GraphSerializationError):
    """Checksum verification failed."""


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@dataclass
class GraphMetadata:
    """Metadata stored alongside the serialized graph."""

    version: str
    created_at: str
    node_count: int
    edge_count: int
    tool_node_count: int
    endpoint_node_count: int
    edge_type_counts: dict[str, int]
    density: float
    connected_components: int
    config: dict[str, Any] = field(default_factory=dict)
    checksum: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Graph save / load
# ---------------------------------------------------------------------------


def _extract_metadata(graph: nx.DiGraph, checksum: str) -> GraphMetadata:
    """Build :class:`GraphMetadata` from a graph."""
    stats: GraphStats | None = graph.graph.get("stats")

    edge_type_counts: dict[str, int]
    if stats is not None:
        tool_count = stats.tool_node_count
        ep_count = stats.endpoint_node_count
        edge_type_counts = dict(stats.edge_counts)
        density = stats.density
        components = stats.connected_components
    else:
        tool_count = 0
        ep_count = 0
        for _n, data in graph.nodes(data=True):
            nt = data.get("node_type")
            if nt == "tool":
                tool_count += 1
            elif nt == "endpoint":
                ep_count += 1
        edge_type_counts = {}
        for _u, _v, data in graph.edges(data=True):
            et = data.get("edge_type", "unknown")
            edge_type_counts[et] = edge_type_counts.get(et, 0) + 1
        density = float(nx.density(graph)) if graph.number_of_nodes() > 0 else 0.0
        components = (
            nx.number_weakly_connected_components(graph)
            if graph.number_of_nodes() > 0
            else 0
        )

    config_raw = graph.graph.get("config")
    if config_raw is not None and hasattr(config_raw, "model_dump"):
        config_dict: dict[str, Any] = config_raw.model_dump()
    elif isinstance(config_raw, dict):
        config_dict = config_raw
    else:
        config_dict = {}

    return GraphMetadata(
        version=CURRENT_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
        node_count=graph.number_of_nodes(),
        edge_count=graph.number_of_edges(),
        tool_node_count=tool_count,
        endpoint_node_count=ep_count,
        edge_type_counts=edge_type_counts,
        density=density,
        connected_components=components,
        config=config_dict,
        checksum=checksum,
    )


def save_graph(
    graph: nx.DiGraph,
    output_path: Path | str,
    compress: bool = True,
) -> GraphMetadata:
    """Serialize *graph* to a binary file at *output_path*."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pickled = pickle.dumps(graph, protocol=pickle.HIGHEST_PROTOCOL)
    checksum = hashlib.sha256(pickled).hexdigest()[:16]

    meta = _extract_metadata(graph, checksum)
    meta_bytes = json.dumps(meta.to_dict()).encode("utf-8")
    meta_len = len(meta_bytes).to_bytes(4, "big")

    inner = meta_len + meta_bytes + pickled

    if compress:
        blob = MAGIC_HEADER + b"\x01" + gzip.compress(inner)
    else:
        blob = MAGIC_HEADER + b"\x00" + inner

    path.write_bytes(blob)
    logger.info(
        "Saved graph to %s (%d nodes, %d edges)",
        path,
        meta.node_count,
        meta.edge_count,
    )
    return meta


def _parse_file(raw: bytes) -> tuple[GraphMetadata, bytes]:
    """Parse header, metadata, and return (metadata, pickled_bytes)."""
    header_len = len(MAGIC_HEADER)

    if len(raw) < header_len + 1 or raw[:header_len] != MAGIC_HEADER:
        raise GraphSerializationError("Invalid file: missing TOOLGRAPH header.")

    compressed = raw[header_len] == 1
    body = raw[header_len + 1 :]

    if compressed:
        try:
            body = gzip.decompress(body)
        except Exception as exc:
            raise GraphSerializationError(f"Decompression failed: {exc}") from exc

    if len(body) < 4:
        raise GraphSerializationError("Truncated file.")

    meta_len = int.from_bytes(body[:4], "big")
    if len(body) < 4 + meta_len:
        raise GraphSerializationError("Truncated metadata.")

    meta_bytes = body[4 : 4 + meta_len]
    pickled = body[4 + meta_len :]

    meta_dict = json.loads(meta_bytes.decode("utf-8"))
    meta = GraphMetadata(**meta_dict)

    return meta, pickled


def _check_version(version: str) -> None:
    """Raise if *version* is not compatible."""
    try:
        major = int(version.split(".")[0])
    except (ValueError, IndexError) as exc:
        raise GraphVersionError(f"Invalid version: {version}") from exc
    if major != _COMPATIBLE_MAJOR:
        raise GraphVersionError(
            f"Version {version} not compatible (expected major={_COMPATIBLE_MAJOR})."
        )


def load_graph(input_path: Path | str) -> tuple[nx.DiGraph, GraphMetadata]:
    """Deserialize a graph from a binary file."""
    path = Path(input_path)
    if not path.exists():
        raise GraphSerializationError(f"File not found: {path}")

    raw = path.read_bytes()
    meta, pickled = _parse_file(raw)

    _check_version(meta.version)

    checksum = hashlib.sha256(pickled).hexdigest()[:16]
    if checksum != meta.checksum:
        raise GraphChecksumError("Graph checksum verification failed.")

    graph: nx.DiGraph = pickle.loads(pickled)  # noqa: S301
    logger.info(
        "Loaded graph from %s (%d nodes, %d edges)",
        path,
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    return graph, meta


def get_graph_info(input_path: Path | str) -> GraphMetadata:
    """Read metadata from a serialized graph without loading the full graph."""
    path = Path(input_path)
    if not path.exists():
        raise GraphSerializationError(f"File not found: {path}")

    raw = path.read_bytes()
    meta, _pickled = _parse_file(raw)
    return meta


# ---------------------------------------------------------------------------
# Embedding save / load (standalone)
# ---------------------------------------------------------------------------


def save_embeddings(embeddings: dict[str, list[float]], output_path: Path | str) -> None:
    """Save embeddings dict to disk via joblib."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving embeddings to %s", path)
    joblib.dump(embeddings, path)


def load_embeddings(input_path: Path | str) -> dict[str, list[float]]:
    """Load embeddings dict from disk. Raises ``FileNotFoundError`` if missing."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Embedding cache not found: {path}")
    logger.info("Loading embeddings from %s", path)
    return joblib.load(path)  # type: ignore[no-any-return]
