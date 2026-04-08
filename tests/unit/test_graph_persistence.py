"""Unit tests for Task 20 — Graph serialization & persistence."""

from __future__ import annotations

import gzip
import json
import pickle
from datetime import datetime
from pathlib import Path

import networkx as nx
import pytest

from tooluse_gen.graph.models import GraphStats
from tooluse_gen.graph.persistence import (
    MAGIC_HEADER,
    GraphChecksumError,
    GraphMetadata,
    GraphSerializationError,
    GraphVersionError,
    get_graph_info,
    load_embeddings,
    load_graph,
    save_embeddings,
    save_graph,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_test_graph() -> nx.DiGraph:
    """Build a small graph mimicking GraphBuilder output."""
    g = nx.DiGraph()
    g.add_node(
        "tool:weather",
        node_type="tool",
        tool_id="weather",
        name="Weather API",
        domain="Weather",
        quality_tier="high",
        endpoint_count=2,
    )
    g.add_node(
        "ep:weather:weather/GET/cur",
        node_type="endpoint",
        endpoint_id="weather/GET/cur",
        tool_id="weather",
        name="Current",
        method="GET",
        path="/current",
        domain="Weather",
    )
    g.add_node(
        "ep:weather:weather/GET/fore",
        node_type="endpoint",
        endpoint_id="weather/GET/fore",
        tool_id="weather",
        name="Forecast",
        method="GET",
        path="/forecast",
        domain="Weather",
    )
    g.add_edge(
        "tool:weather",
        "ep:weather:weather/GET/cur",
        edge_type="same_tool",
        weight=1.0,
    )
    g.add_edge(
        "tool:weather",
        "ep:weather:weather/GET/fore",
        edge_type="same_tool",
        weight=1.0,
    )
    g.add_edge(
        "ep:weather:weather/GET/cur",
        "ep:weather:weather/GET/fore",
        edge_type="same_domain",
        weight=0.9,
    )
    g.graph["stats"] = GraphStats(
        tool_node_count=1,
        endpoint_node_count=2,
        edge_counts={"same_tool": 2, "same_domain": 1},
        density=0.5,
        connected_components=1,
    )
    return g


def _write_fake_graph_file(
    path: Path,
    *,
    version: str = "1.0.0",
    compress: bool = True,
    tamper_pickle: bool = False,
) -> None:
    """Write a graph file with controllable version/corruption for testing."""
    g = nx.DiGraph()
    g.add_node("n1", node_type="tool")
    pickled = pickle.dumps(g, protocol=pickle.HIGHEST_PROTOCOL)

    if tamper_pickle:
        pickled = pickled + b"TAMPERED"

    import hashlib

    checksum = hashlib.sha256(pickled).hexdigest()[:16]
    if tamper_pickle:
        checksum = "wrong_checksum__"

    meta = GraphMetadata(
        version=version,
        created_at=datetime.now().isoformat(),
        node_count=1,
        edge_count=0,
        tool_node_count=1,
        endpoint_node_count=0,
        edge_type_counts={},
        density=0.0,
        connected_components=1,
        config={},
        checksum=checksum,
    )
    meta_bytes = json.dumps(meta.to_dict()).encode("utf-8")
    meta_len = len(meta_bytes).to_bytes(4, "big")
    inner = meta_len + meta_bytes + pickled

    if compress:
        blob = MAGIC_HEADER + b"\x01" + gzip.compress(inner)
    else:
        blob = MAGIC_HEADER + b"\x00" + inner

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)


# ===========================================================================
# save_graph / load_graph round-trip
# ===========================================================================


class TestGraphRoundTrip:
    def test_node_count_preserved(self, tmp_path: Path) -> None:
        g = _build_test_graph()
        p = tmp_path / "g.pkl"
        save_graph(g, p)
        loaded, _meta = load_graph(p)
        assert loaded.number_of_nodes() == g.number_of_nodes()

    def test_edge_count_preserved(self, tmp_path: Path) -> None:
        g = _build_test_graph()
        p = tmp_path / "g.pkl"
        save_graph(g, p)
        loaded, _meta = load_graph(p)
        assert loaded.number_of_edges() == g.number_of_edges()

    def test_node_attributes_preserved(self, tmp_path: Path) -> None:
        g = _build_test_graph()
        p = tmp_path / "g.pkl"
        save_graph(g, p)
        loaded, _meta = load_graph(p)
        assert loaded.nodes["tool:weather"]["name"] == "Weather API"
        assert loaded.nodes["ep:weather:weather/GET/cur"]["method"] == "GET"

    def test_edge_attributes_preserved(self, tmp_path: Path) -> None:
        g = _build_test_graph()
        p = tmp_path / "g.pkl"
        save_graph(g, p)
        loaded, _meta = load_graph(p)
        edge_data = loaded["tool:weather"]["ep:weather:weather/GET/cur"]
        assert edge_data["edge_type"] == "same_tool"
        assert edge_data["weight"] == 1.0

    def test_stats_preserved(self, tmp_path: Path) -> None:
        g = _build_test_graph()
        p = tmp_path / "g.pkl"
        save_graph(g, p)
        loaded, _meta = load_graph(p)
        stats = loaded.graph.get("stats")
        assert isinstance(stats, GraphStats)
        assert stats.tool_node_count == 1
        assert stats.endpoint_node_count == 2

    def test_compressed_round_trip(self, tmp_path: Path) -> None:
        g = _build_test_graph()
        p = tmp_path / "g.pkl"
        save_graph(g, p, compress=True)
        loaded, _meta = load_graph(p)
        assert loaded.number_of_nodes() == 3

    def test_uncompressed_round_trip(self, tmp_path: Path) -> None:
        g = _build_test_graph()
        p = tmp_path / "g.pkl"
        save_graph(g, p, compress=False)
        loaded, _meta = load_graph(p)
        assert loaded.number_of_nodes() == 3


# ===========================================================================
# save_graph metadata
# ===========================================================================


class TestSaveGraphMetadata:
    def test_node_count(self, tmp_path: Path) -> None:
        g = _build_test_graph()
        meta = save_graph(g, tmp_path / "g.pkl")
        assert meta.node_count == 3

    def test_edge_count(self, tmp_path: Path) -> None:
        g = _build_test_graph()
        meta = save_graph(g, tmp_path / "g.pkl")
        assert meta.edge_count == 3

    def test_tool_node_count(self, tmp_path: Path) -> None:
        g = _build_test_graph()
        meta = save_graph(g, tmp_path / "g.pkl")
        assert meta.tool_node_count == 1

    def test_endpoint_node_count(self, tmp_path: Path) -> None:
        g = _build_test_graph()
        meta = save_graph(g, tmp_path / "g.pkl")
        assert meta.endpoint_node_count == 2

    def test_edge_type_counts(self, tmp_path: Path) -> None:
        g = _build_test_graph()
        meta = save_graph(g, tmp_path / "g.pkl")
        assert meta.edge_type_counts == {"same_tool": 2, "same_domain": 1}

    def test_checksum_non_empty(self, tmp_path: Path) -> None:
        g = _build_test_graph()
        meta = save_graph(g, tmp_path / "g.pkl")
        assert len(meta.checksum) == 16

    def test_created_at_parseable(self, tmp_path: Path) -> None:
        g = _build_test_graph()
        meta = save_graph(g, tmp_path / "g.pkl")
        dt = datetime.fromisoformat(meta.created_at)
        assert dt.year >= 2024

    def test_version(self, tmp_path: Path) -> None:
        g = _build_test_graph()
        meta = save_graph(g, tmp_path / "g.pkl")
        assert meta.version == "1.0.0"


# ===========================================================================
# save_graph creates parent dirs
# ===========================================================================


class TestSaveGraphDirs:
    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        g = _build_test_graph()
        p = tmp_path / "a" / "b" / "c" / "g.pkl"
        save_graph(g, p)
        assert p.exists()


# ===========================================================================
# load_graph error handling
# ===========================================================================


class TestLoadGraphErrors:
    def test_file_not_found(self) -> None:
        with pytest.raises(GraphSerializationError, match="File not found"):
            load_graph("/tmp/nonexistent_graph_xyz_test.pkl")

    def test_invalid_header(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.pkl"
        p.write_bytes(b"NOT A GRAPH FILE")
        with pytest.raises(GraphSerializationError, match="header"):
            load_graph(p)

    def test_truncated_file(self, tmp_path: Path) -> None:
        p = tmp_path / "trunc.pkl"
        # Write magic header + compress flag but no body
        p.write_bytes(MAGIC_HEADER + b"\x00")
        with pytest.raises(GraphSerializationError, match="Truncated"):
            load_graph(p)

    def test_tampered_data_checksum_error(self, tmp_path: Path) -> None:
        p = tmp_path / "tampered.pkl"
        _write_fake_graph_file(p, tamper_pickle=True, compress=False)
        with pytest.raises(GraphChecksumError):
            load_graph(p)


# ===========================================================================
# Version compatibility
# ===========================================================================


class TestVersionCompat:
    def test_version_1_0_0(self, tmp_path: Path) -> None:
        p = tmp_path / "v1.pkl"
        _write_fake_graph_file(p, version="1.0.0", compress=False)
        loaded, meta = load_graph(p)
        assert meta.version == "1.0.0"

    def test_version_1_99_0(self, tmp_path: Path) -> None:
        p = tmp_path / "v199.pkl"
        _write_fake_graph_file(p, version="1.99.0", compress=False)
        loaded, meta = load_graph(p)
        assert meta.version == "1.99.0"

    def test_version_2_0_0_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "v2.pkl"
        _write_fake_graph_file(p, version="2.0.0", compress=False)
        with pytest.raises(GraphVersionError):
            load_graph(p)

    def test_invalid_version_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "vbad.pkl"
        _write_fake_graph_file(p, version="abc", compress=False)
        with pytest.raises(GraphVersionError):
            load_graph(p)


# ===========================================================================
# get_graph_info
# ===========================================================================


class TestGetGraphInfo:
    def test_returns_metadata(self, tmp_path: Path) -> None:
        g = _build_test_graph()
        p = tmp_path / "g.pkl"
        save_graph(g, p)
        info = get_graph_info(p)
        assert isinstance(info, GraphMetadata)

    def test_matches_save_metadata(self, tmp_path: Path) -> None:
        g = _build_test_graph()
        p = tmp_path / "g.pkl"
        meta = save_graph(g, p)
        info = get_graph_info(p)
        assert info.node_count == meta.node_count
        assert info.edge_count == meta.edge_count
        assert info.checksum == meta.checksum

    def test_compressed(self, tmp_path: Path) -> None:
        g = _build_test_graph()
        p = tmp_path / "g.pkl"
        save_graph(g, p, compress=True)
        info = get_graph_info(p)
        assert info.node_count == 3

    def test_uncompressed(self, tmp_path: Path) -> None:
        g = _build_test_graph()
        p = tmp_path / "g.pkl"
        save_graph(g, p, compress=False)
        info = get_graph_info(p)
        assert info.node_count == 3

    def test_file_not_found(self) -> None:
        with pytest.raises(GraphSerializationError, match="File not found"):
            get_graph_info("/tmp/nonexistent_info_xyz_test.pkl")


# ===========================================================================
# save_embeddings / load_embeddings
# ===========================================================================


class TestEmbeddingPersistence:
    def test_round_trip(self, tmp_path: Path) -> None:
        data = {"a": [0.1, 0.2], "b": [0.3, 0.4]}
        p = tmp_path / "emb.joblib"
        save_embeddings(data, p)
        loaded = load_embeddings(p)
        assert loaded == data

    def test_load_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_embeddings(tmp_path / "missing.joblib")

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        p = tmp_path / "x" / "y" / "emb.joblib"
        save_embeddings({"k": [1.0]}, p)
        assert p.exists()

    def test_large_dict(self, tmp_path: Path) -> None:
        data = {f"node_{i}": [float(i)] * 128 for i in range(500)}
        p = tmp_path / "large.joblib"
        save_embeddings(data, p)
        loaded = load_embeddings(p)
        assert len(loaded) == 500
        assert loaded["node_0"] == [0.0] * 128

    def test_empty_dict(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.joblib"
        save_embeddings({}, p)
        loaded = load_embeddings(p)
        assert loaded == {}


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_empty_graph_round_trip(self, tmp_path: Path) -> None:
        g = nx.DiGraph()
        p = tmp_path / "empty.pkl"
        meta = save_graph(g, p)
        assert meta.node_count == 0
        assert meta.edge_count == 0
        loaded, loaded_meta = load_graph(p)
        assert loaded.number_of_nodes() == 0
        assert loaded.number_of_edges() == 0

    def test_graph_without_stats(self, tmp_path: Path) -> None:
        g = nx.DiGraph()
        g.add_node("tool:t", node_type="tool")
        g.add_node("ep:t:e", node_type="endpoint")
        g.add_edge("tool:t", "ep:t:e", edge_type="same_tool", weight=1.0)
        # No stats attached
        p = tmp_path / "nostats.pkl"
        meta = save_graph(g, p)
        assert meta.node_count == 2
        assert meta.tool_node_count == 1
        assert meta.endpoint_node_count == 1
        loaded, _meta = load_graph(p)
        assert loaded.number_of_nodes() == 2

    def test_metadata_to_dict(self) -> None:
        meta = GraphMetadata(
            version="1.0.0",
            created_at="2024-01-01T00:00:00",
            node_count=5,
            edge_count=3,
            tool_node_count=2,
            endpoint_node_count=3,
            edge_type_counts={"same_tool": 3},
            density=0.3,
            connected_components=1,
            config={},
            checksum="abc123",
        )
        d = meta.to_dict()
        assert d["version"] == "1.0.0"
        assert d["node_count"] == 5
