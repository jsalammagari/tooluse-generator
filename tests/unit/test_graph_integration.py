"""Integration tests for Task 22 — full graph module pipeline.

Exercises: registry → build graph → verify stats → serialize → reload → query.
EmbeddingService is mocked throughout — no real model is loaded.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import networkx as nx
import numpy as np
import pytest

from tooluse_gen.graph import (
    EdgeType,
    EmbeddingService,
    EndpointNode,
    GraphBuilder,
    GraphChecksumError,
    GraphConfig,
    GraphEdge,
    GraphMetadata,
    GraphSerializationError,
    GraphStats,
    GraphVersionError,
    ToolNode,
    build_endpoint_description,
    build_tool_description,
    compute_node_importance,
    get_chainable_endpoints,
    get_connected_endpoints,
    get_domain_endpoints,
    get_endpoints_for_tool,
    get_graph_info,
    get_graph_stats,
    get_neighbors,
    get_tool_for_endpoint,
    load_embeddings,
    load_graph,
    save_embeddings,
    save_graph,
)
from tooluse_gen.registry.models import (
    Endpoint,
    HttpMethod,
    Parameter,
    ParameterType,
    Tool,
)
from tooluse_gen.registry.registry import ToolRegistry

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REAL_SVC = EmbeddingService()


def _build_integration_registry() -> ToolRegistry:
    """Build a registry with 4 tools across 3 domains, 8 endpoints total."""
    registry = ToolRegistry()
    tools = [
        Tool(
            tool_id="weather",
            name="Weather API",
            domain="Weather",
            description="Weather forecasting service.",
            completeness_score=0.85,
            endpoints=[
                Endpoint(
                    endpoint_id="weather/GET/cur",
                    tool_id="weather",
                    name="Current Weather",
                    description="Get current conditions.",
                    method=HttpMethod.GET,
                    path="/current",
                    parameters=[
                        Parameter(name="city", param_type=ParameterType.STRING, required=True),
                        Parameter(name="units", param_type=ParameterType.STRING),
                    ],
                ),
                Endpoint(
                    endpoint_id="weather/GET/fore",
                    tool_id="weather",
                    name="Forecast",
                    description="5-day forecast.",
                    method=HttpMethod.GET,
                    path="/forecast",
                    parameters=[
                        Parameter(
                            name="city", param_type=ParameterType.STRING, required=True
                        ),
                    ],
                ),
            ],
        ),
        Tool(
            tool_id="maps",
            name="Maps API",
            domain="Weather",
            description="Geolocation and mapping.",
            completeness_score=0.6,
            endpoints=[
                Endpoint(
                    endpoint_id="maps/GET/geo",
                    tool_id="maps",
                    name="Geocode",
                    description="Convert address to coordinates.",
                    method=HttpMethod.GET,
                    path="/geocode",
                    parameters=[
                        Parameter(
                            name="address", param_type=ParameterType.STRING, required=True
                        ),
                    ],
                ),
            ],
        ),
        Tool(
            tool_id="stocks",
            name="Stock API",
            domain="Finance",
            description="Stock market data.",
            completeness_score=0.9,
            endpoints=[
                Endpoint(
                    endpoint_id="stocks/GET/quote",
                    tool_id="stocks",
                    name="Quote",
                    description="Get stock quote.",
                    method=HttpMethod.GET,
                    path="/quote",
                    parameters=[
                        Parameter(
                            name="symbol", param_type=ParameterType.STRING, required=True
                        ),
                    ],
                ),
                Endpoint(
                    endpoint_id="stocks/GET/hist",
                    tool_id="stocks",
                    name="History",
                    description="Historical prices.",
                    method=HttpMethod.GET,
                    path="/history",
                ),
            ],
        ),
        Tool(
            tool_id="news",
            name="News API",
            domain="Media",
            description="News articles and headlines.",
            completeness_score=0.4,
            endpoints=[
                Endpoint(
                    endpoint_id="news/GET/top",
                    tool_id="news",
                    name="Top Headlines",
                    description="Latest news.",
                    method=HttpMethod.GET,
                    path="/top",
                ),
                Endpoint(
                    endpoint_id="news/GET/search",
                    tool_id="news",
                    name="Search",
                    description="Search articles.",
                    method=HttpMethod.GET,
                    path="/search",
                    parameters=[
                        Parameter(name="q", param_type=ParameterType.STRING, required=True),
                    ],
                ),
            ],
        ),
    ]
    registry.add_tools(tools)
    return registry


def _make_mock_embedding_service() -> MagicMock:
    """Return a mocked EmbeddingService with deterministic random embeddings."""
    mock_svc = MagicMock(spec=EmbeddingService)
    rng = np.random.RandomState(42)

    def fake_batch(texts: list[str], **kwargs: object) -> list[list[float]]:
        return [rng.randn(8).tolist() for _ in texts]

    mock_svc.embed_batch.side_effect = fake_batch
    mock_svc.compute_similarity_matrix.side_effect = _REAL_SVC.compute_similarity_matrix
    mock_svc.compute_similarity.side_effect = _REAL_SVC.compute_similarity
    return mock_svc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def registry() -> ToolRegistry:
    return _build_integration_registry()


@pytest.fixture()
def mock_svc() -> MagicMock:
    return _make_mock_embedding_service()


@pytest.fixture()
def graph(registry: ToolRegistry, mock_svc: MagicMock) -> nx.DiGraph:
    """Graph built with default config (all edge types enabled)."""
    config = GraphConfig(similarity_threshold=0.3)
    return GraphBuilder(config=config, embedding_service=mock_svc).build(registry)


# ===========================================================================
# 1. Import completeness
# ===========================================================================


class TestImportCompleteness:
    def test_all_models_importable(self) -> None:
        assert EdgeType is not None
        assert EndpointNode is not None
        assert GraphConfig is not None
        assert GraphEdge is not None
        assert GraphStats is not None
        assert ToolNode is not None

    def test_builder_importable(self) -> None:
        assert GraphBuilder is not None

    def test_embeddings_importable(self) -> None:
        assert EmbeddingService is not None
        assert callable(build_tool_description)
        assert callable(build_endpoint_description)

    def test_persistence_importable(self) -> None:
        assert callable(save_graph)
        assert callable(load_graph)
        assert callable(get_graph_info)
        assert callable(save_embeddings)
        assert callable(load_embeddings)
        assert GraphMetadata is not None
        assert issubclass(GraphSerializationError, Exception)
        assert issubclass(GraphVersionError, GraphSerializationError)
        assert issubclass(GraphChecksumError, GraphSerializationError)

    def test_queries_importable(self) -> None:
        assert callable(get_neighbors)
        assert callable(get_endpoints_for_tool)
        assert callable(get_tool_for_endpoint)
        assert callable(get_domain_endpoints)
        assert callable(get_connected_endpoints)
        assert callable(get_chainable_endpoints)
        assert callable(compute_node_importance)
        assert callable(get_graph_stats)


# ===========================================================================
# 2. Full pipeline
# ===========================================================================


class TestFullPipeline:
    def test_full_pipeline(self, graph: nx.DiGraph, tmp_path: Path) -> None:
        # --- Build verification ---
        tool_nodes = [n for n, d in graph.nodes(data=True) if d.get("node_type") == "tool"]
        ep_nodes = [n for n, d in graph.nodes(data=True) if d.get("node_type") == "endpoint"]
        assert len(tool_nodes) == 4
        assert len(ep_nodes) == 7

        same_tool = [
            (u, v) for u, v, d in graph.edges(data=True) if d.get("edge_type") == "same_tool"
        ]
        assert len(same_tool) == 7

        domain_edges = [
            (u, v) for u, v, d in graph.edges(data=True) if d.get("edge_type") == "same_domain"
        ]
        assert len(domain_edges) > 0

        stats = graph.graph["stats"]
        assert isinstance(stats, GraphStats)
        assert stats.tool_node_count == 4
        assert stats.endpoint_node_count == 7

        # --- Serialize ---
        gp = tmp_path / "graph.pkl"
        meta = save_graph(graph, gp)
        assert meta.node_count == 11
        assert meta.edge_count == graph.number_of_edges()

        # --- Reload ---
        loaded, lmeta = load_graph(gp)
        assert loaded.number_of_nodes() == 11
        assert loaded.number_of_edges() == graph.number_of_edges()

        # --- Query loaded graph ---
        eps = get_endpoints_for_tool(loaded, "weather")
        assert len(eps) == 2
        assert get_tool_for_endpoint(loaded, eps[0]) == "tool:weather"

        weather_eps = get_domain_endpoints(loaded, "Weather")
        assert len(weather_eps) == 3

        pr = compute_node_importance(loaded)
        assert abs(sum(pr.values()) - 1.0) < 0.01


# ===========================================================================
# 3. Graph build verification
# ===========================================================================


class TestBuildVerification:
    def test_build_node_counts(self, graph: nx.DiGraph) -> None:
        tool_count = sum(
            1 for _, d in graph.nodes(data=True) if d.get("node_type") == "tool"
        )
        ep_count = sum(
            1 for _, d in graph.nodes(data=True) if d.get("node_type") == "endpoint"
        )
        assert tool_count == 4
        assert ep_count == 7

    def test_build_same_tool_edges(self, graph: nx.DiGraph) -> None:
        same_tool = [
            (u, v) for u, v, d in graph.edges(data=True) if d.get("edge_type") == "same_tool"
        ]
        assert len(same_tool) == 7

    def test_build_domain_edges_exist(self, graph: nx.DiGraph) -> None:
        domain_edges = [
            (u, v) for u, v, d in graph.edges(data=True) if d.get("edge_type") == "same_domain"
        ]
        # Weather has 3 endpoints → C(3,2)=3 pairs; Finance has 2 → 1 pair; Media has 2 → 1 pair
        assert len(domain_edges) >= 5

    def test_build_all_nodes_have_type(self, graph: nx.DiGraph) -> None:
        for _n, data in graph.nodes(data=True):
            assert "node_type" in data
            assert data["node_type"] in ("tool", "endpoint")

    def test_build_stats_attached(self, graph: nx.DiGraph) -> None:
        stats = graph.graph.get("stats")
        assert isinstance(stats, GraphStats)
        assert stats.total_nodes == 11
        assert stats.total_edges == graph.number_of_edges()

    def test_build_endpoint_has_domain(self, graph: nx.DiGraph) -> None:
        for _n, data in graph.nodes(data=True):
            if data.get("node_type") == "endpoint":
                assert "domain" in data
                assert data["domain"] in ("Weather", "Finance", "Media")


# ===========================================================================
# 4. Serialization round-trip
# ===========================================================================


class TestSerializationRoundTrip:
    def test_serialize_round_trip_nodes(self, graph: nx.DiGraph, tmp_path: Path) -> None:
        p = tmp_path / "g.pkl"
        save_graph(graph, p)
        loaded, _meta = load_graph(p)
        assert loaded.number_of_nodes() == graph.number_of_nodes()

    def test_serialize_round_trip_edges(self, graph: nx.DiGraph, tmp_path: Path) -> None:
        p = tmp_path / "g.pkl"
        save_graph(graph, p)
        loaded, _meta = load_graph(p)
        assert loaded.number_of_edges() == graph.number_of_edges()

    def test_serialize_round_trip_attributes(self, graph: nx.DiGraph, tmp_path: Path) -> None:
        p = tmp_path / "g.pkl"
        save_graph(graph, p)
        loaded, _meta = load_graph(p)
        # Check a tool node attribute
        assert loaded.nodes["tool:weather"]["name"] == "Weather API"
        # Check an edge attribute
        edge_data = loaded["tool:weather"]["ep:weather:weather/GET/cur"]
        assert edge_data["edge_type"] == "same_tool"
        assert edge_data["weight"] == 1.0

    def test_serialize_metadata_correct(self, graph: nx.DiGraph, tmp_path: Path) -> None:
        p = tmp_path / "g.pkl"
        meta = save_graph(graph, p)
        assert meta.node_count == 11
        assert meta.tool_node_count == 4
        assert meta.endpoint_node_count == 7
        assert meta.version == "1.0.0"
        assert len(meta.checksum) == 16


# ===========================================================================
# 5. Query integration
# ===========================================================================


class TestQueryIntegration:
    def test_query_neighbors(self, graph: nx.DiGraph) -> None:
        n = get_neighbors(graph, "ep:weather:weather/GET/cur")
        assert len(n) > 0
        # Should include parent tool and at least one domain neighbor
        assert "tool:weather" in n

    def test_query_endpoints_for_tool(self, graph: nx.DiGraph) -> None:
        eps = get_endpoints_for_tool(graph, "weather")
        assert len(eps) == 2
        assert "ep:weather:weather/GET/cur" in eps
        assert "ep:weather:weather/GET/fore" in eps

    def test_query_tool_for_endpoint(self, graph: nx.DiGraph) -> None:
        tool = get_tool_for_endpoint(graph, "ep:stocks:stocks/GET/quote")
        assert tool == "tool:stocks"

    def test_query_domain_endpoints(self, graph: nx.DiGraph) -> None:
        weather = get_domain_endpoints(graph, "Weather")
        assert len(weather) == 3
        finance = get_domain_endpoints(graph, "Finance")
        assert len(finance) == 2
        media = get_domain_endpoints(graph, "Media")
        assert len(media) == 2

    def test_query_connected_endpoints(self, graph: nx.DiGraph) -> None:
        ce = get_connected_endpoints(graph, "ep:weather:weather/GET/cur", max_hops=2)
        assert len(ce) >= 1
        # Must not include self
        assert "ep:weather:weather/GET/cur" not in ce
        # Must be all endpoints
        for n in ce:
            assert graph.nodes[n].get("node_type") == "endpoint"

    def test_query_graph_stats(self, graph: nx.DiGraph) -> None:
        stats = get_graph_stats(graph)
        assert stats.tool_node_count == 4
        assert stats.endpoint_node_count == 7
        assert stats.total_edges == graph.number_of_edges()


# ===========================================================================
# 6. Embedding cache & graph info
# ===========================================================================


class TestEmbeddingCacheAndInfo:
    def test_embedding_cache_round_trip(self, tmp_path: Path) -> None:
        data = {"node_a": [0.1, 0.2, 0.3], "node_b": [0.4, 0.5, 0.6]}
        p = tmp_path / "emb.joblib"
        save_embeddings(data, p)
        loaded = load_embeddings(p)
        assert loaded == data

    def test_get_graph_info_without_full_load(
        self, graph: nx.DiGraph, tmp_path: Path
    ) -> None:
        p = tmp_path / "g.pkl"
        meta = save_graph(graph, p)
        info = get_graph_info(p)
        assert info.node_count == meta.node_count
        assert info.edge_count == meta.edge_count
        assert info.checksum == meta.checksum


# ===========================================================================
# 7. Configuration variations
# ===========================================================================


class TestConfigVariations:
    def test_build_without_tool_nodes(
        self, registry: ToolRegistry, mock_svc: MagicMock
    ) -> None:
        config = GraphConfig(include_tool_nodes=False, include_semantic_edges=False)
        g = GraphBuilder(config=config, embedding_service=mock_svc).build(registry)
        tool_nodes = [n for n, d in g.nodes(data=True) if d.get("node_type") == "tool"]
        assert len(tool_nodes) == 0
        assert g.number_of_nodes() == 7  # endpoints only
        same_tool = [
            (u, v) for u, v, d in g.edges(data=True) if d.get("edge_type") == "same_tool"
        ]
        assert len(same_tool) == 0

    def test_build_without_semantic_edges(
        self, registry: ToolRegistry, mock_svc: MagicMock
    ) -> None:
        config = GraphConfig(include_semantic_edges=False)
        g = GraphBuilder(config=config, embedding_service=mock_svc).build(registry)
        sem = [
            (u, v)
            for u, v, d in g.edges(data=True)
            if d.get("edge_type") == "semantic_similarity"
        ]
        assert len(sem) == 0
        # SAME_TOOL and SAME_DOMAIN should still exist
        assert g.number_of_edges() > 0

    def test_build_without_domain_edges(
        self, registry: ToolRegistry, mock_svc: MagicMock
    ) -> None:
        config = GraphConfig(include_domain_edges=False, include_semantic_edges=False)
        g = GraphBuilder(config=config, embedding_service=mock_svc).build(registry)
        dom = [
            (u, v)
            for u, v, d in g.edges(data=True)
            if d.get("edge_type") == "same_domain"
        ]
        assert len(dom) == 0
        # Only SAME_TOOL edges remain
        same_tool = [
            (u, v) for u, v, d in g.edges(data=True) if d.get("edge_type") == "same_tool"
        ]
        assert len(same_tool) == 7
