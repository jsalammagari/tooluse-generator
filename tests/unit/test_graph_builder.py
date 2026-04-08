"""Unit tests for Task 19 — Graph builder."""

from __future__ import annotations

from unittest.mock import MagicMock

import networkx as nx
import numpy as np
import pytest

from tooluse_gen.graph.builder import GraphBuilder
from tooluse_gen.graph.embeddings import EmbeddingService
from tooluse_gen.graph.models import EdgeType, GraphConfig, GraphStats
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

_REAL_SVC = EmbeddingService()  # only used for pure-numpy similarity helpers


def _build_test_registry() -> ToolRegistry:
    """Build a small registry: 5 tools, 9 endpoints, 3 domains."""
    registry = ToolRegistry()
    tools = [
        Tool(
            tool_id="weather",
            name="Weather API",
            domain="Weather",
            description="Weather data.",
            completeness_score=0.8,
            endpoints=[
                Endpoint(
                    endpoint_id="weather/GET/cur",
                    tool_id="weather",
                    name="Current",
                    description="Current conditions.",
                    method=HttpMethod.GET,
                    path="/current",
                    parameters=[
                        Parameter(name="city", param_type=ParameterType.STRING, required=True),
                    ],
                ),
                Endpoint(
                    endpoint_id="weather/GET/fore",
                    tool_id="weather",
                    name="Forecast",
                    description="5-day forecast.",
                    method=HttpMethod.GET,
                    path="/forecast",
                ),
            ],
        ),
        Tool(
            tool_id="maps",
            name="Maps API",
            domain="Weather",
            description="Map and geo data.",
            completeness_score=0.6,
            endpoints=[
                Endpoint(
                    endpoint_id="maps/GET/geo",
                    tool_id="maps",
                    name="Geocode",
                    method=HttpMethod.GET,
                    path="/geocode",
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
                    method=HttpMethod.GET,
                    path="/quote",
                ),
                Endpoint(
                    endpoint_id="stocks/GET/hist",
                    tool_id="stocks",
                    name="History",
                    method=HttpMethod.GET,
                    path="/history",
                ),
            ],
        ),
        Tool(
            tool_id="bank",
            name="Banking API",
            domain="Finance",
            description="Banking services.",
            completeness_score=0.5,
            endpoints=[
                Endpoint(
                    endpoint_id="bank/POST/transfer",
                    tool_id="bank",
                    name="Transfer",
                    method=HttpMethod.POST,
                    path="/transfer",
                ),
                Endpoint(
                    endpoint_id="bank/GET/balance",
                    tool_id="bank",
                    name="Balance",
                    method=HttpMethod.GET,
                    path="/balance",
                ),
            ],
        ),
        Tool(
            tool_id="news",
            name="News API",
            domain="Media",
            description="News articles.",
            completeness_score=0.3,
            endpoints=[
                Endpoint(
                    endpoint_id="news/GET/top",
                    tool_id="news",
                    name="Top Headlines",
                    method=HttpMethod.GET,
                    path="/top",
                ),
            ],
        ),
    ]
    registry.add_tools(tools)
    return registry


def _make_mock_service() -> MagicMock:
    """Return a mocked EmbeddingService with deterministic embeddings."""
    mock_svc = MagicMock(spec=EmbeddingService)
    rng = np.random.RandomState(42)

    def fake_batch(texts: list[str], **kwargs: object) -> list[list[float]]:
        return [rng.randn(4).tolist() for _ in texts]

    mock_svc.embed_batch.side_effect = fake_batch
    mock_svc.compute_similarity_matrix.side_effect = _REAL_SVC.compute_similarity_matrix
    mock_svc.compute_similarity.side_effect = _REAL_SVC.compute_similarity
    return mock_svc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def registry() -> ToolRegistry:
    return _build_test_registry()


@pytest.fixture()
def mock_svc() -> MagicMock:
    return _make_mock_service()


@pytest.fixture()
def default_graph(registry: ToolRegistry, mock_svc: MagicMock) -> nx.DiGraph:
    """Graph built with default config."""
    config = GraphConfig(similarity_threshold=0.5)
    builder = GraphBuilder(config=config, embedding_service=mock_svc)
    return builder.build(registry)


# ===========================================================================
# Build basics
# ===========================================================================


class TestBuildBasics:
    def test_returns_digraph(self, default_graph: nx.DiGraph) -> None:
        assert isinstance(default_graph, nx.DiGraph)

    def test_tool_node_count(self, default_graph: nx.DiGraph) -> None:
        tool_nodes = [
            n for n, d in default_graph.nodes(data=True) if d.get("node_type") == "tool"
        ]
        assert len(tool_nodes) == 5

    def test_endpoint_node_count(self, default_graph: nx.DiGraph) -> None:
        ep_nodes = [
            n for n, d in default_graph.nodes(data=True) if d.get("node_type") == "endpoint"
        ]
        assert len(ep_nodes) == 8

    def test_all_nodes_have_node_type(self, default_graph: nx.DiGraph) -> None:
        for _n, data in default_graph.nodes(data=True):
            assert "node_type" in data
            assert data["node_type"] in ("tool", "endpoint")

    def test_tool_node_attributes(self, default_graph: nx.DiGraph) -> None:
        for _n, data in default_graph.nodes(data=True):
            if data.get("node_type") != "tool":
                continue
            for key in ("tool_id", "name", "domain", "quality_tier", "endpoint_count"):
                assert key in data, f"Missing attribute {key} on tool node"

    def test_endpoint_node_attributes(self, default_graph: nx.DiGraph) -> None:
        for _n, data in default_graph.nodes(data=True):
            if data.get("node_type") != "endpoint":
                continue
            for key in ("endpoint_id", "tool_id", "name", "method", "path", "domain"):
                assert key in data, f"Missing attribute {key} on endpoint node"

    def test_node_ids_use_correct_scheme(self, default_graph: nx.DiGraph) -> None:
        for n, data in default_graph.nodes(data=True):
            if data.get("node_type") == "tool":
                assert n.startswith("tool:")
            else:
                assert n.startswith("ep:")


# ===========================================================================
# SAME_TOOL edges
# ===========================================================================


class TestSameToolEdges:
    def test_same_tool_edge_count(self, default_graph: nx.DiGraph) -> None:
        same_tool = [
            (u, v)
            for u, v, d in default_graph.edges(data=True)
            if d.get("edge_type") == EdgeType.SAME_TOOL.value
        ]
        # 8 endpoints → 8 SAME_TOOL edges
        assert len(same_tool) == 8

    def test_same_tool_weight_is_one(self, default_graph: nx.DiGraph) -> None:
        for _u, _v, d in default_graph.edges(data=True):
            if d.get("edge_type") == EdgeType.SAME_TOOL.value:
                assert d["weight"] == 1.0

    def test_same_tool_connects_tool_to_endpoint(self, default_graph: nx.DiGraph) -> None:
        for u, v, d in default_graph.edges(data=True):
            if d.get("edge_type") == EdgeType.SAME_TOOL.value:
                assert default_graph.nodes[u]["node_type"] == "tool"
                assert default_graph.nodes[v]["node_type"] == "endpoint"


# ===========================================================================
# SAME_DOMAIN edges
# ===========================================================================


class TestSameDomainEdges:
    def test_domain_edges_exist(self, default_graph: nx.DiGraph) -> None:
        domain_edges = [
            (u, v)
            for u, v, d in default_graph.edges(data=True)
            if d.get("edge_type") == EdgeType.SAME_DOMAIN.value
        ]
        assert len(domain_edges) > 0

    def test_domain_edge_weights_in_range(self, default_graph: nx.DiGraph) -> None:
        for _u, _v, d in default_graph.edges(data=True):
            if d.get("edge_type") == EdgeType.SAME_DOMAIN.value:
                assert 0.0 < d["weight"] <= 1.0

    def test_domain_edges_connect_same_domain(self, default_graph: nx.DiGraph) -> None:
        for u, v, d in default_graph.edges(data=True):
            if d.get("edge_type") == EdgeType.SAME_DOMAIN.value:
                dom_u = default_graph.nodes[u].get("domain")
                dom_v = default_graph.nodes[v].get("domain")
                assert dom_u == dom_v and dom_u != ""

    def test_no_domain_edges_when_disabled(
        self, registry: ToolRegistry, mock_svc: MagicMock
    ) -> None:
        config = GraphConfig(include_domain_edges=False, include_semantic_edges=False)
        graph = GraphBuilder(config=config, embedding_service=mock_svc).build(registry)
        domain_edges = [
            (u, v)
            for u, v, d in graph.edges(data=True)
            if d.get("edge_type") == EdgeType.SAME_DOMAIN.value
        ]
        assert len(domain_edges) == 0


# ===========================================================================
# SEMANTIC_SIMILARITY edges
# ===========================================================================


class TestSemanticEdges:
    def test_semantic_edges_exist_with_low_threshold(
        self, registry: ToolRegistry, mock_svc: MagicMock
    ) -> None:
        config = GraphConfig(similarity_threshold=0.0, include_domain_edges=False)
        graph = GraphBuilder(config=config, embedding_service=mock_svc).build(registry)
        sem_edges = [
            (u, v)
            for u, v, d in graph.edges(data=True)
            if d.get("edge_type") == EdgeType.SEMANTIC_SIMILARITY.value
        ]
        # With threshold=0 many pairs should connect
        assert len(sem_edges) > 0

    def test_semantic_edge_weights_above_threshold(
        self, registry: ToolRegistry, mock_svc: MagicMock
    ) -> None:
        threshold = 0.3
        config = GraphConfig(similarity_threshold=threshold, include_domain_edges=False)
        graph = GraphBuilder(config=config, embedding_service=mock_svc).build(registry)
        for _u, _v, d in graph.edges(data=True):
            if d.get("edge_type") == EdgeType.SEMANTIC_SIMILARITY.value:
                assert d["weight"] >= threshold

    def test_no_semantic_edges_when_disabled(
        self, registry: ToolRegistry, mock_svc: MagicMock
    ) -> None:
        config = GraphConfig(include_semantic_edges=False)
        graph = GraphBuilder(config=config, embedding_service=mock_svc).build(registry)
        sem_edges = [
            (u, v)
            for u, v, d in graph.edges(data=True)
            if d.get("edge_type") == EdgeType.SEMANTIC_SIMILARITY.value
        ]
        assert len(sem_edges) == 0

    def test_high_threshold_fewer_edges(
        self, registry: ToolRegistry, mock_svc: MagicMock
    ) -> None:
        low_cfg = GraphConfig(similarity_threshold=0.0, include_domain_edges=False)
        high_cfg = GraphConfig(similarity_threshold=0.99, include_domain_edges=False)
        # Need fresh mocks so RNG state matches
        g_low = GraphBuilder(
            config=low_cfg, embedding_service=_make_mock_service()
        ).build(registry)
        g_high = GraphBuilder(
            config=high_cfg, embedding_service=_make_mock_service()
        ).build(registry)
        sem_low = sum(
            1
            for _, _, d in g_low.edges(data=True)
            if d.get("edge_type") == EdgeType.SEMANTIC_SIMILARITY.value
        )
        sem_high = sum(
            1
            for _, _, d in g_high.edges(data=True)
            if d.get("edge_type") == EdgeType.SEMANTIC_SIMILARITY.value
        )
        assert sem_low >= sem_high


# ===========================================================================
# Configuration flags
# ===========================================================================


class TestConfiguration:
    def test_no_tool_nodes(self, registry: ToolRegistry, mock_svc: MagicMock) -> None:
        config = GraphConfig(include_tool_nodes=False, include_semantic_edges=False)
        graph = GraphBuilder(config=config, embedding_service=mock_svc).build(registry)
        tool_nodes = [
            n for n, d in graph.nodes(data=True) if d.get("node_type") == "tool"
        ]
        assert len(tool_nodes) == 0

    def test_no_tool_nodes_means_no_same_tool_edges(
        self, registry: ToolRegistry, mock_svc: MagicMock
    ) -> None:
        config = GraphConfig(include_tool_nodes=False, include_semantic_edges=False)
        graph = GraphBuilder(config=config, embedding_service=mock_svc).build(registry)
        same_tool = [
            (u, v)
            for u, v, d in graph.edges(data=True)
            if d.get("edge_type") == EdgeType.SAME_TOOL.value
        ]
        assert len(same_tool) == 0

    def test_all_disabled_only_endpoints(
        self, registry: ToolRegistry, mock_svc: MagicMock
    ) -> None:
        config = GraphConfig(
            include_tool_nodes=False,
            include_domain_edges=False,
            include_semantic_edges=False,
        )
        graph = GraphBuilder(config=config, embedding_service=mock_svc).build(registry)
        assert graph.number_of_nodes() == 8
        assert graph.number_of_edges() == 0


# ===========================================================================
# Pruning
# ===========================================================================


class TestPruning:
    def test_prune_limits_edges(self, registry: ToolRegistry, mock_svc: MagicMock) -> None:
        config = GraphConfig(
            max_edges_per_node=2,
            similarity_threshold=0.0,
            include_tool_nodes=False,
            include_domain_edges=True,
        )
        graph = GraphBuilder(config=config, embedding_service=mock_svc).build(registry)
        for node in graph.nodes():
            total = graph.in_degree(node) + graph.out_degree(node)
            # Prunable edges should be limited; SAME_TOOL are preserved but we disabled tool nodes
            assert total <= max(
                2,
                sum(
                    1
                    for _, _, d in list(graph.in_edges(node, data=True))
                    + list(graph.out_edges(node, data=True))
                    if d.get("edge_type") == EdgeType.SAME_TOOL.value
                ),
            )

    def test_same_tool_never_pruned(self, registry: ToolRegistry, mock_svc: MagicMock) -> None:
        config = GraphConfig(
            max_edges_per_node=1,
            similarity_threshold=0.0,
            include_domain_edges=True,
        )
        graph = GraphBuilder(config=config, embedding_service=mock_svc).build(registry)
        same_tool = [
            (u, v)
            for u, v, d in graph.edges(data=True)
            if d.get("edge_type") == EdgeType.SAME_TOOL.value
        ]
        # All 8 SAME_TOOL edges should still be present
        assert len(same_tool) == 8


# ===========================================================================
# GraphStats
# ===========================================================================


class TestGraphStats:
    def test_stats_attached(self, default_graph: nx.DiGraph) -> None:
        assert "stats" in default_graph.graph
        assert isinstance(default_graph.graph["stats"], GraphStats)

    def test_stats_tool_count(self, default_graph: nx.DiGraph) -> None:
        stats = default_graph.graph["stats"]
        assert stats.tool_node_count == 5

    def test_stats_endpoint_count(self, default_graph: nx.DiGraph) -> None:
        stats = default_graph.graph["stats"]
        assert stats.endpoint_node_count == 8

    def test_stats_density(self, default_graph: nx.DiGraph) -> None:
        stats = default_graph.graph["stats"]
        assert 0.0 <= stats.density <= 1.0

    def test_stats_connected_components(self, default_graph: nx.DiGraph) -> None:
        stats = default_graph.graph["stats"]
        assert stats.connected_components >= 1

    def test_stats_edge_counts_keys(self, default_graph: nx.DiGraph) -> None:
        stats = default_graph.graph["stats"]
        assert EdgeType.SAME_TOOL.value in stats.edge_counts

    def test_stats_total_edges(self, default_graph: nx.DiGraph) -> None:
        stats = default_graph.graph["stats"]
        assert stats.total_edges == default_graph.number_of_edges()


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_empty_registry(self, mock_svc: MagicMock) -> None:
        config = GraphConfig()
        graph = GraphBuilder(config=config, embedding_service=mock_svc).build(ToolRegistry())
        assert graph.number_of_nodes() == 0
        assert graph.number_of_edges() == 0
        stats = graph.graph["stats"]
        assert stats.total_nodes == 0
        assert stats.total_edges == 0
        assert stats.connected_components == 0

    def test_single_tool_single_endpoint(self, mock_svc: MagicMock) -> None:
        registry = ToolRegistry()
        ep = Endpoint(
            endpoint_id="t/GET/a", tool_id="t", name="EP", method=HttpMethod.GET, path="/a"
        )
        registry.add_tool(
            Tool(tool_id="t", name="T", domain="D", endpoints=[ep], completeness_score=0.5)
        )
        config = GraphConfig(include_semantic_edges=False)
        graph = GraphBuilder(config=config, embedding_service=mock_svc).build(registry)
        assert graph.number_of_nodes() == 2  # 1 tool + 1 endpoint
        same_tool = [
            (u, v)
            for u, v, d in graph.edges(data=True)
            if d.get("edge_type") == EdgeType.SAME_TOOL.value
        ]
        assert len(same_tool) == 1

    def test_tool_with_no_endpoints(self, mock_svc: MagicMock) -> None:
        registry = ToolRegistry()
        registry.add_tool(Tool(tool_id="empty", name="Empty", completeness_score=0.5))
        config = GraphConfig(include_semantic_edges=False)
        graph = GraphBuilder(config=config, embedding_service=mock_svc).build(registry)
        # Tool node exists but no endpoints
        assert graph.number_of_nodes() == 1
        assert graph.number_of_edges() == 0

    def test_all_edges_have_required_attributes(self, default_graph: nx.DiGraph) -> None:
        for u, v, data in default_graph.edges(data=True):
            assert "edge_type" in data, f"Edge ({u}, {v}) missing edge_type"
            assert "weight" in data, f"Edge ({u}, {v}) missing weight"
            assert 0.0 <= data["weight"] <= 1.0

    def test_embeddings_stored_on_nodes(
        self, registry: ToolRegistry, mock_svc: MagicMock
    ) -> None:
        config = GraphConfig(similarity_threshold=0.5)
        graph = GraphBuilder(config=config, embedding_service=mock_svc).build(registry)
        ep_nodes_with_emb = [
            n
            for n, d in graph.nodes(data=True)
            if d.get("node_type") == "endpoint" and d.get("embedding") is not None
        ]
        assert len(ep_nodes_with_emb) == 8
