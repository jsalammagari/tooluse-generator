"""Unit tests for Task 21 — Graph query & traversal utilities."""

from __future__ import annotations

import networkx as nx
import pytest

from tooluse_gen.graph.models import EdgeType, GraphStats
from tooluse_gen.graph.queries import (
    compute_node_importance,
    get_chainable_endpoints,
    get_connected_endpoints,
    get_domain_endpoints,
    get_endpoints_for_tool,
    get_graph_stats,
    get_neighbors,
    get_tool_for_endpoint,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_graph() -> nx.DiGraph:
    """Build a small graph: 3 tools, 5 endpoints, 3 edge types."""
    g = nx.DiGraph()
    # tool nodes
    g.add_node(
        "tool:weather", node_type="tool", tool_id="weather",
        name="Weather API", domain="Weather", quality_tier="high", endpoint_count=2,
    )
    g.add_node(
        "tool:maps", node_type="tool", tool_id="maps",
        name="Maps API", domain="Weather", quality_tier="medium", endpoint_count=1,
    )
    g.add_node(
        "tool:stocks", node_type="tool", tool_id="stocks",
        name="Stock API", domain="Finance", quality_tier="high", endpoint_count=2,
    )
    # endpoint nodes
    g.add_node(
        "ep:weather:weather/GET/cur", node_type="endpoint",
        endpoint_id="weather/GET/cur", tool_id="weather",
        name="Current", method="GET", path="/current", domain="Weather",
    )
    g.add_node(
        "ep:weather:weather/GET/fore", node_type="endpoint",
        endpoint_id="weather/GET/fore", tool_id="weather",
        name="Forecast", method="GET", path="/forecast", domain="Weather",
    )
    g.add_node(
        "ep:maps:maps/GET/geo", node_type="endpoint",
        endpoint_id="maps/GET/geo", tool_id="maps",
        name="Geocode", method="GET", path="/geocode", domain="Weather",
    )
    g.add_node(
        "ep:stocks:stocks/GET/quote", node_type="endpoint",
        endpoint_id="stocks/GET/quote", tool_id="stocks",
        name="Quote", method="GET", path="/quote", domain="Finance",
    )
    g.add_node(
        "ep:stocks:stocks/GET/hist", node_type="endpoint",
        endpoint_id="stocks/GET/hist", tool_id="stocks",
        name="History", method="GET", path="/history", domain="Finance",
    )
    # SAME_TOOL edges (5)
    g.add_edge("tool:weather", "ep:weather:weather/GET/cur", edge_type="same_tool", weight=1.0)
    g.add_edge("tool:weather", "ep:weather:weather/GET/fore", edge_type="same_tool", weight=1.0)
    g.add_edge("tool:maps", "ep:maps:maps/GET/geo", edge_type="same_tool", weight=1.0)
    g.add_edge("tool:stocks", "ep:stocks:stocks/GET/quote", edge_type="same_tool", weight=1.0)
    g.add_edge("tool:stocks", "ep:stocks:stocks/GET/hist", edge_type="same_tool", weight=1.0)
    # SAME_DOMAIN edges (4)
    g.add_edge(
        "ep:weather:weather/GET/cur", "ep:weather:weather/GET/fore",
        edge_type="same_domain", weight=0.9,
    )
    g.add_edge(
        "ep:weather:weather/GET/cur", "ep:maps:maps/GET/geo",
        edge_type="same_domain", weight=0.75,
    )
    g.add_edge(
        "ep:weather:weather/GET/fore", "ep:maps:maps/GET/geo",
        edge_type="same_domain", weight=0.75,
    )
    g.add_edge(
        "ep:stocks:stocks/GET/quote", "ep:stocks:stocks/GET/hist",
        edge_type="same_domain", weight=0.9,
    )
    # SEMANTIC edge (1, cross-domain)
    g.add_edge(
        "ep:weather:weather/GET/cur", "ep:stocks:stocks/GET/quote",
        edge_type="semantic_similarity", weight=0.8,
    )
    return g


# ===========================================================================
# get_neighbors
# ===========================================================================


class TestGetNeighbors:
    def test_endpoint_all_neighbors(self, test_graph: nx.DiGraph) -> None:
        n = get_neighbors(test_graph, "ep:weather:weather/GET/cur")
        # successors: fore, geo, quote  |  predecessors: tool:weather
        assert "tool:weather" in n
        assert "ep:weather:weather/GET/fore" in n
        assert "ep:maps:maps/GET/geo" in n
        assert "ep:stocks:stocks/GET/quote" in n

    def test_tool_node_neighbors(self, test_graph: nx.DiGraph) -> None:
        n = get_neighbors(test_graph, "tool:weather")
        assert "ep:weather:weather/GET/cur" in n
        assert "ep:weather:weather/GET/fore" in n
        assert len(n) == 2

    def test_filter_same_tool(self, test_graph: nx.DiGraph) -> None:
        n = get_neighbors(test_graph, "ep:weather:weather/GET/cur", edge_type=EdgeType.SAME_TOOL)
        assert n == ["tool:weather"]

    def test_filter_same_domain(self, test_graph: nx.DiGraph) -> None:
        n = get_neighbors(
            test_graph, "ep:weather:weather/GET/cur", edge_type=EdgeType.SAME_DOMAIN
        )
        assert "ep:weather:weather/GET/fore" in n
        assert "ep:maps:maps/GET/geo" in n
        assert "tool:weather" not in n

    def test_filter_semantic(self, test_graph: nx.DiGraph) -> None:
        n = get_neighbors(
            test_graph, "ep:weather:weather/GET/cur", edge_type=EdgeType.SEMANTIC_SIMILARITY
        )
        assert n == ["ep:stocks:stocks/GET/quote"]

    def test_nonexistent_node(self, test_graph: nx.DiGraph) -> None:
        assert get_neighbors(test_graph, "nonexistent") == []

    def test_results_sorted(self, test_graph: nx.DiGraph) -> None:
        n = get_neighbors(test_graph, "ep:weather:weather/GET/cur")
        assert n == sorted(n)


# ===========================================================================
# get_endpoints_for_tool
# ===========================================================================


class TestGetEndpointsForTool:
    def test_weather(self, test_graph: nx.DiGraph) -> None:
        eps = get_endpoints_for_tool(test_graph, "weather")
        assert len(eps) == 2
        assert "ep:weather:weather/GET/cur" in eps
        assert "ep:weather:weather/GET/fore" in eps

    def test_stocks(self, test_graph: nx.DiGraph) -> None:
        eps = get_endpoints_for_tool(test_graph, "stocks")
        assert len(eps) == 2

    def test_maps(self, test_graph: nx.DiGraph) -> None:
        eps = get_endpoints_for_tool(test_graph, "maps")
        assert eps == ["ep:maps:maps/GET/geo"]

    def test_nonexistent(self, test_graph: nx.DiGraph) -> None:
        assert get_endpoints_for_tool(test_graph, "nope") == []

    def test_sorted(self, test_graph: nx.DiGraph) -> None:
        eps = get_endpoints_for_tool(test_graph, "weather")
        assert eps == sorted(eps)


# ===========================================================================
# get_tool_for_endpoint
# ===========================================================================


class TestGetToolForEndpoint:
    def test_weather_endpoint(self, test_graph: nx.DiGraph) -> None:
        assert get_tool_for_endpoint(test_graph, "ep:weather:weather/GET/cur") == "tool:weather"

    def test_weather_forecast(self, test_graph: nx.DiGraph) -> None:
        assert get_tool_for_endpoint(test_graph, "ep:weather:weather/GET/fore") == "tool:weather"

    def test_stocks_endpoint(self, test_graph: nx.DiGraph) -> None:
        assert get_tool_for_endpoint(test_graph, "ep:stocks:stocks/GET/quote") == "tool:stocks"

    def test_nonexistent(self, test_graph: nx.DiGraph) -> None:
        assert get_tool_for_endpoint(test_graph, "nonexistent") is None

    def test_tool_node_returns_none(self, test_graph: nx.DiGraph) -> None:
        # A tool node has no SAME_TOOL predecessor
        assert get_tool_for_endpoint(test_graph, "tool:weather") is None


# ===========================================================================
# get_domain_endpoints
# ===========================================================================


class TestGetDomainEndpoints:
    def test_weather_domain(self, test_graph: nx.DiGraph) -> None:
        eps = get_domain_endpoints(test_graph, "Weather")
        assert len(eps) == 3
        assert "ep:weather:weather/GET/cur" in eps
        assert "ep:maps:maps/GET/geo" in eps

    def test_finance_domain(self, test_graph: nx.DiGraph) -> None:
        eps = get_domain_endpoints(test_graph, "Finance")
        assert len(eps) == 2

    def test_nonexistent_domain(self, test_graph: nx.DiGraph) -> None:
        assert get_domain_endpoints(test_graph, "Nope") == []

    def test_sorted(self, test_graph: nx.DiGraph) -> None:
        eps = get_domain_endpoints(test_graph, "Weather")
        assert eps == sorted(eps)


# ===========================================================================
# get_connected_endpoints
# ===========================================================================


class TestGetConnectedEndpoints:
    def test_one_hop(self, test_graph: nx.DiGraph) -> None:
        ce = get_connected_endpoints(test_graph, "ep:weather:weather/GET/cur", max_hops=1)
        # Direct endpoint neighbors: fore (same_domain), geo (same_domain), quote (semantic)
        assert "ep:weather:weather/GET/fore" in ce
        assert "ep:maps:maps/GET/geo" in ce
        assert "ep:stocks:stocks/GET/quote" in ce

    def test_two_hops(self, test_graph: nx.DiGraph) -> None:
        ce = get_connected_endpoints(test_graph, "ep:weather:weather/GET/cur", max_hops=2)
        # Should also reach hist via quote
        assert "ep:stocks:stocks/GET/hist" in ce

    def test_excludes_self(self, test_graph: nx.DiGraph) -> None:
        ce = get_connected_endpoints(test_graph, "ep:weather:weather/GET/cur", max_hops=2)
        assert "ep:weather:weather/GET/cur" not in ce

    def test_excludes_tool_nodes(self, test_graph: nx.DiGraph) -> None:
        ce = get_connected_endpoints(test_graph, "ep:weather:weather/GET/cur", max_hops=2)
        for n in ce:
            assert test_graph.nodes[n].get("node_type") == "endpoint"

    def test_nonexistent_node(self, test_graph: nx.DiGraph) -> None:
        assert get_connected_endpoints(test_graph, "nonexistent") == []

    def test_zero_hops(self, test_graph: nx.DiGraph) -> None:
        assert get_connected_endpoints(test_graph, "ep:weather:weather/GET/cur", max_hops=0) == []

    def test_sorted(self, test_graph: nx.DiGraph) -> None:
        ce = get_connected_endpoints(test_graph, "ep:weather:weather/GET/cur", max_hops=2)
        assert ce == sorted(ce)


# ===========================================================================
# get_chainable_endpoints
# ===========================================================================


class TestGetChainableEndpoints:
    def test_from_weather_cur(self, test_graph: nx.DiGraph) -> None:
        ch = get_chainable_endpoints(test_graph, "ep:weather:weather/GET/cur")
        ids = [c[0] for c in ch]
        assert "ep:weather:weather/GET/fore" in ids
        assert "ep:maps:maps/GET/geo" in ids
        assert "ep:stocks:stocks/GET/quote" in ids

    def test_sorted_by_weight_desc(self, test_graph: nx.DiGraph) -> None:
        ch = get_chainable_endpoints(test_graph, "ep:weather:weather/GET/cur")
        weights = [w for _, w in ch]
        assert weights == sorted(weights, reverse=True)

    def test_excludes_tool_nodes(self, test_graph: nx.DiGraph) -> None:
        ch = get_chainable_endpoints(test_graph, "ep:weather:weather/GET/cur")
        for nid, _w in ch:
            assert test_graph.nodes[nid].get("node_type") == "endpoint"

    def test_nonexistent_node(self, test_graph: nx.DiGraph) -> None:
        assert get_chainable_endpoints(test_graph, "nonexistent") == []

    def test_isolated_endpoint(self) -> None:
        g = nx.DiGraph()
        g.add_node("ep:x:x/GET/a", node_type="endpoint")
        assert get_chainable_endpoints(g, "ep:x:x/GET/a") == []


# ===========================================================================
# compute_node_importance
# ===========================================================================


class TestComputeNodeImportance:
    def test_returns_all_nodes(self, test_graph: nx.DiGraph) -> None:
        pr = compute_node_importance(test_graph)
        assert len(pr) == test_graph.number_of_nodes()

    def test_all_positive(self, test_graph: nx.DiGraph) -> None:
        pr = compute_node_importance(test_graph)
        assert all(v > 0 for v in pr.values())

    def test_sums_to_one(self, test_graph: nx.DiGraph) -> None:
        pr = compute_node_importance(test_graph)
        assert abs(sum(pr.values()) - 1.0) < 0.01

    def test_empty_graph(self) -> None:
        assert compute_node_importance(nx.DiGraph()) == {}

    def test_keys_are_strings(self, test_graph: nx.DiGraph) -> None:
        pr = compute_node_importance(test_graph)
        assert all(isinstance(k, str) for k in pr)


# ===========================================================================
# get_graph_stats
# ===========================================================================


class TestGetGraphStats:
    def test_with_cached_stats(self, test_graph: nx.DiGraph) -> None:
        expected = GraphStats(tool_node_count=99)
        test_graph.graph["stats"] = expected
        assert get_graph_stats(test_graph) is expected

    def test_without_stats_tool_count(self, test_graph: nx.DiGraph) -> None:
        test_graph.graph.pop("stats", None)
        stats = get_graph_stats(test_graph)
        assert stats.tool_node_count == 3

    def test_without_stats_endpoint_count(self, test_graph: nx.DiGraph) -> None:
        test_graph.graph.pop("stats", None)
        stats = get_graph_stats(test_graph)
        assert stats.endpoint_node_count == 5

    def test_without_stats_edge_counts(self, test_graph: nx.DiGraph) -> None:
        test_graph.graph.pop("stats", None)
        stats = get_graph_stats(test_graph)
        assert stats.edge_counts.get("same_tool") == 5
        assert stats.edge_counts.get("same_domain") == 4
        assert stats.edge_counts.get("semantic_similarity") == 1

    def test_without_stats_density(self, test_graph: nx.DiGraph) -> None:
        test_graph.graph.pop("stats", None)
        stats = get_graph_stats(test_graph)
        assert 0.0 <= stats.density <= 1.0

    def test_empty_graph(self) -> None:
        stats = get_graph_stats(nx.DiGraph())
        assert stats.tool_node_count == 0
        assert stats.endpoint_node_count == 0
        assert stats.total_nodes == 0
        assert stats.total_edges == 0
        assert stats.connected_components == 0
