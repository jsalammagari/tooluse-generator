"""Graph query and traversal utilities.

Standalone functions for querying the tool graph built by
:class:`GraphBuilder`.  Every function takes an ``nx.DiGraph`` as its
first argument and returns deterministic (sorted) results.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import networkx as nx

from tooluse_gen.graph.models import EdgeType, GraphStats

# ---------------------------------------------------------------------------
# Neighbor queries
# ---------------------------------------------------------------------------


def get_neighbors(
    graph: nx.DiGraph,
    node_id: str,
    edge_type: EdgeType | None = None,
) -> list[str]:
    """Return sorted neighbor IDs reachable from *node_id* (both directions).

    If *edge_type* is given, only neighbours connected by that type are
    returned.  Returns an empty list when the node is absent.
    """
    if node_id not in graph:
        return []

    et_filter: str | None = edge_type.value if edge_type is not None else None
    neighbors: set[str] = set()

    for _u, v, data in graph.out_edges(node_id, data=True):
        if et_filter is None or data.get("edge_type") == et_filter:
            neighbors.add(v)

    for u, _v, data in graph.in_edges(node_id, data=True):
        if et_filter is None or data.get("edge_type") == et_filter:
            neighbors.add(u)

    return sorted(neighbors)


# ---------------------------------------------------------------------------
# Tool / endpoint lookups
# ---------------------------------------------------------------------------


def get_endpoints_for_tool(graph: nx.DiGraph, tool_id: str) -> list[str]:
    """Return sorted endpoint node IDs connected to *tool_id* via SAME_TOOL."""
    tool_node = f"tool:{tool_id}"
    if tool_node not in graph:
        return []

    result: list[str] = []
    for _u, v, data in graph.out_edges(tool_node, data=True):
        if data.get("edge_type") == EdgeType.SAME_TOOL.value:
            result.append(v)
    return sorted(result)


def get_tool_for_endpoint(graph: nx.DiGraph, endpoint_node_id: str) -> str | None:
    """Return the parent tool node ID for *endpoint_node_id*, or ``None``."""
    if endpoint_node_id not in graph:
        return None

    for u, _v, data in graph.in_edges(endpoint_node_id, data=True):
        if data.get("edge_type") == EdgeType.SAME_TOOL.value:
            return str(u)
    return None


# ---------------------------------------------------------------------------
# Domain queries
# ---------------------------------------------------------------------------


def get_domain_endpoints(graph: nx.DiGraph, domain: str) -> list[str]:
    """Return sorted endpoint node IDs whose *domain* attribute matches."""
    result: list[str] = []
    for node_id, data in graph.nodes(data=True):
        if data.get("node_type") == "endpoint" and data.get("domain") == domain:
            result.append(node_id)
    return sorted(result)


# ---------------------------------------------------------------------------
# Traversal
# ---------------------------------------------------------------------------


def get_connected_endpoints(
    graph: nx.DiGraph,
    endpoint_node_id: str,
    max_hops: int = 2,
) -> list[str]:
    """BFS from *endpoint_node_id* up to *max_hops*, returning endpoint nodes only.

    The graph is treated as undirected for traversal.  The starting node
    is excluded from results.
    """
    if endpoint_node_id not in graph or max_hops <= 0:
        return []

    undirected = graph.to_undirected()
    visited: set[str] = {endpoint_node_id}
    queue: deque[tuple[str, int]] = deque([(endpoint_node_id, 0)])
    result: list[str] = []

    while queue:
        current, depth = queue.popleft()
        if depth >= max_hops:
            continue
        for neighbor in undirected.neighbors(current):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append((neighbor, depth + 1))
            if graph.nodes[neighbor].get("node_type") == "endpoint":
                result.append(neighbor)

    return sorted(result)


def get_chainable_endpoints(
    graph: nx.DiGraph,
    endpoint_node_id: str,
) -> list[tuple[str, float]]:
    """Endpoint nodes 1 hop away, sorted by edge weight descending.

    Both outgoing and incoming edges are considered.  Tool nodes are
    excluded from the results.
    """
    if endpoint_node_id not in graph:
        return []

    candidates: dict[str, float] = {}

    for _u, v, data in graph.out_edges(endpoint_node_id, data=True):
        if graph.nodes[v].get("node_type") == "endpoint":
            w = float(data.get("weight", 0.0))
            if v not in candidates or w > candidates[v]:
                candidates[v] = w

    for u, _v, data in graph.in_edges(endpoint_node_id, data=True):
        if graph.nodes[u].get("node_type") == "endpoint":
            w = float(data.get("weight", 0.0))
            if u not in candidates or w > candidates[u]:
                candidates[u] = w

    return sorted(candidates.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------


def compute_node_importance(graph: nx.DiGraph) -> dict[str, float]:
    """PageRank scores for all nodes. Empty dict for an empty graph."""
    if graph.number_of_nodes() == 0:
        return {}
    scores: dict[Any, float] = nx.pagerank(graph)
    return {str(k): float(v) for k, v in scores.items()}


def get_graph_stats(graph: nx.DiGraph) -> GraphStats:
    """Return :class:`GraphStats` — from cache or computed on the fly."""
    cached = graph.graph.get("stats")
    if isinstance(cached, GraphStats):
        return cached

    tool_count = 0
    ep_count = 0
    for _n, data in graph.nodes(data=True):
        nt = data.get("node_type")
        if nt == "tool":
            tool_count += 1
        elif nt == "endpoint":
            ep_count += 1

    edge_counts: dict[str, int] = {}
    for _u, _v, data in graph.edges(data=True):
        et = data.get("edge_type", "unknown")
        edge_counts[et] = edge_counts.get(et, 0) + 1

    density = float(nx.density(graph)) if graph.number_of_nodes() > 0 else 0.0
    components = (
        nx.number_weakly_connected_components(graph) if graph.number_of_nodes() > 0 else 0
    )

    return GraphStats(
        tool_node_count=tool_count,
        endpoint_node_count=ep_count,
        edge_counts=edge_counts,
        density=density,
        connected_components=components,
    )
