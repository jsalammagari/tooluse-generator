"""Graph builder — constructs a NetworkX DiGraph from a ToolRegistry.

The :class:`GraphBuilder` creates tool and endpoint nodes, then connects
them via three edge types:

* **SAME_TOOL** — parent-child link between a tool and its endpoints.
* **SAME_DOMAIN** — connects endpoints that share a domain.
* **SEMANTIC_SIMILARITY** — connects nodes whose embedding cosine
  similarity exceeds the configured threshold.

After edge creation the builder optionally prunes low-weight edges and
attaches :class:`GraphStats` to the graph metadata.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import networkx as nx

from tooluse_gen.graph.embeddings import (
    EmbeddingService,
    build_endpoint_description,
    build_tool_description,
)
from tooluse_gen.graph.models import (
    EdgeType,
    EndpointNode,
    GraphConfig,
    GraphStats,
    ToolNode,
)
from tooluse_gen.registry.registry import ToolRegistry
from tooluse_gen.utils.logging import get_logger

logger = get_logger("graph.builder")

# Quality-tier → numeric weight for domain edges
_TIER_WEIGHT: dict[str, float] = {
    "high": 0.9,
    "medium": 0.6,
    "low": 0.3,
    "unknown": 0.1,
}


class GraphBuilder:
    """Constructs a NetworkX DiGraph from a :class:`ToolRegistry`."""

    def __init__(
        self,
        config: GraphConfig,
        embedding_service: EmbeddingService,
    ) -> None:
        self._config = config
        self._embedding_service = embedding_service

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, registry: ToolRegistry) -> nx.DiGraph:
        """Build the tool graph from *registry* and return it."""
        graph: nx.DiGraph = nx.DiGraph()

        if self._config.include_tool_nodes:
            self._add_tool_nodes(graph, registry)

        self._add_endpoint_nodes(graph, registry)

        if self._config.include_domain_edges:
            self._add_domain_edges(graph)

        if self._config.include_semantic_edges:
            node_ids, embeddings = self._compute_embeddings(graph, registry)
            if node_ids:
                self._add_semantic_edges(graph, node_ids, embeddings)

        self._prune_edges(graph)

        stats = self._compute_stats(graph)
        graph.graph["stats"] = stats

        logger.info(
            "Graph built: %d nodes (%d tools, %d endpoints), %d edges, %d components",
            stats.total_nodes,
            stats.tool_node_count,
            stats.endpoint_node_count,
            stats.total_edges,
            stats.connected_components,
        )
        return graph

    # ------------------------------------------------------------------
    # Node creation
    # ------------------------------------------------------------------

    def _add_tool_nodes(self, graph: nx.DiGraph, registry: ToolRegistry) -> None:
        """Add tool-level nodes to *graph*."""
        for tool in registry.tools():
            tn = ToolNode.from_tool(tool)
            attrs: dict[str, Any] = {
                "node_type": "tool",
                "tool_id": tn.tool_id,
                "name": tn.name,
                "domain": tn.domain,
                "description": tn.description,
                "quality_tier": tn.quality_tier,
                "endpoint_count": tn.endpoint_count,
            }
            graph.add_node(tn.node_id, **attrs)

    def _add_endpoint_nodes(self, graph: nx.DiGraph, registry: ToolRegistry) -> None:
        """Add endpoint nodes and optional SAME_TOOL edges."""
        for tool in registry.tools():
            tn = ToolNode.from_tool(tool)
            for endpoint in tool.endpoints:
                en = EndpointNode.from_endpoint(endpoint, tool)
                attrs: dict[str, Any] = {
                    "node_type": "endpoint",
                    "endpoint_id": en.endpoint_id,
                    "tool_id": en.tool_id,
                    "name": en.name,
                    "description": en.description,
                    "method": en.method,
                    "path": en.path,
                    "parameter_names": en.parameter_names,
                    "extractable_output_types": en.extractable_output_types,
                    "domain": tool.domain,
                    "quality_tier": tn.quality_tier,
                }
                graph.add_node(en.node_id, **attrs)

                if self._config.include_tool_nodes:
                    graph.add_edge(
                        tn.node_id,
                        en.node_id,
                        edge_type=EdgeType.SAME_TOOL.value,
                        weight=1.0,
                    )

    # ------------------------------------------------------------------
    # Edge creation
    # ------------------------------------------------------------------

    def _add_domain_edges(self, graph: nx.DiGraph) -> None:
        """Connect endpoint nodes that share a domain."""
        domain_groups: dict[str, list[str]] = defaultdict(list)
        for node_id, data in graph.nodes(data=True):
            if data.get("node_type") == "endpoint" and data.get("domain"):
                domain_groups[data["domain"]].append(node_id)

        for _domain, node_ids in domain_groups.items():
            if len(node_ids) < 2:
                continue
            for i in range(len(node_ids)):
                for j in range(i + 1, len(node_ids)):
                    a, b = node_ids[i], node_ids[j]
                    if graph.has_edge(a, b):
                        continue
                    tier_a = graph.nodes[a].get("quality_tier", "unknown")
                    tier_b = graph.nodes[b].get("quality_tier", "unknown")
                    w = (_TIER_WEIGHT.get(tier_a, 0.1) + _TIER_WEIGHT.get(tier_b, 0.1)) / 2.0
                    graph.add_edge(
                        a,
                        b,
                        edge_type=EdgeType.SAME_DOMAIN.value,
                        weight=w,
                    )

    def _add_semantic_edges(
        self,
        graph: nx.DiGraph,
        node_ids: list[str],
        embeddings: list[list[float]],
    ) -> None:
        """Add edges where cosine similarity exceeds the threshold."""
        sim_matrix = self._embedding_service.compute_similarity_matrix(embeddings)
        threshold = self._config.similarity_threshold

        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                sim = float(sim_matrix[i, j])
                if sim >= threshold:
                    a, b = node_ids[i], node_ids[j]
                    if not graph.has_edge(a, b):
                        graph.add_edge(
                            a,
                            b,
                            edge_type=EdgeType.SEMANTIC_SIMILARITY.value,
                            weight=sim,
                        )

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def _compute_embeddings(
        self,
        graph: nx.DiGraph,
        registry: ToolRegistry,
    ) -> tuple[list[str], list[list[float]]]:
        """Compute embeddings for all nodes and return parallel lists."""
        node_ids: list[str] = []
        texts: list[str] = []

        # Endpoint descriptions
        for tool in registry.tools():
            for endpoint in tool.endpoints:
                en = EndpointNode.from_endpoint(endpoint, tool)
                node_ids.append(en.node_id)
                texts.append(build_endpoint_description(endpoint, tool))

        # Tool descriptions (if tool nodes included)
        if self._config.include_tool_nodes:
            for tool in registry.tools():
                tn = ToolNode.from_tool(tool)
                node_ids.append(tn.node_id)
                texts.append(build_tool_description(tool))

        if not texts:
            return [], []

        embeddings = self._embedding_service.embed_batch(texts, show_progress=False)

        # Store embeddings on node data
        for nid, emb in zip(node_ids, embeddings, strict=True):
            if nid in graph:
                graph.nodes[nid]["embedding"] = emb

        return node_ids, embeddings

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def _prune_edges(self, graph: nx.DiGraph) -> None:
        """Remove lowest-weight edges exceeding ``max_edges_per_node``."""
        max_edges = self._config.max_edges_per_node
        edges_to_remove: set[tuple[str, str]] = set()

        for node in graph.nodes():
            # Collect all edges (in + out) for this node
            all_edges: list[tuple[str, str, dict[str, Any]]] = []
            for u, v, d in graph.out_edges(node, data=True):
                all_edges.append((u, v, d))
            for u, v, d in graph.in_edges(node, data=True):
                all_edges.append((u, v, d))

            if len(all_edges) <= max_edges:
                continue

            # Separate SAME_TOOL (never pruned) from the rest
            protected = [
                (u, v) for u, v, d in all_edges if d.get("edge_type") == EdgeType.SAME_TOOL.value
            ]
            prunable = [
                (u, v, d)
                for u, v, d in all_edges
                if d.get("edge_type") != EdgeType.SAME_TOOL.value
            ]

            remaining_budget = max_edges - len(protected)
            if remaining_budget < 0:
                remaining_budget = 0

            if len(prunable) <= remaining_budget:
                continue

            # Sort by weight descending, keep the top ones
            prunable.sort(key=lambda x: x[2].get("weight", 0.0), reverse=True)
            for u, v, _d in prunable[remaining_budget:]:
                edges_to_remove.add((u, v))

        for u, v in edges_to_remove:
            if graph.has_edge(u, v):
                graph.remove_edge(u, v)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def _compute_stats(self, graph: nx.DiGraph) -> GraphStats:
        """Compute summary statistics from the built graph."""
        tool_count = 0
        endpoint_count = 0
        for _n, data in graph.nodes(data=True):
            nt = data.get("node_type")
            if nt == "tool":
                tool_count += 1
            elif nt == "endpoint":
                endpoint_count += 1

        edge_counts: dict[str, int] = defaultdict(int)
        for _u, _v, data in graph.edges(data=True):
            et = data.get("edge_type", "unknown")
            edge_counts[et] += 1

        density = float(nx.density(graph)) if graph.number_of_nodes() > 0 else 0.0
        components = (
            nx.number_weakly_connected_components(graph)
            if graph.number_of_nodes() > 0
            else 0
        )

        return GraphStats(
            tool_node_count=tool_count,
            endpoint_node_count=endpoint_count,
            edge_counts=dict(edge_counts),
            density=density,
            connected_components=components,
        )
