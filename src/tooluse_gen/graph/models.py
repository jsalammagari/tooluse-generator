"""Pydantic data models for the Tool Graph.

Hierarchy:
    ToolNode  ──SAME_TOOL──▶  EndpointNode
       │                          │
       └──SAME_DOMAIN─────────────┘
       └──SEMANTIC_SIMILARITY─────┘

Graph nodes wrap registry :class:`Tool` and :class:`Endpoint` objects with
graph-specific attributes (embeddings, quality tier, node IDs).  Edges
carry a typed weight in ``[0, 1]`` and optional metadata.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field

from tooluse_gen.registry.models import Endpoint, Tool

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EdgeType(str, Enum):
    """Types of edges in the tool graph."""

    SAME_DOMAIN = "same_domain"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    SAME_TOOL = "same_tool"


# ---------------------------------------------------------------------------
# Node models
# ---------------------------------------------------------------------------


def _quality_tier(completeness_score: float) -> str:
    """Map a completeness score to a quality tier label."""
    if completeness_score >= 0.7:
        return "high"
    if completeness_score >= 0.4:
        return "medium"
    return "low"


class ToolNode(BaseModel):
    """Represents a tool as a node in the graph."""

    model_config = ConfigDict(use_enum_values=True)

    tool_id: str = Field(..., description="Unique tool identifier.")
    name: str = Field(..., description="Human-readable tool name.")
    domain: str = Field(default="", description="Primary domain/category.")
    description: str = Field(default="", description="Tool description.")
    quality_tier: str = Field(
        default="unknown",
        description="Quality tier derived from completeness_score: high/medium/low.",
    )
    endpoint_count: int = Field(default=0, description="Number of endpoints.")
    embedding: list[float] | None = Field(
        default=None, description="Embedding vector for semantic similarity."
    )

    @computed_field
    @property
    def node_id(self) -> str:
        """Graph node identifier."""
        return f"tool:{self.tool_id}"

    @classmethod
    def from_tool(cls, tool: Tool) -> ToolNode:
        """Construct a :class:`ToolNode` from a registry :class:`Tool`."""
        return cls(
            tool_id=tool.tool_id,
            name=tool.name,
            domain=tool.domain,
            description=tool.description,
            quality_tier=_quality_tier(tool.completeness_score),
            endpoint_count=tool.endpoint_count,
        )


class EndpointNode(BaseModel):
    """Represents an endpoint as a node in the graph."""

    model_config = ConfigDict(use_enum_values=True)

    endpoint_id: str = Field(..., description="Unique endpoint identifier.")
    tool_id: str = Field(..., description="Parent tool identifier.")
    name: str = Field(..., description="Human-readable endpoint name.")
    description: str = Field(default="", description="Endpoint description.")
    method: str = Field(default="GET", description="HTTP method.")
    path: str = Field(default="", description="URL path template.")
    parameter_names: list[str] = Field(
        default_factory=list, description="Names of all parameters."
    )
    extractable_output_types: list[str] = Field(
        default_factory=list,
        description="Parameter types that could appear in responses, useful for chaining.",
    )
    embedding: list[float] | None = Field(
        default=None, description="Embedding vector for semantic similarity."
    )

    @computed_field
    @property
    def node_id(self) -> str:
        """Graph node identifier."""
        return f"ep:{self.tool_id}:{self.endpoint_id}"

    @classmethod
    def from_endpoint(cls, endpoint: Endpoint, tool: Tool) -> EndpointNode:
        """Construct an :class:`EndpointNode` from a registry :class:`Endpoint`."""
        return cls(
            endpoint_id=endpoint.endpoint_id,
            tool_id=tool.tool_id,
            name=endpoint.name,
            description=endpoint.description,
            method=endpoint.method if isinstance(endpoint.method, str) else endpoint.method,
            path=endpoint.path,
            parameter_names=[p.name for p in endpoint.parameters],
            extractable_output_types=sorted(
                {str(p.param_type) for p in endpoint.parameters}
            ),
        )


# ---------------------------------------------------------------------------
# Edge model
# ---------------------------------------------------------------------------


class GraphEdge(BaseModel):
    """A weighted, typed edge between two graph nodes."""

    model_config = ConfigDict(use_enum_values=True)

    source_id: str = Field(..., description="Source node identifier.")
    target_id: str = Field(..., description="Target node identifier.")
    edge_type: EdgeType = Field(..., description="Type of relationship.")
    weight: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Edge weight in [0, 1]."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Optional edge metadata."
    )


# ---------------------------------------------------------------------------
# Statistics & configuration
# ---------------------------------------------------------------------------


class GraphStats(BaseModel):
    """Summary statistics for a built tool graph."""

    model_config = ConfigDict(use_enum_values=True)

    tool_node_count: int = Field(default=0, description="Number of tool nodes.")
    endpoint_node_count: int = Field(default=0, description="Number of endpoint nodes.")
    edge_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Edge counts keyed by EdgeType value.",
    )
    density: float = Field(default=0.0, description="Graph density.")
    connected_components: int = Field(default=0, description="Number of connected components.")

    @computed_field
    @property
    def total_nodes(self) -> int:
        """Total number of nodes (tool + endpoint)."""
        return self.tool_node_count + self.endpoint_node_count

    @computed_field
    @property
    def total_edges(self) -> int:
        """Total number of edges across all types."""
        return sum(self.edge_counts.values())


class GraphConfig(BaseModel):
    """Configuration for graph construction."""

    model_config = ConfigDict(use_enum_values=True)

    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity for semantic edges.",
    )
    max_edges_per_node: int = Field(
        default=50,
        gt=0,
        description="Maximum edges per node after pruning.",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model name for embeddings.",
    )
    include_tool_nodes: bool = Field(
        default=True, description="Whether to include tool-level nodes."
    )
    include_domain_edges: bool = Field(
        default=True, description="Whether to add same-domain edges."
    )
    include_semantic_edges: bool = Field(
        default=True, description="Whether to add semantic similarity edges."
    )
