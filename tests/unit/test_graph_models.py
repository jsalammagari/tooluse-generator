"""Unit tests for Task 17 — Graph data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tooluse_gen.graph.models import (
    EdgeType,
    EndpointNode,
    GraphConfig,
    GraphEdge,
    GraphStats,
    ToolNode,
)
from tooluse_gen.registry.models import (
    Endpoint,
    HttpMethod,
    Parameter,
    ParameterType,
    Tool,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_tool() -> Tool:
    """A complete Tool with one endpoint and two parameters."""
    return Tool(
        tool_id="weather_api",
        name="Weather API",
        description="Get weather information.",
        domain="Weather",
        completeness_score=0.85,
        endpoints=[
            Endpoint(
                endpoint_id="weather_api/GET/abc12345",
                tool_id="weather_api",
                name="Get Current Weather",
                description="Current conditions for a location.",
                method=HttpMethod.GET,
                path="/weather/current",
                parameters=[
                    Parameter(
                        name="location",
                        description="City name.",
                        param_type=ParameterType.STRING,
                        required=True,
                    ),
                    Parameter(
                        name="units",
                        param_type=ParameterType.STRING,
                        default="celsius",
                    ),
                ],
                completeness_score=0.9,
            ),
        ],
    )


@pytest.fixture()
def sample_endpoint(sample_tool: Tool) -> Endpoint:
    return sample_tool.endpoints[0]


@pytest.fixture()
def medium_tool() -> Tool:
    """A tool with medium completeness."""
    return Tool(
        tool_id="mid_api",
        name="Mid API",
        completeness_score=0.5,
    )


@pytest.fixture()
def low_tool() -> Tool:
    """A tool with low completeness."""
    return Tool(
        tool_id="low_api",
        name="Low API",
        completeness_score=0.2,
    )


# ===========================================================================
# EdgeType
# ===========================================================================


class TestEdgeType:
    def test_all_values_exist(self) -> None:
        assert EdgeType.SAME_DOMAIN is not None
        assert EdgeType.SEMANTIC_SIMILARITY is not None
        assert EdgeType.SAME_TOOL is not None

    def test_values_are_strings(self) -> None:
        assert EdgeType.SAME_DOMAIN == "same_domain"
        assert EdgeType.SEMANTIC_SIMILARITY == "semantic_similarity"
        assert EdgeType.SAME_TOOL == "same_tool"

    def test_enum_count(self) -> None:
        assert len(EdgeType) == 3


# ===========================================================================
# ToolNode
# ===========================================================================


class TestToolNode:
    def test_required_fields_only(self) -> None:
        node = ToolNode(tool_id="t1", name="T1")
        assert node.tool_id == "t1"
        assert node.name == "T1"
        assert node.domain == ""
        assert node.description == ""
        assert node.quality_tier == "unknown"
        assert node.endpoint_count == 0
        assert node.embedding is None

    def test_all_fields(self) -> None:
        emb = [0.1, 0.2, 0.3]
        node = ToolNode(
            tool_id="t2",
            name="T2",
            domain="Finance",
            description="A finance API.",
            quality_tier="high",
            endpoint_count=5,
            embedding=emb,
        )
        assert node.domain == "Finance"
        assert node.embedding == emb

    def test_node_id_computed(self) -> None:
        node = ToolNode(tool_id="abc", name="ABC")
        assert node.node_id == "tool:abc"

    def test_from_tool_high(self, sample_tool: Tool) -> None:
        node = ToolNode.from_tool(sample_tool)
        assert node.tool_id == "weather_api"
        assert node.name == "Weather API"
        assert node.domain == "Weather"
        assert node.quality_tier == "high"
        assert node.endpoint_count == 1
        assert node.embedding is None

    def test_from_tool_medium(self, medium_tool: Tool) -> None:
        node = ToolNode.from_tool(medium_tool)
        assert node.quality_tier == "medium"

    def test_from_tool_low(self, low_tool: Tool) -> None:
        node = ToolNode.from_tool(low_tool)
        assert node.quality_tier == "low"

    def test_from_tool_boundary_high(self) -> None:
        tool = Tool(tool_id="b", name="B", completeness_score=0.7)
        assert ToolNode.from_tool(tool).quality_tier == "high"

    def test_from_tool_boundary_medium(self) -> None:
        tool = Tool(tool_id="b", name="B", completeness_score=0.4)
        assert ToolNode.from_tool(tool).quality_tier == "medium"

    def test_from_tool_boundary_low(self) -> None:
        tool = Tool(tool_id="b", name="B", completeness_score=0.39)
        assert ToolNode.from_tool(tool).quality_tier == "low"

    def test_serialization_round_trip(self) -> None:
        node = ToolNode(tool_id="rt", name="RT", domain="D", embedding=[1.0, 2.0])
        data = node.model_dump()
        restored = ToolNode.model_validate(data)
        assert restored.tool_id == node.tool_id
        assert restored.node_id == node.node_id
        assert restored.embedding == node.embedding

    def test_missing_tool_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            ToolNode(name="NoID")  # type: ignore[call-arg]

    def test_missing_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            ToolNode(tool_id="x")  # type: ignore[call-arg]

    def test_node_id_in_dump(self) -> None:
        node = ToolNode(tool_id="x", name="X")
        data = node.model_dump()
        assert "node_id" in data
        assert data["node_id"] == "tool:x"


# ===========================================================================
# EndpointNode
# ===========================================================================


class TestEndpointNode:
    def test_required_fields_only(self) -> None:
        node = EndpointNode(endpoint_id="e1", tool_id="t1", name="E1")
        assert node.endpoint_id == "e1"
        assert node.tool_id == "t1"
        assert node.name == "E1"
        assert node.description == ""
        assert node.method == "GET"
        assert node.path == ""
        assert node.parameter_names == []
        assert node.extractable_output_types == []
        assert node.embedding is None

    def test_node_id_computed(self) -> None:
        node = EndpointNode(endpoint_id="t1/GET/abc", tool_id="t1", name="EP")
        assert node.node_id == "ep:t1:t1/GET/abc"

    def test_from_endpoint(self, sample_endpoint: Endpoint, sample_tool: Tool) -> None:
        node = EndpointNode.from_endpoint(sample_endpoint, sample_tool)
        assert node.endpoint_id == sample_endpoint.endpoint_id
        assert node.tool_id == sample_tool.tool_id
        assert node.name == "Get Current Weather"
        assert node.method == "GET"
        assert node.path == "/weather/current"
        assert "location" in node.parameter_names
        assert "units" in node.parameter_names
        assert len(node.parameter_names) == 2

    def test_from_endpoint_extractable_types(
        self, sample_endpoint: Endpoint, sample_tool: Tool
    ) -> None:
        node = EndpointNode.from_endpoint(sample_endpoint, sample_tool)
        assert "string" in node.extractable_output_types

    def test_serialization_round_trip(self) -> None:
        node = EndpointNode(
            endpoint_id="e1",
            tool_id="t1",
            name="E1",
            parameter_names=["q", "limit"],
            embedding=[0.5],
        )
        data = node.model_dump()
        restored = EndpointNode.model_validate(data)
        assert restored.endpoint_id == "e1"
        assert restored.parameter_names == ["q", "limit"]
        assert restored.node_id == node.node_id

    def test_missing_endpoint_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            EndpointNode(tool_id="t1", name="E")  # type: ignore[call-arg]

    def test_from_endpoint_no_params(self) -> None:
        ep = Endpoint(
            endpoint_id="t1/GET/xyz",
            tool_id="t1",
            name="NoParams",
            method=HttpMethod.GET,
            path="/empty",
        )
        tool = Tool(tool_id="t1", name="T1")
        node = EndpointNode.from_endpoint(ep, tool)
        assert node.parameter_names == []
        assert node.extractable_output_types == []


# ===========================================================================
# GraphEdge
# ===========================================================================


class TestGraphEdge:
    def test_required_fields(self) -> None:
        edge = GraphEdge(
            source_id="a", target_id="b", edge_type=EdgeType.SAME_DOMAIN
        )
        assert edge.source_id == "a"
        assert edge.target_id == "b"
        assert edge.edge_type == "same_domain"

    def test_weight_default(self) -> None:
        edge = GraphEdge(
            source_id="a", target_id="b", edge_type=EdgeType.SAME_TOOL
        )
        assert edge.weight == 1.0

    def test_weight_zero_valid(self) -> None:
        edge = GraphEdge(
            source_id="a", target_id="b", edge_type=EdgeType.SAME_TOOL, weight=0.0
        )
        assert edge.weight == 0.0

    def test_weight_one_valid(self) -> None:
        edge = GraphEdge(
            source_id="a", target_id="b", edge_type=EdgeType.SAME_TOOL, weight=1.0
        )
        assert edge.weight == 1.0

    def test_weight_negative_invalid(self) -> None:
        with pytest.raises(ValidationError):
            GraphEdge(
                source_id="a", target_id="b", edge_type=EdgeType.SAME_TOOL, weight=-0.1
            )

    def test_weight_above_one_invalid(self) -> None:
        with pytest.raises(ValidationError):
            GraphEdge(
                source_id="a", target_id="b", edge_type=EdgeType.SAME_TOOL, weight=1.1
            )

    def test_metadata_default_empty(self) -> None:
        edge = GraphEdge(
            source_id="a", target_id="b", edge_type=EdgeType.SEMANTIC_SIMILARITY
        )
        assert edge.metadata == {}

    def test_metadata_custom(self) -> None:
        edge = GraphEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.SEMANTIC_SIMILARITY,
            metadata={"similarity": 0.85},
        )
        assert edge.metadata["similarity"] == 0.85

    def test_serialization_round_trip(self) -> None:
        edge = GraphEdge(
            source_id="x",
            target_id="y",
            edge_type=EdgeType.SAME_DOMAIN,
            weight=0.5,
            metadata={"note": "test"},
        )
        data = edge.model_dump()
        restored = GraphEdge.model_validate(data)
        assert restored.weight == 0.5
        assert restored.metadata["note"] == "test"


# ===========================================================================
# GraphStats
# ===========================================================================


class TestGraphStats:
    def test_defaults(self) -> None:
        stats = GraphStats()
        assert stats.tool_node_count == 0
        assert stats.endpoint_node_count == 0
        assert stats.edge_counts == {}
        assert stats.density == 0.0
        assert stats.connected_components == 0

    def test_total_nodes(self) -> None:
        stats = GraphStats(tool_node_count=3, endpoint_node_count=10)
        assert stats.total_nodes == 13

    def test_total_edges(self) -> None:
        stats = GraphStats(
            edge_counts={
                "same_domain": 5,
                "semantic_similarity": 10,
                "same_tool": 3,
            }
        )
        assert stats.total_edges == 18

    def test_total_edges_empty(self) -> None:
        stats = GraphStats()
        assert stats.total_edges == 0

    def test_all_edge_type_keys(self) -> None:
        counts = {et.value: 1 for et in EdgeType}
        stats = GraphStats(edge_counts=counts)
        assert stats.total_edges == 3

    def test_serialization_round_trip(self) -> None:
        stats = GraphStats(
            tool_node_count=2,
            endpoint_node_count=5,
            edge_counts={"same_tool": 5},
            density=0.3,
            connected_components=1,
        )
        data = stats.model_dump()
        restored = GraphStats.model_validate(data)
        assert restored.total_nodes == 7
        assert restored.total_edges == 5


# ===========================================================================
# GraphConfig
# ===========================================================================


class TestGraphConfig:
    def test_defaults(self) -> None:
        cfg = GraphConfig()
        assert cfg.similarity_threshold == 0.7
        assert cfg.max_edges_per_node == 50
        assert cfg.embedding_model == "all-MiniLM-L6-v2"
        assert cfg.include_tool_nodes is True
        assert cfg.include_domain_edges is True
        assert cfg.include_semantic_edges is True

    def test_threshold_zero_valid(self) -> None:
        cfg = GraphConfig(similarity_threshold=0.0)
        assert cfg.similarity_threshold == 0.0

    def test_threshold_one_valid(self) -> None:
        cfg = GraphConfig(similarity_threshold=1.0)
        assert cfg.similarity_threshold == 1.0

    def test_threshold_negative_invalid(self) -> None:
        with pytest.raises(ValidationError):
            GraphConfig(similarity_threshold=-0.1)

    def test_threshold_above_one_invalid(self) -> None:
        with pytest.raises(ValidationError):
            GraphConfig(similarity_threshold=1.1)

    def test_max_edges_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            GraphConfig(max_edges_per_node=0)

    def test_max_edges_negative_invalid(self) -> None:
        with pytest.raises(ValidationError):
            GraphConfig(max_edges_per_node=-1)

    def test_custom_embedding_model(self) -> None:
        cfg = GraphConfig(embedding_model="custom-model")
        assert cfg.embedding_model == "custom-model"

    def test_boolean_flags(self) -> None:
        cfg = GraphConfig(
            include_tool_nodes=False,
            include_domain_edges=False,
            include_semantic_edges=False,
        )
        assert cfg.include_tool_nodes is False
        assert cfg.include_domain_edges is False
        assert cfg.include_semantic_edges is False
