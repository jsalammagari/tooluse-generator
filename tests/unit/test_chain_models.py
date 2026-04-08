"""Unit tests for Task 23 — Chain pattern models."""

from __future__ import annotations

import networkx as nx
import pytest
from pydantic import ValidationError

from tooluse_gen.graph.chain_models import (
    ChainPattern,
    ChainStep,
    ParallelGroup,
    SamplingConstraints,
    ToolChain,
)
from tooluse_gen.registry.completeness import QualityTier
from tooluse_gen.registry.models import HttpMethod

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_step(
    endpoint_id: str = "ep1",
    tool_id: str = "t1",
    tool_name: str = "Tool1",
    endpoint_name: str = "Get Data",
    domain: str = "TestDomain",
    **kwargs: object,
) -> ChainStep:
    return ChainStep(
        endpoint_id=endpoint_id,
        tool_id=tool_id,
        tool_name=tool_name,
        endpoint_name=endpoint_name,
        domain=domain,
        **kwargs,  # type: ignore[arg-type]
    )


def _make_graph() -> nx.DiGraph:
    """Small graph with one tool and two endpoints."""
    g = nx.DiGraph()
    g.add_node(
        "tool:weather",
        node_type="tool",
        tool_id="weather",
        name="Weather API",
        domain="Weather",
    )
    g.add_node(
        "ep:weather:w/GET/cur",
        node_type="endpoint",
        endpoint_id="w/GET/cur",
        tool_id="weather",
        name="Current",
        method="GET",
        path="/current",
        parameter_names=["city", "units"],
        extractable_output_types=["string", "number"],
        description="Get current weather.",
        domain="Weather",
    )
    g.add_node(
        "ep:weather:w/GET/fore",
        node_type="endpoint",
        endpoint_id="w/GET/fore",
        tool_id="weather",
        name="Forecast",
        method="GET",
        path="/forecast",
        parameter_names=["city"],
        extractable_output_types=["string"],
        description="5-day forecast.",
        domain="Weather",
    )
    g.add_edge("tool:weather", "ep:weather:w/GET/cur", edge_type="same_tool", weight=1.0)
    g.add_edge("tool:weather", "ep:weather:w/GET/fore", edge_type="same_tool", weight=1.0)
    return g


# ===========================================================================
# ChainPattern
# ===========================================================================


class TestChainPattern:
    def test_all_values_exist(self) -> None:
        assert ChainPattern.SEQUENTIAL is not None
        assert ChainPattern.PARALLEL is not None
        assert ChainPattern.BRANCH_AND_MERGE is not None
        assert ChainPattern.ITERATIVE is not None

    def test_values_are_strings(self) -> None:
        assert ChainPattern.SEQUENTIAL == "sequential"
        assert ChainPattern.PARALLEL == "parallel"
        assert ChainPattern.BRANCH_AND_MERGE == "branch_and_merge"
        assert ChainPattern.ITERATIVE == "iterative"

    def test_count(self) -> None:
        assert len(ChainPattern) == 4


# ===========================================================================
# ChainStep
# ===========================================================================


class TestChainStep:
    def test_required_fields_only(self) -> None:
        step = _make_step()
        assert step.endpoint_id == "ep1"
        assert step.tool_id == "t1"
        assert step.tool_name == "Tool1"
        assert step.endpoint_name == "Get Data"
        assert step.method == "GET"
        assert step.path == ""
        assert step.expected_params == []
        assert step.expected_output_types == []
        assert step.description == ""
        assert step.domain == "TestDomain"

    def test_all_fields(self) -> None:
        step = _make_step(
            method=HttpMethod.POST,
            path="/data",
            expected_params=["q", "limit"],
            expected_output_types=["string", "integer"],
            description="Search data.",
        )
        assert step.method == "POST"
        assert step.path == "/data"
        assert step.expected_params == ["q", "limit"]
        assert step.expected_output_types == ["string", "integer"]
        assert step.description == "Search data."

    def test_from_graph_node(self) -> None:
        g = _make_graph()
        step = ChainStep.from_graph_node(g, "ep:weather:w/GET/cur")
        assert step.endpoint_id == "w/GET/cur"
        assert step.tool_id == "weather"
        assert step.tool_name == "Weather API"
        assert step.endpoint_name == "Current"
        assert step.method == "GET"
        assert step.path == "/current"
        assert step.expected_params == ["city", "units"]
        assert step.expected_output_types == ["string", "number"]
        assert step.description == "Get current weather."
        assert step.domain == "Weather"

    def test_from_graph_node_missing_tool(self) -> None:
        g = nx.DiGraph()
        g.add_node(
            "ep:orphan:o/GET/x",
            node_type="endpoint",
            endpoint_id="o/GET/x",
            tool_id="orphan",
            name="Orphan EP",
            method="GET",
            path="/x",
            parameter_names=[],
            extractable_output_types=[],
            description="",
            domain="D",
        )
        step = ChainStep.from_graph_node(g, "ep:orphan:o/GET/x")
        assert step.tool_name == "orphan"  # fallback to tool_id

    def test_serialization_round_trip(self) -> None:
        step = _make_step(expected_params=["a"], expected_output_types=["string"])
        data = step.model_dump()
        restored = ChainStep.model_validate(data)
        assert restored.endpoint_id == step.endpoint_id
        assert restored.expected_params == step.expected_params

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            ChainStep(endpoint_id="e1")  # type: ignore[call-arg]


# ===========================================================================
# ParallelGroup
# ===========================================================================


class TestParallelGroup:
    def test_two_steps(self) -> None:
        pg = ParallelGroup(steps=[_make_step(endpoint_id="a"), _make_step(endpoint_id="b")])
        assert len(pg.steps) == 2

    def test_three_steps(self) -> None:
        pg = ParallelGroup(
            steps=[
                _make_step(endpoint_id="a"),
                _make_step(endpoint_id="b"),
                _make_step(endpoint_id="c"),
            ]
        )
        assert len(pg.steps) == 3

    def test_less_than_two_raises(self) -> None:
        with pytest.raises(ValidationError, match="at least 2"):
            ParallelGroup(steps=[_make_step()])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValidationError):
            ParallelGroup(steps=[])

    def test_step_count(self) -> None:
        pg = ParallelGroup(steps=[_make_step(endpoint_id="a"), _make_step(endpoint_id="b")])
        assert pg.step_count == 2

    def test_tool_ids_sorted_unique(self) -> None:
        pg = ParallelGroup(
            steps=[
                _make_step(endpoint_id="a", tool_id="t2"),
                _make_step(endpoint_id="b", tool_id="t1"),
                _make_step(endpoint_id="c", tool_id="t2"),
            ]
        )
        assert pg.tool_ids == ["t1", "t2"]

    def test_endpoint_ids_ordered(self) -> None:
        pg = ParallelGroup(
            steps=[_make_step(endpoint_id="x"), _make_step(endpoint_id="y")]
        )
        assert pg.endpoint_ids == ["x", "y"]

    def test_serialization_round_trip(self) -> None:
        pg = ParallelGroup(steps=[_make_step(endpoint_id="a"), _make_step(endpoint_id="b")])
        data = pg.model_dump()
        restored = ParallelGroup.model_validate(data)
        assert restored.step_count == 2


# ===========================================================================
# ToolChain
# ===========================================================================


class TestToolChain:
    def test_sequential_steps(self) -> None:
        chain = ToolChain(
            steps=[_make_step(endpoint_id="a"), _make_step(endpoint_id="b")],
            pattern=ChainPattern.SEQUENTIAL,
        )
        assert chain.total_step_count == 2

    def test_with_parallel_group(self) -> None:
        pg = ParallelGroup(
            steps=[
                _make_step(endpoint_id="b", tool_id="t2", domain="D2"),
                _make_step(endpoint_id="c", tool_id="t3", domain="D3"),
            ]
        )
        chain = ToolChain(
            steps=[_make_step(endpoint_id="a"), pg],
            pattern=ChainPattern.BRANCH_AND_MERGE,
        )
        assert chain.total_step_count == 3

    def test_domains_involved(self) -> None:
        pg = ParallelGroup(
            steps=[
                _make_step(endpoint_id="b", domain="Finance"),
                _make_step(endpoint_id="c", domain="Media"),
            ]
        )
        chain = ToolChain(
            steps=[_make_step(endpoint_id="a", domain="Weather"), pg],
            pattern=ChainPattern.BRANCH_AND_MERGE,
        )
        assert chain.domains_involved == ["Finance", "Media", "Weather"]

    def test_tool_ids(self) -> None:
        chain = ToolChain(
            steps=[
                _make_step(endpoint_id="a", tool_id="t2"),
                _make_step(endpoint_id="b", tool_id="t1"),
            ],
            pattern=ChainPattern.SEQUENTIAL,
        )
        assert chain.tool_ids == ["t1", "t2"]

    def test_endpoint_ids_flat(self) -> None:
        pg = ParallelGroup(
            steps=[_make_step(endpoint_id="b"), _make_step(endpoint_id="c")]
        )
        chain = ToolChain(
            steps=[_make_step(endpoint_id="a"), pg],
            pattern=ChainPattern.BRANCH_AND_MERGE,
        )
        assert chain.endpoint_ids == ["a", "b", "c"]

    def test_is_multi_tool_true(self) -> None:
        chain = ToolChain(
            steps=[
                _make_step(tool_id="t1"),
                _make_step(endpoint_id="e2", tool_id="t2"),
            ],
            pattern=ChainPattern.SEQUENTIAL,
        )
        assert chain.is_multi_tool is True

    def test_is_multi_tool_false(self) -> None:
        chain = ToolChain(
            steps=[_make_step(tool_id="t1"), _make_step(endpoint_id="e2", tool_id="t1")],
            pattern=ChainPattern.SEQUENTIAL,
        )
        assert chain.is_multi_tool is False

    def test_is_cross_domain_true(self) -> None:
        chain = ToolChain(
            steps=[
                _make_step(domain="Weather"),
                _make_step(endpoint_id="e2", domain="Finance"),
            ],
            pattern=ChainPattern.SEQUENTIAL,
        )
        assert chain.is_cross_domain is True

    def test_is_cross_domain_false(self) -> None:
        chain = ToolChain(
            steps=[
                _make_step(domain="Weather"),
                _make_step(endpoint_id="e2", domain="Weather"),
            ],
            pattern=ChainPattern.SEQUENTIAL,
        )
        assert chain.is_cross_domain is False

    def test_chain_id_default_empty(self) -> None:
        chain = ToolChain(
            steps=[_make_step()], pattern=ChainPattern.SEQUENTIAL
        )
        assert chain.chain_id == ""

    def test_chain_id_custom(self) -> None:
        chain = ToolChain(
            steps=[_make_step()],
            pattern=ChainPattern.SEQUENTIAL,
            chain_id="chain_001",
        )
        assert chain.chain_id == "chain_001"

    def test_metadata_default_empty(self) -> None:
        chain = ToolChain(steps=[_make_step()], pattern=ChainPattern.SEQUENTIAL)
        assert chain.metadata == {}

    def test_metadata_custom(self) -> None:
        chain = ToolChain(
            steps=[_make_step()],
            pattern=ChainPattern.SEQUENTIAL,
            metadata={"score": 0.95},
        )
        assert chain.metadata["score"] == 0.95

    def test_serialization_round_trip(self) -> None:
        pg = ParallelGroup(
            steps=[
                _make_step(endpoint_id="b", tool_id="t2", domain="D2"),
                _make_step(endpoint_id="c", tool_id="t3", domain="D3"),
            ]
        )
        chain = ToolChain(
            chain_id="test",
            steps=[_make_step(endpoint_id="a"), pg],
            pattern=ChainPattern.BRANCH_AND_MERGE,
            metadata={"k": "v"},
        )
        data = chain.model_dump()
        restored = ToolChain.model_validate(data)
        assert restored.chain_id == "test"
        assert restored.total_step_count == 3
        assert restored.is_multi_tool is True


# ===========================================================================
# SamplingConstraints
# ===========================================================================


class TestSamplingConstraints:
    def test_defaults(self) -> None:
        sc = SamplingConstraints()
        assert sc.domains is None
        assert sc.min_steps == 2
        assert sc.max_steps == 5
        assert sc.min_tools == 2
        assert sc.required_tools is None
        assert sc.excluded_tools is None
        assert sc.required_patterns is None
        assert sc.quality_threshold == "fair"

    def test_min_steps_ge_1(self) -> None:
        with pytest.raises(ValidationError):
            SamplingConstraints(min_steps=0)

    def test_max_steps_ge_min_steps(self) -> None:
        with pytest.raises(ValidationError, match="max_steps"):
            SamplingConstraints(min_steps=5, max_steps=3)

    def test_max_steps_equal_min_steps(self) -> None:
        sc = SamplingConstraints(min_steps=3, max_steps=3)
        assert sc.min_steps == 3 and sc.max_steps == 3

    def test_quality_threshold_default_fair(self) -> None:
        sc = SamplingConstraints()
        assert sc.quality_threshold == QualityTier.FAIR.value

    def test_custom_values(self) -> None:
        sc = SamplingConstraints(
            domains=["Finance", "Weather"],
            min_steps=3,
            max_steps=8,
            min_tools=3,
            required_tools=["stocks"],
            excluded_tools=["news"],
            required_patterns=[ChainPattern.SEQUENTIAL],
            quality_threshold=QualityTier.GOOD,
        )
        assert sc.domains == ["Finance", "Weather"]
        assert sc.min_steps == 3
        assert sc.max_steps == 8
        assert sc.required_tools == ["stocks"]
        assert sc.excluded_tools == ["news"]
        assert sc.quality_threshold == "good"

    def test_serialization_round_trip(self) -> None:
        sc = SamplingConstraints(
            domains=["D1"], min_steps=1, max_steps=10, min_tools=1
        )
        data = sc.model_dump()
        restored = SamplingConstraints.model_validate(data)
        assert restored.domains == ["D1"]
        assert restored.max_steps == 10
