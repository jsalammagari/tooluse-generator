"""Unit tests for Task 25 — Chain pattern detection, enforcement, and description."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from tooluse_gen.graph.chain_models import (
    ChainPattern,
    ChainStep,
    ParallelGroup,
    ToolChain,
)
from tooluse_gen.graph.patterns import (
    PatternDetector,
    PatternEnforcer,
    chain_to_description,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _step(
    endpoint_id: str = "ep1",
    tool_id: str = "t1",
    tool_name: str = "Tool1",
    endpoint_name: str = "Action",
    domain: str = "Domain1",
    expected_params: list[str] | None = None,
    expected_output_types: list[str] | None = None,
    method: str = "GET",
    **kw: object,
) -> ChainStep:
    return ChainStep(
        endpoint_id=endpoint_id,
        tool_id=tool_id,
        tool_name=tool_name,
        endpoint_name=endpoint_name,
        domain=domain,
        expected_params=expected_params or [],
        expected_output_types=expected_output_types or [],
        method=method,
        **kw,  # type: ignore[arg-type]
    )


def _chain(
    steps: list[ChainStep | ParallelGroup],
    pattern: ChainPattern = ChainPattern.SEQUENTIAL,
) -> ToolChain:
    return ToolChain(steps=steps, pattern=pattern)


@pytest.fixture()
def graph() -> nx.DiGraph:
    """Small graph: 3 tools, 5 endpoints, 2 domains."""
    g = nx.DiGraph()
    g.add_node("tool:w", node_type="tool", tool_id="w", name="Weather", domain="Weather")
    g.add_node("tool:m", node_type="tool", tool_id="m", name="Maps", domain="Weather")
    g.add_node("tool:s", node_type="tool", tool_id="s", name="Stocks", domain="Finance")

    for nid, tid, name, dom, params, outputs in [
        ("ep:w:a", "w", "Current", "Weather", ["city"], ["string"]),
        ("ep:w:b", "w", "Forecast", "Weather", ["city"], ["string"]),
        ("ep:m:a", "m", "Geocode", "Weather", ["addr"], ["number"]),
        ("ep:s:a", "s", "Quote", "Finance", ["sym"], ["number"]),
        ("ep:s:b", "s", "History", "Finance", ["sym"], ["array"]),
    ]:
        g.add_node(
            nid, node_type="endpoint", endpoint_id=nid.split(":", 2)[2],
            tool_id=tid, name=name, method="GET", path=f"/{name.lower()}",
            domain=dom, parameter_names=params, extractable_output_types=outputs,
            description=f"{name} ep",
        )
        g.add_edge(f"tool:{tid}", nid, edge_type="same_tool", weight=1.0)

    g.add_edge("ep:w:a", "ep:w:b", edge_type="same_domain", weight=0.9)
    g.add_edge("ep:w:a", "ep:m:a", edge_type="same_domain", weight=0.8)
    g.add_edge("ep:w:b", "ep:m:a", edge_type="same_domain", weight=0.8)
    g.add_edge("ep:s:a", "ep:s:b", edge_type="same_domain", weight=0.9)
    g.add_edge("ep:w:a", "ep:s:a", edge_type="semantic_similarity", weight=0.6)
    return g


# ===========================================================================
# PatternDetector — parallel
# ===========================================================================


class TestDetectParallel:
    def test_same_domain_no_dependency(self, graph: nx.DiGraph) -> None:
        s1 = _step(endpoint_id="a", tool_id="w", domain="Weather",
                    expected_params=["city"], expected_output_types=["string"])
        s2 = _step(endpoint_id="b", tool_id="w", endpoint_name="Forecast", domain="Weather",
                    expected_params=["city"], expected_output_types=["string"])
        chain = _chain([s1, s2])
        result = PatternDetector(graph).detect_parallel_opportunities(chain)
        assert any(isinstance(s, ParallelGroup) for s in result.steps)
        assert result.pattern == ChainPattern.PARALLEL.value

    def test_dependency_blocks_parallel(self, graph: nx.DiGraph) -> None:
        s1 = _step(endpoint_id="a", tool_id="w", domain="Weather",
                    expected_output_types=["city_name"])
        s2 = _step(endpoint_id="b", tool_id="w", domain="Weather",
                    expected_params=["city_name"])
        chain = _chain([s1, s2])
        result = PatternDetector(graph).detect_parallel_opportunities(chain)
        assert not any(isinstance(s, ParallelGroup) for s in result.steps)

    def test_three_independent_merged(self, graph: nx.DiGraph) -> None:
        steps = [
            _step(endpoint_id="a", tool_id="w", domain="Weather"),
            _step(endpoint_id="b", tool_id="w", endpoint_name="B", domain="Weather"),
            _step(endpoint_id="c", tool_id="m", endpoint_name="C", domain="Weather"),
        ]
        chain = _chain(steps)
        result = PatternDetector(graph).detect_parallel_opportunities(chain)
        groups = [s for s in result.steps if isinstance(s, ParallelGroup)]
        assert len(groups) == 1
        assert groups[0].step_count == 3

    def test_existing_parallel_group_skipped(self, graph: nx.DiGraph) -> None:
        s1 = _step(endpoint_id="a", domain="Weather")
        s2 = _step(endpoint_id="b", endpoint_name="B", domain="Weather")
        pg = ParallelGroup(steps=[s1, s2])
        chain = _chain([pg])
        result = PatternDetector(graph).detect_parallel_opportunities(chain)
        assert result.steps == chain.steps

    def test_single_step_unchanged(self, graph: nx.DiGraph) -> None:
        chain = _chain([_step()])
        result = PatternDetector(graph).detect_parallel_opportunities(chain)
        assert len(result.steps) == 1

    def test_pattern_set_to_parallel(self, graph: nx.DiGraph) -> None:
        s1 = _step(endpoint_id="a", domain="D")
        s2 = _step(endpoint_id="b", endpoint_name="B", domain="D")
        chain = _chain([s1, s2])
        result = PatternDetector(graph).detect_parallel_opportunities(chain)
        if any(isinstance(s, ParallelGroup) for s in result.steps):
            assert result.pattern == ChainPattern.PARALLEL.value


# ===========================================================================
# PatternDetector — branch_and_merge
# ===========================================================================


class TestDetectBranchAndMerge:
    def test_branch_created(self, graph: nx.DiGraph) -> None:
        s1 = _step(endpoint_id="a", tool_id="w", tool_name="Weather",
                    endpoint_name="Current", domain="Weather")
        s2 = _step(endpoint_id="b", tool_id="w", tool_name="Weather",
                    endpoint_name="Forecast", domain="Weather")
        chain = _chain([s1, s2])
        result = PatternDetector(graph).detect_branch_and_merge(chain)
        assert result.pattern == ChainPattern.BRANCH_AND_MERGE.value
        assert any(isinstance(s, ParallelGroup) for s in result.steps)

    def test_no_graph_neighbors(self) -> None:
        g = nx.DiGraph()
        s1 = _step(endpoint_id="x", tool_id="t")
        s2 = _step(endpoint_id="y", tool_id="t", endpoint_name="Y")
        chain = _chain([s1, s2])
        result = PatternDetector(g).detect_branch_and_merge(chain)
        assert result.pattern == ChainPattern.SEQUENTIAL.value

    def test_pattern_set(self, graph: nx.DiGraph) -> None:
        s1 = _step(endpoint_id="a", tool_id="w", tool_name="Weather",
                    endpoint_name="Current", domain="Weather")
        s2 = _step(endpoint_id="b", tool_id="w", tool_name="Weather",
                    endpoint_name="Forecast", domain="Weather")
        chain = _chain([s1, s2])
        result = PatternDetector(graph).detect_branch_and_merge(chain)
        assert result.pattern == ChainPattern.BRANCH_AND_MERGE.value

    def test_only_first_opportunity(self, graph: nx.DiGraph) -> None:
        s1 = _step(endpoint_id="a", tool_id="w", tool_name="Weather",
                    endpoint_name="Current", domain="Weather")
        s2 = _step(endpoint_id="b", tool_id="w", tool_name="Weather",
                    endpoint_name="Forecast", domain="Weather")
        s3 = _step(endpoint_id="c", tool_id="s", tool_name="Stocks",
                    endpoint_name="Quote", domain="Finance")
        chain = _chain([s1, s2, s3])
        result = PatternDetector(graph).detect_branch_and_merge(chain)
        groups = [s for s in result.steps if isinstance(s, ParallelGroup)]
        assert len(groups) == 1

    def test_branch_contains_original_and_alt(self, graph: nx.DiGraph) -> None:
        s1 = _step(endpoint_id="a", tool_id="w", tool_name="Weather",
                    endpoint_name="Current", domain="Weather")
        s2 = _step(endpoint_id="b", tool_id="w", tool_name="Weather",
                    endpoint_name="Forecast", domain="Weather")
        chain = _chain([s1, s2])
        result = PatternDetector(graph).detect_branch_and_merge(chain)
        for item in result.steps:
            if isinstance(item, ParallelGroup):
                assert item.step_count >= 2
                ids = [s.endpoint_id for s in item.steps]
                assert "b" in ids  # original next step preserved


# ===========================================================================
# PatternDetector — iterative
# ===========================================================================


class TestDetectIterative:
    def test_same_pair_twice(self, graph: nx.DiGraph) -> None:
        s1 = _step(endpoint_id="a", tool_id="w", endpoint_name="Current")
        s2 = _step(endpoint_id="b", tool_id="w", endpoint_name="Current")
        chain = _chain([s1, s2])
        result = PatternDetector(graph).detect_iterative(chain)
        assert result.pattern == ChainPattern.ITERATIVE.value

    def test_same_tool_different_endpoint(self, graph: nx.DiGraph) -> None:
        s1 = _step(endpoint_id="a", tool_id="w", endpoint_name="Current", method="GET")
        s2 = _step(endpoint_id="b", tool_id="w", endpoint_name="Forecast", method="GET")
        chain = _chain([s1, s2])
        result = PatternDetector(graph).detect_iterative(chain)
        assert result.pattern == ChainPattern.ITERATIVE.value

    def test_no_repetition(self, graph: nx.DiGraph) -> None:
        s1 = _step(endpoint_id="a", tool_id="w", endpoint_name="Current", method="GET")
        s2 = _step(endpoint_id="b", tool_id="s", endpoint_name="Quote", method="POST")
        chain = _chain([s1, s2])
        result = PatternDetector(graph).detect_iterative(chain)
        assert result.pattern == ChainPattern.SEQUENTIAL.value

    def test_pattern_set(self, graph: nx.DiGraph) -> None:
        s1 = _step(endpoint_id="a", tool_id="w", endpoint_name="X")
        s2 = _step(endpoint_id="b", tool_id="w", endpoint_name="X")
        chain = _chain([s1, s2])
        result = PatternDetector(graph).detect_iterative(chain)
        assert result.pattern == ChainPattern.ITERATIVE.value


# ===========================================================================
# PatternEnforcer — sequential
# ===========================================================================


class TestEnforceSequential:
    def test_flattens_parallel_groups(self, graph: nx.DiGraph) -> None:
        s1 = _step(endpoint_id="a")
        s2 = _step(endpoint_id="b", endpoint_name="B")
        s3 = _step(endpoint_id="c", endpoint_name="C")
        pg = ParallelGroup(steps=[s2, s3])
        chain = _chain([s1, pg], ChainPattern.PARALLEL)
        rng = np.random.default_rng(42)
        result = PatternEnforcer(graph).enforce_pattern(chain, ChainPattern.SEQUENTIAL, rng)
        assert all(isinstance(s, ChainStep) for s in result.steps)
        assert result.total_step_count == 3
        assert result.pattern == ChainPattern.SEQUENTIAL.value

    def test_already_sequential(self, graph: nx.DiGraph) -> None:
        chain = _chain([_step(endpoint_id="a"), _step(endpoint_id="b", endpoint_name="B")])
        rng = np.random.default_rng(42)
        result = PatternEnforcer(graph).enforce_pattern(chain, ChainPattern.SEQUENTIAL, rng)
        assert result.pattern == ChainPattern.SEQUENTIAL.value
        assert len(result.steps) == 2

    def test_pattern_set(self, graph: nx.DiGraph) -> None:
        chain = _chain([_step()], ChainPattern.ITERATIVE)
        rng = np.random.default_rng(42)
        result = PatternEnforcer(graph).enforce_pattern(chain, ChainPattern.SEQUENTIAL, rng)
        assert result.pattern == ChainPattern.SEQUENTIAL.value


# ===========================================================================
# PatternEnforcer — parallel
# ===========================================================================


class TestEnforceParallel:
    def test_groups_independent_steps(self, graph: nx.DiGraph) -> None:
        s1 = _step(endpoint_id="a", tool_id="w", domain="Weather")
        s2 = _step(endpoint_id="b", tool_id="w", endpoint_name="B", domain="Weather")
        chain = _chain([s1, s2])
        rng = np.random.default_rng(42)
        result = PatternEnforcer(graph).enforce_pattern(chain, ChainPattern.PARALLEL, rng)
        assert result.pattern == ChainPattern.PARALLEL.value

    def test_no_opportunities_still_parallel(self, graph: nx.DiGraph) -> None:
        s1 = _step(endpoint_id="a", tool_id="w", domain="D1")
        s2 = _step(endpoint_id="b", tool_id="s", domain="D2")
        chain = _chain([s1, s2])
        rng = np.random.default_rng(42)
        result = PatternEnforcer(graph).enforce_pattern(chain, ChainPattern.PARALLEL, rng)
        assert result.pattern == ChainPattern.PARALLEL.value

    def test_preserves_existing_groups(self, graph: nx.DiGraph) -> None:
        s1 = _step(endpoint_id="a", domain="D")
        s2 = _step(endpoint_id="b", endpoint_name="B", domain="D")
        pg = ParallelGroup(steps=[s1, s2])
        chain = _chain([pg])
        rng = np.random.default_rng(42)
        result = PatternEnforcer(graph).enforce_pattern(chain, ChainPattern.PARALLEL, rng)
        assert any(isinstance(s, ParallelGroup) for s in result.steps)


# ===========================================================================
# PatternEnforcer — branch_and_merge
# ===========================================================================


class TestEnforceBranchAndMerge:
    def test_inserts_branch(self, graph: nx.DiGraph) -> None:
        s1 = _step(endpoint_id="a", tool_id="w", tool_name="Weather",
                    endpoint_name="Current", domain="Weather")
        s2 = _step(endpoint_id="b", tool_id="w", tool_name="Weather",
                    endpoint_name="Forecast", domain="Weather")
        chain = _chain([s1, s2])
        rng = np.random.default_rng(42)
        result = PatternEnforcer(graph).enforce_pattern(
            chain, ChainPattern.BRANCH_AND_MERGE, rng
        )
        assert result.pattern == ChainPattern.BRANCH_AND_MERGE.value

    def test_pattern_set(self, graph: nx.DiGraph) -> None:
        chain = _chain([_step(endpoint_id="a", tool_id="w", domain="Weather")])
        rng = np.random.default_rng(42)
        result = PatternEnforcer(graph).enforce_pattern(
            chain, ChainPattern.BRANCH_AND_MERGE, rng
        )
        assert result.pattern == ChainPattern.BRANCH_AND_MERGE.value

    def test_works_with_rng(self, graph: nx.DiGraph) -> None:
        s1 = _step(endpoint_id="a", tool_id="w", tool_name="Weather",
                    endpoint_name="Current", domain="Weather")
        s2 = _step(endpoint_id="b", tool_id="w", tool_name="Weather",
                    endpoint_name="Forecast", domain="Weather")
        chain = _chain([s1, s2])
        r1 = PatternEnforcer(graph).enforce_pattern(
            chain, ChainPattern.BRANCH_AND_MERGE, np.random.default_rng(42)
        )
        r2 = PatternEnforcer(graph).enforce_pattern(
            chain, ChainPattern.BRANCH_AND_MERGE, np.random.default_rng(42)
        )
        assert r1.total_step_count == r2.total_step_count


# ===========================================================================
# PatternEnforcer — iterative
# ===========================================================================


class TestEnforceIterative:
    def test_duplicates_step(self, graph: nx.DiGraph) -> None:
        s1 = _step(endpoint_id="a")
        s2 = _step(endpoint_id="b", endpoint_name="B")
        chain = _chain([s1, s2])
        rng = np.random.default_rng(42)
        result = PatternEnforcer(graph).enforce_pattern(chain, ChainPattern.ITERATIVE, rng)
        assert result.total_step_count == 3

    def test_count_increases_by_one(self, graph: nx.DiGraph) -> None:
        steps = [_step(endpoint_id=f"e{i}", endpoint_name=f"A{i}") for i in range(3)]
        chain = _chain(steps)
        rng = np.random.default_rng(42)
        result = PatternEnforcer(graph).enforce_pattern(chain, ChainPattern.ITERATIVE, rng)
        assert result.total_step_count == chain.total_step_count + 1

    def test_pattern_set(self, graph: nx.DiGraph) -> None:
        chain = _chain([_step()])
        rng = np.random.default_rng(42)
        result = PatternEnforcer(graph).enforce_pattern(chain, ChainPattern.ITERATIVE, rng)
        assert result.pattern == ChainPattern.ITERATIVE.value


# ===========================================================================
# chain_to_description
# ===========================================================================


class TestChainToDescription:
    def test_sequential(self) -> None:
        chain = _chain([
            _step(tool_name="Weather", endpoint_name="Current"),
            _step(endpoint_id="e2", tool_name="Stocks", endpoint_name="Quote"),
        ])
        desc = chain_to_description(chain)
        assert desc == "Weather: Current -> Stocks: Quote"

    def test_with_parallel_group(self) -> None:
        s1 = _step(tool_name="W", endpoint_name="A")
        s2 = _step(endpoint_id="e2", tool_name="S", endpoint_name="B")
        s3 = _step(endpoint_id="e3", tool_name="N", endpoint_name="C")
        pg = ParallelGroup(steps=[s2, s3])
        chain = _chain([s1, pg], ChainPattern.BRANCH_AND_MERGE)
        desc = chain_to_description(chain)
        assert desc == "W: A -> [S: B, N: C]"

    def test_single_step(self) -> None:
        chain = _chain([_step(tool_name="T", endpoint_name="X")])
        assert chain_to_description(chain) == "T: X"

    def test_empty_chain(self) -> None:
        chain = _chain([])
        assert chain_to_description(chain) == "(empty chain)"

    def test_mixed_steps_and_groups(self) -> None:
        s1 = _step(tool_name="A", endpoint_name="1")
        s2 = _step(endpoint_id="e2", tool_name="B", endpoint_name="2")
        s3 = _step(endpoint_id="e3", tool_name="C", endpoint_name="3")
        s4 = _step(endpoint_id="e4", tool_name="D", endpoint_name="4")
        pg = ParallelGroup(steps=[s2, s3])
        chain = _chain([s1, pg, s4])
        desc = chain_to_description(chain)
        assert desc == "A: 1 -> [B: 2, C: 3] -> D: 4"

    def test_arrow_separator(self) -> None:
        chain = _chain([
            _step(tool_name="X", endpoint_name="A"),
            _step(endpoint_id="e2", tool_name="Y", endpoint_name="B"),
            _step(endpoint_id="e3", tool_name="Z", endpoint_name="C"),
        ])
        desc = chain_to_description(chain)
        assert desc.count(" -> ") == 2
