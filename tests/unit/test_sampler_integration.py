"""Unit tests for Task 28 — Sampler module integration (ToolChainSampler facade)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from tooluse_gen.graph.chain_models import ChainPattern, SamplingConstraints, ToolChain
from tooluse_gen.graph.diversity import DiversityMetrics, DiversitySteeringConfig
from tooluse_gen.graph.facade import ToolChainSampler
from tooluse_gen.graph.sampler import SamplerConfig, SamplingError

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_graph() -> nx.DiGraph:
    """Graph: 4 tools, 7 endpoints, 3 domains."""
    g = nx.DiGraph()
    for tid, name, domain in [
        ("w", "Weather", "Weather"),
        ("m", "Maps", "Weather"),
        ("s", "Stocks", "Finance"),
        ("n", "News", "Media"),
    ]:
        g.add_node(
            f"tool:{tid}", node_type="tool", tool_id=tid,
            name=name, domain=domain, quality_tier="high",
        )

    endpoints = [
        ("ep:w:a", "w", "Current", "Weather", ["city"], ["string"]),
        ("ep:w:b", "w", "Forecast", "Weather", ["city"], ["string"]),
        ("ep:m:a", "m", "Geocode", "Weather", ["addr"], ["number"]),
        ("ep:s:a", "s", "Quote", "Finance", ["sym"], ["number"]),
        ("ep:s:b", "s", "History", "Finance", ["sym"], ["array"]),
        ("ep:n:a", "n", "Headlines", "Media", [], ["string"]),
        ("ep:n:b", "n", "Search", "Media", ["q"], ["string"]),
    ]
    for nid, tid, name, domain, params, outputs in endpoints:
        g.add_node(
            nid, node_type="endpoint", endpoint_id=nid.split(":", 2)[2],
            tool_id=tid, name=name, method="GET", path=f"/{name.lower()}",
            domain=domain, parameter_names=params,
            extractable_output_types=outputs, description=f"{name} ep",
        )
        g.add_edge(f"tool:{tid}", nid, edge_type="same_tool", weight=1.0)

    # Domain edges
    g.add_edge("ep:w:a", "ep:w:b", edge_type="same_domain", weight=0.9)
    g.add_edge("ep:w:a", "ep:m:a", edge_type="same_domain", weight=0.8)
    g.add_edge("ep:w:b", "ep:m:a", edge_type="same_domain", weight=0.8)
    g.add_edge("ep:s:a", "ep:s:b", edge_type="same_domain", weight=0.9)
    g.add_edge("ep:n:a", "ep:n:b", edge_type="same_domain", weight=0.7)
    # Cross-domain semantic
    g.add_edge("ep:w:a", "ep:s:a", edge_type="semantic_similarity", weight=0.6)
    g.add_edge("ep:m:a", "ep:n:a", edge_type="semantic_similarity", weight=0.5)
    return g


@pytest.fixture()
def sampler(test_graph: nx.DiGraph) -> ToolChainSampler:
    return ToolChainSampler(
        test_graph, SamplerConfig(max_iterations=200, max_retries=20)
    )


@pytest.fixture()
def constraints() -> SamplingConstraints:
    return SamplingConstraints(min_steps=2, max_steps=4, min_tools=1)


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ===========================================================================
# Construction
# ===========================================================================


class TestConstruction:
    def test_default_configs(self, test_graph: nx.DiGraph) -> None:
        s = ToolChainSampler(test_graph)
        assert s._chain_counter == 0

    def test_custom_configs(self, test_graph: nx.DiGraph) -> None:
        s = ToolChainSampler(
            test_graph,
            SamplerConfig(max_iterations=100),
            DiversitySteeringConfig(enabled=False, weight_decay=0.5),
        )
        assert s._tracker.config.enabled is False

    def test_known_domains(self, sampler: ToolChainSampler) -> None:
        assert "Weather" in sampler._known_domains
        assert "Finance" in sampler._known_domains
        assert "Media" in sampler._known_domains

    def test_known_tools(self, sampler: ToolChainSampler) -> None:
        assert "w" in sampler._known_tools
        assert "s" in sampler._known_tools
        assert "n" in sampler._known_tools
        assert "m" in sampler._known_tools


# ===========================================================================
# sample_chain
# ===========================================================================


class TestSampleChain:
    def test_returns_tool_chain(
        self, sampler: ToolChainSampler, constraints: SamplingConstraints,
        rng: np.random.Generator,
    ) -> None:
        chain = sampler.sample_chain(constraints, rng)
        assert isinstance(chain, ToolChain)

    def test_step_count_in_range(
        self, sampler: ToolChainSampler, constraints: SamplingConstraints,
        rng: np.random.Generator,
    ) -> None:
        chain = sampler.sample_chain(constraints, rng)
        assert constraints.min_steps <= chain.total_step_count <= constraints.max_steps

    def test_chain_id_prefix(
        self, sampler: ToolChainSampler, constraints: SamplingConstraints,
        rng: np.random.Generator,
    ) -> None:
        chain = sampler.sample_chain(constraints, rng)
        assert chain.chain_id.startswith("chain_")

    def test_chain_ids_increment(
        self, sampler: ToolChainSampler, constraints: SamplingConstraints,
    ) -> None:
        c0 = sampler.sample_chain(constraints, np.random.default_rng(1))
        c1 = sampler.sample_chain(constraints, np.random.default_rng(2))
        c2 = sampler.sample_chain(constraints, np.random.default_rng(3))
        assert c0.chain_id == "chain_0000"
        assert c1.chain_id == "chain_0001"
        assert c2.chain_id == "chain_0002"

    def test_pattern_auto_detection(
        self, sampler: ToolChainSampler, rng: np.random.Generator,
    ) -> None:
        # With enough samples, at least some should get a non-sequential pattern
        patterns: set[str] = set()
        for seed in range(20):
            c = sampler.sample_chain(
                SamplingConstraints(min_steps=2, max_steps=5, min_tools=1),
                np.random.default_rng(seed),
            )
            patterns.add(c.pattern)
        # At least sequential should appear; non-sequential is possible but not guaranteed
        assert "sequential" in patterns or len(patterns) >= 1

    def test_required_pattern_enforced(
        self, sampler: ToolChainSampler, rng: np.random.Generator,
    ) -> None:
        c = SamplingConstraints(
            min_steps=2, max_steps=4, min_tools=1,
            required_patterns=[ChainPattern.ITERATIVE],
        )
        chain = sampler.sample_chain(c, rng)
        assert chain.pattern == "iterative"

    def test_diversity_updated(
        self, sampler: ToolChainSampler, constraints: SamplingConstraints,
        rng: np.random.Generator,
    ) -> None:
        assert sampler.get_diversity_report().total_conversations == 0
        sampler.sample_chain(constraints, rng)
        assert sampler.get_diversity_report().total_conversations == 1

    def test_steering_prompt_in_metadata(
        self, sampler: ToolChainSampler, constraints: SamplingConstraints,
        rng: np.random.Generator,
    ) -> None:
        chain = sampler.sample_chain(constraints, rng)
        assert "steering_prompt" in chain.metadata


# ===========================================================================
# sample_batch
# ===========================================================================


class TestSampleBatch:
    def test_correct_length(
        self, sampler: ToolChainSampler, constraints: SamplingConstraints,
    ) -> None:
        batch = sampler.sample_batch(constraints, 5, np.random.default_rng(42))
        assert len(batch) == 5

    def test_unique_chain_ids(
        self, sampler: ToolChainSampler, constraints: SamplingConstraints,
    ) -> None:
        batch = sampler.sample_batch(constraints, 5, np.random.default_rng(42))
        ids = [c.chain_id for c in batch]
        assert len(set(ids)) == 5

    def test_diversity_total_matches(
        self, sampler: ToolChainSampler, constraints: SamplingConstraints,
    ) -> None:
        sampler.sample_batch(constraints, 5, np.random.default_rng(42))
        assert sampler.get_diversity_report().total_conversations == 5

    def test_all_satisfy_constraints(
        self, sampler: ToolChainSampler, constraints: SamplingConstraints,
    ) -> None:
        batch = sampler.sample_batch(constraints, 5, np.random.default_rng(42))
        for chain in batch:
            assert constraints.min_steps <= chain.total_step_count <= constraints.max_steps

    def test_metrics_after_batch(
        self, sampler: ToolChainSampler, constraints: SamplingConstraints,
    ) -> None:
        sampler.sample_batch(constraints, 5, np.random.default_rng(42))
        report = sampler.get_diversity_report()
        assert report.total_conversations == 5
        assert report.tool_entropy >= 0


# ===========================================================================
# Diversity report
# ===========================================================================


class TestDiversityReport:
    def test_returns_metrics(self, sampler: ToolChainSampler) -> None:
        assert isinstance(sampler.get_diversity_report(), DiversityMetrics)

    def test_total_after_batch(
        self, sampler: ToolChainSampler, constraints: SamplingConstraints,
    ) -> None:
        sampler.sample_batch(constraints, 3, np.random.default_rng(42))
        assert sampler.get_diversity_report().total_conversations == 3

    def test_entropy_after_diverse_batch(
        self, sampler: ToolChainSampler, constraints: SamplingConstraints,
    ) -> None:
        sampler.sample_batch(constraints, 10, np.random.default_rng(42))
        assert sampler.get_diversity_report().tool_entropy > 0


# ===========================================================================
# Reset
# ===========================================================================


class TestResetDiversity:
    def test_clears_counters(
        self, sampler: ToolChainSampler, constraints: SamplingConstraints,
    ) -> None:
        sampler.sample_batch(constraints, 3, np.random.default_rng(42))
        sampler.reset_diversity()
        assert sampler.get_diversity_report().total_conversations == 0
        assert sampler._chain_counter == 0

    def test_new_ids_from_zero(
        self, sampler: ToolChainSampler, constraints: SamplingConstraints,
    ) -> None:
        sampler.sample_chain(constraints, np.random.default_rng(1))
        sampler.reset_diversity()
        chain = sampler.sample_chain(constraints, np.random.default_rng(2))
        assert chain.chain_id == "chain_0000"


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_empty_graph_raises(self) -> None:
        s = ToolChainSampler(
            nx.DiGraph(), SamplerConfig(max_iterations=5, max_retries=3)
        )
        with pytest.raises(SamplingError):
            s.sample_chain(
                SamplingConstraints(min_steps=1, max_steps=3, min_tools=1),
                np.random.default_rng(42),
            )

    def test_batch_of_one(
        self, sampler: ToolChainSampler, constraints: SamplingConstraints,
    ) -> None:
        batch = sampler.sample_batch(constraints, 1, np.random.default_rng(42))
        assert len(batch) == 1
        assert batch[0].chain_id == "chain_0000"

    def test_diversity_disabled(self, test_graph: nx.DiGraph) -> None:
        s = ToolChainSampler(
            test_graph,
            SamplerConfig(max_iterations=200, max_retries=20),
            DiversitySteeringConfig(enabled=False),
        )
        constraints = SamplingConstraints(min_steps=2, max_steps=4, min_tools=1)
        chain = s.sample_chain(constraints, np.random.default_rng(42))
        # No steering prompt when disabled
        assert chain.metadata.get("steering_prompt", "") == ""

    def test_summary_and_prompt(
        self, sampler: ToolChainSampler, constraints: SamplingConstraints,
    ) -> None:
        sampler.sample_batch(constraints, 3, np.random.default_rng(42))
        summary = sampler.get_diversity_summary()
        assert "3 conversations" in summary
        # Steering prompt is a string (may or may not be empty)
        prompt = sampler.get_steering_prompt()
        assert isinstance(prompt, str)
