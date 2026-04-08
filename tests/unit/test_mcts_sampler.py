"""Unit tests for Task 24 — MCTS sampler core."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest
from pydantic import ValidationError

from tooluse_gen.graph.chain_models import ChainPattern, SamplingConstraints, ToolChain
from tooluse_gen.graph.sampler import MCTSNode, MCTSSampler, SamplerConfig, SamplingError

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_graph() -> nx.DiGraph:
    """Graph: 4 tools, 8 endpoints, 3 domains, rich connectivity."""
    g = nx.DiGraph()
    for tid, name, domain, tier in [
        ("weather", "Weather API", "Weather", "high"),
        ("maps", "Maps API", "Weather", "medium"),
        ("stocks", "Stock API", "Finance", "high"),
        ("news", "News API", "Media", "medium"),
    ]:
        g.add_node(
            f"tool:{tid}",
            node_type="tool",
            tool_id=tid,
            name=name,
            domain=domain,
            quality_tier=tier,
        )

    endpoints = [
        ("ep:weather:w/GET/cur", "weather", "Current", "GET", "/cur", "Weather", ["city"], ["string"]),
        ("ep:weather:w/GET/fore", "weather", "Forecast", "GET", "/fore", "Weather", ["city"], ["string"]),
        ("ep:maps:m/GET/geo", "maps", "Geocode", "GET", "/geo", "Weather", ["addr"], ["number"]),
        ("ep:maps:m/GET/rev", "maps", "Reverse", "GET", "/rev", "Weather", ["lat", "lon"], ["string"]),
        ("ep:stocks:s/GET/q", "stocks", "Quote", "GET", "/q", "Finance", ["sym"], ["number"]),
        ("ep:stocks:s/GET/h", "stocks", "History", "GET", "/h", "Finance", ["sym"], ["array"]),
        ("ep:news:n/GET/top", "news", "Headlines", "GET", "/top", "Media", [], ["string"]),
        ("ep:news:n/GET/s", "news", "Search", "GET", "/s", "Media", ["q"], ["string"]),
    ]
    for nid, tid, name, method, path, domain, params, outputs in endpoints:
        g.add_node(
            nid,
            node_type="endpoint",
            endpoint_id=nid.split(":", 2)[2],
            tool_id=tid,
            name=name,
            method=method,
            path=path,
            domain=domain,
            parameter_names=params,
            extractable_output_types=outputs,
            description=f"{name} endpoint",
        )
        g.add_edge(f"tool:{tid}", nid, edge_type="same_tool", weight=1.0)

    # Domain edges
    weather_eps = [e[0] for e in endpoints if e[5] == "Weather"]
    for i, a in enumerate(weather_eps):
        for b in weather_eps[i + 1 :]:
            g.add_edge(a, b, edge_type="same_domain", weight=0.8)
    finance_eps = [e[0] for e in endpoints if e[5] == "Finance"]
    g.add_edge(finance_eps[0], finance_eps[1], edge_type="same_domain", weight=0.9)
    media_eps = [e[0] for e in endpoints if e[5] == "Media"]
    g.add_edge(media_eps[0], media_eps[1], edge_type="same_domain", weight=0.7)

    # Cross-domain semantic edges
    g.add_edge(
        "ep:weather:w/GET/cur", "ep:stocks:s/GET/q",
        edge_type="semantic_similarity", weight=0.6,
    )
    g.add_edge(
        "ep:maps:m/GET/geo", "ep:news:n/GET/s",
        edge_type="semantic_similarity", weight=0.5,
    )
    g.add_edge(
        "ep:stocks:s/GET/q", "ep:news:n/GET/top",
        edge_type="semantic_similarity", weight=0.4,
    )
    return g


@pytest.fixture()
def sampler(test_graph: nx.DiGraph) -> MCTSSampler:
    return MCTSSampler(test_graph, SamplerConfig(max_iterations=300, max_retries=30))


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ===========================================================================
# SamplerConfig
# ===========================================================================


class TestSamplerConfig:
    def test_defaults(self) -> None:
        cfg = SamplerConfig()
        assert cfg.exploration_weight == pytest.approx(1.414)
        assert cfg.max_iterations == 1000
        assert cfg.max_depth == 7
        assert cfg.rollout_depth == 3
        assert cfg.max_retries == 50

    def test_custom(self) -> None:
        cfg = SamplerConfig(exploration_weight=2.0, max_iterations=500)
        assert cfg.exploration_weight == 2.0
        assert cfg.max_iterations == 500

    def test_exploration_weight_positive(self) -> None:
        with pytest.raises(ValidationError):
            SamplerConfig(exploration_weight=0.0)

    def test_max_iterations_positive(self) -> None:
        with pytest.raises(ValidationError):
            SamplerConfig(max_iterations=0)


# ===========================================================================
# MCTSNode
# ===========================================================================


class TestMCTSNode:
    def test_construction(self) -> None:
        node = MCTSNode(state=["ep1", "ep2"])
        assert node.state == ["ep1", "ep2"]
        assert node.parent is None
        assert node.children == []
        assert node.visits == 0
        assert node.reward == 0.0

    def test_is_terminal(self) -> None:
        node = MCTSNode(state=["ep1"], untried_actions=[])
        assert node.is_terminal is True

    def test_not_terminal_with_children(self) -> None:
        parent = MCTSNode(state=["ep1"])
        child = MCTSNode(state=["ep1", "ep2"], parent=parent)
        parent.children.append(child)
        assert parent.is_terminal is False

    def test_is_fully_expanded(self) -> None:
        node = MCTSNode(state=["ep1"], untried_actions=[])
        assert node.is_fully_expanded is True
        node2 = MCTSNode(state=["ep1"], untried_actions=["ep2"])
        assert node2.is_fully_expanded is False

    def test_ucb1_unvisited(self) -> None:
        node = MCTSNode(state=["ep1"])
        assert node.ucb1(1.414) == float("inf")

    def test_ucb1_visited(self) -> None:
        parent = MCTSNode(state=[])
        parent.visits = 10
        child = MCTSNode(state=["ep1"], parent=parent)
        child.visits = 5
        child.reward = 3.0
        score = child.ucb1(1.414)
        expected = 3.0 / 5 + 1.414 * math.sqrt(math.log(10) / 5)
        assert score == pytest.approx(expected)

    def test_best_child(self) -> None:
        parent = MCTSNode(state=[])
        parent.visits = 10
        c1 = MCTSNode(state=["a"], parent=parent)
        c1.visits = 5
        c1.reward = 2.0
        c2 = MCTSNode(state=["b"], parent=parent)
        c2.visits = 3
        c2.reward = 2.5
        parent.children = [c1, c2]
        best = parent.best_child(1.414)
        assert best is c2  # higher reward/visit ratio + exploration bonus


# ===========================================================================
# Reward function
# ===========================================================================


class TestComputeReward:
    def test_step_count_bonus(self, sampler: MCTSSampler, test_graph: nx.DiGraph) -> None:
        chain = ["ep:weather:w/GET/cur", "ep:weather:w/GET/fore"]
        constraints = SamplingConstraints(min_steps=2, max_steps=5, min_tools=1)
        reward = sampler._compute_reward(chain, constraints)
        assert reward >= 1.0  # at least the step-count bonus

    def test_multi_tool_bonus(self, sampler: MCTSSampler) -> None:
        chain = ["ep:weather:w/GET/cur", "ep:stocks:s/GET/q"]
        constraints = SamplingConstraints(min_steps=2, max_steps=5, min_tools=1)
        reward = sampler._compute_reward(chain, constraints)
        # 2 unique tools → 2 * 0.5 = 1.0 tool bonus
        assert reward >= 2.0

    def test_multi_domain_bonus(self, sampler: MCTSSampler) -> None:
        chain = ["ep:weather:w/GET/cur", "ep:stocks:s/GET/q", "ep:news:n/GET/top"]
        constraints = SamplingConstraints(min_steps=2, max_steps=5, min_tools=1)
        reward = sampler._compute_reward(chain, constraints)
        # 3 domains → 3 * 0.3 = 0.9 domain bonus
        assert reward >= 2.5

    def test_edge_coherence_bonus(self, sampler: MCTSSampler) -> None:
        # These two share a same_domain edge
        chain = ["ep:weather:w/GET/cur", "ep:weather:w/GET/fore"]
        constraints = SamplingConstraints(min_steps=2, max_steps=5, min_tools=1)
        reward = sampler._compute_reward(chain, constraints)
        # Should include 0.2 coherence bonus
        assert reward > 1.5

    def test_constraint_violation_penalty(self, sampler: MCTSSampler) -> None:
        chain = ["ep:weather:w/GET/cur"]
        constraints = SamplingConstraints(min_steps=3, max_steps=5, min_tools=2)
        reward = sampler._compute_reward(chain, constraints)
        # Too few steps (-0.5), too few tools (-0.5), no step bonus
        assert reward < 1.0


# ===========================================================================
# Constraint checking
# ===========================================================================


class TestCheckConstraints:
    def test_valid_chain(self, sampler: MCTSSampler) -> None:
        chain = ["ep:weather:w/GET/cur", "ep:stocks:s/GET/q"]
        constraints = SamplingConstraints(min_steps=2, max_steps=5, min_tools=2)
        assert sampler._check_constraints(chain, constraints) is True

    def test_too_few_steps(self, sampler: MCTSSampler) -> None:
        chain = ["ep:weather:w/GET/cur"]
        constraints = SamplingConstraints(min_steps=2, max_steps=5, min_tools=1)
        assert sampler._check_constraints(chain, constraints) is False

    def test_too_many_steps(self, sampler: MCTSSampler) -> None:
        chain = [
            "ep:weather:w/GET/cur", "ep:weather:w/GET/fore",
            "ep:maps:m/GET/geo", "ep:maps:m/GET/rev",
        ]
        constraints = SamplingConstraints(min_steps=2, max_steps=3, min_tools=1)
        assert sampler._check_constraints(chain, constraints) is False

    def test_too_few_tools(self, sampler: MCTSSampler) -> None:
        chain = ["ep:weather:w/GET/cur", "ep:weather:w/GET/fore"]
        constraints = SamplingConstraints(min_steps=2, max_steps=5, min_tools=3)
        assert sampler._check_constraints(chain, constraints) is False

    def test_excluded_tool(self, sampler: MCTSSampler) -> None:
        chain = ["ep:weather:w/GET/cur", "ep:stocks:s/GET/q"]
        constraints = SamplingConstraints(
            min_steps=2, max_steps=5, min_tools=1, excluded_tools=["stocks"]
        )
        assert sampler._check_constraints(chain, constraints) is False

    def test_required_tool_missing(self, sampler: MCTSSampler) -> None:
        chain = ["ep:weather:w/GET/cur", "ep:weather:w/GET/fore"]
        constraints = SamplingConstraints(
            min_steps=2, max_steps=5, min_tools=1, required_tools=["stocks"]
        )
        assert sampler._check_constraints(chain, constraints) is False

    def test_domain_constraint(self, sampler: MCTSSampler) -> None:
        chain = ["ep:weather:w/GET/cur", "ep:stocks:s/GET/q"]
        constraints = SamplingConstraints(
            min_steps=2, max_steps=5, min_tools=1, domains=["Weather"]
        )
        # stocks is Finance, not Weather
        assert sampler._check_constraints(chain, constraints) is False

    def test_empty_chain(self, sampler: MCTSSampler) -> None:
        constraints = SamplingConstraints(min_steps=1, max_steps=5, min_tools=1)
        assert sampler._check_constraints([], constraints) is False


# ===========================================================================
# Candidate actions
# ===========================================================================


class TestCandidateActions:
    def test_returns_neighbors(self, sampler: MCTSSampler) -> None:
        state = ["ep:weather:w/GET/cur"]
        constraints = SamplingConstraints(min_steps=2, max_steps=5, min_tools=1)
        actions = sampler._get_candidate_actions(state, constraints)
        assert len(actions) > 0
        # Should include domain neighbor and semantic neighbor
        assert "ep:weather:w/GET/fore" in actions or "ep:stocks:s/GET/q" in actions

    def test_excludes_visited(self, sampler: MCTSSampler) -> None:
        state = ["ep:weather:w/GET/cur", "ep:weather:w/GET/fore"]
        constraints = SamplingConstraints(min_steps=2, max_steps=5, min_tools=1)
        actions = sampler._get_candidate_actions(state, constraints)
        assert "ep:weather:w/GET/cur" not in actions
        assert "ep:weather:w/GET/fore" not in actions

    def test_excludes_excluded_tools(self, sampler: MCTSSampler) -> None:
        state = ["ep:weather:w/GET/cur"]
        constraints = SamplingConstraints(
            min_steps=2, max_steps=5, min_tools=1, excluded_tools=["stocks"]
        )
        actions = sampler._get_candidate_actions(state, constraints)
        for a in actions:
            assert sampler._graph.nodes[a].get("tool_id") != "stocks"

    def test_empty_at_max_steps(self, sampler: MCTSSampler) -> None:
        state = ["ep:weather:w/GET/cur", "ep:weather:w/GET/fore"]
        constraints = SamplingConstraints(min_steps=1, max_steps=2, min_tools=1)
        actions = sampler._get_candidate_actions(state, constraints)
        assert actions == []


# ===========================================================================
# Sampling
# ===========================================================================


class TestSampling:
    def test_returns_tool_chain(
        self, sampler: MCTSSampler, rng: np.random.Generator
    ) -> None:
        constraints = SamplingConstraints(min_steps=2, max_steps=5, min_tools=1)
        chain = sampler.sample(constraints, rng)
        assert isinstance(chain, ToolChain)

    def test_step_count_in_range(
        self, sampler: MCTSSampler, rng: np.random.Generator
    ) -> None:
        constraints = SamplingConstraints(min_steps=2, max_steps=4, min_tools=1)
        chain = sampler.sample(constraints, rng)
        assert 2 <= chain.total_step_count <= 4

    def test_min_tools_satisfied(
        self, sampler: MCTSSampler, rng: np.random.Generator
    ) -> None:
        constraints = SamplingConstraints(min_steps=2, max_steps=5, min_tools=2)
        chain = sampler.sample(constraints, rng)
        assert len(chain.tool_ids) >= 2

    def test_domain_constraint(
        self, sampler: MCTSSampler, rng: np.random.Generator
    ) -> None:
        constraints = SamplingConstraints(
            min_steps=2, max_steps=5, min_tools=1, domains=["Weather"]
        )
        chain = sampler.sample(constraints, rng)
        assert all(d == "Weather" for d in chain.domains_involved)

    def test_excluded_tool(
        self, sampler: MCTSSampler, rng: np.random.Generator
    ) -> None:
        constraints = SamplingConstraints(
            min_steps=2, max_steps=5, min_tools=1, excluded_tools=["stocks"]
        )
        chain = sampler.sample(constraints, rng)
        assert "stocks" not in chain.tool_ids

    def test_deterministic(self, test_graph: nx.DiGraph) -> None:
        config = SamplerConfig(max_iterations=200, max_retries=20)
        sampler = MCTSSampler(test_graph, config)
        constraints = SamplingConstraints(min_steps=2, max_steps=4, min_tools=1)
        c1 = sampler.sample(constraints, np.random.default_rng(999))
        c2 = sampler.sample(constraints, np.random.default_rng(999))
        assert c1.endpoint_ids == c2.endpoint_ids

    def test_pattern_is_sequential(
        self, sampler: MCTSSampler, rng: np.random.Generator
    ) -> None:
        constraints = SamplingConstraints(min_steps=2, max_steps=5, min_tools=1)
        chain = sampler.sample(constraints, rng)
        assert chain.pattern == ChainPattern.SEQUENTIAL.value


# ===========================================================================
# Fallback & error
# ===========================================================================


class TestFallback:
    def test_random_walk_produces_chain(self, test_graph: nx.DiGraph) -> None:
        sampler = MCTSSampler(test_graph, SamplerConfig(max_iterations=1, max_retries=30))
        constraints = SamplingConstraints(min_steps=2, max_steps=5, min_tools=1)
        chain = sampler.sample(constraints, np.random.default_rng(42))
        assert isinstance(chain, ToolChain)
        assert chain.total_step_count >= 2

    def test_sampling_error_impossible(self, test_graph: nx.DiGraph) -> None:
        sampler = MCTSSampler(
            test_graph, SamplerConfig(max_iterations=10, max_retries=5)
        )
        # Require a tool that doesn't exist
        constraints = SamplingConstraints(
            min_steps=2, max_steps=5, min_tools=1, required_tools=["nonexistent_tool"]
        )
        with pytest.raises(SamplingError):
            sampler.sample(constraints, np.random.default_rng(42))

    def test_fallback_respects_excluded(self, test_graph: nx.DiGraph) -> None:
        sampler = MCTSSampler(
            test_graph, SamplerConfig(max_iterations=5, max_retries=30)
        )
        constraints = SamplingConstraints(
            min_steps=2, max_steps=5, min_tools=1, excluded_tools=["news"]
        )
        chain = sampler.sample(constraints, np.random.default_rng(42))
        assert "news" not in chain.tool_ids

    def test_empty_graph_raises(self) -> None:
        sampler = MCTSSampler(nx.DiGraph(), SamplerConfig(max_iterations=5, max_retries=3))
        constraints = SamplingConstraints(min_steps=1, max_steps=3, min_tools=1)
        with pytest.raises(SamplingError):
            sampler.sample(constraints, np.random.default_rng(42))
