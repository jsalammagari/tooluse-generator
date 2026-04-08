"""MCTS-based tool-chain sampler.

Uses Monte Carlo Tree Search to traverse the tool graph and produce
:class:`ToolChain` instances that satisfy :class:`SamplingConstraints`.
Falls back to a weighted random walk when MCTS cannot find a valid chain
within the configured iteration budget.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from tooluse_gen.graph.chain_models import (
    ChainPattern,
    ChainStep,
    SamplingConstraints,
    ToolChain,
)
from tooluse_gen.utils.logging import get_logger

logger = get_logger("graph.sampler")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SamplingError(Exception):
    """Raised when the sampler cannot produce a valid chain."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class SamplerConfig(BaseModel):
    """Configuration for the MCTS sampler."""

    model_config = ConfigDict(use_enum_values=True)

    exploration_weight: float = Field(default=1.414, gt=0.0, description="UCB1 constant.")
    max_iterations: int = Field(default=1000, gt=0, description="MCTS iterations per search.")
    max_depth: int = Field(default=7, gt=0, description="Max chain length during search.")
    rollout_depth: int = Field(default=3, gt=0, description="Random simulation depth.")
    max_retries: int = Field(default=50, gt=0, description="Rejection sampling retries.")


# ---------------------------------------------------------------------------
# MCTS tree node
# ---------------------------------------------------------------------------


class MCTSNode:
    """A node in the MCTS search tree."""

    __slots__ = ("state", "parent", "children", "visits", "reward", "untried_actions")

    def __init__(
        self,
        state: list[str],
        parent: MCTSNode | None = None,
        untried_actions: list[str] | None = None,
    ) -> None:
        self.state: list[str] = state
        self.parent: MCTSNode | None = parent
        self.children: list[MCTSNode] = []
        self.visits: int = 0
        self.reward: float = 0.0
        self.untried_actions: list[str] = list(untried_actions or [])

    @property
    def is_terminal(self) -> bool:
        return len(self.untried_actions) == 0 and len(self.children) == 0

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def ucb1(self, exploration_weight: float) -> float:
        if self.visits == 0:
            return float("inf")
        exploitation = self.reward / self.visits
        parent_visits = self.parent.visits if self.parent is not None else self.visits
        exploration = exploration_weight * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration

    def best_child(self, exploration_weight: float) -> MCTSNode:
        return max(self.children, key=lambda c: c.ucb1(exploration_weight))


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class MCTSSampler:
    """MCTS-based chain sampler over a tool graph."""

    def __init__(
        self,
        graph: nx.DiGraph,
        config: SamplerConfig | None = None,
    ) -> None:
        self._graph = graph
        self._config = config or SamplerConfig()
        self._endpoint_nodes: list[str] = [
            n for n, d in graph.nodes(data=True) if d.get("node_type") == "endpoint"
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(
        self,
        constraints: SamplingConstraints,
        rng: np.random.Generator,
    ) -> ToolChain:
        """Sample a tool chain satisfying *constraints*."""
        for attempt in range(self._config.max_retries):
            chain = self._mcts_search(constraints, rng)
            if chain and self._check_constraints(chain, constraints):
                logger.debug("MCTS found valid chain on attempt %d", attempt + 1)
                return self._build_tool_chain(chain)
        # Fallback
        logger.info("MCTS exhausted retries, trying random walk fallback")
        return self._random_walk_fallback(constraints, rng)

    # ------------------------------------------------------------------
    # MCTS core
    # ------------------------------------------------------------------

    def _mcts_search(
        self,
        constraints: SamplingConstraints,
        rng: np.random.Generator,
    ) -> list[str]:
        candidates = self._get_start_candidates(constraints)
        if not candidates:
            return []

        start = str(rng.choice(candidates))
        root = MCTSNode(
            state=[start],
            untried_actions=self._get_candidate_actions([start], constraints),
        )

        for _ in range(self._config.max_iterations):
            node = self._select(root)
            if not node.is_terminal and not node.is_fully_expanded:
                node = self._expand(node, constraints, rng)
            reward = self._rollout(node, constraints, rng)
            self._backpropagate(node, reward)

        return self._extract_best_chain(root)

    def _select(self, node: MCTSNode) -> MCTSNode:
        current = node
        while not current.is_terminal and current.is_fully_expanded:
            if not current.children:
                break
            current = current.best_child(self._config.exploration_weight)
        return current

    def _expand(
        self,
        node: MCTSNode,
        constraints: SamplingConstraints,
        rng: np.random.Generator,
    ) -> MCTSNode:
        if not node.untried_actions:
            return node

        idx = int(rng.integers(len(node.untried_actions)))
        action = node.untried_actions.pop(idx)
        child_state = node.state + [action]
        child = MCTSNode(
            state=child_state,
            parent=node,
            untried_actions=self._get_candidate_actions(child_state, constraints),
        )
        node.children.append(child)
        return child

    def _rollout(
        self,
        node: MCTSNode,
        constraints: SamplingConstraints,
        rng: np.random.Generator,
    ) -> float:
        sim_chain = list(node.state)
        for _ in range(self._config.rollout_depth):
            actions = self._get_candidate_actions(sim_chain, constraints)
            if not actions:
                break
            sim_chain.append(str(rng.choice(actions)))
        return self._compute_reward(sim_chain, constraints)

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        current: MCTSNode | None = node
        while current is not None:
            current.visits += 1
            current.reward += reward
            current = current.parent

    def _extract_best_chain(self, root: MCTSNode) -> list[str]:
        chain = list(root.state)
        current = root
        while current.children:
            current = max(current.children, key=lambda c: c.visits)
            if current.state:
                last = current.state[-1]
                if last not in chain:
                    chain.append(last)
        return chain

    # ------------------------------------------------------------------
    # Candidate actions
    # ------------------------------------------------------------------

    def _get_start_candidates(self, constraints: SamplingConstraints) -> list[str]:
        result: list[str] = []
        excluded = set(constraints.excluded_tools or [])
        allowed_domains = set(constraints.domains) if constraints.domains else None

        for nid in self._endpoint_nodes:
            data = self._graph.nodes[nid]
            if data.get("tool_id", "") in excluded:
                continue
            if allowed_domains is not None and data.get("domain", "") not in allowed_domains:
                continue
            result.append(nid)
        return result

    def _get_candidate_actions(
        self,
        state: list[str],
        constraints: SamplingConstraints,
    ) -> list[str]:
        if len(state) >= constraints.max_steps:
            return []

        last = state[-1]
        visited = set(state)
        excluded = set(constraints.excluded_tools or [])
        allowed_domains = set(constraints.domains) if constraints.domains else None

        neighbors: set[str] = set()
        for _u, v, _d in self._graph.out_edges(last, data=True):
            neighbors.add(v)
        for u, _v, _d in self._graph.in_edges(last, data=True):
            neighbors.add(u)

        result: list[str] = []
        for nid in neighbors:
            if nid in visited:
                continue
            data = self._graph.nodes[nid]
            if data.get("node_type") != "endpoint":
                continue
            if data.get("tool_id", "") in excluded:
                continue
            if allowed_domains is not None and data.get("domain", "") not in allowed_domains:
                continue
            result.append(nid)
        return result

    # ------------------------------------------------------------------
    # Reward & constraint checking
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        chain: list[str],
        constraints: SamplingConstraints,
    ) -> float:
        if not chain:
            return -1.0

        graph = self._graph
        tool_ids: set[str] = set()
        domains: set[str] = set()
        excluded = set(constraints.excluded_tools or [])
        required = set(constraints.required_tools or [])

        for nid in chain:
            data = graph.nodes[nid]
            tool_ids.add(data.get("tool_id", ""))
            domain = data.get("domain", "")
            if domain:
                domains.add(domain)

        score = 0.0

        # Step count bonus
        if constraints.min_steps <= len(chain) <= constraints.max_steps:
            score += 1.0

        # Multi-tool bonus
        score += 0.5 * len(tool_ids)

        # Domain bonus
        score += 0.3 * len(domains)

        # Edge coherence bonus
        for i in range(len(chain) - 1):
            if graph.has_edge(chain[i], chain[i + 1]) or graph.has_edge(
                chain[i + 1], chain[i]
            ):
                score += 0.2

        # Penalties
        for tid in excluded:
            if tid in tool_ids:
                score -= 0.5
        for tid in required:
            if tid not in tool_ids:
                score -= 0.5
        if len(chain) < constraints.min_steps or len(chain) > constraints.max_steps:
            score -= 0.5
        if len(tool_ids) < constraints.min_tools:
            score -= 0.5

        return score

    def _check_constraints(
        self,
        chain: list[str],
        constraints: SamplingConstraints,
    ) -> bool:
        if not chain:
            return False
        if not (constraints.min_steps <= len(chain) <= constraints.max_steps):
            return False

        graph = self._graph
        tool_ids: set[str] = set()
        excluded = set(constraints.excluded_tools or [])
        allowed_domains = set(constraints.domains) if constraints.domains else None

        for nid in chain:
            data = graph.nodes[nid]
            tid = data.get("tool_id", "")
            tool_ids.add(tid)
            if tid in excluded:
                return False
            if allowed_domains is not None and data.get("domain", "") not in allowed_domains:
                return False

        if len(tool_ids) < constraints.min_tools:
            return False

        required = set(constraints.required_tools or [])
        return not (required and not required.issubset(tool_ids))

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _random_walk_fallback(
        self,
        constraints: SamplingConstraints,
        rng: np.random.Generator,
    ) -> ToolChain:
        candidates = self._get_start_candidates(constraints)
        if not candidates:
            raise SamplingError("No valid start endpoints for the given constraints.")

        for _ in range(self._config.max_retries):
            start = str(rng.choice(candidates))
            chain = [start]
            for _ in range(constraints.max_steps - 1):
                actions = self._get_candidate_actions(chain, constraints)
                if not actions:
                    break
                chain.append(str(rng.choice(actions)))

            if self._check_constraints(chain, constraints):
                logger.debug("Random walk fallback found valid chain")
                return self._build_tool_chain(chain)

        raise SamplingError(
            f"Failed to sample a valid chain after {self._config.max_retries} "
            f"MCTS retries and {self._config.max_retries} fallback walks."
        )

    # ------------------------------------------------------------------
    # Chain building
    # ------------------------------------------------------------------

    def _build_tool_chain(self, chain: list[str]) -> ToolChain:
        from tooluse_gen.graph.chain_models import ParallelGroup

        steps: list[ChainStep | ParallelGroup] = [
            ChainStep.from_graph_node(self._graph, nid) for nid in chain
        ]
        return ToolChain(
            steps=steps,
            pattern=ChainPattern.SEQUENTIAL,
        )
