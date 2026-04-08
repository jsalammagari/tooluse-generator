"""Chain pattern detection, enforcement, and description.

:class:`PatternDetector` analyses a flat sequential :class:`ToolChain`
and restructures it into richer execution patterns (parallel,
branch-and-merge, iterative).

:class:`PatternEnforcer` actively modifies a chain to match a
requested :class:`ChainPattern`.

:func:`chain_to_description` renders a chain as a human-readable string
suitable for inclusion in LLM prompts.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from tooluse_gen.graph.chain_models import (
    ChainPattern,
    ChainStep,
    ParallelGroup,
    ToolChain,
    _iter_chain_steps,
)
from tooluse_gen.utils.logging import get_logger

logger = get_logger("graph.patterns")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _endpoint_neighbors(graph: nx.DiGraph, node_id: str) -> list[str]:
    """Return endpoint-type neighbours reachable from *node_id* (both dirs)."""
    if node_id not in graph:
        return []
    result: set[str] = set()
    for _u, v, _d in graph.out_edges(node_id, data=True):
        if graph.nodes[v].get("node_type") == "endpoint":
            result.add(v)
    for u, _v, _d in graph.in_edges(node_id, data=True):
        if graph.nodes[u].get("node_type") == "endpoint":
            result.add(u)
    return sorted(result)


def _has_dependency(step_a: ChainStep, step_b: ChainStep) -> bool:
    """True when *step_b* depends on *step_a*'s output."""
    if not step_a.expected_output_types or not step_b.expected_params:
        return False
    return bool(set(step_b.expected_params) & set(step_a.expected_output_types))


def _are_parallelizable(
    step_a: ChainStep, step_b: ChainStep, graph: nx.DiGraph
) -> bool:
    """Two steps can run in parallel when same-domain/edge-connected and independent."""
    if _has_dependency(step_a, step_b):
        return False
    if step_a.domain and step_a.domain == step_b.domain:
        return True
    # Check graph edge
    nid_a = _node_id_for_step(step_a, graph)
    nid_b = _node_id_for_step(step_b, graph)
    return bool(
        nid_a and nid_b and (graph.has_edge(nid_a, nid_b) or graph.has_edge(nid_b, nid_a))
    )


def _node_id_for_step(step: ChainStep, graph: nx.DiGraph) -> str | None:
    """Best-effort lookup of the graph node ID for a ChainStep."""
    candidate = f"ep:{step.tool_id}:{step.endpoint_id}"
    if candidate in graph:
        return candidate
    # Fallback: scan nodes
    for nid, data in graph.nodes(data=True):
        if data.get("endpoint_id") == step.endpoint_id and data.get("tool_id") == step.tool_id:
            return str(nid)
    return None


def _new_chain(
    steps: list[ChainStep | ParallelGroup],
    pattern: ChainPattern,
    chain: ToolChain,
) -> ToolChain:
    """Build a new ToolChain preserving id and metadata."""
    return ToolChain(
        chain_id=chain.chain_id,
        steps=steps,
        pattern=pattern,
        metadata=dict(chain.metadata),
    )


# ---------------------------------------------------------------------------
# PatternDetector
# ---------------------------------------------------------------------------


class PatternDetector:
    """Detects pattern opportunities in sequential tool chains."""

    def __init__(self, graph: nx.DiGraph) -> None:
        self._graph = graph

    def detect_parallel_opportunities(self, chain: ToolChain) -> ToolChain:
        """Group consecutive independent steps into :class:`ParallelGroup`."""
        if len(chain.steps) < 2:
            return chain

        new_steps: list[ChainStep | ParallelGroup] = []
        i = 0
        found = False

        while i < len(chain.steps):
            item = chain.steps[i]
            if not isinstance(item, ChainStep):
                new_steps.append(item)
                i += 1
                continue

            # Try to accumulate a parallel group starting at i
            group: list[ChainStep] = [item]
            j = i + 1
            while j < len(chain.steps):
                next_item = chain.steps[j]
                if not isinstance(next_item, ChainStep):
                    break
                # Check independence against ALL steps already in the group
                if all(
                    _are_parallelizable(g, next_item, self._graph) for g in group
                ):
                    group.append(next_item)
                    j += 1
                else:
                    break

            if len(group) >= 2:
                new_steps.append(ParallelGroup(steps=group))
                found = True
                i = j
            else:
                new_steps.append(item)
                i += 1

        if found:
            return _new_chain(new_steps, ChainPattern.PARALLEL, chain)
        return chain

    def detect_branch_and_merge(self, chain: ToolChain) -> ToolChain:
        """Insert a branch when a step has alternative graph neighbours."""
        flat = list(chain.steps)
        for idx in range(len(flat) - 1):
            step = flat[idx]
            next_step = flat[idx + 1]
            if not isinstance(step, ChainStep) or not isinstance(next_step, ChainStep):
                continue

            nid = _node_id_for_step(step, self._graph)
            if nid is None:
                continue

            neighbors = _endpoint_neighbors(self._graph, nid)
            next_nid = _node_id_for_step(next_step, self._graph)
            # Find an alternative that is NOT the next step and NOT the current step
            alternatives = [
                n for n in neighbors
                if n != nid and n != next_nid
            ]
            if not alternatives:
                continue

            alt_nid = alternatives[0]
            alt_step = ChainStep.from_graph_node(self._graph, alt_nid)
            branch = ParallelGroup(steps=[next_step, alt_step])
            new_steps: list[ChainStep | ParallelGroup] = (
                list(flat[:idx + 1]) + [branch] + list(flat[idx + 2:])
            )
            return _new_chain(new_steps, ChainPattern.BRANCH_AND_MERGE, chain)

        return chain

    def detect_iterative(self, chain: ToolChain) -> ToolChain:
        """Detect repetition in the chain and mark as iterative."""
        flat = _iter_chain_steps(list(chain.steps))
        seen_pairs: set[tuple[str, str]] = set()
        seen_tool_method: set[tuple[str, str]] = set()

        for step in flat:
            pair = (step.tool_id, step.endpoint_name)
            tm = (step.tool_id, step.method)
            if pair in seen_pairs or tm in seen_tool_method:
                return _new_chain(list(chain.steps), ChainPattern.ITERATIVE, chain)
            seen_pairs.add(pair)
            seen_tool_method.add(tm)

        return chain


# ---------------------------------------------------------------------------
# PatternEnforcer
# ---------------------------------------------------------------------------


class PatternEnforcer:
    """Modifies chains to match a target pattern."""

    def __init__(self, graph: nx.DiGraph) -> None:
        self._graph = graph

    def enforce_pattern(
        self,
        chain: ToolChain,
        target: ChainPattern,
        rng: np.random.Generator,
    ) -> ToolChain:
        """Return a new chain reshaped to *target* pattern."""
        target_val = target if isinstance(target, str) else target.value
        if target_val == ChainPattern.SEQUENTIAL.value:
            return self._enforce_sequential(chain)
        if target_val == ChainPattern.PARALLEL.value:
            return self._enforce_parallel(chain)
        if target_val == ChainPattern.BRANCH_AND_MERGE.value:
            return self._enforce_branch_and_merge(chain, rng)
        if target_val == ChainPattern.ITERATIVE.value:
            return self._enforce_iterative(chain, rng)
        return chain  # pragma: no cover

    def _enforce_sequential(self, chain: ToolChain) -> ToolChain:
        flat: list[ChainStep | ParallelGroup] = []
        for item in chain.steps:
            if isinstance(item, ParallelGroup):
                flat.extend(item.steps)
            else:
                flat.append(item)
        return _new_chain(flat, ChainPattern.SEQUENTIAL, chain)

    def _enforce_parallel(self, chain: ToolChain) -> ToolChain:
        detector = PatternDetector(self._graph)
        result = detector.detect_parallel_opportunities(chain)
        if result.pattern != ChainPattern.PARALLEL.value:
            return _new_chain(list(chain.steps), ChainPattern.PARALLEL, chain)
        return result

    def _enforce_branch_and_merge(
        self, chain: ToolChain, rng: np.random.Generator
    ) -> ToolChain:
        detector = PatternDetector(self._graph)
        result = detector.detect_branch_and_merge(chain)
        if result.pattern == ChainPattern.BRANCH_AND_MERGE.value:
            return result

        # Fallback: pick a random step and try to find any graph neighbor
        flat_steps = [s for s in chain.steps if isinstance(s, ChainStep)]
        if not flat_steps:
            return _new_chain(list(chain.steps), ChainPattern.BRANCH_AND_MERGE, chain)

        rng.shuffle(flat_steps)
        for step in flat_steps:
            nid = _node_id_for_step(step, self._graph)
            if nid is None:
                continue
            neighbors = _endpoint_neighbors(self._graph, nid)
            others = [n for n in neighbors if n != nid]
            if others:
                alt_nid = str(rng.choice(others))
                alt_step = ChainStep.from_graph_node(self._graph, alt_nid)
                branch = ParallelGroup(steps=[step, alt_step])
                new_steps: list[ChainStep | ParallelGroup] = [
                    (branch if s is step else s) for s in chain.steps
                ]
                return _new_chain(new_steps, ChainPattern.BRANCH_AND_MERGE, chain)

        return _new_chain(list(chain.steps), ChainPattern.BRANCH_AND_MERGE, chain)

    def _enforce_iterative(
        self, chain: ToolChain, rng: np.random.Generator
    ) -> ToolChain:
        flat_steps = [s for s in chain.steps if isinstance(s, ChainStep)]
        if not flat_steps:
            return _new_chain(list(chain.steps), ChainPattern.ITERATIVE, chain)

        idx = int(rng.integers(len(flat_steps)))
        chosen = flat_steps[idx]
        duplicate = chosen.model_copy()

        # Insert duplicate right after the original
        new_steps: list[ChainStep | ParallelGroup] = []
        inserted = False
        for item in chain.steps:
            new_steps.append(item)
            if item is chosen and not inserted:
                new_steps.append(duplicate)
                inserted = True

        return _new_chain(new_steps, ChainPattern.ITERATIVE, chain)


# ---------------------------------------------------------------------------
# Description formatter
# ---------------------------------------------------------------------------


def chain_to_description(chain: ToolChain) -> str:
    """Human-readable chain description for prompts."""
    if not chain.steps:
        return "(empty chain)"

    parts: list[str] = []
    for item in chain.steps:
        if isinstance(item, ParallelGroup):
            inner = ", ".join(
                f"{s.tool_name}: {s.endpoint_name}" for s in item.steps
            )
            parts.append(f"[{inner}]")
        else:
            parts.append(f"{item.tool_name}: {item.endpoint_name}")
    return " -> ".join(parts)
