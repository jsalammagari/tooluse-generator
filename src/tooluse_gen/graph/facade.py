"""High-level facade for tool-chain sampling.

:class:`ToolChainSampler` integrates the MCTS sampler, pattern
detection/enforcement, and diversity tracking into a single API.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from tooluse_gen.graph.chain_models import (
    ChainPattern,
    SamplingConstraints,
    ToolChain,
)
from tooluse_gen.graph.diversity import (
    DiversityMetrics,
    DiversitySteeringConfig,
    DiversityTracker,
    build_diversity_summary,
    build_steering_prompt,
    should_steer,
)
from tooluse_gen.graph.patterns import (
    PatternDetector,
    PatternEnforcer,
)
from tooluse_gen.graph.sampler import (
    MCTSSampler,
    SamplerConfig,
    SamplingError,
)
from tooluse_gen.utils.logging import get_logger

logger = get_logger("graph.facade")

_DUPLICATE_RETRIES = 3


class ToolChainSampler:
    """High-level facade integrating MCTS sampling, pattern enforcement, and diversity tracking."""

    def __init__(
        self,
        graph: nx.DiGraph,
        sampler_config: SamplerConfig | None = None,
        diversity_config: DiversitySteeringConfig | None = None,
    ) -> None:
        self._graph = graph
        self._sampler = MCTSSampler(graph, sampler_config)
        self._detector = PatternDetector(graph)
        self._enforcer = PatternEnforcer(graph)

        # Collect known domains and tools from endpoint nodes
        domains: set[str] = set()
        tools: set[str] = set()
        for _n, data in graph.nodes(data=True):
            if data.get("node_type") == "endpoint":
                dom = data.get("domain", "")
                if dom:
                    domains.add(dom)
                tid = data.get("tool_id", "")
                if tid:
                    tools.add(tid)

        self._known_domains = sorted(domains)
        self._known_tools = sorted(tools)

        self._tracker = DiversityTracker(
            config=diversity_config or DiversitySteeringConfig(),
            known_domains=self._known_domains,
            known_tools=self._known_tools,
        )
        self._chain_counter: int = 0

    # ------------------------------------------------------------------
    # Single sample
    # ------------------------------------------------------------------

    def sample_chain(
        self,
        constraints: SamplingConstraints,
        rng: np.random.Generator,
    ) -> ToolChain:
        """Sample a single tool chain with pattern detection and diversity tracking."""
        chain = self._sample_with_diversity(constraints, rng)
        chain = self._apply_pattern(chain, constraints, rng)

        # Assign ID
        chain = chain.model_copy(
            update={
                "chain_id": f"chain_{self._chain_counter:04d}",
                "metadata": {
                    **chain.metadata,
                    "steering_prompt": build_steering_prompt(
                        self._tracker, self._known_domains
                    ),
                },
            }
        )
        self._chain_counter += 1

        # Track diversity
        self._tracker.update(chain)

        return chain

    # ------------------------------------------------------------------
    # Batch sample
    # ------------------------------------------------------------------

    def sample_batch(
        self,
        constraints: SamplingConstraints,
        count: int,
        rng: np.random.Generator,
    ) -> list[ToolChain]:
        """Sample *count* chains with inter-chain diversity tracking."""
        results: list[ToolChain] = []
        for i in range(count):
            try:
                chain = self.sample_chain(constraints, rng)
                results.append(chain)
            except SamplingError:
                logger.warning("Sampling failed for chain %d/%d, skipping", i + 1, count)
                continue
            if (i + 1) % 10 == 0:
                logger.info("Sampled %d/%d chains", i + 1, count)
        return results

    # ------------------------------------------------------------------
    # Diversity accessors
    # ------------------------------------------------------------------

    def get_diversity_report(self) -> DiversityMetrics:
        """Return current diversity metrics."""
        return self._tracker.get_diversity_metrics()

    def get_diversity_summary(self) -> str:
        """Human-readable diversity summary."""
        return build_diversity_summary(self._tracker)

    def get_steering_prompt(self) -> str:
        """Current steering prompt based on diversity state."""
        return build_steering_prompt(self._tracker, self._known_domains)

    def reset_diversity(self) -> None:
        """Reset diversity tracker and chain counter."""
        self._tracker.reset()
        self._chain_counter = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample_with_diversity(
        self,
        constraints: SamplingConstraints,
        rng: np.random.Generator,
    ) -> ToolChain:
        """Sample a chain, re-trying if it duplicates an existing pattern."""
        chain = self._sampler.sample(constraints, rng)

        if should_steer(self._tracker.config):
            for _ in range(_DUPLICATE_RETRIES):
                if not self._tracker.is_duplicate_pattern(chain):
                    break
                logger.debug("Duplicate pattern detected, re-sampling")
                chain = self._sampler.sample(constraints, rng)

        return chain

    def _apply_pattern(
        self,
        chain: ToolChain,
        constraints: SamplingConstraints,
        rng: np.random.Generator,
    ) -> ToolChain:
        """Apply pattern enforcement or auto-detection."""
        patterns = constraints.required_patterns
        if patterns and len(patterns) == 1:
            target = ChainPattern(patterns[0])
            return self._enforcer.enforce_pattern(chain, target, rng)

        # Auto-detect
        result = self._detector.detect_parallel_opportunities(chain)
        if result.pattern != ChainPattern.SEQUENTIAL.value:
            return result

        result = self._detector.detect_branch_and_merge(chain)
        if result.pattern != ChainPattern.SEQUENTIAL.value:
            return result

        result = self._detector.detect_iterative(chain)
        if result.pattern != ChainPattern.SEQUENTIAL.value:
            return result

        return chain
