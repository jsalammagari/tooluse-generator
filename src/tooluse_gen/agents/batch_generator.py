"""Batch conversation generator.

:class:`BatchGenerator` produces multiple conversations with optional
diversity steering by coordinating a :class:`ConversationOrchestrator`
and a :class:`ToolChainSampler`.

:class:`BatchStats` summarises the results of a batch run.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from tooluse_gen.agents.conversation_models import Conversation
from tooluse_gen.agents.orchestrator import ConversationOrchestrator
from tooluse_gen.graph.chain_models import SamplingConstraints
from tooluse_gen.graph.diversity import DiversityMetrics, DiversitySteeringConfig
from tooluse_gen.graph.facade import ToolChainSampler
from tooluse_gen.graph.sampler import SamplingError
from tooluse_gen.utils.logging import get_logger

logger = get_logger("agents.batch_generator")


# ---------------------------------------------------------------------------
# BatchStats
# ---------------------------------------------------------------------------


class BatchStats(BaseModel):
    """Summary of a batch generation run."""

    model_config = ConfigDict(use_enum_values=True)

    total_generated: int = Field(default=0, description="Conversations generated.")
    total_failed: int = Field(default=0, description="Failed attempts.")
    diversity_metrics: DiversityMetrics | None = Field(
        default=None, description="Final diversity metrics."
    )
    average_turns: float = Field(default=0.0, description="Avg turns per conversation.")
    average_tool_calls: float = Field(default=0.0, description="Avg tool calls.")
    average_generation_time_ms: float = Field(default=0.0, description="Avg gen time (ms).")
    tools_coverage: int = Field(default=0, description="Distinct tools used.")
    domain_coverage: int = Field(default=0, description="Distinct domains used.")
    steering_enabled: bool = Field(default=False, description="Steering was on.")


# ---------------------------------------------------------------------------
# BatchGenerator
# ---------------------------------------------------------------------------


class BatchGenerator:
    """Generates batches of conversations with diversity steering."""

    def __init__(
        self,
        orchestrator: ConversationOrchestrator,
        sampler: ToolChainSampler,
        diversity_config: DiversitySteeringConfig | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._sampler = sampler
        self._diversity_config = diversity_config
        self._stats: BatchStats | None = None
        self._logger = logger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        count: int,
        constraints: SamplingConstraints,
        seed: int = 42,
        steering_enabled: bool = True,
    ) -> list[Conversation]:
        """Generate *count* conversations, returning those that succeed."""
        if count <= 0:
            self._stats = self._compute_stats([], 0, steering_enabled)
            return []

        rng = np.random.default_rng(seed)

        if steering_enabled and self._diversity_config is not None:
            self._sampler.reset_diversity()

        conversations: list[Conversation] = []
        failed = 0

        for i in range(count):
            conv_seed = seed + i

            # Sample chain.
            try:
                chain = self._sampler.sample_chain(constraints, rng)
            except (SamplingError, Exception) as exc:  # noqa: BLE001
                self._logger.warning(
                    "Sampling failed for conv %d/%d: %s", i + 1, count, exc
                )
                failed += 1
                continue

            # Generate conversation.
            try:
                conv = self._orchestrator.generate_conversation(
                    chain, seed=conv_seed
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "Generation failed for conv %d/%d: %s", i + 1, count, exc
                )
                failed += 1
                continue

            conversations.append(conv)

            if (len(conversations)) % 10 == 0 or i == count - 1:
                self._log_progress(len(conversations), count, failed)

        self._stats = self._compute_stats(conversations, failed, steering_enabled)
        return conversations

    def get_batch_stats(self) -> BatchStats:
        """Return stats from the last batch. Raises if none generated yet."""
        if self._stats is None:
            raise ValueError("No batch has been generated yet.")
        return self._stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_stats(
        self,
        conversations: list[Conversation],
        failed: int,
        steering_enabled: bool,
    ) -> BatchStats:
        n = len(conversations)
        if n == 0:
            return BatchStats(
                total_generated=0,
                total_failed=failed,
                steering_enabled=steering_enabled,
            )

        all_tools: set[str] = set()
        all_domains: set[str] = set()
        total_turns = 0
        total_calls = 0
        total_time = 0

        for conv in conversations:
            total_turns += conv.metadata.num_turns
            total_calls += conv.metadata.num_tool_calls
            total_time += conv.metadata.generation_time_ms
            all_tools.update(conv.metadata.tools_used)
            all_domains.update(conv.metadata.domains)

        diversity = self._sampler.get_diversity_report()

        return BatchStats(
            total_generated=n,
            total_failed=failed,
            diversity_metrics=diversity,
            average_turns=total_turns / n,
            average_tool_calls=total_calls / n,
            average_generation_time_ms=total_time / n,
            tools_coverage=len(all_tools),
            domain_coverage=len(all_domains),
            steering_enabled=steering_enabled,
        )

    def _log_progress(self, current: int, total: int, failed: int) -> None:
        self._logger.info(
            "Generated %d/%d conversations (%d failed)", current, total, failed
        )
