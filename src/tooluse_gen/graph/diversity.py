"""Cross-conversation diversity tracking and weight adjustment.

:class:`DiversityTracker` monitors tool and domain usage across
generated conversations, provides inverse-frequency weights that steer
the sampler toward underrepresented areas, and computes diversity
metrics (Shannon entropy, domain coverage, tool-pair ratio).
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter

from pydantic import BaseModel, ConfigDict, Field

from tooluse_gen.graph.chain_models import ToolChain
from tooluse_gen.utils.logging import get_logger

logger = get_logger("graph.diversity")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class DiversitySteeringConfig(BaseModel):
    """Configuration for diversity steering."""

    model_config = ConfigDict(use_enum_values=True)

    enabled: bool = Field(default=True, description="Enable diversity steering.")
    weight_decay: float = Field(
        default=0.9,
        gt=0.0,
        le=1.0,
        description="Decay factor for inverse-frequency weights.",
    )
    min_domain_coverage: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Target minimum fraction of domains to cover.",
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class DiversityMetrics(BaseModel):
    """Snapshot of diversity statistics."""

    model_config = ConfigDict(use_enum_values=True)

    tool_entropy: float = Field(
        default=0.0, ge=0.0, description="Shannon entropy of tool usage distribution."
    )
    domain_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of known domains that have been used.",
    )
    unique_tool_pair_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Unique tool pairs / total possible pairs.",
    )
    pattern_repetition_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="1 - (unique patterns / total generated).",
    )
    total_conversations: int = Field(default=0, ge=0)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


def _pattern_hash(tool_ids: list[str]) -> str:
    """Deterministic hash of a sorted tool-id list."""
    key = ",".join(sorted(tool_ids))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


class DiversityTracker:
    """Tracks tool/domain usage across generated conversations for diversity steering."""

    def __init__(
        self,
        config: DiversitySteeringConfig | None = None,
        known_domains: list[str] | None = None,
        known_tools: list[str] | None = None,
    ) -> None:
        self.config = config or DiversitySteeringConfig()
        self.tool_counts: Counter[str] = Counter()
        self.domain_counts: Counter[str] = Counter()
        self.tool_pair_counts: Counter[tuple[str, str]] = Counter()
        self.pattern_hashes: set[str] = set()
        self.total_conversations: int = 0
        self._known_domains: set[str] = set(known_domains or [])
        self._known_tools: set[str] = set(known_tools or [])

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, chain: ToolChain) -> None:
        """Record a newly generated chain."""
        self.total_conversations += 1

        for tid in chain.tool_ids:
            self.tool_counts[tid] += 1
            self._known_tools.add(tid)

        for domain in chain.domains_involved:
            self.domain_counts[domain] += 1
            self._known_domains.add(domain)

        ids = sorted(chain.tool_ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                self.tool_pair_counts[(ids[i], ids[j])] += 1

        self.pattern_hashes.add(_pattern_hash(chain.tool_ids))

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------

    def get_tool_weight(self, tool_id: str) -> float:
        """Inverse-frequency weight — higher for less-used tools."""
        count = self.tool_counts[tool_id]
        return 1.0 / (1.0 + count * self.config.weight_decay)

    def get_domain_weight(self, domain: str) -> float:
        """Inverse-frequency weight — higher for less-used domains."""
        count = self.domain_counts[domain]
        return 1.0 / (1.0 + count * self.config.weight_decay)

    # ------------------------------------------------------------------
    # Duplicate detection
    # ------------------------------------------------------------------

    def is_duplicate_pattern(self, chain: ToolChain) -> bool:
        """True if the sorted tool combination has been seen before."""
        return _pattern_hash(chain.tool_ids) in self.pattern_hashes

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_diversity_metrics(self) -> DiversityMetrics:
        """Compute a snapshot of diversity statistics."""
        return DiversityMetrics(
            tool_entropy=self._tool_entropy(),
            domain_coverage=self._domain_coverage(),
            unique_tool_pair_ratio=self._tool_pair_ratio(),
            pattern_repetition_rate=self._pattern_repetition_rate(),
            total_conversations=self.total_conversations,
        )

    def _tool_entropy(self) -> float:
        total = sum(self.tool_counts.values())
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in self.tool_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        return entropy

    def _domain_coverage(self) -> float:
        if not self._known_domains:
            return 0.0
        used = sum(1 for d in self._known_domains if self.domain_counts[d] > 0)
        return used / len(self._known_domains)

    def _tool_pair_ratio(self) -> float:
        n = len(self._known_tools)
        if n < 2:
            return 0.0
        max_pairs = n * (n - 1) / 2
        return len(self.tool_pair_counts) / max_pairs

    def _pattern_repetition_rate(self) -> float:
        if self.total_conversations == 0:
            return 0.0
        return 1.0 - (len(self.pattern_hashes) / self.total_conversations)

    # ------------------------------------------------------------------
    # Underrepresented queries
    # ------------------------------------------------------------------

    def get_underrepresented_domains(self, threshold: float = 0.0) -> list[str]:
        """Sorted known domains whose usage count is <= *threshold*."""
        return sorted(d for d in self._known_domains if self.domain_counts[d] <= threshold)

    def get_underrepresented_tools(self, threshold: float = 0.0) -> list[str]:
        """Sorted known tools whose usage count is <= *threshold*."""
        return sorted(t for t in self._known_tools if self.tool_counts[t] <= threshold)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all counters but keep known domains and tools."""
        self.tool_counts.clear()
        self.domain_counts.clear()
        self.tool_pair_counts.clear()
        self.pattern_hashes.clear()
        self.total_conversations = 0
