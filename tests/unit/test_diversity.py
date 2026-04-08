"""Unit tests for Task 26 — Diversity-aware sampling."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from tooluse_gen.graph.chain_models import (
    ChainPattern,
    ChainStep,
    ParallelGroup,
    ToolChain,
)
from tooluse_gen.graph.diversity import (
    DiversityMetrics,
    DiversitySteeringConfig,
    DiversityTracker,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chain(
    tool_ids: list[str], domains: list[str] | None = None
) -> ToolChain:
    """Build a simple sequential ToolChain from tool IDs."""
    steps: list[ChainStep | ParallelGroup] = []
    for i, tid in enumerate(tool_ids):
        dom = domains[i] if domains else f"Domain_{tid}"
        steps.append(
            ChainStep(
                endpoint_id=f"ep_{tid}_{i}",
                tool_id=tid,
                tool_name=f"Tool_{tid}",
                endpoint_name=f"Action_{i}",
                domain=dom,
            )
        )
    return ToolChain(steps=steps, pattern=ChainPattern.SEQUENTIAL)


# ===========================================================================
# DiversitySteeringConfig
# ===========================================================================


class TestDiversitySteeringConfig:
    def test_defaults(self) -> None:
        cfg = DiversitySteeringConfig()
        assert cfg.enabled is True
        assert cfg.weight_decay == pytest.approx(0.9)
        assert cfg.min_domain_coverage == pytest.approx(0.5)

    def test_custom(self) -> None:
        cfg = DiversitySteeringConfig(enabled=False, weight_decay=0.5, min_domain_coverage=0.8)
        assert cfg.enabled is False
        assert cfg.weight_decay == 0.5

    def test_weight_decay_positive(self) -> None:
        with pytest.raises(ValidationError):
            DiversitySteeringConfig(weight_decay=0.0)


# ===========================================================================
# DiversityMetrics
# ===========================================================================


class TestDiversityMetrics:
    def test_defaults(self) -> None:
        m = DiversityMetrics()
        assert m.tool_entropy == 0.0
        assert m.domain_coverage == 0.0
        assert m.unique_tool_pair_ratio == 0.0
        assert m.pattern_repetition_rate == 0.0
        assert m.total_conversations == 0

    def test_custom(self) -> None:
        m = DiversityMetrics(
            tool_entropy=1.5,
            domain_coverage=0.8,
            unique_tool_pair_ratio=0.6,
            pattern_repetition_rate=0.2,
            total_conversations=10,
        )
        assert m.tool_entropy == 1.5
        assert m.total_conversations == 10

    def test_round_trip(self) -> None:
        m = DiversityMetrics(tool_entropy=1.0, domain_coverage=0.5, total_conversations=5)
        restored = DiversityMetrics.model_validate(m.model_dump())
        assert restored.tool_entropy == m.tool_entropy


# ===========================================================================
# DiversityTracker — update
# ===========================================================================


class TestTrackerUpdate:
    def test_increments_total(self) -> None:
        t = DiversityTracker()
        t.update(_chain(["a", "b"]))
        assert t.total_conversations == 1
        t.update(_chain(["c"]))
        assert t.total_conversations == 2

    def test_increments_tool_counts(self) -> None:
        t = DiversityTracker()
        t.update(_chain(["a", "b"]))
        assert t.tool_counts["a"] == 1
        assert t.tool_counts["b"] == 1

    def test_increments_domain_counts(self) -> None:
        t = DiversityTracker()
        t.update(_chain(["a"], ["Weather"]))
        assert t.domain_counts["Weather"] == 1

    def test_increments_tool_pair_counts(self) -> None:
        t = DiversityTracker()
        t.update(_chain(["a", "b", "c"]))
        assert t.tool_pair_counts[("a", "b")] == 1
        assert t.tool_pair_counts[("a", "c")] == 1
        assert t.tool_pair_counts[("b", "c")] == 1

    def test_adds_pattern_hash(self) -> None:
        t = DiversityTracker()
        t.update(_chain(["a", "b"]))
        assert len(t.pattern_hashes) == 1

    def test_accumulates(self) -> None:
        t = DiversityTracker()
        t.update(_chain(["a", "b"]))
        t.update(_chain(["a", "c"]))
        assert t.tool_counts["a"] == 2
        assert t.tool_counts["b"] == 1
        assert t.tool_counts["c"] == 1
        assert t.total_conversations == 2


# ===========================================================================
# DiversityTracker — weights
# ===========================================================================


class TestTrackerWeights:
    def test_unused_tool_weight_is_one(self) -> None:
        t = DiversityTracker()
        assert t.get_tool_weight("unused") == 1.0

    def test_tool_weight_decreases(self) -> None:
        t = DiversityTracker(config=DiversitySteeringConfig(weight_decay=0.5))
        t.update(_chain(["a"]))
        t.update(_chain(["a"]))
        w = t.get_tool_weight("a")
        # 1 / (1 + 2 * 0.5) = 0.5
        assert w == pytest.approx(0.5)

    def test_unused_domain_weight_is_one(self) -> None:
        t = DiversityTracker()
        assert t.get_domain_weight("unused") == 1.0

    def test_domain_weight_decreases(self) -> None:
        t = DiversityTracker(config=DiversitySteeringConfig(weight_decay=0.5))
        t.update(_chain(["a"], ["Weather"]))
        w = t.get_domain_weight("Weather")
        # 1 / (1 + 1 * 0.5) = 0.6667
        assert w == pytest.approx(1.0 / 1.5)


# ===========================================================================
# DiversityTracker — duplicate detection
# ===========================================================================


class TestTrackerDuplicate:
    def test_new_pattern_not_duplicate(self) -> None:
        t = DiversityTracker()
        c = _chain(["a", "b"])
        assert t.is_duplicate_pattern(c) is False

    def test_seen_pattern_is_duplicate(self) -> None:
        t = DiversityTracker()
        c = _chain(["a", "b"])
        t.update(c)
        assert t.is_duplicate_pattern(c) is True

    def test_order_independent(self) -> None:
        t = DiversityTracker()
        t.update(_chain(["b", "a"]))
        # Same tools, different order → same hash
        assert t.is_duplicate_pattern(_chain(["a", "b"])) is True


# ===========================================================================
# DiversityTracker — metrics
# ===========================================================================


class TestTrackerMetrics:
    def test_entropy_zero_no_usage(self) -> None:
        t = DiversityTracker()
        assert t.get_diversity_metrics().tool_entropy == 0.0

    def test_entropy_positive(self) -> None:
        t = DiversityTracker()
        t.update(_chain(["a", "b"]))
        assert t.get_diversity_metrics().tool_entropy > 0

    def test_entropy_maximal_equal_usage(self) -> None:
        t = DiversityTracker(known_tools=["a", "b", "c", "d"])
        # Each tool used exactly once
        for tid in ["a", "b", "c", "d"]:
            t.update(_chain([tid]))
        m = t.get_diversity_metrics()
        expected = math.log2(4)  # max entropy for 4 equiprobable items
        assert m.tool_entropy == pytest.approx(expected, abs=0.01)

    def test_domain_coverage(self) -> None:
        t = DiversityTracker(known_domains=["A", "B", "C"])
        t.update(_chain(["x"], ["A"]))
        m = t.get_diversity_metrics()
        assert m.domain_coverage == pytest.approx(1.0 / 3.0)
        t.update(_chain(["y"], ["B"]))
        m2 = t.get_diversity_metrics()
        assert m2.domain_coverage == pytest.approx(2.0 / 3.0)

    def test_tool_pair_ratio(self) -> None:
        t = DiversityTracker(known_tools=["a", "b", "c"])
        t.update(_chain(["a", "b"]))
        m = t.get_diversity_metrics()
        # 1 pair out of C(3,2)=3
        assert m.unique_tool_pair_ratio == pytest.approx(1.0 / 3.0)

    def test_pattern_repetition_rate(self) -> None:
        t = DiversityTracker()
        t.update(_chain(["a", "b"]))
        t.update(_chain(["a", "b"]))  # duplicate
        t.update(_chain(["c", "d"]))  # new
        m = t.get_diversity_metrics()
        # 2 unique patterns out of 3 conversations → 1 - 2/3 = 0.333
        assert m.pattern_repetition_rate == pytest.approx(1.0 / 3.0, abs=0.01)


# ===========================================================================
# DiversityTracker — underrepresented
# ===========================================================================


class TestTrackerUnderrepresented:
    def test_unused_domains(self) -> None:
        t = DiversityTracker(known_domains=["A", "B", "C"])
        t.update(_chain(["x"], ["A"]))
        under = t.get_underrepresented_domains(threshold=0)
        assert "B" in under
        assert "C" in under
        assert "A" not in under

    def test_unused_tools(self) -> None:
        t = DiversityTracker(known_tools=["a", "b", "c"])
        t.update(_chain(["a"]))
        under = t.get_underrepresented_tools(threshold=0)
        assert "b" in under
        assert "c" in under
        assert "a" not in under

    def test_threshold_filtering(self) -> None:
        t = DiversityTracker(known_tools=["a", "b"])
        t.update(_chain(["a"]))
        t.update(_chain(["a"]))
        # a has count 2, b has count 0
        under = t.get_underrepresented_tools(threshold=1)
        assert "b" in under
        assert "a" not in under  # count 2 > threshold 1

    def test_results_sorted(self) -> None:
        t = DiversityTracker(known_tools=["c", "b", "a"])
        under = t.get_underrepresented_tools(threshold=0)
        assert under == sorted(under)


# ===========================================================================
# DiversityTracker — reset
# ===========================================================================


class TestTrackerReset:
    def test_clears_counters(self) -> None:
        t = DiversityTracker(known_tools=["a"], known_domains=["D"])
        t.update(_chain(["a"], ["D"]))
        assert t.total_conversations == 1
        t.reset()
        assert t.total_conversations == 0
        assert t.tool_counts["a"] == 0
        assert t.domain_counts["D"] == 0
        assert len(t.tool_pair_counts) == 0
        assert len(t.pattern_hashes) == 0

    def test_preserves_known(self) -> None:
        t = DiversityTracker(known_tools=["a", "b"], known_domains=["X", "Y"])
        t.update(_chain(["a"], ["X"]))
        t.reset()
        assert "a" in t._known_tools
        assert "b" in t._known_tools
        assert "X" in t._known_domains
        assert "Y" in t._known_domains
