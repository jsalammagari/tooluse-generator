"""Completeness scoring for Tool Registry entities.

Assigns a quality score in ``[0, 1]`` to every :class:`Tool`,
:class:`Endpoint`, and :class:`Parameter` based on how much useful
metadata each carries.  Scores drive quality-tier filtering so the
conversation generator can focus on well-documented APIs.

Scoring methodology
-------------------
Each level (parameter → endpoint → tool) is scored as a weighted sum of
binary or normalised signals.  Default weights are tuned so that
*endpoint quality* dominates the tool score — a tool with one
fully-documented endpoint scores higher than a tool with ten bare stubs.

Quality tiers
-------------
========== ========
Tier       Score
========== ========
EXCELLENT  >= 0.8
GOOD       >= 0.6
FAIR       >= 0.4
POOR       >= 0.2
MINIMAL    < 0.2
========== ========
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

from tooluse_gen.registry.models import Endpoint, Parameter, ParameterType, Tool

# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------


@dataclass
class CompletenessWeights:
    """Configurable weights for completeness scoring.

    Each group (tool-level, endpoint-level, parameter-level) sums to
    approximately 1.0 so that the final score sits in ``[0, 1]``.
    """

    # Tool-level weights
    tool_name: float = 0.10
    tool_description: float = 0.15
    tool_domain: float = 0.10
    tool_has_endpoints: float = 0.15
    endpoint_quality: float = 0.50

    # Endpoint-level weights
    endpoint_name: float = 0.10
    endpoint_description: float = 0.15
    endpoint_path: float = 0.05
    endpoint_method: float = 0.05
    endpoint_parameters: float = 0.35
    endpoint_response_schema: float = 0.30

    # Parameter-level weights
    param_name: float = 0.20
    param_description: float = 0.25
    param_type: float = 0.30
    param_required_flag: float = 0.10
    param_default_value: float = 0.15


DEFAULT_WEIGHTS = CompletenessWeights()


# ---------------------------------------------------------------------------
# Quality tiers
# ---------------------------------------------------------------------------


class QualityTier(str, Enum):
    """Quality tiers based on completeness score."""

    EXCELLENT = "excellent"  # >= 0.8
    GOOD = "good"  # >= 0.6
    FAIR = "fair"  # >= 0.4
    POOR = "poor"  # >= 0.2
    MINIMAL = "minimal"  # < 0.2


_TIER_THRESHOLDS: list[tuple[float, QualityTier]] = [
    (0.8, QualityTier.EXCELLENT),
    (0.6, QualityTier.GOOD),
    (0.4, QualityTier.FAIR),
    (0.2, QualityTier.POOR),
]


def get_quality_tier(score: float) -> QualityTier:
    """Return the quality tier for a completeness *score*."""
    for threshold, tier in _TIER_THRESHOLDS:
        if score >= threshold:
            return tier
    return QualityTier.MINIMAL


_TIER_ORDER: dict[QualityTier, int] = {
    QualityTier.EXCELLENT: 4,
    QualityTier.GOOD: 3,
    QualityTier.FAIR: 2,
    QualityTier.POOR: 1,
    QualityTier.MINIMAL: 0,
}


def filter_by_quality(
    tools: list[Tool],
    min_tier: QualityTier = QualityTier.FAIR,
) -> list[Tool]:
    """Return tools whose quality tier meets or exceeds *min_tier*."""
    min_rank = _TIER_ORDER[min_tier]
    return [t for t in tools if _TIER_ORDER[get_quality_tier(t.completeness_score)] >= min_rank]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

# Trivial descriptions that add no value
_TRIVIAL_PREFIXES = re.compile(
    r"^(the|a|an|this)\s+",
    re.IGNORECASE,
)


def is_meaningful_description(description: str, name: str) -> bool:
    """Check whether *description* adds information beyond *name*.

    Returns ``False`` when the description is:
    - empty or whitespace-only,
    - identical to *name* (case-insensitive),
    - just ``"The <name>"``, ``"A <name>"``, or similar trivial wrappers,
    - shorter than 10 characters.
    """
    desc = description.strip()
    if not desc or len(desc) < 10:
        return False
    if desc.lower() == name.lower():
        return False
    stripped = _TRIVIAL_PREFIXES.sub("", desc).strip()
    return stripped.lower() != name.lower()


def is_explicit_type(param: Parameter) -> bool:
    """True when *param* has a declared (not inferred) type."""
    return param.has_type and not param.inferred_type


def count_documented_params(endpoint: Endpoint) -> tuple[int, int]:
    """Return ``(documented_count, total_count)`` for *endpoint*'s parameters.

    A parameter is *documented* when it has a meaningful description.
    """
    total = len(endpoint.parameters)
    documented = sum(
        1 for p in endpoint.parameters if is_meaningful_description(p.description, p.name)
    )
    return documented, total


# ---------------------------------------------------------------------------
# CompletenessCalculator
# ---------------------------------------------------------------------------


class CompletenessCalculator:
    """Calculate completeness scores for tools, endpoints, and parameters.

    Scores range from 0.0 (empty/stub) to 1.0 (fully documented).
    """

    def __init__(self, weights: CompletenessWeights = DEFAULT_WEIGHTS) -> None:
        self.weights = weights

    # -- Parameter ----------------------------------------------------------

    def calculate_parameter_score(self, param: Parameter) -> float:
        """Score a single parameter.

        Signals:
        - Has a non-empty name (baseline — always true).
        - Has a meaningful description.
        - Has an explicit (non-UNKNOWN, non-inferred) type.
        - Has ``required`` flag set to ``True``.
        - Has a default value (for optional parameters).
        """
        w = self.weights
        score = 0.0

        # Name (always present since it's required)
        score += w.param_name

        # Description
        if is_meaningful_description(param.description, param.name):
            score += w.param_description

        # Type
        if is_explicit_type(param) and param.param_type != ParameterType.UNKNOWN:
            score += w.param_type

        # Required flag
        if param.required:
            score += w.param_required_flag

        # Default value (meaningful for optional params)
        if param.default is not None:
            score += w.param_default_value

        return round(min(score, 1.0), 4)

    # -- Endpoint -----------------------------------------------------------

    def calculate_endpoint_score(self, endpoint: Endpoint) -> float:
        """Score a single endpoint.

        Signals:
        - Has a meaningful name.
        - Has a meaningful description.
        - Has a non-empty path.
        - Has an explicit HTTP method (always true via default).
        - Has parameters with good average scores.
        - Has a response schema.
        """
        w = self.weights
        score = 0.0

        # Name
        if endpoint.name and endpoint.name.strip():
            score += w.endpoint_name

        # Description
        if is_meaningful_description(endpoint.description, endpoint.name):
            score += w.endpoint_description

        # Path
        if endpoint.path and endpoint.path.strip() not in ("", "/"):
            score += w.endpoint_path

        # Method (always has a default)
        score += w.endpoint_method

        # Parameters
        if endpoint.parameters:
            avg_param = sum(
                self.calculate_parameter_score(p) for p in endpoint.parameters
            ) / len(endpoint.parameters)
            score += w.endpoint_parameters * avg_param
        # No parameters is neutral — some endpoints legitimately have none

        # Response schema
        if endpoint.response_schema is not None:
            score += w.endpoint_response_schema

        return round(min(score, 1.0), 4)

    # -- Tool ---------------------------------------------------------------

    def calculate_tool_score(self, tool: Tool) -> float:
        """Score a tool (aggregates endpoint scores).

        Signals:
        - Has a meaningful name.
        - Has a meaningful description.
        - Has a domain/category.
        - Has at least one endpoint.
        - Average quality of endpoints.
        """
        w = self.weights
        score = 0.0

        # Name
        if tool.name and tool.name.strip():
            score += w.tool_name

        # Description
        if is_meaningful_description(tool.description, tool.name):
            score += w.tool_description

        # Domain
        if tool.domain and tool.domain.strip():
            score += w.tool_domain

        # Has endpoints
        if tool.endpoints:
            score += w.tool_has_endpoints
            avg_ep = sum(
                self.calculate_endpoint_score(ep) for ep in tool.endpoints
            ) / len(tool.endpoints)
            score += w.endpoint_quality * avg_ep

        return round(min(score, 1.0), 4)

    # -- Batch --------------------------------------------------------------

    def calculate_all(self, tool: Tool) -> Tool:
        """Calculate and assign scores to *tool* and all its endpoints.

        Mutates *tool* in-place and returns it for chaining.
        """
        for ep in tool.endpoints:
            ep.completeness_score = self.calculate_endpoint_score(ep)
        tool.completeness_score = self.calculate_tool_score(tool)
        return tool


# ---------------------------------------------------------------------------
# Score breakdown
# ---------------------------------------------------------------------------


@dataclass
class ScoreBreakdown:
    """Detailed breakdown of a tool's completeness score."""

    total_score: float
    component_scores: dict[str, float]
    missing_fields: list[str]
    quality_tier: QualityTier
    recommendations: list[str] = field(default_factory=list)


def get_score_breakdown(
    tool: Tool,
    calculator: CompletenessCalculator | None = None,
) -> ScoreBreakdown:
    """Return a detailed breakdown of *tool*'s score."""
    calc = calculator or CompletenessCalculator()
    w = calc.weights

    components: dict[str, float] = {}
    missing: list[str] = []
    recs: list[str] = []

    # Tool-level signals
    has_name = bool(tool.name and tool.name.strip())
    components["tool_name"] = w.tool_name if has_name else 0.0
    if not has_name:
        missing.append("tool.name")
        recs.append("Add a descriptive tool name.")

    has_desc = is_meaningful_description(tool.description, tool.name)
    components["tool_description"] = w.tool_description if has_desc else 0.0
    if not has_desc:
        missing.append("tool.description")
        recs.append("Add a description of at least 10 characters.")

    has_domain = bool(tool.domain and tool.domain.strip())
    components["tool_domain"] = w.tool_domain if has_domain else 0.0
    if not has_domain:
        missing.append("tool.domain")
        recs.append("Assign a domain/category to improve sampling diversity.")

    has_eps = bool(tool.endpoints)
    components["tool_has_endpoints"] = w.tool_has_endpoints if has_eps else 0.0
    if not has_eps:
        missing.append("tool.endpoints")
        recs.append("Define at least one endpoint.")

    if has_eps:
        avg_ep = sum(calc.calculate_endpoint_score(ep) for ep in tool.endpoints) / len(
            tool.endpoints
        )
        components["endpoint_quality"] = round(w.endpoint_quality * avg_ep, 4)
        if avg_ep < 0.5:
            recs.append("Improve endpoint documentation (descriptions, parameters, response schemas).")
    else:
        components["endpoint_quality"] = 0.0

    total = round(min(sum(components.values()), 1.0), 4)
    tier = get_quality_tier(total)

    return ScoreBreakdown(
        total_score=total,
        component_scores=components,
        missing_fields=missing,
        quality_tier=tier,
        recommendations=recs,
    )


# ---------------------------------------------------------------------------
# Aggregate quality report
# ---------------------------------------------------------------------------


def generate_quality_report(tools: list[Tool]) -> dict[str, object]:
    """Generate an aggregate quality report across all *tools*.

    Returns a dict with::

        {
            "total_tools": int,
            "tier_distribution": {str: int},
            "average_score": float,
            "score_histogram": list[int],   # 10 buckets [0.0-0.1), …, [0.9-1.0]
            "top_issues": list[str],
        }
    """
    if not tools:
        return {
            "total_tools": 0,
            "tier_distribution": {t.value: 0 for t in QualityTier},
            "average_score": 0.0,
            "score_histogram": [0] * 10,
            "top_issues": [],
        }

    tier_counts: dict[str, int] = {t.value: 0 for t in QualityTier}
    histogram = [0] * 10
    issue_counter: Counter[str] = Counter()

    calc = CompletenessCalculator()
    for tool in tools:
        score = tool.completeness_score
        tier = get_quality_tier(score)
        tier_counts[tier.value] += 1

        bucket = min(int(score * 10), 9)
        histogram[bucket] += 1

        breakdown = get_score_breakdown(tool, calc)
        for field_name in breakdown.missing_fields:
            issue_counter[field_name] += 1

    avg = round(sum(t.completeness_score for t in tools) / len(tools), 4)
    top_issues = [issue for issue, _ in issue_counter.most_common(5)]

    return {
        "total_tools": len(tools),
        "tier_distribution": tier_counts,
        "average_score": avg,
        "score_histogram": histogram,
        "top_issues": top_issues,
    }
