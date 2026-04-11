"""Comparative diversity and quality report for A/B experiment runs.

:class:`RunMetrics` captures diversity and quality metrics from a single run.
:func:`compute_run_metrics` extracts metrics from a list of records.
:func:`generate_comparison_report` produces a :class:`ComparisonReport` from
two runs.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from tooluse_gen.utils.logging import get_logger

if TYPE_CHECKING:
    from tooluse_gen.core.output_models import ConversationRecord

logger = get_logger("evaluation.diversity_report")


# ---------------------------------------------------------------------------
# RunMetrics
# ---------------------------------------------------------------------------


class RunMetrics(BaseModel):
    """Diversity and quality metrics for a single experiment run."""

    model_config = ConfigDict(use_enum_values=True)

    label: str = Field(default="", description="Run label (e.g. 'A' or 'B').")
    steering_enabled: bool = Field(default=False, description="Whether steering was on.")
    conversation_count: int = Field(default=0, description="Number of conversations.")

    # Diversity metrics
    tool_entropy: float = Field(default=0.0, description="Shannon entropy of tool usage.")
    domain_coverage: float = Field(
        default=0.0, description="Fraction of unique domains used."
    )
    unique_tool_pair_ratio: float = Field(
        default=0.0, description="Unique tool pairs / total possible."
    )
    pattern_repetition_rate: float = Field(
        default=0.0, description="1 - (unique patterns / total)."
    )
    unique_tools: int = Field(default=0, description="Count of distinct tools.")
    unique_domains: int = Field(default=0, description="Count of distinct domains.")
    unique_tool_combos: int = Field(default=0, description="Count of distinct tool sets.")
    total_tool_calls: int = Field(default=0, description="Sum of tool calls.")

    # Quality metrics
    mean_score: float = Field(default=0.0, description="Mean of all 4 dimensions.")
    tool_correctness: float = Field(default=0.0)
    argument_grounding: float = Field(default=0.0)
    task_completion: float = Field(default=0.0)
    naturalness: float = Field(default=0.0)

    # Conversation stats
    avg_turns: float = Field(default=0.0)
    avg_tool_calls: float = Field(default=0.0)
    multi_step_rate: float = Field(
        default=0.0, description="Fraction with >=3 tool calls."
    )
    multi_tool_rate: float = Field(
        default=0.0, description="Fraction with >=2 distinct tools."
    )


# ---------------------------------------------------------------------------
# ComparisonReport
# ---------------------------------------------------------------------------


class ComparisonReport(BaseModel):
    """Side-by-side comparison of two experiment runs."""

    model_config = ConfigDict(use_enum_values=True)

    run_a: RunMetrics = Field(..., description="Metrics for Run A (no steering).")
    run_b: RunMetrics = Field(..., description="Metrics for Run B (with steering).")
    seed: int = Field(default=42, description="Shared seed.")
    analysis: str = Field(default="", description="Textual tradeoff analysis.")


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_run_metrics(
    records: list[ConversationRecord],
    label: str = "",
    steering_enabled: bool = False,
) -> RunMetrics:
    """Compute diversity and quality metrics from conversation records."""
    n = len(records)
    if n == 0:
        return RunMetrics(label=label, steering_enabled=steering_enabled)

    # -- Collect raw data --------------------------------------------------
    tool_counter: Counter[str] = Counter()
    domain_set: set[str] = set()
    tool_combos: set[frozenset[str]] = set()
    tool_pairs: set[tuple[str, str]] = set()
    all_tools: set[str] = set()

    total_calls = 0
    total_turns = 0
    multi_step = 0
    multi_tool = 0

    for rec in records:
        meta = rec.metadata or {}
        tools_used: list[str] = meta.get("tools_used", [])
        domains: list[str] = meta.get("domains", [])
        num_calls: int = meta.get("num_tool_calls", 0)
        num_distinct: int = meta.get("num_distinct_tools", 0)
        num_turns: int = meta.get("num_turns", 0)

        for tool in tools_used:
            tool_counter[tool] += 1
            all_tools.add(tool)
        domain_set.update(domains)
        tool_combos.add(frozenset(tools_used))

        sorted_tools = sorted(set(tools_used))
        for i in range(len(sorted_tools)):
            for j in range(i + 1, len(sorted_tools)):
                tool_pairs.add((sorted_tools[i], sorted_tools[j]))

        total_calls += num_calls
        total_turns += num_turns
        if num_calls >= 3:
            multi_step += 1
        if num_distinct >= 2:
            multi_tool += 1

    # -- Diversity metrics -------------------------------------------------
    total_tool_uses = sum(tool_counter.values())
    entropy = 0.0
    if total_tool_uses > 0:
        for cnt in tool_counter.values():
            p = cnt / total_tool_uses
            if p > 0:
                entropy -= p * math.log2(p)

    unique_tools_count = len(all_tools)
    possible_pairs = unique_tools_count * (unique_tools_count - 1) // 2
    pair_ratio = len(tool_pairs) / possible_pairs if possible_pairs > 0 else 0.0
    repetition = 1.0 - (len(tool_combos) / n) if n > 0 else 0.0

    # -- Quality metrics ---------------------------------------------------
    scored = [
        rec.judge_scores
        for rec in records
        if rec.judge_scores is not None
    ]
    if scored:
        tc = sum(s.get("tool_correctness", 0) for s in scored) / len(scored)
        ag = sum(s.get("argument_grounding", 0) for s in scored) / len(scored)
        comp = sum(s.get("task_completion", 0) for s in scored) / len(scored)
        nat = sum(s.get("naturalness", 0) for s in scored) / len(scored)
        mean = (tc + ag + comp + nat) / 4.0
    else:
        tc = ag = comp = nat = mean = 0.0

    return RunMetrics(
        label=label,
        steering_enabled=steering_enabled,
        conversation_count=n,
        tool_entropy=round(entropy, 4),
        domain_coverage=round(len(domain_set) / max(len(domain_set), 1), 4),
        unique_tool_pair_ratio=round(pair_ratio, 4),
        pattern_repetition_rate=round(repetition, 4),
        unique_tools=unique_tools_count,
        unique_domains=len(domain_set),
        unique_tool_combos=len(tool_combos),
        total_tool_calls=total_calls,
        mean_score=round(mean, 2),
        tool_correctness=round(tc, 2),
        argument_grounding=round(ag, 2),
        task_completion=round(comp, 2),
        naturalness=round(nat, 2),
        avg_turns=round(total_turns / n, 2),
        avg_tool_calls=round(total_calls / n, 2),
        multi_step_rate=round(multi_step / n, 4),
        multi_tool_rate=round(multi_tool / n, 4),
    )


# ---------------------------------------------------------------------------
# Load and compute
# ---------------------------------------------------------------------------


def load_and_compute(
    jsonl_path: Path | str,
    label: str = "",
    steering_enabled: bool = False,
) -> RunMetrics:
    """Load records from JSONL and compute metrics."""
    from tooluse_gen.core.jsonl_io import JSONLReader

    reader = JSONLReader(jsonl_path)
    records = reader.read_all()
    return compute_run_metrics(records, label=label, steering_enabled=steering_enabled)


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------


def generate_comparison_report(
    run_a: RunMetrics,
    run_b: RunMetrics,
    seed: int = 42,
) -> ComparisonReport:
    """Generate a comparison report from two runs' metrics."""
    analysis = _build_analysis(run_a, run_b)
    return ComparisonReport(run_a=run_a, run_b=run_b, seed=seed, analysis=analysis)


def _build_analysis(run_a: RunMetrics, run_b: RunMetrics) -> str:
    """Generate textual analysis of the diversity-quality tradeoff."""
    lines: list[str] = []

    # Entropy
    ent_diff = run_b.tool_entropy - run_a.tool_entropy
    if ent_diff > 0.001:
        lines.append(
            f"Steering increased tool entropy by {ent_diff:.4f} "
            f"({run_a.tool_entropy:.4f} -> {run_b.tool_entropy:.4f}), "
            f"indicating a more uniform tool distribution."
        )
    elif ent_diff < -0.001:
        lines.append(
            f"Steering decreased tool entropy by {abs(ent_diff):.4f} "
            f"({run_a.tool_entropy:.4f} -> {run_b.tool_entropy:.4f})."
        )
    else:
        lines.append("Tool entropy was identical between runs.")

    # Unique tools
    tool_diff = run_b.unique_tools - run_a.unique_tools
    if tool_diff > 0:
        lines.append(
            f"Steering used {tool_diff} more unique tools "
            f"({run_a.unique_tools} -> {run_b.unique_tools})."
        )
    elif tool_diff < 0:
        lines.append(
            f"Steering used {abs(tool_diff)} fewer unique tools "
            f"({run_a.unique_tools} -> {run_b.unique_tools})."
        )

    # Unique combos
    combo_diff = run_b.unique_tool_combos - run_a.unique_tool_combos
    if combo_diff > 0:
        lines.append(
            f"Steering produced {combo_diff} more unique tool combinations "
            f"({run_a.unique_tool_combos} -> {run_b.unique_tool_combos})."
        )

    # Quality tradeoff
    quality_diff = run_b.mean_score - run_a.mean_score
    if quality_diff < -0.01:
        lines.append(
            f"Quality decreased by {abs(quality_diff):.2f} "
            f"({run_a.mean_score:.2f} -> {run_b.mean_score:.2f}) "
            f"— diversity-quality tradeoff observed."
        )
    elif quality_diff > 0.01:
        lines.append(
            f"Quality increased by {quality_diff:.2f} "
            f"({run_a.mean_score:.2f} -> {run_b.mean_score:.2f}) "
            f"— no tradeoff."
        )
    else:
        lines.append("Quality was identical — no tradeoff observed.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_markdown(report: ComparisonReport) -> str:
    """Format the report as a markdown table for DESIGN.md."""
    a, b = report.run_a, report.run_b
    lines = [
        "## Diversity Experiment Results",
        "",
        f"Seed: {report.seed} | Count: {a.conversation_count} per run",
        "",
        "| Metric | Run A (no steering) | Run B (steering) | Delta |",
        "|--------|-------------------|-----------------|-------|",
        (
            f"| Tool entropy | {a.tool_entropy:.4f} | {b.tool_entropy:.4f} "
            f"| {b.tool_entropy - a.tool_entropy:+.4f} |"
        ),
        (
            f"| Unique tools | {a.unique_tools} | {b.unique_tools} "
            f"| {b.unique_tools - a.unique_tools:+d} |"
        ),
        (
            f"| Unique combos | {a.unique_tool_combos} | {b.unique_tool_combos} "
            f"| {b.unique_tool_combos - a.unique_tool_combos:+d} |"
        ),
        (
            f"| Unique domains | {a.unique_domains} | {b.unique_domains} "
            f"| {b.unique_domains - a.unique_domains:+d} |"
        ),
        (
            f"| Pattern repetition | {a.pattern_repetition_rate:.2%} "
            f"| {b.pattern_repetition_rate:.2%} "
            f"| {b.pattern_repetition_rate - a.pattern_repetition_rate:+.2%} |"
        ),
        (
            f"| Mean score | {a.mean_score:.2f} | {b.mean_score:.2f} "
            f"| {b.mean_score - a.mean_score:+.2f} |"
        ),
        (
            f"| tool_correctness | {a.tool_correctness:.1f} | {b.tool_correctness:.1f} "
            f"| {b.tool_correctness - a.tool_correctness:+.1f} |"
        ),
        (
            f"| argument_grounding | {a.argument_grounding:.1f} "
            f"| {b.argument_grounding:.1f} "
            f"| {b.argument_grounding - a.argument_grounding:+.1f} |"
        ),
        (
            f"| task_completion | {a.task_completion:.1f} | {b.task_completion:.1f} "
            f"| {b.task_completion - a.task_completion:+.1f} |"
        ),
        (
            f"| naturalness | {a.naturalness:.1f} | {b.naturalness:.1f} "
            f"| {b.naturalness - a.naturalness:+.1f} |"
        ),
        (
            f"| Multi-step rate | {a.multi_step_rate:.1%} | {b.multi_step_rate:.1%} "
            f"| {b.multi_step_rate - a.multi_step_rate:+.1%} |"
        ),
        (
            f"| Multi-tool rate | {a.multi_tool_rate:.1%} | {b.multi_tool_rate:.1%} "
            f"| {b.multi_tool_rate - a.multi_tool_rate:+.1%} |"
        ),
        "",
        "### Analysis",
        "",
        report.analysis,
    ]
    return "\n".join(lines)


def format_json(report: ComparisonReport) -> str:
    """Format the report as a JSON string."""
    return report.model_dump_json(indent=2)
