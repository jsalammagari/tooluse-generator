"""Tests for diversity report module (Task 70)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tooluse_gen.core.output_models import ConversationRecord
from tooluse_gen.evaluation.diversity_report import (
    ComparisonReport,
    RunMetrics,
    compute_run_metrics,
    format_json,
    format_markdown,
    generate_comparison_report,
    load_and_compute,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------

_RECORDS_A: list[ConversationRecord] = [
    ConversationRecord(
        conversation_id="a1",
        messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}],
        judge_scores={
            "tool_correctness": 4, "argument_grounding": 3,
            "task_completion": 5, "naturalness": 4,
        },
        metadata={
            "seed": 42, "tools_used": ["hotels_api"], "domains": ["Travel"],
            "num_turns": 4, "num_tool_calls": 1, "num_distinct_tools": 1,
        },
    ),
    ConversationRecord(
        conversation_id="a2",
        messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}],
        judge_scores={
            "tool_correctness": 5, "argument_grounding": 4,
            "task_completion": 5, "naturalness": 5,
        },
        metadata={
            "seed": 43, "tools_used": ["weather_api", "hotels_api"], "domains": ["Weather", "Travel"],
            "num_turns": 6, "num_tool_calls": 3, "num_distinct_tools": 2,
        },
    ),
    ConversationRecord(
        conversation_id="a3",
        messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}],
        judge_scores={
            "tool_correctness": 3, "argument_grounding": 2,
            "task_completion": 4, "naturalness": 3,
        },
        metadata={
            "seed": 44, "tools_used": ["flights_api"], "domains": ["Travel"],
            "num_turns": 5, "num_tool_calls": 2, "num_distinct_tools": 1,
        },
    ),
]

_RECORDS_B: list[ConversationRecord] = [
    ConversationRecord(
        conversation_id="b1",
        messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}],
        judge_scores={
            "tool_correctness": 4, "argument_grounding": 3,
            "task_completion": 4, "naturalness": 3,
        },
        metadata={
            "seed": 42, "tools_used": ["music_api", "sports_api"], "domains": ["Music", "Sports"],
            "num_turns": 6, "num_tool_calls": 3, "num_distinct_tools": 2,
        },
    ),
    ConversationRecord(
        conversation_id="b2",
        messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}],
        judge_scores={
            "tool_correctness": 3, "argument_grounding": 3,
            "task_completion": 5, "naturalness": 4,
        },
        metadata={
            "seed": 43, "tools_used": ["weather_api"], "domains": ["Weather"],
            "num_turns": 4, "num_tool_calls": 1, "num_distinct_tools": 1,
        },
    ),
    ConversationRecord(
        conversation_id="b3",
        messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}],
        judge_scores={
            "tool_correctness": 4, "argument_grounding": 2,
            "task_completion": 5, "naturalness": 3,
        },
        metadata={
            "seed": 44, "tools_used": ["finance_api", "hotels_api"], "domains": ["Finance", "Travel"],
            "num_turns": 7, "num_tool_calls": 4, "num_distinct_tools": 2,
        },
    ),
]


def _metrics_a() -> RunMetrics:
    return compute_run_metrics(_RECORDS_A, label="A", steering_enabled=False)


def _metrics_b() -> RunMetrics:
    return compute_run_metrics(_RECORDS_B, label="B", steering_enabled=True)


# ===================================================================
# RunMetrics
# ===================================================================


class TestRunMetrics:
    def test_compute_basic(self) -> None:
        m = _metrics_a()
        assert isinstance(m, RunMetrics)
        assert m.conversation_count == 3

    def test_label_and_steering(self) -> None:
        m = _metrics_a()
        assert m.label == "A"
        assert m.steering_enabled is False

    def test_tool_entropy_computed(self) -> None:
        m = _metrics_a()
        assert m.tool_entropy > 0  # multiple tools

    def test_unique_tools(self) -> None:
        m = _metrics_a()
        # hotels_api, weather_api, flights_api
        assert m.unique_tools == 3

    def test_unique_domains(self) -> None:
        m = _metrics_a()
        # Travel, Weather
        assert m.unique_domains == 2

    def test_unique_tool_combos(self) -> None:
        m = _metrics_a()
        # {hotels_api}, {weather_api, hotels_api}, {flights_api} → 3 unique
        assert m.unique_tool_combos == 3

    def test_total_tool_calls(self) -> None:
        m = _metrics_a()
        assert m.total_tool_calls == 6  # 1 + 3 + 2

    def test_quality_scores_averaged(self) -> None:
        m = _metrics_a()
        # tc: (4+5+3)/3=4.0, ag: (3+4+2)/3=3.0, comp: (5+5+4)/3=4.67, nat: (4+5+3)/3=4.0
        assert m.tool_correctness == 4.0
        assert m.argument_grounding == 3.0
        assert abs(m.mean_score - (4.0 + 3.0 + 4.67 + 4.0) / 4) < 0.1

    def test_multi_step_rate(self) -> None:
        m = _metrics_a()
        # Only a2 has 3 tool calls → 1/3
        assert abs(m.multi_step_rate - 1 / 3) < 0.01

    def test_multi_tool_rate(self) -> None:
        m = _metrics_a()
        # Only a2 has 2 distinct tools → 1/3
        assert abs(m.multi_tool_rate - 1 / 3) < 0.01

    def test_avg_turns(self) -> None:
        m = _metrics_a()
        assert abs(m.avg_turns - 5.0) < 0.01  # (4+6+5)/3

    def test_avg_tool_calls(self) -> None:
        m = _metrics_a()
        assert abs(m.avg_tool_calls - 2.0) < 0.01  # (1+3+2)/3

    def test_empty_records(self) -> None:
        m = compute_run_metrics([])
        assert m.conversation_count == 0
        assert m.tool_entropy == 0.0
        assert m.mean_score == 0.0
        assert m.unique_tools == 0

    def test_records_without_scores(self) -> None:
        records = [
            ConversationRecord(
                conversation_id="x1",
                messages=[{"role": "user", "content": "hi"}],
                judge_scores=None,
                metadata={"tools_used": ["t1"], "num_turns": 2, "num_tool_calls": 1,
                          "num_distinct_tools": 1, "domains": []},
            ),
        ]
        m = compute_run_metrics(records)
        assert m.conversation_count == 1
        assert m.mean_score == 0.0
        assert m.unique_tools == 1

    def test_pattern_repetition_rate(self) -> None:
        m = _metrics_a()
        # 3 unique combos out of 3 conversations → 1 - 3/3 = 0
        assert m.pattern_repetition_rate == 0.0

    def test_pair_ratio(self) -> None:
        m = _metrics_a()
        # 3 tools → 3 possible pairs, a2 contributes (hotels_api, weather_api) = 1 pair
        assert m.unique_tool_pair_ratio > 0


# ===================================================================
# ComparisonReport
# ===================================================================


class TestComparisonReport:
    def test_generate_report(self) -> None:
        report = generate_comparison_report(_metrics_a(), _metrics_b(), seed=42)
        assert isinstance(report, ComparisonReport)
        assert report.seed == 42

    def test_analysis_text(self) -> None:
        report = generate_comparison_report(_metrics_a(), _metrics_b())
        assert len(report.analysis) > 0

    def test_analysis_entropy_change(self) -> None:
        a = _metrics_a()
        b = _metrics_b()
        report = generate_comparison_report(a, b)
        assert "entropy" in report.analysis.lower()

    def test_analysis_quality_tradeoff(self) -> None:
        # Create metrics where B has lower quality
        a = RunMetrics(mean_score=4.0, tool_entropy=2.0)
        b = RunMetrics(mean_score=3.5, tool_entropy=3.0)
        report = generate_comparison_report(a, b)
        assert "tradeoff" in report.analysis.lower()

    def test_analysis_no_tradeoff(self) -> None:
        a = RunMetrics(mean_score=3.5, tool_entropy=2.0)
        b = RunMetrics(mean_score=4.0, tool_entropy=3.0)
        report = generate_comparison_report(a, b)
        assert "no tradeoff" in report.analysis.lower()


# ===================================================================
# format_markdown
# ===================================================================


class TestFormatMarkdown:
    def test_produces_markdown(self) -> None:
        report = generate_comparison_report(_metrics_a(), _metrics_b())
        md = format_markdown(report)
        assert isinstance(md, str)
        assert len(md) > 100

    def test_has_header(self) -> None:
        report = generate_comparison_report(_metrics_a(), _metrics_b())
        md = format_markdown(report)
        assert "## Diversity Experiment Results" in md

    def test_has_table(self) -> None:
        report = generate_comparison_report(_metrics_a(), _metrics_b())
        md = format_markdown(report)
        assert "| Metric |" in md

    def test_has_analysis_section(self) -> None:
        report = generate_comparison_report(_metrics_a(), _metrics_b())
        md = format_markdown(report)
        assert "### Analysis" in md

    def test_has_delta_column(self) -> None:
        report = generate_comparison_report(_metrics_a(), _metrics_b())
        md = format_markdown(report)
        assert "Delta" in md
        assert "+" in md or "-" in md

    def test_has_all_metrics(self) -> None:
        report = generate_comparison_report(_metrics_a(), _metrics_b())
        md = format_markdown(report)
        for metric in [
            "Tool entropy", "Unique tools", "Unique combos",
            "Mean score", "tool_correctness", "naturalness",
            "Multi-step rate", "Multi-tool rate",
        ]:
            assert metric in md, f"Missing metric: {metric}"


# ===================================================================
# format_json
# ===================================================================


class TestFormatJSON:
    def test_produces_valid_json(self) -> None:
        report = generate_comparison_report(_metrics_a(), _metrics_b())
        text = format_json(report)
        data = json.loads(text)
        assert isinstance(data, dict)

    def test_has_run_a_and_b(self) -> None:
        report = generate_comparison_report(_metrics_a(), _metrics_b())
        data = json.loads(format_json(report))
        assert "run_a" in data
        assert "run_b" in data

    def test_has_analysis(self) -> None:
        report = generate_comparison_report(_metrics_a(), _metrics_b())
        data = json.loads(format_json(report))
        assert "analysis" in data
        assert len(data["analysis"]) > 0


# ===================================================================
# load_and_compute
# ===================================================================


class TestLoadAndCompute:
    def test_loads_from_file(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "test.jsonl"
        with open(jsonl, "w") as fh:
            for rec in _RECORDS_A:
                fh.write(rec.to_jsonl() + "\n")
        m = load_and_compute(jsonl, label="test")
        assert m.conversation_count == 3
        assert m.unique_tools == 3

    def test_label_and_steering(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "test.jsonl"
        with open(jsonl, "w") as fh:
            for rec in _RECORDS_A:
                fh.write(rec.to_jsonl() + "\n")
        m = load_and_compute(jsonl, label="X", steering_enabled=True)
        assert m.label == "X"
        assert m.steering_enabled is True

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_and_compute(tmp_path / "nope.jsonl")
