"""E2E test — A/B diversity experiment (Task 68).

Spec requirement: "Run pipeline twice with same seed, steering disabled vs
enabled."

* **Run A** — ``--no-cross-conversation-steering`` (diversity steering off).
* **Run B** — default (diversity steering on).

Both use ``--seed 42 --count 50``.  We compare tool-usage entropy, unique
tool counts, unique tool-combination counts, and mean judge scores.

With small test graphs, steering may produce identical diversity.  Assertions
therefore use ``>=`` (not strict ``>``).  With a production-size graph the
steered run should show measurably higher entropy and coverage.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from tests.conftest import _FALLBACK_TOOLS, MockEmbeddingService
from tooluse_gen.cli.main import app

pytestmark = pytest.mark.e2e

runner = CliRunner()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PATCHES = (
    patch("tooluse_gen.graph.builder.EmbeddingService", MockEmbeddingService),
    patch("tooluse_gen.graph.embeddings.EmbeddingService", MockEmbeddingService),
)


def _parse_eval_report(output: str) -> dict:
    """Extract the JSON report object from evaluate command output."""
    match = re.search(r"\{[\s\S]*\}", output)
    return json.loads(match.group()) if match else {}


def _read_records(path: Path) -> list[dict]:
    records: list[dict] = []
    for line in path.read_text().strip().split("\n"):
        if not line:
            continue
        data = json.loads(line)
        if "conversation_id" in data:
            records.append(data)
    return records


# ---------------------------------------------------------------------------
# Module-scoped fixture — runs the full A/B experiment ONCE
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ab_results(
    tmp_path_factory: pytest.TempPathFactory,
    toolbench_subset: Path | None,
) -> tuple[dict, dict, list[dict], list[dict], Path]:
    """Run the A/B diversity experiment and return parsed results.

    Returns ``(report_a, report_b, records_a, records_b, results_file)``.
    """
    tmp = tmp_path_factory.mktemp("diversity_ab")

    # --- Prepare data directory ------------------------------------------
    if toolbench_subset is not None:
        data_dir = tmp / "data"
        shutil.copytree(toolbench_subset, data_dir)
    else:
        data_dir = tmp / "data"
        data_dir.mkdir()
        for i, tool in enumerate(_FALLBACK_TOOLS):
            (data_dir / f"tool_{i}.json").write_text(json.dumps(tool))

    build_out = tmp / "build"
    out_a = tmp / "run_a.jsonl"
    out_b = tmp / "run_b.jsonl"

    p1, p2 = _PATCHES
    with p1, p2:
        # Build (shared)
        result = runner.invoke(app, [
            "build", "--input-dir", str(data_dir), "--output-dir", str(build_out),
            "--force", "--similarity-threshold", "0.1",
        ])
        assert result.exit_code == 0, f"Build failed:\n{result.output[-500:]}"

        # Run A — steering disabled
        result = runner.invoke(app, [
            "generate", "--output", str(out_a), "--build-dir", str(build_out),
            "--count", "50", "--seed", "42",
            "--min-steps", "1", "--max-steps", "3",
            "--no-cross-conversation-steering",
        ])
        assert result.exit_code == 0, f"Run A failed:\n{result.output[-500:]}"

        # Run B — steering enabled (default)
        result = runner.invoke(app, [
            "generate", "--output", str(out_b), "--build-dir", str(build_out),
            "--count", "50", "--seed", "42",
            "--min-steps", "1", "--max-steps", "3",
        ])
        assert result.exit_code == 0, f"Run B failed:\n{result.output[-500:]}"

        # Evaluate both
        eval_a = runner.invoke(app, ["evaluate", str(out_a), "--format", "json"])
        assert eval_a.exit_code == 0, f"Eval A failed:\n{eval_a.output[-500:]}"

        eval_b = runner.invoke(app, ["evaluate", str(out_b), "--format", "json"])
        assert eval_b.exit_code == 0, f"Eval B failed:\n{eval_b.output[-500:]}"

    # Parse
    report_a = _parse_eval_report(eval_a.output)
    report_b = _parse_eval_report(eval_b.output)
    records_a = _read_records(out_a)
    records_b = _read_records(out_b)

    # --- Write results JSON for DESIGN.md --------------------------------
    def _run_summary(label: str, report: dict, records: list[dict], steering: bool) -> dict:
        div = report.get("diversity", {})
        avg = report.get("average_scores", {})
        scores_vals = [
            avg.get("tool_correctness", 0), avg.get("argument_grounding", 0),
            avg.get("task_completion", 0), avg.get("naturalness", 0),
        ]
        return {
            "label": label,
            "steering": steering,
            "count": len(records),
            "tool_entropy": div.get("tool_entropy", 0),
            "unique_tools": div.get("unique_tools", 0),
            "unique_domains": div.get("unique_domains", 0),
            "unique_tool_combos": div.get("unique_tool_combos", 0),
            "mean_score": round(sum(scores_vals) / 4, 2) if scores_vals else 0,
            "scores": avg,
        }

    results = {
        "run_a": _run_summary("A (no steering)", report_a, records_a, steering=False),
        "run_b": _run_summary("B (steering)", report_b, records_b, steering=True),
    }
    results_file = tmp / "diversity_results.json"
    results_file.write_text(json.dumps(results, indent=2))

    return report_a, report_b, records_a, records_b, results_file


# ===================================================================
# Tests
# ===================================================================


class TestDiversityExperiment:
    """A/B diversity experiment: steering disabled (A) vs enabled (B)."""

    # --- Record counts ---------------------------------------------------

    def test_both_runs_produce_50_conversations(
        self, ab_results: tuple[dict, dict, list[dict], list[dict], Path]
    ) -> None:
        """Both A and B generate 50 conversations."""
        _, _, records_a, records_b, _ = ab_results
        assert len(records_a) == 50
        assert len(records_b) == 50

    # --- Diversity comparisons -------------------------------------------

    def test_tool_entropy_b_gte_a(
        self, ab_results: tuple[dict, dict, list[dict], list[dict], Path]
    ) -> None:
        """Run B has tool entropy >= Run A.

        Steering nudges toward underrepresented tools, producing a more
        uniform distribution.  With small graphs the effect may be zero.
        """
        report_a, report_b, _, _, _ = ab_results
        ent_a = report_a.get("diversity", {}).get("tool_entropy", 0)
        ent_b = report_b.get("diversity", {}).get("tool_entropy", 0)
        assert ent_b >= ent_a, f"Entropy B={ent_b:.4f} < A={ent_a:.4f}"

    def test_unique_tools_b_gte_a(
        self, ab_results: tuple[dict, dict, list[dict], list[dict], Path]
    ) -> None:
        """Run B uses at least as many unique tools as Run A."""
        report_a, report_b, _, _, _ = ab_results
        tools_a = report_a.get("diversity", {}).get("unique_tools", 0)
        tools_b = report_b.get("diversity", {}).get("unique_tools", 0)
        assert tools_b >= tools_a, f"Unique tools B={tools_b} < A={tools_a}"

    def test_unique_combos_b_gte_a(
        self, ab_results: tuple[dict, dict, list[dict], list[dict], Path]
    ) -> None:
        """Run B has at least as many unique tool combinations as Run A."""
        report_a, report_b, _, _, _ = ab_results
        combos_a = report_a.get("diversity", {}).get("unique_tool_combos", 0)
        combos_b = report_b.get("diversity", {}).get("unique_tool_combos", 0)
        assert combos_b >= combos_a, f"Combos B={combos_b} < A={combos_a}"

    # --- Quality ---------------------------------------------------------

    def test_quality_scores_comparable(
        self, ab_results: tuple[dict, dict, list[dict], list[dict], Path]
    ) -> None:
        """Both runs have mean score >= 3.0.

        Steering may slightly reduce quality by forcing less-optimal tool
        combinations — this is the documented diversity-quality tradeoff.
        """
        for label, report in [("A", ab_results[0]), ("B", ab_results[1])]:
            avg = report.get("average_scores", {})
            vals = [
                avg.get("tool_correctness", 0), avg.get("argument_grounding", 0),
                avg.get("task_completion", 0), avg.get("naturalness", 0),
            ]
            mean = sum(vals) / 4
            assert mean >= 3.0, f"Run {label} mean {mean:.2f} < 3.0"

    def test_both_have_judge_scores(
        self, ab_results: tuple[dict, dict, list[dict], list[dict], Path]
    ) -> None:
        """Both runs have judge scores on their records."""
        _, _, records_a, records_b, _ = ab_results
        for label, records in [("A", records_a), ("B", records_b)]:
            with_scores = sum(1 for r in records if r.get("judge_scores"))
            assert with_scores >= len(records) * 0.8, (
                f"Run {label}: only {with_scores}/{len(records)} have scores"
            )

    def test_both_have_tool_calls(
        self, ab_results: tuple[dict, dict, list[dict], list[dict], Path]
    ) -> None:
        """Both runs produce conversations with tool calls."""
        _, _, records_a, records_b, _ = ab_results
        for label, records in [("A", records_a), ("B", records_b)]:
            has_tc = sum(
                1 for r in records
                if any(m.get("tool_calls") for m in r["messages"])
            )
            assert has_tc >= len(records) * 0.5, (
                f"Run {label}: only {has_tc}/{len(records)} have tool calls"
            )

    # --- Results file ----------------------------------------------------

    def test_results_file_written(
        self, ab_results: tuple[dict, dict, list[dict], list[dict], Path]
    ) -> None:
        """A results JSON file is written for DESIGN.md."""
        _, _, _, _, results_file = ab_results
        assert results_file.exists()
        assert results_file.stat().st_size > 10

    def test_results_file_has_both_runs(
        self, ab_results: tuple[dict, dict, list[dict], list[dict], Path]
    ) -> None:
        """Results file contains data for both run_a and run_b."""
        _, _, _, _, results_file = ab_results
        data = json.loads(results_file.read_text())
        assert "run_a" in data
        assert "run_b" in data

    def test_results_file_has_metrics(
        self, ab_results: tuple[dict, dict, list[dict], list[dict], Path]
    ) -> None:
        """Results file includes entropy, unique_tools, scores for both."""
        _, _, _, _, results_file = ab_results
        data = json.loads(results_file.read_text())
        for run_key in ("run_a", "run_b"):
            run = data[run_key]
            assert "tool_entropy" in run
            assert "unique_tools" in run
            assert "mean_score" in run
            assert "scores" in run
            assert "steering" in run
