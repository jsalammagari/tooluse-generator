#!/usr/bin/env python
"""Run the A/B diversity experiment.

Compares generation with cross-conversation steering disabled (Run A)
versus enabled (Run B) using the same seed.

Usage::

    python scripts/run_diversity_experiment.py
    python scripts/run_diversity_experiment.py --seed 42 --count 50
    python scripts/run_diversity_experiment.py --categories Finance,Food,Weather --count 20
    python scripts/run_diversity_experiment.py --input-dir data/toolenv/tools --output-dir output/experiment
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Run A/B diversity experiment: steering off (A) vs on (B).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument(
        "--count", type=int, default=50, help="Conversations per run (default: 50)"
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/toolenv/tools"),
        help="ToolBench data directory (default: data/toolenv/tools)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/diversity_experiment"),
        help="Output directory (default: output/diversity_experiment)",
    )
    p.add_argument(
        "--min-steps", type=int, default=1, help="Min tool calls (default: 1)"
    )
    p.add_argument(
        "--max-steps", type=int, default=3, help="Max tool calls (default: 3)"
    )
    p.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.1,
        help="Graph similarity threshold (default: 0.1)",
    )
    p.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated categories to use (default: all). E.g. Finance,Food,Weather",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run a tooluse CLI command via subprocess and return the result."""
    cmd = [sys.executable, "-m", "tooluse_gen.cli.main", *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603


def _prepare_subset(
    input_dir: Path,
    categories: list[str],
    output_dir: Path,
) -> Path:
    """Copy selected categories to a subset directory."""
    subset = output_dir / "data_subset"
    if subset.exists():
        shutil.rmtree(subset)
    subset.mkdir(parents=True)
    copied = 0
    for cat in categories:
        src = input_dir / cat.strip()
        if src.is_dir():
            shutil.copytree(src, subset / cat.strip())
            copied += 1
        else:
            print(f"  WARNING: category '{cat.strip()}' not found in {input_dir}")
    if copied == 0:
        print(f"ERROR: no valid categories found in {input_dir}")
        sys.exit(1)
    return subset


def _evaluate(jsonl_path: Path) -> dict:
    """Run ``evaluate --format json`` and parse the report."""
    result = run_cli("evaluate", str(jsonl_path), "--format", "json")
    if result.returncode != 0:
        print(f"Evaluate failed for {jsonl_path}:\n{result.stdout}\n{result.stderr}")
        return {}
    match = re.search(r"\{[\s\S]*\}", result.stdout)
    return json.loads(match.group()) if match else {}


def _count_records(jsonl_path: Path) -> int:
    """Count conversation records in a JSONL file."""
    count = 0
    for line in jsonl_path.read_text().strip().split("\n"):
        if line and "conversation_id" in line:
            count += 1
    return count


def _build_results(
    report_a: dict,
    report_b: dict,
    out_a: Path,
    out_b: Path,
    args: argparse.Namespace,
) -> dict:
    """Build the comparison results dict."""

    def summarize(report: dict, path: Path, *, steering: bool) -> dict:
        div = report.get("diversity", {})
        avg = report.get("average_scores", {})
        vals = [
            avg.get("tool_correctness", 0),
            avg.get("argument_grounding", 0),
            avg.get("task_completion", 0),
            avg.get("naturalness", 0),
        ]
        return {
            "steering": steering,
            "count": _count_records(path),
            "tool_entropy": div.get("tool_entropy", 0),
            "unique_tools": div.get("unique_tools", 0),
            "unique_domains": div.get("unique_domains", 0),
            "unique_tool_combos": div.get("unique_tool_combos", 0),
            "mean_score": round(sum(vals) / 4, 2) if vals else 0,
            "scores": avg,
        }

    return {
        "experiment": {
            "seed": args.seed,
            "count": args.count,
            "min_steps": args.min_steps,
            "max_steps": args.max_steps,
            "similarity_threshold": args.similarity_threshold,
            "categories": args.categories,
        },
        "run_a": summarize(report_a, out_a, steering=False),
        "run_b": summarize(report_b, out_b, steering=True),
    }


def _print_comparison(results: dict, elapsed: float) -> None:
    """Print a formatted comparison table."""
    run_a = results["run_a"]
    run_b = results["run_b"]
    exp = results["experiment"]

    print(f"\n{'=' * 65}")
    print(f"DIVERSITY EXPERIMENT RESULTS (seed={exp['seed']}, count={exp['count']})")
    print(f"{'=' * 65}")
    print(f"{'Metric':<25} {'Run A (no steer)':>18} {'Run B (steered)':>18}")
    print(f"{'-' * 65}")
    print(f"{'Conversations':<25} {run_a['count']:>18} {run_b['count']:>18}")
    print(
        f"{'Tool entropy':<25} "
        f"{run_a['tool_entropy']:>18.4f} {run_b['tool_entropy']:>18.4f}"
    )
    print(
        f"{'Unique tools':<25} "
        f"{run_a['unique_tools']:>18} {run_b['unique_tools']:>18}"
    )
    print(
        f"{'Unique combos':<25} "
        f"{run_a['unique_tool_combos']:>18} {run_b['unique_tool_combos']:>18}"
    )
    print(
        f"{'Mean score':<25} "
        f"{run_a['mean_score']:>18.2f} {run_b['mean_score']:>18.2f}"
    )
    for dim in [
        "tool_correctness",
        "argument_grounding",
        "task_completion",
        "naturalness",
    ]:
        va = run_a["scores"].get(dim, 0)
        vb = run_b["scores"].get(dim, 0)
        print(f"{'  ' + dim:<25} {va:>18} {vb:>18}")
    print(f"{'-' * 65}")
    print(f"{'Time':<25} {elapsed:>18.1f}s {'':>18}")
    print(f"{'=' * 65}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the diversity experiment."""
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate input directory
    if not args.input_dir.is_dir():
        print(f"ERROR: Input directory not found: {args.input_dir}")
        sys.exit(1)

    # If categories specified, copy subset to output directory
    if args.categories:
        data_dir = _prepare_subset(
            args.input_dir, args.categories.split(","), output_dir
        )
    else:
        data_dir = args.input_dir

    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Build (shared by both runs)
    # ------------------------------------------------------------------
    build_dir = output_dir / "build"
    print(f"Building from {data_dir} ...")
    result = run_cli(
        "build",
        "--input-dir", str(data_dir),
        "--output-dir", str(build_dir),
        "--force",
        "--similarity-threshold", str(args.similarity_threshold),
    )
    if result.returncode != 0:
        print(f"Build failed:\n{result.stdout}\n{result.stderr}")
        sys.exit(1)
    print("  Build complete.")

    # ------------------------------------------------------------------
    # 2. Run A — steering disabled
    # ------------------------------------------------------------------
    out_a = output_dir / "run_a.jsonl"
    print(f"\nRun A: generating {args.count} conversations (steering OFF) ...")
    result = run_cli(
        "generate",
        "--output", str(out_a),
        "--build-dir", str(build_dir),
        "--count", str(args.count),
        "--seed", str(args.seed),
        "--min-steps", str(args.min_steps),
        "--max-steps", str(args.max_steps),
        "--no-cross-conversation-steering",
    )
    if result.returncode != 0:
        print(f"Run A failed:\n{result.stdout}\n{result.stderr}")
        sys.exit(1)
    print(f"  Run A complete: {out_a}")

    # ------------------------------------------------------------------
    # 3. Run B — steering enabled
    # ------------------------------------------------------------------
    out_b = output_dir / "run_b.jsonl"
    print(f"\nRun B: generating {args.count} conversations (steering ON) ...")
    result = run_cli(
        "generate",
        "--output", str(out_b),
        "--build-dir", str(build_dir),
        "--count", str(args.count),
        "--seed", str(args.seed),
        "--min-steps", str(args.min_steps),
        "--max-steps", str(args.max_steps),
    )
    if result.returncode != 0:
        print(f"Run B failed:\n{result.stdout}\n{result.stderr}")
        sys.exit(1)
    print(f"  Run B complete: {out_b}")

    # ------------------------------------------------------------------
    # 4. Evaluate both
    # ------------------------------------------------------------------
    print("\nEvaluating ...")
    report_a = _evaluate(out_a)
    report_b = _evaluate(out_b)

    # ------------------------------------------------------------------
    # 5. Compare and write results
    # ------------------------------------------------------------------
    results = _build_results(report_a, report_b, out_a, out_b, args)
    results_file = output_dir / "diversity_results.json"
    results_file.write_text(json.dumps(results, indent=2))

    elapsed = time.perf_counter() - t0
    _print_comparison(results, elapsed)

    print(f"\nResults written to: {results_file}")
    print(f"Run A output:       {out_a}")
    print(f"Run B output:       {out_b}")


if __name__ == "__main__":
    main()
