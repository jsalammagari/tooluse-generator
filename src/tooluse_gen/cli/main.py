"""CLI entry point for tooluse-generator."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tooluse_gen.utils.logging import get_logger, setup_logging
from tooluse_gen.utils.seeding import set_global_seed

logger = get_logger("cli")

console = Console()
err_console = Console(stderr=True, style="bold red")

app = typer.Typer(
    name="tooluse",
    help=(
        "[bold]tooluse-generator[/bold] — Offline synthetic data generation system "
        "that produces multi-turn tool-use conversations grounded in ToolBench schemas."
    ),
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)

# ---------------------------------------------------------------------------
# Global state (populated by callback, read by sub-commands)
# ---------------------------------------------------------------------------
_state: dict[str, object] = {
    "verbose": 0,
    "quiet": False,
    "config": Path("config/default.yaml"),
}


@app.callback()
def main(
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Enable verbose output. Repeat (-vv) for more verbosity.",
        ),
    ] = 0,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress non-essential output."),
    ] = False,
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to YAML configuration file.",
            exists=False,  # don't error if missing — commands will validate
        ),
    ] = Path("config/default.yaml"),
) -> None:
    """[bold]tooluse-generator[/bold]: build → generate → evaluate."""
    _state["verbose"] = verbose
    _state["quiet"] = quiet
    _state["config"] = config
    setup_logging(verbosity=verbose, quiet=quiet)
    logger.debug("CLI started: verbosity=%d quiet=%s config=%s", verbose, quiet, config)


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
@app.command()
def build(
    input_dir: Annotated[
        Path,
        typer.Option(
            "--input-dir",
            "-i",
            help="Path to directory containing raw ToolBench JSON files.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory to write built artifacts (graph, registry, indexes).",
        ),
    ] = Path("./output/build"),
    embedding_model: Annotated[
        str,
        typer.Option(
            "--embedding-model",
            help="Sentence-transformer model name for semantic edges.",
        ),
    ] = "all-MiniLM-L6-v2",
    similarity_threshold: Annotated[
        float,
        typer.Option(
            "--similarity-threshold",
            min=0.0,
            max=1.0,
            help="Cosine similarity threshold for adding semantic edges (0–1).",
        ),
    ] = 0.7,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite existing artifacts."),
    ] = False,
    generate_pools: Annotated[
        bool,
        typer.Option("--generate-pools", help="Generate value pools for mock responses."),
    ] = False,
) -> None:
    """Ingest ToolBench data, build tool registry and graph."""
    quiet = bool(_state["quiet"])

    if not quiet:
        console.print(
            Panel.fit(
                f"[bold cyan]build[/bold cyan]\n"
                f"  input-dir          : {input_dir}\n"
                f"  output-dir         : {output_dir}\n"
                f"  embedding-model    : {embedding_model}\n"
                f"  similarity-threshold: {similarity_threshold}\n"
                f"  force              : {force}\n"
                f"  generate-pools     : {generate_pools}\n"
                f"  config             : {_state['config']}",
                title="tooluse build",
            )
        )

    # Late imports to keep CLI startup fast
    from tooluse_gen.agents.value_generator import ValuePool
    from tooluse_gen.graph.builder import GraphBuilder
    from tooluse_gen.graph.embeddings import EmbeddingService
    from tooluse_gen.graph.models import GraphConfig
    from tooluse_gen.graph.persistence import save_embeddings, save_graph
    from tooluse_gen.graph.queries import get_graph_stats
    from tooluse_gen.registry.completeness import QualityTier, generate_quality_report
    from tooluse_gen.registry.registry import RegistryBuilder
    from tooluse_gen.registry.serialization import save_registry

    t0 = time.perf_counter()

    try:
        # ----------------------------------------------------------
        # Step 1: Validate inputs
        # ----------------------------------------------------------
        if not input_dir.is_dir():
            err_console.print(f"Input directory does not exist: {input_dir}")
            raise typer.Exit(code=1)

        _artifact_names = ["registry.json", "graph.pkl", "embeddings.joblib"]
        if output_dir.exists() and not force:
            existing = [n for n in _artifact_names if (output_dir / n).exists()]
            if existing:
                err_console.print(
                    f"Output directory already contains artifacts: {', '.join(existing)}. "
                    f"Use --force to overwrite."
                )
                raise typer.Exit(code=1)

        output_dir.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------------
        # Step 2: Load ToolBench data
        # ----------------------------------------------------------
        registry = (
            RegistryBuilder()
            .load_from_directory(input_dir)
            .calculate_completeness()
            .filter_by_quality(QualityTier.FAIR)
            .build()
        )

        total_endpoints = sum(len(t.endpoints) for t in registry.tools())
        if not quiet:
            console.print(
                f"  [bold green]✓[/bold green] Loaded [bold]{len(registry)}[/bold] tools "
                f"([bold]{total_endpoints}[/bold] endpoints) from {input_dir}"
            )

        if len(registry) == 0:
            err_console.print("No tools passed quality filter. Aborting.")
            raise typer.Exit(code=1)

        # ----------------------------------------------------------
        # Step 3: Save registry
        # ----------------------------------------------------------
        registry_path = output_dir / "registry.json"
        save_registry(registry, registry_path)
        if not quiet:
            console.print(
                f"  [bold green]✓[/bold green] Saved registry to {registry_path}"
            )

        # ----------------------------------------------------------
        # Step 4: Build embeddings
        # ----------------------------------------------------------
        include_semantic = True
        try:
            embedding_service = EmbeddingService(model_name=embedding_model)
        except Exception:
            logger.warning(
                "Failed to initialise EmbeddingService (%s) — "
                "semantic edges will be skipped.",
                embedding_model,
                exc_info=True,
            )
            embedding_service = EmbeddingService.__new__(EmbeddingService)
            embedding_service._model = None  # type: ignore[attr-defined]
            embedding_service._cache_dir = None  # type: ignore[attr-defined]
            include_semantic = False

        # ----------------------------------------------------------
        # Step 5: Build tool graph
        # ----------------------------------------------------------
        graph_config = GraphConfig(
            similarity_threshold=similarity_threshold,
            embedding_model=embedding_model,
            include_tool_nodes=True,
            include_domain_edges=True,
            include_semantic_edges=include_semantic,
        )

        graph_builder = GraphBuilder(config=graph_config, embedding_service=embedding_service)
        graph = graph_builder.build(registry)
        stats = get_graph_stats(graph)

        if not quiet:
            console.print(
                f"  [bold green]✓[/bold green] Built graph: "
                f"[bold]{stats.tool_node_count}[/bold] tool nodes, "
                f"[bold]{stats.endpoint_node_count}[/bold] endpoint nodes, "
                f"[bold]{stats.total_edges}[/bold] edges"
            )

        # Save embeddings from graph node data
        embeddings: dict[str, list[float]] = {}
        for nid, ndata in graph.nodes(data=True):
            emb = ndata.get("embedding")
            if emb is not None:
                embeddings[nid] = emb

        embeddings_path = output_dir / "embeddings.joblib"
        save_embeddings(embeddings, embeddings_path)
        if not quiet:
            console.print(
                f"  [bold green]✓[/bold green] Saved [bold]{len(embeddings)}[/bold] "
                f"embeddings to {embeddings_path}"
            )

        # Save graph
        graph_path = output_dir / "graph.pkl"
        save_graph(graph, graph_path)
        if not quiet:
            console.print(f"  [bold green]✓[/bold green] Saved graph to {graph_path}")

        # ----------------------------------------------------------
        # Step 6: Optionally generate value pools
        # ----------------------------------------------------------
        if generate_pools:
            pool = ValuePool()
            pools_path = output_dir / "value_pools.json"
            pool.save(pools_path)
            if not quiet:
                console.print(
                    f"  [bold green]✓[/bold green] Saved value pools to {pools_path}"
                )

        # ----------------------------------------------------------
        # Step 7: Print summary
        # ----------------------------------------------------------
        elapsed = time.perf_counter() - t0

        if not quiet:
            console.print()

            # Quality report
            quality = generate_quality_report(list(registry.tools()))
            tier_dist = quality.get("tier_distribution", {})

            summary = Table(title="Build Summary", show_header=False, padding=(0, 2))
            summary.add_column("key", style="bold")
            summary.add_column("value")
            summary.add_row("Tools", str(len(registry)))
            summary.add_row("Endpoints", str(total_endpoints))
            summary.add_row(
                "Quality",
                ", ".join(f"{tier}: {cnt}" for tier, cnt in sorted(tier_dist.items())),  # type: ignore[union-attr]
            )
            summary.add_row("Graph nodes", str(stats.total_nodes))
            summary.add_row("Graph edges", str(stats.total_edges))
            summary.add_row("Graph density", f"{stats.density:.4f}")
            summary.add_row("Embeddings", str(len(embeddings)))

            # Artifact sizes
            artifacts = ["registry.json", "graph.pkl", "embeddings.joblib"]
            if generate_pools:
                artifacts.append("value_pools.json")
            sizes = []
            for name in artifacts:
                p = output_dir / name
                if p.exists():
                    sz = p.stat().st_size
                    if sz >= 1_048_576:
                        sizes.append(f"{name} ({sz / 1_048_576:.1f} MB)")
                    elif sz >= 1024:
                        sizes.append(f"{name} ({sz / 1024:.1f} KB)")
                    else:
                        sizes.append(f"{name} ({sz} B)")
            summary.add_row("Artifacts", ", ".join(sizes))
            summary.add_row("Time", f"{elapsed:.1f}s")

            console.print(summary)

        logger.info("Build completed in %.1fs", elapsed)

    except typer.Exit:
        raise
    except Exception as exc:
        logger.exception("Build failed: %s", exc)
        if not quiet:
            console.print(
                Panel(
                    f"[bold red]Build failed[/bold red]\n\n{exc}",
                    title="Error",
                    border_style="red",
                )
            )
        raise typer.Exit(code=1) from exc


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------
@app.command()
def generate(
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output JSONL file path for generated conversations.",
        ),
    ],
    build_dir: Annotated[
        Path,
        typer.Option(
            "--build-dir",
            "-b",
            help="Path to pre-built artifacts directory (from `tooluse build`).",
        ),
    ] = Path("./output/build"),
    count: Annotated[
        int,
        typer.Option("--count", "-n", min=1, help="Number of conversations to generate."),
    ] = 100,
    seed: Annotated[
        int,
        typer.Option("--seed", "-s", help="Random seed for reproducibility."),
    ] = 42,
    min_steps: Annotated[
        int,
        typer.Option("--min-steps", min=1, help="Minimum tool calls per conversation."),
    ] = 2,
    max_steps: Annotated[
        int,
        typer.Option("--max-steps", min=1, help="Maximum tool calls per conversation."),
    ] = 5,
    domains: Annotated[
        str | None,
        typer.Option(
            "--domains",
            help="Comma-separated list of ToolBench domains to include (e.g. Travel,Finance).",
        ),
    ] = None,
    no_cross_conversation_steering: Annotated[
        bool,
        typer.Option(
            "--no-cross-conversation-steering",
            help="Disable cross-conversation diversity steering (Run A mode).",
        ),
    ] = False,
    max_retries: Annotated[
        int,
        typer.Option("--max-retries", min=0, help="Max repair attempts per conversation."),
    ] = 3,
    quality_threshold: Annotated[
        float,
        typer.Option(
            "--quality-threshold",
            min=0.0,
            max=5.0,
            help="Minimum average LLM-as-judge score to accept a conversation (0–5).",
        ),
    ] = 3.5,
) -> None:
    """Generate synthetic conversations with tool use."""
    if min_steps > max_steps:
        err_console.print("--min-steps must be <= --max-steps")
        raise typer.Exit(code=1)

    set_global_seed(seed)
    logger.info("Initialized with seed: %d", seed)

    steering = not no_cross_conversation_steering
    domain_list = [d.strip() for d in domains.split(",")] if domains else []

    quiet = bool(_state["quiet"])

    if not quiet:
        table = Table(title="tooluse generate", show_header=False, box=None, padding=(0, 2))
        rows = [
            ("build-dir", str(build_dir)),
            ("output", str(output)),
            ("count", str(count)),
            ("seed", str(seed)),
            ("steps", f"{min_steps}–{max_steps}"),
            ("domains", ", ".join(domain_list) if domain_list else "(all)"),
            ("steering", str(steering)),
            ("max-retries", str(max_retries)),
            ("quality-threshold", str(quality_threshold)),
            ("config", str(_state["config"])),
        ]
        for k, v in rows:
            table.add_row(f"[bold]{k}[/bold]", v)
        console.print(table)

    # Late imports to keep CLI startup fast
    from tooluse_gen.agents.assistant_agent import AssistantAgent
    from tooluse_gen.agents.batch_generator import BatchGenerator
    from tooluse_gen.agents.orchestrator import ConversationOrchestrator
    from tooluse_gen.agents.tool_executor import ToolExecutor
    from tooluse_gen.agents.user_simulator import UserSimulator
    from tooluse_gen.core.config import load_config
    from tooluse_gen.core.jsonl_io import JSONLWriter
    from tooluse_gen.core.output_models import from_conversation
    from tooluse_gen.core.reproducibility import embed_config_in_output, serialize_run_config
    from tooluse_gen.evaluation.judge import JudgeAgent
    from tooluse_gen.evaluation.models import EvaluationConfig
    from tooluse_gen.evaluation.pipeline import EvaluationPipeline
    from tooluse_gen.evaluation.repair import RepairLoop
    from tooluse_gen.evaluation.validator import ConversationValidator
    from tooluse_gen.graph.chain_models import SamplingConstraints
    from tooluse_gen.graph.diversity import DiversitySteeringConfig
    from tooluse_gen.graph.facade import ToolChainSampler
    from tooluse_gen.graph.persistence import load_graph
    from tooluse_gen.registry.serialization import load_registry

    t0 = time.perf_counter()

    try:
        # ----------------------------------------------------------
        # Step 1: Validate inputs
        # ----------------------------------------------------------
        registry_path = build_dir / "registry.json"
        graph_path = build_dir / "graph.pkl"

        if not build_dir.is_dir():
            err_console.print(f"Build directory does not exist: {build_dir}")
            raise typer.Exit(code=1)
        if not registry_path.exists():
            err_console.print(f"Missing registry: {registry_path}")
            raise typer.Exit(code=1)
        if not graph_path.exists():
            err_console.print(f"Missing graph: {graph_path}")
            raise typer.Exit(code=1)

        # ----------------------------------------------------------
        # Step 2: Load build artifacts
        # ----------------------------------------------------------
        registry, _ = load_registry(registry_path)
        graph, _ = load_graph(graph_path)

        total_endpoints = sum(len(t.endpoints) for t in registry.tools())
        if not quiet:
            console.print(
                f"  [bold green]✓[/bold green] Loaded registry: "
                f"[bold]{len(registry)}[/bold] tools "
                f"([bold]{total_endpoints}[/bold] endpoints) from {build_dir}"
            )
            console.print(
                f"  [bold green]✓[/bold green] Loaded graph: "
                f"[bold]{graph.number_of_nodes()}[/bold] nodes, "
                f"[bold]{graph.number_of_edges()}[/bold] edges"
            )

        # ----------------------------------------------------------
        # Step 3: Load config
        # ----------------------------------------------------------
        config_path = Path(str(_state["config"]))
        app_config = load_config(str(config_path) if config_path.exists() else None)

        # ----------------------------------------------------------
        # Step 4: Initialize agents (offline mode, no LLM)
        # ----------------------------------------------------------
        user_sim = UserSimulator()
        assistant = AssistantAgent(registry=registry)
        executor = ToolExecutor(registry=registry)
        orchestrator = ConversationOrchestrator(
            user_sim=user_sim, assistant=assistant, executor=executor,
        )

        # ----------------------------------------------------------
        # Step 5: Initialize sampler with diversity config
        # ----------------------------------------------------------
        diversity_config = DiversitySteeringConfig(enabled=steering)
        sampler = ToolChainSampler(graph, diversity_config=diversity_config)

        # ----------------------------------------------------------
        # Step 6: Generate conversations
        # ----------------------------------------------------------
        constraints = SamplingConstraints(
            min_steps=min_steps,
            max_steps=max_steps,
            min_tools=1,
            domains=domain_list if domain_list else None,
        )
        batch_gen = BatchGenerator(
            orchestrator=orchestrator, sampler=sampler, diversity_config=diversity_config,
        )
        conversations = batch_gen.generate_batch(
            count=count, constraints=constraints, seed=seed, steering_enabled=steering,
        )
        batch_stats = batch_gen.get_batch_stats()

        if not quiet:
            console.print(
                f"  [bold green]✓[/bold green] Generated "
                f"[bold]{batch_stats.total_generated}[/bold] conversations "
                f"(seed={seed}, steering={'enabled' if steering else 'disabled'})"
            )

        # ----------------------------------------------------------
        # Step 7: Evaluate conversations
        # ----------------------------------------------------------
        validator = ConversationValidator(registry=registry)
        judge = JudgeAgent()  # offline heuristic mode
        eval_config = EvaluationConfig(
            min_score=quality_threshold, max_retries=max_retries,
        )
        repair = RepairLoop(
            orchestrator=orchestrator, validator=validator, judge=judge, config=eval_config,
        )
        pipeline = EvaluationPipeline(
            validator=validator, judge=judge, repair_loop=repair, config=eval_config,
        )
        eval_report = pipeline.evaluate_batch(conversations, chains=None, seed=seed)

        if not quiet:
            console.print(
                f"  [bold green]✓[/bold green] Evaluated: "
                f"[bold]{eval_report.passed}[/bold] passed, "
                f"[bold]{eval_report.failed}[/bold] failed "
                f"(pass rate: {eval_report.pass_rate:.1%})"
            )

        # ----------------------------------------------------------
        # Step 8: Build output records and write JSONL
        # ----------------------------------------------------------
        cli_args = {
            "count": count, "seed": seed, "min_steps": min_steps,
            "max_steps": max_steps, "domains": domain_list,
            "steering": steering, "max_retries": max_retries,
            "quality_threshold": quality_threshold,
        }
        run_config = serialize_run_config(app_config, seed, cli_args=cli_args)

        records = []
        eval_results = pipeline.get_results()
        for i, conv in enumerate(conversations):
            scores = eval_results[i].scores if i < len(eval_results) else None
            rec = from_conversation(conv, eval_scores=scores)
            records.append(rec)

        records = embed_config_in_output(records, run_config)

        output.parent.mkdir(parents=True, exist_ok=True)
        writer = JSONLWriter(output)
        writer.write_header(run_config)
        writer.write_batch(records)

        if not quiet:
            console.print(
                f"  [bold green]✓[/bold green] Wrote "
                f"[bold]{writer.count}[/bold] records to {output}"
            )

        # ----------------------------------------------------------
        # Step 9: Print summary
        # ----------------------------------------------------------
        elapsed = time.perf_counter() - t0

        if not quiet:
            console.print()

            summary = Table(title="Generation Summary", show_header=False, padding=(0, 2))
            summary.add_column("key", style="bold")
            summary.add_column("value")
            summary.add_row("Conversations", str(batch_stats.total_generated))
            summary.add_row("Failed", str(batch_stats.total_failed))
            summary.add_row("Passed eval", str(eval_report.passed))
            summary.add_row("Failed eval", str(eval_report.failed))
            summary.add_row("Pass rate", f"{eval_report.pass_rate:.1%}")

            if eval_report.average_scores is not None:
                avg = eval_report.average_scores
                summary.add_row(
                    "Avg scores",
                    f"tc={avg.tool_correctness} ag={avg.argument_grounding} "
                    f"comp={avg.task_completion} nat={avg.naturalness} "
                    f"(mean={avg.average:.2f})",
                )

            if batch_stats.diversity_metrics is not None:
                dm = batch_stats.diversity_metrics
                summary.add_row(
                    "Diversity",
                    f"entropy={dm.tool_entropy:.2f} "
                    f"coverage={dm.domain_coverage:.2f} "
                    f"pair_ratio={dm.unique_tool_pair_ratio:.2f} "
                    f"repetition={dm.pattern_repetition_rate:.2f}",
                )

            summary.add_row("Avg turns", f"{batch_stats.average_turns:.1f}")
            summary.add_row("Avg tool calls", f"{batch_stats.average_tool_calls:.1f}")
            summary.add_row("Steering", "enabled" if steering else "disabled")
            summary.add_row("Output", str(output))
            summary.add_row("Time", f"{elapsed:.1f}s")

            console.print(summary)

        logger.info("Generate completed in %.1fs", elapsed)

    except typer.Exit:
        raise
    except Exception as exc:
        logger.exception("Generate failed: %s", exc)
        if not quiet:
            console.print(
                Panel(
                    f"[bold red]Generate failed[/bold red]\n\n{exc}",
                    title="Error",
                    border_style="red",
                )
            )
        raise typer.Exit(code=1) from exc


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------
@app.command()
def evaluate(
    input_path: Annotated[
        Path,
        typer.Argument(
            help="Path to JSONL file of generated conversations to evaluate.",
            metavar="INPUT",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Write evaluation report to this file (default: stdout).",
        ),
    ] = None,
    fmt: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format: json | table | markdown.",
        ),
    ] = "table",
    rescore: Annotated[
        bool,
        typer.Option("--rescore", help="Re-run the LLM judge on all conversations."),
    ] = False,
) -> None:
    """Evaluate generated conversations and compute metrics."""
    valid_formats = {"json", "table", "markdown"}
    if fmt not in valid_formats:
        err_console.print(f"--format must be one of: {', '.join(sorted(valid_formats))}")
        raise typer.Exit(code=1)

    if not _state["quiet"]:
        console.print(
            Panel.fit(
                f"[bold cyan]evaluate[/bold cyan]\n"
                f"  input   : {input_path}\n"
                f"  output  : {output or '(stdout)'}\n"
                f"  format  : {fmt}\n"
                f"  rescore : {rescore}\n"
                f"  config  : {_state['config']}",
                title="tooluse evaluate",
            )
        )
    typer.echo("Not implemented yet")


if __name__ == "__main__":
    app()
