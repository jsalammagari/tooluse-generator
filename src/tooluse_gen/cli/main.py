"""CLI entry point for tooluse-generator."""

from __future__ import annotations

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
) -> None:
    """Ingest ToolBench data, build tool registry and graph."""
    if not _state["quiet"]:
        console.print(
            Panel.fit(
                f"[bold cyan]build[/bold cyan]\n"
                f"  input-dir          : {input_dir}\n"
                f"  output-dir         : {output_dir}\n"
                f"  embedding-model    : {embedding_model}\n"
                f"  similarity-threshold: {similarity_threshold}\n"
                f"  force              : {force}\n"
                f"  config             : {_state['config']}",
                title="tooluse build",
            )
        )
    typer.echo("Not implemented yet")


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

    if not _state["quiet"]:
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

    typer.echo("Not implemented yet")


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
