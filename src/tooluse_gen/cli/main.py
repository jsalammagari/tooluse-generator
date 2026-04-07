"""CLI entry point for tooluse-generator."""

from pathlib import Path

import typer

app = typer.Typer(
    name="tooluse",
    help="Multi-Agent Tool-Use Conversation Generator",
    add_completion=False,
)


@app.command()
def build(
    data_dir: Path = typer.Option(
        Path("data/toolbench"),
        "--data-dir",
        "-d",
        help="Directory containing raw ToolBench JSON files.",
    ),
    output_dir: Path = typer.Option(
        Path("output/artifacts"),
        "--output-dir",
        "-o",
        help="Directory to write built artifacts (graph, registry, indexes).",
    ),
    config: Path = typer.Option(
        Path("config/default.yaml"),
        "--config",
        "-c",
        help="Path to configuration YAML file.",
    ),
) -> None:
    """Ingest ToolBench data and build all derived artifacts (graph, indexes, etc.)."""
    typer.echo(f"Building artifacts from {data_dir} → {output_dir}")
    typer.echo("(build command not yet implemented)")


@app.command()
def generate(
    artifacts_dir: Path = typer.Option(
        Path("output/artifacts"),
        "--artifacts-dir",
        "-a",
        help="Directory containing built artifacts.",
    ),
    output_file: Path = typer.Option(
        Path("output/conversations.jsonl"),
        "--output",
        "-o",
        help="Path to write generated conversations (JSONL).",
    ),
    num_conversations: int = typer.Option(
        100,
        "--num",
        "-n",
        help="Number of conversations to generate.",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Random seed for reproducibility.",
    ),
    no_cross_conversation_steering: bool = typer.Option(
        False,
        "--no-cross-conversation-steering",
        help="Disable cross-conversation diversity steering (Run A mode).",
    ),
    config: Path = typer.Option(
        Path("config/default.yaml"),
        "--config",
        "-c",
        help="Path to configuration YAML file.",
    ),
) -> None:
    """Generate multi-turn tool-use conversations."""
    steering = not no_cross_conversation_steering
    typer.echo(f"Generating {num_conversations} conversations (seed={seed}, steering={steering})")
    typer.echo(f"Output → {output_file}")
    typer.echo("(generate command not yet implemented)")


@app.command()
def evaluate(
    input_file: Path = typer.Option(
        Path("output/conversations.jsonl"),
        "--input",
        "-i",
        help="JSONL file of generated conversations to evaluate.",
    ),
    output_file: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional path to write evaluation results (defaults to stdout).",
    ),
    config: Path = typer.Option(
        Path("config/default.yaml"),
        "--config",
        "-c",
        help="Path to configuration YAML file.",
    ),
) -> None:
    """Validate generated conversations and compute evaluation metrics."""
    typer.echo(f"Evaluating conversations from {input_file}")
    typer.echo("(evaluate command not yet implemented)")


if __name__ == "__main__":
    app()
