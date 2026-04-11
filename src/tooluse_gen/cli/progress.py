"""CLI progress reporting and interrupt handling.

:class:`BuildProgress` tracks build pipeline steps with timing.
:class:`GenerationProgress` wraps tqdm for batch generation.
:class:`InterruptHandler` catches SIGINT so partial output can be saved.
"""

from __future__ import annotations

import signal
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from tqdm import tqdm

from tooluse_gen.utils.logging import get_logger
from tooluse_gen.utils.progress import create_progress_bar

logger = get_logger("cli.progress")


# ---------------------------------------------------------------------------
# BuildProgress
# ---------------------------------------------------------------------------


class BuildProgress:
    """Tracks build pipeline steps with timing and artifact reporting."""

    def __init__(self, console: Console, quiet: bool = False) -> None:
        self._console = console
        self._quiet = quiet
        self._steps: list[dict[str, Any]] = []
        self._start_time: float = 0.0

    def start(self) -> None:
        """Record the pipeline start time."""
        self._start_time = time.perf_counter()

    def step(self, description: str, **details: Any) -> None:
        """Record a completed step and print a checkmark (unless quiet)."""
        elapsed = time.perf_counter() - self._start_time if self._start_time else 0.0
        self._steps.append({"description": description, "elapsed": elapsed, **details})
        if not self._quiet:
            self._console.print(f"  [bold green]✓[/bold green] {description}")

    @property
    def steps(self) -> list[dict[str, Any]]:
        """Return a copy of recorded steps."""
        return list(self._steps)

    @property
    def step_count(self) -> int:
        """Number of completed steps."""
        return len(self._steps)


# ---------------------------------------------------------------------------
# GenerationProgress
# ---------------------------------------------------------------------------


class GenerationProgress:
    """Progress bar for batch conversation generation."""

    def __init__(
        self,
        total: int,
        console: Console,
        quiet: bool = False,
    ) -> None:
        self._total = total
        self._console = console
        self._quiet = quiet
        self._completed = 0
        self._failed = 0
        self._pbar: tqdm | None = None  # type: ignore[type-arg]

    def start(self) -> None:
        """Open the progress bar."""
        self._pbar = create_progress_bar(
            total=self._total,
            description="Generating",
            disable=self._quiet,
            unit="conv",
            colour="cyan",
        )

    def advance(self, success: bool = True) -> None:
        """Record one conversation attempt and tick the bar."""
        if success:
            self._completed += 1
        else:
            self._failed += 1
        if self._pbar is not None:
            self._pbar.update(1)

    def finish(self) -> None:
        """Close the progress bar."""
        if self._pbar is not None:
            self._pbar.close()

    @property
    def completed(self) -> int:
        return self._completed

    @property
    def failed(self) -> int:
        return self._failed

    @property
    def total(self) -> int:
        return self._total


# ---------------------------------------------------------------------------
# InterruptHandler
# ---------------------------------------------------------------------------


class InterruptHandler:
    """Context manager that catches SIGINT and sets a flag instead of raising.

    Usage::

        handler = InterruptHandler()
        with handler:
            for i in range(100):
                if handler.interrupted:
                    break
                do_work(i)
        if handler.interrupted:
            save_partial_output()
    """

    def __init__(self) -> None:
        self._interrupted = False
        self._original_handler: Any = None

    @property
    def interrupted(self) -> bool:
        """Whether an interrupt was received."""
        return self._interrupted

    def __enter__(self) -> InterruptHandler:
        self._original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle)
        return self

    def __exit__(self, *args: object) -> None:
        signal.signal(signal.SIGINT, self._original_handler)

    def _handle(self, signum: int, frame: Any) -> None:
        self._interrupted = True
        logger.warning(
            "Interrupt received — finishing current conversation and saving partial output"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def format_file_size(size_bytes: int) -> str:
    """Format bytes as human-readable file size."""
    if size_bytes >= 1_048_576:
        return f"{size_bytes / 1_048_576:.1f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def print_artifact_summary(
    console: Console,
    output_dir: Path,
    artifact_names: list[str],
    quiet: bool = False,
) -> None:
    """Print a summary of generated artifacts with file sizes."""
    if quiet:
        return
    for name in artifact_names:
        p = output_dir / name
        if p.exists():
            size = format_file_size(p.stat().st_size)
            console.print(f"    {name}: {size}")
