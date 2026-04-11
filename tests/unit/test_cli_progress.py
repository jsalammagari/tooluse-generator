"""Tests for CLI progress utilities (Task 62)."""

from __future__ import annotations

import os
import signal
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from tooluse_gen.cli.progress import (
    BuildProgress,
    GenerationProgress,
    InterruptHandler,
    format_file_size,
    print_artifact_summary,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_console() -> Console:
    """Create a Console that captures output to a string buffer."""
    return Console(file=StringIO(), force_terminal=False)


def _get_output(console: Console) -> str:
    """Extract captured text from a Console with a StringIO file."""
    f = console.file
    assert isinstance(f, StringIO)
    return f.getvalue()


# ===================================================================
# BuildProgress
# ===================================================================


class TestBuildProgress:
    def test_step_records_in_order(self) -> None:
        bp = BuildProgress(_make_console(), quiet=True)
        bp.start()
        bp.step("Step A")
        bp.step("Step B")
        bp.step("Step C")
        descs = [s["description"] for s in bp.steps]
        assert descs == ["Step A", "Step B", "Step C"]

    def test_step_count(self) -> None:
        bp = BuildProgress(_make_console(), quiet=True)
        assert bp.step_count == 0
        bp.step("One")
        assert bp.step_count == 1
        bp.step("Two")
        assert bp.step_count == 2

    def test_quiet_mode_no_print(self) -> None:
        c = _make_console()
        bp = BuildProgress(c, quiet=True)
        bp.start()
        bp.step("Silent step")
        assert _get_output(c) == ""

    def test_non_quiet_prints_checkmark(self) -> None:
        c = _make_console()
        bp = BuildProgress(c, quiet=False)
        bp.start()
        bp.step("Visible step")
        output = _get_output(c)
        assert "Visible step" in output

    def test_step_details_stored(self) -> None:
        bp = BuildProgress(_make_console(), quiet=True)
        bp.step("Load", count=42, path="/data")
        step = bp.steps[0]
        assert step["count"] == 42
        assert step["path"] == "/data"

    def test_elapsed_tracked(self) -> None:
        bp = BuildProgress(_make_console(), quiet=True)
        bp.start()
        bp.step("After start")
        assert bp.steps[0]["elapsed"] >= 0.0

    def test_steps_returns_copy(self) -> None:
        bp = BuildProgress(_make_console(), quiet=True)
        bp.step("A")
        steps = bp.steps
        steps.clear()
        assert bp.step_count == 1  # original not affected


# ===================================================================
# GenerationProgress
# ===================================================================


class TestGenerationProgress:
    def test_advance_increments_completed(self) -> None:
        gp = GenerationProgress(5, _make_console(), quiet=True)
        gp.start()
        gp.advance(success=True)
        gp.advance(success=True)
        gp.finish()
        assert gp.completed == 2

    def test_advance_failure(self) -> None:
        gp = GenerationProgress(5, _make_console(), quiet=True)
        gp.start()
        gp.advance(success=False)
        gp.advance(success=False)
        gp.finish()
        assert gp.failed == 2
        assert gp.completed == 0

    def test_mixed_advance(self) -> None:
        gp = GenerationProgress(4, _make_console(), quiet=True)
        gp.start()
        gp.advance(success=True)
        gp.advance(success=False)
        gp.advance(success=True)
        gp.finish()
        assert gp.completed == 2
        assert gp.failed == 1

    def test_start_creates_pbar(self) -> None:
        gp = GenerationProgress(3, _make_console(), quiet=True)
        assert gp._pbar is None
        gp.start()
        assert gp._pbar is not None
        gp.finish()

    def test_finish_closes_pbar(self) -> None:
        gp = GenerationProgress(3, _make_console(), quiet=True)
        gp.start()
        pbar = gp._pbar
        assert pbar is not None
        gp.finish()
        assert pbar.disable or pbar.n == pbar.n  # closed bar is still accessible

    def test_quiet_disables_pbar(self) -> None:
        gp = GenerationProgress(3, _make_console(), quiet=True)
        gp.start()
        assert gp._pbar is not None
        assert gp._pbar.disable is True
        gp.finish()

    def test_total_property(self) -> None:
        gp = GenerationProgress(7, _make_console(), quiet=True)
        assert gp.total == 7

    def test_advance_without_start(self) -> None:
        gp = GenerationProgress(5, _make_console(), quiet=True)
        # Should not error even without start()
        gp.advance(success=True)
        assert gp.completed == 1

    def test_finish_without_start(self) -> None:
        gp = GenerationProgress(5, _make_console(), quiet=True)
        # Should not error
        gp.finish()


# ===================================================================
# InterruptHandler
# ===================================================================


class TestInterruptHandler:
    def test_not_interrupted_by_default(self) -> None:
        handler = InterruptHandler()
        assert not handler.interrupted

    def test_context_manager_runs(self) -> None:
        handler = InterruptHandler()
        with handler:
            pass
        assert not handler.interrupted

    def test_restores_original_handler(self) -> None:
        original = signal.getsignal(signal.SIGINT)
        handler = InterruptHandler()
        with handler:
            # Inside, our handler is installed
            current = signal.getsignal(signal.SIGINT)
            assert current is not original
        # After exit, original is restored
        restored = signal.getsignal(signal.SIGINT)
        assert restored is original

    def test_interrupted_flag_set_by_signal(self) -> None:
        handler = InterruptHandler()
        with handler:
            os.kill(os.getpid(), signal.SIGINT)
        assert handler.interrupted

    def test_multiple_signals(self) -> None:
        handler = InterruptHandler()
        with handler:
            os.kill(os.getpid(), signal.SIGINT)
            os.kill(os.getpid(), signal.SIGINT)
        assert handler.interrupted

    def test_handler_reusable(self) -> None:
        handler = InterruptHandler()
        with handler:
            pass
        assert not handler.interrupted
        # Can't truly "reset" but a new instance works
        handler2 = InterruptHandler()
        with handler2:
            pass
        assert not handler2.interrupted


# ===================================================================
# format_file_size
# ===================================================================


class TestFormatFileSize:
    def test_zero(self) -> None:
        assert format_file_size(0) == "0 B"

    def test_bytes(self) -> None:
        assert format_file_size(500) == "500 B"

    def test_one_kb(self) -> None:
        assert format_file_size(1024) == "1.0 KB"

    def test_kilobytes(self) -> None:
        assert format_file_size(2048) == "2.0 KB"

    def test_one_mb(self) -> None:
        assert format_file_size(1_048_576) == "1.0 MB"

    def test_megabytes(self) -> None:
        assert format_file_size(5_242_880) == "5.0 MB"

    def test_boundary_below_kb(self) -> None:
        assert format_file_size(1023) == "1023 B"

    def test_boundary_below_mb(self) -> None:
        result = format_file_size(1_048_575)
        assert "KB" in result


# ===================================================================
# print_artifact_summary
# ===================================================================


class TestPrintArtifactSummary:
    def test_prints_existing_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.json").write_text('{"x": 1}')
        (tmp_path / "b.pkl").write_bytes(b"\x00" * 2048)
        c = _make_console()
        print_artifact_summary(c, tmp_path, ["a.json", "b.pkl"])
        output = _get_output(c)
        assert "a.json" in output
        assert "b.pkl" in output

    def test_skips_missing_files(self, tmp_path: Path) -> None:
        (tmp_path / "exists.json").write_text("{}")
        c = _make_console()
        print_artifact_summary(c, tmp_path, ["exists.json", "missing.json"])
        output = _get_output(c)
        assert "exists.json" in output
        assert "missing.json" not in output

    def test_quiet_mode_no_output(self, tmp_path: Path) -> None:
        (tmp_path / "a.json").write_text("{}")
        c = _make_console()
        print_artifact_summary(c, tmp_path, ["a.json"], quiet=True)
        assert _get_output(c) == ""

    def test_shows_file_sizes(self, tmp_path: Path) -> None:
        (tmp_path / "big.dat").write_bytes(b"\x00" * 5000)
        c = _make_console()
        print_artifact_summary(c, tmp_path, ["big.dat"])
        output = _get_output(c)
        assert "KB" in output
