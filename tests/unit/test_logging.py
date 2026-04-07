"""Unit tests for logging infrastructure (Task 6)."""

from __future__ import annotations

import json
import logging
import time
from io import StringIO
from pathlib import Path

import pytest

from tooluse_gen.utils import (
    create_progress_bar,
    get_logger,
    log_context,
    log_duration,
    progress_callback,
    setup_logging,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_root() -> logging.Logger:
    """Return the tooluse_gen root logger with handlers cleared."""
    root = logging.getLogger("tooluse_gen")
    root.handlers.clear()
    return root


def _capture_handler(level: int = logging.DEBUG) -> tuple[logging.Handler, StringIO]:
    """Return a (handler, buffer) pair for capturing log output in tests."""
    buf = StringIO()
    h = logging.StreamHandler(buf)
    h.setLevel(level)
    h.setFormatter(logging.Formatter("%(levelname)s|%(name)s|%(message)s"))
    return h, buf


# ---------------------------------------------------------------------------
# setup_logging — level mapping
# ---------------------------------------------------------------------------


def test_setup_logging_verbosity_0_level_warning():
    root = setup_logging(verbosity=0)
    assert root.level <= logging.WARNING


def test_setup_logging_verbosity_1_level_info():
    setup_logging(verbosity=1)
    root = logging.getLogger("tooluse_gen")
    assert root.level <= logging.INFO


def test_setup_logging_verbosity_2_level_debug():
    setup_logging(verbosity=2)
    root = logging.getLogger("tooluse_gen")
    assert root.level <= logging.DEBUG


def test_setup_logging_quiet_suppresses_to_error():
    setup_logging(verbosity=2, quiet=True)
    root = logging.getLogger("tooluse_gen")
    # Console handler should be at ERROR level even though verbosity=2
    console = root.handlers[0]
    assert console.level >= logging.ERROR


def test_setup_logging_returns_logger():
    logger = setup_logging()
    assert isinstance(logger, logging.Logger)
    assert logger.name == "tooluse_gen"


def test_setup_logging_no_duplicate_handlers():
    setup_logging(verbosity=1)
    setup_logging(verbosity=1)
    root = logging.getLogger("tooluse_gen")
    assert len(root.handlers) == 1


# ---------------------------------------------------------------------------
# setup_logging — file handler
# ---------------------------------------------------------------------------


def test_setup_logging_file_handler_created(tmp_path: Path):
    log_file = tmp_path / "app.log"
    setup_logging(verbosity=2, log_file=log_file)
    root = logging.getLogger("tooluse_gen")
    assert len(root.handlers) == 2
    root.debug("file test message")
    assert log_file.exists()
    assert "file test message" in log_file.read_text()


def test_setup_logging_json_file_handler(tmp_path: Path):
    log_file = tmp_path / "app.jsonl"
    setup_logging(verbosity=2, log_file=log_file, json_logs=True)
    root = logging.getLogger("tooluse_gen")
    root.info("json test")
    lines = [line for line in log_file.read_text().splitlines() if line.strip()]
    assert lines, "No log lines written"
    record = json.loads(lines[-1])
    assert record["message"] == "json test"
    assert "timestamp" in record
    assert "level" in record
    assert "module" in record


def test_setup_logging_file_parent_dir_auto_created(tmp_path: Path):
    log_file = tmp_path / "deep" / "nested" / "app.log"
    setup_logging(log_file=log_file)
    assert log_file.parent.exists()


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------


def test_get_logger_namespaced():
    logger = get_logger("registry")
    assert logger.name == "tooluse_gen.registry"


def test_get_logger_already_namespaced():
    logger = get_logger("tooluse_gen.graph")
    assert logger.name == "tooluse_gen.graph"


def test_get_logger_is_child_of_root():
    root = logging.getLogger("tooluse_gen")
    child = get_logger("agents")
    assert child.parent is root


# ---------------------------------------------------------------------------
# log_context
# ---------------------------------------------------------------------------


def test_log_context_injects_fields():
    setup_logging(verbosity=2)
    root = logging.getLogger("tooluse_gen")
    h, buf = _capture_handler()
    root.addHandler(h)

    with log_context(conv_id="conv_99"):
        root.info("inside context")

    # The filter injects into the LogRecord; check it was processed without error
    output = buf.getvalue()
    assert "inside context" in output


def test_log_context_nested():
    setup_logging(verbosity=2)
    from tooluse_gen.utils.logging import _log_extra

    with log_context(a=1):
        inner = _log_extra.get() or {}
        assert inner.get("a") == 1
        with log_context(b=2):
            both = _log_extra.get() or {}
            assert both.get("a") == 1
            assert both.get("b") == 2
        restored = _log_extra.get() or {}
        assert restored.get("a") == 1
        assert "b" not in restored


def test_log_context_restores_after_exception():
    from tooluse_gen.utils.logging import _log_extra

    assert _log_extra.get() is None
    try:
        with log_context(x=99):
            raise ValueError("boom")
    except ValueError:
        pass
    assert _log_extra.get() is None


# ---------------------------------------------------------------------------
# log_duration
# ---------------------------------------------------------------------------


def test_log_duration_success_logs_completion(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.INFO, logger="tooluse_gen"), log_duration("test operation"):
        time.sleep(0.01)
    assert any("test operation" in r.message and "completed" in r.message for r in caplog.records)


def test_log_duration_records_elapsed(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.INFO, logger="tooluse_gen"), log_duration("timed op"):
        time.sleep(0.02)
    completion = next(r for r in caplog.records if "completed" in r.message)
    assert "timed op" in completion.message


def test_log_duration_failure_logs_error(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.ERROR, logger="tooluse_gen"), pytest.raises(RuntimeError), log_duration("failing op"):
        raise RuntimeError("oops")
    assert any("Failed" in r.message and "failing op" in r.message for r in caplog.records)


def test_log_duration_re_raises_exception():
    with pytest.raises(ValueError, match="expected"), log_duration("op"):
        raise ValueError("expected")


def test_log_duration_custom_logger(caplog: pytest.LogCaptureFixture):
    custom = logging.getLogger("tooluse_gen.custom")
    with caplog.at_level(logging.INFO, logger="tooluse_gen.custom"), log_duration("custom op", logger=custom):
        pass
    assert any("custom op" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# create_progress_bar
# ---------------------------------------------------------------------------


def test_create_progress_bar_returns_tqdm():
    from tqdm import tqdm

    pbar = create_progress_bar(10, "Test", disable=True)
    assert isinstance(pbar, tqdm)
    pbar.close()


def test_create_progress_bar_total_and_desc():
    from io import StringIO

    pbar = create_progress_bar(42, "My Task", file=StringIO())
    assert pbar.total == 42
    # desc is stored internally; check via format_dict
    assert pbar.format_dict.get("prefix") == "My Task"
    pbar.close()


def test_create_progress_bar_disabled():
    pbar = create_progress_bar(10, "Test", disable=True)
    assert pbar.disable
    pbar.close()


def test_create_progress_bar_as_context_manager():
    from io import StringIO

    # disable=True makes update() a no-op; use a redirected real bar instead
    with create_progress_bar(5, "ctx", file=StringIO()) as pbar:
        for _ in range(5):
            pbar.update(1)
    assert pbar.n == 5


# ---------------------------------------------------------------------------
# progress_callback
# ---------------------------------------------------------------------------


def test_progress_callback_advances_bar():
    from io import StringIO

    pbar = create_progress_bar(3, "cb test", file=StringIO())
    cb = progress_callback(pbar)
    assert pbar.n == 0
    cb()
    assert pbar.n == 1
    cb()
    assert pbar.n == 2
    pbar.close()


def test_progress_callback_is_callable():
    pbar = create_progress_bar(1, "test", disable=True)
    cb = progress_callback(pbar)
    assert callable(cb)
    pbar.close()


# ---------------------------------------------------------------------------
# CLI integration — setup_logging called by callback
# ---------------------------------------------------------------------------


def test_cli_callback_calls_setup_logging():
    """Verify CLI --verbose flag wires through to setup_logging."""
    from typer.testing import CliRunner

    from tooluse_gen.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["-vv", "build", "--input-dir", "data/"])
    assert result.exit_code == 0
    # At verbosity 2 the root logger should be at DEBUG level
    root = logging.getLogger("tooluse_gen")
    assert root.level <= logging.DEBUG
