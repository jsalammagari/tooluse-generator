"""Logging infrastructure for tooluse-generator.

Verbosity levels:
    0  (default)  — WARNING and above only
    1  (-v)       — INFO  (progress, major steps)
    2  (-vv)      — DEBUG (detailed operations)
    3+ (-vvv)     — DEBUG + noisy third-party loggers silenced less

Handlers:
    Console  — RichHandler (falls back to StreamHandler) with coloured output
    File     — TimedRotatingFileHandler with structured plain-text format
    JSON     — Same file handler but emitting newline-delimited JSON
"""

from __future__ import annotations

import contextlib
import json
import logging
import logging.handlers
import time
from collections.abc import Generator
from contextvars import ContextVar
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Package-level root logger name
# ---------------------------------------------------------------------------
_ROOT = "tooluse_gen"

# ContextVar that carries extra fields injected by log_context()
_log_extra: ContextVar[dict[str, Any] | None] = ContextVar("_log_extra", default=None)

# Verbosity → stdlib log level mapping
_VERBOSITY_TO_LEVEL = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG,
    3: logging.DEBUG,  # same level, but third-party loggers stay verbose too
}

# ---------------------------------------------------------------------------
# Custom formatters
# ---------------------------------------------------------------------------

_CONSOLE_FMT = "[%(asctime)s] %(levelname)-8s | %(name)s | %(message)s"
_CONSOLE_DATE = "%H:%M:%S"
_FILE_FMT = (
    "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
)
_FILE_DATE = "%Y-%m-%d %H:%M:%S"


class _ContextInjectingFilter(logging.Filter):
    """Inject ContextVar extra fields into every LogRecord."""

    def filter(self, record: logging.LogRecord) -> bool:
        for k, v in (_log_extra.get() or {}).items():
            setattr(record, k, v)
        return True


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, _FILE_DATE),
            "level": record.levelname,
            "module": record.name,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # Attach any extra context fields
        extra = _log_extra.get()
        if extra:
            payload["extra"] = extra
        return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def setup_logging(
    verbosity: int = 0,
    quiet: bool = False,
    log_file: Path | None = None,
    json_logs: bool = False,
) -> logging.Logger:
    """Configure application-wide logging.

    Args:
        verbosity: 0–3.  Controls console detail level.
        quiet:     Suppress all non-error console output (overrides verbosity).
        log_file:  If given, also write logs to this file (rotated daily).
        json_logs: Write JSON-formatted lines to *log_file* instead of plain text.

    Returns:
        The ``tooluse_gen`` root logger.
    """
    root = logging.getLogger(_ROOT)

    # Avoid adding duplicate handlers on repeated calls (e.g. in tests)
    if root.handlers:
        root.handlers.clear()

    # Set the effective level to the most permissive we'll ever emit
    effective_verbosity = max(0, min(verbosity, 3))
    console_level = logging.ERROR if quiet else _VERBOSITY_TO_LEVEL[effective_verbosity]
    root.setLevel(min(console_level, logging.DEBUG))  # file handler may need DEBUG

    context_filter = _ContextInjectingFilter()

    # ── Console handler ───────────────────────────────────────────────────────
    try:
        from rich.logging import RichHandler

        console_handler: logging.Handler = RichHandler(
            level=console_level,
            show_time=verbosity >= 1,
            show_path=verbosity >= 2,
            rich_tracebacks=True,
            markup=True,
            log_time_format=_CONSOLE_DATE,
        )
    except ImportError:  # pragma: no cover
        console_handler = logging.StreamHandler()
        fmt = _CONSOLE_FMT if not quiet else "%(levelname)s: %(message)s"
        console_handler.setFormatter(logging.Formatter(fmt, datefmt=_CONSOLE_DATE))
        console_handler.setLevel(console_level)

    console_handler.addFilter(context_filter)
    root.addHandler(console_handler)

    # ── File handler (optional) ───────────────────────────────────────────────
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler: logging.Handler = logging.handlers.TimedRotatingFileHandler(
            filename=log_file,
            when="midnight",
            backupCount=7,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)  # capture everything to file
        if json_logs:
            file_handler.setFormatter(_JsonFormatter())
        else:
            file_handler.setFormatter(logging.Formatter(_FILE_FMT, datefmt=_FILE_DATE))
        file_handler.addFilter(context_filter)
        root.addHandler(file_handler)

    # Silence noisy third-party loggers unless very verbose
    if effective_verbosity < 3:
        for noisy in ("httpx", "httpcore", "openai", "sentence_transformers", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    return root


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``tooluse_gen`` namespace.

    Args:
        name: Dotted sub-name, e.g. ``"registry"`` → ``tooluse_gen.registry``.

    Returns:
        A :class:`logging.Logger` instance.
    """
    if name.startswith(_ROOT):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT}.{name}")


@contextlib.contextmanager
def log_context(**kwargs: Any) -> Generator[None, None, None]:
    """Context manager that attaches extra fields to all log records within.

    Example::

        with log_context(conversation_id="conv_0042", seed=42):
            logger.info("Generating conversation")
            # LogRecord will include extra={"conversation_id": "conv_0042", "seed": 42}
    """
    token = _log_extra.set({**(_log_extra.get() or {}), **kwargs})
    try:
        yield
    finally:
        _log_extra.reset(token)


@contextlib.contextmanager
def log_duration(
    operation: str, logger: logging.Logger | None = None
) -> Generator[None, None, None]:
    """Context manager that logs the wall-clock duration of an operation.

    Args:
        operation: Human-readable name for the operation being timed.
        logger:    Logger to emit to.  Defaults to the ``tooluse_gen`` root logger.

    Example::

        with log_duration("build tool graph"):
            graph = build_graph(tools)
        # → INFO tooluse_gen | build tool graph completed in 3.41s
    """
    _logger = logger or logging.getLogger(_ROOT)
    start = time.perf_counter()
    _logger.debug("Starting: %s", operation)
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - start
        _logger.error("Failed: %s (after %.2fs)", operation, elapsed)
        raise
    else:
        elapsed = time.perf_counter() - start
        _logger.info("%s completed in %.2fs", operation, elapsed)
