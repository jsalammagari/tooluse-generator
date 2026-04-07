"""Deterministic seeding utilities for reproducible random operations.

Verbosity:
    INFO  — base seed initialization
    DEBUG — per-component derived seeds
"""

from __future__ import annotations

import contextlib
import random
from collections.abc import Generator
from typing import Any

import numpy as np

from tooluse_gen.utils.logging import get_logger

logger = get_logger("seeding")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_seed_manager: SeedManager | None = None


# ---------------------------------------------------------------------------
# SeedManager
# ---------------------------------------------------------------------------


class SeedManager:
    """Manages deterministic seeding across all random sources.

    Ensures reproducibility by seeding:
    - Python's ``random`` module
    - NumPy's random generator
    - PyTorch (if available, for sentence-transformers)

    Also provides methods to create derived seeds for different
    components (sampling, mock generation, etc.) to allow
    partial reproducibility.
    """

    def __init__(self, seed: int) -> None:
        self.base_seed = seed
        self._component_seeds: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Set all random seeds.  Call once at startup."""
        random.seed(self.base_seed)
        np.random.seed(self.base_seed)  # legacy API — also seeds default RNG
        _try_seed_torch(self.base_seed)
        logger.info("Random seed initialized: %d", self.base_seed)

    def get_component_seed(self, component: str) -> int:
        """Return a deterministic seed for *component*.

        The derivation is::

            component_seed = hash(f"{base_seed}:{component}") % 2**32

        The same ``(base_seed, component)`` pair always yields the same value.

        Args:
            component: One of ``'sampling'``, ``'mock_generation'``,
                ``'user_simulator'``, ``'assistant'``, ``'judge'``,
                ``'diversity'`` (or any other string key).
        """
        if component not in self._component_seeds:
            derived = hash(f"{self.base_seed}:{component}") % (2**32)
            self._component_seeds[component] = derived
            logger.debug("Derived seed for '%s': %d", component, derived)
        return self._component_seeds[component]

    def get_rng(self, component: str) -> random.Random:
        """Return an isolated :class:`random.Random` instance for *component*."""
        rng = random.Random()
        rng.seed(self.get_component_seed(component))
        return rng

    def get_numpy_rng(self, component: str) -> np.random.Generator:
        """Return an isolated :class:`numpy.random.Generator` for *component*."""
        return np.random.default_rng(self.get_component_seed(component))


# ---------------------------------------------------------------------------
# Global helpers
# ---------------------------------------------------------------------------


def set_global_seed(seed: int) -> SeedManager:
    """Initialize the global :class:`SeedManager` and seed all RNGs.

    Args:
        seed: Base integer seed.

    Returns:
        The newly created :class:`SeedManager`.
    """
    global _seed_manager
    _seed_manager = SeedManager(seed)
    _seed_manager.initialize()
    return _seed_manager


def get_seed_manager() -> SeedManager:
    """Return the global :class:`SeedManager`.

    Raises:
        RuntimeError: If :func:`set_global_seed` has not been called yet.
    """
    if _seed_manager is None:
        raise RuntimeError("Global seed manager not initialized — call set_global_seed() first.")
    return _seed_manager


def get_rng(component: str) -> random.Random:
    """Convenience wrapper: return a component-isolated :class:`random.Random`."""
    return get_seed_manager().get_rng(component)


# ---------------------------------------------------------------------------
# Reproducibility context manager
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def reproducible_context(seed: int) -> Generator[None, None, None]:
    """Context manager that saves/restores random state around a block.

    Seeds Python ``random``, NumPy, and (optionally) PyTorch to *seed*
    at entry, then restores the previous state on exit — even if an
    exception is raised.

    Args:
        seed: Seed to apply for the duration of the block.

    Example::

        with reproducible_context(42):
            result = random.random()   # always the same value
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = _get_torch_state()

    random.seed(seed)
    np.random.seed(seed)
    _try_seed_torch(seed)

    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        _restore_torch_state(torch_state)


# ---------------------------------------------------------------------------
# Non-determinism warning
# ---------------------------------------------------------------------------


def warn_nondeterministic(source: str, reason: str) -> None:
    """Log a WARNING about a source of non-determinism.

    Call this when operations that cannot be seeded are used
    (e.g., approximate nearest-neighbour search, network I/O).

    Args:
        source: Name of the subsystem introducing non-determinism.
        reason: Human-readable explanation.
    """
    logger.warning(
        "Non-deterministic operation in '%s': %s — results may not be reproducible.",
        source,
        reason,
    )


# ---------------------------------------------------------------------------
# State serialization
# ---------------------------------------------------------------------------


def save_random_state() -> dict[str, Any]:
    """Capture the current state of all random generators.

    Returns:
        A dict suitable for passing to :func:`restore_random_state`.
    """
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": _get_torch_state(),
    }
    return state


def restore_random_state(state: dict[str, Any]) -> None:
    """Restore random generator state from a snapshot produced by
    :func:`save_random_state`.

    Args:
        state: Snapshot dict.
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    _restore_torch_state(state.get("torch"))


# ---------------------------------------------------------------------------
# Internal PyTorch helpers (optional dependency)
# ---------------------------------------------------------------------------


def _try_seed_torch(seed: int) -> None:
    """Seed PyTorch if it is installed."""
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _get_torch_state() -> Any:
    """Return PyTorch RNG state, or *None* if PyTorch is not installed."""
    try:
        import torch

        return torch.get_rng_state()
    except ImportError:
        return None


def _restore_torch_state(state: Any) -> None:
    """Restore PyTorch RNG state if possible."""
    if state is None:
        return
    try:
        import torch

        torch.set_rng_state(state)
    except ImportError:
        pass
