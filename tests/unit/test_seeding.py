"""Unit tests for deterministic seeding utilities (Task 7)."""

from __future__ import annotations

import random

import numpy as np
import pytest

from tooluse_gen.utils.seeding import (
    SeedManager,
    get_rng,
    get_seed_manager,
    reproducible_context,
    restore_random_state,
    save_random_state,
    set_global_seed,
    warn_nondeterministic,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_global() -> None:
    """Clear the module-level _seed_manager between tests."""
    import tooluse_gen.utils.seeding as _mod

    _mod._seed_manager = None


# ---------------------------------------------------------------------------
# SeedManager — initialization
# ---------------------------------------------------------------------------


def test_seed_manager_stores_base_seed():
    sm = SeedManager(42)
    assert sm.base_seed == 42


def test_initialize_sets_python_random():
    sm = SeedManager(7)
    sm.initialize()
    val1 = random.random()

    sm2 = SeedManager(7)
    sm2.initialize()
    val2 = random.random()

    assert val1 == val2


def test_initialize_sets_numpy_random():
    sm = SeedManager(99)
    sm.initialize()
    arr1 = np.random.rand(5).tolist()

    sm2 = SeedManager(99)
    sm2.initialize()
    arr2 = np.random.rand(5).tolist()

    assert arr1 == arr2


def test_different_seeds_produce_different_sequences():
    SeedManager(1).initialize()
    v1 = random.random()

    SeedManager(2).initialize()
    v2 = random.random()

    assert v1 != v2


# ---------------------------------------------------------------------------
# SeedManager — component seeds
# ---------------------------------------------------------------------------


def test_component_seed_deterministic():
    sm = SeedManager(42)
    s1 = sm.get_component_seed("sampling")
    s2 = sm.get_component_seed("sampling")
    assert s1 == s2


def test_component_seed_same_base_same_result():
    s1 = SeedManager(42).get_component_seed("sampling")
    s2 = SeedManager(42).get_component_seed("sampling")
    assert s1 == s2


def test_component_seed_different_components_differ():
    sm = SeedManager(42)
    assert sm.get_component_seed("sampling") != sm.get_component_seed("mock_generation")


def test_component_seed_different_bases_differ():
    s1 = SeedManager(1).get_component_seed("sampling")
    s2 = SeedManager(2).get_component_seed("sampling")
    assert s1 != s2


def test_component_seed_in_uint32_range():
    sm = SeedManager(12345)
    for comp in ("sampling", "mock_generation", "user_simulator", "assistant", "judge", "diversity"):
        seed = sm.get_component_seed(comp)
        assert 0 <= seed < 2**32


def test_component_seed_cached():
    sm = SeedManager(42)
    sm.get_component_seed("sampling")
    assert "sampling" in sm._component_seeds


# ---------------------------------------------------------------------------
# SeedManager — get_rng
# ---------------------------------------------------------------------------


def test_get_rng_returns_random_instance():
    sm = SeedManager(42)
    rng = sm.get_rng("sampling")
    assert isinstance(rng, random.Random)


def test_get_rng_same_seed_same_sequence():
    sm1 = SeedManager(42)
    sm2 = SeedManager(42)
    seq1 = [sm1.get_rng("sampling").random() for _ in range(5)]
    seq2 = [sm2.get_rng("sampling").random() for _ in range(5)]
    assert seq1 == seq2


def test_get_rng_components_isolated():
    sm = SeedManager(42)
    rng_a = sm.get_rng("sampling")
    rng_b = sm.get_rng("mock_generation")
    # Consuming from one should not affect the other
    _ = rng_a.random()
    val_b1 = rng_b.random()

    sm2 = SeedManager(42)
    val_b2 = sm2.get_rng("mock_generation").random()
    assert val_b1 == val_b2


# ---------------------------------------------------------------------------
# SeedManager — get_numpy_rng
# ---------------------------------------------------------------------------


def test_get_numpy_rng_returns_generator():
    sm = SeedManager(42)
    rng = sm.get_numpy_rng("diversity")
    assert isinstance(rng, np.random.Generator)


def test_get_numpy_rng_same_seed_same_sequence():
    sm1 = SeedManager(42)
    sm2 = SeedManager(42)
    arr1 = sm1.get_numpy_rng("diversity").random(10).tolist()
    arr2 = sm2.get_numpy_rng("diversity").random(10).tolist()
    assert arr1 == arr2


def test_get_numpy_rng_different_components_differ():
    sm = SeedManager(42)
    arr_a = sm.get_numpy_rng("sampling").random(5).tolist()
    arr_b = sm.get_numpy_rng("diversity").random(5).tolist()
    assert arr_a != arr_b


# ---------------------------------------------------------------------------
# Global seed functions
# ---------------------------------------------------------------------------


def test_set_global_seed_returns_seed_manager():
    _reset_global()
    sm = set_global_seed(42)
    assert isinstance(sm, SeedManager)
    assert sm.base_seed == 42


def test_get_seed_manager_after_set():
    _reset_global()
    set_global_seed(7)
    sm = get_seed_manager()
    assert sm.base_seed == 7


def test_get_seed_manager_raises_without_init():
    _reset_global()
    with pytest.raises(RuntimeError, match="not initialized"):
        get_seed_manager()


def test_get_rng_convenience():
    _reset_global()
    set_global_seed(42)
    rng = get_rng("sampling")
    assert isinstance(rng, random.Random)


def test_set_global_seed_replaces_previous():
    _reset_global()
    set_global_seed(1)
    set_global_seed(99)
    assert get_seed_manager().base_seed == 99


# ---------------------------------------------------------------------------
# reproducible_context
# ---------------------------------------------------------------------------


def test_reproducible_context_same_results():
    with reproducible_context(42):
        v1 = random.random()
    with reproducible_context(42):
        v2 = random.random()
    assert v1 == v2


def test_reproducible_context_restores_state():
    random.seed(0)
    before = random.getstate()

    with reproducible_context(999):
        random.random()

    after = random.getstate()
    assert before == after


def test_reproducible_context_restores_on_exception():
    random.seed(0)
    before = random.getstate()

    with pytest.raises(ValueError), reproducible_context(999):
        random.random()
        raise ValueError("boom")

    assert random.getstate() == before


def test_reproducible_context_numpy_same_results():
    with reproducible_context(42):
        arr1 = np.random.rand(5).tolist()
    with reproducible_context(42):
        arr2 = np.random.rand(5).tolist()
    assert arr1 == arr2


# ---------------------------------------------------------------------------
# save_random_state / restore_random_state
# ---------------------------------------------------------------------------


def test_save_restore_python_random():
    random.seed(55)
    state = save_random_state()
    v1 = random.random()

    restore_random_state(state)
    v2 = random.random()

    assert v1 == v2


def test_save_restore_numpy():
    np.random.seed(55)
    state = save_random_state()
    arr1 = np.random.rand(5).tolist()

    restore_random_state(state)
    arr2 = np.random.rand(5).tolist()

    assert arr1 == arr2


def test_save_state_contains_expected_keys():
    state = save_random_state()
    assert "python" in state
    assert "numpy" in state
    assert "torch" in state


# ---------------------------------------------------------------------------
# warn_nondeterministic
# ---------------------------------------------------------------------------


def test_warn_nondeterministic_logs_warning(caplog: pytest.LogCaptureFixture):
    import logging

    with caplog.at_level(logging.WARNING, logger="tooluse_gen.seeding"):
        warn_nondeterministic("faiss", "ANN search is not deterministic")

    assert any(
        "faiss" in r.message and "ANN search" in r.message for r in caplog.records
    )
