from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math


def test_reflect_unit_interval_handles_large_overshoot():
    from core.native_loop import _reflect_unit_interval

    assert _reflect_unit_interval(-0.25) == 0.25
    assert _reflect_unit_interval(1.25) == 0.75
    assert _reflect_unit_interval(2.25) == 0.25


def test_reflect_unit_interval_handles_nonfinite_without_looping():
    from core.native_loop import _reflect_unit_interval

    assert math.isnan(_reflect_unit_interval(float("nan")))
    assert math.isnan(_reflect_unit_interval(float("inf")))
    assert math.isnan(_reflect_unit_interval(float("-inf")))


def test_stochastic_native_seed_resolution_is_thread_safe_and_reproducible_when_seeded():
    from core.models import FullModelConfig
    from core.solver import _resolve_stochastic_seed
    from core.stochastic_rng import reset_rng, seed_all

    cfg = FullModelConfig()
    seed_all(12345)
    assert _resolve_stochastic_seed(cfg, 0.05, True) == 12345

    reset_rng(None)
    with ThreadPoolExecutor(max_workers=8) as pool:
        seeds = list(pool.map(lambda _: _resolve_stochastic_seed(cfg, 0.05, True), range(16)))

    assert len(set(seeds)) > 1


def test_thread_local_rng_does_not_replace_global_rng_from_worker_thread():
    import core.stochastic_rng as rng_mod

    rng_mod.reset_rng(123)
    main_rng = rng_mod.get_rng()

    with ThreadPoolExecutor(max_workers=1) as pool:
        worker_rng = pool.submit(rng_mod.get_rng).result()

    assert worker_rng is not main_rng
    assert rng_mod._GLOBAL_RNG is main_rng


def test_jacobian_cache_has_lock_guard():
    import core.jacobian as jac

    assert hasattr(jac, "_LEGACY_JACOBIAN_CACHE_LOCK")
    jac.clear_legacy_jacobian_cache()
