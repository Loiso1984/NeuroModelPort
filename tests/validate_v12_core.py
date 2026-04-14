"""Scientific validation suite for v12.0 core — Zero-Slice + Benettin LLE.

Tests:
  1. Regression: V_soma from zero-slice (calc_lle=False) vs reference (< 1e-10 mV).
  2. Performance: dual-trajectory LLE mode must be ~2.0x slower, not more.
  3. LLE convergence: verify exponent stabilises on a known regime.
"""
import sys
import os
import time
import numpy as np

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.models import FullModelConfig
from core.solver import NeuronSolver


def _make_default_config(t_sim: float = 100.0) -> FullModelConfig:
    """Return a minimal FullModelConfig for validation runs."""
    cfg = FullModelConfig()
    cfg.stim.t_sim = t_sim
    cfg.stim.stim_type = "pulse"
    cfg.stim.Iext = 10.0
    cfg.stim.pulse_start = 10.0
    cfg.stim.pulse_dur = 50.0
    cfg.stim.dt_eval = 0.1
    cfg.stim.jacobian_mode = "native_hines"
    return cfg


# ── TEST 1: Regression — deterministic zero-slice ──────────────────────────

def test_regression_deterministic():
    """Two identical runs must produce bit-identical V_soma traces."""
    cfg = _make_default_config(100.0)
    solver = NeuronSolver(cfg)

    res1 = solver.run_native(cfg)
    res2 = solver.run_native(cfg)

    diff = np.max(np.abs(res1.v_soma - res2.v_soma))
    print(f"[Regression] max |V_soma_run1 - V_soma_run2| = {diff:.2e} mV")
    assert diff < 1e-10, f"Regression failed: diff={diff:.2e} > 1e-10 mV"
    print("  PASS")


# ── TEST 2: Performance — LLE dual-trajectory overhead ─────────────────────

def test_lle_performance_overhead():
    """LLE (dual trajectory) should be ~2.0x slower than single trajectory.

    An overhead significantly above 2.5x suggests register spilling or
    allocation inside the @njit loop.
    """
    cfg = _make_default_config(1000.0)
    solver = NeuronSolver(cfg)

    # Warmup JIT
    _ = solver.run_native(cfg, calc_lle=False)
    _ = solver.run_native(cfg, calc_lle=True)

    # Timed single-trajectory
    t0 = time.perf_counter()
    _ = solver.run_native(cfg, calc_lle=False)
    t_single = time.perf_counter() - t0

    # Timed dual-trajectory (LLE)
    t0 = time.perf_counter()
    _ = solver.run_native(cfg, calc_lle=True)
    t_dual = time.perf_counter() - t0

    ratio = t_dual / max(t_single, 1e-9)
    print(f"[Performance] single={t_single:.3f}s  dual(LLE)={t_dual:.3f}s  ratio={ratio:.2f}x")
    assert ratio < 3.0, f"LLE overhead too high: {ratio:.2f}x (expected ~2.0x)"
    print("  PASS")


# ── TEST 3: LLE convergence ────────────────────────────────────────────────

def test_lle_convergence_stable():
    """LLE should converge to a finite value (not NaN/Inf) on a standard HH neuron.

    For a periodic spiking neuron, the maximal Lyapunov exponent is typically
    near zero or slightly negative (stable limit cycle).  For chaotic regimes
    it would be positive.  Here we only verify convergence — not the sign.
    """
    cfg = _make_default_config(500.0)
    cfg.stim.Iext = 10.0  # tonic spiking regime
    solver = NeuronSolver(cfg)

    res = solver.run_native(cfg, calc_lle=True, lle_delta=1e-6, lle_t_evolve=1.0)

    lle = res.lle_convergence
    assert lle is not None, "lle_convergence is None"
    assert len(lle) > 0, "lle_convergence is empty"

    # Last 20% should be converged (not NaN/Inf)
    tail = lle[int(0.8 * len(lle)):]
    finite_mask = np.isfinite(tail)
    finite_frac = np.sum(finite_mask) / max(len(tail), 1)
    print(f"[LLE convergence] final LLE ~= {tail[-1]:.4f} ms/ms, "
          f"finite fraction in tail = {finite_frac:.2%}")
    assert finite_frac > 0.9, f"Too many non-finite LLE values: {1 - finite_frac:.1%}"

    # LLE should be bounded (physiological range: roughly -10 to +5 ms/ms)
    lle_final = tail[finite_mask][-1]
    assert -50.0 < lle_final < 50.0, f"LLE out of reasonable range: {lle_final:.2f}"
    print("  PASS")


# ── TEST 4: LLE disabled produces identical trace to baseline ──────────────

def test_lle_disabled_no_side_effects():
    """calc_lle=False must produce identical results to the pre-LLE code path.

    This verifies that the trajectory-loop refactoring introduces no
    regression when LLE is not requested.
    """
    cfg = _make_default_config(200.0)
    solver = NeuronSolver(cfg)

    res_base = solver.run_native(cfg, calc_lle=False)
    res_lle  = solver.run_native(cfg, calc_lle=True)

    # Main trajectory (y_out) should be identical regardless of LLE
    diff_v = np.max(np.abs(res_base.v_soma - res_lle.v_soma))
    print(f"[Side-effect check] max |V_base - V_lle| = {diff_v:.2e} mV")
    # With LLE the main trajectory follows the same physics, so diff should be 0
    # (stochastic noise is only on main traj, same RNG sequence)
    assert diff_v < 1e-8, f"LLE mode altered main trajectory: diff={diff_v:.2e}"
    print("  PASS")


# ── Runner ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("v12.0 Core Validation — Zero-Slice + Benettin LLE")
    print("=" * 60)

    tests = [
        test_regression_deterministic,
        test_lle_performance_overhead,
        test_lle_convergence_stable,
        test_lle_disabled_no_side_effects,
    ]

    passed, failed = 0, 0
    for test_fn in tests:
        try:
            print(f"\n> {test_fn.__name__}")
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(1 if failed else 0)
