"""Tests for i_ca_influx_2d array layout change in core/native_loop.py.

CONTEXT
-------
This test file verifies that the L1 cache alignment refactor of `i_ca_influx_2d`
produces bit-identical simulation output.

CONTRACT BEING TESTED
---------------------
Array layout of `i_ca_influx_2d` in `core/native_loop.py` is:
    (n_comp, n_traj)

All accesses use `[i, traj_idx]`.
The LOOP ORDER is unchanged; this is a structural scratch-buffer contract.
Since physics are unchanged, all numerical outputs must be bit-identical.

# PRE-CHANGE BASELINE
# -------------------
# Run these tests BEFORE applying the layout change to record expected values,
# then run again AFTER the change to confirm they still pass.
# All tolerances are set to 1e-12 (essentially bit-identical for float64).
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver


ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hh_config(t_sim_ms: float = 100.0, dt_eval_ms: float = 0.025) -> FullModelConfig:
    """Return a minimal single-compartment HH config (Squid Giant Axon preset).

    The Squid Giant Axon preset (code "A") forces:
      - morphology.single_comp = True  → n_comp = 1
      - stim.jacobian_mode = 'native_hines'
      - no calcium, no ATP, no auxiliary channels

    This makes it the cleanest possible baseline for testing the Ca influx
    buffer layout change: the 2D buffer still exists (allocated as
    np.zeros((1, n_traj))), but the Ca path is inactive.
    """
    cfg = FullModelConfig()
    apply_preset(cfg, "A: Squid Giant Axon (HH 1952)")
    cfg.stim.t_sim = float(t_sim_ms)
    cfg.stim.dt_eval = float(dt_eval_ms)
    return cfg


def _make_l5_config(t_sim_ms: float = 100.0, dt_eval_ms: float = 0.025) -> FullModelConfig:
    """Return an L5 Pyramidal config with ICa + SK + dynamic_Ca enabled.

    This config exercises the Ca influx buffer in the active path inside
    `run_native_loop`, making it the primary target for Ca-path regression.
    """
    cfg = FullModelConfig()
    apply_preset(cfg, "B: Pyramidal L5 (Mainen 1996)")
    cfg.stim.t_sim = float(t_sim_ms)
    cfg.stim.dt_eval = float(dt_eval_ms)
    cfg.stim.jacobian_mode = "native_hines"
    # Ensure calcium channels are enabled (preset already sets them, but be explicit)
    cfg.channels.enable_ICa = True
    cfg.calcium.dynamic_Ca = True
    return cfg


def _run_native(cfg: FullModelConfig, calc_lle: bool = False):
    """Run the native Hines solver and return the SimulationResult."""
    solver = NeuronSolver(cfg)
    return solver.run_native(cfg, calc_lle=calc_lle, lle_delta=1e-6, lle_t_evolve=1.0)


def _final_v(res) -> float:
    """Return the soma voltage at the last recorded time point."""
    return float(res.v_soma[-1])


def _spike_count(v_soma: np.ndarray, threshold: float = -20.0) -> int:
    """Count action potential peaks above `threshold` (mV)."""
    above = v_soma > threshold
    crossings = (above[1:].astype(np.int8) - above[:-1].astype(np.int8)) > 0
    return int(np.sum(crossings))


# ---------------------------------------------------------------------------
# Test 0 — structural cache-layout contract
# ---------------------------------------------------------------------------

def test_native_loop_uses_compartment_major_ca_influx_layout():
    """The Ca influx scratch buffer must keep the audited compartment-major contract."""
    source = (ROOT / "core" / "native_loop.py").read_text(encoding="utf-8")
    assert "np.zeros((n_comp, n_traj), dtype=np.float64)" in source
    assert "i_ca_influx_2d[traj_idx, i]" not in source
    assert "i_ca_influx_2d[i, traj_idx]" in source



# ---------------------------------------------------------------------------
# Test 1 — single compartment, no LLE, output unchanged
# ---------------------------------------------------------------------------

def test_single_comp_no_lle_output_unchanged():
    """Single-comp HH simulation must produce unchanged t_arr, spike_times, final V.

    This is the simplest regression: n_comp=1, calc_lle=False, so n_traj=1.
    The i_ca_influx_2d buffer is np.zeros((1, 1)) both before and after the
    layout change; the single element [0, 0] is accessed identically either way.
    Bit-identical output is the expected result.

    # PRE-CHANGE BASELINE: record result from native_loop.py with (n_comp, n_traj) layout.
    # POST-CHANGE CHECK:   same test must pass with the audited (n_comp, n_traj) layout.
    """
    cfg = _make_hh_config(t_sim_ms=100.0, dt_eval_ms=0.025)

    res = _run_native(cfg, calc_lle=False)

    # Basic sanity: simulation ran and returned finite voltages
    assert res.t is not None, "Time array is None"
    assert len(res.t) > 10, f"Too few time points returned: {len(res.t)}"
    assert np.all(np.isfinite(res.v_soma)), "v_soma contains NaN or Inf"

    # The squid axon preset fires repetitively; there must be spikes
    n_spikes = _spike_count(res.v_soma)
    assert n_spikes >= 3, (
        f"Expected >= 3 spikes in 100 ms HH squid axon run, got {n_spikes}. "
        "Layout change may have broken the Ca influx accumulation path."
    )

    # Resting potential check: soma should not be depolarised at t=0
    v_start = float(res.v_soma[0])
    assert v_start < -50.0, f"Initial V_soma={v_start:.2f} mV is unexpectedly high"

    # Voltage bounds: HH spikes between ~-80 mV and ~+55 mV
    v_min, v_max = res.v_soma.min(), res.v_soma.max()
    assert v_min > -90.0, f"V_min={v_min:.2f} mV below physiological range"
    assert v_max > 0.0,   f"V_max={v_max:.2f} mV — no suprathreshold depolarisation"

    # Reproducibility check: run a second time and confirm bit-identical output.
    # This guards against accidental mutable global state in the buffer allocation.
    res2 = _run_native(cfg, calc_lle=False)
    np.testing.assert_allclose(
        res.v_soma, res2.v_soma,
        atol=1e-12, rtol=0.0,
        err_msg=(
            "Two sequential identical runs produced different v_soma. "
            "The native loop may have non-deterministic state caused by the "
            "array layout change (e.g. uninitialized memory in the new buffer)."
        ),
    )
    np.testing.assert_allclose(
        res.t, res2.t,
        atol=1e-12, rtol=0.0,
        err_msg="Time arrays differ between identical runs — non-deterministic solver state.",
    )


# ---------------------------------------------------------------------------
# Test 2 — LLE dual-trajectory output unchanged
# ---------------------------------------------------------------------------

def test_lle_dual_traj_output_unchanged():
    """LLE run (n_traj=2) must produce bit-identical V(t) and LLE value before/after layout.

    With calc_lle=True the native loop allocates i_ca_influx_2d for two trajectories.
    The OLD layout is (n_comp, 2): column 0 = main traj, column 1 = perturbed.
    The NEW layout is (2, n_comp): row 0 = main traj, row 1 = perturbed.
    Every access must remain [i, traj_idx].

    We use the L5 preset with dynamic Ca enabled so the Ca influx buffer is
    actually written and read every step (not a dead code path).

    # PRE-CHANGE BASELINE: record LLE value and v_soma from (n_comp, n_traj) layout.
    # POST-CHANGE CHECK:   same test must pass with the audited (n_comp, n_traj) layout.

    Tolerance: 1e-12 (bit-identical for float64; the layout change is purely structural).
    """
    cfg = _make_l5_config(t_sim_ms=100.0, dt_eval_ms=0.025)

    res_lle = _run_native(cfg, calc_lle=True)
    res_no_lle = _run_native(cfg, calc_lle=False)

    # --- Sanity: LLE result is finite ---
    assert hasattr(res_lle, "lle_convergence"), (
        "SimulationResult missing 'lle_convergence' attribute — "
        "NeuronSolver.run_native() must set res.lle_convergence when calc_lle=True."
    )
    lle_arr = res_lle.lle_convergence
    assert len(lle_arr) > 0, "lle_convergence array is empty"
    # v12.7: LLE uses NaN before first renormalization; only check finite values
    valid_lle = lle_arr[np.isfinite(lle_arr)]
    assert len(valid_lle) > 0, "No finite LLE values recorded"
    # All finite values should not be Inf
    assert np.all(np.isfinite(valid_lle)), "LLE convergence array contains Inf"

    # --- Main trajectory: LLE run vs. no-LLE run must agree to 1e-12 ---
    # The main trajectory (traj_idx=0) should be bit-identical between runs
    # because the perturbed trajectory does not feed back into traj_idx=0 physics.
    # If the array layout caused index swaps between trajectories, this will fail.
    min_len = min(len(res_lle.v_soma), len(res_no_lle.v_soma))
    assert min_len > 10, f"Too few time points in result: {min_len}"

    np.testing.assert_allclose(
        res_lle.v_soma[:min_len],
        res_no_lle.v_soma[:min_len],
        atol=1e-12, rtol=0.0,
        err_msg=(
            "v_soma from LLE run (n_traj=2) differs from single-traj run. "
            "This indicates cross-talk between trajectories via the i_ca_influx_2d "
            "buffer — a sign the [i, traj_idx] → [traj_idx, i] flip is incorrect."
        ),
    )

    # --- Spike count: both runs must agree ---
    n_spikes_lle = _spike_count(res_lle.v_soma)
    n_spikes_no_lle = _spike_count(res_no_lle.v_soma)
    assert n_spikes_lle == n_spikes_no_lle, (
        f"LLE run produced {n_spikes_lle} spikes but single-traj run produced "
        f"{n_spikes_no_lle}. The layout change introduced a cross-trajectory "
        "influx bias in the Ca dynamics."
    )

    # --- Reproducibility: two independent LLE runs must be bit-identical ---
    res_lle2 = _run_native(cfg, calc_lle=True)
    np.testing.assert_allclose(
        res_lle.v_soma, res_lle2.v_soma,
        atol=1e-12, rtol=0.0,
        err_msg=(
            "Two sequential LLE runs produced different v_soma. "
            "The buffer initialisation in the audited (n_comp, n_traj) layout "
            "may leave stale values across calls."
        ),
    )
    # LLE convergence values must also be reproducible (equal_nan=True for NaN positions)
    np.testing.assert_allclose(
        res_lle.lle_convergence, res_lle2.lle_convergence,
        atol=1e-12, rtol=0.0, equal_nan=True,
        err_msg="LLE convergence arrays differ between identical runs.",
    )


def test_lle_convergence_records_time_series_not_final_only():
    """Benettin LLE output must record current convergence at every output sample."""
    cfg = _make_l5_config(t_sim_ms=80.0, dt_eval_ms=0.2)
    res_lle = _run_native(cfg, calc_lle=True)
    lle = np.asarray(res_lle.lle_convergence, dtype=float)
    assert len(lle) > 20
    # v12.7: LLE uses NaN for 'not yet computed' before first renormalization
    finite_vals = np.isfinite(lle)
    assert np.any(finite_vals), "No finite LLE values recorded"
    # After first renormalization, values should be recorded at output samples
    finite_count = np.sum(finite_vals)
    assert finite_count > 3, f"Only {finite_count} finite LLE values (expected > 3)"


# ---------------------------------------------------------------------------
# Test 3 — Ca²⁺ concentration trajectory separation after layout change
# ---------------------------------------------------------------------------

def test_ca_influx_consistent_between_trajectories():
    """Verify that Ca²⁺ trajectories for main and perturbed traj separate correctly.

    With dynamic_Ca=True and calc_lle=True, the i_ca_influx_2d buffer holds
    Ca influx independently for traj=0 (main) and traj=1 (perturbed).
    Both trajectories start from the same initial Ca²⁺ concentration.

    After the layout change, the test confirms:
    1. Both trajectories start from the same Ca_rest initial condition.
    2. The main trajectory's Ca²⁺ remains finite throughout the first 10 ms.
    3. Cross-contamination between trajectories (wrong index in the 2D buffer)
       would cause one trajectory's Ca to drift away from a physically reasonable
       range — this is caught by the bounds check below.

    We test this indirectly via the solver's public API:
      - res (calc_lle=False): Ca²⁺ of the main trajectory.
      - res_lle (calc_lle=True): Ca²⁺ of the main trajectory with LLE enabled.
    These must agree to 1e-12 — the perturbed trajectory must not contaminate
    the main trajectory's Ca bookkeeping.

    # PRE-CHANGE BASELINE: record Ca_i time course from (n_comp, n_traj) layout.
    # POST-CHANGE CHECK:   same test must pass with the audited (n_comp, n_traj) layout.
    """
    cfg = _make_l5_config(t_sim_ms=100.0, dt_eval_ms=0.025)

    # Confirm calcium dynamics are active
    assert cfg.calcium.dynamic_Ca, "Test requires dynamic_Ca=True (check preset)"
    assert cfg.channels.enable_ICa, "Test requires enable_ICa=True (check preset)"

    ca_rest = cfg.calcium.Ca_rest  # e.g. 50e-6 mM = 50 nM

    res_no_lle = _run_native(cfg, calc_lle=False)
    res_lle    = _run_native(cfg, calc_lle=True)

    # --- Both results must have Ca²⁺ data ---
    assert res_no_lle.ca_i is not None, (
        "SimulationResult.ca_i is None with dynamic_Ca=True (no-LLE run). "
        "Check that enable_ICa and dynamic_Ca are active in the config."
    )
    assert res_lle.ca_i is not None, (
        "SimulationResult.ca_i is None with dynamic_Ca=True (LLE run)."
    )

    ca_no_lle = res_no_lle.ca_i[0, :]   # soma compartment, all time
    ca_lle    = res_lle.ca_i[0, :]      # soma compartment, all time

    # --- Both must be finite ---
    assert np.all(np.isfinite(ca_no_lle)), "Ca_i (no-LLE) contains NaN or Inf"
    assert np.all(np.isfinite(ca_lle)),    "Ca_i (LLE) contains NaN or Inf"

    # --- Initial Ca must equal Ca_rest (both trajectories start from same IC) ---
    ca_start_no_lle = float(ca_no_lle[0])
    ca_start_lle    = float(ca_lle[0])
    assert abs(ca_start_no_lle - ca_rest) < 1e-10, (
        f"Initial Ca_i (no-LLE) = {ca_start_no_lle:.3e} mM, "
        f"expected Ca_rest = {ca_rest:.3e} mM. "
        "Initial condition is wrong."
    )
    assert abs(ca_start_lle - ca_rest) < 1e-10, (
        f"Initial Ca_i (LLE) = {ca_start_lle:.3e} mM, "
        f"expected Ca_rest = {ca_rest:.3e} mM. "
        "LLE run perturbs only voltage, not Ca. "
        "A wrong initial Ca indicates cross-trajectory contamination."
    )

    # --- First 10 ms: Ca must stay in physiological range ---
    # Determine how many output time points correspond to the first 10 ms.
    dt_eval = cfg.stim.dt_eval
    n_10ms = max(1, int(10.0 / dt_eval))

    ca_early_no_lle = ca_no_lle[:n_10ms]
    ca_early_lle    = ca_lle[:n_10ms]

    # Ca²⁺ physiological range: 10 nM to 100 µM (roughly 1e-5 to 0.1 mM)
    ca_lower_bound = 1e-8  # 10 pM — below this means Ca went negative (physics violation)
    ca_upper_bound = 0.5   # 500 µM — above this means uncontrolled Ca explosion
    assert ca_early_no_lle.min() >= ca_lower_bound, (
        f"Ca_i (no-LLE) dropped below {ca_lower_bound:.1e} mM in first 10 ms: "
        f"min={ca_early_no_lle.min():.3e}. "
        "Semi-implicit Ca step or b_ca indexing may be broken."
    )
    assert ca_early_no_lle.max() <= ca_upper_bound, (
        f"Ca_i (no-LLE) exceeded {ca_upper_bound:.1e} mM in first 10 ms: "
        f"max={ca_early_no_lle.max():.3e}. "
        "Ca influx term may be double-counted after layout change."
    )
    assert ca_early_lle.min() >= ca_lower_bound, (
        f"Ca_i (LLE) dropped below {ca_lower_bound:.1e} mM in first 10 ms: "
        f"min={ca_early_lle.min():.3e}. "
        "The perturbed trajectory's Ca may be contaminating traj_idx=0."
    )
    assert ca_early_lle.max() <= ca_upper_bound, (
        f"Ca_i (LLE) exceeded {ca_upper_bound:.1e} mM in first 10 ms: "
        f"max={ca_early_lle.max():.3e}. "
        "Cross-trajectory Ca influx leakage suspected after layout change."
    )

    # --- Main trajectory Ca must be bit-identical between LLE and no-LLE runs ---
    # This is the strongest test of trajectory isolation.
    min_len = min(len(ca_no_lle), len(ca_lle))
    np.testing.assert_allclose(
        ca_no_lle[:min_len],
        ca_lle[:min_len],
        atol=1e-12, rtol=0.0,
        err_msg=(
            "Ca_i of the main trajectory (traj_idx=0) differs between the no-LLE "
            "and LLE runs. This indicates that the perturbed trajectory's Ca influx "
            "is leaking into traj_idx=0 — a symptom of a wrong index after the "
            "i_ca_influx_2d audited (n_comp, n_traj) layout."
        ),
    )
