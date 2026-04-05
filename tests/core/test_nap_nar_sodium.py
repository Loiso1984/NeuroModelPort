"""Test: Persistent (I_NaP) and Resurgent (I_NaR) sodium currents — Stage 5.3.

I_NaP: Magistretti & Alonso 1999, J Gen Physiol 114:491-509
I_NaR: Raman & Bean 2001, Biophys J 80:729-737 (phenomenological HH adaptation)

Verifies:
- Simulation stability with NaP/NaR enabled
- Currents extracted in post-processing
- Kinetics validation (V½, τ)
- All Jacobian modes
- NaP enhances subthreshold excitability
- Coexistence with all other channels
"""

import numpy as np
import pytest
from core.models import FullModelConfig
from core.solver import NeuronSolver
from core.kinetics import ax_NaP, bx_NaP, ay_NaR, by_NaR, aj_NaR, bj_NaR


def _make_nap_config(jac="sparse_fd") -> FullModelConfig:
    cfg = FullModelConfig()
    cfg.morphology.single_comp = True
    cfg.channels.enable_NaP = True
    cfg.channels.gNaP_max = 0.1
    cfg.stim.t_sim = 80.0
    cfg.stim.stim_type = "const"
    cfg.stim.Iext = 8.0
    cfg.stim.jacobian_mode = jac
    return cfg


def _make_nar_config(jac="sparse_fd") -> FullModelConfig:
    cfg = FullModelConfig()
    cfg.morphology.single_comp = True
    cfg.channels.enable_NaR = True
    cfg.channels.gNaR_max = 0.2
    cfg.stim.t_sim = 80.0
    cfg.stim.stim_type = "const"
    cfg.stim.Iext = 10.0
    cfg.stim.jacobian_mode = jac
    return cfg


# ─── NaP basic tests ──────────────────────────────────────────────

def test_nap_simulation_runs():
    result = NeuronSolver(_make_nap_config()).run_single()
    assert np.all(np.isfinite(result.v_soma))


def test_nap_current_extracted():
    result = NeuronSolver(_make_nap_config()).run_single()
    assert "NaP" in result.currents
    assert np.all(np.isfinite(result.currents["NaP"]))


def test_nap_kinetics_half_activation():
    """V½ of NaP should be near -52 mV (Magistretti 1999)."""
    V = np.array([-52.0])
    a, b = float(ax_NaP(V)[0]), float(bx_NaP(V)[0])
    x_inf = a / (a + b)
    assert abs(x_inf - 0.5) < 0.02, f"x_inf at -52 mV: {x_inf}"


def test_nap_kinetics_fast_tau():
    """τ_x at V½ should be very fast (<1 ms)."""
    V = np.array([-52.0])
    a, b = float(ax_NaP(V)[0]), float(bx_NaP(V)[0])
    tau = 1.0 / (a + b)
    assert tau < 1.0, f"τ_x should be <1 ms, got {tau:.3f} ms"


def test_nap_closed_at_rest():
    """x_inf at -65 mV should be small (channel mostly closed at rest)."""
    V = np.array([-65.0])
    a, b = float(ax_NaP(V)[0]), float(bx_NaP(V)[0])
    x_inf = a / (a + b)
    assert x_inf < 0.1, f"x_inf at -65 mV should be <0.1, got {x_inf:.3f}"


def test_nap_enhances_excitability():
    """NaP should increase spike count (subthreshold amplification)."""
    # Baseline
    cfg_base = FullModelConfig()
    cfg_base.morphology.single_comp = True
    cfg_base.stim.t_sim = 200.0
    cfg_base.stim.Iext = 7.0  # near threshold
    cfg_base.stim.jacobian_mode = "sparse_fd"
    res_base = NeuronSolver(cfg_base).run_single()

    # With NaP
    cfg_nap = FullModelConfig()
    cfg_nap.morphology.single_comp = True
    cfg_nap.stim.t_sim = 200.0
    cfg_nap.stim.Iext = 7.0
    cfg_nap.stim.jacobian_mode = "sparse_fd"
    cfg_nap.channels.enable_NaP = True
    cfg_nap.channels.gNaP_max = 0.2
    res_nap = NeuronSolver(cfg_nap).run_single()

    thr = -20.0
    spikes_base = np.sum((res_base.v_soma[:-1] < thr) & (res_base.v_soma[1:] >= thr))
    spikes_nap = np.sum((res_nap.v_soma[:-1] < thr) & (res_nap.v_soma[1:] >= thr))
    assert spikes_nap >= spikes_base, (
        f"NaP should enhance excitability: base={spikes_base}, NaP={spikes_nap}"
    )


# ─── NaR basic tests ──────────────────────────────────────────────

def test_nar_simulation_runs():
    result = NeuronSolver(_make_nar_config()).run_single()
    assert np.all(np.isfinite(result.v_soma))


def test_nar_current_extracted():
    result = NeuronSolver(_make_nar_config()).run_single()
    assert "NaR" in result.currents
    assert np.all(np.isfinite(result.currents["NaR"]))


def test_nar_kinetics_activation_v_half():
    """NaR activation V½ should be near -48 mV."""
    V = np.array([-48.0])
    a, b = float(ay_NaR(V)[0]), float(by_NaR(V)[0])
    y_inf = a / (a + b)
    assert abs(y_inf - 0.5) < 0.02, f"y_inf at -48 mV: {y_inf}"


def test_nar_kinetics_block_v_half():
    """NaR block/inactivation V½ should be near -33 mV."""
    V = np.array([-33.0])
    a, b = float(aj_NaR(V)[0]), float(bj_NaR(V)[0])
    j_inf = a / (a + b)
    assert abs(j_inf - 0.5) < 0.02, f"j_inf at -33 mV: {j_inf}"


def test_nar_window_current_peaks_at_intermediate_v():
    """Product y_inf * j_inf should peak around -40 mV (window current)."""
    voltages = np.arange(-80.0, 20.0, 1.0)
    products = []
    for v_val in voltages:
        V = np.array([v_val])
        ay, by = float(ay_NaR(V)[0]), float(by_NaR(V)[0])
        aj, bj = float(aj_NaR(V)[0]), float(bj_NaR(V)[0])
        y_inf = ay / (ay + by)
        j_inf = aj / (aj + bj)
        products.append(y_inf * j_inf)
    peak_idx = np.argmax(products)
    peak_v = voltages[peak_idx]
    assert -50 < peak_v < -30, f"Window current should peak ~-40 mV, got {peak_v}"


# ─── Jacobian modes ───────────────────────────────────────────────

@pytest.mark.parametrize("jac_mode", ["dense_fd", "sparse_fd", "analytic_sparse"])
def test_nap_jacobian_modes(jac_mode):
    result = NeuronSolver(_make_nap_config(jac=jac_mode)).run_single()
    assert np.all(np.isfinite(result.v_soma))
    assert "NaP" in result.currents


@pytest.mark.parametrize("jac_mode", ["dense_fd", "sparse_fd", "analytic_sparse"])
def test_nar_jacobian_modes(jac_mode):
    result = NeuronSolver(_make_nar_config(jac=jac_mode)).run_single()
    assert np.all(np.isfinite(result.v_soma))
    assert "NaR" in result.currents


# ─── Coexistence ──────────────────────────────────────────────────

def test_nap_nar_with_all_channels():
    """NaP + NaR should coexist with all other channels."""
    cfg = FullModelConfig()
    cfg.morphology.single_comp = True
    cfg.stim.t_sim = 60.0
    cfg.stim.Iext = 10.0
    cfg.stim.jacobian_mode = "sparse_fd"
    cfg.channels.enable_Ih = True
    cfg.channels.enable_ICa = True
    cfg.channels.enable_IA = True
    cfg.channels.enable_SK = True
    cfg.channels.enable_ITCa = True
    cfg.channels.enable_IM = True
    cfg.channels.enable_NaP = True
    cfg.channels.enable_NaR = True
    cfg.calcium.dynamic_Ca = True
    cfg.calcium.Ca_rest = 5e-5
    cfg.calcium.Ca_ext = 2.0
    cfg.calcium.tau_Ca = 200.0
    cfg.calcium.B_Ca = 1e-5
    result = NeuronSolver(cfg).run_single()

    assert np.all(np.isfinite(result.v_soma))
    assert "NaP" in result.currents
    assert "NaR" in result.currents
    assert "IM" in result.currents
    assert "ITCa" in result.currents
