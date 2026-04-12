"""Test: M-type potassium current (I_M, KCNQ2/3) integration — Stage 5.2.

Literature source: Yamada, Koch & Adams 1989
  (Methods in Neuronal Modeling, Koch & Segev, MIT Press, pp 97-133)

Verifies:
- Simulation stability with IM enabled
- IM current extracted in post-processing
- Correct gate kinetics (V½ ≈ -35 mV, τ_w ≈ 151 ms at V½)
- Coexistence with other channels
- All three Jacobian modes
- Spike-frequency adaptation effect (IM reduces late firing rate)
"""

import numpy as np
import pytest
from core.models import FullModelConfig
from core.solver import NeuronSolver
from core.kinetics import aw_IM, bw_IM


def _make_im_config(jacobian_mode: str = "sparse_fd") -> FullModelConfig:
    """Single-compartment config with M-type K enabled."""
    cfg = FullModelConfig()
    cfg.morphology.single_comp = True
    cfg.channels.gNa_max = 120.0
    cfg.channels.gK_max = 36.0
    cfg.channels.gL = 0.3
    cfg.channels.ENa = 50.0
    cfg.channels.EK = -77.0
    cfg.channels.EL = -65.0
    # Enable M-type K
    cfg.channels.enable_IM = True
    cfg.channels.gIM_max = 0.5
    # Short simulation
    cfg.stim.t_sim = 100.0
    cfg.stim.stim_type = "const"
    cfg.stim.Iext = 10.0
    cfg.stim.jacobian_mode = jacobian_mode
    return cfg


# ─── Basic integration tests ───────────────────────────────────────

def test_im_simulation_runs():
    """M-type K simulation should complete without error and produce finite results."""
    cfg = _make_im_config()
    result = NeuronSolver(cfg).run_single()

    assert len(result.t) > 10
    assert np.all(np.isfinite(result.v_soma)), "V_soma contains NaN/Inf"


def test_im_current_extracted():
    """Post-processing should extract IM current density."""
    cfg = _make_im_config()
    result = NeuronSolver(cfg).run_single()

    assert "IM" in result.currents, "IM current not in post-processed currents"
    im = result.currents["IM"]
    assert len(im) == len(result.t)
    assert np.all(np.isfinite(im)), "IM current contains NaN/Inf"


def test_im_current_is_outward():
    """IM should be outward (positive) during depolarization above E_K."""
    cfg = _make_im_config()
    result = NeuronSolver(cfg).run_single()

    im = result.currents["IM"]
    # During spiking, V > E_K, so IM should be predominantly positive (outward)
    # Allow some negative values during deep AHP
    mean_im = np.mean(im[len(im)//2:])
    assert mean_im > 0, f"Mean IM current should be outward (positive), got {mean_im}"


# ─── Kinetics validation ───────────────────────────────────────────

def test_im_kinetics_half_activation():
    """V½ of M-type activation should be near -35 mV (Yamada 1989)."""
    V_test = np.array([-35.0])
    a = float(aw_IM(V_test)[0])
    b = float(bw_IM(V_test)[0])
    w_inf = a / (a + b)
    # At V½, w_inf should be 0.5
    assert abs(w_inf - 0.5) < 0.02, f"w_inf at V½=-35 mV should be ~0.5, got {w_inf}"


def test_im_kinetics_time_constant():
    """τ_w at V½ should be ~151 ms (Yamada 1989)."""
    V_test = np.array([-35.0])
    a = float(aw_IM(V_test)[0])
    b = float(bw_IM(V_test)[0])
    tau_w = 1.0 / (a + b)
    # Expected: 1000 / (3.3 * 2) ≈ 151.5 ms
    assert 140 < tau_w < 160, f"τ_w at V½ should be ~151 ms, got {tau_w:.1f} ms"


def test_im_kinetics_slow_at_rest():
    """τ_w at resting potential (-65 mV) should be slow (>50 ms)."""
    V_test = np.array([-65.0])
    a = float(aw_IM(V_test)[0])
    b = float(bw_IM(V_test)[0])
    tau_w = 1.0 / (a + b)
    assert tau_w > 50, f"τ_w at -65 mV should be >50 ms (slow channel), got {tau_w:.1f} ms"


def test_im_kinetics_gate_closed_at_rest():
    """w_inf at resting potential (-65 mV) should be near 0 (channel closed)."""
    V_test = np.array([-65.0])
    a = float(aw_IM(V_test)[0])
    b = float(bw_IM(V_test)[0])
    w_inf = a / (a + b)
    assert w_inf < 0.1, f"w_inf at -65 mV should be near 0, got {w_inf:.3f}"


def test_im_kinetics_gate_open_depolarized():
    """w_inf at depolarized voltage (0 mV) should be near 1 (channel open)."""
    V_test = np.array([0.0])
    a = float(aw_IM(V_test)[0])
    b = float(bw_IM(V_test)[0])
    w_inf = a / (a + b)
    assert w_inf > 0.9, f"w_inf at 0 mV should be near 1, got {w_inf:.3f}"


# ─── Channel coexistence ───────────────────────────────────────────

def test_im_with_all_channels():
    """IM should coexist with all other channels without instability."""
    cfg = _make_im_config()
    cfg.channels.enable_Ih = True
    cfg.channels.gIh_max = 0.02
    cfg.channels.enable_ICa = True
    cfg.channels.gCa_max = 0.5
    cfg.channels.enable_IA = True
    cfg.channels.gA_max = 0.4
    cfg.channels.enable_SK = True
    cfg.channels.gSK_max = 2.0
    cfg.channels.enable_ITCa = True
    cfg.channels.gTCa_max = 1.0
    cfg.calcium.dynamic_Ca = True
    cfg.calcium.Ca_rest = 5e-5
    cfg.calcium.Ca_ext = 2.0
    cfg.calcium.tau_Ca = 200.0
    cfg.calcium.B_Ca = 1e-5

    result = NeuronSolver(cfg).run_single()

    assert np.all(np.isfinite(result.v_soma)), "V_soma NaN/Inf with all channels"
    assert "IM" in result.currents


def test_im_without_other_optional_channels():
    """IM should work with only core HH (no Ih, ICa, IA, SK, ITCa)."""
    cfg = _make_im_config()
    cfg.channels.enable_Ih = False
    cfg.channels.enable_ICa = False
    cfg.channels.enable_IA = False
    cfg.channels.enable_SK = False
    cfg.channels.enable_ITCa = False

    result = NeuronSolver(cfg).run_single()

    assert np.all(np.isfinite(result.v_soma))
    assert "IM" in result.currents
    assert "Ih" not in result.currents
    assert "ICa" not in result.currents


# ─── Jacobian modes ────────────────────────────────────────────────

@pytest.mark.parametrize("jac_mode", ["dense_fd", "sparse_fd", "analytic_sparse"])
def test_im_jacobian_modes(jac_mode):
    """All Jacobian modes should work with IM enabled."""
    cfg = _make_im_config(jacobian_mode=jac_mode)
    result = NeuronSolver(cfg).run_single()

    assert np.all(np.isfinite(result.v_soma)), f"V_soma NaN/Inf with {jac_mode}"
    assert "IM" in result.currents


# ─── Physiological effect: spike-frequency adaptation ──────────────

def test_im_causes_spike_frequency_adaptation():
    """Enabling IM should reduce spike count compared to baseline HH.

    The M-current is a slow outward K+ current that builds up during
    sustained depolarization, progressively opposing further spiking.
    This is the hallmark spike-frequency adaptation mechanism.
    """
    # Baseline: HH only
    cfg_base = FullModelConfig()
    cfg_base.morphology.single_comp = True
    cfg_base.stim.t_sim = 200.0
    cfg_base.stim.stim_type = "pulse"
    cfg_base.stim.Iext = 35.0
    cfg_base.stim.pulse_start = 10.0
    cfg_base.stim.pulse_dur = 180.0
    cfg_base.stim.jacobian_mode = "sparse_fd"
    res_base = NeuronSolver(cfg_base).run_single()

    # With IM
    cfg_im = FullModelConfig()
    cfg_im.morphology.single_comp = True
    cfg_im.stim.t_sim = 200.0
    cfg_im.stim.stim_type = "pulse"
    cfg_im.stim.Iext = 35.0
    cfg_im.stim.pulse_start = 10.0
    cfg_im.stim.pulse_dur = 180.0
    cfg_im.stim.jacobian_mode = "sparse_fd"
    cfg_im.channels.enable_IM = True
    cfg_im.channels.gIM_max = 1.0  # Strong IM for clear effect
    res_im = NeuronSolver(cfg_im).run_single()

    # Count threshold crossings (simple spike count)
    v_base = res_base.v_soma
    v_im = res_im.v_soma
    threshold = -20.0

    crossings_base = np.sum((v_base[:-1] < threshold) & (v_base[1:] >= threshold))
    crossings_im = np.sum((v_im[:-1] < threshold) & (v_im[1:] >= threshold))

    assert crossings_im < crossings_base, (
        f"IM should reduce spike count: baseline={crossings_base}, with IM={crossings_im}"
    )
