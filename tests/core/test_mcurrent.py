"""Test: M-type potassium current (I_M, KCNQ/Kv7) integration — Stage 5.2.

Verifies that enabling enable_IM produces valid simulation with:
- Finite voltage traces
- IM current extracted in post-processing
- Spike-frequency adaptation effect (fewer spikes with I_M enabled)
- All three Jacobian modes work
"""

import numpy as np
import pytest
from core.models import FullModelConfig
from core.solver import NeuronSolver


def _make_im_config(jacobian_mode: str = "sparse_fd") -> FullModelConfig:
    """Single-compartment config with M-current enabled."""
    cfg = FullModelConfig()
    cfg.morphology.single_comp = True
    cfg.channels.gNa_max = 120.0
    cfg.channels.gK_max = 36.0
    cfg.channels.gL = 0.3
    cfg.channels.ENa = 50.0
    cfg.channels.EK = -77.0
    cfg.channels.EL = -54.387
    # Enable M-current
    cfg.channels.enable_IM = True
    cfg.channels.gIM_max = 0.5  # Strong enough to see adaptation effect
    # Long pulse to observe adaptation
    cfg.stim.t_sim = 200.0
    cfg.stim.stim_type = "pulse"
    cfg.stim.Iext = 15.0
    cfg.stim.pulse_start = 10.0
    cfg.stim.pulse_dur = 180.0
    cfg.stim.jacobian_mode = jacobian_mode
    return cfg


def test_im_simulation_runs():
    """M-current simulation should complete and produce finite results."""
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


def test_im_reduces_firing_rate():
    """M-current should reduce spike count (spike-frequency adaptation)."""
    threshold = -20.0

    # Without I_M
    cfg_off = _make_im_config()
    cfg_off.channels.enable_IM = False
    res_off = NeuronSolver(cfg_off).run_single()
    v_off = res_off.v_soma
    n_off = int(np.sum((v_off[:-1] < threshold) & (v_off[1:] >= threshold)))

    # With I_M
    cfg_on = _make_im_config()
    res_on = NeuronSolver(cfg_on).run_single()
    v_on = res_on.v_soma
    n_on = int(np.sum((v_on[:-1] < threshold) & (v_on[1:] >= threshold)))

    assert n_off > 0, f"Baseline (no I_M) should produce spikes, got {n_off}"
    assert n_on < n_off, (
        f"I_M should reduce spike count: {n_on} >= {n_off}"
    )


@pytest.mark.parametrize("jac_mode", ["dense_fd", "sparse_fd", "analytic_sparse"])
def test_im_jacobian_modes(jac_mode):
    """All Jacobian modes should work with M-current enabled."""
    cfg = _make_im_config(jacobian_mode=jac_mode)
    cfg.stim.t_sim = 50.0  # Short sim for speed
    result = NeuronSolver(cfg).run_single()

    assert np.all(np.isfinite(result.v_soma)), f"V_soma NaN/Inf with {jac_mode}"
    assert "IM" in result.currents
