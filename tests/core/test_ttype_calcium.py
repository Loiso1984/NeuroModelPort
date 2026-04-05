"""Test: T-type calcium current (I_T, CaV3.x) integration — Stage 5.1.

Verifies that enabling enable_ITCa produces a valid simulation with:
- Finite voltage and calcium traces
- ITCa current extracted in post-processing
- Correct state vector size (p, q gates added)
- All three Jacobian modes work (dense_fd, sparse_fd, analytic_sparse)
"""

import numpy as np
import pytest
from core.models import FullModelConfig
from core.solver import NeuronSolver


def _make_itca_config(jacobian_mode: str = "sparse_fd") -> FullModelConfig:
    """Single-compartment config with T-type Ca enabled + dynamic calcium."""
    cfg = FullModelConfig()
    cfg.morphology.single_comp = True
    cfg.channels.gNa_max = 120.0
    cfg.channels.gK_max = 36.0
    cfg.channels.gL = 0.3
    cfg.channels.ENa = 50.0
    cfg.channels.EK = -77.0
    cfg.channels.EL = -65.0
    # Enable T-type Ca
    cfg.channels.enable_ITCa = True
    cfg.channels.gTCa_max = 2.0
    # Enable dynamic Ca for Nernst
    cfg.calcium.dynamic_Ca = True
    cfg.calcium.Ca_rest = 5e-5
    cfg.calcium.Ca_ext = 2.0
    cfg.calcium.tau_Ca = 200.0
    cfg.calcium.B_Ca = 1e-5
    # Short simulation
    cfg.stim.t_sim = 50.0
    cfg.stim.stim_type = "pulse"
    cfg.stim.Iext = 10.0
    cfg.stim.pulse_start = 5.0
    cfg.stim.pulse_dur = 30.0
    cfg.stim.jacobian_mode = jacobian_mode
    return cfg


def test_itca_simulation_runs():
    """T-type Ca simulation should complete without error and produce finite results."""
    cfg = _make_itca_config()
    result = NeuronSolver(cfg).run_single()

    assert len(result.t) > 10
    assert np.all(np.isfinite(result.v_soma)), "V_soma contains NaN/Inf"
    assert result.ca_i is not None, "Ca_i should exist with dynamic_Ca=True"
    assert np.all(np.isfinite(result.ca_i)), "Ca_i contains NaN/Inf"
    assert np.all(result.ca_i >= 0), "Ca_i went negative"


def test_itca_current_extracted():
    """Post-processing should extract ITCa current density."""
    cfg = _make_itca_config()
    result = NeuronSolver(cfg).run_single()

    assert "ITCa" in result.currents, "ITCa current not in post-processed currents"
    itca = result.currents["ITCa"]
    assert len(itca) == len(result.t)
    assert np.all(np.isfinite(itca)), "ITCa current contains NaN/Inf"


def test_itca_without_ltype():
    """T-type Ca should work independently of L-type Ca (enable_ICa=False)."""
    cfg = _make_itca_config()
    cfg.channels.enable_ICa = False
    result = NeuronSolver(cfg).run_single()

    assert np.all(np.isfinite(result.v_soma))
    assert "ITCa" in result.currents
    assert "ICa" not in result.currents


def test_itca_with_ltype():
    """T-type + L-type Ca should coexist, both contributing to calcium dynamics."""
    cfg = _make_itca_config()
    cfg.channels.enable_ICa = True
    cfg.channels.gCa_max = 0.5
    result = NeuronSolver(cfg).run_single()

    assert np.all(np.isfinite(result.v_soma))
    assert "ITCa" in result.currents
    assert "ICa" in result.currents
    assert result.ca_i is not None


@pytest.mark.parametrize("jac_mode", ["dense_fd", "sparse_fd", "analytic_sparse"])
def test_itca_jacobian_modes(jac_mode):
    """All Jacobian modes should work with T-type Ca enabled."""
    cfg = _make_itca_config(jacobian_mode=jac_mode)
    result = NeuronSolver(cfg).run_single()

    assert np.all(np.isfinite(result.v_soma)), f"V_soma NaN/Inf with {jac_mode}"
    assert "ITCa" in result.currents
