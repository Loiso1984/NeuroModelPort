"""Test: Load L5 config and run minimal simulation via NeuronSolver."""

import numpy as np
from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver


def test_l5_solver_basic():
    """L5 Pyramidal preset should produce a valid simulation result."""
    cfg = FullModelConfig()
    apply_preset(cfg, "B: Pyramidal L5 (Mainen 1996)")
    cfg.stim.t_sim = 50.0

    solver = NeuronSolver(cfg)
    result = solver.run_single()

    assert len(result.t) > 10, "Too few timepoints"
    assert result.v_soma.min() < -50, f"V_min too high: {result.v_soma.min():.1f}"
    assert result.v_soma.max() > -80, f"V_max too low: {result.v_soma.max():.1f}"
    if result.ca_i is not None:
        assert np.all(np.isfinite(result.ca_i)), "Ca_i contains NaN/Inf"
