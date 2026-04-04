"""Test: All presets should run without errors and produce finite voltages."""

import pytest
import numpy as np
from core.models import FullModelConfig
from core.presets import get_preset_names, apply_preset
from core.solver import NeuronSolver


@pytest.mark.parametrize("preset_name", get_preset_names())
def test_preset_runs_and_finite(preset_name):
    """Each preset must run to completion with finite voltage output."""
    cfg = FullModelConfig()
    apply_preset(cfg, preset_name)
    cfg.stim.stim_type = 'const'
    cfg.stim.t_sim = 50.0

    solver = NeuronSolver(cfg)
    result = solver.run_single()

    assert len(result.t) > 0, f"{preset_name}: empty result"
    assert np.all(np.isfinite(result.v_soma)), f"{preset_name}: NaN/Inf in v_soma"
    assert result.v_soma.max() < 80, f"{preset_name}: V_max={result.v_soma.max():.1f} unrealistic"
    assert result.v_soma.min() > -120, f"{preset_name}: V_min={result.v_soma.min():.1f} unrealistic"
