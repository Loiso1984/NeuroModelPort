"""Test: Voltage ranges for key presets (unit system validation)."""

import pytest
import numpy as np
from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver


@pytest.mark.parametrize("preset_name, v_min_bound, v_max_bound", [
    ("A: Squid Giant Axon (HH 1952)", -100, 50),
    ("B: Pyramidal L5 (Mainen 1996)", -95, 60),
    ("C: FS Interneuron (Wang-Buzsaki)", -100, 60),
])
def test_preset_voltage_range(preset_name, v_min_bound, v_max_bound):
    """Each preset should produce voltages within physiological bounds."""
    cfg = FullModelConfig()
    apply_preset(cfg, preset_name)
    cfg.stim.t_sim = 50.0

    solver = NeuronSolver(cfg)
    result = solver.run_single()

    v_min, v_max = result.v_soma.min(), result.v_soma.max()
    assert v_min >= v_min_bound, f"{preset_name}: V_min={v_min:.1f} below {v_min_bound}"
    assert v_max <= v_max_bound, f"{preset_name}: V_max={v_max:.1f} above {v_max_bound}"
