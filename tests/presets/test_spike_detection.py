"""Test: L5 Pyramidal spike detection validation."""

import numpy as np
from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver


def test_l5_produces_spikes():
    """L5 Pyramidal with default stimulus must generate spikes."""
    cfg = FullModelConfig()
    apply_preset(cfg, "B: Pyramidal L5 (Mainen 1996)")

    solver = NeuronSolver(cfg)
    result = solver.run_single()

    v_soma = result.v_soma
    spike_threshold = -20

    assert v_soma.max() > spike_threshold, (
        f"No spikes detected: V_max = {v_soma.max():.1f} mV"
    )
    assert v_soma.min() < -50, f"Resting potential too high: {v_soma.min():.1f}"
