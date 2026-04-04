"""Test: Squid Giant Axon (HH 1952) — golden standard reference."""

import numpy as np
from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver


def test_squid_generates_spikes():
    """Squid HH 1952 with const stimulus must produce action potentials."""
    cfg = FullModelConfig()
    apply_preset(cfg, "A: Squid Giant Axon (HH 1952)")
    cfg.stim.stim_type = 'const'

    solver = NeuronSolver(cfg)
    result = solver.run_single()

    v_soma = result.v_soma
    spike_threshold = -20

    # Must have voltage excursions above threshold
    assert v_soma.max() > spike_threshold, (
        f"No spikes: V_max = {v_soma.max():.1f} mV, expected > {spike_threshold}"
    )

    # Voltage range must be physiological for squid axon
    assert v_soma.min() >= -100, f"V_min unrealistic: {v_soma.min():.1f}"
    assert v_soma.max() <= 60, f"V_max unrealistic: {v_soma.max():.1f}"
