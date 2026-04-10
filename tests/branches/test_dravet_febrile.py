"""
Test Dravet Syndrome febrile mode verification.

Verifies that switching to febrile mode (fever) reduces spiking compared to baseline
due to thermal hyper-sensitivity of Na channels.
"""
import pytest
import numpy as np
from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver
from core.analysis import detect_spikes


def test_dravet_febrile_mode():
    """Test that Dravet febrile mode reduces spiking compared to baseline."""
    # Baseline mode
    cfg_baseline = FullModelConfig()
    cfg_baseline.preset_modes.dravet_mode = "baseline"
    apply_preset(cfg_baseline, "Pathology: Dravet Syndrome (SCN1A LOF)")
    cfg_baseline.stim.Iext = 60.0  # Increase stimulus to get more spikes
    
    solver_baseline = NeuronSolver(cfg_baseline)
    result_baseline = solver_baseline.run_single()
    t_baseline = np.asarray(result_baseline.t, dtype=float)
    v_baseline = np.asarray(result_baseline.v_soma, dtype=float)
    
    kwargs_baseline = {
        'threshold': -20.0,
        'refractory_ms': 2.0,
        'prominence': 10.0
    }
    _, spike_times_baseline, _ = detect_spikes(v_baseline, t_baseline, **kwargs_baseline)
    n_spikes_baseline = len(spike_times_baseline)
    
    # Febrile mode
    cfg_febrile = FullModelConfig()
    cfg_febrile.preset_modes.dravet_mode = "febrile"
    apply_preset(cfg_febrile, "Pathology: Dravet Syndrome (SCN1A LOF)")
    cfg_febrile.stim.Iext = 60.0  # Same stimulus for fair comparison
    
    solver_febrile = NeuronSolver(cfg_febrile)
    result_febrile = solver_febrile.run_single()
    t_febrile = np.asarray(result_febrile.t, dtype=float)
    v_febrile = np.asarray(result_febrile.v_soma, dtype=float)
    
    kwargs_febrile = {
        'threshold': -20.0,
        'refractory_ms': 2.0,
        'prominence': 10.0
    }
    _, spike_times_febrile, _ = detect_spikes(v_febrile, t_febrile, **kwargs_febrile)
    n_spikes_febrile = len(spike_times_febrile)
    
    print(f"\nDravet Baseline spikes: {n_spikes_baseline}")
    print(f"Dravet Febrile spikes: {n_spikes_febrile}")
    if n_spikes_baseline > 0:
        print(f"Reduction: {100 * (1 - n_spikes_febrile / n_spikes_baseline):.1f}%")
    
    # Febrile mode should reduce spiking (at least 10% reduction if baseline has >2 spikes)
    if n_spikes_baseline >= 2:
        assert n_spikes_febrile < n_spikes_baseline * 0.9, \
            f"Febrile mode should reduce spiking: baseline={n_spikes_baseline}, febrile={n_spikes_febrile}"
    
    # Verify temperature is set correctly
    assert cfg_febrile.env.T_celsius == 40.0, "Febrile mode should set T_celsius=40.0"
    assert cfg_febrile.env.Q10_Na == 3.0, "Febrile mode should set Q10_Na=3.0"
    assert cfg_febrile.channels.gNa_max == 42.0, "Febrile mode should set gNa_max=42.0"


if __name__ == "__main__":
    test_dravet_febrile_mode()
    print("\n✅ Dravet febrile mode test passed")
