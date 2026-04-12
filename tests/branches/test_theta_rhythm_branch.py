"""
Test CA1 Theta Rhythm - Synaptic Train Verification

Tests that the CA1 preset correctly responds to 7Hz rhythmic synaptic input
to produce theta rhythm firing. This is a system effect that emerges from
external rhythmic stimulation, not constant current injection.
"""

import numpy as np
import pytest
from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver


def _spike_times(v: np.ndarray, t: np.ndarray, threshold: float = -20.0) -> np.ndarray:
    """Extract spike times from voltage trace."""
    idx = np.where((v[:-1] < threshold) & (v[1:] >= threshold))[0] + 1
    if len(idx) == 0:
        return np.array([], dtype=float)
    st = t[idx]
    # Filter: minimum 1ms between spikes
    keep = [0]
    for i in range(1, len(st)):
        if st[i] - st[keep[-1]] >= 1.0:
            keep.append(i)
    return st[keep]


def test_ca1_theta_dual_stimulation_config():
    """Test that CA1 preset uses a physiology-first theta surrogate protocol."""
    cfg = FullModelConfig()
    apply_preset(cfg, "L: Hippocampal CA1 Pyramidal (Adapting)")

    assert cfg.dual_stimulation is not None, "CA1 should have dual stimulation configured"
    assert cfg.dual_stimulation.enabled is True, "Dual stimulation should be enabled"

    assert cfg.morphology.single_comp is False
    assert cfg.stim_location.location == "dendritic_filtered"
    assert cfg.stim.stim_type == "AMPA"
    assert cfg.stim.Iext == 2.0
    assert cfg.stim.synaptic_train_type == "regular"
    assert cfg.stim.synaptic_train_freq_hz == 7.0

    assert cfg.dual_stimulation.secondary_stim_type == "GABAA"
    assert cfg.dual_stimulation.secondary_Iext == 0.8
    assert cfg.dual_stimulation.secondary_train_type == "regular"
    assert cfg.dual_stimulation.secondary_train_freq_hz == 7.0
    assert cfg.dual_stimulation.secondary_train_duration_ms == 5000.0
    assert cfg.dual_stimulation.secondary_location == "soma"
    assert cfg.stim.noise_sigma == 0.25


def test_ca1_theta_firing_response():
    """Test that CA1 neuron fires in response to theta-paced distal input."""
    cfg = FullModelConfig()
    apply_preset(cfg, "L: Hippocampal CA1 Pyramidal (Adapting)")

    res = NeuronSolver(cfg).run_single()
    st = _spike_times(res.v_soma, res.t)

    assert len(st) >= 5, f"Expected at least 5 spikes with theta surrogate, got {len(st)}"

    if len(st) > 1:
        isi = np.diff(st)
        burst_starts = [st[0]]
        for i, dt in enumerate(isi):
            if dt >= 50.0:
                burst_starts.append(st[i + 1])
        if len(burst_starts) > 1:
            burst_intervals = np.diff(np.asarray(burst_starts))
            mean_burst_interval = np.mean(burst_intervals)
            assert 80 <= mean_burst_interval <= 220, (
                f"Expected theta-paced burst spacing, got {mean_burst_interval:.1f}ms"
            )


def test_ca1_theta_physiological_channels():
    """Test that CA1 has correct channel configuration for theta resonance."""
    cfg = FullModelConfig()
    apply_preset(cfg, "L: Hippocampal CA1 Pyramidal (Adapting)")
    
    # Theta resonance requires IA and Ih
    assert cfg.channels.enable_IA == True
    assert cfg.channels.enable_Ih == True
    
    # Check physiological densities (Storm 1990, Magee 1998)
    assert 0.3 <= cfg.channels.gA_max <= 0.5  # IA density
    assert 0.01 <= cfg.channels.gIh_max <= 0.02  # Ih density
    
    # Current CA1 model includes calcium-dependent adaptation terms.
    assert cfg.channels.enable_SK is True
    assert cfg.channels.enable_ICa is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
