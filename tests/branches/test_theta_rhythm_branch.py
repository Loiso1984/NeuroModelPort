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
    """Test that CA1 preset has correct dual stimulation configuration for theta rhythm."""
    cfg = FullModelConfig()
    apply_preset(cfg, "L: Hippocampal CA1 Pyramidal (Adapting)")
    
    # Verify dual stimulation is enabled
    assert cfg.dual_stimulation is not None, "CA1 should have dual stimulation configured"
    assert cfg.dual_stimulation.enabled == True, "Dual stimulation should be enabled"
    
    # Verify primary stimulus: weak tonic current for readiness
    assert cfg.stim.stim_type == "const"
    assert cfg.stim.Iext == 2.0
    
    # Verify secondary stimulus: AMPA at 7Hz (theta)
    assert cfg.dual_stimulation.secondary_stim_type == "AMPA"
    assert cfg.dual_stimulation.secondary_Iext == 5.0
    assert cfg.dual_stimulation.secondary_train_type == "regular"
    assert cfg.dual_stimulation.secondary_train_freq_hz == 7.0
    assert cfg.dual_stimulation.secondary_train_duration_ms == 5000.0
    
    # Verify stochastic facilitation
    assert cfg.stim.noise_sigma == 0.8


def test_ca1_theta_firing_response():
    """Test that CA1 neuron fires in response to dual-drive stimulation at 7Hz."""
    cfg = FullModelConfig()
    apply_preset(cfg, "L: Hippocampal CA1 Pyramidal (Adapting)")
    
    res = NeuronSolver(cfg).run_single()
    st = _spike_times(res.v_soma, res.t)
    
    # Should fire multiple spikes (at least 5-10) over 5000ms with dual-drive
    # Primary: 2.0 µA/cm² tonic + Secondary: 5.0 mS/cm² AMPA at 7Hz
    assert len(st) >= 5, f"Expected at least 5 spikes with dual-drive, got {len(st)}"
    
    # Check that firing follows the 7Hz rhythm (ISI ~143ms)
    if len(st) > 1:
        isi = np.diff(st)
        mean_isi = np.mean(isi)
        assert 100 <= mean_isi <= 200, f"Expected ISI ~143ms for 7Hz, got {mean_isi:.1f}ms"


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
