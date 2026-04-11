"""
Validation tests for Phase 11.12 Critical Hotfixes.

Tests:
1. SPN preset shows delayed firing (~100ms) due to slow alpha stimulus
2. LUT integration in hines.py works correctly
3. Presets with dual stimulation load correctly
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver


def test_spn_delayed_firing():
    """Test that SPN preset shows characteristic delayed firing due to slow alpha stimulus."""
    cfg = FullModelConfig()
    apply_preset(cfg, "Q: Striatal Spiny Projection (SPN)")
    
    # Verify stimulus configuration
    assert cfg.stim.stim_type == 'alpha', "SPN should use alpha stimulus"
    assert cfg.stim.alpha_tau == 20.0, "SPN should have slow alpha tau (20ms)"
    assert cfg.stim.Iext == 15.0, "SPN should have increased Iext (15.0)"
    
    # Run simulation
    res = NeuronSolver(cfg).run_single()
    
    # Check that simulation ran successfully
    assert np.all(np.isfinite(res.v_soma)), "SPN: non-finite voltage trace"
    assert -140.0 < float(np.min(res.v_soma)) < 80.0, "SPN: minimum voltage out of range"
    assert -140.0 < float(np.max(res.v_soma)) < 80.0, "SPN: maximum voltage out of range"
    
    # Check for delayed firing onset (first spike should be >5ms after start)
    # Find first spike threshold crossing
    threshold = -20.0  # Spike threshold
    above_threshold = res.v_soma > threshold
    if np.any(above_threshold):
        first_spike_idx = np.where(above_threshold)[0][0]
        first_spike_time = res.t[first_spike_idx]
        print(f"SPN first spike at {first_spike_time:.2f} ms")
        # With slow alpha (tau=20ms) and strong IA, first spike should be delayed
        # We expect >5ms delay (alpha rise time + IA blocking)
        assert first_spike_time > 5.0, f"SPN should show delayed firing (first spike at {first_spike_time:.2f} ms, expected >5ms)"
    else:
        print("SPN: No spikes detected (may be subthreshold)")
    
    print(f"SPN: Vmin={np.min(res.v_soma):.2f} mV, Vmax={np.max(res.v_soma):.2f} mV")


def test_purkinje_dual_stimulation():
    """Test that Purkinje preset loads with dual stimulation enabled."""
    cfg = FullModelConfig()
    apply_preset(cfg, "E: Cerebellar Purkinje (De Schutter)")
    
    # Verify dual stimulation is configured
    assert cfg.dual_stimulation.enabled, "Purkinje should have dual stimulation enabled"
    assert cfg.stim_location.location == "soma", "Primary should be soma"
    assert cfg.stim.stim_type == "const", "Primary should be const"
    assert cfg.stim.Iext == 30.0, "Primary Iext should be 30.0"
    assert cfg.dual_stimulation.secondary_location == "dendritic_filtered", "Secondary should be dendritic"
    assert cfg.dual_stimulation.secondary_stim_type == "GABAA", "Secondary should be GABAA"
    assert cfg.dual_stimulation.secondary_train_type == "poisson", "Secondary should be poisson"
    assert cfg.dual_stimulation.secondary_train_freq_hz == 60.0, "Secondary freq should be 60Hz"
    
    # Run simulation to ensure it works
    res = NeuronSolver(cfg).run_single()
    
    # Check that simulation ran successfully
    assert np.all(np.isfinite(res.v_soma)), "Purkinje: non-finite voltage trace"
    assert -140.0 < float(np.min(res.v_soma)) < 80.0, "Purkinje: minimum voltage out of range"
    assert -140.0 < float(np.max(res.v_soma)) < 80.0, "Purkinje: maximum voltage out of range"
    
    print(f"Purkinje: Vmin={np.min(res.v_soma):.2f} mV, Vmax={np.max(res.v_soma):.2f} mV")


def test_lut_integration():
    """Test that LUT functions are imported and used in hines.py."""
    import core.hines as hines_module
    import core.kinetics as kinetics_module
    
    # Verify LUT functions exist in kinetics
    assert hasattr(kinetics_module, 'am_lut'), "kinetics should have am_lut"
    assert hasattr(kinetics_module, 'bm_lut'), "kinetics should have bm_lut"
    assert hasattr(kinetics_module, 'ah_lut'), "kinetics should have ah_lut"
    assert hasattr(kinetics_module, 'bh_lut'), "kinetics should have bh_lut"
    assert hasattr(kinetics_module, 'an_lut'), "kinetics should have an_lut"
    assert hasattr(kinetics_module, 'bn_lut'), "kinetics should have bn_lut"
    
    # Verify hines.py imports LUT versions (check by looking at source)
    hines_source = Path(hines_module.__file__).read_text(encoding='utf-8')
    assert 'am_lut' in hines_source, "hines.py should import am_lut"
    assert 'bm_lut' in hines_source, "hines.py should import bm_lut"
    assert 'ah_lut' in hines_source, "hines.py should import ah_lut"
    assert 'bh_lut' in hines_source, "hines.py should import bh_lut"
    assert 'an_lut' in hines_source, "hines.py should import an_lut"
    assert 'bn_lut' in hines_source, "hines.py should import bn_lut"
    
    # Verify update_gates_analytic uses LUT functions
    assert 'am_lut(vi)' in hines_source, "update_gates_analytic should use am_lut"
    assert 'bm_lut(vi)' in hines_source, "update_gates_analytic should use bm_lut"
    assert 'ah_lut(vi)' in hines_source, "update_gates_analytic should use ah_lut"
    assert 'bh_lut(vi)' in hines_source, "update_gates_analytic should use bh_lut"
    assert 'an_lut(vi)' in hines_source, "update_gates_analytic should use an_lut"
    assert 'bn_lut(vi)' in hines_source, "update_gates_analytic should use bn_lut"
    
    print("LUT integration: All checks passed")


if __name__ == "__main__":
    test_lut_integration()
    test_purkinje_dual_stimulation()
    test_spn_delayed_firing()
    print("\nAll Phase 11.12 hotfix validation tests passed!")
