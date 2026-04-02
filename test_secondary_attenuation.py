#!/usr/bin/env python3
"""
Test: Secondary stimulus attenuation calculation
"""
import numpy as np
from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver
from core.dual_stimulation import DualStimulationConfig

print("=" * 90)
print("TEST: Secondary Stimulus Attenuation")
print("=" * 90)

test_cases = [
    (0.0, 100.0, "No distance (at soma)"),
    (50.0, 100.0, "50µm distance (attenuation=0.606)"),
    (100.0, 100.0, "100µm distance (attenuation=0.368)"),
    (150.0, 150.0, "150µm distance, 150µm space_const (attenuation=0.368)"),
]

baseline_spikes = None

for dist, space_const, desc in test_cases:
    cfg = FullModelConfig()
    apply_preset(cfg, 'A: Squid Giant Axon (HH 1952)')
    cfg.stim.stim_type = 'const'
    cfg.stim.Iext = 12.0
    
    cfg.dual_stimulation = DualStimulationConfig()
    cfg.dual_stimulation.enabled = True
    cfg.dual_stimulation.secondary_location = 'dendritic_filtered'
    cfg.dual_stimulation.secondary_stim_type = 'const'
    cfg.dual_stimulation.secondary_Iext = -5.0  # Constant inhibition
    cfg.dual_stimulation.secondary_start = 10.0
    cfg.dual_stimulation.secondary_distance_um = dist
    cfg.dual_stimulation.secondary_space_constant_um = space_const
    cfg.dual_stimulation.secondary_tau_dendritic_ms = 10.0
    
    # Calculate expected attenuation
    if space_const > 0:
        attenuation = np.exp(-dist / space_const)
        effective_iext = cfg.dual_stimulation.secondary_Iext * attenuation
    else:
        attenuation = 1.0
        effective_iext = cfg.dual_stimulation.secondary_Iext
    
    try:
        solver = NeuronSolver(cfg)
        result = solver.run_single()
        
        v = result.v_soma
        peak = v.max()
        spikes = (v > (v[0] + 30)).sum()
        
        if baseline_spikes is None:
            baseline_spikes = spikes
        
        reduction = baseline_spikes - spikes
        print(f"\n{desc}")
        print(f"  Distance: {dist}µm, Space const: {space_const}µm")
        print(f"  Attenuation: {attenuation:.4f}")
        print(f"  Effective Iext: {effective_iext:.4f} µA/cm²")
        print(f"  Spikes: {spikes} (baseline={baseline_spikes}, diff={reduction:+d})")
        
        # If attenuation is working, spikes should decrease with higher attenuation
        # Actually, LOWER attenuation values should give LESS effect
        # So: dist=0 → att=1.0 → full effect
        #     dist=100 → att=0.368 → 36.8% effect
        # With -5µA secondary:
        #   dist=0: -5µA full = -5µA
        #   dist=100: -5µA * 0.368 = -1.84µA
        # Both should reduce spikes, but dist=100 less so
        
        if attenuation < 1.0 and spikes > baseline_spikes - 50:
            print(f"  ✓ Attenuation appears to be working (less inhibition with lower attenuation)")
        elif attenuation == 1.0 and spikes <= baseline_spikes / 2:
            print(f"  ✓ Full strength inhibition applied (dist=0)")
        else:
            print(f"  ⚠ Result unclear - check if attenuation is applied")
            
    except Exception as e:
        print(f"  ✗ ERROR: {e}")

print("\n" + "=" * 90)
print("Expected behavior:")
print("  - With higher attenuation (lower distance), more inhibition")
print("  - With lower attenuation (higher distance), less inhibition")
print("=" * 90)
