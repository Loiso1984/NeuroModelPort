#!/usr/bin/env python3
"""
Test: Does secondary stimulus attenuation work correctly?
Use stronger inhibition to see the effect clearly.
"""
import numpy as np
from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver
from core.dual_stimulation import DualStimulationConfig

print("=" * 90)
print("TEST: Secondary Stimulus with Strong Inhibition")
print("=" * 90)

# First, get baseline (no secondary)
cfg_baseline = FullModelConfig()
apply_preset(cfg_baseline, 'A: Squid Giant Axon (HH 1952)')
cfg_baseline.stim.stim_type = 'const'
cfg_baseline.stim.Iext = 12.0

solver_baseline = NeuronSolver(cfg_baseline)
result_baseline = solver_baseline.run_single()
spikes_baseline = (result_baseline.v_soma > (result_baseline.v_soma[0] + 30)).sum()

print(f"Baseline (no secondary): {spikes_baseline} spikes\n")

# Test with STRONG secondary inhibition
test_cases = [
    (0.0, 100.0, -20.0, "No distance, strong inhibition (-20µA)"),
    (100.0, 100.0, -20.0, "100µm distance, strong inhibition (-20µA * 0.368 = -7.36µA)"),
    (150.0, 150.0, -20.0, "150µm distance, strong inhibition (-20µA * 0.368 = -7.36µA)"),
]

print("Comparison:")
print("─" * 90)

for dist, space_const, iext_secondary, desc in test_cases:
    cfg = FullModelConfig()
    apply_preset(cfg, 'A: Squid Giant Axon (HH 1952)')
    cfg.stim.stim_type = 'const'
    cfg.stim.Iext = 12.0
    
    cfg.dual_stimulation = DualStimulationConfig()
    cfg.dual_stimulation.enabled = True
    cfg.dual_stimulation.secondary_location = 'dendritic_filtered'
    cfg.dual_stimulation.secondary_stim_type = 'const'
    cfg.dual_stimulation.secondary_Iext = iext_secondary
    cfg.dual_stimulation.secondary_start = 10.0
    cfg.dual_stimulation.secondary_distance_um = dist
    cfg.dual_stimulation.secondary_space_constant_um = space_const
    cfg.dual_stimulation.secondary_tau_dendritic_ms = 10.0
    
    # Calculate expected attenuation
    if space_const > 0:
        attenuation = np.exp(-dist / space_const)
        effective_iext = iext_secondary * attenuation
    else:
        attenuation = 1.0
        effective_iext = iext_secondary
    
    try:
        solver = NeuronSolver(cfg)
        result = solver.run_single()
        
        v = result.v_soma
        spikes = (v > (v[0] + 30)).sum()
        reduction = spikes_baseline - spikes
        reduction_pct = 100 * reduction / spikes_baseline if spikes_baseline > 0 else 0
        
        print(f"\n{desc}")
        print(f"  Attenuation: {attenuation:.4f}")
        print(f"  Effective secondary: {effective_iext:.2f} µA/cm²")
        print(f"  Result: {spikes} spikes (baseline={spikes_baseline}, reduction={reduction:+d}, {reduction_pct:+.1f}%)")
        
        if reduction > 0:
            print(f"  ✓ Inhibition is working")
        else:
            print(f"  ✗ Inhibition is NOT working")
            
    except Exception as e:
        print(f"  ✗ ERROR: {e}")

print("\n" + "=" * 90)
print("Interpretation:")
print("  - If attenuation works: dist=0 should have MORE inhibition than dist=100")
print("  - reduction should decrease as attenuation decreases")
print("=" * 90)
