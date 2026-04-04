#!/usr/bin/env python3
"""
Debug dendritic filter implementation.
Check if dendritic_filtered stimulation location works correctly.
"""
import numpy as np
from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver

print("=" * 90)
print("DEBUG: Dendritic Filter Scaling and Effectiveness")
print("=" * 90)

cfg = FullModelConfig()
apply_preset(cfg, 'B: Pyramidal L5 (Mainen 1996)')

print(f"\nL5 Preset Configuration:")
print(f"  Stim location: {cfg.stim_location.location}")
print(f"  Iext: {cfg.stim.Iext} µA/cm²")
print(f"  Dendritic filter enabled: {cfg.dendritic_filter.enabled}")
print(f"  Dendritic filter distance: {cfg.dendritic_filter.distance_um} µm")
print(f"  Dendritic filter space constant: {cfg.dendritic_filter.space_constant_um} µm")
print(f"  Dendritic filter tau: {cfg.dendritic_filter.tau_dendritic_ms} ms")

# Calculate attenuation factor
if cfg.dendritic_filter.space_constant_um > 0:
    attenuation = np.exp(-cfg.dendritic_filter.distance_um / cfg.dendritic_filter.space_constant_um)
    print(f"  Attenuation factor: {attenuation:.4f}")
    effective_current = cfg.stim.Iext * attenuation
    print(f"  Effective soma current: {effective_current:.4f} µA/cm²")
    expected_spikes = "~5 spikes (less than direct soma stim)"
else:
    print(f"  Attenuation: N/A (space_constant=0)")

# Test different stimulus locations
test_cases = [
    ('soma', None, 'Direct soma stimulation'),
    ('dendritic_filtered', cfg.dendritic_filter, 'L5 dendritic_filtered settings'),
]

print("\n" + "-" * 90)
print("Comparing stimulus locations:")
print("-" * 90)

results = {}

for location, dfilter_params, desc in test_cases:
    cfg_test = FullModelConfig()
    apply_preset(cfg_test, 'B: Pyramidal L5 (Mainen 1996)')
    cfg_test.stim_location.location = location
    
    if dfilter_params is None and location == 'soma':
        cfg_test.dendritic_filter.enabled = False
        cfg_test.stim.Iext = 6.0  # Use preset value for soma
    else:
        # Keep dendritic_filter params as-is from preset
        pass
    
    print(f"\n✓ {desc}")
    print(f"  Location: {cfg_test.stim_location.location}")
    print(f"  Iext: {cfg_test.stim.Iext:.2f} µA/cm²")
    print(f"  Filter enabled: {cfg_test.dendritic_filter.enabled}")
    
    try:
        solver = NeuronSolver(cfg_test)
        result = solver.run_single()
        v = result.v_soma
        peak = v.max()
        rest = v[0]
        spikes = (v > (rest + 30)).sum()
        
        print(f"  Peak V: {peak:.2f} mV")
        print(f"  Resting V: {rest:.2f} mV")
        print(f"  Spikes in 150ms: {spikes}")
        print(f"  Firing rate: {spikes / (result.t[-1] / 1000.0):.1f} Hz")
        
        results[location] = {'peak': peak, 'spikes': spikes, 'freq': spikes / (result.t[-1] / 1000.0)}
    except Exception as e:
        print(f"  ERROR: {e}")
        results[location] = {'error': str(e)[:60]}

# Summary
print("\n" + "=" * 90)
print("Analysis:")
print("=" * 90)

if 'soma' in results and 'dendritic_filtered' in results:
    r_soma = results['soma']
    r_dend = results['dendritic_filtered']
    
    if 'error' not in r_soma and 'error' not in r_dend:
        spike_reduction = (r_soma['spikes'] - r_dend['spikes']) / r_soma['spikes']
        print(f"\nDendritic filter reduces spikes by: {100*spike_reduction:.1f}%")
        print(f"  Soma stimulus: {r_soma['spikes']} spikes")
        print(f"  Dendritic stimulus: {r_dend['spikes']} spikes")
        
        if spike_reduction > 0.2:
            print(f"✓ Dendritic filtering is working as expected (>20% reduction)")
        else:
            print(f"⚠ Dendritic filtering may not be effective (<20% reduction)")

print("\n" + "=" * 90)
