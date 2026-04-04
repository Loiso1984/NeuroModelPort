#!/usr/bin/env python3
"""
Debug: Why does dendritic GABA inhibition not reduce firing?
Test the secondary stimulus parameter flow explicitly.
"""
import numpy as np
from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver
from core.dual_stimulation import DualStimulationConfig

print("=" * 90)
print("DEBUG: Dendritic GABA Inhibition - Parameter Flow Test")
print("=" * 90)

# Use Squid (simple neurons, no extra channels) for clarity
presets_to_test = [
    ('A: Squid Giant Axon (HH 1952)', 12.0),
    ('B: Pyramidal L5 (Mainen 1996)', 8.0),
]

for preset_name, iext_primary in presets_to_test:
    print(f"\n\n{'-' * 90}")
    print(f"Testing: {preset_name}")
    print(f"{'-' * 90}")
    
    # TEST 1: Single primary stimulus only
    print(f"\n[TEST 1] Primary stimulus only")
    cfg1 = FullModelConfig()
    apply_preset(cfg1, preset_name)
    cfg1.stim.stim_type = 'const'
    cfg1.stim.Iext = iext_primary
    cfg1.stim.pulse_start = 10.0
    cfg1.stim.pulse_dur = 1.0  # Not used for const
    
    solver1 = NeuronSolver(cfg1)
    result1 = solver1.run_single()
    v1 = result1.v_soma
    spikes1 = (v1 > (v1[0] + 30)).sum()
    peak1 = v1.max()
    
    print(f"  Iext: {cfg1.stim.Iext:.2f} µA/cm²")
    print(f"  Peak V: {peak1:.2f} mV")
    print(f"  Spikes: {spikes1}")
    
    # TEST 2: Primary + Secondary (inhibitory) both at soma
    print(f"\n[TEST 2] Primary + Secondary inhibition (both soma)")
    cfg2 = FullModelConfig()
    apply_preset(cfg2, preset_name)
    cfg2.stim.stim_type = 'const'
    cfg2.stim.Iext = iext_primary
    cfg2.stim.pulse_start = 10.0
    
    # Add secondary inhibitory stimulus
    cfg2.dual_stimulation = DualStimulationConfig()
    cfg2.dual_stimulation.enabled = True
    cfg2.dual_stimulation.secondary_location = 'soma'
    cfg2.dual_stimulation.secondary_stim_type = 'const'  # Use const for clarity
    cfg2.dual_stimulation.secondary_Iext = -5.0  # Negative = inhibitory
    cfg2.dual_stimulation.secondary_start = 10.0  # Same time as primary
    cfg2.dual_stimulation.secondary_duration = 1.0
    
    try:
        solver2 = NeuronSolver(cfg2)
        result2 = solver2.run_single()
        v2 = result2.v_soma
        spikes2 = (v2 > (v2[0] + 30)).sum()
        peak2 = v2.max()
        
        print(f"  Primary: {cfg2.stim.Iext:.2f} µA/cm²")
        print(f"  Secondary: {cfg2.dual_stimulation.secondary_Iext:.2f} µA/cm² (inhibitory)")
        print(f"  Both at: soma, starting @ {cfg2.dual_stimulation.secondary_start} ms")
        print(f"  Peak V: {peak2:.2f} mV (diff: {peak2-peak1:+.2f})")
        print(f"  Spikes: {spikes2} (diff: {spikes2-spikes1:+d})")
        
        if spikes2 < spikes1:
            print(f"  ✓ Inhibition worked (spikes reduced by {spikes1-spikes2})")
        else:
            print(f"  ⚠ Inhibition did NOT work (spikes same or increased)")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # TEST 3: Primary soma + Secondary dendritic inhibition
    print(f"\n[TEST 3] Primary soma + Secondary dendritic inhibition")
    cfg3 = FullModelConfig()
    apply_preset(cfg3, preset_name)
    cfg3.stim.stim_type = 'const'
    cfg3.stim.Iext = iext_primary
    cfg3.stim.pulse_start = 10.0
    cfg3.stim_location.location = 'soma'  # Force soma for primary
    
    # Add secondary inhibitory stimulus at dendrite
    cfg3.dual_stimulation = DualStimulationConfig()
    cfg3.dual_stimulation.enabled = True
    cfg3.dual_stimulation.secondary_location = 'dendritic_filtered'
    cfg3.dual_stimulation.secondary_stim_type = 'const'
    cfg3.dual_stimulation.secondary_Iext = -5.0  # Negative = inhibitory
    cfg3.dual_stimulation.secondary_start = 10.0
    
    # Dendritic filter parameters for secondary stim
    cfg3.dual_stimulation.secondary_distance_um = 150.0
    cfg3.dual_stimulation.secondary_space_constant_um = 150.0
    cfg3.dual_stimulation.secondary_tau_dendritic_ms = 10.0
    
    try:
        solver3 = NeuronSolver(cfg3)
        result3 = solver3.run_single()
        v3 = result3.v_soma
        spikes3 = (v3 > (v3[0] + 30)).sum()
        peak3 = v3.max()
        
        print(f"  Primary: {cfg3.stim.Iext:.2f} µA/cm² @ soma")
        print(f"  Secondary: {cfg3.dual_stimulation.secondary_Iext:.2f} µA/cm² @ dendrite (inhibitory)")
        print(f"  Dendritic filter: distance={cfg3.dual_stimulation.secondary_distance_um}µm, τ={cfg3.dual_stimulation.secondary_tau_dendritic_ms}ms")
        print(f"  Peak V: {peak3:.2f} mV (diff from test1: {peak3-peak1:+.2f})")
        print(f"  Spikes: {spikes3} (diff from test1: {spikes3-spikes1:+d})")
        
        if spikes3 < spikes1:
            pct = 100 * (spikes1 - spikes3) / spikes1
            print(f"  ✓ Dendritic inhibition worked ({pct:.1f}% reduction)")
        else:
            print(f"  ⚠ Dendritic inhibition did NOT work")
    except Exception as e:
        print(f"  ✗ Error: {e}")

print("\n" + "=" * 90)
print(f"Summary: Testing whether dual stimulation secondary parameters are used correctly")
print("=" * 90)
