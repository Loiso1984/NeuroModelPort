#!/usr/bin/env python3
"""
Test: Does ANY secondary stimulus get applied when location is dendritic_filtered?
"""
import numpy as np
from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver
from core.dual_stimulation import DualStimulationConfig

print("=" * 90)
print("TEST: Secondary Stimulus Application")
print("=" * 90)

cfg_base = FullModelConfig()
apply_preset(cfg_base, 'A: Squid Giant Axon (HH 1952)')
cfg_base.stim.stim_type = 'const'
cfg_base.stim.Iext = 12.0

# TEST 1: Secondary at SOMA with pulse stimulus
print("\nTEST 1: Secondary stimulus at SOMA (should work)")
print("-" * 90)

cfg1 = FullModelConfig()
apply_preset(cfg1, 'A: Squid Giant Axon (HH 1952)')
cfg1.stim.stim_type = 'const'
cfg1.stim.Iext = 12.0
cfg1.dual_stimulation = DualStimulationConfig()
cfg1.dual_stimulation.enabled = True
cfg1.dual_stimulation.secondary_location = 'soma'
cfg1.dual_stimulation.secondary_stim_type = 'pulse'  # Use pulse for clear timing
cfg1.dual_stimulation.secondary_Iext = -20.0  # Strong inhibition
cfg1.dual_stimulation.secondary_start = 25.0  # Start at 25ms
cfg1.dual_stimulation.secondary_duration = 10.0  # Duration 10ms

try:
    solver1 = NeuronSolver(cfg1)
    result1 = solver1.run_single()
    
    v1 = result1.v_soma
    t1 = result1.t
    
    # Count spikes before and after secondary stim
    idx_25 = np.argmin(np.abs(t1 - 25.0))
    idx_35 = np.argmin(np.abs(t1 - 35.0))
    
    spikes_before = (v1[:idx_25] > (v1[0] + 30)).sum()
    spikes_during = (v1[idx_25:idx_35] > (v1[0] + 30)).sum()
    spikes_after = (v1[idx_35:] > (v1[0] + 30)).sum()
    
    print(f"  Spikes before inhibition (0-25ms): {spikes_before}")
    print(f"  Spikes during inhibition (25-35ms): {spikes_during}")
    print(f"  Spikes after inhibition (35-150ms): {spikes_after}")
    
    if spikes_during == 0:
        print(f"  ✓ Strong inhibition is working (blocked spikes)")
    else:
        print(f"  ⚠ Inhibition did NOT block spikes")
        
except Exception as e:
    print(f"  ✗ ERROR: {e}")

# TEST 2: Secondary at DENDRITIC with pulse stimulus (same as TEST 1 but location changed)
print("\nTEST 2: Secondary stimulus at DENDRITIC_FILTERED (testing if applied at all)")
print("-" * 90)

cfg2 = FullModelConfig()
apply_preset(cfg2, 'A: Squid Giant Axon (HH 1952)')
cfg2.stim.stim_type = 'const'
cfg2.stim.Iext = 12.0
cfg2.dual_stimulation = DualStimulationConfig()
cfg2.dual_stimulation.enabled = True
cfg2.dual_stimulation.secondary_location = 'dendritic_filtered'
cfg2.dual_stimulation.secondary_stim_type = 'pulse'
cfg2.dual_stimulation.secondary_Iext = -20.0  # Same strong inhibition
cfg2.dual_stimulation.secondary_start = 25.0
cfg2.dual_stimulation.secondary_duration = 10.0
cfg2.dual_stimulation.secondary_distance_um = 0.0  # At soma (distance=0)
cfg2.dual_stimulation.secondary_space_constant_um = 100.0

try:
    solver2 = NeuronSolver(cfg2)
    result2 = solver2.run_single()
    
    v2 = result2.v_soma
    t2 = result2.t
    
    idx_25 = np.argmin(np.abs(t2 - 25.0))
    idx_35 = np.argmin(np.abs(t2 - 35.0))
    
    spikes_before = (v2[:idx_25] > (v2[0] + 30)).sum()
    spikes_during = (v2[idx_25:idx_35] > (v2[0] + 30)).sum()
    spikes_after = (v2[idx_35:] > (v2[0] + 30)).sum()
    
    # With distance=0, attenuation should be exp(0) = 1.0, so same as TEST 1
    attenuation = np.exp(-0.0 / 100.0) if 100.0 > 0 else 1.0
    
    print(f"  Secondary location: dendritic_filtered")
    print(f"  Distance: {cfg2.dual_stimulation.secondary_distance_um} µm (at soma)")
    print(f"  Attenuation: {attenuation:.4f} (should be 1.0)")
    print(f"  Effective secondary Iext: {cfg2.dual_stimulation.secondary_Iext * attenuation:.2f}")
    print(f"  Spikes before inhibition (0-25ms): {spikes_before}")
    print(f"  Spikes during inhibition (25-35ms): {spikes_during}")
    print(f"  Spikes after inhibition (35-150ms): {spikes_after}")
    
    if spikes_during == 0:
        print(f"  ✓ Dendritic secondary stimulus IS being applied")
    else:
        print(f"  ✗ Dendritic secondary stimulus is NOT being applied (spikes not blocked)")
        
except Exception as e:
    print(f"  ✗ ERROR: {e}")

# TEST 3: Compare soma vs dendritic secondary
print("\nTEST 3: Compare soma vs dendritic secondary (with attenuation)")
print("-" * 90)

# Soma secondary
cfg3a = FullModelConfig()
apply_preset(cfg3a, 'A: Squid Giant Axon (HH 1952)')
cfg3a.stim.stim_type = 'const'
cfg3a.stim.Iext = 12.0
cfg3a.dual_stimulation = DualStimulationConfig()
cfg3a.dual_stimulation.enabled = True
cfg3a.dual_stimulation.secondary_location = 'soma'
cfg3a.dual_stimulation.secondary_Iext = -5.0
cfg3a.dual_stimulation.secondary_start = 10.0

# Dendritic secondary (with attenuation that gives -5 * 1.0 = -5.0)
cfg3b = FullModelConfig()
apply_preset(cfg3b, 'A: Squid Giant Axon (HH 1952)')
cfg3b.stim.stim_type = 'const'
cfg3b.stim.Iext = 12.0
cfg3b.dual_stimulation = DualStimulationConfig()
cfg3b.dual_stimulation.enabled = True
cfg3b.dual_stimulation.secondary_location = 'dendritic_filtered'
cfg3b.dual_stimulation.secondary_Iext = -5.0
cfg3b.dual_stimulation.secondary_start = 10.0
cfg3b.dual_stimulation.secondary_distance_um = 0.0  # No attenuation
cfg3b.dual_stimulation.secondary_space_constant_um = 100.0

results_3a = None
results_3b = None

try:
    solver3a = NeuronSolver(cfg3a)
    result3a = solver3a.run_single()
    spikes_3a = (result3a.v_soma > (result3a.v_soma[0] + 30)).sum()
    results_3a = spikes_3a
    print(f"  Soma secondary: {spikes_3a} spikes")
except Exception as e:
    print(f"  Soma secondary: ERROR: {e}")

try:
    solver3b = NeuronSolver(cfg3b)
    result3b = solver3b.run_single()
    spikes_3b = (result3b.v_soma > (result3b.v_soma[0] + 30)).sum()
    results_3b = spikes_3b
    print(f"  Dendritic secondary (distance=0): {spikes_3b} spikes")
except Exception as e:
    print(f"  Dendritic secondary: ERROR: {e}")

if results_3a is not None and results_3b is not None:
    if results_3a == results_3b:
        print(f"  ✓ Both equal ({results_3a} spikes) → Secondary stimulus location works")
    else:
        print(f"  ✗ Different: soma={results_3a} vs dendrite={results_3b}")
        print(f"    → Dendritic secondary may NOT be applied correctly")

print("\n" + "=" * 90)
