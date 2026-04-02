#!/usr/bin/env python3
"""
Debug: Trace dual stim parameters passed to RHS solver
"""
import numpy as np
from core.models import FullModelConfig
from core.presets import apply_preset
from core.dual_stimulation import DualStimulationConfig

# Monkey-patch the solver to log parameters
original_rhs = None
debug_log = []

def debug_rhs_wrapper(t, y, n_comp, *args, **kwargs):
    """Wrapper to log RHS calls"""
    # Extract dual stim parameters (they're at the end)
    if len(args) > 35:  # empirically find the dual stim args
        dual_stim_enabled = args[34]  # This should be the dual_stim_enabled flag
        if dual_stim_enabled == 1 and len(debug_log) == 0:  # Log only once
            debug_log.append({
                'dual_stim_enabled': dual_stim_enabled,
                'args_count': len(args),
            })
    return original_rhs(t, y, n_comp, *args, **kwargs)

# Patch the RHS
from core import rhs
original_rhs = rhs.rhs_multicompartment
# rhs.rhs_multicompartment = debug_rhs_wrapper  # Can't patch numba function

print("=" * 90)
print("DEBUG: Dual Stimulation Parameter Validation")
print("=" * 90)

cfg = FullModelConfig()
apply_preset(cfg, 'A: Squid Giant Axon (HH 1952)')

# Setup dual stim manually with explicit values
cfg.stim.stim_type = 'const'
cfg.stim.Iext = 12.0
cfg.stim.pulse_start = 10.0

cfg.dual_stimulation = DualStimulationConfig()
cfg.dual_stimulation.enabled = True
cfg.dual_stimulation.secondary_location = 'dendritic_filtered'
cfg.dual_stimulation.secondary_stim_type = 'const'
cfg.dual_stimulation.secondary_Iext = -5.0
cfg.dual_stimulation.secondary_start = 10.0
cfg.dual_stimulation.secondary_distance_um = 150.0
cfg.dual_stimulation.secondary_space_constant_um = 150.0
cfg.dual_stimulation.secondary_tau_dendritic_ms = 10.0

print("\nConfiguration:")
print(f"  Primary stimulus:")
print(f"    Location: {cfg.stim_location.location}")
print(f"    Type: {cfg.stim.stim_type}")
print(f"    Iext: {cfg.stim.Iext:.2f} µA/cm²")
print(f"    Start: {cfg.stim.pulse_start} ms")

print(f"\n  Secondary stimulus:")
print(f"    Enabled: {cfg.dual_stimulation.enabled}")
print(f"    Location: {cfg.dual_stimulation.secondary_location}")
print(f"    Type: {cfg.dual_stimulation.secondary_stim_type}")
print(f"    Iext: {cfg.dual_stimulation.secondary_Iext:.2f} µA/cm²")
print(f"    Start: {cfg.dual_stimulation.secondary_start} ms")
print(f"    Distance: {cfg.dual_stimulation.secondary_distance_um} µm")
print(f"    Space constant: {cfg.dual_stimulation.secondary_space_constant_um} µm")
print(f"    Tau: {cfg.dual_stimulation.secondary_tau_dendritic_ms} ms")

# Calculate what attenuation SHOULD be
if cfg.dual_stimulation.secondary_space_constant_um > 0:
    attenuation_expected = np.exp(
        -cfg.dual_stimulation.secondary_distance_um / cfg.dual_stimulation.secondary_space_constant_um
    )
    print(f"    Expected attenuation: {attenuation_expected:.4f}")
    print(f"    Expected effective Iext: {cfg.dual_stimulation.secondary_Iext * attenuation_expected:.4f} µA/cm²")

# Look for issues
print("\nParameter validation:")

# Check if secondary location is valid
valid_locations = ['soma', 'ais', 'dendritic_filtered']
if cfg.dual_stimulation.secondary_location not in valid_locations:
    print(f"  ✗ Invalid secondary location: {cfg.dual_stimulation.secondary_location}")
else:
    print(f"  ✓ Secondary location valid: {cfg.dual_stimulation.secondary_location}")

# Check if space constant is > 0
if cfg.dual_stimulation.secondary_space_constant_um <= 0:
    print(f"  ✗ Space constant must be > 0, got {cfg.dual_stimulation.secondary_space_constant_um}")
else:
    print(f"  ✓ Space constant > 0: {cfg.dual_stimulation.secondary_space_constant_um}")

# Check if tau> 0
if cfg.dual_stimulation.secondary_tau_dendritic_ms <= 0:
    print(f"  ✗ Dendritic tau must be > 0, got {cfg.dual_stimulation.secondary_tau_dendritic_ms}")
else:
    print(f"  ✓ Dendritic tau > 0: {cfg.dual_stimulation.secondary_tau_dendritic_ms}")

# Now run simulation with verbose output
print("\n" + "─" * 90)
print("Running simulation...")
print("─" * 90)

from core.solver import NeuronSolver

try:
    solver = NeuronSolver(cfg)
    result = solver.run_single()
    
    v = result.v_soma
    peak = v.max()
    spikes = (v > (v[0] + 30)).sum()
    
    print(f"\n✓ Simulation completed")
    print(f"  Peak V: {peak:.2f} mV")
    print(f"  Spikes: {spikes}")
    print(f"  Expected: With attenuation={attenuation_expected:.4f}, secondary Iext={cfg.dual_stimulation.secondary_Iext * attenuation_expected:.4f}")
    print(f"  Should reduce spikes from ~422 to ~380-400")
    
    if spikes < 400:
        print(f"  ✓ Secondary inhibition appears to be working")
    else:
        print(f"  ⚠ Secondary inhibition may NOT be applied")

except Exception as e:
    import traceback
    print(f"\n✗ ERROR: {e}")
    traceback.print_exc()

print("\n" + "=" * 90)
