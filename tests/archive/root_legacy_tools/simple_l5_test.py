"""simple_l5_test.py - Simple L5 test to check what's happening"""

from core.solver import NeuronSolver
from core.models import FullModelConfig
from core.presets import apply_preset
import numpy as np

def get_frequency(t, v_soma, v_threshold=-30.0):
    """Extract firing frequency from soma voltage trace."""
    dv = np.diff(v_soma)
    upstrokes = np.where(dv > 0)[0]
    
    threshold_crossings = 0
    prev_below = True
    for i in range(len(v_soma)):
        if v_soma[i] < v_threshold:
            prev_below = True
        elif prev_below and v_soma[i] >= v_threshold:
            threshold_crossings += 1
            prev_below = False
    
    t_sim_sec = t[-1] / 1000.0
    if t_sim_sec > 0:
        frequency = threshold_crossings / t_sim_sec
    else:
        frequency = 0.0
    
    return frequency, threshold_crossings

print("="*70)
print("L5 SIMPLE TEST")
print("="*70)

cfg = FullModelConfig()
print(f"\nDefault config:")
print(f"  stim.Iext: {cfg.stim.Iext}")
print(f"  stim_location: {cfg.stim_location.location}")

apply_preset(cfg, 'B: Pyramidal L5 (Mainen 1996)')

print(f"\nAfter L5 preset:")
print(f"  stim.Iext: {cfg.stim.Iext}")
print(f"  stim_location: {cfg.stim_location.location}")
print(f"  stim.pulse_start: {cfg.stim.pulse_start}")
print(f"  stim.t_sim: {cfg.stim.t_sim}")
print(f"  dendritic_filter.enabled: {cfg.dendritic_filter.enabled}")
print(f"  dendritic_filter.distance_um: {cfg.dendritic_filter.distance_um}")
print(f"  dendritic_filter.space_constant_um: {cfg.dendritic_filter.space_constant_um}")
print(f"  dendritic_filter.tau_dendritic_ms: {cfg.dendritic_filter.tau_dendritic_ms}")

print(f"\nRunning simulation...")
solver = NeuronSolver(cfg)
result = solver.run_single()

freq, n_spikes = get_frequency(result.t, result.v_soma)

print(f"\nSimulation results:")
print(f"  V_soma range: [{result.v_soma.min():.1f}, {result.v_soma.max():.1f}] mV")
print(f"  Firing frequency: {freq:.2f} Hz")
print(f"  Number of threshold crossings: {n_spikes}")
print(f"  First spike time (approx): {result.t[np.where(result.v_soma > -30)[0][0]] if n_spikes > 0 else 'N/A'} ms")

# Print V_soma values around key time points
print(f"\nV_soma samples:")
indices = [0, len(result.t)//4, len(result.t)//2, 3*len(result.t)//4, -1]
for idx in indices:
    print(f"  t={result.t[idx]:7.1f}ms: V={result.v_soma[idx]:7.2f}mV")
