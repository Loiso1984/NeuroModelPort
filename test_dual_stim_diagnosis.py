#!/usr/bin/env python3
"""
Diagnostic: Check how ion channels vary across presets
and understand dual stimulation failures.
"""
import numpy as np
from core.models import FullModelConfig
from core.presets import apply_preset, get_preset_names
from core.solver import NeuronSolver

print("=" * 100)
print("ION CHANNEL CONFIGURATION AUDIT ACROSS ALL PRESETS")
print("=" * 100)

presets = get_preset_names()

# Collect channel configurations
channel_configs = {}

for preset_name in presets:
    try:
        cfg = FullModelConfig()
        apply_preset(cfg, preset_name)
        
        channel_configs[preset_name] = {
            'enable_Ih': cfg.channels.enable_Ih,
            'enable_ICa': cfg.channels.enable_ICa,
            'enable_IA': cfg.channels.enable_IA,
            'enable_SK': cfg.channels.enable_SK,
            'dynamic_Ca': cfg.calcium.dynamic_Ca,
            'gNa': cfg.channels.gNa_max,
            'gK': cfg.channels.gK_max,
            'gL': cfg.channels.gL,
            'gIh': cfg.channels.gIh_max if cfg.channels.enable_Ih else 0.0,
            'gCa': cfg.channels.gCa_max if cfg.channels.enable_ICa else 0.0,
            'gIA': cfg.channels.gIA_max if cfg.channels.enable_IA else 0.0,
            'gSK': cfg.channels.gSK_max if cfg.channels.enable_SK else 0.0,
            'stim_type': cfg.stim.stim_type,
            'Iext': cfg.stim.Iext,
        }
    except Exception as e:
        print(f"✗ {preset_name}: {str(e)[:50]}")

print("\n┌─ PRESET CHANNEL CONFIGURATION SUMMARY ─┐\n")
print(f"{'Preset':<42} | Channels" + " " * 20 + "| Stim")
print(f"{'-' * 42} | {'-' * 45} | {'-' * 20}")

for preset_name, config in sorted(channel_configs.items()):
    chans = ""
    if config['enable_Ih']:
        chans += "Ih "
    if config['enable_ICa']:
        chans += "ICa "
    if config['enable_IA']:
        chans += "IA "
    if config['enable_SK']:
        chans += "SK "
    if config['dynamic_Ca']:
        chans += "[dCa] "
    if not chans:
        chans = "Na,K,L only"
    
    stim = f"{config['stim_type']}({config['Iext']:.1f})"
    print(f"{preset_name:<42} | {chans:<45} | {stim:<20}")

print("\n" + "=" * 100)
print("TESTING DUAL STIMULATION WITH DIFFERENT ION CHANNEL CONFIGURATIONS")
print("=" * 100)

# Test dual stim with various configurations
test_cases = [
    {
        'name': 'Simple (Na,K,L only)',
        'preset': 'A: Squid Giant Axon (HH 1952)',
        'stim_type': 'const',
        'Iext': 10.0,
    },
    {
        'name': 'L5 Pyr (baseline)',
        'preset': 'B: Pyramidal L5 (Mainen 1996)',
        'stim_type': 'const',
        'Iext': 6.0,
    },
    {
        'name': 'Thalamic (Ih + ICa)',
        'preset': 'K: Thalamic Relay (Ih + ICa + Burst)',
        'stim_type': 'const',
        'Iext': 10.0,
    },
    {
        'name': 'Purkinje (Ih + ICa + IA + SK)',
        'preset': 'E: Cerebellar Purkinje (De Schutter)',
        'stim_type': 'const',
        'Iext': 30.0,
    },
]

print("\nTest: Single stimulation on each neuron type")
print("─" * 80)

from core.dual_stimulation import DualStimulationConfig

results = {}
for test in test_cases:
    cfg = FullModelConfig()
    apply_preset(cfg, test['preset'])
    cfg.stim.stim_type = test['stim_type']
    cfg.stim.Iext = test['Iext']
    
    try:
        solver = NeuronSolver(cfg)
        result = solver.run_single()
        
        v_soma = result.v_soma
        peak = v_soma.max()
        rest = v_soma[0]
        spikes = (v_soma > (rest + 30)).sum()
        
        results[test['name']] = {
            'success': True,
            'peak': peak,
            'spikes': spikes,
            'channels': f"Ih={cfg.channels.enable_Ih}, ICa={cfg.channels.enable_ICa}, IA={cfg.channels.enable_IA}, SK={cfg.channels.enable_SK}",
        }
        
        print(f"✓ {test['name']:<25} | Peak={peak:7.2f}mV | Spikes={spikes:3d} | {results[test['name']]['channels']}")
    except Exception as e:
        print(f"✗ {test['name']:<25} | ERROR: {str(e)[:50]}")
        results[test['name']] = {'success': False, 'error': str(e)[:50]}

print("\n" + "=" * 100)
print("DUAL STIMULATION: Primary vs Secondary Parameter Tracking")
print("=" * 100)

# Check if dual stimulation correctly uses both primary and secondary parameters
print("\nTest: Verify RHS receives dual stim parameters correctly")
print("─" * 80)

cfg = FullModelConfig()
apply_preset(cfg, 'B: Pyramidal L5 (Mainen 1996)')

# Enable dual stim
cfg.dual_stimulation = DualStimulationConfig()
cfg.dual_stimulation.enabled = True
cfg.dual_stimulation.secondary_location = 'soma'  # Also going to soma for test
cfg.dual_stimulation.secondary_Iext = 5.0
cfg.dual_stimulation.secondary_start = 50.0  # Different start time

print(f"\nPrimary stim (from cfg.stim):")
print(f"  Type: {cfg.stim.stim_type}")
print(f"  Iext: {cfg.stim.Iext} µA/cm²")
print(f"  Start: {cfg.stim.pulse_start} ms")

print(f"\nSecondary stim (from cfg.dual_stimulation):")
print(f"  Type: {cfg.dual_stimulation.secondary_stim_type}")
print(f"  Iext: {cfg.dual_stimulation.secondary_Iext} µA/cm²")
print(f"  Start: {cfg.dual_stimulation.secondary_start} ms")
print(f"  Location: {cfg.dual_stimulation.secondary_location}")

try:
    solver = NeuronSolver(cfg)
    result = solver.run_single()
    
    v_soma = result.v_soma
    peak = v_soma.max()
    rest = v_soma[0]
    
    # Check for evidence of secondary stim
    # If secondary stim kicks in at t=50ms, we should see a change in behavior
    t = result.t
    idx_50 = np.argmin(np.abs(t - 50.0))
    
    v_before_50 = v_soma[:idx_50]
    v_after_50 = v_soma[idx_50:]
    
    print(f"\n✓ Dual stim simulation completed")
    print(f"  Peak V: {peak:.2f}mV")
    print(f"  Resting V: {rest:.2f}mV")
    print(f"  V_before_50ms: [{v_before_50.min():.1f}, {v_before_50.max():.1f}]")
    print(f"  V_after_50ms: [{v_after_50.min():.1f}, {v_after_50.max():.1f}]")
    
    # If secondary stim is working, there should be a noticeable change
    if v_after_50.max() > v_before_50.max():
        print(f"  ✓ Secondary stimulus appears to be having effect (peak increased after 50ms)")
    else:
        print(f"  ⚠ Secondary stimulus may NOT be affecting dynamics (peak did not increase)")

except Exception as e:
    print(f"\n✗ ERROR: {e}")

print("\n" + "=" * 100)
