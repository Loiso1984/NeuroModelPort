"""
PHASE 4: Comprehensive Preset Stress Testing
Test all 15 presets across wide parameter ranges and channel combinations
Validate: stability, parameter ranges, multi-channel interactions
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset, get_preset_names
from core.solver import NeuronSolver

print("=" * 100)
print("PHASE 4: COMPREHENSIVE PRESET STRESS TESTING")
print("=" * 100)

presets = get_preset_names()

# ============================================================================
# TEST A: Channel Flag Audit
# ============================================================================
print("\n[TEST A] Channel Configuration Audit")
print("-" * 100)

channel_audit = {}
for preset_name in presets:
    cfg = FullModelConfig()
    apply_preset(cfg, preset_name)
    
    channel_audit[preset_name] = {
        'IA': cfg.channels.enable_IA,
        'ICa': cfg.channels.enable_ICa,
        'Ih': cfg.channels.enable_Ih,
        'SK': cfg.channels.enable_SK if hasattr(cfg.channels, 'enable_SK') else False,
        'gA_max': cfg.channels.gA_max,
        'gCa_max': cfg.channels.gCa_max if hasattr(cfg.channels, 'gCa_max') else 0,
        'gIh_max': cfg.channels.gIh_max if hasattr(cfg.channels, 'gIh_max') else 0,
        'gSK_max': cfg.channels.gSK_max if hasattr(cfg.channels, 'gSK_max') else 0,
    }

print(f"\n{'Preset':45} | {'IA':4} | {'ICa':4} | {'Ih':4} | {'SK':4} | {'gA_max':8} | {'gCa_max':8} | {'gIh_max':8}")
print("-" * 100)

for preset_name, flags in channel_audit.items():
    ia_str = "✓" if flags['IA'] else "·"
    ica_str = "✓" if flags['ICa'] else "·"
    ih_str = "✓" if flags['Ih'] else "·"
    sk_str = "✓" if flags['SK'] else "·"
    
    print(f"{preset_name:45} | {ia_str:4} | {ica_str:4} | {ih_str:4} | {sk_str:4} | {flags['gA_max']:8.2f} | {flags['gCa_max']:8.2f} | {flags['gIh_max']:8.3f}")

summary = {}
for key in ['IA', 'ICa', 'Ih', 'SK']:
    summary[key] = sum(1 for f in channel_audit.values() if f[key])

print()
print(f"Summary: IA enabled={summary['IA']}, ICa enabled={summary['ICa']}, Ih enabled={summary['Ih']}, SK enabled={summary['SK']}")

# ============================================================================
# TEST B: Single Run Validation (all presets)
# ============================================================================
print("\n[TEST B] Single Run Validation - All Presets")
print("-" * 100)

validation_results = []

for preset_name in presets:
    cfg = FullModelConfig()
    apply_preset(cfg, preset_name)
    
    try:
        solver = NeuronSolver(cfg)
        result = solver.run_single()
        
        v_soma = result.v_soma
        rest = v_soma[0]
        peak = v_soma.max()
        spike_threshold = rest + 30
        spikes = (v_soma > spike_threshold).sum()
        
        status = "✓" if spikes > 0 else "✪" if peak > (rest + 5) else "✗"
        validation_results.append((preset_name, status, rest, peak, spikes))
        
    except Exception as e:
        validation_results.append((preset_name, "✗", np.nan, np.nan, -1))

print(f"\n{'Preset':45} | {'Status':6} | {'V_rest':11} | {'V_peak':11} | {'Spikes':8}")
print("-" * 100)

for preset_name, status, rest, peak, spikes in validation_results:
    if spikes >= 0:
        print(f"{preset_name:45} | {status:6} | {rest:11.1f} | {peak:11.1f} | {spikes:8d}")
    else:
        print(f"{preset_name:45} | {status:6} | {'ERROR':>11} | {'ERROR':>11} | {'ERROR':>8}")

pass_count = sum(1 for _, status, _, _, _ in validation_results if status in ["✓", "✪"])
print(f"\nValidation: {pass_count}/{len(presets)} presets passed")

# ============================================================================
# TEST C: Current Injection Stress Test
# ============================================================================
print("\n[TEST C] Current Injection Stress Test (Wide Range)")
print("-" * 100)

test_presets = ['A: Squid Giant Axon (HH 1952)', 
                'B: Pyramidal L5 (Mainen 1996)',
                'C: FS Interneuron (Wang-Buzsaki)',
                'K: Thalamic Relay (Ih + ICa + Burst)',
                'E: Cerebellar Purkinje (De Schutter)']

current_range = [0.5, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

print(f"\nStress testing {len(test_presets)} key presets across current range ({current_range[0]}-{current_range[-1]} µA)")

for preset_name in test_presets:
    print(f"\n  {preset_name}:")
    stress_results = []
    
    for Iext in current_range:
        cfg = FullModelConfig()
        apply_preset(cfg, preset_name)
        cfg.stim.Iext = Iext
        cfg.stim.stim_type = 'const'
        
        try:
            solver = NeuronSolver(cfg)
            result = solver.run_single()
            
            v_soma = result.v_soma
            rest = v_soma[0]
            peak = v_soma.max()
            spike_threshold = rest + 30
            spikes = (v_soma > spike_threshold).sum()
            
            # Sanity check
            if peak < -100 or peak > 100:
                stress_results.append((Iext, "BAD", peak, spikes))
            else:
                stress_results.append((Iext, "OK", peak, spikes))
                
        except Exception as e:
            stress_results.append((Iext, "ERR", np.nan, -1))
    
    print(f"    {'I (µA)':>8} | {'Status':>6} | {'V_peak (mV)':>14} | {'Spikes':>8}")
    print(f"    " + "-" * 50)
    for Iext, status, peak, spikes in stress_results:
        if spikes >= 0:
            print(f"    {Iext:8.1f} | {status:>6} | {peak:14.1f} | {spikes:8d}")
        else:
            print(f"    {Iext:8.1f} | {status:>6} | {'ERROR':>14} | {'ERROR':>8}")

# ============================================================================
# TEST D: Temperature Robustness
# ============================================================================
print("\n[TEST D] Temperature Robustness Test")
print("-" * 100)

temperature_range = [6.3, 20.0, 37.0]
temp_test_presets = ['A: Squid Giant Axon (HH 1952)',
                     'B: Pyramidal L5 (Mainen 1996)',
                     'C: FS Interneuron (Wang-Buzsaki)']

print(f"\nTesting temperature sensitivity at {temperature_range} °C")

for preset_name in temp_test_presets:
    print(f"\n  {preset_name}:")
    print(f"    {'T (°C)':>8} | {'V_rest':>11} | {'V_peak':>11} | {'Spikes':>8}")
    print(f"    " + "-" * 50)
    
    for T in temperature_range:
        cfg = FullModelConfig()
        apply_preset(cfg, preset_name)
        cfg.env.T_celsius = T
        
        try:
            solver = NeuronSolver(cfg)
            result = solver.run_single()
            
            v_soma = result.v_soma
            rest = v_soma[0]
            peak = v_soma.max()
            spike_threshold = rest + 30
            spikes = (v_soma > spike_threshold).sum()
            
            print(f"    {T:8.1f} | {rest:11.1f} | {peak:11.1f} | {spikes:8d}")
            
        except Exception as e:
            print(f"    {T:8.1f} | {'ERROR':>11} | {'ERROR':>11} | {'ERROR':>8}")

# ============================================================================
# TEST E: Multi-Channel Interaction Matrix
# ============================================================================
print("\n[TEST E] Multi-Channel Interaction Test")
print("-" * 100)

multi_channel_presets = [
    ('K: Thalamic Relay (Ih + ICa + Burst)', ['IA', 'ICa', 'Ih']),
    ('E: Cerebellar Purkinje (De Schutter)', ['IA', 'ICa']),
    ('L: Hippocampal CA1 (Theta rhythm)', ['ICa', 'Ih']),
]

print("\nTesting multi-channel stability:")
print(f"{'Preset':45} | {'Channels':25} | {'V_rest':11} | {'V_peak':11} | {'Status':8}")
print("-" * 100)

for preset_name, channels in multi_channel_presets:
    cfg = FullModelConfig()
    apply_preset(cfg, preset_name)
    
    channel_str = "+".join(channels)
    
    try:
        solver = NeuronSolver(cfg)
        result = solver.run_single()
        
        v_soma = result.v_soma
        rest = v_soma[0]
        peak = v_soma.max()
        
        # Check for pathological behavior
        if -120 < rest < -30 and -100 < peak < 80:
            status = "✓"
        else:
            status = "⚠"
        
        print(f"{preset_name:45} | {channel_str:25} | {rest:11.1f} | {peak:11.1f} | {status:8}")
        
    except Exception as e:
        print(f"{preset_name:45} | {channel_str:25} | {'ERROR':>11} | {'ERROR':>11} | {'✗':>8}")

# ============================================================================
# TEST F: Parameter Range Validation
# ============================================================================
print("\n[TEST F] Parameter Range Validation")
print("-" * 100)

print("\nChecking all conductances are within physiological ranges:")

param_issues = []
for preset_name in presets:
    cfg = FullModelConfig()
    apply_preset(cfg, preset_name)
    
    issues = []
    
    # Check voltage parameters
    if cfg.channels.ENa < 40 or cfg.channels.ENa > 65:
        issues.append(f"ENa={cfg.channels.ENa}")
    if cfg.channels.EK < -100 or cfg.channels.EK > -50:
        issues.append(f"EK={cfg.channels.EK}")
    
    # Check conductances (mS/cm²)
    if cfg.channels.gNa_max < 10 or cfg.channels.gNa_max > 200:
        issues.append(f"gNa={cfg.channels.gNa_max}")
    if cfg.channels.gK_max < 1 or cfg.channels.gK_max > 100:
        issues.append(f"gK={cfg.channels.gK_max}")
    if cfg.channels.gL < 0 or cfg.channels.gL > 1.0:
        issues.append(f"gL={cfg.channels.gL}")
    
    # Check channel-specific parameters
    if cfg.channels.enable_IA and (cfg.channels.gA_max < 0.01 or cfg.channels.gA_max > 5):
        issues.append(f"gA={cfg.channels.gA_max}")
    
    if issues:
        param_issues.append((preset_name, issues))

if param_issues:
    print("\n⚠️  Potential parameter issues found:")
    for preset_name, issues in param_issues:
        print(f"  {preset_name}:")
        for issue in issues:
            print(f"    - {issue}")
else:
    print("  ✓ All parameters within physiological ranges")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 100)
print("PHASE 4 COMPLETE - SUMMARY")
print("=" * 100)

print(f"\nChannel Configuration:")
print(f"  IA enabled in {summary['IA']}/15 presets")
print(f"  ICa enabled in {summary['ICa']}/15 presets")
print(f"  Ih enabled in {summary['Ih']}/15 presets")
print(f"  SK enabled in {summary['SK']}/15 presets")

print(f"\nValidation Status:")
print(f"  Single-run: {pass_count}/15 presets functional")
print(f"  Parameter issues: {'✓ None' if not param_issues else f'⚠ {len(param_issues)} found'}")

print(f"\nNext Phase 5: Multi-channel stress testing with parameter sweeps")
print(f"Focus: Ih+ICa interactions, IA+ICa dynamics, all-channel combinations")
