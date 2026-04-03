"""
PHASE 5: Multi-Channel Interaction Stress Testing
Test combinations: Ih+ICa, IA+ICa, IA+Ih, all channels
Validate numerical stability and biophysical realism
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver

print("=" * 100)
print("PHASE 5: MULTI-CHANNEL INTERACTION STRESS TESTING")
print("=" * 100)

# ============================================================================
# TEST A: Ih + ICa Interaction (Thalamic Relay Model)
# ============================================================================
print("\n[TEST A] Ih + ICa Interaction Test")
print("-" * 100)
print("Base: K (Thalamic Relay) with both Ih and ICa enabled")
print()

configs_to_test = [
    ('Both enabled (default)', {'enable_Ih': True, 'enable_ICa': True}),
    ('Only Ih', {'enable_Ih': True, 'enable_ICa': False}),
    ('Only ICa', {'enable_Ih': False, 'enable_ICa': True}),
    ('Both disabled (baseline)', {'enable_Ih': False, 'enable_ICa': False}),
]

print(f"{'Config':30} | {'V_rest':11} | {'V_peak':11} | {'Spikes':8} | {'Status':10}")
print("-" * 80)

for config_name, channel_flags in configs_to_test:
    cfg = FullModelConfig()
    apply_preset(cfg, 'K: Thalamic Relay (Ih + ICa + Burst)')
    
    cfg.channels.enable_Ih = channel_flags['enable_Ih']
    cfg.channels.enable_ICa = channel_flags['enable_ICa']
    
    try:
        solver = NeuronSolver(cfg)
        result = solver.run_single()
        
        v_soma = result.v_soma
        rest = v_soma[0]
        peak = v_soma.max()
        spike_threshold = rest + 30
        spikes = (v_soma > spike_threshold).sum()
        
        # Validate
        if -120 < rest < -30 and -100 < peak < 80 and spikes >= 0:
            status = "✓ OK"
        else:
            status = "⚠ WARN"
        
        print(f"{config_name:30} | {rest:11.1f} | {peak:11.1f} | {spikes:8d} | {status:10}")
        
    except Exception as e:
        print(f"{config_name:30} | {'ERROR':>11} | {'ERROR':>11} | {'ERROR':>8} | {'✗':>10}")

print("\nInterpretation:")
print("  Ih: Pacemaker inward current (depolarizing, slow)")
print("  ICa: Calcium inward current (depolarizing, faster than Ih)")
print("  Expected: Combined Ih+ICa may interact - check for numerical issues")

# ============================================================================
# TEST B: IA + ICa Interaction (with IA in presets)
# ============================================================================
print("\n[TEST B] IA + ICa Interaction Test")
print("-" * 100)
print("Testing IA-enabled presets with calcium channel variations")
print()

ia_presets = [
    'C: FS Interneuron (Wang-Buzsaki)',
    'D: alpha-Motoneuron (Powers 2001)',
    'E: Cerebellar Purkinje (De Schutter)',
]

for preset_name in ia_presets:
    print(f"\n  {preset_name}:")
    
    # Test with and without ICa
    configs = [
        ('With ICa (if present)', {}),  # Keep as-is
        ('ICa forced off', {'enable_ICa': False}),
    ]
    
    print(f"    {'Config':30} | {'V_rest':11} | {'V_peak':11} | {'IA status':8}")
    print(f"    " + "-" * 70)
    
    for config_name, overrides in configs:
        cfg = FullModelConfig()
        apply_preset(cfg, preset_name)
        
        for key, val in overrides.items():
            setattr(cfg.channels, key, val)
        
        try:
            solver = NeuronSolver(cfg)
            result = solver.run_single()
            
            v_soma = result.v_soma
            rest = v_soma[0]
            peak = v_soma.max()
            
            ia_status = "✓" if cfg.channels.enable_IA else "✗"
            
            print(f"    {config_name:30} | {rest:11.1f} | {peak:11.1f} | {ia_status:8}")
            
        except Exception as e:
            print(f"    {config_name:30} | {'ERROR':>11} | {'ERROR':>11} | {'ERR':>8}")

# ============================================================================
# TEST C: Conductance Variation Test
# ============================================================================
print("\n[TEST C] Conductance Variation Stress Test")
print("-" * 100)
print("Test how neuron responds to 10-fold conductance variations")
print()

test_params = [
    ('gNa_max', 'Sodium channel conductance'),
    ('gK_max', 'Potassium channel conductance'),
    ('gL', 'Leak conductance'),
    ('gCa_max', 'Calcium channel conductance (if enabled)'),
    ('gA_max', 'A-current conductance (if enabled)'),
]

preset_for_test = 'K: Thalamic Relay (Ih + ICa + Burst)'
cfg_base = FullModelConfig()
apply_preset(cfg_base, preset_for_test)

multipliers = [0.1, 0.5, 1.0, 2.0, 10.0]

print(f"Testing {preset_for_test}")
print(f"{'Param':15} | {'0.1x':11} | {'0.5x':11} | {'1.0x (base)':11} | {'2.0x':11} | {'10x':11} | {'Status':10}")
print("-" * 110)

for param_name, param_desc in test_params:
    if not hasattr(cfg_base.channels, param_name):
        continue
    
    base_val = getattr(cfg_base.channels, param_name)
    if base_val == 0:
        continue  # Skip disabled channels
    
    results = []
    
    for mult in multipliers:
        cfg = FullModelConfig()
        apply_preset(cfg, preset_for_test)
        setattr(cfg.channels, param_name, base_val * mult)
        
        try:
            solver = NeuronSolver(cfg)
            result = solver.run_single()
            
            v_soma = result.v_soma
            peak = v_soma.max()
            
            # Check for pathology
            if -100 < peak < 80:
                results.append(f"{peak:6.1f}")
            else:
                results.append("ABNORM")
                
        except:
            results.append("ERROR")
    
    result_str = " | ".join([f"{r:11}" for r in results])
    status = "✓" if all("ABNORM" not in r and "ERROR" not in r for r in results) else "⚠"
    print(f"{param_name:15} | {result_str} | {status:10}")

# ============================================================================
# TEST D: Complex Multi-Channel Sweep
# ============================================================================
print("\n[TEST D] Complex Multi-Channel Sweep")
print("-" * 100)
print("Systematic variation of multiple channels simultaneously")
print()

cfg_base = FullModelConfig()
apply_preset(cfg_base, 'K: Thalamic Relay (Ih + ICa + Burst)')

# Create 3x3 matrix: gCa_max vs gIh_max
gCa_values = [0.04, 0.08, 0.16]  # 0.5x, 1x, 2x of default
gIh_values = [0.015, 0.03, 0.06]  # 0.5x, 1x, 2x of default

print("Matrix: V_peak (mV) for different Ih and Ca conductance combinations")
print()
print(f"{'gIh_max':>10}", end="")
for g_ca in gCa_values:
    print(f" | {'gCa=' + f'{g_ca:.2f}':>12}", end="")
print()
print("-" * 60)

for g_ih in gIh_values:
    print(f"{g_ih:10.3f}", end="")
    
    for g_ca in gCa_values:
        cfg = FullModelConfig()
        apply_preset(cfg, 'K: Thalamic Relay (Ih + ICa + Burst)')
        cfg.channels.gIh_max = g_ih
        cfg.channels.gCa_max = g_ca
        
        try:
            solver = NeuronSolver(cfg)
            result = solver.run_single()
            v_peak = result.v_soma.max()
            print(f" | {v_peak:12.1f}", end="")
        except:
            print(f" | {'ERROR':>12}", end="")
    
    print()

# ============================================================================
# TEST E: Frequency Response Test
# ============================================================================
print("\n[TEST E] Frequency Response (ISI analysis)")
print("-" * 100)
print("Testing how multi-channel combinations affect firing rate")
print()

configs = [
    ('Baseline (HH only)', {'enable_IA': False, 'enable_ICa': False, 'enable_Ih': False}),
    ('IA only', {'enable_IA': True, 'enable_ICa': False, 'enable_Ih': False}),
    ('ICa only', {'enable_IA': False, 'enable_ICa': True, 'enable_Ih': False}),
    ('Ih only', {'enable_IA': False, 'enable_ICa': False, 'enable_Ih': True}),
    ('All channels', {'enable_IA': True, 'enable_ICa': True, 'enable_Ih': True}),
]

print(f"{'Config':25} | {'Firing Rate (Hz)':>20} | {'ISI mean (ms)':>16} | {'CV ISI':>10}")
print("-" * 85)

for config_name, channel_flags in configs:
    cfg = FullModelConfig()
    apply_preset(cfg, 'K: Thalamic Relay (Ih + ICa + Burst)')
    
    for key, val in channel_flags.items():
        setattr(cfg.channels, key, val)
    
    try:
        solver = NeuronSolver(cfg)
        result = solver.run_single()
        
        v_soma = result.v_soma
        t = result.t
        
        # Find spikes
        rest = v_soma[0]
        threshold = rest + 30
        spike_times = t[v_soma > threshold]
        
        if len(spike_times) > 1:
            isi = np.diff(spike_times)
            rate_hz = 1000.0 / isi.mean()
            cv = isi.std() / isi.mean() if isi.mean() > 0 else 0
            print(f"{config_name:25} | {rate_hz:20.1f} | {isi.mean():16.2f} | {cv:10.3f}")
        else:
            print(f"{config_name:25} | {'N/A':>20} | {'N/A':>16} | {'N/A':>10}")
            
    except Exception as e:
        print(f"{config_name:25} | {'ERROR':>20} | {'ERROR':>16} | {'ERROR':>10}")

print("\n" + "=" * 100)
print("PHASE 5 COMPLETE")
print("=" * 100)
print("\nKey findings from multi-channel interactions will guide:")
print("  - Channel parameter optimization")
print("  - Identification of problematic combinations")
print("  - Guidelines for future preset design")
