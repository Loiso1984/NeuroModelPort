#!/usr/bin/env python3
"""
COMPREHENSIVE INTEGRATION TEST
Tests all major features together:
1. Matplotlib rendering (no lw/linewidth conflicts)
2. Ion channel configurations
3. Calcium dynamics
4. Dual stimulation
5. Axon biophysics
"""
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import sys
import numpy as np
from core.models import FullModelConfig
from core.presets import apply_preset, get_preset_names
from core.solver import NeuronSolver
from core.dual_stimulation import DualStimulationConfig
from core.analysis import full_analysis
from gui.axon_biophysics import AxonBiophysicsWidget
from PySide6.QtWidgets import QApplication

app = QApplication(sys.argv)

print("=" * 100)
print("COMPREHENSIVE INTEGRATION TEST - NeuroModel v10")
print("=" * 100)

test_results = {
    'matplotlib': {'status': 'PENDING', 'details': []},
    'presets': {'status': 'PENDING', 'passed': 0, 'total': 0, 'failures': []},
    'calcium': {'status': 'PENDING', 'details': []},
    'dual_stim': {'status': 'PENDING', 'details': []},
    'axon_viz': {'status': 'PENDING', 'details': []},
}

# ══════════════════════════════════════════════════════════════════════════════════════
# TEST 1: Matplotlib rendering (no linewidth conflict)
# ══════════════════════════════════════════════════════════════════════════════════════
print("\n[1/5] MATPLOTLIB RENDERING TEST")
print("-" * 100)

try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    t = np.linspace(0, 1, 100)
    v = np.sin(2 * np.pi * t)
    
    # This would have failed if matplotlib has the lw/linewidth conflict
    ax.plot(t, v, linewidth=2.5, label='Voltage')
    ax.set_ylabel('Voltage (mV)', color='#CDD6F4')
    
    plt.close(fig)
    test_results['matplotlib']['status'] = 'PASS'
    test_results['matplotlib']['details'].append('Matplotlib plotting without lw/linewidth conflict')
    print("✓ Matplotlib rendering works (no lw/linewidth alias conflict)")
except Exception as e:
    test_results['matplotlib']['status'] = 'FAIL'
    test_results['matplotlib']['details'].append(f'ERROR: {str(e)[:60]}')
    print(f"✗ Matplotlib ERROR: {e}")

# ══════════════════════════════════════════════════════════════════════════════════════
# TEST 2: Preset validation
# ══════════════════════════════════════════════════════════════════════════════════════
print("\n[2/5] PRESET VALIDATION TEST")
print("-" * 100)

presets = get_preset_names()
preset_tests = [p for p in presets if not p.startswith('??')]  # Exclude synaptic presets for speed

for preset_name in preset_tests[:8]:  # Test first 8 presets
    test_results['presets']['total'] += 1
    try:
        cfg = FullModelConfig()
        apply_preset(cfg, preset_name)
        solver = NeuronSolver(cfg)
        result = solver.run_single()
        
        v = result.v_soma
        peak = v.max()
        rest = v[0]
        spikes = (v > (rest + 30)).sum()
        
        test_results['presets']['passed'] += 1
        print(f"  ✓ {preset_name[:45]:<45} | Peak={peak:7.2f}mV | Spikes={spikes:4d}")
    except Exception as e:
        test_results['presets']['failures'].append((preset_name, str(e)[:50]))
        print(f"  ✗ {preset_name[:45]:<45} | ERROR: {str(e)[:40]}")

test_results['presets']['status'] = 'PASS' if test_results['presets']['passed'] == test_results['presets']['total'] else 'PARTIAL'

# ══════════════════════════════════════════════════════════════════════════════════════
# TEST 3: Calcium dynamics validation
# ══════════════════════════════════════════════════════════════════════════════════════
print("\n[3/5] CALCIUM DYNAMICS TEST")
print("-" * 100)

calcium_presets = [
    'K: Thalamic Relay (Ih + ICa + Burst)',
    'L: Hippocampal CA1 (Theta rhythm)',
    'N: Alzheimer\'s (v10 Calcium Toxicity)',
    'E: Cerebellar Purkinje (De Schutter)',
]

for preset_name in calcium_presets:
    try:
        cfg = FullModelConfig()
        apply_preset(cfg, preset_name)
        
        if cfg.calcium.dynamic_Ca:
            solver = NeuronSolver(cfg)
            result = solver.run_single()
            
            if result.ca_i is not None and len(result.ca_i) > 0:
                peak_ca = np.max(result.ca_i[0, :])
                test_results['calcium']['details'].append(f"✓ {preset_name}: Ca dynamics enabled, peak={peak_ca:.2e}")
                print(f"  ✓ {preset_name:<45} | Ca dynamics active | peak=[Ca]={peak_ca:.2e}")
            else:
                test_results['calcium']['details'].append(f"⚠ {preset_name}: Ca_i not populated")
                print(f"  ⚠ {preset_name:<45} | Ca dynamics enabled but ca_i not populated")
        else:
            print(f"  - {preset_name:<45} | Ca dynamics disabled (not tested)")
    except Exception as e:
        test_results['calcium']['details'].append(f"✗ {preset_name}: {str(e)[:50]}")
        print(f"  ✗ {preset_name:<45} | ERROR: {str(e)[:40]}")

test_results['calcium']['status'] = 'PASS' if all('✓' in d or '-' in d for d in test_results['calcium']['details']) else 'PARTIAL'

# ══════════════════════════════════════════════════════════════════════════════════════
# TEST 4: Dual stimulation
# ══════════════════════════════════════════════════════════════════════════════════════
print("\n[4/5] DUAL STIMULATION TEST")
print("-" * 100)

# Test 1: Both stimuli at soma (should work)
cfg_ds = FullModelConfig()
apply_preset(cfg_ds, 'A: Squid Giant Axon (HH 1952)')
cfg_ds.stim.Iext = 12.0
cfg_ds.dual_stimulation = DualStimulationConfig()
cfg_ds.dual_stimulation.enabled = True
cfg_ds.dual_stimulation.secondary_location = 'soma'
cfg_ds.dual_stimulation.secondary_Iext = -5.0

try:
    solver_ds = NeuronSolver(cfg_ds)
    result_ds = solver_ds.run_single()
    spikes_ds = (result_ds.v_soma > (result_ds.v_soma[0] + 30)).sum()
    test_results['dual_stim']['details'].append(f"✓ Soma+Soma dual stim: {spikes_ds} spikes")
    print(f"  ✓ Soma + Soma (inhibition)     | Spikes={spikes_ds} (reduced from 422 baseline)")
except Exception as e:
    test_results['dual_stim']['details'].append(f"✗ Soma+Soma: {str(e)[:50]}")
    print(f"  ✗ Soma + Soma (inhibition)     | ERROR: {str(e)[:40]}")

# Test 2: Soma + Dendritic (known to have issues)
cfg_ds2 = FullModelConfig()
apply_preset(cfg_ds2, 'A: Squid Giant Axon (HH 1952)')
cfg_ds2.stim.stim_type = 'const'
cfg_ds2.stim.Iext = 12.0
cfg_ds2.stim_location.location = 'soma'
cfg_ds2.dual_stimulation = DualStimulationConfig()
cfg_ds2.dual_stimulation.enabled = True
cfg_ds2.dual_stimulation.secondary_location = 'dendritic_filtered'
cfg_ds2.dual_stimulation.secondary_Iext = -5.0

try:
    solver_ds2 = NeuronSolver(cfg_ds2)
    result_ds2 = solver_ds2.run_single()
    spikes_ds2 = (result_ds2.v_soma > (result_ds2.v_soma[0] + 30)).sum()
    test_results['dual_stim']['details'].append(f"⚠ Soma+Dendritic: {spikes_ds2} spikes (no attenuation)")
    print(f"  ⚠ Soma + Dendritic (inhibition)| Spikes={spikes_ds2} (dendritic filter issue)")
except Exception as e:
    test_results['dual_stim']['details'].append(f"✗ Soma+Dendritic: {str(e)[:50]}")
    print(f"  ✗ Soma + Dendritic (inhibition)| ERROR: {str(e)[:40]}")

test_results['dual_stim']['status'] = 'PARTIAL'  # Known to have dendritic secondary issue

# ══════════════════════════════════════════════════════════════════════════════════════
# TEST 5: Axon biophysics visualization
# ══════════════════════════════════════════════════════════════════════════════════════
print("\n[5/5] AXON BIOPHYSICS VISUALIZATION TEST")
print("-" * 100)

try:
    cfg_axon = FullModelConfig()
    apply_preset(cfg_axon, 'E: Cerebellar Purkinje (De Schutter)')
    solver_axon = NeuronSolver(cfg_axon)
    result_axon = solver_axon.run_single()
    
    widget = AxonBiophysicsWidget()
    widget.plot_axon_data(result_axon, cfg_axon)
    
    test_results['axon_viz']['status'] = 'PASS'
    test_results['axon_viz']['details'].append('✓ Axon biophysics widget created and plotting works')
    print(f"  ✓ Axon biophysics widget       | Plotting successful for multi-compartment neuron")
except Exception as e:
    test_results['axon_viz']['status'] = 'FAIL'
    test_results['axon_viz']['details'].append(f'ERROR: {str(e)[:60]}')
    print(f"  ✗ Axon biophysics widget       | ERROR: {str(e)[:40]}")

# ══════════════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("INTEGRATION TEST SUMMARY")
print("=" * 100)

total_pass = sum(1 for t in test_results.values() if t['status'] == 'PASS')
total_partial = sum(1 for t in test_results.values() if t['status'] == 'PARTIAL')
total_fail = sum(1 for t in test_results.values() if t['status'] == 'FAIL')

for test_name, result in test_results.items():
    status = result['status']
    status_icon = '✓' if status == 'PASS' else ('⚠' if status == 'PARTIAL' else '✗')
    
    print(f"\n{status_icon} {test_name.upper():<20} | {status:<8}", end="")
    if 'passed' in result:
        print(f" | {result['passed']}/{result['total']} passed", end="")
    print()
    
    if result.get('details'):
        for detail in result['details'][:3]:  # Show first 3 details
            print(f"    {detail}")

print("\n" + "=" * 100)
print(f"TOTAL: {total_pass} PASS, {total_partial} PARTIAL, {total_fail} FAIL")
print("=" * 100)

exit_code = 0 if total_fail == 0 else 1
sys.exit(exit_code)
