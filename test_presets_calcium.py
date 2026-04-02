#!/usr/bin/env python3
"""Test presets with updated calcium dynamics"""
from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver

test_presets = [
    ('N: Alzheimer\'s (v10 Calcium Toxicity)', 'Model calcium toxicity'),
    ('B: Pyramidal L5 (Mainen 1996)', 'Control: standard pyramidal'),
    ('O: Hypoxia (v10 ATP-pump failure)', 'Model calcium overload'),
    ('K: Thalamic Relay (Ih + ICa + Burst)', 'Existing thalamic'),
]

print('='*90)
print('TESTING UPDATED PRESETS WITH CALCIUM DYNAMICS')
print('='*90)

for preset_name, description in test_presets:
    try:
        config = FullModelConfig()
        apply_preset(config, preset_name)
        
        # Get calcium settings
        ca_enabled = config.calcium.dynamic_Ca
        ica_enabled = config.channels.enable_ICa
        sk_enabled = config.channels.enable_SK
        tau_ca = config.calcium.tau_Ca
        b_ca = config.calcium.B_Ca
        gca_max = config.channels.gCa_max
        
        # Run simulation
        solver = NeuronSolver(config)
        result = solver.run_single()
        
        v_soma = result.v_soma
        peak = v_soma.max()
        base = v_soma[0]
        spike_count = (v_soma > (base + 20)).sum()
        
        # Print results
        print(f'\n{preset_name:45}')
        print(f'  Description: {description}')
        print(f'  Ca-dynamics={ca_enabled:5} | ICa={ica_enabled:5} | SK={sk_enabled:5} | '
              f'tau_Ca={tau_ca:7.0f}ms | B_Ca={b_ca:.2f} | gCa={gca_max:.2f}')
        print(f'  → Peak: {peak:7.2f}mV | Spikes: {spike_count:3d} | ✓ PASS')
        
    except Exception as e:
        print(f'\n{preset_name:45}')
        print(f'  ✗ FAIL: {str(e)[:80]}')

print('\n' + '='*90)
print('Test completed')
