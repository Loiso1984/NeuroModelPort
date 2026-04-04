#!/usr/bin/env python3
"""
Test all neuron presets with calcium dynamics enabled.
Validates: Alzheimer's, Hypoxia, Thalamic, Purkinje, Hippocampal CA1
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver

def test_preset(name: str, duration_ms: float = 500.0) -> dict:
    """Test a preset and return metrics."""
    try:
        cfg = FullModelConfig()
        apply_preset(cfg, name)
        
        # Set simulation duration
        cfg.stim.t_sim = duration_ms
        
        # Create solver and simulate
        solver = NeuronSolver(cfg)
        result = solver.run_single()
        
        # Calculate metrics
        V_soma = result.v_soma
        peak_V = np.max(V_soma)
        min_V = np.min(V_soma)
        
        # Count action potentials (threshold: -20 mV)
        threshold = -20.0
        crossings = np.sum(np.diff(np.sign(V_soma - threshold)) > 0)
        spikes = crossings
        
        # Calcium metrics
        peak_Ca = 0.0
        if result.ca_i is not None and len(result.ca_i) > 0:
            peak_Ca = np.max(result.ca_i[0, :]) * 1e9  # convert to nM
        
        return {
            'name': name,
            'status': '✓ PASS',
            'peak_V': peak_V,
            'min_V': min_V,
            'spikes': spikes,
            'peak_Ca_nM': peak_Ca,
            'dynamic_Ca': cfg.calcium.dynamic_Ca,
            'tau_Ca': cfg.calcium.tau_Ca,
            'B_Ca': cfg.calcium.B_Ca,
        }
    except Exception as e:
        import traceback
        return {
            'name': name,
            'status': f'✗ FAIL: {str(e)[:60]}',
            'peak_V': None,
            'min_V': None,
            'spikes': None,
            'peak_Ca_nM': None,
        }

# Test all calcium-enabled presets
test_presets = [
    "Alzheimer's (v10 Calcium Toxicity)",
    "Cerebellar Purkinje (De Schutter)",
    "Hippocampal CA1 (Theta rhythm)",
    "Hypoxia (v10 ATP-pump failure)",
    "Thalamic Relay (Ih + ICa + Burst)",
]

print("=" * 100)
print("TESTING ALL CALCIUM-ENABLED PRESETS")
print("=" * 100)

results = []
for preset_name in test_presets:
    result = test_preset(preset_name)
    results.append(result)
    
    status = result['status']
    name = result['name']
    peak_V = result.get('peak_V')
    spikes = result.get('spikes')
    
    if peak_V is not None and spikes is not None:
        print(f"\n{name}")
        print(f"  Peak: {peak_V:7.2f}mV | Spikes: {spikes:5d} | {status}")
    else:
        print(f"\n{name}")
        print(f"  {status}")

print("\n" + "=" * 100)
print(f"SUMMARY: {sum(1 for r in results if '✓' in r['status'])}/{len(results)} tests passed")
print("=" * 100)
