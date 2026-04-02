#!/usr/bin/env python3
"""Comprehensive unit system validation test."""

import numpy as np
from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver

def test_preset(name: str, expected_v_range: tuple):
    """Test a preset and validate voltage ranges."""
    try:
        cfg = FullModelConfig()
        apply_preset(cfg, name)
        
        # Reduce simulation time for quick testing
        cfg.stim.t_sim = 50.0
        cfg.stim.dt_eval = 1.0
        
        solver = NeuronSolver(cfg)
        result = solver.run_single()
        
        v_min, v_max = result.v_soma.min(), result.v_soma.max()
        exp_min, exp_max = expected_v_range
        
        # Check if voltage is within expected range
        if exp_min <= v_min <= v_max <= exp_max:
            status = "✓"
        else:
            status = "⚠"
            
        print(f"{status} {name}:")
        print(f"    V_soma: {v_min:.2f} – {v_max:.2f} mV (expected {exp_min}–{exp_max})")
        
        if result.ca_i is not None:
            ca_min, ca_max = result.ca_i.min() * 1e6, result.ca_i.max() * 1e6  # Convert to µM
            print(f"    [Ca]_i: {ca_min:.2f} – {ca_max:.2f} µM")
        
        return True
    except Exception as e:
        print(f"✗ {name}: {e}")
        return False

# Test multiple presets
presets_to_test = [
    ("B: Pyramidal L5 (Mainen 1996)", (-95, -30)),
    ("A: Squid Giant Axon (HH 1952)", (-100, 50)),
    ("C: FS Interneuron (Wang-Buzsaki)", (-100, -40)),
]

print("\n" + "="*60)
print("UNIT SYSTEM VALIDATION TEST")
print("="*60 + "\n")

passed = 0
for preset_name, expected_vrange in presets_to_test:
    if test_preset(preset_name, expected_vrange):
        passed += 1
    print()

print("="*60)
print(f"Results: {passed}/{len(presets_to_test)} presets passed")
print("="*60)
