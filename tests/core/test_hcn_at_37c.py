"""
HCN Validation - With Correct Temperature (37°C)
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver

print("=" * 70)
print("🧪 HCN VALIDATION AT 37°C (CORRECT TEMPERATURE)")
print("=" * 70)

hcn_presets = [
    "K: Thalamic Relay (Ih + ICa + Burst)",
    "L: Hippocampal CA1 (Theta rhythm)"
]

for preset_name in hcn_presets:
    print(f"\n{'=' * 70}")
    print(f"Testing: {preset_name}")
    print('=' * 70)
    
    cfg = FullModelConfig()
    apply_preset(cfg, preset_name)
    
    # Fix temperature to 37°C (mammalian)
    cfg.env.T_celsius = 37.0
    cfg.env.T_ref = 37.0
    
    # Compute phi_T manually
    Q10 = cfg.env.Q10
    phi_T = Q10 ** ((cfg.env.T_celsius - cfg.env.T_ref) / 10.0)
    
    print(f"\nTemperature:")
    print(f"  T_celsius: {cfg.env.T_celsius}°C")
    print(f"  T_ref: {cfg.env.T_ref}°C")
    print(f"  phi_T: {phi_T:.6f}")
    
    print(f"\nConfiguration:")
    print(f"  enable_Ih: {cfg.channels.enable_Ih}")
    print(f"  enable_ICa: {cfg.channels.enable_ICa}")
    print(f"  gIh_max: {cfg.channels.gIh_max}")
    print(f"  Iext: {cfg.stim.Iext}")
    
    # Test 1: Resting potential (no stimulus)
    print(f"\nTest 1: Resting potential (Iext=0)...")
    cfg.stim.Iext = 0.0
    cfg.stim.stim_type = 'const'
    
    try:
        solver = NeuronSolver(cfg)
        result = solver.run_single()
        
        v_soma = result.v_soma
        v_rest = v_soma[-100:].mean()
        v_std = v_soma[-100:].std()
        
        print(f"  ✅ Completed")
        print(f"     V_rest = {v_rest:.2f} ± {v_std:.2f} mV")
        
        if -75 <= v_rest <= -50:
            print(f"     ✅ V_rest in physiological range")
        else:
            print(f"     ⚠️  V_rest outside typical range")
            
    except Exception as e:
        print(f"  ❌ ERROR: {str(e)[:60]}")
    
    # Test 2: With stimulus
    print(f"\nTest 2: Response to stimulus (Iext=10 µA)...")
    cfg.stim.Iext = 10.0
    cfg.stim.stim_type = 'const'
    
    try:
        solver = NeuronSolver(cfg)
        result = solver.run_single()
        
        v_soma = result.v_soma
        peak = v_soma.max()
        
        # Count spikes (threshold = rest + 30mV)
        rest = v_soma[0]
        threshold = rest + 30
        spikes = (v_soma > threshold).sum()
        
        print(f"  ✅ Completed")
        print(f"     Peak: {peak:.2f} mV")
        print(f"     Spikes: {spikes}")
        
    except Exception as e:
        print(f"  ❌ ERROR: {str(e)[:60]}")

print("\n" + "=" * 70)
print("✅ SUMMARY: All tests at 37°C")
print("=" * 70)
