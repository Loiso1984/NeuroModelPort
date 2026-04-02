"""
Debug HCN current and input resistance
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver

def debug_hcn_current():
    """Debug HCN current and input resistance"""
    print("🔍 Debug: HCN Current and Input Resistance")
    print("=" * 60)
    
    # Test with and without HCN
    configs = [
        ("With HCN", True),
        ("Without HCN", False)
    ]
    
    for name, enable_hcn in configs:
        print(f"\n{name}:")
        
        cfg = FullModelConfig()
        apply_preset(cfg, "K: Thalamic Relay (Ih + ICa + Burst)")
        
        cfg.channels.enable_Ih = enable_hcn
        cfg.channels.gIh_max = 0.03 if enable_hcn else 0.0
        
        print(f"  enable_Ih: {cfg.channels.enable_Ih}")
        print(f"  gIh_max: {cfg.channels.gIh_max}")
        
        # Test 1: Resting potential (no current)
        cfg.stim.Iext = 0.0
        cfg.stim.stim_type = 'const'
        
        solver = NeuronSolver(cfg)
        result = solver.run_single()
        
        v_soma = result.v_soma
        v_rest = v_soma[-100:].mean()
        
        print(f"  V_rest (0 µA): {v_rest:.1f} mV")
        
        # Test 2: Hyperpolarizing current
        cfg.stim.Iext = -0.5  # µA/cm²
        
        solver = NeuronSolver(cfg)
        result = solver.run_single()
        
        v_soma = result.v_soma
        v_hyper = v_soma[-100:].mean()
        
        print(f"  V_hyper (-0.5 µA): {v_hyper:.1f} mV")
        
        # Calculate input resistance
        delta_v = v_hyper - v_rest
        r_input = abs(delta_v / 0.5)  # MΩ
        
        print(f"  ΔV = {delta_v:.1f} mV")
        print(f"  R_in = {r_input:.1f} MΩ")
        
        # Test 3: Depolarizing current
        cfg.stim.Iext = 0.5  # µA/cm²
        
        solver = NeuronSolver(cfg)
        result = solver.run_single()
        
        v_soma = result.v_soma
        v_depol = v_soma[-100:].mean()
        
        print(f"  V_depol (+0.5 µA): {v_depol:.1f} mV")
        
        # Check if HCN is working as expected
        if enable_hcn:
            print(f"  Expected: HCN should reduce R_in and stabilize V")
        else:
            print(f"  Expected: Higher R_in, larger voltage changes")
    
    print("\n" + "=" * 60)
    print("Analysis:")
    print("- HCN channels open at hyperpolarized potentials")
    print("- More open channels = more inward current = depolarization")
    print("- This should reduce input resistance (shunting effect)")
    print("- If R_in is HIGHER with HCN, there's a problem")

if __name__ == "__main__":
    debug_hcn_current()
