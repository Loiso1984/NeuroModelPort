"""
Test HCN with different reversal potentials
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver

def test_hcn_reversal():
    """Test HCN with different E_Ih values"""
    print("🧪 Testing HCN Reversal Potentials")
    print("=" * 50)
    
    # Test different E_Ih values
    eih_values = [-43.0, -30.0, -20.0]  # Current, more physiological
    
    for eih in eih_values:
        print(f"\nE_Ih = {eih} mV:")
        
        cfg = FullModelConfig()
        apply_preset(cfg, "K: Thalamic Relay (Ih + ICa + Burst)")
        
        cfg.channels.E_Ih = eih
        cfg.channels.enable_Ih = True
        cfg.channels.gIh_max = 0.03
        
        # Test with small hyperpolarizing current
        cfg.stim.Iext = -0.05
        cfg.stim.stim_type = 'const'
        
        solver = NeuronSolver(cfg)
        result = solver.run_single()
        
        v_soma = result.v_soma
        v_rest = v_soma[:100].mean()
        v_hyper = v_soma[-100:].mean()
        
        delta_v = v_hyper - v_rest
        r_input = abs(delta_v / 0.05)  # MΩ
        
        print(f"  V_rest: {v_rest:.1f} mV")
        print(f"  V_hyper: {v_hyper:.1f} mV")
        print(f"  ΔV: {delta_v:.2f} mV")
        print(f"  R_in: {r_input:.1f} MΩ")
        
        # Calculate expected current at rest
        v_diff = v_rest - eih
        print(f"  V - E_Ih: {v_diff:.1f} mV")
        
        if v_diff < 0:
            print(f"  Current direction: inward (depolarizing)")
        else:
            print(f"  Current direction: outward (hyperpolarizing)")
    
    print("\n" + "=" * 50)
    print("Analysis:")
    print("- More negative E_Ih = stronger depolarizing drive")
    print("- E_Ih = -20 to -30 mV typical for HCN")
    print("- Should reduce input resistance (shunting)")

if __name__ == "__main__":
    test_hcn_reversal()
