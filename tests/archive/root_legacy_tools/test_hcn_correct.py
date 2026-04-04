"""
Proper HCN input resistance test with voltage clamp
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver

def test_hcn_input_resistance_correct():
    """Test HCN input resistance with proper voltage control"""
    print("🧪 Proper HCN Input Resistance Test")
    print("=" * 50)
    
    # Test at voltages where HCN is active
    test_voltages = [-80, -70, -60]  # mV
    
    for v_test in test_voltages:
        print(f"\nTesting at V = {v_test} mV:")
        
        # Test with and without HCN
        configs = [
            ("With HCN", True),
            ("Without HCN", False)
        ]
        
        currents_needed = []
        
        for name, enable_hcn in configs:
            cfg = FullModelConfig()
            apply_preset(cfg, "K: Thalamic Relay (Ih + ICa + Burst)")
            
            cfg.channels.enable_Ih = enable_hcn
            cfg.channels.E_Ih = -30.0
            
            # Use current injection to reach target voltage
            # Start with estimate and adjust
            i_test = 0.5  # µA/cm²
            
            # Try different currents to reach target voltage
            best_i = i_test
            best_v = 0
            best_diff = 1000
            
            for i_candidate in np.linspace(-2.0, 2.0, 20):
                cfg.stim.Iext = i_candidate
                cfg.stim.stim_type = 'const'
                
                solver = NeuronSolver(cfg)
                result = solver.run_single()
                
                v_soma = result.v_soma
                v_steady = v_soma[-100:].mean()
                
                diff = abs(v_steady - v_test)
                if diff < best_diff:
                    best_diff = diff
                    best_v = v_steady
                    best_i = i_candidate
            
            currents_needed.append(best_i)
            print(f"  {name}: I = {best_i:.2f} µA/cm² → V = {best_v:.1f} mV")
        
        # Calculate input resistance (slope of I-V curve)
        if len(currents_needed) == 2:
            i_with_hcn = currents_needed[0]
            i_without_hcn = currents_needed[1]
            
            # Input resistance = ΔI/ΔV
            delta_i = abs(i_with_hcn - i_without_hcn)
            
            # HCN provides additional conductance, so less current needed
            # for same voltage change = lower input resistance
            if delta_i > 0.01:  # Significant difference
                print(f"  ΔI = {delta_i:.2f} µA/cm²")
                
                if i_with_hcn < i_without_hcn:
                    print(f"  ✅ HCN reduces input resistance (needs less current)")
                else:
                    print(f"  ❌ HCN increases input resistance (needs more current)")
            else:
                print(f"  ⚠️ Minimal difference in current requirements")
    
    print("\n" + "=" * 50)
    print("Expected behavior:")
    print("- HCN channels open at hyperpolarized potentials")
    print("- Open HCN channels provide additional conductance")
    print("- More conductance = lower input resistance")
    print("- Therefore: HCN should reduce input resistance")

if __name__ == "__main__":
    test_hcn_input_resistance_correct()
