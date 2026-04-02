"""
Test HCN input resistance at different voltages
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver
from core.kinetics import ar_Ih, br_Ih

def test_hcn_voltage_dependence():
    """Test HCN effect at different voltages"""
    print("🧪 HCN Voltage-Dependent Input Resistance")
    print("=" * 60)
    
    # Test voltages
    v_test = np.array([-80, -70, -60, -50])  # mV
    
    print("V (mV) | r_inf | HCN open | Expected R_in effect")
    print("-" * 55)
    
    for v in v_test:
        r_inf = ar_Ih(v) / (ar_Ih(v) + br_Ih(v))
        
        if r_inf > 0.5:
            hcn_effect = "Strong shunt (↓R_in)"
        elif r_inf > 0.2:
            hcn_effect = "Moderate shunt (↓R_in)"
        elif r_inf > 0.05:
            hcn_effect = "Weak shunt (↓R_in)"
        else:
            hcn_effect = "Minimal shunt"
        
        print(f"{v:6.0f} | {r_inf:5.3f} | {r_inf*100:5.1f}% | {hcn_effect}")
    
    print("\n" + "=" * 60)
    print("Test with voltage clamp:")
    
    # Test with voltage clamp at different potentials
    for v_clamp in [-80, -70, -60]:
        print(f"\nVoltage clamp at {v_clamp} mV:")
        
        configs = [
            ("With HCN", True),
            ("Without HCN", False)
        ]
        
        for name, enable_hcn in configs:
            cfg = FullModelConfig()
            apply_preset(cfg, "K: Thalamic Relay (Ih + ICa + Burst)")
            
            cfg.channels.enable_Ih = enable_hcn
            cfg.channels.E_Ih = -30.0
            
            # Apply current to reach target voltage
            cfg.stim.stim_type = 'const'
            
            # Estimate current needed
            if enable_hcn:
                # With HCN, need more current to overcome shunt
                cfg.stim.Iext = 2.0 if v_clamp == -60 else 1.0
            else:
                # Without HCN, less current needed
                cfg.stim.Iext = 1.0 if v_clamp == -60 else 0.5
            
            solver = NeuronSolver(cfg)
            result = solver.run_single()
            
            v_soma = result.v_soma
            v_steady = v_soma[-100:].mean()
            
            print(f"  {name}: V = {v_steady:.1f} mV, I = {cfg.stim.Iext:.1f} µA/cm²")
        
        print(f"  At {v_clamp} mV, HCN should reduce R_in")

if __name__ == "__main__":
    test_hcn_voltage_dependence()
