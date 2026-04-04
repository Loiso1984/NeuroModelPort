"""
Check HCN channel activation curves
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.kinetics import ar_Ih, br_Ih

def check_hcn_kinetics():
    """Check HCN channel activation"""
    print("🧪 HCN Channel Activation Curves")
    print("=" * 50)
    
    # Test voltages
    v_test = np.array([-120, -100, -80, -60, -40, -20, 0, 20])
    
    print("V (mV) | r_inf | r_tau | Current direction")
    print("-" * 45)
    
    for v in v_test:
        r_inf = ar_Ih(v) / (ar_Ih(v) + br_Ih(v))
        r_tau = 1.0 / (ar_Ih(v) + br_Ih(v))
        
        # HCN is activated by hyperpolarization (more open at negative V)
        direction = "activated" if v < -78 else "deactivated"
        
        print(f"{v:6.0f} | {r_inf:5.3f} | {r_tau:6.1f} | {direction}")
    
    print("\n" + "=" * 50)
    print("Analysis:")
    print("- HCN channels activate on hyperpolarization (negative voltages)")
    print("- V_½ ≈ -78 mV (activation midpoint)")
    print("- Should be more open at rest (-70 mV) than at depolarized potentials")
    print("- More open channels = lower input resistance")
    
    # Check at resting potential
    v_rest = -70.0
    r_inf_rest = ar_Ih(v_rest) / (ar_Ih(v_rest) + br_Ih(v_rest))
    
    print(f"\nAt V_rest = {v_rest} mV:")
    print(f"  r_inf = {r_inf_rest:.3f} ({r_inf_rest*100:.1f}% open)")
    
    if r_inf_rest > 0.1:
        print("  ✅ Significant HCN activation at rest")
    else:
        print("  ⚠️ Low HCN activation at rest")

if __name__ == "__main__":
    check_hcn_kinetics()
