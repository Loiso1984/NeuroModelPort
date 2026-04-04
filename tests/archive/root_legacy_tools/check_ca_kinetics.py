"""
Check calcium channel activation curves
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.kinetics import as_Ca, bs_Ca, au_Ca, bu_Ca

def check_ca_kinetics():
    """Check calcium channel activation"""
    print("🧪 Calcium Channel Activation Curves")
    print("=" * 50)
    
    # Test voltages
    v_test = np.array([-80, -60, -40, -20, 0, 20, 40, 60])
    
    print("V (mV) | s_inf | u_inf | s_tau | u_tau | Product")
    print("-" * 55)
    
    for v in v_test:
        s_inf = as_Ca(v) / (as_Ca(v) + bs_Ca(v))
        u_inf = au_Ca(v) / (au_Ca(v) + bu_Ca(v))
        s_tau = 1.0 / (as_Ca(v) + bs_Ca(v))
        u_tau = 1.0 / (au_Ca(v) + bu_Ca(v))
        product = s_inf**2 * u_inf  # (s^2) * u
        
        print(f"{v:6.0f} | {s_inf:5.3f} | {u_inf:5.3f} | {s_tau:6.1f} | {u_tau:6.1f} | {product:7.4f}")
    
    print("\n" + "=" * 50)
    print("Analysis:")
    print("- s gate: activation (opens with depolarization)")
    print("- u gate: inactivation (closes with depolarization)")
    print("- Current = gCa * s^2 * u * (V - E_Ca)")
    print("\nAt spike peak (V ≈ 47 mV):")
    
    v_spike = 47.0
    s_inf_spike = as_Ca(v_spike) / (as_Ca(v_spike) + bs_Ca(v_spike))
    u_inf_spike = au_Ca(v_spike) / (au_Ca(v_spike) + bu_Ca(v_spike))
    product_spike = s_inf_spike**2 * u_inf_spike
    
    print(f"  s_inf = {s_inf_spike:.3f}")
    print(f"  u_inf = {u_inf_spike:.3f}")
    print(f"  s^2 * u = {product_spike:.4f}")
    
    if product_spike < 0.01:
        print("  ❌ Very low activation - channels barely open!")
    elif product_spike < 0.1:
        print("  ⚠️ Low activation - channels partially open")
    else:
        print("  ✅ Good activation - channels open")

if __name__ == "__main__":
    check_ca_kinetics()
