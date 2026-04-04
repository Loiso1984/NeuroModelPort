"""
Debug calcium current direction during spike
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset
from core.morphology import MorphologyBuilder
from core.solver import NeuronSolver
from core.rhs import nernst_ca_ion

def debug_current_direction():
    """Debug calcium current direction"""
    print("🔍 Debug: Calcium Current Direction")
    print("=" * 50)
    
    # Setup
    cfg = FullModelConfig()
    apply_preset(cfg, "K: Thalamic Relay (Ih + ICa + Burst)")
    
    # Calculate E_Ca
    ca_rest = cfg.calcium.Ca_rest
    ca_ext = cfg.calcium.Ca_ext
    t_kelvin = cfg.env.T_celsius + 273.15
    e_ca = nernst_ca_ion(ca_rest, ca_ext, t_kelvin)
    
    print(f"Physiology:")
    print(f"  E_Ca = {e_ca:.1f} mV")
    print(f"  [Ca²⁺]ᵢ = {ca_rest*1e6:.1f} nM")
    print(f"  [Ca²⁺]ₑₓₜ = {ca_ext:.1f} mM")
    
    # Run simulation
    solver = NeuronSolver(cfg)
    result = solver.run_single()
    
    # Find spike
    v_soma = result.v_soma
    spike_idx = np.argmax(v_soma)
    
    print(f"\nSpike analysis:")
    print(f"  Spike at index: {spike_idx}")
    print(f"  V_max: {v_soma[spike_idx]:.1f} mV")
    print(f"  V - E_Ca: {v_soma[spike_idx] - e_ca:.1f} mV")
    
    # Check calcium around spike
    if hasattr(result, 'ca_i') and result.ca_i is not None:
        soma_ca = result.ca_i[0, :]
        
        print(f"\nCalcium around spike:")
        for offset in [-10, -5, 0, 5, 10]:
            idx = spike_idx + offset
            if 0 <= idx < len(soma_ca):
                ca = soma_ca[idx]
                v = v_soma[idx]
                v_diff = v - e_ca
                current_dir = "inward" if v_diff < 0 else "outward"
                print(f"  t-{offset:2d}: V={v:6.1f}mV, V-E_Ca={v_diff:7.1f}mV ({current_dir}), Ca={ca*1e6:6.2f}µM")
        
        # Check trend
        pre_spike = soma_ca[spike_idx-5:spike_idx].mean()
        post_spike = soma_ca[spike_idx:spike_idx+5].mean()
        change = post_spike - pre_spike
        
        print(f"\nTrend analysis:")
        print(f"  Pre-spike mean: {pre_spike*1e6:.2f} µM")
        print(f"  Post-spike mean: {post_spike*1e6:.2f} µM")
        print(f"  Change: {change*1e6:.2f} µM")
        print(f"  Direction: {'Increase' if change > 0 else 'Decrease'}")
        
        # Physics check
        print(f"\nPhysics check:")
        if v_soma[spike_idx] < e_ca:
            print(f"  V ({v_soma[spike_idx]:.1f}) < E_Ca ({e_ca:.1f}) → Inward current expected")
            if change < 0:
                print(f"  ❌ Calcium decreases when influx expected!")
            else:
                print(f"  ✅ Calcium increases as expected")
        else:
            print(f"  V ({v_soma[spike_idx]:.1f}) > E_Ca ({e_ca:.1f}) → Outward current expected")
            if change < 0:
                print(f"  ✅ Calcium decreases as expected")
            else:
                print(f"  ❌ Calcium increases when efflux expected!")

if __name__ == "__main__":
    debug_current_direction()
