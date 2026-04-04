"""
Debug calcium influx in test_current_balance
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset
from core.morphology import MorphologyBuilder
from core.solver import NeuronSolver

def debug_current_balance():
    """Debug current balance test"""
    print("🔍 Debug: Current Balance Test")
    print("=" * 50)
    
    # Reproduce test_current_balance exactly
    cfg = FullModelConfig()
    apply_preset(cfg, "K: Thalamic Relay (Ih + ICa + Burst)")
    
    print(f"Config:")
    print(f"  enable_ICa: {cfg.channels.enable_ICa}")
    print(f"  gCa_max: {cfg.channels.gCa_max}")
    print(f"  dynamic_Ca: {cfg.calcium.dynamic_Ca}")
    print(f"  B_Ca: {cfg.calcium.B_Ca}")
    
    # Build morphology
    morph = MorphologyBuilder.build(cfg)
    
    # Create solver
    solver = NeuronSolver(cfg)
    
    # Run simulation (same as test)
    result = solver.run_single()
    
    print(f"\nSimulation results:")
    print(f"  V_max: {result.v_soma.max():.1f} mV")
    print(f"  V_min: {result.v_soma.min():.1f} mV")
    
    # Check calcium data
    if hasattr(result, 'ca_i') and result.ca_i is not None:
        ca_i_trace = result.ca_i
        print(f"  ca_i shape: {ca_i_trace.shape}")
        print(f"  ca_i range: {ca_i_trace.min():.2e} to {ca_i_trace.max():.2e} M")
        
        # Detect spikes (same as test)
        v_soma = result.v_soma
        threshold = -20.0  # mV
        spike_indices = np.where(v_soma > threshold)[0]
        print(f"  Spike count: {len(spike_indices)}")
        
        if len(spike_indices) > 5:
            # Analyze same spike as test
            spike_idx = spike_indices[2]  # third spike
            soma_ca_trace = ca_i_trace[0, :]  # Soma
            
            pre_spike_ca = soma_ca_trace[max(0, spike_idx-5)]
            post_spike_ca = soma_ca_trace[min(len(soma_ca_trace)-1, spike_idx+5)]
            
            ca_increase = post_spike_ca - pre_spike_ca
            
            print(f"\nSpike analysis:")
            print(f"  Spike index: {spike_idx}")
            print(f"  Pre-spike Ca: {pre_spike_ca:.2e} M")
            print(f"  Post-spike Ca: {post_spike_ca:.2e} M")
            print(f"  Ca increase: {ca_increase:.2e} M")
            print(f"  Ca increase: {ca_increase*1e6:.2f} µM")
            
            # Check if this meets test threshold
            if ca_increase < 1e-9:
                print(f"  ❌ Test threshold: {ca_increase:.2e} < 1e-9")
            else:
                print(f"  ✅ Test threshold: {ca_increase:.2e} >= 1e-9")
        else:
            print(f"  ❌ Not enough spikes: {len(spike_indices)} <= 5")
    else:
        print(f"  ❌ No calcium data in results")

if __name__ == "__main__":
    debug_current_balance()
