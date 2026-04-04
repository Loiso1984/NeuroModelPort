"""
Check calcium changes with higher resolution
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset
from core.morphology import MorphologyBuilder
from core.solver import NeuronSolver

def check_calcium_resolution():
    """Check calcium changes with temporal resolution"""
    print("🔍 High Resolution Calcium Analysis")
    print("=" * 50)
    
    cfg = FullModelConfig()
    apply_preset(cfg, "K: Thalamic Relay (Ih + ICa + Burst)")
    
    # Shorter simulation for higher resolution
    cfg.stim.t_sim = 100.0  # ms
    cfg.stim.dt_eval = 0.01  # 0.01 ms for high resolution
    
    solver = NeuronSolver(cfg)
    result = solver.run_single()
    
    v_soma = result.v_soma
    soma_ca = result.ca_i[0, :]
    t = result.t
    
    print(f"Simulation:")
    print(f"  Duration: {t[-1]:.1f} ms")
    print(f"  Time points: {len(t)}")
    print(f"  dt: {t[1]-t[0]:.3f} ms")
    
    # Find first spike
    threshold = -20.0
    spike_indices = np.where(v_soma > threshold)[0]
    
    if len(spike_indices) > 0:
        spike_idx = spike_indices[0]
        spike_time = t[spike_idx]
        
        print(f"\nFirst spike:")
        print(f"  Index: {spike_idx}")
        print(f"  Time: {spike_time:.2f} ms")
        print(f"  Voltage: {v_soma[spike_idx]:.1f} mV")
        
        # High-resolution analysis around spike
        window = 50  # points
        start_idx = max(0, spike_idx - window)
        end_idx = min(len(t), spike_idx + window)
        
        print(f"\nCalcium around spike (±{window*(t[1]-t[0]):.2f} ms):")
        
        # Find min and max in window
        ca_window = soma_ca[start_idx:end_idx]
        ca_min_idx = start_idx + np.argmin(ca_window)
        ca_max_idx = start_idx + np.argmax(ca_window)
        
        ca_min = soma_ca[ca_min_idx]
        ca_max = soma_ca[ca_max_idx]
        ca_change = ca_max - ca_min
        
        print(f"  Min Ca: {ca_min*1e6:.3f} µM at t={t[ca_min_idx]:.2f} ms")
        print(f"  Max Ca: {ca_max*1e6:.3f} µM at t={t[ca_max_idx]:.2f} ms")
        print(f"  Change: {ca_change*1e6:.3f} µM")
        print(f"  Change: {ca_change:.2e} M")
        
        # Check if significant
        if abs(ca_change) > 1e-9:  # > 1 nM
            print(f"  ✅ Significant change detected!")
        else:
            print(f"  ❌ Change too small: {ca_change*1e9:.3f} nM")
        
        # Show detailed trace
        print(f"\nDetailed trace (every 5 points):")
        for i in range(start_idx, end_idx, 5):
            if i < len(t):
                print(f"  t={t[i]:6.2f}ms: V={v_soma[i]:6.1f}mV, Ca={soma_ca[i]*1e6:7.3f}µM")
    
    else:
        print("No spikes detected")

if __name__ == "__main__":
    check_calcium_resolution()
