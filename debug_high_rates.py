"""
Debug high firing rates in HCN simulations
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver
from core.kinetics import ar_Ih, br_Ih

def debug_high_firing_rates():
    """Debug why HCN simulations show 2000+ Hz firing"""
    print("🔍 Debug: High Firing Rates Analysis")
    print("=" * 60)
    
    # Test Thalamic Relay preset
    cfg = FullModelConfig()
    apply_preset(cfg, "K: Thalamic Relay (Ih + ICa + Burst)")
    
    print("Configuration:")
    print(f"  gNa_max: {cfg.channels.gNa_max}")
    print(f"  gK_max: {cfg.channels.gK_max}")
    print(f"  gL: {cfg.channels.gL}")
    print(f"  gIh_max: {cfg.channels.gIh_max}")
    print(f"  gCa_max: {cfg.channels.gCa_max}")
    print(f"  Iext: {cfg.stim.Iext}")
    print(f"  stim_type: {cfg.stim.stim_type}")
    print(f"  alpha_tau: {cfg.stim.alpha_tau}")
    
    # Run simulation
    solver = NeuronSolver(cfg)
    result = solver.run_single()
    
    # Analyze firing pattern
    v_soma = result.v_soma
    t = result.t
    
    # Count spikes
    threshold = -20.0
    spike_indices = np.where(v_soma > threshold)[0]
    
    if len(spike_indices) > 0:
        # Calculate firing rate
        sim_duration = t[-1]
        firing_rate = len(spike_indices) / (sim_duration / 1000.0)
        
        # Calculate ISI distribution
        if len(spike_indices) > 1:
            isi_times = t[spike_indices[1:]] - t[spike_indices[:-1]]
            isi_mean = isi_times.mean()
            isi_cv = isi_times.std() / isi_mean if isi_mean > 0 else 0
            
            print(f"\nFiring Analysis:")
            print(f"  Duration: {sim_duration:.1f} ms")
            print(f"  Spike count: {len(spike_indices)}")
            print(f"  Firing rate: {firing_rate:.1f} Hz")
            print(f"  Mean ISI: {isi_mean:.2f} ms")
            print(f"  ISI CV: {isi_cv:.3f}")
            
            # Check if this is realistic
            if firing_rate > 100:
                print(f"  ❌ UNREALISTIC: {firing_rate:.1f} Hz > 100 Hz")
                
                # Check for multiple spikes per depolarization
                print(f"\nSpike pattern analysis:")
                print(f"  First 20 spike times: {t[spike_indices[:20]]}")
                
                # Check voltage levels
                v_max = v_soma.max()
                v_min = v_soma.min()
                v_rest = v_soma[:100].mean()
                
                print(f"\nVoltage analysis:")
                print(f"  V_max: {v_max:.1f} mV")
                print(f"  V_min: {v_min:.1f} mV")
                print(f"  V_rest: {v_rest:.1f} mV")
                
                # Check for depolarization block
                if v_rest > -50:
                    print(f"  ⚠️ Depolarized rest: {v_rest:.1f} mV")
                
                # Check stimulus
                if cfg.stim.stim_type == 'const':
                    print(f"  Constant stimulus: {cfg.stim.Iext} µA/cm²")
                    if cfg.stim.Iext > 10:
                        print(f"  ❌ STIMULUS TOO HIGH: {cfg.stim.Iext} µA/cm²")
            else:
                print(f"  ✅ Realistic: {firing_rate:.1f} Hz")
        else:
            print(f"  Only {len(spike_indices)} spike(s)")
    else:
        print("  No spikes detected")
    
    # Test with reduced stimulus
    print(f"\n" + "=" * 60)
    print("Testing with reduced stimulus:")
    
    test_currents = [30.0, 15.0, 10.0, 5.0, 2.0, 1.0]
    
    for i_test in test_currents:
        cfg_test = FullModelConfig()
        apply_preset(cfg_test, "K: Thalamic Relay (Ih + ICa + Burst)")
        
        cfg_test.stim.Iext = i_test
        cfg_test.stim.stim_type = 'const'
        
        solver_test = NeuronSolver(cfg_test)
        result_test = solver_test.run_single()
        
        v_test = result_test.v_soma
        spikes_test = np.where(v_test > threshold)[0]
        
        rate_test = len(spikes_test) / (result_test.t[-1] / 1000.0)
        v_max_test = v_test.max()
        
        print(f"  Iext = {i_test:4.1f} µA/cm²: {rate_test:6.1f} Hz, V_max = {v_max_test:5.1f} mV")
        
        # Stop when we get reasonable rates
        if 4 <= rate_test <= 12:
            print(f"  ✅ THETA RANGE: {rate_test:.1f} Hz")
            break

if __name__ == "__main__":
    debug_high_firing_rates()
