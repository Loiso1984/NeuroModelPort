#!/usr/bin/env python3
"""
Improved Preset Calibration - Fine-tuning difficult presets

This branch focuses on the remaining problematic presets K and L.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver
from core.analysis import detect_spikes


def fine_calibrate_k_thalamic():
    """Fine-tune K: Thalamic Relay with more granular approach"""
    print("\n🔧 FINE CALIBRATING K: Thalamic Relay...")
    
    # Current: 20 spikes at gK_max=10, target: 2-8
    # Need more precise range around 15-20 where transition happens
    
    gk_values = [13, 14, 15, 16, 17, 18, 19]
    
    for gk in gk_values:
        cfg = FullModelConfig()
        apply_preset(cfg, 'K: Thalamic Relay (Ih + ICa + Burst)')
        cfg.channels.gK_max = gk
        cfg.stim.t_sim = 200.0
        cfg.stim.dt_eval = 0.2
        
        solver = NeuronSolver(cfg)
        result = solver.run_single()
        
        spike_idx, spike_times, _ = detect_spikes(
            result.v_soma, result.t, threshold=-20.0, baseline_threshold=-50.0
        )
        
        n_spikes = len(spike_times)
        firing_rate = n_spikes / (result.t[-1] / 1000.0)
        
        print(f"   gK_max={gk}: {n_spikes} spikes, {firing_rate:.1f} Hz")
        
        if 2 <= n_spikes <= 8:  # Within target range
            print(f"   ✅ OPTIMAL: gK_max={gk}")
            return {'optimal_gk': gk, 'n_spikes': n_spikes, 'rate': firing_rate}
    
    return {'optimal_gk': None, 'failed': True}


def fine_calibrate_l_ca1():
    """Fine-tune L: Hippocampal CA1 with more granular approach"""
    print("\n🔧 FINE CALIBRATING L: Hippocampal CA1...")
    
    # Current: 0 spikes at gK_max=8, target: 3-10
    # Need to find the threshold where spiking starts
    
    gk_values = [9, 10, 11, 12, 13, 14, 15]
    
    for gk in gk_values:
        cfg = FullModelConfig()
        apply_preset(cfg, 'L: Hippocampal CA1 (Theta rhythm)')
        cfg.channels.gK_max = gk
        cfg.stim.t_sim = 200.0
        cfg.stim.dt_eval = 0.2
        
        solver = NeuronSolver(cfg)
        result = solver.run_single()
        
        spike_idx, spike_times, _ = detect_spikes(
            result.v_soma, result.t, threshold=-20.0, baseline_threshold=-50.0
        )
        
        n_spikes = len(spike_times)
        firing_rate = n_spikes / (result.t[-1] / 1000.0)
        
        print(f"   gK_max={gk}: {n_spikes} spikes, {firing_rate:.1f} Hz")
        
        if 3 <= n_spikes <= 10:  # Within target range
            print(f"   ✅ OPTIMAL: gK_max={gk}")
            return {'optimal_gk': gk, 'n_spikes': n_spikes, 'rate': firing_rate}
    
    return {'optimal_gk': None, 'failed': True}


def test_calibrated_presets():
    """Test the calibrated parameters"""
    print("\n📊 TESTING CALIBRATED PRESETS:")
    
    calibrated_params = {
        'B: FS Interneuron (GABA)': {'Iext': 25},
        'K: Thalamic Relay (Ih + ICa + Burst)': {'gK_max': 17},  # From fine calibration
        'L: Hippocampal CA1 (Theta rhythm)': {'gK_max': 12}   # From fine calibration
    }
    
    for preset_name, params in calibrated_params.items():
        cfg = FullModelConfig()
        apply_preset(cfg, preset_name)
        
        # Apply calibrated parameters
        if 'Iext' in params:
            cfg.stim.Iext = params['Iext']
        if 'gK_max' in params:
            cfg.channels.gK_max = params['gK_max']
        
        cfg.stim.t_sim = 200.0
        cfg.stim.dt_eval = 0.2
        
        solver = NeuronSolver(cfg)
        result = solver.run_single()
        
        spike_idx, spike_times, _ = detect_spikes(
            result.v_soma, result.t, threshold=-20.0, baseline_threshold=-50.0
        )
        
        n_spikes = len(spike_times)
        firing_rate = n_spikes / (result.t[-1] / 1000.0)
        
        print(f"   {preset_name}: {n_spikes} spikes, {firing_rate:.1f} Hz")
        
        # Check if within target
        target_ranges = {
            'B: FS Interneuron (GABA)': (10, 50),
            'K: Thalamic Relay (Ih + ICa + Burst)': (2, 8),
            'L: Hippocampal CA1 (Theta rhythm)': (3, 10)
        }
        
        target = target_ranges[preset_name]
        if target[0] <= n_spikes <= target[1]:
            print(f"      ✅ WITHIN TARGET")
        else:
            print(f"      ❌ OUTSIDE TARGET ({target[0]}-{target[1]})")


if __name__ == "__main__":
    print("=" * 70)
    print("=" * 25 + " FINE PRESET CALIBRATION " + "=" * 25)
    print("=" * 70)
    
    # Fine calibration
    k_result = fine_calibrate_k_thalamic()
    l_result = fine_calibrate_l_ca1()
    
    print("\n" + "=" * 70)
    print("FINE CALIBRATION RESULTS:")
    print("=" * 70)
    
    if k_result.get('optimal_gk'):
        print(f"   K: ✅ gK_max={k_result['optimal_gk']}")
    else:
        print(f"   K: ❌ Still needs work")
    
    if l_result.get('optimal_gk'):
        print(f"   L: ✅ gK_max={l_result['optimal_gk']}")
    else:
        print(f"   L: ❌ Still needs work")
    
    # Test all calibrated presets
    test_calibrated_presets()
