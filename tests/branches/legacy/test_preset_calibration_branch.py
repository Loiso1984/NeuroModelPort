#!/usr/bin/env python3
"""
Preset Calibration Test Branch

Tests and calibrates preset parameters to achieve physiological firing rates.
This is a TEST BRANCH - changes should be moved to main presets after validation.
"""

import sys
import numpy as np
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver
from core.analysis import detect_spikes


class PresetCalibrationTestBranch:
    """Test branch for calibrating preset parameters"""
    
    def __init__(self):
        self.results = {}
        self.physiological_targets = {
            'A: Classic Squid (HH 1952)': {'spikes': (5, 20), 'rate': (25, 100)},
            'B: FS Interneuron (GABA)': {'spikes': (10, 50), 'rate': (50, 250)},
            'C: α-Motoneuron (Soma)': {'spikes': (3, 15), 'rate': (15, 75)},
            'D: AIS (Axon Initial Segment)': {'spikes': (5, 25), 'rate': (25, 125)},
            'E: Dendritic Filter (EPSP)': {'spikes': (0, 5), 'rate': (0, 25)},
            'G: Purkinje Cell (Cerebellar)': {'spikes': (2, 10), 'rate': (10, 50)},
            'H: Layer 5 Pyramidal (Cortex)': {'spikes': (3, 12), 'rate': (15, 60)},
            'I: O-LM Interneuron (Hippocampus)': {'spikes': (5, 20), 'rate': (25, 100)},
            'K: Thalamic Relay (Ih + ICa + Burst)': {'spikes': (2, 8), 'rate': (10, 40)},
            'L: Hippocampal CA1 (Theta rhythm)': {'spikes': (3, 10), 'rate': (15, 50)},
            'M: Epilepsy (v10 SCN1A mutation)': {'spikes': (20, 100), 'rate': (100, 500)},
            'N: Alzheimer\'s (v10 Calcium Toxicity)': {'spikes': (0, 5), 'rate': (0, 25)},
            'O: Hypoxia (v10 ATP-pump failure)': {'spikes': (0, 3), 'rate': (0, 15)}
        }
    
    def test_current_preset(self, preset_name: str) -> dict:
        """Test current preset parameters"""
        cfg = FullModelConfig()
        apply_preset(cfg, preset_name)
        cfg.stim.t_sim = 200.0
        cfg.stim.dt_eval = 0.2
        
        solver = NeuronSolver(cfg)
        result = solver.run_single()
        
        spike_idx, spike_times, spike_amps = detect_spikes(
            result.v_soma, result.t, threshold=-20.0, baseline_threshold=-50.0
        )
        
        n_spikes = len(spike_times)
        firing_rate = n_spikes / (result.t[-1] / 1000.0)  # Hz
        rest_v = np.mean(result.v_soma[:100])  # First 100 points
        peak_v = np.max(result.v_soma)
        
        return {
            'n_spikes': n_spikes,
            'firing_rate': firing_rate,
            'rest_v': rest_v,
            'peak_v': peak_v,
            'gNa_max': cfg.channels.gNa_max,
            'gK_max': cfg.channels.gK_max,
            'Iext': cfg.stim.Iext,
            'within_target': self._check_target(preset_name, n_spikes, firing_rate)
        }
    
    def _check_target(self, preset_name: str, n_spikes: int, firing_rate: float) -> bool:
        """Check if within physiological target range"""
        if preset_name not in self.physiological_targets:
            return True
        
        target = self.physiological_targets[preset_name]
        spike_range = target['spikes']
        rate_range = target['rate']
        
        return spike_range[0] <= n_spikes <= spike_range[1] and \
               rate_range[0] <= firing_rate <= rate_range[1]
    
    def calibrate_b_fs_interneuron(self) -> dict:
        """Calibrate B: FS Interneuron - reduce Iext from 40 to physiological range"""
        print("\n🔧 Calibrating B: FS Interneuron (GABA)...")
        
        # Current: 27 spikes (target: 10-50), Iext=40 too high
        # Try reducing Iext
        iext_values = [20, 25, 30, 35]
        
        for iext in iext_values:
            cfg = FullModelConfig()
            apply_preset(cfg, 'B: FS Interneuron (GABA)')
            cfg.stim.Iext = iext
            cfg.stim.t_sim = 200.0
            cfg.stim.dt_eval = 0.2
            
            solver = NeuronSolver(cfg)
            result = solver.run_single()
            
            spike_idx, spike_times, _ = detect_spikes(
                result.v_soma, result.t, threshold=-20.0, baseline_threshold=-50.0
            )
            
            n_spikes = len(spike_times)
            firing_rate = n_spikes / (result.t[-1] / 1000.0)
            
            print(f"   Iext={iext}: {n_spikes} spikes, {firing_rate:.1f} Hz")
            
            if 10 <= n_spikes <= 50:  # Within target range
                print(f"   ✅ OPTIMAL: Iext={iext}")
                return {'optimal_iext': iext, 'n_spikes': n_spikes, 'rate': firing_rate}
        
        return {'optimal_iext': None, 'failed': True}
    
    def calibrate_k_thalamic_relay(self) -> dict:
        """Calibrate K: Thalamic Relay - increase gK_max from 10 to reduce firing"""
        print("\n🔧 Calibrating K: Thalamic Relay (Ih + ICa + Burst)...")
        
        # Current: 23 spikes (target: 2-8), gK_max=10 too low
        # Try increasing gK_max
        gk_values = [15, 20, 25, 30]
        
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
    
    def calibrate_l_hippocampal_ca1(self) -> dict:
        """Calibrate L: Hippocampal CA1 - increase gK_max from 8 to enable spiking"""
        print("\n🔧 Calibrating L: Hippocampal CA1 (Theta rhythm)...")
        
        # Current: 0 spikes (target: 3-10), gK_max=8 too low
        # Try increasing gK_max
        gk_values = [12, 16, 20, 24]
        
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
    
    def run_calibration_tests(self):
        """Run all calibration tests"""
        print("=" * 70)
        print("=" * 35 + " PRESET CALIBRATION (TEST BRANCH) " + "=" * 35)
        print("=" * 70)
        print("Testing and calibrating preset parameters...")
        print()
        
        # Test current state
        print("📊 CURRENT PRESET STATE:")
        for preset_name in ['B: FS Interneuron (GABA)', 'K: Thalamic Relay (Ih + ICa + Burst)', 'L: Hippocampal CA1 (Theta rhythm)']:
            result = self.test_current_preset(preset_name)
            status = "✅ WITHIN TARGET" if result['within_target'] else "❌ NEEDS CALIBRATION"
            print(f"   {preset_name}: {result['n_spikes']} spikes, {result['firing_rate']:.1f} Hz - {status}")
        
        # Calibrate problematic presets
        calibrations = {}
        
        # Calibrate B: FS Interneuron
        calibrations['B'] = self.calibrate_b_fs_interneuron()
        
        # Calibrate K: Thalamic Relay  
        calibrations['K'] = self.calibrate_k_thalamic_relay()
        
        # Calibrate L: Hippocampal CA1
        calibrations['L'] = self.calibrate_l_hippocampal_ca1()
        
        # Summary
        print("\n" + "=" * 70)
        print("CALIBRATION SUMMARY:")
        print("=" * 70)
        
        for preset, result in calibrations.items():
            if 'failed' in result:
                print(f"   {preset}: ❌ CALIBRATION FAILED")
            else:
                print(f"   {preset}: ✅ CALIBRATED")
        
        return calibrations


if __name__ == "__main__":
    branch = PresetCalibrationTestBranch()
    results = branch.run_calibration_tests()
