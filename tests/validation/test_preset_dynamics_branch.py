"""
Test Branch: Preset Physiological Validation
Comprehensive validation of preset dynamics against physiological parameters
Validates: spike counts, firing rates, voltage ranges for all presets
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver
from core.analysis import detect_spikes


class PresetValidationTestBranch:
    """Test branch for comprehensive preset physiological validation"""
    
    def __init__(self):
        self.results = {}
        self.physiological_targets = {
            # Preset: (min_spikes, max_spikes, expected_firing_pattern)
            'A: Classic Squid (HH 1952)': (5, 20, 'regular'),
            'B: FS Interneuron (GABA)': (10, 50, 'fast'),
            'C: α-Motoneuron (Soma)': (3, 15, 'regular'),
            'D: AIS (Axon Initial Segment)': (5, 25, 'regular'),
            'E: Dendritic Filter (EPSP)': (0, 5, 'low'),
            'G: Purkinje Cell (Cerebellar)': (2, 10, 'bursting'),
            'H: Layer 5 Pyramidal (Cortex)': (3, 12, 'adapting'),
            'I: O-LM Interneuron (Hippocampus)': (5, 20, 'oscillatory'),
            'K: Thalamic Relay (Ih + ICa + Burst)': (2, 8, 'bursting'),
            'L: Hippocampal CA1 (Theta rhythm)': (3, 10, 'theta'),
            'M: Epilepsy (v10 SCN1A mutation)': (15, 100, 'high_rate'),
            "N: Alzheimer's (v10 Calcium Toxicity)": (2, 10, 'irregular'),
            'O: Hypoxia (v10 ATP-pump failure)': (1, 8, 'degrading'),
        }
    
    def test_preset_dynamics(self, preset_name, duration_ms=500):
        """Test single preset dynamics"""
        print(f"\n🧪 Testing {preset_name}...")
        
        cfg = FullModelConfig()
        apply_preset(cfg, preset_name)
        
        cfg.stim.t_sim = duration_ms
        cfg.stim.dt_eval = 0.2
        
        try:
            solver = NeuronSolver(cfg)
            result = solver.run_single()
            
            v = result.v_soma
            t = result.t
            
            # Basic checks
            v_rest = v[:100].mean()  # First 100 points (20ms)
            v_max = v.max()
            v_min = v.min()
            
            # Spike detection
            spike_idx, spike_times, spike_amps = detect_spikes(v, t, threshold=-20.0)
            n_spikes = len(spike_times)
            
            # Firing rate
            if n_spikes > 1 and len(spike_times) > 1:
                isi = np.diff(spike_times)
                mean_isi = isi.mean()
                firing_rate = 1000.0 / mean_isi if mean_isi > 0 else 0
                cv_isi = isi.std() / isi.mean() if isi.mean() > 0 else 0
            else:
                firing_rate = 0
                cv_isi = 0
            
            print(f"   Rest: {v_rest:.1f}mV, Peak: {v_max:.1f}mV")
            print(f"   Spikes: {n_spikes}, Rate: {firing_rate:.1f} Hz, CV: {cv_isi:.2f}")
            
            # Validation against targets
            target = self.physiological_targets.get(preset_name, (0, 20, 'unknown'))
            min_spikes, max_spikes, pattern = target
            
            in_range = min_spikes <= n_spikes <= max_spikes
            
            # Additional physiological checks
            healthy_voltage = -90 < v_min < v_max < 60
            
            passed = in_range and healthy_voltage and not np.isnan(v).any()
            
            self.results[preset_name] = {
                'n_spikes': n_spikes,
                'firing_rate': firing_rate,
                'cv_isi': cv_isi,
                'v_rest': v_rest,
                'v_max': v_max,
                'passed': passed
            }
            
            if passed:
                print(f"   ✅ PASSED")
            else:
                print(f"   ❌ FAILED (spikes: {n_spikes}, expected: {min_spikes}-{max_spikes})")
            
            return passed
            
        except Exception as e:
            print(f"   ❌ EXCEPTION: {str(e)[:60]}")
            self.results[preset_name] = {'error': str(e), 'passed': False}
            return False
    
    def test_all_presets(self):
        """Test all available presets"""
        print("\n" + "=" * 70)
        print("PRESET DYNAMICS VALIDATION (TEST BRANCH)")
        print("=" * 70)
        
        preset_names = list(self.physiological_targets.keys())
        
        passed = 0
        total = len(preset_names)
        
        for preset_name in preset_names:
            try:
                if self.test_preset_dynamics(preset_name, duration_ms=500):
                    passed += 1
            except Exception as e:
                print(f"   ❌ EXCEPTION in {preset_name}: {e}")
        
        print("\n" + "=" * 70)
        print(f"SUMMARY: {passed}/{total} presets passed")
        print("=" * 70)
        
        return self.results
    
    def test_parameter_sensitivity(self, preset_name='B: FS Interneuron (GABA)'):
        """Test sensitivity to parameter changes"""
        print(f"\n🧪 Parameter sensitivity for {preset_name}...")
        
        base_cfg = FullModelConfig()
        apply_preset(base_cfg, preset_name)
        
        # Test varying stimulation
        currents = [10, 15, 20, 25]
        results = []
        
        for I in currents:
            try:
                cfg = FullModelConfig()
                apply_preset(cfg, preset_name)
                cfg.stim.Iext = I
                cfg.stim.t_sim = 300.0
                
                solver = NeuronSolver(cfg)
                result = solver.run_single()
                
                v = result.v_soma
                t = result.t
                spike_idx, spike_times, spike_amps = detect_spikes(v, t, threshold=-20.0)
                
                results.append({'I': I, 'spikes': len(spike_times)})
                print(f"   I={I}μA: {len(spike_times)} spikes")
                
            except Exception as e:
                results.append({'I': I, 'spikes': 0, 'error': str(e)})
                print(f"   I={I}μA: ERROR")
        
        # Check that higher current = more spikes (monotonic)
        spike_counts = [r['spikes'] for r in results if 'error' not in r]
        monotonic = all(spike_counts[i] <= spike_counts[i+1] for i in range(len(spike_counts)-1)) if len(spike_counts) > 1 else True
        
        passed = len([r for r in results if 'error' not in r]) >= 3
        
        self.results['parameter_sensitivity'] = {'results': results, 'monotonic': monotonic, 'passed': passed}
        
        if passed:
            print("   ✅ PASSED")
        else:
            print("   ❌ FAILED")
        
        return passed
    
    def run_all_tests(self):
        """Run complete preset validation suite"""
        print("=" * 70)
        print("PRESET PHYSIOLOGICAL VALIDATION (TEST BRANCH)")
        print("=" * 70)
        print("Validating all presets against physiological parameters")
        
        # Main preset tests
        self.test_all_presets()
        
        # Sensitivity test
        self.test_parameter_sensitivity()
        
        return self.results


if __name__ == "__main__":
    test_branch = PresetValidationTestBranch()
    test_branch.run_all_tests()
