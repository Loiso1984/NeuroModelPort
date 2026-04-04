"""
Test Branch: Multi-Channel Stress Tests
Stress tests for multi-channel interactions
Validates: Ih+ICa, IA+SK, all channels combined
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.solver import NeuronSolver
from core.analysis import detect_spikes


class MultichannelStressTestBranch:
    """Test branch for multi-channel interaction stress tests"""
    
    def __init__(self):
        self.results = {}
    
    def create_multichannel_config(self, channels_enabled):
        """Create config with specific channel combinations"""
        cfg = FullModelConfig()
        
        # Baseline
        cfg.channels.gNa_max = 100.0
        cfg.channels.gK_max = 10.0
        cfg.channels.gL = 0.3
        
        # Optional channels
        cfg.channels.gIh_max = 0.5 if channels_enabled.get('Ih', False) else 0.0
        cfg.channels.gCa_max = 0.08 if channels_enabled.get('ICa', False) else 0.0
        cfg.channels.gA_max = 0.4 if channels_enabled.get('IA', False) else 0.0
        cfg.channels.gSK_max = 0.5 if channels_enabled.get('SK', False) else 0.0
        
        cfg.channels.enable_Ih = channels_enabled.get('Ih', False)
        cfg.channels.enable_ICa = channels_enabled.get('ICa', False)
        cfg.channels.enable_IA = channels_enabled.get('IA', False)
        cfg.channels.enable_SK = channels_enabled.get('SK', False)
        
        # Stimulation
        cfg.stim.stim_type = 'pulse'
        cfg.stim.Iext = 15.0
        cfg.stim.pulse_start = 50.0
        cfg.stim.pulse_dur = 200.0
        cfg.stim.t_sim = 300.0
        cfg.stim.dt_eval = 0.2
        
        return cfg
    
    def test_ih_ica_interaction(self):
        """Test Ih + ICa interaction - known problematic combination"""
        print("\n🧪 Testing Ih + ICa interaction...")
        print("   ⚠️  Known issue: This combination can cause instability")
        
        cfg = self.create_multichannel_config({'Ih': True, 'ICa': True, 'IA': False, 'SK': False})
        
        try:
            solver = NeuronSolver(cfg)
            result = solver.run_single()
            
            v = result.v_soma
            t = result.t
            
            # Check for numerical stability
            v_range = v.max() - v.min()
            v_nan = np.isnan(v).any()
            v_inf = np.isinf(v).any()
            
            print(f"   V range: [{v.min():.1f}, {v.max():.1f}] mV")
            print(f"   NaN present: {v_nan}, Inf present: {v_inf}")
            
            # Detect spikes
            spike_idx, spike_times, spike_amps = detect_spikes(v, t, threshold=-20.0)
            print(f"   Spikes detected: {len(spike_times)}")
            
            stable = not v_nan and not v_inf and v_range < 200
            
            passed = stable
            self.results['ih_ica'] = {'stable': stable, 'spikes': len(spike_times), 'passed': passed}
            
            if passed:
                print("   ✅ PASSED - combination stable")
            else:
                print("   ❌ FAILED - instability detected")
            
            return passed
            
        except Exception as e:
            print(f"   ❌ EXCEPTION: {str(e)[:60]}")
            self.results['ih_ica'] = {'error': str(e), 'passed': False}
            return False
    
    def test_ia_sk_interaction(self):
        """Test IA + SK interaction"""
        print("\n🧪 Testing IA + SK interaction...")
        
        cfg = self.create_multichannel_config({'Ih': False, 'ICa': False, 'IA': True, 'SK': True})
        
        try:
            solver = NeuronSolver(cfg)
            result = solver.run_single()
            
            v = result.v_soma
            t = result.t
            
            v_nan = np.isnan(v).any()
            spike_idx, spike_times, spike_amps = detect_spikes(v, t, threshold=-20.0)
            
            print(f"   V range: [{v.min():.1f}, {v.max():.1f}] mV")
            print(f"   Spikes: {len(spike_times)}")
            
            stable = not v_nan
            passed = stable
            
            self.results['ia_sk'] = {'stable': stable, 'spikes': len(spike_times), 'passed': passed}
            
            if passed:
                print("   ✅ PASSED")
            else:
                print("   ❌ FAILED")
            
            return passed
            
        except Exception as e:
            print(f"   ❌ EXCEPTION: {str(e)[:60]}")
            self.results['ia_sk'] = {'error': str(e), 'passed': False}
            return False
    
    def test_all_channels(self):
        """Test all channels enabled simultaneously"""
        print("\n🧪 Testing ALL channels simultaneously...")
        print("   ⚠️  Maximum complexity scenario")
        
        cfg = self.create_multichannel_config({'Ih': True, 'ICa': True, 'IA': True, 'SK': True})
        
        try:
            solver = NeuronSolver(cfg)
            result = solver.run_single()
            
            v = result.v_soma
            t = result.t
            
            v_nan = np.isnan(v).any()
            v_inf = np.isinf(v).any()
            
            spike_idx, spike_times, spike_amps = detect_spikes(v, t, threshold=-20.0)
            
            print(f"   V range: [{v.min():.1f}, {v.max():.1f}] mV")
            print(f"   Spikes: {len(spike_times)}")
            
            stable = not v_nan and not v_inf
            passed = stable
            
            self.results['all_channels'] = {
                'stable': stable,
                'spikes': len(spike_times),
                'passed': passed
            }
            
            if passed:
                print("   ✅ PASSED - all channels stable together")
            else:
                print("   ❌ FAILED - instability with all channels")
            
            return passed
            
        except Exception as e:
            print(f"   ❌ EXCEPTION: {str(e)[:60]}")
            self.results['all_channels'] = {'error': str(e), 'passed': False}
            return False
    
    def test_parameter_sweep(self):
        """Quick parameter sweep for critical combinations"""
        print("\n🧪 Testing parameter sweep (simplified)...")
        
        # Test different gCa_max values with Ih
        gCa_values = [0.04, 0.08, 0.12]
        results = []
        
        for gCa in gCa_values:
            try:
                cfg = self.create_multichannel_config({'Ih': True, 'ICa': True, 'IA': False, 'SK': False})
                cfg.channels.gCa_max = gCa
                
                solver = NeuronSolver(cfg)
                result = solver.run_single()
                
                v = result.v_soma
                stable = not np.isnan(v).any() and not np.isinf(v).any()
                
                results.append({'gCa': gCa, 'stable': stable})
                print(f"   gCa={gCa}: {'✅' if stable else '❌'}")
                
            except Exception as e:
                results.append({'gCa': gCa, 'stable': False, 'error': str(e)})
                print(f"   gCa={gCa}: ❌ ({str(e)[:40]})")
        
        passed = all(r['stable'] for r in results)
        self.results['parameter_sweep'] = {'results': results, 'passed': passed}
        
        return passed
    
    def run_all_tests(self):
        """Run complete stress test suite"""
        print("=" * 70)
        print("MULTI-CHANNEL STRESS TESTS (TEST BRANCH)")
        print("=" * 70)
        print("Testing numerical stability of channel combinations")
        
        tests = [
            self.test_ih_ica_interaction,
            self.test_ia_sk_interaction,
            self.test_all_channels,
            self.test_parameter_sweep,
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            try:
                if test():
                    passed += 1
            except Exception as e:
                print(f"   ❌ EXCEPTION: {e}")
        
        print("\n" + "=" * 70)
        print(f"SUMMARY: {passed}/{total} tests passed")
        print("=" * 70)
        
        return self.results


if __name__ == "__main__":
    test_branch = MultichannelStressTestBranch()
    test_branch.run_all_tests()
