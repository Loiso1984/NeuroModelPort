"""
IA Channel Validation Tests
Tests for A-type potassium channel (Connor-Stevens kinetics)
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver
from core.analysis import detect_spikes
from core.kinetics import aa_IA, ba_IA, ab_IA, bb_IA

class IAChannelValidator:
    """Validator for A-type potassium channels"""
    
    def __init__(self):
        self.results = {}
    
    def test_kinetics(self):
        """Test IA channel kinetics functions"""
        print("🧪 Testing IA channel kinetics...")
        
        # Test voltages
        v_test = np.array([-80, -60, -40, -20, 0, 20])
        
        print("V (mV) | am | bm | m_inf | ah | bh | h_inf")
        print("-" * 60)
        
        for v in v_test:
            am = aa_IA(v)
            bm = ba_IA(v)
            ah = ab_IA(v)
            bh = bb_IA(v)
            
            m_inf = am / (am + bm) if (am + bm) > 0 else 0
            h_inf = ah / (ah + bh) if (ah + bh) > 0 else 0
            
            print(f"{v:6.0f} | {am:.3f} | {bm:.3f} | {m_inf:.3f} | {ah:.4f} | {bh:.3f} | {h_inf:.3f}")
        
        # Check activation curve
        v_range = np.linspace(-100, 40, 50)
        m_infs = []
        h_infs = []
        
        for v in v_range:
            am = aa_IA(v)
            bm = ba_IA(v)
            ah = ab_IA(v)
            bh = bb_IA(v)
            m_infs.append(am / (am + bm) if (am + bm) > 0 else 0)
            h_infs.append(ah / (ah + bh) if (ah + bh) > 0 else 0)
        
        # Find half-activation voltages
        m_arr = np.array(m_infs)
        h_arr = np.array(h_infs)
        
        # m_inf should be 0.5 around -40 mV
        v_m05_idx = np.argmin(np.abs(m_arr - 0.5))
        v_m05 = v_range[v_m05_idx]
        
        # h_inf should be 0.5 around -60 mV
        v_h05_idx = np.argmin(np.abs(h_arr - 0.5))
        v_h05 = v_range[v_h05_idx]
        
        print(f"\nV_½ activation (m): {v_m05:.1f} mV (expected: -40 mV)")
        print(f"V_½ inactivation (h): {v_h05:.1f} mV (expected: -60 mV)")
        
        results = {
            'v_half_m': v_m05,
            'v_half_h': v_h05,
            'passed': (-50 < v_m05 < -30) and (-70 < v_h05 < -50)
        }
        
        if results['passed']:
            print("✅ Kinetics test PASSED")
        else:
            print("❌ Kinetics test FAILED")
        
        return results
    
    def test_spike_delay(self):
        """Test that IA causes spike delay"""
        print("\n🧪 Testing spike delay effect...")
        
        # Test with and without IA
        configs = [
            ("With IA", True, 0.4),
            ("Without IA", False, 0.0)
        ]
        
        first_spike_times = []
        
        for name, enable_ia, gA in configs:
            cfg = FullModelConfig()
            cfg.channels.gNa_max = 100.0
            cfg.channels.gK_max = 10.0
            cfg.channels.gL = 0.3
            cfg.channels.ENa = 50.0
            cfg.channels.EK = -77.0
            cfg.channels.EL = -54.0
            cfg.channels.enable_IA = enable_ia
            cfg.channels.gA_max = gA
            cfg.stim.Iext = 10.0
            cfg.stim.stim_type = 'const'
            
            solver = NeuronSolver(cfg)
            result = solver.run_single()
            
            v = result.v_soma
            t = result.t
            
            # Detect first spike
            spike_indices, spike_times, _ = detect_spikes(v, t, threshold=-20.0)
            
            if len(spike_times) > 0:
                first_spike_time = spike_times[0]
                first_spike_times.append(first_spike_time)
                print(f"   {name}: First spike at {first_spike_time:.2f} ms ({len(spike_times)} total spikes)")
            else:
                first_spike_times.append(None)
                print(f"   {name}: No spikes detected")
        
        # Check delay
        if len(first_spike_times) == 2 and first_spike_times[0] is not None and first_spike_times[1] is not None:
            delay = first_spike_times[0] - first_spike_times[1]
            print(f"   IA delay: {delay:.2f} ms")
            
            if delay > 0:
                print("✅ IA causes spike delay (PASSED)")
                return {'delay_ms': delay, 'passed': True}
            else:
                print("❌ IA does not cause delay (FAILED)")
                return {'delay_ms': delay, 'passed': False}
        else:
            print("⚠️ Could not measure delay (insufficient spikes)")
            return {'delay_ms': None, 'passed': False}
    
    def test_presets_with_ia(self):
        """Test presets that should have IA"""
        print("\n🧪 Testing presets with IA...")
        
        presets_with_ia = [
            "FS Interneuron (Wang-Buzsaki)",
            "alpha-Motoneuron (Powers 2001)",
            "Purkinje Cell (De Schutter 1994)",
            "L: Hippocampal CA1 (Theta rhythm)"
        ]
        
        results = []
        
        for preset_name in presets_with_ia:
            cfg = FullModelConfig()
            apply_preset(cfg, preset_name)
            
            has_ia = cfg.channels.enable_IA and cfg.channels.gA_max > 0
            gA_value = cfg.channels.gA_max if has_ia else 0
            
            # Quick simulation test
            if has_ia:
                solver = NeuronSolver(cfg)
                result = solver.run_single()
                v = result.v_soma
                spike_indices, _, _ = detect_spikes(v, result.t, threshold=-20.0)
                n_spikes = len(spike_indices)
                v_max = v.max()
                
                status = "✅" if n_spikes > 0 else "⚠️"
                print(f"   {status} {preset_name}: IA enabled (gA={gA_value}), {n_spikes} spikes, V_max={v_max:.1f}mV")
                
                results.append({
                    'preset': preset_name,
                    'has_ia': True,
                    'gA_max': gA_value,
                    'n_spikes': n_spikes,
                    'v_max': v_max,
                    'working': n_spikes > 0
                })
            else:
                print(f"   ❌ {preset_name}: IA NOT enabled (should have IA)")
                results.append({
                    'preset': preset_name,
                    'has_ia': False,
                    'working': False
                })
        
        all_working = all(r['working'] for r in results if r['has_ia'])
        if all_working:
            print("✅ All IA-enabled presets working correctly")
        else:
            print("⚠️ Some IA presets may have issues")
        
        return results
    
    def run_all_tests(self):
        """Run complete IA validation suite"""
        print("=" * 80)
        print("IA CHANNEL VALIDATION SUITE")
        print("=" * 80)
        
        self.results['kinetics'] = self.test_kinetics()
        self.results['spike_delay'] = self.test_spike_delay()
        self.results['presets'] = self.test_presets_with_ia()
        
        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        passed = 0
        total = 0
        
        for test_name, result in self.results.items():
            if isinstance(result, dict) and 'passed' in result:
                total += 1
                if result['passed']:
                    passed += 1
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        return self.results

if __name__ == "__main__":
    validator = IAChannelValidator()
    validator.run_all_tests()
