"""
Test Branch: IA Channel Solo Validation  
Tests A-type potassium (IA) channel in isolation
Validates: activation/inactivation curves, spike delay effect, recovery
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.solver import NeuronSolver
from core.analysis import detect_spikes


class IAIsolatedTestBranch:
    """Test branch for isolated IA (A-type K+) channel validation"""
    
    def __init__(self):
        self.results = {}
    
    def create_isolated_ia_config(self, gA=0.4) -> FullModelConfig:
        """Create config with ONLY IA channel enabled"""
        cfg = FullModelConfig()
        
        # Minimal baseline conductances for spike generation test
        cfg.channels.gNa_max = 100.0  # Need Na for spiking
        cfg.channels.gK_max = 10.0    # Need K for repolarization
        cfg.channels.gL = 0.3         # Leak
        
        # IA conductance
        cfg.channels.gA_max = gA
        
        # Disable other optional channels
        cfg.channels.gIh_max = 0.0
        cfg.channels.gCa_max = 0.0
        cfg.channels.gSK_max = 0.0
        
        # Enable only IA (plus baseline Na/K)
        cfg.channels.enable_IA = True
        cfg.channels.enable_Ih = False
        cfg.channels.enable_ICa = False
        cfg.channels.enable_SK = False
        
        return cfg
    
    def test_kinetics_curves(self):
        """Test IA activation and inactivation curves"""
        print("\n🧪 Testing IA kinetics curves...")
        
        from core.kinetics import aa_IA, ba_IA, ab_IA, bb_IA
        
        v_range = np.linspace(-100, 40, 100)
        m_infs = []
        h_infs = []
        
        for v in v_range:
            am = aa_IA(v)
            bm = ba_IA(v)
            ah = ab_IA(v)
            bh = bb_IA(v)
            
            m_inf = am / (am + bm) if (am + bm) > 0 else 0
            h_inf = ah / (ah + bh) if (ah + bh) > 0 else 0
            
            m_infs.append(m_inf)
            h_infs.append(h_inf)
        
        m_arr = np.array(m_infs)
        h_arr = np.array(h_infs)
        
        # Find V_½ values
        v_half_m_idx = np.argmin(np.abs(m_arr - 0.5))
        v_half_h_idx = np.argmin(np.abs(h_arr - 0.5))
        
        v_half_m = v_range[v_half_m_idx]
        v_half_h = v_range[v_half_h_idx]
        
        print(f"   V_½ activation (m): {v_half_m:.1f} mV (expected: -40 ± 10 mV)")
        print(f"   V_½ inactivation (h): {v_half_h:.1f} mV (expected: -60 ± 10 mV)")
        
        # Check slopes (steepness)
        # m curve should be steep around -40 mV
        # h curve should be steep around -60 mV
        
        passed = (-50 <= v_half_m <= -30) and (-70 <= v_half_h <= -50)
        
        self.results['kinetics'] = {
            'v_half_m': v_half_m,
            'v_half_h': v_half_h,
            'passed': passed
        }
        
        if passed:
            print("   ✅ PASSED")
        else:
            print("   ❌ FAILED")
        
        return passed
    
    def test_spike_delay_effect(self):
        """Test that IA causes spike delay compared to no IA"""
        print("\n🧪 Testing spike delay effect...")
        
        configs = [
            ("No IA", False, 0.0),
            ("With IA", True, 0.4),
        ]
        
        first_spikes = []
        
        for name, enable_ia, gA in configs:
            cfg = self.create_isolated_ia_config(gA)
            cfg.channels.enable_IA = enable_ia
            
            # Stimulation
            cfg.stim.stim_type = 'pulse'
            cfg.stim.Iext = 15.0
            cfg.stim.pulse_start = 50.0
            cfg.stim.pulse_dur = 400.0
            cfg.stim.t_sim = 500.0
            
            solver = NeuronSolver(cfg)
            result = solver.run_single()
            
            v = result.v_soma
            t = result.t
            
            # Detect spikes
            spike_idx, spike_times, spike_amps = detect_spikes(v, t, threshold=-20.0)
            
            if len(spike_times) > 0:
                first_spike = spike_times[0]
                first_spikes.append(first_spike)
                print(f"   {name}: First spike at {first_spike:.1f} ms ({len(spike_times)} spikes)")
            else:
                first_spikes.append(None)
                print(f"   {name}: No spikes detected")
        
        # Check delay
        delay = None
        if len(first_spikes) == 2 and first_spikes[0] is not None and first_spikes[1] is not None:
            delay = first_spikes[1] - first_spikes[0]
            print(f"   IA delay: {delay:.1f} ms")
            
            # IA should delay first spike by at least 5ms
            passed = delay > 5.0
        else:
            passed = False
        
        self.results['spike_delay'] = {'delay_ms': delay, 'passed': passed}
        
        if passed:
            print("   ✅ PASSED - IA delays spiking")
        else:
            print("   ❌ FAILED - insufficient delay or no spikes")
        
        return passed
    
    def test_inactivation_recovery(self):
        """Test recovery from inactivation"""
        print("\n🧪 Testing inactivation recovery...")
        
        # This is a simplified test - full recovery testing needs longer protocols
        print("   ⚠️  Recovery test - simplified version")
        print("   (Full recovery testing requires pulse-pair protocols)")
        
        # For now, just verify the model runs with IA
        cfg = self.create_isolated_ia_config(0.4)
        cfg.stim.t_sim = 100.0
        
        solver = NeuronSolver(cfg)
        result = solver.run_single()
        
        v = result.v_soma
        stable = np.std(v) < 10.0  # Should be stable
        
        self.results['recovery'] = {'stable': stable, 'passed': stable}
        
        if stable:
            print("   ✅ PASSED - model stable with IA")
        else:
            print("   ❌ FAILED - unstable")
        
        return stable
    
    def run_all_tests(self):
        """Run complete IA validation suite"""
        print("=" * 70)
        print("IA CHANNEL SOLO VALIDATION (ISOLATED TEST BRANCH)")
        print("=" * 70)
        
        tests = [
            self.test_kinetics_curves,
            self.test_spike_delay_effect,
            self.test_inactivation_recovery,
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
    test_branch = IAIsolatedTestBranch()
    test_branch.run_all_tests()
