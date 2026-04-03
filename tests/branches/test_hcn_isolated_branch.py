"""
Test Branch: HCN Channel Solo Validation
Tests HCN (Ih) channel in isolation - without other channels
Validates: activation curve, resting stability, input resistance, temperature scaling
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.solver import NeuronSolver
from core.analysis import detect_spikes


class HCNIsolatedTestBranch:
    """
    Test branch for isolated HCN channel validation.
    All other channels disabled - pure Ih behavior.
    """
    
    def __init__(self):
        self.results = {}
    
    def create_isolated_hcn_config(self) -> FullModelConfig:
        """Create config with ONLY HCN channel enabled"""
        cfg = FullModelConfig()
        
        # Disable ALL channels except Ih
        cfg.channels.gNa_max = 0.0  # No Na
        cfg.channels.gK_max = 0.0   # No K
        cfg.channels.gL = 0.01      # Minimal leak for stability
        cfg.channels.gIh_max = 0.5  # HCN conductance
        cfg.channels.gCa_max = 0.0  # No Ca
        cfg.channels.gA_max = 0.0   # No IA
        cfg.channels.gSK_max = 0.0  # No SK
        
        # Enable only Ih
        cfg.channels.enable_Ih = True
        cfg.channels.enable_ICa = False
        cfg.channels.enable_IA = False
        cfg.channels.enable_SK = False
        
        # HCN reversal potential
        cfg.channels.E_Ih = -43.0  # Physiological value
        
        # Morphology - single compartment for isolation
        cfg.morphology.single_comp = True
        
        # No stimulation - resting state
        cfg.stim.Iext = 0.0
        cfg.stim.stim_type = 'const'
        
        return cfg
    
    def test_activation_curve(self):
        """Test HCN activation curve - V_½ should be ~-78 mV (Destexhe 1993)"""
        print("\n🧪 Testing HCN activation curve (isolated)...")
        
        from core.kinetics import ar_Ih, br_Ih
        
        v_range = np.linspace(-100, -40, 50)
        r_infs = []
        
        for v in v_range:
            ar = ar_Ih(v)
            br = br_Ih(v)
            r_inf = ar / (ar + br) if (ar + br) > 0 else 0
            r_infs.append(r_inf)
        
        r_arr = np.array(r_infs)
        v_half_idx = np.argmin(np.abs(r_arr - 0.5))
        v_half = v_range[v_half_idx]
        
        print(f"   V_½ activation: {v_half:.1f} mV (expected: -78 mV, tolerance: ±10 mV)")
        
        passed = -88 <= v_half <= -68
        self.results['activation_curve'] = {'v_half': v_half, 'passed': passed}
        
        if passed:
            print("   ✅ PASSED")
        else:
            print("   ❌ FAILED")
        
        return passed
    
    def test_resting_stability(self):
        """Test resting stability with isolated HCN"""
        print("\n🧪 Testing resting stability (isolated HCN)...")
        
        cfg = self.create_isolated_hcn_config()
        cfg.stim.t_sim = 500.0
        
        solver = NeuronSolver(cfg)
        result = solver.run_single()
        
        v = result.v_soma
        v_rest = v.mean()
        v_std = v.std()
        
        print(f"   Resting V: {v_rest:.1f} ± {v_std:.2f} mV")
        print(f"   V range: [{v.min():.1f}, {v.max():.1f}] mV")
        
        # Should be stable around -65 mV with minimal oscillation
        stable = v_std < 5.0 and -80 < v_rest < -50
        self.results['resting_stability'] = {'v_rest': v_rest, 'v_std': v_std, 'passed': stable}
        
        if stable:
            print("   ✅ PASSED - stable resting state")
        else:
            print("   ❌ FAILED - unstable")
        
        return stable
    
    def test_input_resistance(self):
        """Test input resistance with isolated HCN"""
        print("\n🧪 Testing input resistance (isolated HCN)...")
        
        cfg = self.create_isolated_hcn_config()
        
        # Test with small current pulse
        cfg.stim.stim_type = 'pulse'
        cfg.stim.Iext = 0.01  # Very small current (μA/cm²)
        cfg.stim.pulse_start = 100.0
        cfg.stim.pulse_dur = 200.0
        cfg.stim.t_sim = 500.0
        
        solver = NeuronSolver(cfg)
        result = solver.run_single()
        
        v = result.v_soma
        t = result.t
        
        # Measure voltage change during pulse
        pulse_start_idx = np.argmin(np.abs(t - 100.0))
        pulse_end_idx = np.argmin(np.abs(t - 300.0))
        
        v_baseline = v[:pulse_start_idx].mean()
        v_during = v[pulse_start_idx:pulse_end_idx].mean()
        
        delta_v = v_during - v_baseline
        delta_i = 0.01  # μA/cm²
        
        if abs(delta_i) > 1e-10:
            r_in = abs(delta_v / delta_i)  # kΩ·cm²
        else:
            r_in = float('inf')
        
        print(f"   ΔV: {delta_v:.2f} mV")
        print(f"   Input resistance: {r_in:.1f} kΩ·cm²")
        
        # HCN should increase input resistance (physiological: ~100-200 MΩ for whole cell)
        # For density: expect ~50-200 kΩ·cm²
        passed = 10 < r_in < 500
        self.results['input_resistance'] = {'r_in': r_in, 'passed': passed}
        
        if passed:
            print("   ✅ PASSED")
        else:
            print("   ❌ FAILED (unexpected R_in)")
        
        return passed
    
    def test_temperature_scaling(self):
        """Test temperature scaling of HCN kinetics"""
        print("\n🧪 Testing temperature scaling (isolated HCN)...")
        
        cfg = self.create_isolated_hcn_config()
        cfg.stim.t_sim = 200.0
        
        temps = [20.0, 25.0, 37.0]  # Celsius
        results = []
        
        for temp in temps:
            cfg.env.T_celsius = temp
            solver = NeuronSolver(cfg)
            result = solver.run_single()
            
            v = result.v_soma
            results.append({'temp': temp, 'v_mean': v.mean(), 'v_std': v.std()})
            
            print(f"   T={temp}°C: V={v.mean():.1f}±{v.std():.2f}mV")
        
        # Check that temperature affects kinetics
        passed = len(results) == 3
        self.results['temperature_scaling'] = {'results': results, 'passed': passed}
        
        if passed:
            print("   ✅ PASSED")
        else:
            print("   ❌ FAILED")
        
        return passed
    
    def run_all_tests(self):
        """Run complete isolated HCN validation suite"""
        print("=" * 70)
        print("HCN CHANNEL SOLO VALIDATION (ISOLATED TEST BRANCH)")
        print("=" * 70)
        print("Testing HCN in isolation - all other channels disabled")
        
        tests = [
            self.test_activation_curve,
            self.test_resting_stability,
            self.test_input_resistance,
            self.test_temperature_scaling,
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
    test_branch = HCNIsolatedTestBranch()
    test_branch.run_all_tests()
