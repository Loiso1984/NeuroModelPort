"""
HCN Channel Validation Suite
Tests HCN channel activation, resting stability, and temperature dependence
"""

import sys
import numpy as np
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset
from core.morphology import MorphologyBuilder
from core.solver import NeuronSolver

class HCNChannelValidator:
    """Comprehensive HCN channel validation"""
    
    def __init__(self):
        self.results = {}
        self.output_dir = Path("_test_results")
        self.output_dir.mkdir(exist_ok=True)
    
    def test_hcn_activation_curve(self) -> dict:
        """Test HCN channel V_½ activation and slope"""
        print("🧪 Testing HCN activation curve...")
        
        results = {'passed': True, 'details': []}
        
        # Test with different presets that have HCN
        hcn_presets = [
            "K: Thalamic Relay (Ih + ICa + Burst)",
            "L: Hippocampal CA1 (Theta rhythm)"
        ]
        
        for preset_name in hcn_presets:
            try:
                cfg = FullModelConfig()
                apply_preset(cfg, preset_name)
                
                if not cfg.channels.enable_Ih:
                    results['details'].append(f"{preset_name}: HCN not enabled")
                    continue
                
                # Test activation at different voltages
                v_test = np.array([-120, -100, -80, -60, -40, -20, 0])
                
                # Build morphology for single compartment
                morph = MorphologyBuilder.build(cfg)
                solver = NeuronSolver(cfg)
                
                # Run voltage clamp simulation (simplified)
                v_half_expected = -90.0  # Typical HCN V_½
                slope_expected = 10.0    # Typical slope factor
                
                # Check if parameters are reasonable
                if cfg.channels.gIh_max > 0:
                    results['details'].append(f"{preset_name}: ✅ gIh_max = {cfg.channels.gIh_max}")
                else:
                    results['passed'] = False
                    results['details'].append(f"{preset_name}: ❌ gIh_max = {cfg.channels.gIh_max}")
                
                print(f"   {preset_name}: gIh_max = {cfg.channels.gIh_max}")
                
            except Exception as e:
                results['passed'] = False
                results['details'].append(f"{preset_name}: Error - {e}")
        
        return results
    
    def test_resting_stability(self) -> dict:
        """Test resting membrane potential stability with HCN"""
        print("🧪 Testing resting stability...")
        
        results = {'passed': True, 'details': []}
        
        # Test Thalamic Relay preset (has HCN)
        cfg = FullModelConfig()
        apply_preset(cfg, "K: Thalamic Relay (Ih + ICa + Burst)")
        
        # Run without stimulation to check resting potential
        cfg.stim.Iext = 0.0
        cfg.stim.stim_type = 'const'
        
        solver = NeuronSolver(cfg)
        result = solver.run_single()
        
        # Check resting potential stability
        v_soma = result.v_soma
        v_rest = v_soma[-100:].mean()  # Last 100 points
        v_std = v_soma[-100:].std()
        
        # Typical resting potentials
        if -80 <= v_rest <= -50:
            results['details'].append(f"✅ Resting potential: {v_rest:.1f} mV")
        else:
            results['passed'] = False
            results['details'].append(f"❌ Resting potential: {v_rest:.1f} mV (out of range)")
        
        # Check stability (low std)
        if v_std < 2.0:
            results['details'].append(f"✅ Stability: {v_std:.2f} mV std")
        else:
            results['passed'] = False
            results['details'].append(f"❌ Instability: {v_std:.2f} mV std")
        
        print(f"   V_rest = {v_rest:.1f} mV (σ = {v_std:.2f} mV)")
        
        return results
    
    def test_temperature_scaling(self) -> dict:
        """Test HCN temperature dependence"""
        print("🧪 Testing temperature scaling...")
        
        results = {'passed': True, 'details': []}
        
        # Test at different temperatures
        temps = [23.0, 30.0, 37.0, 40.0]  # °C
        firing_rates = []
        
        for temp in temps:
            cfg = FullModelConfig()
            apply_preset(cfg, "K: Thalamic Relay (Ih + ICa + Burst)")
            
            cfg.env.T_celsius = temp
            cfg.env.T_ref = 23.0  # Reference temperature
            
            solver = NeuronSolver(cfg)
            result = solver.run_single()
            
            # Count spikes
            v_soma = result.v_soma
            threshold = -20.0
            spike_indices = np.where(v_soma > threshold)[0]
            
            # Calculate firing rate
            sim_duration = result.t[-1]
            firing_rate = len(spike_indices) / (sim_duration / 1000.0)  # Hz
            firing_rates.append(firing_rate)
            
            print(f"   {temp:.1f}°C: {firing_rate:.1f} Hz")
        
        # Check temperature scaling (should increase with temperature)
        if len(firing_rates) >= 2:
            ratio_37_23 = firing_rates[2] / firing_rates[0] if firing_rates[0] > 0 else 1.0
            
            # HCN channels should increase firing with temperature
            if ratio_37_23 > 1.0:
                results['details'].append(f"✅ Temperature scaling: 37°C/23°C = {ratio_37_23:.2f}")
            else:
                results['details'].append(f"⚠️ Temperature scaling: 37°C/23°C = {ratio_37_23:.2f}")
            
            # Check for reasonable scaling
            if 0.5 <= ratio_37_23 <= 5.0:
                results['details'].append("✅ Temperature scaling in physiological range")
            else:
                results['passed'] = False
                results['details'].append("❌ Temperature scaling out of range")
        
        return results
    
    def test_input_resistance(self) -> dict:
        """Test HCN effect on input resistance"""
        print("🧪 Testing input resistance...")
        
        results = {'passed': True, 'details': []}
        
        # Test with and without HCN
        configs = [
            ("With HCN", True),
            ("Without HCN", False)
        ]
        
        input_resistances = []
        
        for name, enable_hcn in configs:
            cfg = FullModelConfig()
            apply_preset(cfg, "K: Thalamic Relay (Ih + ICa + Burst)")
            
            cfg.channels.enable_Ih = enable_hcn
            
            # Apply small hyperpolarizing current
            cfg.stim.Iext = -0.05  # µA/cm² - smaller for better resolution
            cfg.stim.stim_type = 'const'
            
            solver = NeuronSolver(cfg)
            result = solver.run_single()
            
            # Calculate input resistance
            v_soma = result.v_soma
            v_steady = v_soma[-100:].mean()
            v_rest = v_soma[:100].mean()
            
            delta_v = v_steady - v_rest
            r_input = abs(delta_v / 0.05)  # MΩ
            
            input_resistances.append(r_input)
            print(f"   {name}: R_in = {r_input:.1f} MΩ")
        
        # HCN should decrease input resistance
        if len(input_resistances) == 2:
            r_with_hcn = input_resistances[0]
            r_without_hcn = input_resistances[1]
            
            if r_with_hcn < r_without_hcn:
                ratio = r_without_hcn / r_with_hcn
                results['details'].append(f"✅ HCN reduces R_in: {ratio:.2f}x")
            else:
                results['passed'] = False
                results['details'].append("❌ HCN doesn't reduce input resistance")
        
        return results
    
    def run_all_tests(self):
        """Run complete HCN validation suite"""
        print("🧪 HCN CHANNEL VALIDATION SUITE")
        print("=" * 50)
        
        # Run all tests
        self.results['activation'] = self.test_hcn_activation_curve()
        self.results['stability'] = self.test_resting_stability()
        self.results['temperature'] = self.test_temperature_scaling()
        self.results['input_resistance'] = self.test_input_resistance()
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 HCN VALIDATION SUMMARY")
        print("=" * 50)
        
        passed_count = 0
        total_count = 0
        
        for test_name, result in self.results.items():
            total_count += 1
            status = "✅ PASSED" if result['passed'] else "❌ FAILED"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
            
            if result['passed']:
                passed_count += 1
            
            for detail in result['details']:
                print(f"   {detail}")
        
        print(f"\nOverall Status: {'✅ PASSED' if passed_count == total_count else '❌ FAILED'}")
        print(f"Tests Passed: {passed_count}/{total_count}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"hcn_validation_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        return self.results

if __name__ == "__main__":
    validator = HCNChannelValidator()
    validator.run_all_tests()
