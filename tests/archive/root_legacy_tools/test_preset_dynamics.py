"""
Test preset dynamics before/after parameter changes
Проверка динамики пресетов перед и после изменений параметров
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver
from core.analysis import detect_spikes

def test_preset_dynamics(preset_name, description=""):
    """Test single preset dynamics with detailed output"""
    print(f"\n🔍 Testing: {preset_name}")
    if description:
        print(f"   {description}")
    print("-" * 60)
    
    try:
        cfg = FullModelConfig()
        apply_preset(cfg, preset_name)
        
        # Print channel configuration
        channels = []
        if cfg.channels.gNa_max > 0:
            channels.append(f"Na({cfg.channels.gNa_max:.1f})")
        if cfg.channels.gK_max > 0:
            channels.append(f"K({cfg.channels.gK_max:.1f})")
        if cfg.channels.enable_ICa and cfg.channels.gCa_max > 0:
            channels.append(f"Ca({cfg.channels.gCa_max:.2f})")
        if cfg.channels.enable_Ih and cfg.channels.gIh_max > 0:
            channels.append(f"HCN({cfg.channels.gIh_max:.2f})")
        if cfg.channels.enable_IA and cfg.channels.gA_max > 0:
            channels.append(f"IA({cfg.channels.gA_max:.2f})")
        if cfg.channels.enable_SK and cfg.channels.gSK_max > 0:
            channels.append(f"SK({cfg.channels.gSK_max:.1f})")
        
        print(f"   Channels: {' + '.join(channels)}")
        print(f"   Stimulus: {cfg.stim.stim_type}, Iext={cfg.stim.Iext} µA/cm²")
        
        # Run simulation
        solver = NeuronSolver(cfg)
        result = solver.run_single()
        
        v_soma = result.v_soma
        t = result.t
        
        # Detect spikes
        spike_indices, spike_times, _ = detect_spikes(v_soma, t, threshold=-20.0)
        
        # Analyze dynamics
        n_spikes = len(spike_indices)
        sim_duration = t[-1]
        firing_rate = n_spikes / (sim_duration / 1000.0) if sim_duration > 0 else 0
        
        # Voltage analysis
        v_max = v_soma.max()
        v_min = v_soma.min()
        v_rest = v_soma[:100].mean() if len(v_soma) > 100 else v_soma.mean()
        v_final = v_soma[-100:].mean() if len(v_soma) > 100 else v_soma.mean()
        
        print(f"   Spikes: {n_spikes} ({firing_rate:.1f} Hz)")
        print(f"   Voltage: rest={v_rest:.1f}mV, max={v_max:.1f}mV, min={v_min:.1f}mV, final={v_final:.1f}mV")
        
        # Check for issues
        issues = []
        if n_spikes == 0 and cfg.stim.Iext > 0:
            issues.append("NO SPIKES despite stimulation")
        if v_max > 50:
            issues.append("OVERSHOOT >50mV")
        if v_min < -100:
            issues.append("EXCESSIVE HYPERPOLARIZATION <-100mV")
        if abs(v_final - v_rest) > 10:
            issues.append("DRIFT in resting potential")
        if n_spikes > 0 and firing_rate > 500:
            issues.append(f"UNREALISTIC firing rate: {firing_rate:.1f} Hz")
        
        if issues:
            print(f"   ⚠️ Issues: {', '.join(issues)}")
        else:
            print(f"   ✅ Dynamics look reasonable")
        
        return {
            'preset': preset_name,
            'channels': channels,
            'n_spikes': n_spikes,
            'firing_rate': firing_rate,
            'v_rest': v_rest,
            'v_max': v_max,
            'v_min': v_min,
            'issues': issues,
            'success': len(issues) == 0
        }
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            'preset': preset_name,
            'success': False,
            'error': str(e)
        }

def test_all_presets():
    """Test all presets and report summary"""
    print("=" * 80)
    print("PRESET DYNAMICS VALIDATION")
    print("Testing BEFORE parameter changes to establish baseline")
    print("=" * 80)
    
    presets = [
        ("FS Interneuron (Wang-Buzsaki)", "Fast-spiking inhibitory interneuron"),
        ("alpha-Motoneuron (Powers 2001)", "Spinal motor neuron"),
        ("Purkinje Cell (De Schutter 1994)", "Cerebellar Purkinje neuron"),
        ("K: Thalamic Relay (Ih + ICa + Burst)", "Thalamic relay with burst capability"),
        ("L: Hippocampal CA1 (Theta rhythm)", "CA1 pyramidal for theta rhythm"),
        ("M: Epilepsy (v10 SCN1A mutation)", "Epilepsy model with hyperexcitability"),
        ("N: Alzheimer's (v10 Calcium Toxicity)", "Alzheimer's with calcium dysregulation"),
        ("O: Hypoxia (v10 ATP-pump failure)", "Hypoxia with impaired pumps")
    ]
    
    results = []
    for preset_name, description in presets:
        result = test_preset_dynamics(preset_name, description)
        results.append(result)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for r in results if r.get('success', False))
    failed = len(results) - passed
    
    print(f"Passed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    
    if failed > 0:
        print("\nFailed presets:")
        for r in results:
            if not r.get('success', False):
                print(f"  ❌ {r['preset']}: {r.get('error', r.get('issues', ['Unknown']))}")
    
    # Table of results
    print("\nDetailed Results:")
    print(f"{'Preset':<40} {'Spikes':<8} {'Rate(Hz)':<10} {'V_rest':<8} {'V_max':<8} {'Status':<8}")
    print("-" * 80)
    for r in results:
        if 'error' not in r:
            status = '✅' if r.get('success', False) else '❌'
            print(f"{r['preset']:<40} {r['n_spikes']:<8} {r['firing_rate']:<10.1f} {r['v_rest']:<8.1f} {r['v_max']:<8.1f} {status:<8}")
    
    return results

if __name__ == "__main__":
    results = test_all_presets()
    
    # Save results for comparison
    import json
    from datetime import datetime
    
    output = {
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    
    with open('_test_results/preset_baseline.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n💾 Baseline saved to: _test_results/preset_baseline.json")
