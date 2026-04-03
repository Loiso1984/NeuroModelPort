"""
test_dual_stim.py - Dual Stimulation Integration Tests

Test scenarios:
1. Single stim backward compatibility (Phase 6 baseline)
2. Soma + Dendritic (GABA-A inhibition)
3. Soma + AIS (with phase offset)
4. Verify no regressions
"""

import numpy as np
import sys
from core.solver import NeuronSolver
from core.models import FullModelConfig
from core.dual_stimulation import DualStimulationConfig
from core.presets import apply_preset

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def get_soma_frequency(t, v_soma, v_threshold=-30.0):
    """Extract firing frequency from soma voltage trace."""
    # Find upstrokes (dV/dt > 0 and V crosses threshold)
    dv = np.diff(v_soma)
    upstrokes = np.where(dv > 0)[0]
    
    # Count threshold crossings
    threshold_crossings = 0
    for i in upstrokes:
        if v_soma[i] < v_threshold and v_soma[i+1] >= v_threshold:
            threshold_crossings += 1
    
    # Calculate frequency (spikes per second)
    t_sim_sec = t[-1] / 1000.0  # Convert ms to sec
    if t_sim_sec > 0:
        frequency = threshold_crossings / t_sim_sec
    else:
        frequency = 0.0
    
    return frequency, threshold_crossings


def test_single_stim_baseline():
    """TEST 1: Single stim backward compatibility (Phase 6 baseline)."""
    print("\n" + "="*70)
    print("TEST 1: Single Stim Backward Compatibility (Phase 6 Baseline)")
    print("="*70)
    
    cfg = FullModelConfig()
    apply_preset(cfg, 'B: Pyramidal L5 (Mainen 1996)')
    
    print(f"\nConfiguration:")
    print(f"  Neuron type: L5 Pyramidal")
    print(f"  Stim location: {cfg.stim_location.location}")
    print(f"  Stim type: {cfg.stim.stim_type}")
    print(f"  Iext: {cfg.stim.Iext} µA/cm²")
    print(f"  Simulation time: {cfg.stim.t_sim} ms")
    
    solver = NeuronSolver(cfg)
    result = solver.run_single()
    
    freq, n_spikes = get_soma_frequency(result.t, result.v_soma)
    print(f"\nResults:")
    print(f"  Firing frequency: {freq:.2f} Hz")
    print(f"  Number of spikes: {n_spikes}")
    print(f"  V_soma range: [{result.v_soma.min():.1f}, {result.v_soma.max():.1f}] mV")
    
    # Expected: L5 fires at ~7.5 Hz (Phase 6 baseline)
    expected_freq = 7.5
    freq_tolerance = 1.0  # Hz
    
    if abs(freq - expected_freq) < freq_tolerance:
        print(f"\n✅ PASS: Frequency {freq:.2f} Hz is within expected range ({expected_freq}±{freq_tolerance})")
        return True
    else:
        print(f"\n❌ FAIL: Frequency {freq:.2f} Hz is outside expected range ({expected_freq}±{freq_tolerance})")
        return False


def test_soma_dendritic_inhibition():
    """TEST 2: Soma excitation + Dendritic GABA-A inhibition."""
    print("\n" + "="*70)
    print("TEST 2: Soma Excitation + Dendritic GABA-A Inhibition")
    print("="*70)
    
    # Create baseline (soma only)
    cfg_baseline = FullModelConfig()
    apply_preset(cfg_baseline, 'B: Pyramidal L5 (Mainen 1996)')
    
    solver_baseline = NeuronSolver(cfg_baseline)
    result_baseline = solver_baseline.run_single()
    freq_baseline, _ = get_soma_frequency(result_baseline.t, result_baseline.v_soma)
    
    # Create dual stim config
    cfg_dual = FullModelConfig()
    apply_preset(cfg_dual, 'B: Pyramidal L5 (Mainen 1996)')
    
    cfg_dual.dual_stimulation = DualStimulationConfig()
    cfg_dual.dual_stimulation.enabled = True
    cfg_dual.dual_stimulation.primary_location = 'soma'
    cfg_dual.dual_stimulation.primary_stim_type = 'const'
    cfg_dual.dual_stimulation.primary_Iext = cfg_baseline.stim.Iext  # Same as baseline
    cfg_dual.dual_stimulation.primary_start = cfg_baseline.stim.pulse_start
    
    cfg_dual.dual_stimulation.secondary_location = 'dendritic_filtered'
    cfg_dual.dual_stimulation.secondary_stim_type = 'GABAA'
    cfg_dual.dual_stimulation.secondary_Iext = 5.0  # Inhibitory
    cfg_dual.dual_stimulation.secondary_start = cfg_baseline.stim.pulse_start
    
    print(f"\nBaseline configuration (soma only):")
    print(f"  Stim location: {cfg_baseline.stim_location.location}")
    print(f"  Iext: {cfg_baseline.stim.Iext} µA/cm²")
    print(f"  Frequency: {freq_baseline:.2f} Hz")
    
    print(f"\nDual stim configuration:")
    print(f"  Primary: {cfg_dual.dual_stimulation.primary_location} {cfg_dual.dual_stimulation.primary_stim_type}")
    print(f"  Primary Iext: {cfg_dual.dual_stimulation.primary_Iext} µA/cm²")
    print(f"  Secondary: {cfg_dual.dual_stimulation.secondary_location} {cfg_dual.dual_stimulation.secondary_stim_type}")
    print(f"  Secondary Iext: {cfg_dual.dual_stimulation.secondary_Iext} µA/cm²")
    
    solver_dual = NeuronSolver(cfg_dual)
    result_dual = solver_dual.run_single()
    freq_dual, _ = get_soma_frequency(result_dual.t, result_dual.v_soma)
    
    print(f"\nResults:")
    print(f"  Baseline frequency: {freq_baseline:.2f} Hz")
    print(f"  Dual stim frequency: {freq_dual:.2f} Hz")
    print(f"  Frequency change: {freq_baseline - freq_dual:.2f} Hz ({100*(freq_baseline - freq_dual)/freq_baseline:.1f}%)")
    
    # Expected: Inhibition should reduce frequency by at least 10%
    freq_reduction = (freq_baseline - freq_dual) / freq_baseline
    
    if freq_reduction > 0.1:  # At least 10% reduction
        print(f"\n✅ PASS: Inhibition reduced frequency by {100*freq_reduction:.1f}%")
        return True
    else:
        print(f"\n❌ FAIL: Inhibition only reduced frequency by {100*freq_reduction:.1f}% (expected >10%)")
        return False


def test_soma_ais_offset():
    """TEST 3: Soma + AIS with phase offset."""
    print("\n" + "="*70)
    print("TEST 3: Soma + AIS Excitation with Phase Offset")
    print("="*70)
    
    cfg = FullModelConfig()
    apply_preset(cfg, 'B: Pyramidal L5 (Mainen 1996)')
    
    cfg.dual_stimulation = DualStimulationConfig()
    cfg.dual_stimulation.enabled = True
    cfg.dual_stimulation.primary_location = 'soma'
    cfg.dual_stimulation.primary_stim_type = 'const'
    cfg.dual_stimulation.primary_Iext = 5.0
    cfg.dual_stimulation.primary_start = 100.0
    
    cfg.dual_stimulation.secondary_location = 'ais'
    cfg.dual_stimulation.secondary_stim_type = 'const'
    cfg.dual_stimulation.secondary_Iext = 5.0
    cfg.dual_stimulation.secondary_start = 150.0  # 50ms delay
    
    print(f"\nConfiguration:")
    print(f"  Primary: {cfg.dual_stimulation.primary_location} @ {cfg.dual_stimulation.primary_start}ms, Iext={cfg.dual_stimulation.primary_Iext}")
    print(f"  Secondary: {cfg.dual_stimulation.secondary_location} @ {cfg.dual_stimulation.secondary_start}ms, Iext={cfg.dual_stimulation.secondary_Iext}")
    print(f"  Phase offset: {cfg.dual_stimulation.secondary_start - cfg.dual_stimulation.primary_start}ms")
    
    solver = NeuronSolver(cfg)
    result = solver.run_single()
    freq, n_spikes = get_soma_frequency(result.t, result.v_soma)
    
    print(f"\nResults:")
    print(f"  Firing frequency: {freq:.2f} Hz")
    print(f"  Number of spikes: {n_spikes}")
    print(f"  V_soma range: [{result.v_soma.min():.1f}, {result.v_soma.max():.1f}] mV")
    
    # Expected: Should have some spiking (combined stimulus > single soma)
    if n_spikes > 0:
        print(f"\n✅ PASS: Dual AIS/soma stimulus produced {n_spikes} spikes")
        return True
    else:
        print(f"\n❌ FAIL: Dual AIS/soma stimulus produced no spikes")
        return False


def main():
    """Run all dual stimulation integration tests."""
    print("\n" + "="*70)
    print("DUAL STIMULATION INTEGRATION TEST SUITE")
    print("="*70)
    
    results = {}
    
    try:
        results['test_1_backward_compat'] = test_single_stim_baseline()
    except Exception as e:
        print(f"\n❌ TEST 1 EXCEPTION: {e}")
        results['test_1_backward_compat'] = False
    
    try:
        results['test_2_soma_dend'] = test_soma_dendritic_inhibition()
    except Exception as e:
        print(f"\n❌ TEST 2 EXCEPTION: {e}")
        results['test_2_soma_dend'] = False
    
    try:
        results['test_3_soma_ais'] = test_soma_ais_offset()
    except Exception as e:
        print(f"\n❌ TEST 3 EXCEPTION: {e}")
        results['test_3_soma_ais'] = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    n_passed = sum(1 for v in results.values() if v)
    n_total = len(results)
    
    print(f"\nTotal: {n_passed}/{n_total} tests passed")
    
    if n_passed == n_total:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️  {n_total - n_passed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit(main())
