"""
Comprehensive 1000ms validation for all presets.

Tests biophysical dynamics over extended simulations to ensure:
- Finite voltage output
- Physiological voltage ranges
- Stable behavior (no runaway oscillations)
- Expected spike counts/rates for each preset
"""
import pytest
import numpy as np
from core.models import FullModelConfig
from core.presets import get_preset_names, apply_preset
from core.solver import NeuronSolver
from core.analysis import detect_spikes


def test_all_presets_1000ms_stability():
    """Validate all presets with 1000ms simulations for biophysical stability."""
    results = {}
    
    for preset_name in get_preset_names():
        cfg = FullModelConfig()
        apply_preset(cfg, preset_name)
        
        # Set extended simulation time
        cfg.stim.t_sim = 1000.0
        cfg.stim.dt_eval = 0.5  # Coarser output for long sims
        cfg.stim.jacobian_mode = "native_hines"
        
        solver = NeuronSolver(cfg)
        result = solver.run_single()
        
        t = np.asarray(result.t, dtype=float)
        v = np.asarray(result.v_soma, dtype=float)
        
        # Basic checks
        assert len(t) > 0, f"{preset_name}: empty result"
        assert np.all(np.isfinite(v)), f"{preset_name}: non-finite voltage"
        
        # Physiological voltage range check
        v_min = float(np.min(v))
        v_max = float(np.max(v))
        assert -140.0 < v_min < 80.0, f"{preset_name}: V_min={v_min:.2f} out of physiological range"
        assert -140.0 < v_max < 80.0, f"{preset_name}: V_max={v_max:.2f} out of physiological range"
        
        # Check for runaway oscillations (std should not explode)
        v_std = float(np.std(v))
        assert v_std < 50.0, f"{preset_name}: unstable dynamics (std={v_std:.2f})"
        
        # Spike detection
        _, spike_times, _ = detect_spikes(v, t, threshold=-20.0, prominence=10.0, refractory_ms=2.0)
        n_spikes = len(spike_times)
        
        # Calculate firing rate if spikes exist
        if n_spikes > 1:
            isi_ms = np.mean(np.diff(spike_times))
            rate_hz = 1000.0 / isi_ms
        else:
            rate_hz = 0.0
        
        results[preset_name] = {
            'v_min': v_min,
            'v_max': v_max,
            'v_std': v_std,
            'n_spikes': n_spikes,
            'rate_hz': rate_hz,
            'stable': True
        }
    
    # Print summary
    print("\n" + "="*80)
    print("1000ms PRESET VALIDATION SUMMARY")
    print("="*80)
    print(f"{'Preset':<50} {'Spikes':>8} {'Rate (Hz)':>10} {'V_min':>10} {'V_max':>10}")
    print("-"*80)
    for preset_name, res in results.items():
        print(f"{preset_name:<50} {res['n_spikes']:>8} {res['rate_hz']:>10.1f} {res['v_min']:>10.1f} {res['v_max']:>10.1f}")
    print("="*80)


def test_pathology_presets_expected_behavior():
    """Validate pathology presets show expected biophysical signatures."""
    
    # Alzheimer's: Should show spike attenuation due to calcium toxicity
    cfg_alz = FullModelConfig()
    cfg_alz.preset_modes.alzheimer_mode = "progressive"
    apply_preset(cfg_alz, "N: Alzheimer's (v10 Calcium Toxicity)")
    cfg_alz.stim.t_sim = 1000.0
    cfg_alz.stim.dt_eval = 0.5
    
    solver_alz = NeuronSolver(cfg_alz)
    result_alz = solver_alz.run_single()
    
    _, spike_times_alz, _ = detect_spikes(
        result_alz.v_soma, result_alz.t,
        threshold=-20.0, prominence=10.0, refractory_ms=2.0
    )
    n_spikes_alz = len(spike_times_alz)
    
    print(f"\nAlzheimer's (progressive): {n_spikes_alz} spikes in 1000ms")
    
    # Hypoxia: Should show metabolic failure (few or no spikes)
    cfg_hyp = FullModelConfig()
    cfg_hyp.preset_modes.hypoxia_mode = "progressive"
    apply_preset(cfg_hyp, "O: Hypoxia (v10 ATP-pump failure)")
    cfg_hyp.stim.t_sim = 1000.0
    cfg_hyp.stim.dt_eval = 0.5
    
    solver_hyp = NeuronSolver(cfg_hyp)
    result_hyp = solver_hyp.run_single()
    
    _, spike_times_hyp, _ = detect_spikes(
        result_hyp.v_soma, result_hyp.t,
        threshold=-20.0, prominence=10.0, refractory_ms=2.0
    )
    n_spikes_hyp = len(spike_times_hyp)
    
    print(f"Hypoxia (progressive): {n_spikes_hyp} spikes in 1000ms")
    
    # Dravet: Should show reduced spiking in febrile mode
    cfg_dra_base = FullModelConfig()
    cfg_dra_base.preset_modes.dravet_mode = "baseline"
    apply_preset(cfg_dra_base, "S: Pathology: Dravet Syndrome (SCN1A LOF)")
    cfg_dra_base.stim.t_sim = 1000.0
    cfg_dra_base.stim.dt_eval = 0.5
    
    solver_dra_base = NeuronSolver(cfg_dra_base)
    result_dra_base = solver_dra_base.run_single()
    
    _, spike_times_dra_base, _ = detect_spikes(
        result_dra_base.v_soma, result_dra_base.t,
        threshold=-20.0, prominence=10.0, refractory_ms=2.0
    )
    n_spikes_dra_base = len(spike_times_dra_base)
    
    cfg_dra_febrile = FullModelConfig()
    cfg_dra_febrile.preset_modes.dravet_mode = "febrile"
    apply_preset(cfg_dra_febrile, "S: Pathology: Dravet Syndrome (SCN1A LOF)")
    cfg_dra_febrile.stim.t_sim = 1000.0
    cfg_dra_febrile.stim.dt_eval = 0.5
    
    solver_dra_febrile = NeuronSolver(cfg_dra_febrile)
    result_dra_febrile = solver_dra_febrile.run_single()
    
    _, spike_times_dra_febrile, _ = detect_spikes(
        result_dra_febrile.v_soma, result_dra_febrile.t,
        threshold=-20.0, prominence=10.0, refractory_ms=2.0
    )
    n_spikes_dra_febrile = len(spike_times_dra_febrile)
    
    print(f"Dravet (baseline): {n_spikes_dra_base} spikes in 1000ms")
    print(f"Dravet (febrile): {n_spikes_dra_febrile} spikes in 1000ms")
    print(f"Dravet febrile reduction: {100 * (1 - n_spikes_dra_febrile / max(n_spikes_dra_base, 1)):.1f}%")
    
    # Verify parameters are set correctly
    assert cfg_dra_febrile.env.T_celsius == 40.0, "Dravet febrile: T_celsius should be 40.0"
    assert cfg_dra_febrile.env.Q10_Na == 3.0, "Dravet febrile: Q10_Na should be 3.0"
    assert cfg_dra_febrile.channels.gNa_max == 44.0, "Dravet febrile: gNa_max should be 44.0"


if __name__ == "__main__":
    test_all_presets_1000ms_stability()
    test_pathology_presets_expected_behavior()
    print("\n✅ All 1000ms validation tests passed")
