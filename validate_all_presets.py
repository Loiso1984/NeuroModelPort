"""
Comprehensive Preset Validation v10.1
Runs all 15 presets, measures spike characteristics, compares against literature.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from core.models import FullModelConfig
from core.presets import get_preset_names, apply_preset
from core.solver import NeuronSolver
from core.analysis import detect_spikes

# Literature target frequencies (Hz) for SUSTAINED stimulation
# Source: various electrophysiology studies
TARGETS = {
    "Squid":       {"freq_range": (30, 80),   "mode": "repetitive",  "note": "HH 1952: 60-70 Hz at ~10 uA/cm2"},
    "Pyramidal L5":{"freq_range": (5, 20),    "mode": "regular",     "note": "Mainen 1996: 5-15 Hz regular spiking"},
    "FS Interneuron":{"freq_range": (40, 200),"mode": "fast",        "note": "Wang-Buzsaki: 40-200 Hz, no adaptation"},
    "alpha-Motoneuron":{"freq_range": (8, 35),"mode": "regular",     "note": "Powers 2001: 8-30 Hz steady-state"},
    "Purkinje":    {"freq_range": (30, 80),   "mode": "complex",     "note": "De Schutter: 40-80 Hz simple spikes"},
    "Multiple Sclerosis":{"freq_range": (0, 15),"mode": "impaired",  "note": "Conduction slowed or blocked by demyelination"},
    "Anesthesia":  {"freq_range": (0, 0),     "mode": "blocked",     "note": "90% Na block -> no spikes expected"},
    "Hyperkalemia":{"freq_range": (0, 40),    "mode": "depolarized", "note": "Depolarization block or reduced firing"},
    "In Vitro":    {"freq_range": (3, 15),    "mode": "slow",        "note": "23C slows kinetics 2-3x vs 37C"},
    "C-Fiber":     {"freq_range": (2, 15),    "mode": "slow",        "note": "Unmyelinated: 2-10 Hz typical"},
    "Thalamic":    {"freq_range": (5, 30),    "mode": "tonic/burst", "note": "McCormick 1992: tonic 5-30 Hz, bursts 200+ Hz"},
    "Hippocampal CA1":{"freq_range": (4, 12), "mode": "theta",       "note": "Theta rhythm: 4-8 Hz, up to 12 Hz"},
    "Epilepsy":    {"freq_range": (100, 500), "mode": "paroxysmal",  "note": "SCN1A GoF: high-freq bursts, hyperexcitable"},
    "Alzheimer's": {"freq_range": (3, 15),    "mode": "impaired",    "note": "Similar to L5 but Ca toxicity causes dysfunction"},
    "Hypoxia":     {"freq_range": (0, 30),    "mode": "depolarized", "note": "Ion imbalance: reduced drive or depol. block"},
}

def find_target(name):
    for key, val in TARGETS.items():
        if key in name:
            return val
    return None

def run_validation():
    names = get_preset_names()
    results = []

    for name in names:
        cfg = FullModelConfig()
        apply_preset(cfg, name)

        # Use long simulation for frequency measurement
        cfg.stim.t_sim = 500.0
        cfg.stim.dt_eval = 0.025  # 40 kHz sampling for accurate spike detection

        target = find_target(name)

        try:
            solver = NeuronSolver(cfg)
            res = solver.run_single()

            # Detect spikes (threshold = -20 mV, minimum height)
            v_soma = res.v_soma
            t = res.t

            # Skip first 50ms transient
            mask = t > 50.0
            v_ss = v_soma[mask]
            t_ss = t[mask]

            peak_idx, spike_times, spike_amps = detect_spikes(v_ss, t_ss)
            n_spikes = len(peak_idx)

            if n_spikes > 1:
                isis = np.diff(spike_times)
                freq = 1000.0 / np.mean(isis)
                freq_cv = np.std(isis) / np.mean(isis) if np.mean(isis) > 0 else 0
            elif n_spikes == 1:
                freq = 0.0  # Single spike, can't measure frequency
                freq_cv = 0.0
            else:
                freq = 0.0
                freq_cv = 0.0

            v_peak = float(np.max(spike_amps)) if n_spikes > 0 else float(np.max(v_soma))
            v_min = float(np.min(v_ss)) if len(v_ss) > 0 else -65.0

            # Check attenuation if dendritic filtered
            attenuation = 1.0
            if cfg.stim_location.location == "dendritic_filtered" and cfg.dendritic_filter.enabled:
                attenuation = np.exp(-cfg.dendritic_filter.distance_um / cfg.dendritic_filter.space_constant_um)

            # Effective current at soma (steady-state for const, peak for alpha)
            if cfg.stim.stim_type == 'const':
                i_eff = cfg.stim.Iext * attenuation
            elif cfg.stim.stim_type == 'alpha':
                # Alpha peak is Iext at t=tau, but filtered by dendritic tau
                # Effective peak after filtering: reduced by factor ~tau_alpha/(tau_alpha+tau_dend)
                tau_a = cfg.stim.alpha_tau
                tau_d = cfg.dendritic_filter.tau_dendritic_ms if cfg.dendritic_filter.enabled else 0.0
                if tau_d > 0 and cfg.stim_location.location == "dendritic_filtered":
                    filter_loss = tau_a / (tau_a + tau_d)
                    i_eff = cfg.stim.Iext * attenuation * filter_loss
                else:
                    i_eff = cfg.stim.Iext * attenuation
            else:
                i_eff = cfg.stim.Iext * attenuation

            # Evaluate against target
            status = "?"
            if target:
                fmin, fmax = target["freq_range"]
                if fmin == 0 and fmax == 0:
                    status = "OK" if n_spikes == 0 else f"FAIL: {n_spikes} spikes (expect 0)"
                elif n_spikes == 0 and fmin > 0:
                    status = f"FAIL: no spikes (expect {fmin}-{fmax} Hz)"
                elif freq > 0:
                    if fmin <= freq <= fmax:
                        status = "OK"
                    elif freq < fmin:
                        status = f"LOW: {freq:.1f} Hz (expect {fmin}-{fmax})"
                    else:
                        status = f"HIGH: {freq:.1f} Hz (expect {fmin}-{fmax})"
                elif n_spikes == 1 and fmin > 0:
                    status = f"FAIL: only 1 spike (expect sustained {fmin}-{fmax} Hz)"
                elif n_spikes > 0 and fmin == 0:
                    status = "OK (some spikes, impaired range)"

            results.append({
                'name': name,
                'n_spikes': n_spikes,
                'freq': freq,
                'freq_cv': freq_cv,
                'v_peak': v_peak,
                'v_min': v_min,
                'stim_type': cfg.stim.stim_type,
                'stim_loc': cfg.stim_location.location,
                'Iext': cfg.stim.Iext,
                'i_eff': i_eff,
                'attenuation': attenuation,
                'phi': cfg.env.phi,
                'status': status,
                'target': target,
            })

            print(f"[{'OK' if 'OK' in status else 'XX'}] {name}")
            print(f"    Spikes: {n_spikes}, Freq: {freq:.1f} Hz, Peak: {v_peak:.1f} mV, AHP: {v_min:.1f} mV")
            print(f"    Stim: {cfg.stim.stim_type} {cfg.stim.Iext:.1f} uA/cm2 @ {cfg.stim_location.location}")
            print(f"    Atten: {attenuation:.3f}, I_eff: {i_eff:.2f} uA/cm2, phi: {cfg.env.phi:.2f}")
            print(f"    Status: {status}")
            if target:
                print(f"    Target: {target['note']}")
            print()

        except Exception as e:
            print(f"[ER] {name}: {e}")
            results.append({'name': name, 'error': str(e)})
            print()

    # Summary
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    ok = sum(1 for r in results if 'status' in r and 'OK' in r.get('status', ''))
    fail = sum(1 for r in results if 'status' in r and ('FAIL' in r.get('status', '') or 'HIGH' in r.get('status', '') or 'LOW' in r.get('status', '')))
    err = sum(1 for r in results if 'error' in r)
    print(f"OK: {ok}/15, FAIL: {fail}/15, ERROR: {err}/15")
    print()

    for r in results:
        if 'error' in r:
            print(f"  ERROR: {r['name']}: {r['error']}")
        elif 'FAIL' in r.get('status', '') or 'HIGH' in r.get('status', '') or 'LOW' in r.get('status', ''):
            print(f"  NEEDS FIX: {r['name']}: {r['status']}")

    return results

if __name__ == "__main__":
    run_validation()
