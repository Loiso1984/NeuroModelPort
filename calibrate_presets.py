"""
Automated Preset Calibration v10.1
Finds optimal Iext for each preset to match target firing frequency.
Uses bisection on Iext with const stimulus for frequency calibration.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from core.models import FullModelConfig
from core.presets import get_preset_names, apply_preset
from core.solver import NeuronSolver
from core.analysis import detect_spikes

# Target frequencies (Hz) and special modes
TARGETS = {
    "Squid":            {"freq": 55,  "range": (30, 80),   "note": "HH repetitive firing"},
    "Pyramidal L5":     {"freq": 10,  "range": (5, 20),    "note": "Regular spiking"},
    "FS Interneuron":   {"freq": 80,  "range": (40, 200),  "note": "Fast spiking"},
    "alpha-Motoneuron": {"freq": 20,  "range": (8, 35),    "note": "Regular firing"},
    "Purkinje":         {"freq": 50,  "range": (30, 80),   "note": "Simple spikes"},
    "Multiple Sclerosis":{"freq": 8,  "range": (0, 15),    "note": "Impaired conduction"},
    "Anesthesia":       {"freq": 0,   "range": (0, 0),     "note": "No spikes (blocked)"},
    "Hyperkalemia":     {"freq": 0,   "range": (0, 5),     "note": "Depol. block possible"},
    "In Vitro":         {"freq": 8,   "range": (3, 15),    "note": "Slow (23C)"},
    "C-Fiber":          {"freq": 5,   "range": (2, 15),    "note": "Slow unmyelinated"},
    "Thalamic":         {"freq": 15,  "range": (5, 30),    "note": "Tonic mode"},
    "Hippocampal CA1":  {"freq": 7,   "range": (4, 12),    "note": "Theta"},
    "Epilepsy":         {"freq": 150, "range": (100, 500), "note": "Hyperexcitable"},
    "Alzheimer's":      {"freq": 8,   "range": (3, 15),    "note": "Impaired L5"},
    "Hypoxia":          {"freq": 10,  "range": (0, 30),    "note": "Depolarized"},
}

def find_target(name):
    for key, val in TARGETS.items():
        if key in name:
            return val
    return None

def measure_freq(cfg, t_sim=500.0):
    """Run simulation and measure firing frequency (Hz)."""
    cfg.stim.t_sim = t_sim
    cfg.stim.dt_eval = 0.025
    try:
        solver = NeuronSolver(cfg)
        res = solver.run_single()
        v = res.v_soma
        t = res.t
        # Skip first 50ms transient
        mask = t > 50.0
        peaks, sp_t, sp_amp = detect_spikes(v[mask], t[mask])
        n_spikes = len(peaks)
        if n_spikes > 1:
            freq = 1000.0 / np.mean(np.diff(sp_t))
        else:
            freq = 0.0
        v_peak = float(np.max(sp_amp)) if n_spikes > 0 else float(np.max(v))
        return n_spikes, freq, v_peak
    except Exception as e:
        return 0, 0.0, -65.0

def find_rheobase(name, i_min=0.1, i_max=2000.0, tol=0.5, max_iter=20):
    """Find minimum Iext for at least 2 spikes using bisection."""
    for _ in range(max_iter):
        i_test = (i_min + i_max) / 2.0
        cfg = FullModelConfig()
        apply_preset(cfg, name)
        cfg.stim.stim_type = 'const'  # Force const for calibration
        cfg.stim.Iext = i_test
        n_sp, freq, vpk = measure_freq(cfg, t_sim=300.0)

        if n_sp >= 2:
            i_max = i_test
        else:
            i_min = i_test

        if i_max - i_min < tol:
            break

    return i_max

def find_iext_for_freq(name, target_freq, i_min=0.1, i_max=5000.0, max_iter=25):
    """Find Iext that produces target frequency using bisection."""
    best_iext = None
    best_freq = 0.0
    best_diff = 1e9

    for iteration in range(max_iter):
        i_test = (i_min + i_max) / 2.0
        cfg = FullModelConfig()
        apply_preset(cfg, name)
        cfg.stim.stim_type = 'const'
        cfg.stim.Iext = i_test
        n_sp, freq, vpk = measure_freq(cfg, t_sim=500.0)

        diff = abs(freq - target_freq)
        if diff < best_diff:
            best_diff = diff
            best_iext = i_test
            best_freq = freq

        if abs(freq - target_freq) < 2.0:  # Within 2 Hz
            break

        if freq < target_freq:
            i_min = i_test
        else:
            i_max = i_test

        if i_max - i_min < 0.1:
            break

    return best_iext, best_freq

def main():
    names = get_preset_names()

    print("=" * 90)
    print("PRESET CALIBRATION - Finding optimal Iext for target frequencies")
    print("=" * 90)
    print()

    results = {}

    for name in names:
        target = find_target(name)
        if not target:
            print(f"SKIP: {name} (no target)")
            continue

        target_freq = target["freq"]
        freq_range = target["range"]

        print(f"\n{'='*80}")
        print(f"CALIBRATING: {name}")
        print(f"Target: {target_freq} Hz ({freq_range[0]}-{freq_range[1]} Hz) - {target['note']}")
        print(f"{'='*80}")

        # Skip no-spike presets
        if target_freq == 0:
            cfg = FullModelConfig()
            apply_preset(cfg, name)
            n_sp, freq, vpk = measure_freq(cfg, t_sim=300.0)
            status = "OK (no spikes)" if n_sp == 0 else f"ISSUE: {n_sp} spikes"
            print(f"  Result: {status}")
            results[name] = {'status': status, 'iext': cfg.stim.Iext, 'stim_type': cfg.stim.stim_type}
            continue

        # Step 1: Find rheobase
        print("  Step 1: Finding rheobase...")
        rheobase = find_rheobase(name)
        print(f"  Rheobase (const): ~{rheobase:.1f} uA/cm2")

        # Step 2: Find Iext for target frequency
        print(f"  Step 2: Finding Iext for {target_freq} Hz...")
        optimal_iext, achieved_freq = find_iext_for_freq(
            name, target_freq,
            i_min=rheobase * 0.9,
            i_max=rheobase * 50.0
        )

        # Step 3: Verify
        cfg = FullModelConfig()
        apply_preset(cfg, name)
        cfg.stim.stim_type = 'const'
        cfg.stim.Iext = optimal_iext
        n_sp, freq, vpk = measure_freq(cfg, t_sim=500.0)

        in_range = freq_range[0] <= freq <= freq_range[1]
        status = "OK" if in_range else f"{'HIGH' if freq > freq_range[1] else 'LOW'}"

        print(f"  RESULT: Iext={optimal_iext:.1f} uA/cm2 -> {freq:.1f} Hz, {n_sp} spikes, Vpeak={vpk:.1f} mV [{status}]")

        # Get current preset info for comparison
        cfg_orig = FullModelConfig()
        apply_preset(cfg_orig, name)

        # Calculate effective current
        atten = 1.0
        if cfg.stim_location.location == "dendritic_filtered" and cfg.dendritic_filter.enabled:
            atten = np.exp(-cfg.dendritic_filter.distance_um / cfg.dendritic_filter.space_constant_um)

        print(f"  Attenuation: {atten:.3f}, I_eff at soma: {optimal_iext * atten:.1f} uA/cm2")
        print(f"  Original: stim_type={cfg_orig.stim.stim_type}, Iext={cfg_orig.stim.Iext:.1f}")

        results[name] = {
            'status': status,
            'iext_optimal': optimal_iext,
            'freq_achieved': freq,
            'n_spikes': n_sp,
            'vpeak': vpk,
            'rheobase': rheobase,
            'attenuation': atten,
            'iext_original': cfg_orig.stim.Iext,
            'stim_type_original': cfg_orig.stim.stim_type,
        }

    # Summary
    print("\n" + "=" * 90)
    print("CALIBRATION SUMMARY")
    print("=" * 90)
    print(f"{'Preset':<45} {'Iext_opt':>8} {'Freq':>7} {'Status':>8} {'Orig_Iext':>10} {'Orig_type':>10}")
    print("-" * 90)

    for name in names:
        target = find_target(name)
        if not target or name not in results:
            continue
        r = results[name]
        if 'iext_optimal' in r:
            print(f"{name:<45} {r['iext_optimal']:>8.1f} {r['freq_achieved']:>6.1f}Hz {r['status']:>8} {r['iext_original']:>10.1f} {r['stim_type_original']:>10}")
        else:
            print(f"{name:<45} {'N/A':>8} {'N/A':>7} {r['status']:>8} {r.get('iext','?'):>10} {r.get('stim_type','?'):>10}")

    # Generate preset code
    print("\n" + "=" * 90)
    print("RECOMMENDED PRESET UPDATES (const stimulus, calibrated Iext):")
    print("=" * 90)
    for name in names:
        if name not in results:
            continue
        r = results[name]
        if 'iext_optimal' in r and r['status'] == 'OK':
            target = find_target(name)
            print(f"# {name}: {r['freq_achieved']:.1f} Hz (target {target['freq']} Hz)")
            print(f"cfg.stim.stim_type = 'const'")
            print(f"cfg.stim.Iext = {r['iext_optimal']:.1f}")
            print()

    return results

if __name__ == "__main__":
    main()
