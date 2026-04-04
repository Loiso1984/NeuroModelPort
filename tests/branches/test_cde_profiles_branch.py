"""
Branch validation for C/D/E preset operating profiles.

Goal:
- keep firing regimes within physiologically plausible ranges,
- ensure robust spiking and sane voltage peaks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver


def _spike_times(v: np.ndarray, t: np.ndarray, threshold: float = -20.0) -> np.ndarray:
    idx = np.where((v[:-1] < threshold) & (v[1:] >= threshold))[0] + 1
    if len(idx) == 0:
        return np.array([], dtype=float)
    st = t[idx]
    keep = [0]
    for i in range(1, len(st)):
        if st[i] - st[keep[-1]] >= 1.0:
            keep.append(i)
    return st[keep]


def _run_profile(preset: str, t_sim: float = 220.0, dt_eval: float = 0.2) -> dict:
    cfg = FullModelConfig()
    apply_preset(cfg, preset)
    cfg.stim.t_sim = t_sim
    cfg.stim.dt_eval = dt_eval
    cfg.stim.jacobian_mode = "sparse_fd"

    res = NeuronSolver(cfg).run_single()
    st = _spike_times(res.v_soma, res.t)
    freq = float(1000.0 / np.mean(np.diff(st))) if len(st) > 1 else 0.0
    return {
        "n_spikes": int(len(st)),
        "freq_hz": freq,
        "v_peak": float(np.max(res.v_soma)),
        "v_tail": float(np.mean(res.v_soma[-100:])),
        "stable": bool(np.all(np.isfinite(res.v_soma))),
    }


def test_c_fs_profile():
    row = _run_profile("C: FS Interneuron (Wang-Buzsaki)")
    assert row["stable"], "C preset produced non-finite trace"
    assert row["n_spikes"] >= 10, "C preset should be robustly spiking"
    assert 80.0 <= row["freq_hz"] <= 220.0, f"C frequency out of FS range: {row['freq_hz']:.2f} Hz"
    assert 20.0 <= row["v_peak"] <= 60.0, f"C spike peak out of range: {row['v_peak']:.2f} mV"


def test_d_motoneuron_profile():
    row = _run_profile("D: alpha-Motoneuron (Powers 2001)")
    assert row["stable"], "D preset produced non-finite trace"
    assert row["n_spikes"] >= 8, "D preset should produce repeated output under default stimulus"
    assert 60.0 <= row["freq_hz"] <= 130.0, f"D frequency out of expected range: {row['freq_hz']:.2f} Hz"
    assert 20.0 <= row["v_peak"] <= 50.0, f"D spike peak out of range: {row['v_peak']:.2f} mV"


def test_e_purkinje_profile():
    row = _run_profile("E: Cerebellar Purkinje (De Schutter)")
    assert row["stable"], "E preset produced non-finite trace"
    assert row["n_spikes"] >= 8, "E preset should produce tonic spiking under default stimulus"
    assert 80.0 <= row["freq_hz"] <= 150.0, f"E frequency out of expected range: {row['freq_hz']:.2f} Hz"
    assert 20.0 <= row["v_peak"] <= 45.0, f"E spike peak out of range: {row['v_peak']:.2f} mV"


def _run_as_script() -> int:
    tests = [test_c_fs_profile, test_d_motoneuron_profile, test_e_purkinje_profile]
    passed = 0
    for fn in tests:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] {fn.__name__}: {exc}")
    print(f"\nSummary: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(_run_as_script())
