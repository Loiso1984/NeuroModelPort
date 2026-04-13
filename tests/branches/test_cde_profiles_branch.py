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
from tests.shared_utils import _spike_times


def _run_profile(preset: str, t_sim: float = 220.0, dt_eval: float = 0.2) -> dict:
    cfg = FullModelConfig()
    apply_preset(cfg, preset)
    cfg.stim.t_sim = t_sim
    cfg.stim.dt_eval = dt_eval
    cfg.stim.jacobian_mode = "native_hines"

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


def _run_profile_scaled(
    preset: str,
    i_scale: float,
    *,
    t_sim: float = 150.0,
    dt_eval: float = 0.25,
) -> dict:
    cfg = FullModelConfig()
    apply_preset(cfg, preset)
    cfg.stim.Iext = float(cfg.stim.Iext * i_scale)
    cfg.stim.t_sim = t_sim
    cfg.stim.dt_eval = dt_eval
    cfg.stim.jacobian_mode = "native_hines"
    res = NeuronSolver(cfg).run_single()
    st = _spike_times(res.v_soma, res.t)
    return {
        "n_spikes": int(len(st)),
        "v_peak": float(np.max(res.v_soma)),
        "v_tail": float(np.mean(res.v_soma[-60:])),
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
    assert 100.0 <= row["freq_hz"] <= 260.0, f"E frequency out of expected range: {row['freq_hz']:.2f} Hz"
    assert 20.0 <= row["v_peak"] <= 45.0, f"E spike peak out of range: {row['v_peak']:.2f} mV"


def test_cde_drive_sweep_stability():
    presets = {
        "C: FS Interneuron (Wang-Buzsaki)": [0.7, 1.0, 1.3],
        "D: alpha-Motoneuron (Powers 2001)": [0.7, 1.0, 1.3],
        "E: Cerebellar Purkinje (De Schutter)": [0.7, 1.0, 1.3],
    }
    rows = {}
    for preset, scales in presets.items():
        rows[preset] = [_run_profile_scaled(preset, s) for s in scales]
        for r in rows[preset]:
            assert r["stable"], f"{preset}: non-finite trace in drive sweep"
            assert -140.0 < r["v_tail"] < 10.0, f"{preset}: tail out of guard range ({r['v_tail']:.2f} mV)"
            assert -120.0 < r["v_peak"] < 65.0, f"{preset}: peak out of guard range ({r['v_peak']:.2f} mV)"

    c_rows = rows["C: FS Interneuron (Wang-Buzsaki)"]
    assert c_rows[2]["n_spikes"] >= c_rows[0]["n_spikes"], "C should not lose excitability with stronger drive"

    d_rows = rows["D: alpha-Motoneuron (Powers 2001)"]
    assert d_rows[2]["n_spikes"] >= d_rows[0]["n_spikes"] - 1, "D sweep should remain robust under stronger drive"

    e_rows = rows["E: Cerebellar Purkinje (De Schutter)"]
    assert e_rows[1]["n_spikes"] >= 6, "E baseline drive should remain clearly spiking"
    assert e_rows[2]["n_spikes"] >= e_rows[1]["n_spikes"], "E should not weaken when drive increases"


def test_e_moderate_low_drive_not_silent_at_37c():
    row = _run_profile_scaled("E: Cerebellar Purkinje (De Schutter)", 0.8)
    assert row["stable"], "E preset produced non-finite trace at moderate low drive"
    assert row["n_spikes"] >= 6, (
        "E preset should stay excitable at 37C under moderate low drive (0.8x), "
        "not collapse into a silent island"
    )
    assert 15.0 <= row["v_peak"] <= 45.0, f"E low-drive spike peak out of guard range: {row['v_peak']:.2f} mV"


def _run_as_script() -> int:
    tests = [
        test_c_fs_profile,
        test_d_motoneuron_profile,
        test_e_purkinje_profile,
        test_cde_drive_sweep_stability,
        test_e_moderate_low_drive_not_silent_at_37c,
    ]
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
