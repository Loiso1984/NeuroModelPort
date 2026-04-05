"""Deterministic impedance report for ZAP/chirp stimulation cases.

Outputs JSON artifact:
    _test_results/impedance_zap_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

ARTIFACT = Path("_test_results/impedance_zap_report.json")


CASES = [
    {
        "id": "P_TRN",
        "preset": "P: Thalamic Reticular Nucleus (TRN Spindles)",
        "i_scale": 0.5,
        "f_res_range": (0.5, 40.0),
    },
    {
        "id": "R_ACh",
        "preset": "R: Cholinergic Neuromodulation (ACh)",
        "i_scale": 0.5,
        "f_res_range": (0.5, 40.0),
    },
    {
        "id": "K_Thalamic",
        "preset": "K: Thalamic Relay (Ih + IT + Burst)",
        "i_scale": 0.5,
        "f_res_range": (0.5, 40.0),
    },
]


def run_case(case: dict, *, fmin_hz: float, fmax_hz: float) -> dict:
    from core.analysis import compute_membrane_impedance, reconstruct_stimulus_trace
    from core.models import FullModelConfig
    from core.presets import apply_preset
    from core.solver import NeuronSolver

    cfg = FullModelConfig()
    apply_preset(cfg, case["preset"])

    cfg.stim.stim_type = "zap"
    cfg.stim.pulse_start = 50.0
    cfg.stim.pulse_dur = 900.0
    cfg.stim.t_sim = 1100.0
    cfg.stim.dt_eval = 0.25
    cfg.stim.zap_f0_hz = 0.5
    cfg.stim.zap_f1_hz = 40.0
    cfg.stim.Iext = max(2.0, float(cfg.stim.Iext) * float(case.get("i_scale", 0.5)))

    res = NeuronSolver(cfg).run_single()
    i_stim = reconstruct_stimulus_trace(res)
    imp = compute_membrane_impedance(res.t, res.v_soma, i_stim, fmin_hz=fmin_hz, fmax_hz=fmax_hz)

    f_res = float(imp.get("f_res_hz", np.nan))
    z_res = float(imp.get("z_res_kohm_cm2", np.nan))
    valid = bool(imp.get("valid", False))

    case_lo, case_hi = case.get("f_res_range", (fmin_hz, fmax_hz))
    lo = max(float(case_lo), float(fmin_hz))
    hi = min(float(case_hi), float(fmax_hz))
    guard_reasons: list[str] = []
    if not valid:
        guard_reasons.append("invalid_impedance")
    if not np.isfinite(f_res):
        guard_reasons.append("non_finite_f_res")
    if not np.isfinite(z_res):
        guard_reasons.append("non_finite_z_res")
    if z_res <= 0.0:
        guard_reasons.append("non_positive_z_res")
    if lo > hi:
        guard_reasons.append("empty_expected_range")
    elif not (lo <= f_res <= hi):
        guard_reasons.append("f_res_out_of_expected_range")
    guard_ok = len(guard_reasons) == 0

    return {
        "id": case["id"],
        "preset": case["preset"],
        "valid": valid,
        "guard_ok": guard_ok,
        "guard_reasons": guard_reasons,
        "f_res_hz": f_res,
        "z_res_kohm_cm2": z_res,
        "f_res_expected_range_hz": [float(lo), float(hi)],
        "v_peak_mV": float(np.max(res.v_soma)),
        "v_min_mV": float(np.min(res.v_soma)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic ZAP impedance report")
    parser.add_argument("--fmin", type=float, default=0.5, help="Lower impedance analysis frequency bound")
    parser.add_argument("--fmax", type=float, default=80.0, help="Upper impedance analysis frequency bound")
    parser.add_argument("--strict", action="store_true", help="Return non-zero if any guard fails")
    args = parser.parse_args()
    if args.fmin <= 0.0 or args.fmax <= args.fmin:
        print("[WARN] invalid frequency bounds: require 0 < --fmin < --fmax")
        return 2

    try:
        import pydantic  # noqa:F401
    except Exception as exc:
        print(f"[WARN] missing dependency for report execution: {exc}")
        return 2

    rows = []
    for case in CASES:
        try:
            rows.append(run_case(case, fmin_hz=float(args.fmin), fmax_hz=float(args.fmax)))
        except Exception as exc:
            rows.append({
                "id": case["id"],
                "preset": case["preset"],
                "valid": False,
                "guard_ok": False,
                "guard_reasons": ["exception"],
                "error": str(exc),
            })

    ok = int(sum(1 for r in rows if r.get("guard_ok", False)))
    total = len(rows)
    artifact = {
        "rows": rows,
        "ok": ok,
        "total": total,
        "analysis_band_hz": [float(args.fmin), float(args.fmax)],
        "strict_mode": bool(args.strict),
        "all_guard_ok": (ok == total),
        "failed_case_ids": [r["id"] for r in rows if not r.get("guard_ok", False)],
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print(f"Saved: {ARTIFACT}")
    print(f"Guard OK: {ok}/{total}")

    if args.strict and ok != total:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
