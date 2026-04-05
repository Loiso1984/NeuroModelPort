"""
Deterministic extended conduction validation for demyelination (F) vs control (D).

Checks across a configurable grid of:
- temperature
- F-specific Ra multiplier
- F-specific gL multiplier
- drive scaling

Acceptance per case:
- D and F remain spiking/stable,
- F propagation delay exceeds D by margin,
- and F shows reduced relative transfer ratio OR clear absolute attenuation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.utils.runtime_import_guard import dependency_diagnostic

try:
    from core.models import FullModelConfig
    from core.presets import apply_preset
    from core.solver import NeuronSolver
except ModuleNotFoundError as exc:
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _parse_csv_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


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


def _first_cross(v: np.ndarray, t: np.ndarray, threshold: float = 0.0) -> float:
    idx = np.where((v[:-1] < threshold) & (v[1:] >= threshold))[0]
    return float(t[idx[0] + 1]) if len(idx) else float("nan")


def _run_case(
    preset: str,
    *,
    t_celsius: float,
    i_mult: float,
    ra_mult: float,
    gl_mult: float,
    t_sim: float,
    dt_eval: float,
) -> dict:
    cfg = FullModelConfig()
    apply_preset(cfg, preset)
    cfg.env.T_celsius = float(t_celsius)
    cfg.stim.Iext *= float(i_mult)
    cfg.morphology.Ra *= float(ra_mult)
    cfg.channels.gL *= float(gl_mult)
    cfg.stim.t_sim = float(t_sim)
    cfg.stim.dt_eval = float(dt_eval)
    cfg.stim.jacobian_mode = "sparse_fd"

    res = NeuronSolver(cfg).run_single()
    if cfg.morphology.N_trunk > 0:
        j = min(1 + cfg.morphology.N_ais + cfg.morphology.N_trunk - 1, res.n_comp - 1)
    elif cfg.morphology.N_ais > 0:
        j = min(cfg.morphology.N_ais, res.n_comp - 1)
    else:
        j = min(1, res.n_comp - 1)

    peak_s = float(np.max(res.v_soma))
    peak_j = float(np.max(res.v_all[j, :]))
    ts = _first_cross(res.v_soma, res.t, 0.0)
    tt = _first_cross(res.v_all[-1, :], res.t, 0.0)
    delay = float(tt - ts) if np.isfinite(ts) and np.isfinite(tt) else float("nan")

    return {
        "n_spikes": int(len(_spike_times(res.v_soma, res.t))),
        "soma_peak_mV": peak_s,
        "junction_peak_mV": peak_j,
        "prop_ratio": float(peak_j / max(peak_s, 1e-9)),
        "term_delay_ms": delay,
        "stable": bool(np.all(np.isfinite(res.v_soma))),
    }


def main() -> int:
    if _IMPORT_ERROR is not None:
        print(dependency_diagnostic("f-conduction-extended", _IMPORT_ERROR), file=sys.stderr)
        return 2

    parser = argparse.ArgumentParser(description="Extended F-vs-D conduction validation")
    parser.add_argument("--temps", type=str, default="23,30,37")
    parser.add_argument("--f-ra-mults", type=str, default="0.9,1.0,1.2")
    parser.add_argument("--f-gl-mults", type=str, default="0.85,1.0,1.2")
    parser.add_argument("--i-mults", type=str, default="0.8,1.0,1.2")
    parser.add_argument("--t-sim", type=float, default=320.0)
    parser.add_argument("--dt-eval", type=float, default=0.15)
    parser.add_argument("--delay-margin-ms", type=float, default=0.3)
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=0.3,
        help="Target upper bound for F propagation ratio (junction_peak/soma_peak).",
    )
    parser.add_argument("--output", type=str, default="_test_results/pathology_f_conduction_extended.json")
    args = parser.parse_args()

    temps = _parse_csv_floats(args.temps)
    f_ra = _parse_csv_floats(args.f_ra_mults)
    f_gl = _parse_csv_floats(args.f_gl_mults)
    i_mults = _parse_csv_floats(args.i_mults)

    rows = []
    anomalies = []
    total = len(temps) * len(f_ra) * len(f_gl) * len(i_mults)
    done = 0

    for t_c in temps:
        for ra_mult in f_ra:
            for gl_mult in f_gl:
                for i_mult in i_mults:
                    d = _run_case(
                        "D: alpha-Motoneuron (Powers 2001)",
                        t_celsius=t_c,
                        i_mult=i_mult,
                        ra_mult=1.0,
                        gl_mult=1.0,
                        t_sim=args.t_sim,
                        dt_eval=args.dt_eval,
                    )
                    f = _run_case(
                        "F: Multiple Sclerosis (Demyelination)",
                        t_celsius=t_c,
                        i_mult=i_mult,
                        ra_mult=ra_mult,
                        gl_mult=gl_mult,
                        t_sim=args.t_sim,
                        dt_eval=args.dt_eval,
                    )

                    delay_ok = bool(
                        np.isfinite(d["term_delay_ms"])
                        and np.isfinite(f["term_delay_ms"])
                        and (f["term_delay_ms"] > d["term_delay_ms"] + args.delay_margin_ms)
                    )
                    ratio_ok = bool(f["prop_ratio"] < d["prop_ratio"])
                    ratio_target_ok = bool(f["prop_ratio"] <= float(args.target_ratio))
                    abs_peak_ok = bool(f["junction_peak_mV"] < d["junction_peak_mV"] - 3.0)
                    spike_ok = bool(d["n_spikes"] >= 1 and f["n_spikes"] >= 1)
                    stable_ok = bool(d["stable"] and f["stable"])
                    ok = bool(delay_ok and (ratio_target_ok or abs_peak_ok) and spike_ok and stable_ok)

                    row = {
                        "t_celsius": float(t_c),
                        "f_ra_mult": float(ra_mult),
                        "f_gl_mult": float(gl_mult),
                        "i_mult": float(i_mult),
                        "D": d,
                        "F": f,
                        "ok": ok,
                    }
                    rows.append(row)
                    done += 1
                    print(
                        f"{done:03d}/{total} T={t_c:.1f} ra={ra_mult:.2f} gl={gl_mult:.2f} i={i_mult:.2f} ok={ok}",
                        flush=True,
                    )

                    if not ok:
                        anomalies.append(
                            {
                                "t_celsius": float(t_c),
                                "f_ra_mult": float(ra_mult),
                                "f_gl_mult": float(gl_mult),
                                "i_mult": float(i_mult),
                                "delay_ok": delay_ok,
                                "ratio_ok": ratio_ok,
                                "ratio_target_ok": ratio_target_ok,
                                "abs_peak_ok": abs_peak_ok,
                                "spike_ok": spike_ok,
                                "stable_ok": stable_ok,
                                "D": d,
                                "F": f,
                            }
                        )

    out = {
        "config": {
            "temps": temps,
            "f_ra_mults": f_ra,
            "f_gl_mults": f_gl,
            "i_mults": i_mults,
            "t_sim": float(args.t_sim),
            "dt_eval": float(args.dt_eval),
            "delay_margin_ms": float(args.delay_margin_ms),
        },
        "summary": {
            "total_cases": len(rows),
            "anomaly_count": len(anomalies),
            "pass_ratio": float((len(rows) - len(anomalies)) / max(1, len(rows))),
        },
        "anomalies": anomalies,
        "rows": rows,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    print(f"Anomalies: {len(anomalies)} / {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
