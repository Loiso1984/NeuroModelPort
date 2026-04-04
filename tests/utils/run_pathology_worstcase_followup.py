"""
Select near-boundary pathology cases from hard matrix and revalidate with stricter settings.

Input:
- `_test_results/pathology_hard_matrix.json` produced by hard deterministic sweep.

Output:
- `_test_results/pathology_worstcase_followup.json`
"""

from __future__ import annotations

import argparse
import json
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


def _first_cross(v: np.ndarray, t: np.ndarray, threshold: float = 0.0) -> float:
    idx = np.where((v[:-1] < threshold) & (v[1:] >= threshold))[0]
    return float(t[idx[0] + 1]) if len(idx) else float("nan")


def _run_mode_case(
    preset: str,
    mode: str,
    i_scale: float,
    t_c: float,
    gl_mult: float,
    tau_mult: float,
    gca_mult: float,
    *,
    t_sim: float,
    dt_eval: float,
) -> dict:
    cfg = FullModelConfig()
    if preset.startswith("N:"):
        cfg.preset_modes.alzheimer_mode = mode
    else:
        cfg.preset_modes.hypoxia_mode = mode
    apply_preset(cfg, preset)
    cfg.stim.Iext *= i_scale
    cfg.env.T_celsius = t_c
    cfg.channels.gL *= gl_mult
    if cfg.calcium.dynamic_Ca:
        cfg.calcium.tau_Ca *= tau_mult
        cfg.channels.gCa_max *= gca_mult
    cfg.stim.t_sim = t_sim
    cfg.stim.dt_eval = dt_eval
    cfg.stim.jacobian_mode = "sparse_fd"
    res = NeuronSolver(cfg).run_single()
    st = _spike_times(res.v_soma, res.t)
    mid = 0.5 * t_sim
    return {
        "n_spikes": int(len(st)),
        "first_half": int(np.sum(st < mid)),
        "second_half": int(np.sum(st >= mid)),
        "v_peak_mV": float(np.max(res.v_soma)),
        "v_tail_mV": float(np.mean(res.v_soma[-100:])),
        "stable": bool(np.all(np.isfinite(res.v_soma))),
    }


def _run_conduction_case(
    preset: str,
    i_scale: float,
    t_c: float,
    *,
    t_sim: float,
    dt_eval: float,
) -> dict:
    cfg = FullModelConfig()
    apply_preset(cfg, preset)
    cfg.stim.Iext *= i_scale
    cfg.env.T_celsius = t_c
    cfg.stim.t_sim = t_sim
    cfg.stim.dt_eval = dt_eval
    cfg.stim.jacobian_mode = "sparse_fd"
    res = NeuronSolver(cfg).run_single()
    j = min(1 + cfg.morphology.N_ais + cfg.morphology.N_trunk, res.n_comp - 1)
    peak_s = float(np.max(res.v_soma))
    peak_j = float(np.max(res.v_all[j, :]))
    ts = _first_cross(res.v_soma, res.t, 0.0)
    tt = _first_cross(res.v_all[-1, :], res.t, 0.0)
    return {
        "n_spikes": int(len(_spike_times(res.v_soma, res.t))),
        "soma_peak_mV": peak_s,
        "junction_peak_mV": peak_j,
        "prop_ratio": float(peak_j / max(peak_s, 1e-9)),
        "term_delay_ms": float(tt - ts) if np.isfinite(ts) and np.isfinite(tt) else float("nan"),
        "stable": bool(np.all(np.isfinite(res.v_soma))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run worst-case pathology follow-up checks")
    parser.add_argument("--input", type=str, default="_test_results/pathology_hard_matrix.json")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--t-sim", type=float, default=300.0)
    parser.add_argument("--dt-eval", type=float, default=0.2)
    parser.add_argument("--output", type=str, default="_test_results/pathology_worstcase_followup.json")
    args = parser.parse_args()

    hard = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = hard["rows"]

    mode_rows = [r for r in rows if r.get("kind") == "mode"]
    cond_rows = [r for r in rows if r.get("kind") == "cond"]

    risky_mode = []
    for r in mode_rows:
        preset = r["preset"]
        prog = r["prog"]
        term = r["term"]
        # Smaller margins => closer to violation.
        margin_order = float(prog["n"] - term["n"])
        margin_early = float(prog["first"] - prog["second"]) if preset.startswith("O:") else 999.0
        risk = min(margin_order, margin_early)
        risky_mode.append((risk, r))
    risky_mode.sort(key=lambda x: x[0])
    top_mode = [r for _, r in risky_mode[: max(1, args.top_k)]]

    risky_cond = []
    for r in cond_rows:
        d = r["D"]
        f = r["F"]
        dly = float(f["delay"] - d["delay"]) if np.isfinite(f["delay"]) and np.isfinite(d["delay"]) else -999.0
        ratio_margin = float(d["ratio"] - f["ratio"])
        abs_margin = float(d["j_peak"] - f["j_peak"])
        # lower is riskier
        risk = min(dly - 0.3, max(ratio_margin, abs_margin / 3.0))
        risky_cond.append((risk, r))
    risky_cond.sort(key=lambda x: x[0])
    top_cond = [r for _, r in risky_cond[: max(1, args.top_k)]]

    follow_mode = []
    anomalies = []
    for r in top_mode:
        preset = r["preset"]
        i = float(r["i"])
        t_c = float(r["T"])
        gl = float(r["gl"])
        tau = float(r["tau"])
        gca = float(r["gca"])
        prog = _run_mode_case(preset, "progressive", i, t_c, gl, tau, gca, t_sim=args.t_sim, dt_eval=args.dt_eval)
        term = _run_mode_case(preset, "terminal", i, t_c, gl, tau, gca, t_sim=args.t_sim, dt_eval=args.dt_eval)
        follow_mode.append({
            "preset": preset,
            "i_scale": i,
            "t_celsius": t_c,
            "gl_mult": gl,
            "tau_mult": tau,
            "gca_mult": gca,
            "progressive": prog,
            "terminal": term,
        })
        if term["n_spikes"] > prog["n_spikes"]:
            anomalies.append({
                "type": "mode_order_violation",
                "preset": preset,
                "i_scale": i,
                "t_celsius": t_c,
                "prog_spikes": prog["n_spikes"],
                "term_spikes": term["n_spikes"],
            })
        if preset.startswith("O:") and prog["second_half"] > prog["first_half"]:
            anomalies.append({
                "type": "hypoxia_late_gain",
                "i_scale": i,
                "t_celsius": t_c,
                "first_half": prog["first_half"],
                "second_half": prog["second_half"],
            })

    follow_cond = []
    for r in top_cond:
        i = float(r["i"])
        t_c = float(r["T"])
        d = _run_conduction_case("D: alpha-Motoneuron (Powers 2001)", i, t_c, t_sim=args.t_sim, dt_eval=args.dt_eval)
        f = _run_conduction_case("F: Multiple Sclerosis (Demyelination)", i, t_c, t_sim=args.t_sim, dt_eval=args.dt_eval)
        follow_cond.append({"i_scale": i, "t_celsius": t_c, "D": d, "F": f})

        delay_ok = bool(np.isfinite(d["term_delay_ms"]) and np.isfinite(f["term_delay_ms"]) and (f["term_delay_ms"] > d["term_delay_ms"] + 0.3))
        ratio_ok = bool(f["prop_ratio"] < d["prop_ratio"])
        abs_peak_ok = bool(f["junction_peak_mV"] < d["junction_peak_mV"] - 3.0)
        if not (delay_ok and (ratio_ok or abs_peak_ok)):
            anomalies.append({
                "type": "f_conduction_signature_violation",
                "i_scale": i,
                "t_celsius": t_c,
                "D": d,
                "F": f,
            })

    out = {
        "config": {
            "input": args.input,
            "top_k": int(args.top_k),
            "t_sim": float(args.t_sim),
            "dt_eval": float(args.dt_eval),
        },
        "mode_followup": follow_mode,
        "conduction_followup": follow_cond,
        "anomalies": anomalies,
        "summary": {
            "mode_cases": len(follow_mode),
            "conduction_cases": len(follow_cond),
            "anomaly_count": len(anomalies),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out_path}")
    print(f"Anomalies: {len(anomalies)}")
    if anomalies:
        for i, a in enumerate(anomalies[:10], start=1):
            print(f"{i:02d}. {a}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
