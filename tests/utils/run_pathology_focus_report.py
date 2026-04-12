"""
Focused pathology report for N/O/F presets.

Checks:
1) N/O mode ordering under deterministic perturbations:
   terminal should be less excitable than progressive.
2) F demyelination conduction phenotype vs D control:
   soma spikes preserved, axonal propagation delayed/weakened.
"""

from __future__ import annotations

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
    cfg.stim.Iext *= float(i_scale)
    cfg.env.T_celsius = float(t_c)
    cfg.stim.t_sim = t_sim
    cfg.stim.dt_eval = dt_eval
    cfg.stim.jacobian_mode = "native_hines"

    res = NeuronSolver(cfg).run_single()
    st = _spike_times(res.v_soma, res.t)
    mid = 0.5 * t_sim
    first = int(np.sum(st < mid))
    second = int(np.sum(st >= mid))
    return {
        "preset": preset,
        "mode": mode,
        "i_scale": float(i_scale),
        "t_celsius": float(t_c),
        "n_spikes": int(len(st)),
        "first_half": first,
        "second_half": second,
        "v_peak_mV": float(np.max(res.v_soma)),
        "v_tail_mV": float(np.mean(res.v_soma[-80:])),
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
    cfg.stim.Iext *= float(i_scale)
    cfg.env.T_celsius = float(t_c)
    cfg.stim.t_sim = t_sim
    cfg.stim.dt_eval = dt_eval
    cfg.stim.jacobian_mode = "native_hines"

    res = NeuronSolver(cfg).run_single()
    st = _spike_times(res.v_soma, res.t)
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
    delay = float(tt - ts) if not (np.isnan(ts) or np.isnan(tt)) else float("nan")
    return {
        "preset": preset,
        "i_scale": float(i_scale),
        "t_celsius": float(t_c),
        "n_spikes": int(len(st)),
        "soma_peak_mV": peak_s,
        "junction_peak_mV": peak_j,
        "prop_ratio": float(peak_j / max(peak_s, 1e-9)),
        "term_delay_ms": delay,
        "stable": bool(np.all(np.isfinite(res.v_soma))),
    }


def main() -> int:
    out_dir = Path("_test_results")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "pathology_focus_report.json"

    i_scales = [0.8, 1.0, 1.2]
    temps = [23.0, 30.0, 37.0]

    mode_rows = []
    anomalies = []
    for preset in [
        "N: Alzheimer's (v10 Calcium Toxicity)",
        "O: Hypoxia (v10 ATP-pump failure)",
    ]:
        for i_scale in i_scales:
            for t_c in temps:
                prog = _run_mode_case(preset, "progressive", i_scale, t_c, t_sim=220.0, dt_eval=0.3)
                term = _run_mode_case(preset, "terminal", i_scale, t_c, t_sim=220.0, dt_eval=0.3)
                mode_rows.extend([prog, term])

                if term["n_spikes"] > prog["n_spikes"]:
                    anomalies.append({
                        "type": "mode_order_violation",
                        "preset": preset,
                        "i_scale": i_scale,
                        "t_celsius": t_c,
                        "progressive_spikes": prog["n_spikes"],
                        "terminal_spikes": term["n_spikes"],
                    })

                if preset.startswith("O:") and prog["first_half"] < prog["second_half"]:
                    anomalies.append({
                        "type": "hypoxia_progressive_late_gain",
                        "preset": preset,
                        "i_scale": i_scale,
                        "t_celsius": t_c,
                        "first_half": prog["first_half"],
                        "second_half": prog["second_half"],
                    })

    conduction_rows = []
    for i_scale in i_scales:
        for t_c in temps:
            row_d = _run_conduction_case("D: alpha-Motoneuron (Powers 2001)", i_scale, t_c, t_sim=220.0, dt_eval=0.2)
            row_f = _run_conduction_case("F: Multiple Sclerosis (Demyelination)", i_scale, t_c, t_sim=220.0, dt_eval=0.2)
            conduction_rows.extend([row_d, row_f])

            if row_f["n_spikes"] <= 0:
                anomalies.append({
                    "type": "f_no_soma_spikes",
                    "i_scale": i_scale,
                    "t_celsius": t_c,
                    "f_row": row_f,
                })
            d_delay = row_d["term_delay_ms"]
            f_delay = row_f["term_delay_ms"]
            delay_ok = bool(np.isfinite(d_delay) and np.isfinite(f_delay) and (f_delay > d_delay + 0.3))
            ratio_ok = bool(row_f["prop_ratio"] < row_d["prop_ratio"])
            abs_peak_ok = bool(row_f["junction_peak_mV"] < row_d["junction_peak_mV"] - 3.0)
            # Accept either reduced relative transfer ratio or clear absolute attenuation,
            # but always require delayed propagation.
            if not (delay_ok and (ratio_ok or abs_peak_ok)):
                anomalies.append({
                    "type": "f_conduction_signature_violation",
                    "i_scale": i_scale,
                    "t_celsius": t_c,
                    "d_delay_ms": d_delay,
                    "f_delay_ms": f_delay,
                    "d_prop_ratio": row_d["prop_ratio"],
                    "f_prop_ratio": row_f["prop_ratio"],
                    "d_junction_peak_mV": row_d["junction_peak_mV"],
                    "f_junction_peak_mV": row_f["junction_peak_mV"],
                })

    mode_pairs = 0
    mode_pairs_ok = 0
    for preset in ["N: Alzheimer's (v10 Calcium Toxicity)", "O: Hypoxia (v10 ATP-pump failure)"]:
        for i_scale in i_scales:
            for t_c in temps:
                mode_pairs += 1
                prog = next(r for r in mode_rows if r["preset"] == preset and r["mode"] == "progressive" and r["i_scale"] == i_scale and r["t_celsius"] == t_c)
                term = next(r for r in mode_rows if r["preset"] == preset and r["mode"] == "terminal" and r["i_scale"] == i_scale and r["t_celsius"] == t_c)
                if term["n_spikes"] <= prog["n_spikes"]:
                    mode_pairs_ok += 1

    out = {
        "config": {
            "i_scales": i_scales,
            "temps_celsius": temps,
            "mode_t_sim": 220.0,
            "mode_dt_eval": 0.3,
            "conduction_t_sim": 220.0,
            "conduction_dt_eval": 0.2,
        },
        "summary": {
            "mode_pairs": mode_pairs,
            "mode_pairs_ok": mode_pairs_ok,
            "mode_pairs_ok_ratio": float(mode_pairs_ok / max(1, mode_pairs)),
            "anomaly_count": len(anomalies),
        },
        "mode_rows": mode_rows,
        "conduction_rows": conduction_rows,
        "anomalies": anomalies,
    }
    out_file.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved: {out_file}")
    print(
        f"Mode ordering: {mode_pairs_ok}/{mode_pairs} "
        f"({100.0 * mode_pairs_ok / max(1, mode_pairs):.1f}%)"
    )
    print(f"Anomalies: {len(anomalies)}")
    if anomalies:
        for i, a in enumerate(anomalies[:10], start=1):
            print(f"{i:02d}. {a['type']} -> {a}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
