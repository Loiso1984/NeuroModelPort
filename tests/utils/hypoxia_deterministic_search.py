"""
Deterministic coarse-to-fine search for hypoxia progressive mode.

No random sampling:
1) 20 fixed broad cases,
2) local deterministic refinement around top candidates.
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


def _evaluate(case: dict, t_sim: float, dt_eval: float) -> dict:
    cfg = FullModelConfig()
    cfg.preset_modes.hypoxia_mode = "progressive"
    apply_preset(cfg, "O: Hypoxia (v10 ATP-pump failure)")
    cfg.stim.stim_type = "const"
    cfg.stim.t_sim = t_sim
    cfg.stim.dt_eval = dt_eval
    cfg.channels.EK = case["EK"]
    cfg.channels.EL = case["EL"]
    cfg.channels.gL = case["gL"]
    cfg.stim.Iext = case["Iext"]
    cfg.calcium.tau_Ca = case["tau_Ca"]
    cfg.calcium.B_Ca = case["B_Ca"]
    cfg.channels.gCa_max = case["gCa_max"]

    res = NeuronSolver(cfg).run_single()
    st = _spike_times(res.v_soma, res.t)
    half = t_sim / 2.0
    first = int(np.sum(st < half))
    second = int(np.sum(st >= half))
    v_peak = float(np.max(res.v_soma))
    v_tail = float(np.mean(res.v_soma[-80:]))
    score = (first - second) * 4.0 + first * 1.5 - second * 2.0 - abs(v_peak - 30.0) / 6.0
    return {
        **case,
        "n_spikes": int(len(st)),
        "first_half": first,
        "second_half": second,
        "v_peak_mV": v_peak,
        "v_tail_mV": v_tail,
        "score": float(score),
    }


def _coarse_cases() -> list[dict]:
    return [
        {"EK": -62.0, "EL": -50.0, "gL": 0.12, "Iext": 24.0, "tau_Ca": 700.0, "B_Ca": 1e-5, "gCa_max": 0.06},
        {"EK": -62.0, "EL": -48.0, "gL": 0.15, "Iext": 28.0, "tau_Ca": 700.0, "B_Ca": 1e-5, "gCa_max": 0.08},
        {"EK": -62.0, "EL": -46.0, "gL": 0.20, "Iext": 32.0, "tau_Ca": 900.0, "B_Ca": 1e-5, "gCa_max": 0.08},
        {"EK": -60.0, "EL": -50.0, "gL": 0.12, "Iext": 30.0, "tau_Ca": 900.0, "B_Ca": 1e-5, "gCa_max": 0.08},
        {"EK": -60.0, "EL": -48.0, "gL": 0.15, "Iext": 34.0, "tau_Ca": 900.0, "B_Ca": 1e-5, "gCa_max": 0.10},
        {"EK": -60.0, "EL": -45.0, "gL": 0.20, "Iext": 38.0, "tau_Ca": 900.0, "B_Ca": 1e-5, "gCa_max": 0.10},
        {"EK": -58.0, "EL": -50.0, "gL": 0.12, "Iext": 34.0, "tau_Ca": 900.0, "B_Ca": 1e-5, "gCa_max": 0.08},
        {"EK": -58.0, "EL": -48.0, "gL": 0.15, "Iext": 38.0, "tau_Ca": 900.0, "B_Ca": 1e-5, "gCa_max": 0.10},
        {"EK": -58.0, "EL": -45.0, "gL": 0.20, "Iext": 42.0, "tau_Ca": 1050.0, "B_Ca": 1e-5, "gCa_max": 0.10},
        {"EK": -55.0, "EL": -50.0, "gL": 0.12, "Iext": 38.0, "tau_Ca": 900.0, "B_Ca": 1e-5, "gCa_max": 0.08},
        {"EK": -55.0, "EL": -48.0, "gL": 0.15, "Iext": 42.0, "tau_Ca": 1050.0, "B_Ca": 1e-5, "gCa_max": 0.10},
        {"EK": -55.0, "EL": -45.0, "gL": 0.20, "Iext": 46.0, "tau_Ca": 1050.0, "B_Ca": 1e-5, "gCa_max": 0.12},
        {"EK": -52.0, "EL": -48.0, "gL": 0.15, "Iext": 46.0, "tau_Ca": 1050.0, "B_Ca": 1e-5, "gCa_max": 0.10},
        {"EK": -52.0, "EL": -45.0, "gL": 0.20, "Iext": 50.0, "tau_Ca": 1050.0, "B_Ca": 1e-5, "gCa_max": 0.12},
        {"EK": -50.0, "EL": -45.0, "gL": 0.20, "Iext": 50.0, "tau_Ca": 1200.0, "B_Ca": 1e-5, "gCa_max": 0.12},
        {"EK": -60.0, "EL": -48.0, "gL": 0.18, "Iext": 32.0, "tau_Ca": 750.0, "B_Ca": 1.5e-5, "gCa_max": 0.08},
        {"EK": -58.0, "EL": -46.0, "gL": 0.18, "Iext": 36.0, "tau_Ca": 900.0, "B_Ca": 1.5e-5, "gCa_max": 0.08},
        {"EK": -55.0, "EL": -46.0, "gL": 0.18, "Iext": 40.0, "tau_Ca": 900.0, "B_Ca": 1.5e-5, "gCa_max": 0.10},
        {"EK": -58.0, "EL": -48.0, "gL": 0.25, "Iext": 34.0, "tau_Ca": 900.0, "B_Ca": 1e-5, "gCa_max": 0.08},
        {"EK": -55.0, "EL": -45.0, "gL": 0.25, "Iext": 38.0, "tau_Ca": 1050.0, "B_Ca": 1e-5, "gCa_max": 0.10},
    ]


def _refine_cases(seed: dict) -> list[dict]:
    out = []
    for d_ek in (-2.0, 0.0, 2.0):
        for d_el in (-2.0, 0.0, 2.0):
            for d_i in (-4.0, 0.0, 4.0):
                c = {
                    "EK": max(-70.0, min(-45.0, seed["EK"] + d_ek)),
                    "EL": max(-55.0, min(-40.0, seed["EL"] + d_el)),
                    "gL": round(max(0.08, min(0.35, seed["gL"])), 3),
                    "Iext": max(16.0, min(60.0, seed["Iext"] + d_i)),
                    "tau_Ca": seed["tau_Ca"],
                    "B_Ca": seed["B_Ca"],
                    "gCa_max": seed["gCa_max"],
                }
                out.append(c)
    # keep deterministic unique order
    uniq = []
    seen = set()
    for c in out:
        key = tuple(c[k] for k in ("EK", "EL", "gL", "Iext", "tau_Ca", "B_Ca", "gCa_max"))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def main() -> int:
    out_dir = Path("_test_results")
    out_dir.mkdir(exist_ok=True)

    coarse_rows = []
    for i, case in enumerate(_coarse_cases(), start=1):
        row = _evaluate(case, t_sim=220.0, dt_eval=0.25)
        coarse_rows.append(row)
        print(
            f"coarse {i:02d}/20 score={row['score']:.2f} "
            f"sp={row['n_spikes']} first={row['first_half']} second={row['second_half']} vpk={row['v_peak_mV']:.1f}"
        )

    coarse_sorted = sorted(coarse_rows, key=lambda x: x["score"], reverse=True)
    seeds = coarse_sorted[:2]

    refined_rows = []
    for si, seed in enumerate(seeds, start=1):
        local_cases = _refine_cases(seed)
        print(f"\nrefine seed {si}: score={seed['score']:.2f}, cases={len(local_cases)}")
        for j, case in enumerate(local_cases, start=1):
            row = _evaluate(case, t_sim=260.0, dt_eval=0.25)
            refined_rows.append(row)
            if j % 9 == 0 or j == len(local_cases):
                print(
                    f"  refine {si}.{j:02d}/{len(local_cases)} last_score={row['score']:.2f} "
                    f"sp={row['n_spikes']} first={row['first_half']} second={row['second_half']}"
                )

    all_rows = coarse_rows + refined_rows
    all_sorted = sorted(all_rows, key=lambda x: x["score"], reverse=True)

    out = {
        "top10": all_sorted[:10],
        "coarse": coarse_rows,
        "refined": refined_rows,
    }
    out_file = out_dir / "hypoxia_deterministic_search.json"
    out_file.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nTOP 10")
    for i, row in enumerate(all_sorted[:10], start=1):
        print(
            f"{i:02d}. score={row['score']:.2f} sp={row['n_spikes']} "
            f"first={row['first_half']} second={row['second_half']} "
            f"EK={row['EK']} EL={row['EL']} gL={row['gL']} I={row['Iext']}"
        )
    print(f"\nSaved: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
