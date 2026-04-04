"""
Deterministic coarse-to-fine search for hypoxia progressive mode.

Key principles:
1) No random sampling.
2) Two-stage evaluation (quick screen -> full run) for speed.
3) Physiological guard-constraints in scoring.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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


def _simulate_case(
    case: dict,
    t_sim: float,
    dt_eval: float,
    *,
    single_comp: bool = False,
    jacobian_mode: str = "sparse_fd",
) -> dict:
    cfg = FullModelConfig()
    cfg.preset_modes.hypoxia_mode = "progressive"
    apply_preset(cfg, "O: Hypoxia (v10 ATP-pump failure)")
    if single_comp:
        cfg.morphology.single_comp = True
        cfg.stim_location.location = "soma"
        cfg.dendritic_filter.enabled = False
    cfg.stim.stim_type = "const"
    cfg.stim.t_sim = t_sim
    cfg.stim.dt_eval = dt_eval
    cfg.stim.jacobian_mode = jacobian_mode
    cfg.channels.EK = case["EK"]
    cfg.channels.EL = case["EL"]
    cfg.channels.gL = case["gL"]
    cfg.stim.Iext = case["Iext"]
    cfg.calcium.tau_Ca = case["tau_Ca"]
    cfg.calcium.B_Ca = case["B_Ca"]
    cfg.channels.gCa_max = case["gCa_max"]

    res = NeuronSolver(cfg).run_single()
    st = _spike_times(res.v_soma, res.t)
    half = float(t_sim / 2.0)
    first = int(np.sum(st < half))
    second = int(np.sum(st >= half))
    v_peak = float(np.max(res.v_soma))
    v_tail = float(np.mean(res.v_soma[-80:]))

    ca_peak_nM = float(np.max(res.ca_i[0, :]) * 1e6) if res.ca_i is not None else None
    stable = bool(np.all(np.isfinite(res.v_soma)))
    return {
        **case,
        "n_spikes": int(len(st)),
        "first_half": first,
        "second_half": second,
        "v_peak_mV": v_peak,
        "v_tail_mV": v_tail,
        "ca_peak_nM": ca_peak_nM,
        "stable": stable,
    }


def _physiology_penalty(row: dict) -> float:
    penalty = 0.0
    if not row["stable"]:
        penalty += 200.0
    if row["v_peak_mV"] < 0.0 or row["v_peak_mV"] > 60.0:
        penalty += 60.0
    if row["v_tail_mV"] < -95.0 or row["v_tail_mV"] > -20.0:
        penalty += 40.0
    if row["ca_peak_nM"] is not None and row["ca_peak_nM"] > 5000.0:
        penalty += 50.0
    return penalty


def _progressive_score(row: dict) -> float:
    # Prefer "early spikes then attenuation" while staying physiological.
    first = row["first_half"]
    second = row["second_half"]
    v_peak = row["v_peak_mV"]
    base = (first - second) * 5.0 + first * 1.5 - second * 2.5 - abs(v_peak - 30.0) / 5.0
    return float(base - _physiology_penalty(row))


def _screen_case(
    case: dict,
    *,
    quick_t: float,
    quick_dt: float,
    quick_jacobian_mode: str,
) -> dict:
    quick = _simulate_case(
        case,
        quick_t,
        quick_dt,
        single_comp=True,
        jacobian_mode=quick_jacobian_mode,
    )
    quick["quick_score"] = _progressive_score(quick)
    quick["screen_reject"] = bool((not quick["stable"]) or quick["n_spikes"] == 0 or quick["v_peak_mV"] < 5.0)
    quick["score"] = quick["quick_score"] - (20.0 if quick["screen_reject"] else 0.0)
    return quick


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


def _case_key(case: dict) -> tuple:
    return (
        float(case["EK"]),
        float(case["EL"]),
        float(case["gL"]),
        float(case["Iext"]),
        float(case["tau_Ca"]),
        float(case["B_Ca"]),
        float(case["gCa_max"]),
    )


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _refine_cases(seed: dict) -> list[dict]:
    # Deterministic local neighborhood including all tunable dimensions.
    deltas = {
        "EK": (-2.0, 0.0, 2.0),
        "EL": (-2.0, 0.0, 2.0),
        "gL": (-0.03, 0.0, 0.03),
        "Iext": (-4.0, 0.0, 4.0),
        "tau_Ca": (-150.0, 0.0, 150.0),
        "gCa_max": (-0.02, 0.0, 0.02),
    }
    b_candidates = [1e-5, 1.5e-5, 2e-5]

    out = []
    # Axis-wise perturbations (faster and more interpretable than full cartesian).
    for p_name, p_deltas in deltas.items():
        for d in p_deltas:
            if d == 0.0:
                continue
            c = {
                "EK": seed["EK"],
                "EL": seed["EL"],
                "gL": seed["gL"],
                "Iext": seed["Iext"],
                "tau_Ca": seed["tau_Ca"],
                "B_Ca": seed["B_Ca"],
                "gCa_max": seed["gCa_max"],
            }
            c[p_name] = c[p_name] + d
            out.append(c)

    # Limited pairwise interactions that often matter for hypoxia.
    pairwise = [
        ("EK", -2.0, "gL", +0.03),
        ("EK", +2.0, "gL", -0.03),
        ("EL", -2.0, "Iext", +4.0),
        ("EL", +2.0, "Iext", -4.0),
        ("tau_Ca", +150.0, "gCa_max", +0.02),
        ("tau_Ca", -150.0, "gCa_max", -0.02),
    ]
    for p1, d1, p2, d2 in pairwise:
        c = {
            "EK": seed["EK"],
            "EL": seed["EL"],
            "gL": seed["gL"],
            "Iext": seed["Iext"],
            "tau_Ca": seed["tau_Ca"],
            "B_Ca": seed["B_Ca"],
            "gCa_max": seed["gCa_max"],
        }
        c[p1] = c[p1] + d1
        c[p2] = c[p2] + d2
        out.append(c)

    # Explicit B_Ca alternatives (often sensitive in Ca overload patterns).
    for b in b_candidates:
        if b == seed["B_Ca"]:
            continue
        c = {
            "EK": seed["EK"],
            "EL": seed["EL"],
            "gL": seed["gL"],
            "Iext": seed["Iext"],
            "tau_Ca": seed["tau_Ca"],
            "B_Ca": b,
            "gCa_max": seed["gCa_max"],
        }
        out.append(c)

    # Clamp to valid ranges.
    for c in out:
        c["EK"] = max(-70.0, min(-45.0, c["EK"]))
        c["EL"] = max(-55.0, min(-40.0, c["EL"]))
        c["gL"] = round(max(0.08, min(0.35, c["gL"])), 3)
        c["Iext"] = max(16.0, min(60.0, c["Iext"]))
        c["tau_Ca"] = max(500.0, min(1400.0, c["tau_Ca"]))
        c["gCa_max"] = round(max(0.04, min(0.14, c["gCa_max"])), 3)

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
    parser = argparse.ArgumentParser(description="Deterministic hypoxia search")
    parser.add_argument("--coarse-cases", type=int, default=12)
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument("--coarse-full-count", type=int, default=3)
    parser.add_argument("--max-refined-full", type=int, default=4)
    parser.add_argument("--quick-t", type=float, default=70.0)
    parser.add_argument("--quick-dt", type=float, default=0.40)
    parser.add_argument("--full-t", type=float, default=150.0)
    parser.add_argument("--full-dt", type=float, default=0.30)
    parser.add_argument("--final-validate-count", type=int, default=3)
    parser.add_argument("--final-t", type=float, default=260.0)
    parser.add_argument("--final-dt", type=float, default=0.25)
    parser.add_argument("--quick-jacobian-mode", type=str, default="sparse_fd")
    parser.add_argument("--full-jacobian-mode", type=str, default="sparse_fd")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_dir = Path("_test_results")
    out_dir.mkdir(exist_ok=True)
    checkpoint_file = out_dir / "hypoxia_deterministic_search.jsonl"

    # Warm up numba/JIT once so the first measured coarse case is not misleadingly slow.
    warm_case = _coarse_cases()[0]
    _simulate_case(
        warm_case,
        20.0,
        0.6,
        single_comp=True,
        jacobian_mode=args.quick_jacobian_mode,
    )

    coarse_rows = []
    coarse_screen_rows = []
    seen = set()
    if args.resume and checkpoint_file.exists():
        with checkpoint_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "EK" in row:
                    seen.add(_case_key(row))

    coarse_cases = _coarse_cases()[: max(1, args.coarse_cases)]
    t0 = time.time()
    for i, case in enumerate(coarse_cases, start=1):
        if _case_key(case) in seen:
            continue
        row = _screen_case(
            case,
            quick_t=args.quick_t,
            quick_dt=args.quick_dt,
            quick_jacobian_mode=args.quick_jacobian_mode,
        )
        coarse_screen_rows.append(row)
        _append_jsonl(checkpoint_file, {"phase": "coarse_screen", **row})
        print(
            f"coarse-screen {i:02d}/{len(coarse_cases)} score={row['score']:.2f} "
            f"sp={row['n_spikes']} first={row['first_half']} second={row['second_half']} "
            f"vpk={row['v_peak_mV']:.1f} reject={row.get('screen_reject', False)}"
        )

    coarse_screen_sorted = sorted(coarse_screen_rows, key=lambda x: x["score"], reverse=True)
    if len(coarse_screen_sorted) == 0:
        print("No new coarse rows computed (resume mode).")
        return 0

    coarse_full_candidates = [r for r in coarse_screen_sorted if not r.get("screen_reject", False)]
    coarse_full_candidates = coarse_full_candidates[: max(1, args.coarse_full_count)]
    for i, row in enumerate(coarse_full_candidates, start=1):
        case = {
            "EK": row["EK"],
            "EL": row["EL"],
            "gL": row["gL"],
            "Iext": row["Iext"],
            "tau_Ca": row["tau_Ca"],
            "B_Ca": row["B_Ca"],
            "gCa_max": row["gCa_max"],
        }
        full = _simulate_case(
            case,
            args.full_t,
            args.full_dt,
            single_comp=False,
            jacobian_mode=args.full_jacobian_mode,
        )
        full["quick_score"] = row["quick_score"]
        full["screen_reject"] = False
        full["score"] = _progressive_score(full)
        coarse_rows.append(full)
        _append_jsonl(checkpoint_file, {"phase": "coarse_full", **full})
        print(
            f"coarse-full {i:02d}/{len(coarse_full_candidates)} score={full['score']:.2f} "
            f"sp={full['n_spikes']} first={full['first_half']} second={full['second_half']}"
        )

    coarse_sorted = sorted(coarse_rows if coarse_rows else coarse_screen_rows, key=lambda x: x["score"], reverse=True)
    seeds = coarse_sorted[: max(1, args.seed_count)]

    refined_rows = []
    for si, seed in enumerate(seeds, start=1):
        local_cases = _refine_cases(seed)
        print(f"\nrefine seed {si}: score={seed['score']:.2f}, cases={len(local_cases)}")
        # quick-stage all local cases, run full only for top-N quick scores
        quick_local = []
        for case in local_cases:
            q = _screen_case(
                case,
                quick_t=args.quick_t,
                quick_dt=args.quick_dt,
                quick_jacobian_mode=args.quick_jacobian_mode,
            )
            quick_local.append((q["quick_score"], case, q))
        quick_local.sort(key=lambda x: x[0], reverse=True)
        top_local = quick_local[: max(1, args.max_refined_full)]

        for j, (_qscore, case, qrow) in enumerate(top_local, start=1):
            if _case_key(case) in seen:
                continue
            if qrow["n_spikes"] == 0 or not qrow["stable"]:
                row = dict(qrow)
                row["screen_reject"] = True
                row["score"] = qrow["quick_score"] - 20.0
            else:
                row = _simulate_case(
                    case,
                    args.full_t + 40.0,
                    args.full_dt,
                    single_comp=False,
                    jacobian_mode=args.full_jacobian_mode,
                )
                row["quick_score"] = qrow["quick_score"]
                row["screen_reject"] = False
                row["score"] = _progressive_score(row)
            refined_rows.append(row)
            _append_jsonl(checkpoint_file, {"phase": f"refine_seed_{si}", **row})
            if j % 4 == 0 or j == len(top_local):
                print(
                    f"  refine {si}.{j:02d}/{len(top_local)} last_score={row['score']:.2f} "
                    f"sp={row['n_spikes']} first={row['first_half']} second={row['second_half']} "
                    f"reject={row.get('screen_reject', False)}"
                )

    all_rows = coarse_rows + refined_rows
    all_sorted = sorted(all_rows, key=lambda x: x["score"], reverse=True)

    final_validation = []
    for row in all_sorted[: max(1, args.final_validate_count)]:
        case = {
            "EK": row["EK"],
            "EL": row["EL"],
            "gL": row["gL"],
            "Iext": row["Iext"],
            "tau_Ca": row["tau_Ca"],
            "B_Ca": row["B_Ca"],
            "gCa_max": row["gCa_max"],
        }
        vrow = _simulate_case(
            case,
            args.final_t,
            args.final_dt,
            single_comp=False,
            jacobian_mode=args.full_jacobian_mode,
        )
        vrow["score"] = _progressive_score(vrow)
        final_validation.append(vrow)
        _append_jsonl(checkpoint_file, {"phase": "final_validate", **vrow})

    out = {
        "config": {
            "coarse_cases": len(coarse_cases),
            "coarse_full_count": args.coarse_full_count,
            "seed_count": len(seeds),
            "max_refined_full": args.max_refined_full,
            "quick_t": args.quick_t,
            "quick_dt": args.quick_dt,
            "full_t": args.full_t,
            "full_dt": args.full_dt,
            "final_validate_count": args.final_validate_count,
            "final_t": args.final_t,
            "final_dt": args.final_dt,
            "quick_jacobian_mode": args.quick_jacobian_mode,
            "full_jacobian_mode": args.full_jacobian_mode,
        },
        "top10": all_sorted[:10],
        "coarse_screen": coarse_screen_rows,
        "coarse": coarse_rows,
        "refined": refined_rows,
        "final_validation": final_validation,
        "elapsed_sec": time.time() - t0,
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
    print(f"Checkpoint: {checkpoint_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
