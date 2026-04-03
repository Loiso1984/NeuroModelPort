"""
Parallel sweep utility for pathology mode tuning.

Focused on test-time acceleration:
- multiprocessing across parameter cases,
- checkpointed jsonl output,
- resumable runs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver


def _spike_times_crossing(v: np.ndarray, t: np.ndarray, threshold: float = -20.0) -> np.ndarray:
    idx = np.where((v[:-1] < threshold) & (v[1:] >= threshold))[0] + 1
    if len(idx) == 0:
        return np.array([], dtype=float)
    st = t[idx]
    keep = [0]
    for i in range(1, len(st)):
        if st[i] - st[keep[-1]] >= 1.0:
            keep.append(i)
    return st[keep]


def _evaluate_hypoxia_case(case: dict) -> dict:
    cfg = FullModelConfig()
    apply_preset(cfg, "O: Hypoxia (v10 ATP-pump failure)")
    cfg.preset_modes.hypoxia_mode = "progressive"
    apply_preset(cfg, "O: Hypoxia (v10 ATP-pump failure)")

    cfg.channels.EK = case["EK"]
    cfg.channels.EL = case["EL"]
    cfg.channels.gL = case["gL"]
    cfg.stim.Iext = case["Iext"]
    cfg.calcium.tau_Ca = case["tau_Ca"]
    cfg.calcium.B_Ca = case["B_Ca"]
    cfg.channels.gCa_max = case["gCa_max"]
    cfg.stim.t_sim = case["t_sim"]
    cfg.stim.dt_eval = case["dt_eval"]
    cfg.stim.stim_type = "const"

    res = NeuronSolver(cfg).run_single()
    st = _spike_times_crossing(res.v_soma, res.t)
    mid = 0.5 * cfg.stim.t_sim
    first_half = int(np.sum(st < mid))
    second_half = int(np.sum(st >= mid))

    return {
        **case,
        "n_spikes": int(len(st)),
        "spikes_first_half": first_half,
        "spikes_second_half": second_half,
        "v_peak": float(np.max(res.v_soma)),
        "v_rest_tail": float(np.mean(res.v_soma[-80:])),
        "ca_peak_nM": float(np.max(res.ca_i[0, :]) * 1e6) if res.ca_i is not None else None,
    }


def _iter_hypoxia_cases(t_sim: float, dt_eval: float) -> Iterable[dict]:
    for ek, el, gl, iext, tau_ca, b_ca, gca in itertools.product(
        [-65.0, -60.0, -55.0, -50.0],
        [-52.0, -48.0, -45.0, -42.0],
        [0.10, 0.15, 0.20, 0.25, 0.30],
        [24.0, 30.0, 36.0, 42.0, 50.0],
        [700.0, 900.0, 1100.0],
        [1e-5, 2e-5],
        [0.06, 0.08, 0.10, 0.12],
    ):
        yield {
            "EK": ek,
            "EL": el,
            "gL": gl,
            "Iext": iext,
            "tau_Ca": tau_ca,
            "B_Ca": b_ca,
            "gCa_max": gca,
            "t_sim": t_sim,
            "dt_eval": dt_eval,
        }


def _score(record: dict) -> tuple:
    # Prefer clear early activity followed by attenuation.
    return (
        record["spikes_first_half"] - record["spikes_second_half"],
        -abs(record["spikes_first_half"] - 3),  # target modest early burst count
        -record["spikes_second_half"],
        -abs(record["v_peak"] - 30.0),
    )


def run_hypoxia_sweep(args: argparse.Namespace) -> int:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    if args.resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (
                    row["EK"], row["EL"], row["gL"], row["Iext"],
                    row["tau_Ca"], row["B_Ca"], row["gCa_max"],
                )
                seen.add(key)

    cases = list(_iter_hypoxia_cases(args.t_sim, args.dt_eval))
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    if seen:
        cases = [
            c for c in cases
            if (c["EK"], c["EL"], c["gL"], c["Iext"], c["tau_Ca"], c["B_Ca"], c["gCa_max"]) not in seen
        ]

    total = len(cases)
    if total == 0:
        print("No new cases to run.")
        return 0

    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 4) - 1)
    print(f"Running {total} cases with {workers} workers...")

    completed = 0
    top_records: list[dict] = []
    with out_path.open("a", encoding="utf-8") as out_f:
        executor = None
        try:
            executor = ProcessPoolExecutor(max_workers=workers)
        except PermissionError:
            print("ProcessPool unavailable (permission). Falling back to ThreadPoolExecutor.")
            executor = ThreadPoolExecutor(max_workers=workers)
        with executor as ex:
            futures = [ex.submit(_evaluate_hypoxia_case, c) for c in cases]
            for fut in as_completed(futures):
                completed += 1
                try:
                    row = fut.result(timeout=args.case_timeout_s)
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    top_records.append(row)
                except Exception as exc:
                    out_f.write(json.dumps({"error": str(exc)}, ensure_ascii=False) + "\n")
                if completed % args.progress_every == 0 or completed == total:
                    print(f"[{completed}/{total}]")

    top_records = [r for r in top_records if "n_spikes" in r]
    top_records.sort(key=_score, reverse=True)
    print("\nTop 10 candidates:")
    for i, row in enumerate(top_records[:10], start=1):
        print(
            f"{i:2d}. spikes={row['n_spikes']} first={row['spikes_first_half']} second={row['spikes_second_half']} "
            f"Vpeak={row['v_peak']:.1f} EK={row['EK']} EL={row['EL']} gL={row['gL']} I={row['Iext']}"
        )
    print(f"\nSaved: {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Parallel pathology mode sweep")
    parser.add_argument("--target", choices=["hypoxia"], default="hypoxia")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--t-sim", type=float, default=260.0)
    parser.add_argument("--dt-eval", type=float, default=0.2)
    parser.add_argument("--case-timeout-s", type=float, default=120.0)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path("_test_results") / "hypoxia_mode_sweep.jsonl"),
    )
    args = parser.parse_args()

    if args.target == "hypoxia":
        return run_hypoxia_sweep(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
