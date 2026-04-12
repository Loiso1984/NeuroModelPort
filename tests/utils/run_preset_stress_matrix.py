"""
Deterministic all-preset stress matrix runner.

Runs a grid over:
- preset name
- Iext scale
- environment temperature

and records stability/guard metrics for quick physiology-first screening.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset, get_preset_names
from core.solver import NeuronSolver


def _warm_native_hines() -> dict:
    """Compile the hot native path once before matrix execution."""
    t0 = time.time()
    try:
        cfg = FullModelConfig()
        apply_preset(cfg, "A: Squid Giant Axon (HH 1952)")
        cfg.stim.t_sim = 20.0
        cfg.stim.dt_eval = 0.2
        cfg.stim.jacobian_mode = "native_hines"
        NeuronSolver(cfg).run_single()
        return {"status": "pass", "elapsed_sec": time.time() - t0}
    except Exception as exc:
        return {"status": "warn", "elapsed_sec": time.time() - t0, "message": f"{type(exc).__name__}: {exc}"}


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


def _parse_csv_floats(raw: str) -> list[float]:
    out = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(float(token))
    return out


def _evaluate_case(
    preset: str,
    i_scale: float,
    t_celsius: float,
    *,
    t_sim: float,
    dt_eval: float,
    single_comp_proxy: bool,
) -> dict:
    cfg = FullModelConfig()
    apply_preset(cfg, preset)
    cfg.stim.t_sim = t_sim
    cfg.stim.dt_eval = dt_eval
    cfg.stim.Iext = float(cfg.stim.Iext * i_scale)
    cfg.env.T_celsius = float(t_celsius)
    if single_comp_proxy:
        cfg.morphology.single_comp = True
        cfg.stim_location.location = "soma"
        cfg.dendritic_filter.enabled = False
    if preset.startswith(("F:", "K:", "N:", "O:")):
        cfg.stim.jacobian_mode = "sparse_fd"

    t0 = time.time()
    res = NeuronSolver(cfg).run_single()
    elapsed = time.time() - t0

    st = _spike_times(res.v_soma, res.t)
    duration_ms = float(res.t[-1] - res.t[0]) if len(res.t) > 1 else 0.0
    freq_global = float(1000.0 * len(st) / duration_ms) if duration_ms > 0 else 0.0

    v_peak = float(np.max(res.v_soma))
    v_min = float(np.min(res.v_soma))
    v_tail = float(np.mean(res.v_soma[-80:]))
    stable = bool(np.all(np.isfinite(res.v_soma)))

    ca_max_nM = None
    ca_ok = True
    if res.ca_i is not None:
        ca = res.ca_i[0, :]
        ca_max_nM = float(np.max(ca) * 1e6)
        ca_ok = bool(np.all(ca >= 0.0) and ca_max_nM <= 10000.0)

    voltage_ok = bool(-140.0 < v_min < 80.0 and -140.0 < v_peak < 80.0 and -120.0 < v_tail < 20.0)
    guard_ok = bool(stable and voltage_ok and ca_ok)

    return {
        "preset": preset,
        "i_scale": float(i_scale),
        "t_celsius": float(t_celsius),
        "stim_Iext": float(cfg.stim.Iext),
        "n_spikes": int(len(st)),
        "freq_global_hz": freq_global,
        "v_peak_mV": v_peak,
        "v_min_mV": v_min,
        "v_tail_mV": v_tail,
        "ca_max_nM": ca_max_nM,
        "stable_finite": stable,
        "guard_ok": guard_ok,
        "elapsed_sec": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic all-preset stress matrix")
    parser.add_argument("--i-scales", type=str, default="0.8,1.0,1.2")
    parser.add_argument("--temps", type=str, default="23,37")
    parser.add_argument("--t-sim", type=float, default=140.0)
    parser.add_argument("--dt-eval", type=float, default=0.3)
    parser.add_argument("--multicomp", action="store_true", help="Run full multi-comp matrix (much slower).")
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2))), help="Parallel worker count for independent matrix cases.")
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True, help="Warm the native solver once before matrix execution.")
    parser.add_argument("--output", type=str, default="_test_results/preset_stress_matrix.json")
    args = parser.parse_args()

    i_scales = _parse_csv_floats(args.i_scales)
    temps = _parse_csv_floats(args.temps)
    presets = get_preset_names()

    warmup = _warm_native_hines() if args.warmup else {"status": "skipped", "elapsed_sec": 0.0}
    cases = []
    t0 = time.time()
    total = len(presets) * len(i_scales) * len(temps)
    for preset in presets:
        for i_scale in i_scales:
            for t_c in temps:
                cases.append((preset, i_scale, t_c))

    def _run_matrix_case(case):
        preset, i_scale, t_c = case
        return _evaluate_case(
            preset,
            i_scale,
            t_c,
            t_sim=args.t_sim,
            dt_eval=args.dt_eval,
            single_comp_proxy=not args.multicomp,
        )

    rows = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        for done, row in enumerate(pool.map(_run_matrix_case, cases), start=1):
            rows.append(row)
            print(
                f"{done:03d}/{total} {row['preset'][:28]:28} "
                f"Ix={row['i_scale']:.2f} T={row['t_celsius']:.1f}C guard={row['guard_ok']} spikes={row['n_spikes']}",
                flush=True,
            )

    n_ok = int(sum(1 for r in rows if r["guard_ok"]))
    by_preset = {}
    for p in presets:
        p_rows = [r for r in rows if r["preset"] == p]
        by_preset[p] = {
            "cases": len(p_rows),
            "guard_ok_cases": int(sum(1 for r in p_rows if r["guard_ok"])),
            "max_peak_mV": float(max(r["v_peak_mV"] for r in p_rows)),
            "min_peak_mV": float(min(r["v_peak_mV"] for r in p_rows)),
        }

    out = {
        "config": {
            "i_scales": i_scales,
            "temps": temps,
            "t_sim": float(args.t_sim),
            "dt_eval": float(args.dt_eval),
            "multicomp": bool(args.multicomp),
            "workers": int(args.workers),
            "warmup": warmup,
        },
        "summary": {
            "total_cases": len(rows),
            "guard_ok_cases": n_ok,
            "guard_ok_ratio": float(n_ok / max(1, len(rows))),
            "elapsed_sec": float(time.time() - t0),
        },
        "by_preset": by_preset,
        "rows": rows,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    print(
        f"Guard pass: {n_ok}/{len(rows)} "
        f"({100.0 * n_ok / max(1, len(rows)):.1f}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
