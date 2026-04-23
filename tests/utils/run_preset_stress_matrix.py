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

from core.batch_validator import run_validation_batch
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


from tests.shared_utils import _spike_times


def _parse_csv_floats(raw: str) -> list[float]:
    out = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(float(token))
    return out


def _evaluate_case(
    cfg: FullModelConfig,
    preset: str,
    i_scale: float,
    t_celsius: float,
) -> dict:
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
        "status_code": "OK" if guard_ok else "UNSTABLE",
        "pruned": False,
    }


def _build_case_cfg(
    preset: str,
    i_scale: float,
    t_celsius: float,
    *,
    t_sim: float,
    dt_eval: float,
    single_comp_proxy: bool,
    prefer_native_hines: bool,
) -> FullModelConfig:
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
    if (not prefer_native_hines) and preset.startswith(("F:", "K:", "N:", "O:")):
        cfg.stim.jacobian_mode = "sparse_fd"
    return cfg


def _run_single_rows(cases: list[tuple[str, float, float]], args) -> list[dict]:
    def _run_matrix_case(case):
        preset, i_scale, t_c = case
        cfg = _build_case_cfg(
            preset,
            i_scale,
            t_c,
            t_sim=args.t_sim,
            dt_eval=args.dt_eval,
            single_comp_proxy=not args.multicomp,
            prefer_native_hines=False,
        )
        return _evaluate_case(cfg, preset, i_scale, t_c)

    rows = []
    total = len(cases)
    with ThreadPoolExecutor(max_workers=max(1, int(args.single_workers))) as pool:
        for done, row in enumerate(pool.map(_run_matrix_case, cases), start=1):
            rows.append(row)
            print(
                f"{done:03d}/{total} {row['preset'][:28]:28} "
                f"Ix={row['i_scale']:.2f} T={row['t_celsius']:.1f}C "
                f"guard={row['guard_ok']} spikes={row['n_spikes']}",
                flush=True,
            )
    return rows


def _run_batch_rows(cases: list[tuple[str, float, float]], args) -> tuple[list[dict], dict[str, int], float]:
    cfgs = []
    meta = []
    for preset, i_scale, t_c in cases:
        cfg = _build_case_cfg(
            preset,
            i_scale,
            t_c,
            t_sim=args.t_sim,
            dt_eval=args.dt_eval,
            single_comp_proxy=not args.multicomp,
            prefer_native_hines=bool(args.batch_prefer_native_hines),
        )
        cfgs.append(cfg)
        meta.append((preset, i_scale, t_c, float(cfg.stim.Iext)))

    batch = run_validation_batch(
        cfgs,
        workers=max(1, int(args.batch_workers)),
        compact_native=bool(args.compact_native),
        quick_prune_ms=float(args.quick_prune_ms) if args.quick_prune_ms > 0 else None,
        compact_dt_eval_ms=float(args.batch_dt_eval) if args.batch_dt_eval > 0 else None,
        parallel_backend=str(args.batch_backend),
    )

    rows = []
    total = len(meta)
    for idx, (preset, i_scale, t_c, stim_iext) in enumerate(meta):
        m = batch.rows[idx]
        row = {
            "preset": preset,
            "i_scale": float(i_scale),
            "t_celsius": float(t_c),
            "stim_Iext": stim_iext,
            "n_spikes": int(m["n_spikes"]),
            "freq_global_hz": float(m["freq_global_hz"]),
            "v_peak_mV": float(m["v_peak_mV"]),
            "v_min_mV": float(m["v_min_mV"]),
            "v_tail_mV": float(m["v_tail_mV"]),
            "ca_max_nM": None if m["ca_max_nM"] is None else float(m["ca_max_nM"]),
            "stable_finite": bool(m["stable_finite"]),
            "guard_ok": bool(m["guard_ok"]),
            "elapsed_sec": float(m["elapsed_sec"]),
            "status_code": str(m["status_code"]),
            "pruned": bool(m.get("pruned", False)),
            "dt_eval_ms": float(m.get("dt_eval_ms", cfgs[idx].stim.dt_eval)),
        }
        rows.append(row)
        print(
            f"{idx + 1:03d}/{total} {row['preset'][:28]:28} "
            f"Ix={row['i_scale']:.2f} T={row['t_celsius']:.1f}C "
            f"status={row['status_code']} spikes={row['n_spikes']}",
            flush=True,
        )
    return rows, batch.status_counts, batch.elapsed_sec


def _row_key(row: dict) -> tuple[str, float, float]:
    return (
        str(row["preset"]),
        float(row["i_scale"]),
        float(row["t_celsius"]),
    )


def _save_baseline_vs_optimized(single_rows: list[dict], batch_rows: list[dict], path: Path) -> dict:
    single_map = {_row_key(r): r for r in single_rows}
    batch_map = {_row_key(r): r for r in batch_rows}
    common_keys = [k for k in single_map.keys() if k in batch_map]

    def _arr(rows_map, metric):
        return np.asarray([float(rows_map[k][metric]) for k in common_keys], dtype=float)

    if not common_keys:
        payload = {"common_cases": 0}
    else:
        single_n = _arr(single_map, "n_spikes")
        batch_n = _arr(batch_map, "n_spikes")
        single_peak = _arr(single_map, "v_peak_mV")
        batch_peak = _arr(batch_map, "v_peak_mV")
        single_tail = _arr(single_map, "v_tail_mV")
        batch_tail = _arr(batch_map, "v_tail_mV")
        single_freq = _arr(single_map, "freq_global_hz")
        batch_freq = _arr(batch_map, "freq_global_hz")
        payload = {
            "common_cases": int(len(common_keys)),
            "mean_abs_delta": {
                "n_spikes": float(np.mean(np.abs(single_n - batch_n))),
                "v_peak_mV": float(np.mean(np.abs(single_peak - batch_peak))),
                "v_tail_mV": float(np.mean(np.abs(single_tail - batch_tail))),
                "freq_global_hz": float(np.mean(np.abs(single_freq - batch_freq))),
            },
            "max_abs_delta": {
                "n_spikes": float(np.max(np.abs(single_n - batch_n))),
                "v_peak_mV": float(np.max(np.abs(single_peak - batch_peak))),
                "v_tail_mV": float(np.max(np.abs(single_tail - batch_tail))),
                "freq_global_hz": float(np.max(np.abs(single_freq - batch_freq))),
            },
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic all-preset stress matrix")
    parser.add_argument("--i-scales", type=str, default="0.8,1.0,1.2")
    parser.add_argument("--temps", type=str, default="23,37")
    parser.add_argument("--t-sim", type=float, default=140.0)
    parser.add_argument("--dt-eval", type=float, default=0.3)
    parser.add_argument("--multicomp", action="store_true", help="Run full multi-comp matrix (much slower).")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, (os.cpu_count() or 2))),
        help="Legacy alias: applies to both single and batch workers unless explicit overrides are provided.",
    )
    parser.add_argument("--single-workers", type=int, default=None, help="Workers for baseline/single engine.")
    parser.add_argument("--batch-workers", type=int, default=None, help="Workers for batch engine.")
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True, help="Warm the native solver once before matrix execution.")
    parser.add_argument("--engine", choices=["single", "batch", "compare"], default="batch", help="Execution engine for matrix cases.")
    parser.add_argument("--compact-native", action=argparse.BooleanOptionalAction, default=True, help="Skip heavy current/ATP post-processing for native_hines in batch mode.")
    parser.add_argument("--quick-prune-ms", type=float, default=180.0, help="Quick prune horizon for SILENT/D_BLOCK screening in batch mode (<=0 disables).")
    parser.add_argument("--batch-dt-eval", type=float, default=1.0, help="Optional coarser output sampling for batch compact mode (ms, <=0 disables).")
    parser.add_argument("--batch-prefer-native-hines", action=argparse.BooleanOptionalAction, default=True, help="Prefer native_hines for all presets in batch mode.")
    parser.add_argument("--batch-backend", choices=["thread", "process", "serial"], default="thread", help="Parallel backend for batch engine.")
    parser.add_argument("--baseline-artifact", type=str, default="_test_results/baseline_vs_optimized_preset_stress_matrix.json")
    parser.add_argument("--output", type=str, default="_test_results/preset_stress_matrix.json")
    args = parser.parse_args()
    if args.single_workers is None:
        args.single_workers = int(args.workers)
    if args.batch_workers is None:
        # Aggressive default for batch compute on CPU, bounded to avoid oversubscription.
        args.batch_workers = max(1, min(12, (os.cpu_count() or 2)))
    args.single_workers = max(1, int(args.single_workers))
    args.batch_workers = max(1, int(args.batch_workers))

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

    batch_status_counts = {}
    baseline_vs_optimized = None
    if args.engine == "single":
        rows = _run_single_rows(cases, args)
    elif args.engine == "batch":
        rows, batch_status_counts, _batch_elapsed = _run_batch_rows(cases, args)
    else:
        t_single = time.time()
        single_rows = _run_single_rows(cases, args)
        single_elapsed = float(time.time() - t_single)
        t_batch = time.time()
        batch_rows, batch_status_counts, _ = _run_batch_rows(cases, args)
        batch_elapsed = float(time.time() - t_batch)
        rows = batch_rows
        artifact = Path(args.baseline_artifact)
        baseline_vs_optimized = _save_baseline_vs_optimized(single_rows, batch_rows, artifact)
        baseline_vs_optimized["single_elapsed_sec"] = single_elapsed
        baseline_vs_optimized["batch_elapsed_sec"] = batch_elapsed
        baseline_vs_optimized["speedup_x"] = float(single_elapsed / max(batch_elapsed, 1e-9))
        artifact.write_text(json.dumps(baseline_vs_optimized, indent=2, ensure_ascii=False), encoding="utf-8")

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
            "single_workers": int(args.single_workers),
            "batch_workers": int(args.batch_workers),
            "warmup": warmup,
            "engine": str(args.engine),
            "compact_native": bool(args.compact_native),
            "quick_prune_ms": float(args.quick_prune_ms),
            "batch_dt_eval": float(args.batch_dt_eval),
            "batch_prefer_native_hines": bool(args.batch_prefer_native_hines),
            "batch_backend": str(args.batch_backend),
        },
        "summary": {
            "total_cases": len(rows),
            "guard_ok_cases": n_ok,
            "guard_ok_ratio": float(n_ok / max(1, len(rows))),
            "elapsed_sec": float(time.time() - t0),
            "batch_status_counts": batch_status_counts,
            "baseline_vs_optimized": baseline_vs_optimized,
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
