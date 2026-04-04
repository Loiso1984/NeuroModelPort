from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver


def _spike_count(v: np.ndarray) -> int:
    idx = np.where((v[:-1] < -20.0) & (v[1:] >= -20.0))[0] + 1
    if len(idx) == 0:
        return 0
    keep = [idx[0]]
    for i in idx[1:]:
        if i - keep[-1] > 3:
            keep.append(i)
    return len(keep)


def _run_case(preset: str, mode: str, t_sim: float, dt_eval: float) -> dict:
    cfg = FullModelConfig()
    apply_preset(cfg, preset)
    cfg.stim.t_sim = t_sim
    cfg.stim.dt_eval = dt_eval
    cfg.stim.jacobian_mode = mode

    t0 = time.perf_counter()
    res = NeuronSolver(cfg).run_single()
    elapsed = time.perf_counter() - t0
    return {
        "sec": elapsed,
        "n_spikes": _spike_count(res.v_soma),
        "v_peak_mV": float(np.max(res.v_soma)),
        "v_tail_mV": float(np.mean(res.v_soma[-80:])),
        "stable": bool(np.all(np.isfinite(res.v_soma))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark dense/sparse/analytic Jacobian modes")
    parser.add_argument("--preset", type=str, default="O: Hypoxia (v10 ATP-pump failure)")
    parser.add_argument("--t-sim", type=float, default=180.0)
    parser.add_argument("--dt-eval", type=float, default=0.3)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=str, default="_test_results/jacobian_mode_benchmark.json")
    args = parser.parse_args()

    modes = ["dense_fd", "sparse_fd", "analytic_sparse"]
    out = {
        "preset": args.preset,
        "t_sim": args.t_sim,
        "dt_eval": args.dt_eval,
        "repeats": args.repeats,
        "modes": {},
    }

    # Warmup run outside measurements.
    _run_case(args.preset, "dense_fd", min(30.0, args.t_sim), max(args.dt_eval, 0.4))

    for mode in modes:
        rows = []
        for _ in range(args.repeats):
            rows.append(_run_case(args.preset, mode, args.t_sim, args.dt_eval))
        secs = [r["sec"] for r in rows]
        out["modes"][mode] = {
            "runs": rows,
            "sec_mean": float(statistics.mean(secs)),
            "sec_median": float(statistics.median(secs)),
        }
        print(
            f"{mode:>15}: mean={out['modes'][mode]['sec_mean']:.3f}s "
            f"median={out['modes'][mode]['sec_median']:.3f}s"
        )

    dense_mean = out["modes"]["dense_fd"]["sec_mean"]
    out["speedup_vs_dense"] = {
        "sparse_fd": float(dense_mean / max(out["modes"]["sparse_fd"]["sec_mean"], 1e-12)),
        "analytic_sparse": float(dense_mean / max(out["modes"]["analytic_sparse"]["sec_mean"], 1e-12)),
    }
    print(
        f"speedup sparse_fd={out['speedup_vs_dense']['sparse_fd']:.2f}x, "
        f"analytic_sparse={out['speedup_vs_dense']['analytic_sparse']:.2f}x"
    )

    output = Path(args.output)
    output.parent.mkdir(exist_ok=True, parents=True)
    output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved benchmark: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
