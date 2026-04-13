"""
Extended deterministic report for C/D/E presets.

Sweeps:
- drive scaling
- temperature

Reports:
- spike count
- global/active frequency
- voltage guards
- simple per-preset physiology flags
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
from tests.shared_utils import _spike_times


def _parse_csv_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _run_case(
    preset: str,
    *,
    i_scale: float,
    t_celsius: float,
    t_sim: float,
    dt_eval: float,
) -> dict:
    cfg = FullModelConfig()
    apply_preset(cfg, preset)
    cfg.stim.Iext *= float(i_scale)
    cfg.env.T_celsius = float(t_celsius)
    cfg.stim.t_sim = float(t_sim)
    cfg.stim.dt_eval = float(dt_eval)
    cfg.stim.jacobian_mode = "sparse_fd"

    res = NeuronSolver(cfg).run_single()
    st = _spike_times(res.v_soma, res.t)
    dur = float(res.t[-1] - res.t[0]) if len(res.t) > 1 else 0.0
    fg = float(1000.0 * len(st) / dur) if dur > 0 else 0.0
    fa = float(1000.0 * (len(st) - 1) / max(st[-1] - st[0], 1e-9)) if len(st) > 1 else 0.0

    return {
        "preset": preset,
        "i_scale": float(i_scale),
        "t_celsius": float(t_celsius),
        "stim_Iext": float(cfg.stim.Iext),
        "n_spikes": int(len(st)),
        "freq_global_hz": fg,
        "freq_active_hz": fa,
        "v_peak_mV": float(np.max(res.v_soma)),
        "v_tail_mV": float(np.mean(res.v_soma[-100:])),
        "v_min_mV": float(np.min(res.v_soma)),
        "stable": bool(np.all(np.isfinite(res.v_soma))),
    }


def _in_range(x: float, lo: float, hi: float) -> bool:
    return bool(lo <= x <= hi)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extended C/D/E deterministic report")
    parser.add_argument("--i-scales", type=str, default="0.6,0.8,1.0,1.2,1.4")
    parser.add_argument("--temps", type=str, default="23,30,37")
    parser.add_argument("--t-sim", type=float, default=220.0)
    parser.add_argument("--dt-eval", type=float, default=0.2)
    parser.add_argument("--output", type=str, default="_test_results/cde_extended_report.json")
    args = parser.parse_args()

    i_scales = _parse_csv_floats(args.i_scales)
    temps = _parse_csv_floats(args.temps)
    presets = [
        "C: FS Interneuron (Wang-Buzsaki)",
        "D: alpha-Motoneuron (Powers 2001)",
        "E: Cerebellar Purkinje (De Schutter)",
    ]

    rows = []
    anomalies = []
    total = len(presets) * len(i_scales) * len(temps)
    done = 0
    for preset in presets:
        for i_scale in i_scales:
            for t_c in temps:
                row = _run_case(
                    preset,
                    i_scale=i_scale,
                    t_celsius=t_c,
                    t_sim=args.t_sim,
                    dt_eval=args.dt_eval,
                )
                rows.append(row)
                done += 1
                print(
                    f"{done:03d}/{total} {preset[:28]:28} I={i_scale:.2f} T={t_c:.1f} "
                    f"sp={row['n_spikes']} fg={row['freq_global_hz']:.1f}Hz",
                    flush=True,
                )

                # Hard guards
                if not row["stable"] or not (-140.0 < row["v_peak_mV"] < 80.0 and -140.0 < row["v_min_mV"] < 80.0):
                    anomalies.append({"type": "numeric_or_voltage_guard", **row})
                    continue

                # Baseline physiology windows near i_scale=1.0 and 37C.
                if abs(i_scale - 1.0) < 1e-9 and abs(t_c - 37.0) < 1e-9:
                    if preset.startswith("C:") and not _in_range(row["freq_active_hz"], 80.0, 260.0):
                        anomalies.append({"type": "c_baseline_freq_out", **row})
                    if preset.startswith("D:") and not _in_range(row["freq_active_hz"], 50.0, 150.0):
                        anomalies.append({"type": "d_baseline_freq_out", **row})
                    if preset.startswith("E:") and not _in_range(row["freq_active_hz"], 70.0, 180.0):
                        anomalies.append({"type": "e_baseline_freq_out", **row})

    by_preset = {}
    for p in presets:
        p_rows = [r for r in rows if r["preset"] == p]
        by_preset[p] = {
            "cases": len(p_rows),
            "spike_cases": int(sum(1 for r in p_rows if r["n_spikes"] >= 1)),
            "max_freq_active_hz": float(max(r["freq_active_hz"] for r in p_rows)),
            "min_freq_active_hz": float(min(r["freq_active_hz"] for r in p_rows)),
            "max_peak_mV": float(max(r["v_peak_mV"] for r in p_rows)),
            "min_peak_mV": float(min(r["v_peak_mV"] for r in p_rows)),
        }

    out = {
        "config": {
            "i_scales": i_scales,
            "temps": temps,
            "t_sim": float(args.t_sim),
            "dt_eval": float(args.dt_eval),
        },
        "summary": {
            "cases": len(rows),
            "anomalies": len(anomalies),
            "pass_ratio": float((len(rows) - len(anomalies)) / max(1, len(rows))),
        },
        "by_preset": by_preset,
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

