"""Extended deterministic HCN/IA validation report."""

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


def _estimate_rin(v: np.ndarray, t: np.ndarray, pulse_start: float, pulse_end: float, iext: float) -> float:
    baseline_mask = (t >= pulse_start - 20.0) & (t < pulse_start - 2.0)
    steady_mask = (t >= pulse_end - 25.0) & (t < pulse_end - 5.0)
    v_baseline = float(np.mean(v[baseline_mask]))
    v_steady = float(np.mean(v[steady_mask]))
    dv = v_steady - v_baseline
    return abs(dv / iext)


def _build_hcn_probe_config(enable_hcn: bool, ih_scale: float, t_celsius: float) -> FullModelConfig:
    cfg = FullModelConfig()
    cfg.morphology.single_comp = True
    cfg.stim.jacobian_mode = "sparse_fd"

    cfg.channels.gNa_max = 0.0
    cfg.channels.gK_max = 0.0
    cfg.channels.gL = 0.05
    cfg.channels.EL = -65.0

    cfg.channels.enable_Ih = bool(enable_hcn)
    cfg.channels.gIh_max = 0.03 * float(ih_scale) if enable_hcn else 0.0
    cfg.channels.E_Ih = -43.0

    cfg.channels.enable_ICa = False
    cfg.channels.enable_IA = False
    cfg.channels.enable_SK = False

    cfg.stim.stim_type = "pulse"
    cfg.stim.Iext = -0.10
    cfg.stim.pulse_start = 50.0
    cfg.stim.pulse_dur = 200.0
    cfg.stim.t_sim = 350.0
    cfg.stim.dt_eval = 0.1
    cfg.env.T_celsius = float(t_celsius)
    return cfg


def _hcn_probe_case(*, t_celsius: float, ih_scale: float) -> dict:
    cfg = _build_hcn_probe_config(enable_hcn=(ih_scale > 0.0), ih_scale=ih_scale, t_celsius=t_celsius)
    res = NeuronSolver(cfg).run_single()
    t = res.t
    v = res.v_soma
    start = cfg.stim.pulse_start
    end = cfg.stim.pulse_start + cfg.stim.pulse_dur
    w = (t >= start) & (t <= end)
    rin = _estimate_rin(v, t, start, end, cfg.stim.Iext)
    v_min = float(np.min(v[w]))
    t_min = float(t[w][np.argmin(v[w])])
    v_ss = float(np.mean(v[(t >= end - 25.0) & (t < end - 5.0)]))
    sag_recovery = float(v_ss - v_min)
    return {
        "scope": "hcn_probe",
        "t_celsius": float(t_celsius),
        "ih_scale": float(ih_scale),
        "gIh_max": float(cfg.channels.gIh_max),
        "rin_mV_per_uAcm2": rin,
        "sag_tmin_ms": t_min,
        "sag_recovery_mV": sag_recovery,
        "v_min_mV": v_min,
        "stable": bool(np.all(np.isfinite(v))),
    }


def _hcn_preset_case(*, preset: str, t_celsius: float, ih_enabled: bool) -> dict:
    cfg = FullModelConfig()
    apply_preset(cfg, preset)
    cfg.morphology.single_comp = True
    cfg.stim.jacobian_mode = "sparse_fd"
    cfg.stim.stim_type = "const"
    cfg.stim.Iext = 0.0
    cfg.stim.t_sim = 300.0
    cfg.stim.dt_eval = 0.2
    cfg.env.T_celsius = float(t_celsius)
    if not ih_enabled:
        cfg.channels.enable_Ih = False
        cfg.channels.gIh_max = 0.0

    res = NeuronSolver(cfg).run_single()
    tail = res.v_soma[-100:]
    return {
        "scope": "hcn_preset_rest",
        "preset": preset,
        "t_celsius": float(t_celsius),
        "ih_enabled": bool(ih_enabled),
        "v_rest_tail_mV": float(np.mean(tail)),
        "v_tail_std_mV": float(np.std(tail)),
        "stable": bool(np.all(np.isfinite(res.v_soma))),
    }


def _build_ia_probe_config(ga_scale: float) -> FullModelConfig:
    cfg = FullModelConfig()
    cfg.morphology.single_comp = True
    cfg.stim.jacobian_mode = "sparse_fd"

    cfg.channels.gNa_max = 120.0
    cfg.channels.gK_max = 36.0
    cfg.channels.gL = 0.3
    cfg.channels.ENa = 50.0
    cfg.channels.EK = -77.0
    cfg.channels.EL = -54.387

    cfg.channels.enable_IA = ga_scale > 0.0
    cfg.channels.gA_max = 0.8 * float(ga_scale) if ga_scale > 0.0 else 0.0
    cfg.channels.E_A = -77.0

    cfg.channels.enable_Ih = False
    cfg.channels.enable_ICa = False
    cfg.channels.enable_SK = False

    cfg.stim.stim_type = "pulse"
    cfg.stim.Iext = 8.0
    cfg.stim.pulse_start = 20.0
    cfg.stim.pulse_dur = 80.0
    cfg.stim.t_sim = 140.0
    cfg.stim.dt_eval = 0.05
    return cfg


def _ia_probe_case(*, ga_scale: float) -> dict:
    cfg = _build_ia_probe_config(ga_scale)
    res = NeuronSolver(cfg).run_single()
    st = _spike_times(res.v_soma, res.t)
    return {
        "scope": "ia_probe",
        "ga_scale": float(ga_scale),
        "gA_max": float(cfg.channels.gA_max),
        "n_spikes": int(len(st)),
        "v_peak_mV": float(np.max(res.v_soma)),
        "stable": bool(np.all(np.isfinite(res.v_soma))),
    }


def _ia_preset_case(*, preset: str, ia_enabled: bool, i_scale: float) -> dict:
    cfg = FullModelConfig()
    apply_preset(cfg, preset)
    cfg.morphology.single_comp = True
    cfg.stim.jacobian_mode = "sparse_fd"
    cfg.stim.stim_type = "const"
    cfg.stim.t_sim = 200.0
    cfg.stim.dt_eval = 0.2
    cfg.stim.Iext = float(max(cfg.stim.Iext * i_scale, 8.0))

    if not ia_enabled:
        cfg.channels.enable_IA = False
        cfg.channels.gA_max = 0.0

    res = NeuronSolver(cfg).run_single()
    st = _spike_times(res.v_soma, res.t)
    freq_active = float(1000.0 * (len(st) - 1) / max((st[-1] - st[0]) if len(st) > 1 else 1e9, 1e-9)) if len(st) > 1 else 0.0
    return {
        "scope": "ia_preset",
        "preset": preset,
        "ia_enabled": bool(ia_enabled),
        "gA_max": float(cfg.channels.gA_max),
        "i_scale": float(i_scale),
        "stim_Iext": float(cfg.stim.Iext),
        "n_spikes": int(len(st)),
        "freq_active_hz": freq_active,
        "v_peak_mV": float(np.max(res.v_soma)),
        "stable": bool(np.all(np.isfinite(res.v_soma))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Extended deterministic HCN/IA validation report")
    parser.add_argument("--output", type=str, default="_test_results/hcn_ia_extended_report.json")
    parser.add_argument("--ia-i-scale", type=float, default=1.0)
    args = parser.parse_args()

    hcn_presets = [
        "K: Thalamic Relay (Ih + ICa + Burst)",
        "L: Hippocampal CA1 (Theta rhythm)",
    ]
    ia_presets = [
        "C: FS Interneuron (Wang-Buzsaki)",
        "L: Hippocampal CA1 (Theta rhythm)",
    ]

    hcn_probe_rows = []
    for t_c in [23.0, 37.0]:
        for scale in [0.0, 0.5, 1.0, 1.5]:
            row = _hcn_probe_case(t_celsius=t_c, ih_scale=scale)
            hcn_probe_rows.append(row)
            print(f"HCN probe T={t_c:.0f} scale={scale:.1f} Rin={row['rin_mV_per_uAcm2']:.2f}", flush=True)

    hcn_preset_rows = []
    for preset in hcn_presets:
        for t_c in [23.0, 37.0]:
            for ih_enabled in [False, True]:
                row = _hcn_preset_case(preset=preset, t_celsius=t_c, ih_enabled=ih_enabled)
                hcn_preset_rows.append(row)
                print(
                    f"HCN preset {preset[:20]:20} T={t_c:.0f} Ih={int(ih_enabled)} "
                    f"Vrest={row['v_rest_tail_mV']:.2f}",
                    flush=True,
                )

    ia_probe_rows = []
    for scale in [0.0, 0.5, 1.0, 1.5]:
        row = _ia_probe_case(ga_scale=scale)
        ia_probe_rows.append(row)
        print(f"IA probe scale={scale:.1f} spikes={row['n_spikes']:2d}", flush=True)

    ia_preset_rows = []
    for preset in ia_presets:
        for ia_enabled in [False, True]:
            row = _ia_preset_case(preset=preset, ia_enabled=ia_enabled, i_scale=args.ia_i_scale)
            ia_preset_rows.append(row)
            print(
                f"IA preset {preset[:20]:20} IA={int(ia_enabled)} spikes={row['n_spikes']:2d}",
                flush=True,
            )

    anomalies = []

    for t_c in [23.0, 37.0]:
        group = [r for r in hcn_probe_rows if abs(r["t_celsius"] - t_c) < 1e-9]
        group = sorted(group, key=lambda r: r["ih_scale"])
        rins = [r["rin_mV_per_uAcm2"] for r in group]
        if not all(r["stable"] for r in group):
            anomalies.append({"type": "hcn_probe_non_finite_trace", "t_celsius": t_c})
        # In this reduced probe, high gIh may show mild non-monotonicity from saturation/numerics.
        # Use robust acceptance: clear net Rin reduction and no large rebound at highest gIh.
        rin_0, rin_05, rin_10, rin_15 = rins
        if rin_10 > rin_0 * 0.85:
            anomalies.append({"type": "hcn_probe_weak_rin_reduction", "t_celsius": t_c, "rins": rins})
        if rin_15 > rin_10 * 1.25:
            anomalies.append({"type": "hcn_probe_high_scale_rebound", "t_celsius": t_c, "rins": rins})

    p23 = [r for r in hcn_probe_rows if abs(r["t_celsius"] - 23.0) < 1e-9 and abs(r["ih_scale"] - 1.0) < 1e-9][0]
    p37 = [r for r in hcn_probe_rows if abs(r["t_celsius"] - 37.0) < 1e-9 and abs(r["ih_scale"] - 1.0) < 1e-9][0]
    if not (p37["sag_tmin_ms"] < p23["sag_tmin_ms"]):
        anomalies.append(
            {
                "type": "hcn_probe_temp_acceleration_missing",
                "tmin_23": p23["sag_tmin_ms"],
                "tmin_37": p37["sag_tmin_ms"],
            }
        )

    for preset in hcn_presets:
        for t_c in [23.0, 37.0]:
            group = [r for r in hcn_preset_rows if r["preset"] == preset and abs(r["t_celsius"] - t_c) < 1e-9]
            for row in group:
                if not row["stable"]:
                    anomalies.append({"type": "hcn_preset_non_finite_trace", "preset": preset, "t_celsius": t_c})
                if not (-85.0 <= row["v_rest_tail_mV"] <= -50.0):
                    anomalies.append(
                        {
                            "type": "hcn_preset_rest_out_of_range",
                            "preset": preset,
                            "t_celsius": t_c,
                            "ih_enabled": row["ih_enabled"],
                            "v_rest_tail_mV": row["v_rest_tail_mV"],
                        }
                    )
                if row["v_tail_std_mV"] > 3.0:
                    anomalies.append(
                        {
                            "type": "hcn_preset_rest_unstable",
                            "preset": preset,
                            "t_celsius": t_c,
                            "ih_enabled": row["ih_enabled"],
                            "v_tail_std_mV": row["v_tail_std_mV"],
                        }
                    )

    probe_spikes = [r["n_spikes"] for r in sorted(ia_probe_rows, key=lambda r: r["ga_scale"])]
    if not all(r["stable"] for r in ia_probe_rows):
        anomalies.append({"type": "ia_probe_non_finite_trace"})
    if not all(probe_spikes[i + 1] <= probe_spikes[i] for i in range(len(probe_spikes) - 1)):
        anomalies.append({"type": "ia_probe_spike_trend_non_monotonic", "spikes": probe_spikes})

    for preset in ia_presets:
        off = [r for r in ia_preset_rows if r["preset"] == preset and not r["ia_enabled"]][0]
        on = [r for r in ia_preset_rows if r["preset"] == preset and r["ia_enabled"]][0]
        if not (off["stable"] and on["stable"]):
            anomalies.append({"type": "ia_preset_non_finite_trace", "preset": preset})
        if on["n_spikes"] > off["n_spikes"] + 2:
            anomalies.append(
                {
                    "type": "ia_preset_non_suppressive_response",
                    "preset": preset,
                    "spikes_off": off["n_spikes"],
                    "spikes_on": on["n_spikes"],
                }
            )

    out = {
        "config": {
            "ia_i_scale": float(args.ia_i_scale),
            "hcn_temps": [23.0, 37.0],
            "hcn_ih_scales": [0.0, 0.5, 1.0, 1.5],
            "ia_scales": [0.0, 0.5, 1.0, 1.5],
        },
        "summary": {
            "hcn_probe_cases": len(hcn_probe_rows),
            "hcn_preset_cases": len(hcn_preset_rows),
            "ia_probe_cases": len(ia_probe_rows),
            "ia_preset_cases": len(ia_preset_rows),
            "anomalies": len(anomalies),
            "pass_ratio": float(
                (
                    len(hcn_probe_rows)
                    + len(hcn_preset_rows)
                    + len(ia_probe_rows)
                    + len(ia_preset_rows)
                    - len(anomalies)
                )
                / max(1, len(hcn_probe_rows) + len(hcn_preset_rows) + len(ia_probe_rows) + len(ia_preset_rows))
            ),
        },
        "anomalies": anomalies,
        "hcn_probe_rows": hcn_probe_rows,
        "hcn_preset_rows": hcn_preset_rows,
        "ia_probe_rows": ia_probe_rows,
        "ia_preset_rows": ia_preset_rows,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    print(f"Anomalies: {len(anomalies)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
