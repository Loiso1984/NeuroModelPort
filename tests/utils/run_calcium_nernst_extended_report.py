"""
Extended deterministic calcium/Nernst validation report.

Focus presets: K, L, M, N, O (+ N/O mode variants).
Checks:
- Ca_i range and non-negativity
- E_Ca range and temperature trend
- ICa inward influx proxy
- explicit export of B_Ca and gCa_max for audit
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
from core.rhs import nernst_ca_ion
from core.solver import NeuronSolver
from tests.shared_utils import _spike_times


def _build_cfg(preset_key: str, *, t_celsius: float, i_scale: float) -> FullModelConfig:
    cfg = FullModelConfig()

    if preset_key == "N_progressive":
        cfg.preset_modes.alzheimer_mode = "progressive"
        apply_preset(cfg, "N: Alzheimer's (v10 Calcium Toxicity)")
    elif preset_key == "N_terminal":
        cfg.preset_modes.alzheimer_mode = "terminal"
        apply_preset(cfg, "N: Alzheimer's (v10 Calcium Toxicity)")
    elif preset_key == "O_progressive":
        cfg.preset_modes.hypoxia_mode = "progressive"
        apply_preset(cfg, "O: Hypoxia (v10 ATP-pump failure)")
    elif preset_key == "O_terminal":
        cfg.preset_modes.hypoxia_mode = "terminal"
        apply_preset(cfg, "O: Hypoxia (v10 ATP-pump failure)")
    else:
        apply_preset(cfg, preset_key)

    cfg.env.T_celsius = float(t_celsius)
    cfg.stim.Iext *= float(i_scale)
    cfg.stim.t_sim = 220.0
    cfg.stim.dt_eval = 0.2
    cfg.stim.jacobian_mode = "sparse_fd"
    return cfg


def _run_case(preset_key: str, *, t_celsius: float, i_scale: float) -> dict:
    cfg = _build_cfg(preset_key, t_celsius=t_celsius, i_scale=i_scale)
    res = NeuronSolver(cfg).run_single()
    spikes = _spike_times(res.v_soma, res.t)

    row = {
        "preset_key": preset_key,
        "t_celsius": float(t_celsius),
        "i_scale": float(i_scale),
        "enable_ICa": bool(cfg.channels.enable_ICa),
        "dynamic_Ca": bool(cfg.calcium.dynamic_Ca),
        "gCa_max": float(cfg.channels.gCa_max),
        "B_Ca": float(cfg.calcium.B_Ca),
        "tau_Ca": float(cfg.calcium.tau_Ca),
        "n_spikes": int(len(spikes)),
        "v_peak_mV": float(np.max(res.v_soma)),
        "stable": bool(np.all(np.isfinite(res.v_soma))),
    }

    if res.ca_i is not None:
        ca = np.maximum(res.ca_i[0, :], 1e-12)  # mM
        eca = np.array([nernst_ca_ion(float(c), cfg.calcium.Ca_ext, 273.15 + cfg.env.T_celsius) for c in ca])
        row.update(
            {
                "ca_min_nM": float(np.min(ca) * 1e6),
                "ca_max_nM": float(np.max(ca) * 1e6),
                "ca_rest_nM": float(cfg.calcium.Ca_rest * 1e6),
                "eca_min_mV": float(np.min(eca)),
                "eca_med_mV": float(np.median(eca)),
                "eca_max_mV": float(np.max(eca)),
            }
        )
    else:
        row.update(
            {
                "ca_min_nM": None,
                "ca_max_nM": None,
                "ca_rest_nM": float(cfg.calcium.Ca_rest * 1e6),
                "eca_min_mV": None,
                "eca_med_mV": None,
                "eca_max_mV": None,
            }
        )

    ica = res.currents.get("ICa", None) if hasattr(res, "currents") and isinstance(res.currents, dict) else None
    if ica is not None:
        dt = float(np.mean(np.diff(res.t))) if len(res.t) > 1 else 0.0
        inward = np.maximum(-np.asarray(ica), 0.0)  # inward proxy
        row["ica_inward_integral"] = float(np.sum(inward) * dt)
        row["ica_peak_abs"] = float(np.max(np.abs(ica)))
    else:
        row["ica_inward_integral"] = None
        row["ica_peak_abs"] = None

    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Extended calcium/Nernst validation report")
    parser.add_argument("--temps", type=str, default="23,37")
    parser.add_argument("--i-scales", type=str, default="0.8,1.0")
    parser.add_argument("--output", type=str, default="_test_results/calcium_nernst_extended_report.json")
    args = parser.parse_args()

    temps = [float(x.strip()) for x in args.temps.split(",") if x.strip()]
    i_scales = [float(x.strip()) for x in args.i_scales.split(",") if x.strip()]
    presets = [
        "K: Thalamic Relay (Ih + ITCa + Burst)",
        "L: Hippocampal CA1 Pyramidal (Adapting)",
        "M: Epilepsy (v10 SCN1A mutation)",
        "N_progressive",
        "N_terminal",
        "O_progressive",
        "O_terminal",
    ]

    rows = []
    anomalies = []
    total = len(presets) * len(temps) * len(i_scales)
    done = 0

    for p in presets:
        for t_c in temps:
            for i_scale in i_scales:
                row = _run_case(p, t_celsius=t_c, i_scale=i_scale)
                rows.append(row)
                done += 1
                print(
                    f"{done:03d}/{total} {p[:22]:22} T={t_c:.0f} I={i_scale:.2f} "
                    f"sp={row['n_spikes']} eca={row['eca_med_mV']}",
                    flush=True,
                )

                if not row["stable"]:
                    anomalies.append({"type": "non_finite_voltage", **row})
                    continue

                if row["enable_ICa"] and row["dynamic_Ca"]:
                    if row["ca_min_nM"] is None or row["ca_max_nM"] is None:
                        anomalies.append({"type": "missing_ca_trace", **row})
                        continue
                    if row["ca_min_nM"] < 0.0:
                        anomalies.append({"type": "negative_ca", **row})
                    if row["ca_max_nM"] > 5000.0:
                        anomalies.append({"type": "ca_overload", **row})
                    if row["eca_min_mV"] is None or row["eca_max_mV"] is None:
                        anomalies.append({"type": "missing_eca", **row})
                    else:
                        if row["eca_min_mV"] < 95.0 or row["eca_max_mV"] > 190.0:
                            anomalies.append({"type": "eca_out_of_range", **row})
                    if row["ica_inward_integral"] is not None and row["ica_inward_integral"] <= 0.0:
                        anomalies.append({"type": "no_inward_ica", **row})

    # Temperature trend check: E_Ca median should increase from 23C to 37C for same preset+drive.
    for p in presets:
        for i_scale in i_scales:
            r23 = [r for r in rows if r["preset_key"] == p and abs(r["t_celsius"] - 23.0) < 1e-9 and abs(r["i_scale"] - i_scale) < 1e-9]
            r37 = [r for r in rows if r["preset_key"] == p and abs(r["t_celsius"] - 37.0) < 1e-9 and abs(r["i_scale"] - i_scale) < 1e-9]
            if not r23 or not r37:
                continue
            r23, r37 = r23[0], r37[0]
            if (
                r23["enable_ICa"]
                and r23["dynamic_Ca"]
                and r23["eca_med_mV"] is not None
                and r37["eca_med_mV"] is not None
                and not (r37["eca_med_mV"] > r23["eca_med_mV"])
            ):
                anomalies.append(
                    {
                        "type": "eca_temp_trend_fail",
                        "preset_key": p,
                        "i_scale": float(i_scale),
                        "eca_med_23": r23["eca_med_mV"],
                        "eca_med_37": r37["eca_med_mV"],
                    }
                )

    out = {
        "config": {"presets": presets, "temps": temps, "i_scales": i_scales},
        "summary": {
            "cases": len(rows),
            "anomalies": len(anomalies),
            "pass_ratio": float((len(rows) - len(anomalies)) / max(1, len(rows))),
        },
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

