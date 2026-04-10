"""
Unified preset protocol runner.

Produces a single JSON report with physiology-oriented metrics for:
- all standard presets,
- mode variants (K/N/O),
- demyelination conduction comparison (D vs F).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset, get_preset_names
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


def _collect_single(cfg: FullModelConfig, name: str) -> dict:
    start_time = time.perf_counter()
    res = NeuronSolver(cfg).run_single()
    wall_time_ms = (time.perf_counter() - start_time) * 1000.0
    st = _spike_times(res.v_soma, res.t)
    freq_inst = float(1000.0 / np.mean(np.diff(st))) if len(st) > 1 else 0.0
    total_dur_ms = float(res.t[-1] - res.t[0]) if len(res.t) > 1 else 0.0
    freq_global = float(1000.0 * len(st) / total_dur_ms) if total_dur_ms > 0 else 0.0
    active_dur_ms = float(st[-1] - st[0]) if len(st) > 1 else 0.0
    freq_active = float(1000.0 * (len(st) - 1) / active_dur_ms) if active_dur_ms > 0 else 0.0
    row = {
        "preset": name,
        "n_spikes": int(len(st)),
        "freq_hz": freq_inst,
        "freq_global_hz": freq_global,
        "freq_active_window_hz": freq_active,
        "v_rest_tail_mV": float(np.mean(res.v_soma[-100:])),
        "v_peak_mV": float(np.max(res.v_soma)),
        "v_min_mV": float(np.min(res.v_soma)),
        "ca_peak_nM": float(np.max(res.ca_i[0, :]) * 1e6) if res.ca_i is not None else None,
        "stable_finite": bool(np.all(np.isfinite(res.v_soma))),
        "wall_time_ms": wall_time_ms,
    }
    if res.n_comp > 1:
        if cfg.morphology.N_trunk > 0:
            j_idx = min(1 + cfg.morphology.N_ais + cfg.morphology.N_trunk - 1, res.n_comp - 1)
        elif cfg.morphology.N_ais > 0:
            j_idx = min(cfg.morphology.N_ais, res.n_comp - 1)
        else:
            j_idx = min(1, res.n_comp - 1)
        row["terminal_peak_mV"] = float(np.max(res.v_all[-1, :]))
        row["junction_peak_mV"] = float(np.max(res.v_all[j_idx, :]))
        t_s = _first_cross(res.v_soma, res.t, 0.0)
        t_t = _first_cross(res.v_all[-1, :], res.t, 0.0)
        row["delay_term_minus_soma_ms"] = float(t_t - t_s) if not (np.isnan(t_s) or np.isnan(t_t)) else None
    return row


def _base_cfg(name: str, *, t_sim: float = 1000.0, dt_eval: float = 0.5) -> FullModelConfig:
    cfg = FullModelConfig()
    apply_preset(cfg, name)
    cfg.stim.t_sim = t_sim
    cfg.stim.dt_eval = dt_eval
    cfg.stim.jacobian_mode = "native_hines"
    return cfg


def _build_standard_probe_cfg(name: str) -> FullModelConfig:
    """
    Unified probe protocol for cross-preset comparability.
    """
    cfg = FullModelConfig()
    apply_preset(cfg, name)
    probe_amp = float(abs(cfg.stim.Iext))
    probe_amp = max(8.0, min(30.0, probe_amp if probe_amp > 0 else 10.0))
    cfg.stim.stim_type = "const"
    cfg.stim.Iext = probe_amp
    cfg.stim.t_sim = 1000.0
    cfg.stim.dt_eval = 0.5
    cfg.stim.jacobian_mode = "native_hines"
    return cfg


def main() -> int:
    out_dir = Path("_test_results")
    out_dir.mkdir(exist_ok=True)

    report = {
        "default_behavior": [],
        "standard_probe": [],
        "mode_variants": [],
        "demyelination_conduction_check": {},
    }

    for preset in get_preset_names():
        cfg = _base_cfg(preset, t_sim=1000.0, dt_eval=0.5)
        report["default_behavior"].append(_collect_single(cfg, preset))
        probe_cfg = _build_standard_probe_cfg(preset)
        report["standard_probe"].append(_collect_single(probe_cfg, preset))

    # Mode variants: K baseline/activated, N progressive/terminal, O progressive/terminal
    for mode in ["baseline", "activated"]:
        cfg = FullModelConfig()
        cfg.preset_modes.k_mode = mode
        apply_preset(cfg, "K: Thalamic Relay (Ih + ICa + Burst)")
        cfg.stim.t_sim = 1000.0
        cfg.stim.dt_eval = 0.5
        cfg.stim.jacobian_mode = "native_hines"
        row = _collect_single(cfg, f"K_mode={mode}")
        row["mode_family"] = "K"
        report["mode_variants"].append(row)

    for mode in ["progressive", "terminal"]:
        cfg = FullModelConfig()
        cfg.preset_modes.alzheimer_mode = mode
        apply_preset(cfg, "N: Alzheimer's (v10 Calcium Toxicity)")
        cfg.stim.t_sim = 1000.0
        cfg.stim.dt_eval = 0.5
        cfg.stim.jacobian_mode = "native_hines"
        row = _collect_single(cfg, f"N_mode={mode}")
        row["mode_family"] = "N"
        report["mode_variants"].append(row)

    for mode in ["progressive", "terminal"]:
        cfg = FullModelConfig()
        cfg.preset_modes.hypoxia_mode = mode
        apply_preset(cfg, "O: Hypoxia (v10 ATP-pump failure)")
        cfg.stim.t_sim = 320.0
        cfg.stim.dt_eval = 0.2
        cfg.stim.jacobian_mode = "native_hines"
        res = NeuronSolver(cfg).run_single()
        st = _spike_times(res.v_soma, res.t)
        total_dur_ms = float(res.t[-1] - res.t[0]) if len(res.t) > 1 else 0.0
        freq_global = float(1000.0 * len(st) / total_dur_ms) if total_dur_ms > 0 else 0.0
        active_dur_ms = float(st[-1] - st[0]) if len(st) > 1 else 0.0
        freq_active = float(1000.0 * (len(st) - 1) / active_dur_ms) if active_dur_ms > 0 else 0.0
        row = {
            "preset": f"O_mode={mode}",
            "mode_family": "O",
            "n_spikes": int(len(st)),
            "freq_hz": float(1000.0 / np.mean(np.diff(st))) if len(st) > 1 else 0.0,
            "freq_global_hz": freq_global,
            "freq_active_window_hz": freq_active,
            "v_rest_tail_mV": float(np.mean(res.v_soma[-100:])),
            "v_peak_mV": float(np.max(res.v_soma)),
            "v_min_mV": float(np.min(res.v_soma)),
            "ca_peak_nM": float(np.max(res.ca_i[0, :]) * 1e6) if res.ca_i is not None else None,
            "stable_finite": bool(np.all(np.isfinite(res.v_soma))),
        }
        row["spikes_first_half"] = int(np.sum(st < 500.0))
        row["spikes_second_half"] = int(np.sum(st >= 500.0))
        report["mode_variants"].append(row)

    # Demyelination conduction check (D vs F)
    cfg_d = _base_cfg("D: alpha-Motoneuron (Powers 2001)", t_sim=1000.0, dt_eval=0.5)
    cfg_f = _base_cfg("F: Multiple Sclerosis (Demyelination)", t_sim=1000.0, dt_eval=0.5)
    row_d = _collect_single(cfg_d, "D")
    row_f = _collect_single(cfg_f, "F")
    report["demyelination_conduction_check"] = {
        "control_D": row_d,
        "pathology_F": row_f,
        "delay_increase_ms": (
            (row_f.get("delay_term_minus_soma_ms") or 0.0) - (row_d.get("delay_term_minus_soma_ms") or 0.0)
        ),
    }

    out_file = out_dir / "unified_preset_protocol.json"
    out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved report: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
