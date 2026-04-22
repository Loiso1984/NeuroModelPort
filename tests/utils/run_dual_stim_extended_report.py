"""
Deterministic dual-stimulation extended validation report.

Covers branch-equivalent scenarios with exported metrics artifact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.dual_stimulation import DualStimulationConfig
from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver
from tests.shared_utils import _spike_times


def _metrics(tag: str, res) -> dict:
    st = _spike_times(res.v_soma, res.t)
    dur = float(res.t[-1] - res.t[0]) if len(res.t) > 1 else 0.0
    return {
        "tag": tag,
        "stable": bool(np.all(np.isfinite(res.v_soma))),
        "n_spikes": int(len(st)),
        "v_peak_mV": float(np.max(res.v_soma)),
        "v_tail_mV": float(np.mean(res.v_soma[-80:])),
        "freq_global_hz": float(1000.0 * len(st) / dur) if dur > 0 else 0.0,
    }


def _build_l5() -> FullModelConfig:
    cfg = FullModelConfig()
    apply_preset(cfg, "B: Pyramidal L5 (Mainen 1996)")
    cfg.stim.t_sim = 220.0
    cfg.stim.dt_eval = 0.2
    cfg.stim.jacobian_mode = "sparse_fd"
    return cfg


def _build_k_activated() -> FullModelConfig:
    cfg = FullModelConfig()
    cfg.preset_modes.k_mode = "activated"
    apply_preset(cfg, "K: Thalamic Relay (Ih + ITCa + Burst)")
    cfg.stim.t_sim = 300.0
    cfg.stim.dt_eval = 0.2
    cfg.stim.jacobian_mode = "sparse_fd"
    return cfg


def main() -> int:
    rows = []
    anomalies = []

    # 1) Dual disabled ~ baseline
    cfg_base = _build_l5()
    res_base = NeuronSolver(cfg_base).run_single()
    m_base = _metrics("l5_base", res_base)
    rows.append(m_base)

    cfg_dual_off = _build_l5()
    dual_off = DualStimulationConfig()
    dual_off.enabled = False
    cfg_dual_off.dual_stimulation = dual_off
    res_dual_off = NeuronSolver(cfg_dual_off).run_single()
    m_dual_off = _metrics("l5_dual_off", res_dual_off)
    rows.append(m_dual_off)

    if abs(m_dual_off["n_spikes"] - m_base["n_spikes"]) > 1 or abs(m_dual_off["v_peak_mV"] - m_base["v_peak_mV"]) > 3.0:
        anomalies.append({"type": "dual_off_mismatch_baseline", "base": m_base, "dual_off": m_dual_off})

    # 2) L5 inhibitory secondary reduces spiking
    cfg_l5_inh = _build_l5()
    dual_inh = DualStimulationConfig()
    dual_inh.enabled = True
    dual_inh.secondary_location = "soma"
    dual_inh.secondary_stim_type = "GABAB"
    dual_inh.secondary_Iext = 10.0
    dual_inh.secondary_start = 0.0
    dual_inh.secondary_duration = cfg_l5_inh.stim.t_sim
    cfg_l5_inh.dual_stimulation = dual_inh
    m_l5_inh = _metrics("l5_inhibitory_secondary", NeuronSolver(cfg_l5_inh).run_single())
    rows.append(m_l5_inh)
    if m_l5_inh["n_spikes"] >= m_base["n_spikes"]:
        anomalies.append({"type": "l5_inhibition_not_reducing", "base": m_base, "inh": m_l5_inh})

    # 3) L5 soma+AIS remains excitable
    cfg_soma_ais = _build_l5()
    cfg_soma_ais.stim.stim_type = "alpha"
    cfg_soma_ais.stim.Iext = 8.0
    cfg_soma_ais.stim.pulse_start = 20.0
    cfg_soma_ais.stim.pulse_dur = 3.0
    cfg_soma_ais.stim.alpha_tau = 2.0
    dual_sa = DualStimulationConfig()
    dual_sa.enabled = True
    dual_sa.secondary_location = "ais"
    dual_sa.secondary_stim_type = "alpha"
    dual_sa.secondary_Iext = 6.0
    dual_sa.secondary_start = 35.0
    dual_sa.secondary_duration = 3.0
    dual_sa.secondary_alpha_tau = 2.0
    cfg_soma_ais.dual_stimulation = dual_sa
    m_soma_ais = _metrics("l5_soma_plus_ais", NeuronSolver(cfg_soma_ais).run_single())
    rows.append(m_soma_ais)
    if (not m_soma_ais["stable"]) or m_soma_ais["n_spikes"] < 1:
        anomalies.append({"type": "l5_soma_ais_unstable_or_silent", "row": m_soma_ais})

    # 4) L5 main primary with neutral dual secondary
    cfg_override = _build_l5()
    cfg_override.stim.stim_type = "const"
    cfg_override.stim.Iext = 12.0
    cfg_override.stim.pulse_start = 0.0
    cfg_override.stim.pulse_dur = cfg_override.stim.t_sim
    dual_ov = DualStimulationConfig()
    dual_ov.enabled = True
    dual_ov.secondary_location = "soma"
    dual_ov.secondary_stim_type = "const"
    dual_ov.secondary_Iext = 0.0
    dual_ov.secondary_start = 0.0
    dual_ov.secondary_duration = cfg_override.stim.t_sim
    cfg_override.dual_stimulation = dual_ov
    m_override = _metrics("l5_primary_with_neutral_secondary", NeuronSolver(cfg_override).run_single())
    rows.append(m_override)
    if m_override["n_spikes"] < 1:
        anomalies.append({"type": "l5_primary_override_failed", "row": m_override})

    # 5) K activated inhibitory modulation (rebound-aware criterion)
    cfg_k_base = _build_k_activated()
    res_k_base = NeuronSolver(cfg_k_base).run_single()
    m_k_base = _metrics("k_activated_base", res_k_base)
    rows.append(m_k_base)

    cfg_k_inh = _build_k_activated()
    dual_k = DualStimulationConfig()
    dual_k.enabled = True
    dual_k.secondary_location = "soma"
    dual_k.secondary_stim_type = "const"
    dual_k.secondary_Iext = -8.0
    dual_k.secondary_start = 0.0
    dual_k.secondary_duration = cfg_k_inh.stim.t_sim
    cfg_k_inh.dual_stimulation = dual_k
    res_k_inh = NeuronSolver(cfg_k_inh).run_single()
    m_k_inh = _metrics("k_activated_inhibitory_secondary", res_k_inh)
    rows.append(m_k_inh)

    if (not m_k_base["stable"]) or (not m_k_inh["stable"]):
        anomalies.append({"type": "k_activated_unstable", "base": m_k_base, "inh": m_k_inh})

    out = {
        "summary": {
            "cases": len(rows),
            "anomalies": len(anomalies),
            "pass_ratio": float((len(rows) - len(anomalies)) / max(1, len(rows))),
        },
        "anomalies": anomalies,
        "rows": rows,
    }

    out_path = Path("_test_results/dual_stim_extended_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out_path}")
    print(f"Anomalies: {len(anomalies)} / {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
