"""
Branch validation for dual stimulation logic.

Checks:
1. Disabled dual stimulation does not alter baseline behavior.
2. Inhibitory secondary stimulus can reduce excitability.
3. Mixed-location stimulation (soma + AIS) remains functional.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.dual_stimulation import DualStimulationConfig
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


def _build_l5() -> FullModelConfig:
    cfg = FullModelConfig()
    apply_preset(cfg, "B: Pyramidal L5 (Mainen 1996)")
    cfg.stim.t_sim = 220.0
    cfg.stim.dt_eval = 0.2
    cfg.stim.jacobian_mode = "sparse_fd"
    return cfg


def test_dual_disabled_matches_baseline():
    cfg_base = _build_l5()
    res_base = NeuronSolver(cfg_base).run_single()

    cfg_dual_off = _build_l5()
    dual = DualStimulationConfig()
    dual.enabled = False
    cfg_dual_off.dual_stimulation = dual
    res_dual_off = NeuronSolver(cfg_dual_off).run_single()

    n_base = len(_spike_times(res_base.v_soma, res_base.t))
    n_off = len(_spike_times(res_dual_off.v_soma, res_dual_off.t))
    assert abs(n_base - n_off) <= 1, f"Dual-off should match baseline: base={n_base}, dual_off={n_off}"
    assert abs(float(np.max(res_base.v_soma)) - float(np.max(res_dual_off.v_soma))) < 3.0


def test_secondary_inhibition_reduces_spiking():
    cfg_base = _build_l5()
    res_base = NeuronSolver(cfg_base).run_single()
    n_base = len(_spike_times(res_base.v_soma, res_base.t))

    cfg_inh = _build_l5()
    dual = DualStimulationConfig()
    dual.enabled = True
    dual.primary_location = "soma"
    dual.primary_stim_type = "const"
    dual.primary_Iext = cfg_inh.stim.Iext
    dual.primary_start = 0.0
    dual.secondary_location = "soma"
    dual.secondary_stim_type = "GABAA"
    dual.secondary_Iext = 25.0
    dual.secondary_start = 0.0
    dual.secondary_duration = cfg_inh.stim.t_sim
    cfg_inh.dual_stimulation = dual

    res_inh = NeuronSolver(cfg_inh).run_single()
    n_inh = len(_spike_times(res_inh.v_soma, res_inh.t))
    assert n_inh < n_base, f"Inhibitory secondary stimulus should reduce spiking: base={n_base}, inh={n_inh}"


def test_soma_plus_ais_stimulation_runs():
    cfg = _build_l5()
    dual = DualStimulationConfig()
    dual.enabled = True
    dual.primary_location = "soma"
    dual.primary_stim_type = "alpha"
    dual.primary_Iext = 8.0
    dual.primary_start = 20.0
    dual.primary_duration = 3.0
    dual.primary_alpha_tau = 2.0
    dual.secondary_location = "ais"
    dual.secondary_stim_type = "alpha"
    dual.secondary_Iext = 6.0
    dual.secondary_start = 35.0
    dual.secondary_duration = 3.0
    dual.secondary_alpha_tau = 2.0
    cfg.dual_stimulation = dual

    res = NeuronSolver(cfg).run_single()
    n_spikes = len(_spike_times(res.v_soma, res.t))
    assert np.all(np.isfinite(res.v_soma)), "Dual soma+AIS produced non-finite trace"
    assert n_spikes >= 1, "Dual soma+AIS should remain excitable"


def _run_as_script() -> int:
    tests = [
        test_dual_disabled_matches_baseline,
        test_secondary_inhibition_reduces_spiking,
        test_soma_plus_ais_stimulation_runs,
    ]
    passed = 0
    for fn in tests:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] {fn.__name__}: {exc}")
    print(f"\nSummary: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(_run_as_script())
