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
    cfg.stim.jacobian_mode = "native_hines"
    return cfg


def _build_k_activated() -> FullModelConfig:
    cfg = FullModelConfig()
    cfg.preset_modes.k_mode = "activated"
    apply_preset(cfg, "K: Thalamic Relay (Ih + ICa + Burst)")
    cfg.stim.t_sim = 300.0
    cfg.stim.dt_eval = 0.2
    cfg.stim.jacobian_mode = "native_hines"
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
    # Mirror the baseline primary stimulus so we isolate secondary effect.
    dual.primary_location = "soma"
    dual.primary_stim_type = cfg_inh.stim.stim_type
    dual.primary_Iext = cfg_inh.stim.Iext
    dual.primary_start = cfg_inh.stim.pulse_start
    dual.primary_duration = cfg_inh.stim.pulse_dur
    dual.primary_alpha_tau = cfg_inh.stim.alpha_tau
    dual.secondary_location = "soma"
    # Use slow inhibitory kinetics to enforce sustained reduction of excitability.
    dual.secondary_stim_type = "GABAB"
    dual.secondary_Iext = 10.0
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


def test_dual_primary_configuration_overrides_main_stimulus():
    cfg = _build_l5()
    # Main stimulus intentionally set subthreshold/silent.
    cfg.stim.stim_type = "const"
    cfg.stim.Iext = 0.0

    dual = DualStimulationConfig()
    dual.enabled = True
    dual.primary_location = "soma"
    dual.primary_stim_type = "const"
    dual.primary_Iext = 12.0
    dual.primary_start = 0.0
    dual.primary_duration = cfg.stim.t_sim
    # Secondary disabled by zero amplitude.
    dual.secondary_location = "soma"
    dual.secondary_stim_type = "const"
    dual.secondary_Iext = 0.0
    dual.secondary_start = 0.0
    dual.secondary_duration = cfg.stim.t_sim
    cfg.dual_stimulation = dual

    res = NeuronSolver(cfg).run_single()
    n_spikes = len(_spike_times(res.v_soma, res.t))
    assert n_spikes >= 1, (
        "Dual primary stimulus should drive spiking even if main cfg.stim is silent"
    )


def test_secondary_dendritic_filter_tau_modulates_early_response():
    base = _build_l5()
    base.stim.stim_type = "const"
    base.stim.Iext = 0.0

    cfg_fast = _build_l5()
    cfg_fast.stim.stim_type = "const"
    cfg_fast.stim.Iext = 0.0
    dual_fast = DualStimulationConfig()
    dual_fast.enabled = True
    dual_fast.primary_location = "soma"
    dual_fast.primary_stim_type = "const"
    dual_fast.primary_Iext = 0.0
    dual_fast.secondary_location = "dendritic_filtered"
    dual_fast.secondary_stim_type = "alpha"
    dual_fast.secondary_Iext = 20.0
    dual_fast.secondary_start = 20.0
    dual_fast.secondary_duration = 5.0
    dual_fast.secondary_alpha_tau = 2.0
    dual_fast.secondary_distance_um = 150.0
    dual_fast.secondary_space_constant_um = 150.0
    dual_fast.secondary_tau_dendritic_ms = 1.0
    cfg_fast.dual_stimulation = dual_fast

    cfg_slow = _build_l5()
    cfg_slow.stim.stim_type = "const"
    cfg_slow.stim.Iext = 0.0
    dual_slow = DualStimulationConfig()
    dual_slow.enabled = True
    dual_slow.primary_location = "soma"
    dual_slow.primary_stim_type = "const"
    dual_slow.primary_Iext = 0.0
    dual_slow.secondary_location = "dendritic_filtered"
    dual_slow.secondary_stim_type = "alpha"
    dual_slow.secondary_Iext = 20.0
    dual_slow.secondary_start = 20.0
    dual_slow.secondary_duration = 5.0
    dual_slow.secondary_alpha_tau = 2.0
    dual_slow.secondary_distance_um = 150.0
    dual_slow.secondary_space_constant_um = 150.0
    dual_slow.secondary_tau_dendritic_ms = 30.0
    cfg_slow.dual_stimulation = dual_slow

    res_fast = NeuronSolver(cfg_fast).run_single()
    res_slow = NeuronSolver(cfg_slow).run_single()

    w_fast = (res_fast.t >= 20.0) & (res_fast.t <= 45.0)
    w_slow = (res_slow.t >= 20.0) & (res_slow.t <= 45.0)
    early_peak_fast = float(np.max(res_fast.v_soma[w_fast]))
    early_peak_slow = float(np.max(res_slow.v_soma[w_slow]))
    assert early_peak_fast > early_peak_slow + 0.5, (
        f"Dendritic filter tau should shape early response: fast={early_peak_fast:.2f}, slow={early_peak_slow:.2f}"
    )


def test_secondary_inhibition_modulates_thalamic_activated_throughput():
    cfg_base = _build_k_activated()
    res_base = NeuronSolver(cfg_base).run_single()
    n_base = len(_spike_times(res_base.v_soma, res_base.t))

    cfg_inh = _build_k_activated()
    dual = DualStimulationConfig()
    dual.enabled = True
    # Mirror active primary stimulus.
    dual.primary_location = "soma"
    dual.primary_stim_type = cfg_inh.stim.stim_type
    dual.primary_Iext = cfg_inh.stim.Iext
    dual.primary_start = cfg_inh.stim.pulse_start
    dual.primary_duration = cfg_inh.stim.pulse_dur
    dual.primary_alpha_tau = cfg_inh.stim.alpha_tau
    # Add sustained inhibitory secondary drive (explicit negative current).
    dual.secondary_location = "soma"
    dual.secondary_stim_type = "const"
    dual.secondary_Iext = -8.0
    dual.secondary_start = 0.0
    dual.secondary_duration = cfg_inh.stim.t_sim
    cfg_inh.dual_stimulation = dual

    res_inh = NeuronSolver(cfg_inh).run_single()
    n_inh = len(_spike_times(res_inh.v_soma, res_inh.t))
    # Thalamic relay with Ih can show post-inhibitory rebound; inhibitory drive should
    # strongly modulate throughput (not necessarily reduce it monotonically).
    assert n_base >= 10, f"K activated baseline should be robustly spiking, got n={n_base}"
    assert abs(n_inh - n_base) >= 10, (
        f"Inhibitory secondary should significantly modulate K activated throughput: base={n_base}, inh={n_inh}"
    )


def _run_as_script() -> int:
    tests = [
        test_dual_disabled_matches_baseline,
        test_secondary_inhibition_reduces_spiking,
        test_soma_plus_ais_stimulation_runs,
        test_dual_primary_configuration_overrides_main_stimulus,
        test_secondary_dendritic_filter_tau_modulates_early_response,
        test_secondary_inhibition_modulates_thalamic_activated_throughput,
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
