"""
Unified branch protocol for physiology-first preset validation.

Focus:
- robust spike counting via threshold transitions,
- demyelination phenotype (soma spikes with impaired axonal propagation),
- mode behavior for K/N/O presets.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.analysis import detect_spikes
from core.models import FullModelConfig
from core.presets import apply_preset, get_preset_names
from core.solver import NeuronSolver


def _spike_times_by_crossing(v: np.ndarray, t: np.ndarray, threshold: float = -20.0) -> np.ndarray:
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


def _run_preset(
    preset_name: str,
    *,
    t_sim: float = 260.0,
    dt_eval: float = 0.2,
    k_mode: str | None = None,
    alz_mode: str | None = None,
    hyp_mode: str | None = None,
):
    cfg = FullModelConfig()
    if k_mode is not None:
        cfg.preset_modes.k_mode = k_mode
    if alz_mode is not None:
        cfg.preset_modes.alzheimer_mode = alz_mode
    if hyp_mode is not None:
        cfg.preset_modes.hypoxia_mode = hyp_mode
    apply_preset(cfg, preset_name)
    cfg.stim.t_sim = t_sim
    cfg.stim.dt_eval = dt_eval
    cfg.stim.jacobian_mode = "native_hines"
    res = NeuronSolver(cfg).run_single()
    st = _spike_times_by_crossing(res.v_soma, res.t)
    return cfg, res, st


def test_spike_counting_is_transition_based():
    t = np.linspace(0.0, 100.0, 1001)
    v = np.full_like(t, -65.0)
    # One long suprathreshold plateau should count as one event, not many samples.
    v[(t >= 20.0) & (t <= 40.0)] = 15.0

    transition_spikes = _spike_times_by_crossing(v, t, threshold=-20.0)
    naive_sample_count = int(np.sum(v > -20.0))
    assert len(transition_spikes) == 1, "Spike must be counted by threshold transition"
    assert naive_sample_count > 1, "Naive sample count is intentionally inflated for this check"


def test_all_presets_run_without_numerical_break():
    for preset in get_preset_names():
        _, res, _ = _run_preset(preset, t_sim=180.0, dt_eval=0.25)
        assert np.all(np.isfinite(res.v_soma)), f"{preset}: non-finite voltage trace"
        assert -140.0 < float(np.min(res.v_soma)) < 80.0, f"{preset}: minimum voltage out of guard range"
        assert -140.0 < float(np.max(res.v_soma)) < 80.0, f"{preset}: maximum voltage out of guard range"


def test_demyelination_preserves_soma_spike_but_impairs_axon_propagation():
    _, res_d, _ = _run_preset("D: alpha-Motoneuron (Powers 2001)", t_sim=220.0, dt_eval=0.15)
    _, res_f, _ = _run_preset("F: Multiple Sclerosis (Demyelination)", t_sim=220.0, dt_eval=0.15)

    peak_d_soma = float(np.max(res_d.v_soma))
    peak_f_soma = float(np.max(res_f.v_soma))
    assert peak_d_soma > 20.0, "Control motoneuron should produce robust somatic spikes"
    assert peak_f_soma > 10.0, "Demyelination preset should still spike at soma"

    j_d = min(1 + 5 + 35, res_d.n_comp - 1)
    j_f = min(1 + 5 + 35, res_f.n_comp - 1)
    ratio_d = float(np.max(res_d.v_all[j_d, :]) / max(peak_d_soma, 1e-6))
    ratio_f = float(np.max(res_f.v_all[j_f, :]) / max(peak_f_soma, 1e-6))

    t_d_soma = _first_cross(res_d.v_soma, res_d.t, 0.0)
    t_d_term = _first_cross(res_d.v_all[-1, :], res_d.t, 0.0)
    t_f_soma = _first_cross(res_f.v_soma, res_f.t, 0.0)
    t_f_term = _first_cross(res_f.v_all[-1, :], res_f.t, 0.0)

    assert not np.isnan(t_d_soma) and not np.isnan(t_d_term), "Control should propagate across axon"
    assert not np.isnan(t_f_soma) and not np.isnan(t_f_term), "Pathology should still conduct, but worse"
    delay_d = t_d_term - t_d_soma
    delay_f = t_f_term - t_f_soma
    assert delay_f > delay_d + 0.5, (
        f"Demyelination should slow conduction: delay_d={delay_d:.2f}ms, delay_f={delay_f:.2f}ms"
    )
    assert ratio_f < ratio_d, (
        f"Demyelination should reduce proximal propagation effectiveness: ratio_d={ratio_d:.3f}, ratio_f={ratio_f:.3f}"
    )


def test_k_modes_baseline_vs_activated():
    _, res_base, st_base = _run_preset(
        "K: Thalamic Relay (Ih + ICa + Burst)",
        t_sim=300.0,
        dt_eval=0.2,
        k_mode="baseline",
    )
    _, res_act, st_act = _run_preset(
        "K: Thalamic Relay (Ih + ICa + Burst)",
        t_sim=300.0,
        dt_eval=0.2,
        k_mode="activated",
    )
    dur_base = float(res_base.t[-1] - res_base.t[0]) if len(res_base.t) > 1 else 0.0
    dur_act = float(res_act.t[-1] - res_act.t[0]) if len(res_act.t) > 1 else 0.0
    fg_base = float(1000.0 * len(st_base) / dur_base) if dur_base > 0 else 0.0
    fg_act = float(1000.0 * len(st_act) / dur_act) if dur_act > 0 else 0.0

    assert 4.0 <= fg_base <= 25.0, f"K baseline global frequency should stay low-throughput/theta-like, got {fg_base:.2f} Hz"
    assert len(st_act) >= len(st_base), "Activated thalamic mode should not be less excitable than baseline"
    assert len(st_act) >= 3, "Activated thalamic mode should produce robust relay output"
    assert fg_act >= fg_base + 10.0, f"K activated should clearly exceed baseline throughput ({fg_base:.2f} -> {fg_act:.2f} Hz)"


def test_k_default_mode_is_baseline_and_low_throughput():
    cfg, res, st = _run_preset(
        "K: Thalamic Relay (Ih + ICa + Burst)",
        t_sim=300.0,
        dt_eval=0.2,
    )
    assert cfg.preset_modes.k_mode == "baseline", "K default mode should be baseline"
    dur = float(res.t[-1] - res.t[0]) if len(res.t) > 1 else 0.0
    fg = float(1000.0 * len(st) / dur) if dur > 0 else 0.0
    assert 4.0 <= fg <= 25.0, f"K default baseline throughput should remain low, got {fg:.2f} Hz"


def test_alzheimer_progressive_vs_terminal_modes():
    _, res_prog, st_prog = _run_preset(
        "N: Alzheimer's (v10 Calcium Toxicity)",
        t_sim=320.0,
        dt_eval=0.2,
        alz_mode="progressive",
    )
    _, _, st_term = _run_preset(
        "N: Alzheimer's (v10 Calcium Toxicity)",
        t_sim=320.0,
        dt_eval=0.2,
        alz_mode="terminal",
    )
    assert len(st_prog) >= 1, "Progressive Alzheimer mode should show at least initial spiking"
    assert len(st_term) <= len(st_prog), "Terminal Alzheimer mode should be equal or less excitable"
    assert float(np.max(res_prog.v_soma)) > 20.0, "Progressive Alzheimer mode should show visible spikes"


def test_hypoxia_progressive_vs_terminal_modes():
    _, res_prog, st_prog = _run_preset(
        "O: Hypoxia (v10 ATP-pump failure)",
        t_sim=320.0,
        dt_eval=0.2,
        hyp_mode="progressive",
    )
    _, res_term, st_term = _run_preset(
        "O: Hypoxia (v10 ATP-pump failure)",
        t_sim=320.0,
        dt_eval=0.2,
        hyp_mode="terminal",
    )
    first_half_prog = int(np.sum(st_prog < 160.0))
    second_half_prog = int(np.sum(st_prog >= 160.0))
    assert float(np.max(res_prog.v_soma)) >= float(np.max(res_term.v_soma)), "Progressive hypoxia should not be less excitable than terminal"
    assert first_half_prog >= 1 or float(np.max(res_prog.v_soma)) > -10.0, "Progressive hypoxia should show early excitatory activity"
    assert second_half_prog <= first_half_prog, "Progressive hypoxia should not gain late activity"
    assert len(st_term) <= len(st_prog), "Terminal hypoxia mode should be equal or less excitable"


def test_detect_spikes_consistency_with_transition_counter():
    _, res, _ = _run_preset("B: Pyramidal L5 (Mainen 1996)", t_sim=220.0, dt_eval=0.2)
    _, st_detect, _ = detect_spikes(res.v_soma, res.t, threshold=-20.0, baseline_threshold=-50.0)
    st_cross = _spike_times_by_crossing(res.v_soma, res.t, threshold=-20.0)
    # Allow small differences from repolarization logic, but prohibit major divergence.
    assert abs(len(st_detect) - len(st_cross)) <= 2, (
        f"Spike counters diverged: detect_spikes={len(st_detect)}, crossing={len(st_cross)}"
    )


def test_heavy_presets_use_native_hines_jacobian_mode():
    heavy = [
        "F: Multiple Sclerosis (Demyelination)",
        "K: Thalamic Relay (Ih + ICa + Burst)",
        "N: Alzheimer's (v10 Calcium Toxicity)",
        "O: Hypoxia (v10 ATP-pump failure)",
    ]
    for preset in heavy:
        cfg = FullModelConfig()
        apply_preset(cfg, preset)
        # Note: We now force native_hines in all branch tests per "Hines First" policy
        # This test verifies the branch test infrastructure enforces native_hines
        assert cfg.stim.jacobian_mode == "native_hines", (
            f"{preset}: expected native_hines for heavy preset, got {cfg.stim.jacobian_mode}"
        )


def _run_as_script() -> int:
    tests = [
        test_spike_counting_is_transition_based,
        test_all_presets_run_without_numerical_break,
        test_demyelination_preserves_soma_spike_but_impairs_axon_propagation,
        test_k_modes_baseline_vs_activated,
        test_k_default_mode_is_baseline_and_low_throughput,
        test_alzheimer_progressive_vs_terminal_modes,
        test_hypoxia_progressive_vs_terminal_modes,
        test_detect_spikes_consistency_with_transition_counter,
        test_heavy_presets_use_native_hines_jacobian_mode,
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
