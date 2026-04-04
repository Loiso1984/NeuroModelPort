"""
Branch tests for physiology-first validation of optional channels.

Rules enforced by this branch suite:
1. Optional channel conductances must propagate from config to morphology.
2. HCN must reduce apparent input resistance under hyperpolarizing pulse.
3. IA must produce non-zero transient outward current when enabled.
4. Calcium Nernst potential must stay in physiologically realistic range.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.morphology import MorphologyBuilder
from core.presets import apply_preset, get_preset_names
from core.kinetics import ar_Ih, br_Ih
from core.rhs import nernst_ca_ion
from core.solver import NeuronSolver
from core.analysis import detect_spikes


def _build_hcn_pulse_config(enable_hcn: bool) -> FullModelConfig:
    cfg = FullModelConfig()
    cfg.morphology.single_comp = True
    cfg.stim.jacobian_mode = "sparse_fd"

    # Disable spike-generating channels to isolate passive+Ih behavior.
    cfg.channels.gNa_max = 0.0
    cfg.channels.gK_max = 0.0
    cfg.channels.gL = 0.05
    cfg.channels.EL = -65.0

    cfg.channels.enable_Ih = enable_hcn
    cfg.channels.gIh_max = 0.03
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
    return cfg


def _build_ia_probe_config() -> FullModelConfig:
    cfg = FullModelConfig()
    cfg.morphology.single_comp = True
    cfg.stim.jacobian_mode = "sparse_fd"

    cfg.channels.gNa_max = 120.0
    cfg.channels.gK_max = 36.0
    cfg.channels.gL = 0.3
    cfg.channels.ENa = 50.0
    cfg.channels.EK = -77.0
    cfg.channels.EL = -54.387

    cfg.channels.enable_IA = True
    cfg.channels.gA_max = 0.8
    cfg.channels.E_A = -77.0

    cfg.channels.enable_Ih = False
    cfg.channels.enable_ICa = False
    cfg.channels.enable_SK = False

    cfg.stim.stim_type = "pulse"
    cfg.stim.Iext = 20.0
    cfg.stim.pulse_start = 20.0
    cfg.stim.pulse_dur = 80.0
    cfg.stim.t_sim = 140.0
    cfg.stim.dt_eval = 0.05
    return cfg


def _build_calcium_probe_config() -> FullModelConfig:
    cfg = FullModelConfig()
    cfg.morphology.single_comp = True
    cfg.stim.jacobian_mode = "sparse_fd"

    cfg.channels.gNa_max = 0.0
    cfg.channels.gK_max = 0.0
    cfg.channels.gL = 0.05
    cfg.channels.EL = -65.0

    cfg.channels.enable_Ih = False
    cfg.channels.enable_IA = False
    cfg.channels.enable_SK = False
    cfg.channels.enable_ICa = True
    cfg.channels.gCa_max = 0.08

    cfg.calcium.dynamic_Ca = True
    cfg.calcium.Ca_rest = 50e-6
    cfg.calcium.Ca_ext = 2.0
    cfg.calcium.tau_Ca = 200.0
    cfg.calcium.B_Ca = 0.001

    cfg.stim.stim_type = "pulse"
    cfg.stim.Iext = 1.0
    cfg.stim.pulse_start = 20.0
    cfg.stim.pulse_dur = 50.0
    cfg.stim.t_sim = 120.0
    cfg.stim.dt_eval = 0.05
    return cfg


def _estimate_rin(v: np.ndarray, t: np.ndarray, pulse_start: float, pulse_end: float, iext: float) -> float:
    baseline_mask = (t >= pulse_start - 20.0) & (t < pulse_start - 2.0)
    steady_mask = (t >= pulse_end - 25.0) & (t < pulse_end - 5.0)
    v_baseline = float(np.mean(v[baseline_mask]))
    v_steady = float(np.mean(v[steady_mask]))
    dv = v_steady - v_baseline
    return abs(dv / iext)


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


def test_optional_conductances_propagate_to_morphology():
    cfg_k = FullModelConfig()
    apply_preset(cfg_k, "K: Thalamic Relay (Ih + ICa + Burst)")
    morph_k = MorphologyBuilder.build(cfg_k)
    assert np.max(morph_k["gIh_v"]) > 0.0, "Ih enabled in preset K, but gIh_v is zero"
    assert np.max(morph_k["gCa_v"]) > 0.0, "ICa enabled in preset K, but gCa_v is zero"
    assert np.max(morph_k["gA_v"]) == 0.0, "IA disabled in preset K, but gA_v is non-zero"

    cfg_l = FullModelConfig()
    apply_preset(cfg_l, "L: Hippocampal CA1 (Theta rhythm)")
    morph_l = MorphologyBuilder.build(cfg_l)
    assert np.max(morph_l["gIh_v"]) > 0.0, "Ih enabled in preset L, but gIh_v is zero"
    assert np.max(morph_l["gCa_v"]) == 0.0, "ICa should be disabled in CA1 theta preset"
    assert np.max(morph_l["gA_v"]) > 0.0, "IA enabled in preset L, but gA_v is zero"

    cfg_b = FullModelConfig()
    apply_preset(cfg_b, "B: Pyramidal L5 (Mainen 1996)")
    morph_b = MorphologyBuilder.build(cfg_b)
    assert np.max(morph_b["gIh_v"]) == 0.0
    assert np.max(morph_b["gCa_v"]) == 0.0
    assert np.max(morph_b["gA_v"]) == 0.0
    assert np.max(morph_b["gSK_v"]) == 0.0


def test_membrane_capacitance_propagates_to_morphology():
    cfg = FullModelConfig()
    apply_preset(cfg, "B: Pyramidal L5 (Mainen 1996)")
    morph = MorphologyBuilder.build(cfg)
    cm_unique = np.unique(np.round(morph["Cm_v"], 6))
    expected = round(cfg.channels.Cm, 6)
    assert cm_unique.size == 1 and abs(cm_unique[0] - expected) < 1e-9, (
        f"Cm_v must equal cfg.channels.Cm ({expected}), got {cm_unique}"
    )


def test_hcn_reduces_input_resistance():
    cfg_no_hcn = _build_hcn_pulse_config(enable_hcn=False)
    cfg_hcn = _build_hcn_pulse_config(enable_hcn=True)

    res_no_hcn = NeuronSolver(cfg_no_hcn).run_single()
    res_hcn = NeuronSolver(cfg_hcn).run_single()

    pulse_start = cfg_hcn.stim.pulse_start
    pulse_end = cfg_hcn.stim.pulse_start + cfg_hcn.stim.pulse_dur
    iext = cfg_hcn.stim.Iext

    rin_no_hcn = _estimate_rin(res_no_hcn.v_soma, res_no_hcn.t, pulse_start, pulse_end, iext)
    rin_hcn = _estimate_rin(res_hcn.v_soma, res_hcn.t, pulse_start, pulse_end, iext)

    # Ih is a depolarizing inward current during hyperpolarization; it lowers apparent Rin.
    assert rin_hcn < rin_no_hcn * 0.95, (
        f"Expected Ih to reduce Rin by >=5%, got Rin_with={rin_hcn:.3f}, Rin_without={rin_no_hcn:.3f}"
    )


def test_hcn_vhalf_activation_is_physiological():
    v = np.linspace(-130.0, -30.0, 10001)
    r_inf = ar_Ih(v) / (ar_Ih(v) + br_Ih(v))
    i_half = int(np.argmin(np.abs(r_inf - 0.5)))
    v_half = float(v[i_half])
    # Typical HCN activation midpoint is in hyperpolarized range (~ -70 to -95 mV).
    assert -95.0 <= v_half <= -60.0, f"HCN V1/2 out of expected range: {v_half:.2f} mV"


def test_hcn_temperature_accelerates_sag_kinetics():
    cfg_23 = _build_hcn_pulse_config(enable_hcn=True)
    cfg_37 = _build_hcn_pulse_config(enable_hcn=True)
    cfg_23.env.T_celsius = 23.0
    cfg_37.env.T_celsius = 37.0

    res_23 = NeuronSolver(cfg_23).run_single()
    res_37 = NeuronSolver(cfg_37).run_single()

    start = cfg_23.stim.pulse_start
    end = cfg_23.stim.pulse_start + cfg_23.stim.pulse_dur
    w23 = (res_23.t >= start) & (res_23.t <= end)
    w37 = (res_37.t >= start) & (res_37.t <= end)

    t_min_23 = float(res_23.t[w23][np.argmin(res_23.v_soma[w23])])
    t_min_37 = float(res_37.t[w37][np.argmin(res_37.v_soma[w37])])
    # Higher temperature should speed gating and shift sag minimum earlier in time.
    assert t_min_37 < t_min_23 - 20.0, (
        f"HCN sag not sufficiently faster at 37C: t_min_23={t_min_23:.2f}, t_min_37={t_min_37:.2f}"
    )


def test_ia_current_is_nonzero_when_enabled():
    cfg = _build_ia_probe_config()
    res = NeuronSolver(cfg).run_single()
    assert "IA" in res.currents
    ia_peak = float(np.max(np.abs(res.currents["IA"])))
    assert ia_peak > 1e-4, f"Expected non-zero IA current, got peak {ia_peak:.6g}"


def test_ia_suppresses_repetitive_spiking_near_threshold():
    cfg_off = _build_ia_probe_config()
    cfg_on = _build_ia_probe_config()
    cfg_off.channels.enable_IA = False
    cfg_off.channels.gA_max = 0.0
    # Near-threshold current where IA effect is pronounced in this reduced model.
    cfg_off.stim.Iext = 8.0
    cfg_on.stim.Iext = 8.0

    res_off = NeuronSolver(cfg_off).run_single()
    res_on = NeuronSolver(cfg_on).run_single()
    n_off = len(_spike_times(res_off.v_soma, res_off.t))
    n_on = len(_spike_times(res_on.v_soma, res_on.t))
    assert n_on <= max(1, n_off - 4), (
        f"IA should suppress repetitive near-threshold spiking: without={n_off}, with={n_on}"
    )


def test_ia_sweep_monotonic_reduction_trend():
    spike_counts = []
    for ga in [0.0, 0.2, 0.4, 0.6, 0.8]:
        cfg = _build_ia_probe_config()
        cfg.stim.Iext = 8.0
        cfg.channels.enable_IA = ga > 0.0
        cfg.channels.gA_max = ga
        res = NeuronSolver(cfg).run_single()
        spike_counts.append(len(_spike_times(res.v_soma, res.t)))
    # Expect non-increasing trend with stronger IA under fixed drive.
    assert all(spike_counts[i + 1] <= spike_counts[i] for i in range(len(spike_counts) - 1)), (
        f"IA sweep should show non-increasing excitability trend, got {spike_counts}"
    )


def test_calcium_nernst_is_physiological_and_temp_sensitive():
    # Typical physiology: [Ca]o=2 mM, [Ca]i(rest)=50 nM.
    eca_23 = float(nernst_ca_ion(50e-6, 2.0, 296.15))
    eca_37 = float(nernst_ca_ion(50e-6, 2.0, 310.15))

    assert 130.0 <= eca_37 <= 150.0, f"E_Ca at 37C out of range: {eca_37:.2f} mV"
    assert 120.0 <= eca_23 <= 145.0, f"E_Ca at 23C out of range: {eca_23:.2f} mV"
    assert eca_37 > eca_23, "E_Ca should increase with temperature in Nernst equation"


def test_inward_ica_increases_intracellular_calcium():
    cfg = _build_calcium_probe_config()
    res = NeuronSolver(cfg).run_single()
    ca = res.ca_i[0, :]
    ca_rest = cfg.calcium.Ca_rest

    assert float(np.max(ca)) > ca_rest * 1.001, (
        f"Expected calcium to rise above rest during inward ICa; max={np.max(ca):.6g}, rest={ca_rest:.6g}"
    )
    assert float(np.min(ca)) >= 0.0, f"Calcium concentration must stay non-negative, got min={np.min(ca):.6g}"


def test_double_stimulation_disabled_by_default_for_all_presets():
    for preset in get_preset_names():
        cfg = FullModelConfig()
        apply_preset(cfg, preset)
        if cfg.dual_stimulation is None:
            continue
        enabled = getattr(cfg.dual_stimulation, "enabled", False)
        assert not enabled, f"Preset '{preset}' must have dual stimulation disabled by default"


def test_hcn_presets_have_stable_rest_without_stimulus():
    for preset in ["K: Thalamic Relay (Ih + ICa + Burst)", "L: Hippocampal CA1 (Theta rhythm)"]:
        cfg = FullModelConfig()
        apply_preset(cfg, preset)
        cfg.stim.Iext = 0.0
        cfg.stim.stim_type = "const"
        cfg.stim.t_sim = 300.0
        cfg.stim.dt_eval = 0.2
        cfg.stim.jacobian_mode = "sparse_fd"

        res = NeuronSolver(cfg).run_single()
        tail = res.v_soma[-100:]
        v_rest = float(np.mean(tail))
        v_std = float(np.std(tail))

        assert -80.0 <= v_rest <= -55.0, f"{preset}: non-physiological rest {v_rest:.2f} mV"
        assert v_std < 2.0, f"{preset}: unstable rest (std={v_std:.2f} mV)"


def test_hcn_presets_remain_excitable_with_default_stimulus():
    for preset in ["K: Thalamic Relay (Ih + ICa + Burst)", "L: Hippocampal CA1 (Theta rhythm)"]:
        cfg = FullModelConfig()
        apply_preset(cfg, preset)
        cfg.stim.t_sim = 200.0
        cfg.stim.dt_eval = 0.2
        cfg.stim.jacobian_mode = "sparse_fd"

        res = NeuronSolver(cfg).run_single()
        peaks, spike_times, _ = detect_spikes(res.v_soma, res.t, threshold=-20.0, baseline_threshold=-50.0)
        assert len(spike_times) > 0, f"{preset}: lost excitability under default stimulus"
        assert float(np.max(res.v_soma)) > 20.0, f"{preset}: spike peak too low ({np.max(res.v_soma):.2f} mV)"


def test_ca1_theta_preset_has_theta_band_rate():
    cfg = FullModelConfig()
    apply_preset(cfg, "L: Hippocampal CA1 (Theta rhythm)")
    cfg.stim.t_sim = 500.0
    cfg.stim.dt_eval = 0.2
    cfg.stim.jacobian_mode = "sparse_fd"

    res = NeuronSolver(cfg).run_single()
    _, spike_times, _ = detect_spikes(res.v_soma, res.t, threshold=-20.0, baseline_threshold=-50.0)
    assert len(spike_times) >= 2, "CA1 theta preset should produce repetitive spiking for theta-rate estimation"
    freq_hz = 1000.0 / float(np.mean(np.diff(spike_times)))
    assert 4.0 <= freq_hz <= 12.0, f"CA1 theta preset out of theta band: {freq_hz:.2f} Hz"


def test_dynamic_calcium_presets_have_bounded_calcium_range():
    presets = [
        "E: Cerebellar Purkinje (De Schutter)",
        "K: Thalamic Relay (Ih + ICa + Burst)",
        "M: Epilepsy (v10 SCN1A mutation)",
        "N: Alzheimer's (v10 Calcium Toxicity)",
        "O: Hypoxia (v10 ATP-pump failure)",
    ]
    for preset in presets:
        cfg = FullModelConfig()
        apply_preset(cfg, preset)
        assert cfg.calcium.dynamic_Ca, f"{preset}: expected dynamic calcium enabled"
        cfg.stim.jacobian_mode = "sparse_fd"

        res = NeuronSolver(cfg).run_single()
        ca = res.ca_i[0, :] * 1e6  # mM -> nM
        ca_min = float(np.min(ca))
        ca_max = float(np.max(ca))

        assert ca_min >= 0.0, f"{preset}: negative calcium ({ca_min:.2f} nM)"
        assert ca_max <= 5000.0, f"{preset}: unrealistic calcium overload ({ca_max:.2f} nM)"


def test_dynamic_calcium_presets_have_physiological_eca_and_temp_behavior():
    presets = [
        "E: Cerebellar Purkinje (De Schutter)",
        "K: Thalamic Relay (Ih + ICa + Burst)",
        "M: Epilepsy (v10 SCN1A mutation)",
        "N: Alzheimer's (v10 Calcium Toxicity)",
        "O: Hypoxia (v10 ATP-pump failure)",
    ]
    for preset in presets:
        cfg = FullModelConfig()
        apply_preset(cfg, preset)
        cfg.stim.jacobian_mode = "sparse_fd"
        cfg.stim.t_sim = 180.0
        cfg.stim.dt_eval = 0.25
        assert cfg.calcium.dynamic_Ca, f"{preset}: expected dynamic calcium enabled"

        res = NeuronSolver(cfg).run_single()
        ca_i = np.maximum(res.ca_i[0, :], 1e-9)  # mM

        eca_23 = np.array([nernst_ca_ion(float(c), cfg.calcium.Ca_ext, 296.15) for c in ca_i])
        eca_37 = np.array([nernst_ca_ion(float(c), cfg.calcium.Ca_ext, 310.15) for c in ca_i])

        assert np.all(np.isfinite(eca_23)) and np.all(np.isfinite(eca_37)), f"{preset}: non-finite E_Ca"
        assert 95.0 <= float(np.min(eca_37)) <= 170.0, f"{preset}: E_Ca(37C) minimum out of range"
        assert 110.0 <= float(np.max(eca_37)) <= 190.0, f"{preset}: E_Ca(37C) maximum out of range"
        assert float(np.median(eca_37)) > float(np.median(eca_23)), (
            f"{preset}: E_Ca should increase with temperature"
        )


def _run_as_script() -> int:
    tests = [
        test_optional_conductances_propagate_to_morphology,
        test_membrane_capacitance_propagates_to_morphology,
        test_hcn_reduces_input_resistance,
        test_hcn_vhalf_activation_is_physiological,
        test_hcn_temperature_accelerates_sag_kinetics,
        test_ia_current_is_nonzero_when_enabled,
        test_ia_suppresses_repetitive_spiking_near_threshold,
        test_ia_sweep_monotonic_reduction_trend,
        test_calcium_nernst_is_physiological_and_temp_sensitive,
        test_inward_ica_increases_intracellular_calcium,
        test_double_stimulation_disabled_by_default_for_all_presets,
        test_hcn_presets_have_stable_rest_without_stimulus,
        test_hcn_presets_remain_excitable_with_default_stimulus,
        test_ca1_theta_preset_has_theta_band_rate,
        test_dynamic_calcium_presets_have_bounded_calcium_range,
        test_dynamic_calcium_presets_have_physiological_eca_and_temp_behavior,
    ]

    passed = 0
    for fn in tests:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] {fn.__name__}: {exc}")

    total = len(tests)
    print(f"\nSummary: {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(_run_as_script())
