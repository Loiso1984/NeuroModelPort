"""
Branch validation for solver-side parameter guards and custom exceptions.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.errors import SimulationParameterError
from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver
from core.validation import validate_simulation_config, build_preset_mode_warnings
from core.dual_stimulation import DualStimulationConfig


def test_invalid_dt_eval_vs_t_sim_raises_custom_error():
    cfg = FullModelConfig()
    apply_preset(cfg, "A: Squid Giant Axon (HH 1952)")
    cfg.stim.t_sim = 10.0
    cfg.stim.dt_eval = 20.0
    try:
        NeuronSolver(cfg).run_single()
    except SimulationParameterError:
        return
    raise AssertionError("Expected SimulationParameterError for dt_eval > t_sim")


def test_invalid_dynamic_calcium_tau_raises_custom_error():
    cfg = FullModelConfig()
    apply_preset(cfg, "N: Alzheimer's (v10 Calcium Toxicity)")
    cfg.calcium.dynamic_Ca = True
    cfg.calcium.tau_Ca = 0.0
    try:
        NeuronSolver(cfg).run_single()
    except SimulationParameterError:
        return
    raise AssertionError("Expected SimulationParameterError for tau_Ca <= 0")


def test_invalid_enabled_channel_with_zero_conductance_raises_custom_error():
    cfg = FullModelConfig()
    apply_preset(cfg, "L: Hippocampal CA1 (Theta rhythm)")
    cfg.channels.enable_IA = True
    cfg.channels.gA_max = 0.0
    try:
        NeuronSolver(cfg).run_single()
    except SimulationParameterError:
        return
    raise AssertionError("Expected SimulationParameterError for IA enabled with gA_max <= 0")


def test_valid_config_still_runs_after_validation_layer():
    cfg = FullModelConfig()
    apply_preset(cfg, "B: Pyramidal L5 (Mainen 1996)")
    cfg.stim.t_sim = 80.0
    cfg.stim.dt_eval = 0.2
    cfg.stim.jacobian_mode = "sparse_fd"
    res = NeuronSolver(cfg).run_single()
    assert len(res.t) > 0


def test_validation_warns_on_nonphysiological_iext():
    cfg = FullModelConfig()
    apply_preset(cfg, "A: Squid Giant Axon (HH 1952)")
    cfg.stim.Iext = 300.0
    warnings = validate_simulation_config(cfg)
    assert any("High |Iext|" in w for w in warnings), "Expected high-Iext warning"


def test_validation_warns_on_heavy_runtime_estimate():
    cfg = FullModelConfig()
    apply_preset(cfg, "F: Multiple Sclerosis (Demyelination)")
    cfg.stim.t_sim = 800.0
    cfg.stim.dt_eval = 0.05
    warnings = validate_simulation_config(cfg)
    assert any("Heavy simulation estimate" in w for w in warnings), (
        "Expected heavy simulation runtime warning"
    )


def test_validation_warns_that_dual_stim_overrides_primary_fields():
    cfg = FullModelConfig()
    apply_preset(cfg, "B: Pyramidal L5 (Mainen 1996)")
    dual = DualStimulationConfig()
    dual.enabled = True
    cfg.dual_stimulation = dual
    warnings = validate_simulation_config(cfg)
    assert any("Dual stimulation is enabled" in w for w in warnings), (
        "Expected warning about dual-stim overriding primary stimulation fields"
    )


def test_validation_warns_on_terminal_pathology_modes():
    cfg_n = FullModelConfig()
    cfg_n.preset_modes.alzheimer_mode = "terminal"
    apply_preset(cfg_n, "N: Alzheimer's (v10 Calcium Toxicity)")
    wn = build_preset_mode_warnings(cfg_n, "N: Alzheimer's (v10 Calcium Toxicity)")
    assert any("N mode=terminal" in w for w in wn), "Expected terminal-stage warning for N terminal mode"

    cfg_o = FullModelConfig()
    cfg_o.preset_modes.hypoxia_mode = "terminal"
    apply_preset(cfg_o, "O: Hypoxia (v10 ATP-pump failure)")
    wo = build_preset_mode_warnings(cfg_o, "O: Hypoxia (v10 ATP-pump failure)")
    assert any("O mode=terminal" in w for w in wo), "Expected terminal-stage warning for O terminal mode"


def test_validation_warns_modes_ignored_for_non_kno_presets():
    cfg = FullModelConfig()
    cfg.preset_modes.alzheimer_mode = "terminal"
    apply_preset(cfg, "F: Multiple Sclerosis (Demyelination)")
    w = build_preset_mode_warnings(cfg, "F: Multiple Sclerosis (Demyelination)")
    assert any("single-stage" in s for s in w), "Expected explicit single-stage note for F preset"
    assert any("ignored for this preset" in s for s in w), (
        "Expected warning that K/N/O mode flags are ignored for non-K/N/O preset"
    )


def test_validation_reports_thalamic_mode_note():
    cfg = FullModelConfig()
    cfg.preset_modes.k_mode = "activated"
    apply_preset(cfg, "K: Thalamic Relay (Ih + ICa + Burst)")
    wk = build_preset_mode_warnings(cfg, "K: Thalamic Relay (Ih + ICa + Burst)")
    assert any("K mode=activated" in w for w in wk), "Expected K activated mode note"


def _run_as_script() -> int:
    tests = [
        test_invalid_dt_eval_vs_t_sim_raises_custom_error,
        test_invalid_dynamic_calcium_tau_raises_custom_error,
        test_invalid_enabled_channel_with_zero_conductance_raises_custom_error,
        test_valid_config_still_runs_after_validation_layer,
        test_validation_warns_on_nonphysiological_iext,
        test_validation_warns_on_heavy_runtime_estimate,
        test_validation_warns_that_dual_stim_overrides_primary_fields,
        test_validation_warns_on_terminal_pathology_modes,
        test_validation_warns_modes_ignored_for_non_kno_presets,
        test_validation_reports_thalamic_mode_note,
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
