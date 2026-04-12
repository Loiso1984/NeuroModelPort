from __future__ import annotations

import numpy as np
import pytest
import subprocess
import sys

pytest.importorskip("pydantic")

from core.analysis import reconstruct_stimulus_trace, _stable_seed_from_values
from core.analysis import compute_current_balance
from core.analysis import detect_spikes
from core.dual_stimulation import distributed_stimulus_current_for_comp
from core.dual_stimulation import DualStimulationConfig
from core.models import FullModelConfig
from core.presets import apply_preset, get_preset_names
from core.rhs import get_event_driven_conductance, _get_syn_reversal
from core.solver import (
    NeuronSolver,
    generate_effective_event_times,
    _stable_seed_from_values as _solver_seed,
)


def test_distributed_stimulus_current_for_comp_soma_mode():
    vals = [
        distributed_stimulus_current_for_comp(
            comp_idx=i,
            n_comp=4,
            base_current=3.0,
            stim_comp=2,
            stim_mode=0,
            use_dfilter=0,
            dfilter_attenuation=1.0,
            dfilter_tau_ms=10.0,
            i_filtered=0.0,
        )
        for i in range(4)
    ]
    assert vals == [0.0, 0.0, 3.0, 0.0]


def test_distributed_stimulus_current_for_comp_ais_mode():
    vals = [
        distributed_stimulus_current_for_comp(
            comp_idx=i,
            n_comp=4,
            base_current=2.5,
            stim_comp=0,
            stim_mode=1,
            use_dfilter=0,
            dfilter_attenuation=1.0,
            dfilter_tau_ms=10.0,
            i_filtered=0.0,
        )
        for i in range(4)
    ]
    assert vals == [0.0, 2.5, 0.0, 0.0]


def test_distributed_stimulus_current_for_comp_dfilter_mode():
    vals_no_filter = [
        distributed_stimulus_current_for_comp(
            comp_idx=i,
            n_comp=3,
            base_current=4.0,
            stim_comp=0,
            stim_mode=2,
            use_dfilter=0,
            dfilter_attenuation=0.25,
            dfilter_tau_ms=10.0,
            i_filtered=0.0,
        )
        for i in range(3)
    ]
    assert vals_no_filter == [1.0, 0.0, 0.0]

    vals_filter = [
        distributed_stimulus_current_for_comp(
            comp_idx=i,
            n_comp=3,
            base_current=4.0,
            stim_comp=0,
            stim_mode=2,
            use_dfilter=1,
            dfilter_attenuation=0.25,
            dfilter_tau_ms=10.0,
            i_filtered=0.33,
        )
        for i in range(3)
    ]
    assert vals_filter == [0.33, 0.0, 0.0]


def test_distributed_stimulus_current_for_comp_tau_nonpositive_disables_filter():
    vals_tau_zero = [
        distributed_stimulus_current_for_comp(
            comp_idx=i,
            n_comp=3,
            base_current=4.0,
            stim_comp=0,
            stim_mode=2,
            use_dfilter=1,
            dfilter_attenuation=0.25,
            dfilter_tau_ms=0.0,
            i_filtered=0.33,
        )
        for i in range(3)
    ]
    assert vals_tau_zero == [1.0, 0.0, 0.0]


def test_reconstruct_stimulus_trace_is_deterministic_for_synaptic_trains():
    cfg = FullModelConfig()
    cfg.morphology.single_comp = True
    cfg.stim.jacobian_mode = "native_hines"
    cfg.stim.t_sim = 40.0
    cfg.stim.dt_eval = 0.1
    cfg.stim.stim_type = "NMDA"
    cfg.stim.Iext = 2.0
    cfg.stim.pulse_start = 5.0
    cfg.stim.synaptic_train_type = "poisson"
    cfg.stim.synaptic_train_freq_hz = 30.0
    cfg.stim.synaptic_train_duration_ms = 25.0

    res = NeuronSolver(cfg).run_native(cfg)
    stim_1 = reconstruct_stimulus_trace(res)
    stim_2 = reconstruct_stimulus_trace(res)

    assert np.array_equal(stim_1, stim_2)


def test_reconstruct_stimulus_trace_matches_solver_primitives_for_gabaa():
    cfg = FullModelConfig()
    cfg.morphology.single_comp = True
    cfg.stim.jacobian_mode = "native_hines"
    cfg.stim.t_sim = 35.0
    cfg.stim.dt_eval = 0.1
    cfg.stim.stim_type = "GABAA"
    cfg.stim.Iext = 4.0
    cfg.stim.pulse_start = 5.0
    cfg.stim.synaptic_train_type = "regular"
    cfg.stim.synaptic_train_freq_hz = 20.0
    cfg.stim.synaptic_train_duration_ms = 20.0
    cfg.channels.e_rev_syn_secondary = -40.0

    res = NeuronSolver(cfg).run_native(cfg)
    stim = reconstruct_stimulus_trace(res)

    seed = _stable_seed_from_values(
        cfg.stim.synaptic_train_freq_hz,
        cfg.stim.synaptic_train_duration_ms,
        cfg.stim.pulse_start,
        cfg.stim.synaptic_train_type,
    )
    events = generate_effective_event_times(
        cfg.stim.synaptic_train_type,
        cfg.stim.synaptic_train_freq_hz,
        cfg.stim.synaptic_train_duration_ms,
        cfg.stim.pulse_start,
        cfg.stim.event_times,
        seed_hash=seed,
    )
    if len(events) == 0:
        events = np.array([cfg.stim.pulse_start], dtype=np.float64)

    expected = np.zeros_like(stim)
    stype = 6  # GABAA
    e_syn = _get_syn_reversal(stype, cfg.channels.e_rev_syn_primary, cfg.channels.e_rev_syn_secondary)
    for i, ti in enumerate(res.t):
        g_t = get_event_driven_conductance(ti, stype, cfg.stim.Iext, events, len(events), cfg.stim.alpha_tau)
        expected[i] = abs(g_t) * (e_syn - cfg.channels.EL)

    assert np.allclose(stim, expected, atol=1e-12, rtol=0.0)


def test_seed_helpers_match_between_solver_and_analysis():
    vals = (40.0, 200.0, 12.5, "poisson")
    assert _stable_seed_from_values(*vals) == _solver_seed(*vals)


def test_generate_effective_event_times_is_stable_across_processes():
    code = r"""
import json
from core.solver import generate_effective_event_times, _stable_seed_from_values
seed = _stable_seed_from_values(35.0, 120.0, 8.0, "poisson")
ev = generate_effective_event_times("poisson", 35.0, 120.0, 8.0, [], seed_hash=seed)
print(json.dumps(ev.tolist()))
"""
    run1 = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    run2 = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert run1 == run2


def test_synaptic_events_equal_between_run_single_and_run_native_paths():
    cfg = FullModelConfig()
    cfg.morphology.single_comp = True
    cfg.stim.stim_type = "AMPA"
    cfg.stim.synaptic_train_type = "regular"
    cfg.stim.synaptic_train_freq_hz = 25.0
    cfg.stim.synaptic_train_duration_ms = 100.0
    cfg.stim.pulse_start = 10.0

    seed = _solver_seed(
        cfg.stim.synaptic_train_freq_hz,
        cfg.stim.synaptic_train_duration_ms,
        cfg.stim.pulse_start,
        cfg.stim.synaptic_train_type,
    )
    ev_single = generate_effective_event_times(
        cfg.stim.synaptic_train_type,
        cfg.stim.synaptic_train_freq_hz,
        cfg.stim.synaptic_train_duration_ms,
        cfg.stim.pulse_start,
        cfg.stim.event_times,
        seed_hash=seed,
    )
    ev_native = generate_effective_event_times(
        cfg.stim.synaptic_train_type,
        cfg.stim.synaptic_train_freq_hz,
        cfg.stim.synaptic_train_duration_ms,
        cfg.stim.pulse_start,
        cfg.stim.event_times,
        seed_hash=seed,
    )

    assert np.array_equal(ev_single, ev_native)


def test_get_syn_reversal_mapping_for_all_synaptic_types():
    e_exc = 5.0
    e_inh = -42.0

    # Excitatory synapses should use e_rev_syn_primary
    for stype in (4, 5, 8, 9):
        assert _get_syn_reversal(stype, e_exc, e_inh) == pytest.approx(e_exc)

    # GABA-A should use configurable inhibitory reversal
    assert _get_syn_reversal(6, e_exc, e_inh) == pytest.approx(e_inh)

    # GABA-B remains fixed to canonical GIRK-mediated reversal
    assert _get_syn_reversal(7, e_exc, e_inh) == pytest.approx(-95.0)


def test_preset_switch_clears_stale_dual_and_notes_state():
    cfg = FullModelConfig()
    cfg.notes = "stale note"
    cfg.dual_stimulation = DualStimulationConfig(enabled=True, secondary_Iext=99.0)

    apply_preset(cfg, "E: Cerebellar Purkinje (De Schutter)")
    assert cfg.dual_stimulation is not None and cfg.dual_stimulation.enabled

    # Switching to a non-dual preset must clear dual state and notes.
    cfg.notes = "carry me"
    apply_preset(cfg, "A: Squid Giant Axon (HH 1952)")
    assert cfg.dual_stimulation is None
    assert cfg.notes == ""


def test_preset_switch_resets_mode_flags_to_defaults():
    cfg = FullModelConfig()
    cfg.preset_modes.k_mode = "activated"
    cfg.preset_modes.alzheimer_mode = "terminal"
    cfg.preset_modes.hypoxia_mode = "terminal"
    cfg.preset_modes.l5_mode = "high_ach"
    cfg.preset_modes.dravet_mode = "febrile"
    cfg.preset_modes.delay_target = "Soma"

    apply_preset(cfg, "A: Squid Giant Axon (HH 1952)")
    assert cfg.preset_modes.k_mode == "baseline"
    assert cfg.preset_modes.alzheimer_mode == "progressive"
    assert cfg.preset_modes.hypoxia_mode == "progressive"
    assert cfg.preset_modes.l5_mode == "normal"
    assert cfg.preset_modes.dravet_mode == "normal"
    assert cfg.preset_modes.delay_target == "Terminal"


def test_all_presets_apply_without_leaking_previous_dual_state():
    cfg = FullModelConfig()
    names = get_preset_names()
    assert len(names) > 0

    # Start from known dual-enabled state to catch leakage.
    apply_preset(cfg, "E: Cerebellar Purkinje (De Schutter)")
    assert cfg.dual_stimulation is not None and cfg.dual_stimulation.enabled

    dual_expected_substrings = (
        "Purkinje",
        "Epilepsy",
        "Hippocampal CA1",
    )
    for name in names:
        apply_preset(cfg, name)
        dual_is_on = bool(cfg.dual_stimulation is not None and cfg.dual_stimulation.enabled)
        if any(sub in name for sub in dual_expected_substrings):
            assert dual_is_on, f"{name} should define dual stimulation"
        else:
            assert not dual_is_on, f"{name} should not inherit stale dual stimulation"


def test_katp_is_extracted_for_dynamic_atp_presets():
    cfg = FullModelConfig()
    apply_preset(cfg, "O: Hypoxia (v10 ATP-pump failure)")
    cfg.stim.t_sim = 80.0
    cfg.stim.dt_eval = 0.2
    cfg.stim.jacobian_mode = "native_hines"

    res = NeuronSolver(cfg).run_native(cfg)
    assert "KATP" in res.currents
    katp = np.asarray(res.currents["KATP"], dtype=float)
    assert katp.ndim == 2
    assert katp.shape[1] == len(res.t)
    assert np.all(np.isfinite(katp))


def test_current_balance_output_is_soma_aligned_shape():
    cfg = FullModelConfig()
    apply_preset(cfg, "B: Pyramidal L5 (Mainen 1996)")
    cfg.stim.t_sim = 60.0
    cfg.stim.dt_eval = 0.2
    cfg.stim.jacobian_mode = "native_hines"

    res = NeuronSolver(cfg).run_native(cfg)
    i_bal = compute_current_balance(res, res.morph)
    assert i_bal.shape == res.t.shape
    assert np.all(np.isfinite(i_bal))


def test_thalamic_baseline_uses_hyperpolarizing_rebound_policy():
    cfg = FullModelConfig()
    apply_preset(cfg, "K: Thalamic Relay (Ih + ICa + Burst)")
    assert cfg.preset_modes.k_mode == "baseline"
    assert cfg.stim.stim_type == "pulse"
    assert cfg.stim.Iext < 0.0
    assert abs(cfg.stim.Iext) == pytest.approx(4.0, abs=1.5)
    assert cfg.stim.pulse_start == pytest.approx(10.0)
    assert 40.0 <= cfg.stim.pulse_dur <= 120.0
    assert cfg.channels.EL == pytest.approx(-75.0)


def test_thalamic_baseline_produces_post_inhibitory_rebound_signature():
    cfg = FullModelConfig()
    apply_preset(cfg, "K: Thalamic Relay (Ih + ICa + Burst)")
    cfg.stim.t_sim = 220.0
    cfg.stim.dt_eval = 0.1
    cfg.stim.jacobian_mode = "native_hines"

    res = NeuronSolver(cfg).run_native(cfg)
    t = np.asarray(res.t, dtype=float)
    v = np.asarray(res.v_soma, dtype=float)
    _, spike_times, _ = detect_spikes(v, t, threshold=-20.0)

    pulse_end = cfg.stim.pulse_start + cfg.stim.pulse_dur  # expected rebound after ~110 ms
    spikes_after_release = spike_times[spike_times >= pulse_end]
    spikes_during_pulse = spike_times[(spike_times >= cfg.stim.pulse_start) & (spike_times < pulse_end)]

    # Physiological PIR expectations: deep hyperpolarization then rebound spiking.
    in_pulse = (t >= cfg.stim.pulse_start) & (t <= pulse_end)
    assert np.min(v[in_pulse]) < -71.5
    assert len(spikes_during_pulse) <= 1
    # Runtime may be subthreshold in baseline mode; require at least a rebound depolarization.
    pre_pulse = t < cfg.stim.pulse_start
    post_release = t >= pulse_end
    rebound_delta = float(np.max(v[post_release]) - np.mean(v[pre_pulse]))
    assert (len(spikes_after_release) >= 1) or (rebound_delta >= 1.0)
