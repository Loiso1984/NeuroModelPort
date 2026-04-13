from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver
from tests.shared_utils import _spike_times


def _run(preset: str, *, t_sim: float = 150.0, dt_eval: float = 0.1, **mode_overrides):
    cfg = FullModelConfig()
    for field, value in mode_overrides.items():
        setattr(cfg.preset_modes, field, value)
    apply_preset(cfg, preset)
    cfg.stim.t_sim = max(float(cfg.stim.t_sim), float(t_sim))
    cfg.stim.dt_eval = min(float(cfg.stim.dt_eval), float(dt_eval))
    cfg.stim.jacobian_mode = "native_hines"
    res = NeuronSolver(cfg).run_single()
    st = _spike_times(np.asarray(res.v_soma), np.asarray(res.t))
    return cfg, res, st


def test_ms_keeps_soma_spiking_but_blocks_trunk_propagation():
    _, res, st = _run("F: Multiple Sclerosis (Demyelination)", t_sim=180.0, dt_eval=0.1)
    assert len(st) >= 3, f"MS preset should still spike at soma, got {len(st)} spikes"
    assert float(np.max(res.v_soma)) > 20.0
    j = min(1 + res.config.morphology.N_ais + res.config.morphology.N_trunk - 1, res.n_comp - 1)
    assert float(np.max(res.v_all[j, :])) < 0.0, "MS junction should show strong attenuation/block"


def test_trn_is_rebound_burst_surrogate_not_silent():
    _, res, st = _run("P: Thalamic Reticular Nucleus (TRN Spindles)", t_sim=180.0, dt_eval=0.1)
    assert len(st) >= 2, f"TRN surrogate should emit a delayed rebound burst, got {len(st)} spikes"
    assert float(st[0]) >= 100.0, f"TRN rebound should emerge after release from hyperpolarization, got first spike at {st[0]:.2f} ms"
    assert float(np.max(res.v_soma)) > 25.0, "TRN rebound spike should remain clearly suprathreshold"
    assert float(np.min(res.v_soma)) < -95.0, "TRN should hyperpolarize before rebound"


def test_l5_default_restores_adapting_spike_train():
    _, res, st = _run("B: Pyramidal L5 (Mainen 1996)", t_sim=500.0, dt_eval=0.2)
    assert len(st) >= 5, f"L5 should sustain an adapting train, got {len(st)} spikes"
    assert float(np.max(res.v_soma)) > 35.0
    assert len(st) <= 20, f"L5 default should stay in a moderate adapting regime, got {len(st)} spikes"
    if len(st) > 2:
        isi = np.diff(st)
        assert float(np.mean(isi[-2:])) >= float(np.mean(isi[:2])) * 0.9, "L5 should not accelerate into an FS-like regime"


def test_spn_default_is_delayed_spiking_not_immediate():
    _, _, st = _run("Q: Striatal Spiny Projection (SPN)", t_sim=180.0, dt_eval=0.1)
    assert len(st) >= 3, f"SPN should show delayed recruitment, got {len(st)} spikes"
    assert float(st[0]) >= 20.0, f"SPN first spike should be delayed, got {st[0]:.2f} ms"


def test_ach_modes_separate_sleep_and_arousal_output():
    _, _, st_sleep = _run(
        "R: Cholinergic Neuromodulation (ACh)",
        ach_mode="sleep",
        t_sim=180.0,
        dt_eval=0.1,
    )
    _, _, st_arousal = _run(
        "R: Cholinergic Neuromodulation (ACh)",
        ach_mode="arousal",
        t_sim=180.0,
        dt_eval=0.1,
    )
    assert len(st_sleep) >= 1, "Sleep ACh mode should retain sparse output"
    assert len(st_arousal) > len(st_sleep) + 5, "Arousal mode should be markedly more excitable"


def test_thalamic_relay_modes_span_rebound_tonic_and_delta_like_regimes():
    _, _, st_baseline = _run(
        "K: Thalamic Relay (Ih + ITCa + Burst)",
        k_mode="baseline",
        t_sim=220.0,
        dt_eval=0.1,
    )
    _, _, st_activated = _run(
        "K: Thalamic Relay (Ih + ITCa + Burst)",
        k_mode="activated",
        t_sim=220.0,
        dt_eval=0.1,
    )
    _, _, st_delta = _run(
        "K: Thalamic Relay (Ih + ITCa + Burst)",
        k_mode="delta_oscillator",
        t_sim=1000.0,
        dt_eval=0.2,
    )
    assert len(st_baseline) >= 2, "Relay baseline should retain rebound output"
    assert len(st_activated) > len(st_baseline) + 5, "Activated relay mode should be much more tonic/excitable"
    assert 4 <= len(st_delta) <= 8, f"Delta-like surrogate should stay slow and sparse, got {len(st_delta)} spikes"
    assert float(st_delta[0]) >= 200.0, f"Delta-like mode should emerge slowly, got first spike at {st_delta[0]:.1f} ms"


def test_alzheimer_progressive_differs_from_terminal():
    _, _, st_prog = _run(
        "N: Alzheimer's (v10 Calcium Toxicity)",
        alzheimer_mode="progressive",
        t_sim=500.0,
        dt_eval=0.2,
    )
    _, _, st_term = _run(
        "N: Alzheimer's (v10 Calcium Toxicity)",
        alzheimer_mode="terminal",
        t_sim=500.0,
        dt_eval=0.2,
    )
    assert len(st_prog) >= 2, "Progressive Alzheimer mode should retain an early response before collapse"
    assert len(st_term) <= 1, "Terminal Alzheimer mode should be near-silent or retain only a vestigial first response"
    assert float(st_prog[-1]) > 200.0, "Progressive Alzheimer should show prolonged sparse activity"


def test_hypoxia_progressive_shows_ion_drift_before_terminal_silence():
    cfg_prog, res_prog, st_prog = _run(
        "O: Hypoxia (v10 ATP-pump failure)",
        hypoxia_mode="progressive",
        t_sim=500.0,
        dt_eval=0.2,
    )
    cfg_term, res_term, st_term = _run(
        "O: Hypoxia (v10 ATP-pump failure)",
        hypoxia_mode="terminal",
        t_sim=500.0,
        dt_eval=0.2,
    )
    assert res_prog.k_o is not None and res_prog.na_i is not None, "Hypoxia should export dynamic ion-gradient states"
    assert len(st_prog) >= 5, "Progressive hypoxia should show a brief early spiking epoch before collapse"
    assert len(st_term) <= 1, "Terminal hypoxia should be effectively silent"
    assert float(res_prog.k_o[0, -1]) > cfg_prog.metabolism.k_o_rest_mM + 0.2, "Progressive hypoxia should elevate extracellular K+"
    assert float(res_prog.na_i[0, -1]) > cfg_prog.metabolism.na_i_rest_mM, "Progressive hypoxia should accumulate intracellular Na+"
    assert float(res_term.k_o[0, -1]) >= cfg_term.metabolism.k_o_rest_mM, "Terminal hypoxia should keep the ion-drift path active"
    assert float(st_prog[-1]) < 200.0, "Progressive hypoxia should collapse before late sustained firing"


def test_anesthesia_modes_partial_vs_full_block():
    _, res_partial, st_partial = _run(
        "G: Local Anesthesia (gNa Block)",
        anesthesia_mode="partial_block",
        t_sim=180.0,
        dt_eval=0.1,
    )
    _, res_full, st_full = _run(
        "G: Local Anesthesia (gNa Block)",
        anesthesia_mode="full_block",
        t_sim=180.0,
        dt_eval=0.1,
    )
    assert len(st_partial) >= 1, "Partial Na block should allow a blunted spike response"
    assert len(st_full) == 0, "Full Na block should suppress spikes"
    assert float(np.max(res_partial.v_soma)) > float(np.max(res_full.v_soma)) + 5.0


def test_dravet_baseline_is_weaker_than_control_and_febrile_is_weaker_than_baseline():
    _, res_base, st_base = _run(
        "S: Pathology: Dravet Syndrome (SCN1A LOF)",
        dravet_mode="baseline",
        t_sim=300.0,
        dt_eval=0.1,
    )
    _, res_febrile, st_febrile = _run(
        "S: Pathology: Dravet Syndrome (SCN1A LOF)",
        dravet_mode="febrile",
        t_sim=300.0,
        dt_eval=0.1,
    )
    assert len(st_base) >= 1, "Baseline Dravet should retain at least one compromised inhibitory spike"
    assert len(st_febrile) < len(st_base), "Febrile Dravet should further degrade firing reserve"
    assert float(np.max(res_febrile.v_soma)) < float(np.max(res_base.v_soma)), "Febrile mode should blunt spike amplitude"
