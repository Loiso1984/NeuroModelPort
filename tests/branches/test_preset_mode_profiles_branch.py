from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
    assert len(st) >= 2, f"TRN surrogate should emit rebound burst, got {len(st)} spikes"
    assert float(np.min(res.v_soma)) < -90.0, "TRN should hyperpolarize before rebound"


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
    assert len(st_term) == 0, "Terminal Alzheimer mode should be near-silent"
    assert float(st_prog[-1]) > 200.0, "Progressive Alzheimer should show prolonged sparse activity"


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
