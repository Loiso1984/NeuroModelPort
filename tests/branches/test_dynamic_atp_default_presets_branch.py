from __future__ import annotations

import numpy as np

from core.analysis import detect_spikes
from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver


def _run_with_dynamic_atp(preset: str, *, t_sim: float) -> tuple[FullModelConfig, object, np.ndarray]:
    cfg = FullModelConfig()
    apply_preset(cfg, preset)
    cfg.metabolism.enable_dynamic_atp = True
    cfg.stim.t_sim = max(float(cfg.stim.t_sim), float(t_sim))
    cfg.stim.dt_eval = min(float(cfg.stim.dt_eval), 0.2)
    cfg.stim.jacobian_mode = "native_hines"
    res = NeuronSolver(cfg).run_single()
    _, spike_times, _ = detect_spikes(
        np.asarray(res.v_soma, dtype=float),
        np.asarray(res.t, dtype=float),
        threshold=-20.0,
        prominence=10.0,
        refractory_ms=2.0,
    )
    return cfg, res, np.asarray(spike_times, dtype=float)


def _assert_dynamic_atp_exports(res) -> None:
    assert getattr(res, "atp_level", None) is not None
    assert getattr(res, "na_i", None) is not None
    assert getattr(res, "k_o", None) is not None
    assert np.all(np.isfinite(np.asarray(res.atp_level, dtype=float)))
    assert np.all(np.isfinite(np.asarray(res.na_i, dtype=float)))
    assert np.all(np.isfinite(np.asarray(res.k_o, dtype=float)))


def test_dynamic_atp_preserves_repetitive_squid_spiking():
    _, res, spike_times = _run_with_dynamic_atp("A: Squid Giant Axon (HH 1952)", t_sim=150.0)
    _assert_dynamic_atp_exports(res)
    assert len(spike_times) >= 8, f"Squid + ATP should remain repetitively spiking, got {len(spike_times)} spikes"
    assert float(np.max(res.v_soma)) > 30.0


def test_dynamic_atp_preserves_detectable_l5_spikes():
    _, res, spike_times = _run_with_dynamic_atp("B: Pyramidal L5 (Mainen 1996)", t_sim=500.0)
    _assert_dynamic_atp_exports(res)
    assert len(spike_times) >= 4, f"L5 + ATP should remain clearly spiking, got {len(spike_times)} spikes"
    assert float(np.max(res.v_soma)) > 30.0, "L5 + ATP spike peaks should remain comfortably above detection threshold"


def test_dynamic_atp_preserves_motoneuron_tonic_output():
    _, res, spike_times = _run_with_dynamic_atp("D: alpha-Motoneuron (Powers 2001)", t_sim=300.0)
    _assert_dynamic_atp_exports(res)
    assert len(spike_times) >= 10, f"Motoneuron + ATP should retain tonic output, got {len(spike_times)} spikes"
    assert float(np.max(res.v_soma)) > 20.0
