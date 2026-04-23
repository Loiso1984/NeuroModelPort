from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.advanced_sim import run_sweep
from core.batch_validator import (
    STATUS_D_BLOCK,
    STATUS_OK,
    STATUS_SILENT,
    STATUS_UNSTABLE,
    run_validation_batch,
)
from core.models import FullModelConfig
from core.physics_params import build_env_params, build_state_offsets, unpack_env_params
from core.presets import apply_preset
from core.solver import NeuronSolver
from tests.shared_utils import _spike_times


def test_sweep_accepts_legacy_and_dotted_iext_paths():
    cfg = FullModelConfig()
    apply_preset(cfg, "A: Squid Giant Axon (HH 1952)")
    cfg.stim.t_sim = 120.0
    cfg.stim.dt_eval = 0.2
    cfg.stim.jacobian_mode = "native_hines"
    values = np.array([4.0, 6.0, 8.0], dtype=float)

    legacy = run_sweep(cfg, "Iext", values)
    dotted = run_sweep(cfg, "stim.Iext", values)

    assert len(legacy) == len(values)
    assert len(dotted) == len(values)
    for (v_legacy, res_legacy), (v_dotted, res_dotted) in zip(legacy, dotted):
        assert v_legacy == v_dotted
        assert res_legacy is not None, f"legacy sweep failed for {v_legacy}"
        assert res_dotted is not None, f"dotted sweep failed for {v_dotted}"
        assert len(res_legacy.t) == len(res_dotted.t)
        assert np.isfinite(np.max(res_legacy.v_soma))
        assert np.isfinite(np.max(res_dotted.v_soma))


def test_long_simulation_supports_eight_seconds():
    cfg = FullModelConfig()
    apply_preset(cfg, "A: Squid Giant Axon (HH 1952)")
    cfg.stim.t_sim = 8000.0
    cfg.stim.dt_eval = 1.0
    cfg.stim.jacobian_mode = "native_hines"
    res = NeuronSolver(cfg).run_single()

    assert float(res.t[-1]) >= 7999.0
    assert len(res.t) >= 7000
    assert np.all(np.isfinite(res.v_soma))


def test_dynamic_atp_exports_l5_state_and_preserves_spiking():
    cfg = FullModelConfig()
    apply_preset(cfg, "B: Pyramidal L5 (Mainen 1996)")
    cfg.metabolism.enable_dynamic_atp = True
    cfg.stim.t_sim = 500.0
    cfg.stim.dt_eval = 0.2
    cfg.stim.jacobian_mode = "native_hines"
    res = NeuronSolver(cfg).run_single()
    st = _spike_times(np.asarray(res.v_soma), np.asarray(res.t))

    assert res.atp is not None and res.atp_level is not None, "Dynamic ATP should export ATP pool state"
    assert res.na_i is not None and res.k_o is not None, "Dynamic ATP should export ion-drift states"
    assert res.atp_level.shape == (res.n_comp, len(res.t))
    assert res.na_i.shape == res.atp_level.shape == res.k_o.shape
    assert len(st) >= 5, f"L5 with default ATP should remain clearly spiking, got {len(st)} spikes"
    assert float(np.max(res.v_soma)) > 20.0, "L5 with default ATP should keep suprathreshold spikes"
    assert 0.0 < float(res.atp_level[0, -1]) <= cfg.metabolism.atp_max_mM


def test_state_offsets_match_dynamic_atp_result_layout():
    cfg = FullModelConfig()
    apply_preset(cfg, "B: Pyramidal L5 (Mainen 1996)")
    cfg.metabolism.enable_dynamic_atp = True
    cfg.stim.t_sim = 80.0
    cfg.stim.dt_eval = 0.2
    cfg.stim.jacobian_mode = "native_hines"
    res = NeuronSolver(cfg).run_single()

    dual = getattr(cfg, "dual_stimulation", None)
    use_dfilter_secondary = int(
        bool(dual is not None and getattr(dual, "enabled", False))
        and getattr(dual, "secondary_location", "soma") == "dendritic_filtered"
        and getattr(dual, "secondary_tau_dendritic_ms", 0.0) > 0.0
    )
    offsets = build_state_offsets(
        res.n_comp,
        en_ih=cfg.channels.enable_Ih,
        en_ica=cfg.channels.enable_ICa,
        en_ia=cfg.channels.enable_IA,
        en_sk=cfg.channels.enable_SK,
        dyn_ca=cfg.calcium.dynamic_Ca,
        en_itca=cfg.channels.enable_ITCa,
        en_im=cfg.channels.enable_IM,
        en_nap=cfg.channels.enable_NaP,
        en_nar=cfg.channels.enable_NaR,
        dyn_atp=cfg.metabolism.enable_dynamic_atp,
        use_dfilter_primary=int(
            cfg.stim_location.location == "dendritic_filtered"
            and cfg.dendritic_filter.enabled
            and cfg.dendritic_filter.tau_dendritic_ms > 0.0
        ),
        use_dfilter_secondary=use_dfilter_secondary,
    )

    assert int(offsets.n_state) == int(res.y.shape[0])
    assert int(offsets.off_ca) >= 0
    assert int(offsets.off_atp) > int(offsets.off_ca)
    assert int(offsets.off_na_i) > int(offsets.off_atp)
    assert int(offsets.off_k_o) > int(offsets.off_na_i)


def test_env_params_roundtrip_preserves_ordering():
    values = (
        310.15, 2.0, 5e-5, 120.0, 1.2, 10.0, 0.7,
        0.6, 0.5, 2.0, 0.18, 18.0, 145.0, 140.0, 4.5, 8e-5, 1600.0,
    )
    packed = build_env_params(*values)
    unpacked = unpack_env_params(packed)
    np.testing.assert_allclose(np.asarray(unpacked), np.asarray(values), rtol=0.0, atol=1e-12)


def test_native_compact_postprocess_toggle_keeps_voltage_trace():
    cfg = FullModelConfig()
    apply_preset(cfg, "A: Squid Giant Axon (HH 1952)")
    cfg.stim.t_sim = 120.0
    cfg.stim.dt_eval = 0.2
    cfg.stim.jacobian_mode = "native_hines"

    solver = NeuronSolver(cfg)
    res_full = solver.run_native(cfg, post_process=True)
    res_compact = solver.run_native(cfg, post_process=False)

    np.testing.assert_allclose(res_full.v_soma, res_compact.v_soma, rtol=0.0, atol=1e-9)
    assert isinstance(res_compact.currents, dict)
    assert len(res_compact.currents) == 0


def test_batch_validator_contract_returns_status_and_metrics():
    cfg_spiking = FullModelConfig()
    apply_preset(cfg_spiking, "A: Squid Giant Axon (HH 1952)")
    cfg_spiking.stim.t_sim = 220.0
    cfg_spiking.stim.dt_eval = 0.25
    cfg_spiking.stim.jacobian_mode = "native_hines"
    cfg_spiking.stim.Iext = 8.0

    cfg_silent = FullModelConfig()
    apply_preset(cfg_silent, "A: Squid Giant Axon (HH 1952)")
    cfg_silent.stim.t_sim = 220.0
    cfg_silent.stim.dt_eval = 0.25
    cfg_silent.stim.jacobian_mode = "native_hines"
    cfg_silent.stim.Iext = 0.0

    out = run_validation_batch(
        [cfg_spiking, cfg_silent],
        workers=1,
        compact_native=True,
        quick_prune_ms=180.0,
    )
    assert out.n_cases == 2
    assert len(out.rows) == 2
    assert sum(out.status_counts.values()) == 2

    valid_codes = {STATUS_OK, STATUS_SILENT, STATUS_D_BLOCK, STATUS_UNSTABLE}
    for row in out.rows:
        assert row["status_code"] in valid_codes
        assert "n_spikes" in row
        assert "v_peak_mV" in row
        assert "guard_ok" in row
