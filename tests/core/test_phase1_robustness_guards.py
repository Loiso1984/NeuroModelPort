from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def test_event_driven_conductance_caps_event_loop():
    from core.rhs import MAX_EVENTS_TO_PROCESS, get_event_driven_conductance

    events = np.zeros(MAX_EVENTS_TO_PROCESS + 500, dtype=np.float64)
    g_capped = get_event_driven_conductance(1.0, 4, 1.0, events, len(events), 1.0)
    g_limit = get_event_driven_conductance(1.0, 4, 1.0, events[:MAX_EVENTS_TO_PROCESS], MAX_EVENTS_TO_PROCESS, 1.0)
    assert g_capped == g_limit


def test_nernst_zero_valence_returns_safe_zero():
    from core.rhs import nernst_mono_ion

    assert nernst_mono_ion(1.0, 10.0, 0.0, 310.15) == 0.0


def test_tau_ca_extreme_value_emits_warning():
    from core.models import FullModelConfig
    from core.validation import validate_simulation_config

    cfg = FullModelConfig()
    cfg.calcium.dynamic_Ca = True
    cfg.calcium.tau_Ca = 0.05
    warnings = validate_simulation_config(cfg)
    assert any("tau_Ca" in warning for warning in warnings)


def test_lle_mask_rejects_out_of_bounds_offsets():
    from core.native_loop import make_lle_subspace_mask

    offsets = SimpleNamespace(
        n_state=4,
        off_m=3,
        off_h=-1,
        off_n=-1,
        off_r=-1,
        off_s=-1,
        off_u=-1,
        off_a=-1,
        off_b=-1,
        off_p=-1,
        off_q=-1,
        off_w=-1,
        off_x=-1,
        off_y=-1,
        off_j=-1,
        off_zsk=-1,
        off_ca=-1,
        off_atp=-1,
        off_na_i=-1,
        off_k_o=-1,
        off_ifilt_primary=-1,
        off_ifilt_secondary=-1,
    )
    import pytest
    with pytest.raises(ValueError):
        make_lle_subspace_mask(2, offsets, include_v=False, include_gates=["m"])


def test_config_manager_rolls_back_failed_preset(monkeypatch):
    import gui.config_manager as cm

    mgr = cm.ConfigManager()
    mgr.config.stim.Iext = 17.0
    mgr.mark_custom_config("before")

    def fail_apply(_cfg, _name):
        _cfg.stim.Iext = 999.0
        raise RuntimeError("boom")

    monkeypatch.setattr(cm, "apply_preset", fail_apply)
    assert mgr.load_preset("broken") is False
    assert mgr.config.stim.Iext == 17.0
    assert mgr.current_preset_name == "before"


def test_sparse_structure_signature_uses_compact_hash_not_large_tuples():
    import numpy as np
    from core.jacobian import _sparse_structure_signature

    indices = np.arange(100, dtype=np.int64)
    indptr = np.arange(20, dtype=np.int64)
    sig1 = _sparse_structure_signature(indices, indptr)
    sig2 = _sparse_structure_signature(indices.copy(), indptr.copy())

    assert sig1 == sig2
    assert len(sig1) == 4
    assert not any(isinstance(part, tuple) and len(part) > 10 for part in sig1)
