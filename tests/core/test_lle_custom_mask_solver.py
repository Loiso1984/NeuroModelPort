from __future__ import annotations

import numpy as np


def test_solver_builds_custom_lle_mask_from_analysis_params(monkeypatch):
    import core.native_loop as native_loop
    from core.models import FullModelConfig
    from core.solver import NeuronSolver

    captured = {}

    def fake_loop(*args):
        n_state = int(args[0].shape[0])
        captured["mask"] = args[-2]
        return (
            np.array([0.0], dtype=np.float64),
            np.zeros((n_state, 1), dtype=np.float64),
            False,
            np.array([0.0], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
        )

    monkeypatch.setattr(native_loop, "run_native_loop", fake_loop)

    cfg = FullModelConfig()
    cfg.morphology.single_comp = True
    cfg.stim.jacobian_mode = "native_hines"
    cfg.stim.t_sim = 0.01
    cfg.analysis.lle_subspace = "Custom"
    cfg.analysis.lle_custom_v = True
    cfg.analysis.lle_custom_gates = "m"
    cfg.analysis.lle_custom_ca = False
    cfg.analysis.lle_custom_atp = False

    NeuronSolver(cfg).run_native(cfg, calc_lle=True, lle_subspace_mode="Custom")

    mask = captured["mask"]
    assert mask is not None
    assert mask.dtype == np.bool_
    assert bool(mask[0]) is True
    assert np.count_nonzero(mask) >= 2  # V + m for one compartment
