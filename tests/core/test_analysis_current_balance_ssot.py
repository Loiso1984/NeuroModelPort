"""Regression tests for current-balance stimulus source of truth."""

from __future__ import annotations

import numpy as np

from core.analysis import compute_current_balance
from core.models import FullModelConfig


class DummyResult:
    pass


def test_current_balance_uses_solver_i_stim_total_without_reconstruction(monkeypatch):
    """compute_current_balance must not reconstruct stimulus when solver output exists."""
    import core.rhs

    def fail_get_stim_current(*_args, **_kwargs):
        raise AssertionError("stimulus reconstruction should not be called")

    monkeypatch.setattr(core.rhs, "get_stim_current", fail_get_stim_current)

    cfg = FullModelConfig()
    cfg.stim.stim_type = "pulse"
    cfg.channels.Cm = 1.0

    result = DummyResult()
    result.t = np.array([0.0, 1.0, 2.0], dtype=float)
    result.v_soma = np.array([-65.0, -64.0, -63.0], dtype=float)
    result.config = cfg
    result.currents = {}
    result.n_comp = 1
    result.i_stim_total = np.array([0.0, 1.0, 1.0], dtype=float)

    balance = compute_current_balance(result, morph={})

    np.testing.assert_allclose(balance, np.array([0.0, 0.0, 0.0]), atol=1e-12)
