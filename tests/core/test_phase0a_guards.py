from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError


def test_calcium_external_concentration_rejects_zero_on_create_and_assignment():
    from core.models import FullModelConfig

    with pytest.raises(ValidationError):
        FullModelConfig(calcium={"Ca_ext": 0.0})

    cfg = FullModelConfig()
    with pytest.raises(ValidationError):
        cfg.calcium.Ca_ext = 0.0


def test_nernst_helpers_guard_zero_external_calcium_and_zero_valence():
    from core.rhs import nernst_ca_ion, nernst_mono_ion

    e_ca = float(nernst_ca_ion(50e-6, 0.0, 310.15))
    e_mono = float(nernst_mono_ion(1.0, 10.0, 0.0, 310.15))

    assert math.isfinite(e_ca)
    assert math.isfinite(e_mono)


def test_plot_downsample_ignores_nan_for_1d_and_2d_traces():
    from gui.plots import _downsample_xy

    t = np.linspace(0.0, 10.0, 1000)
    y = np.sin(t)
    y[100:120] = np.nan
    td, yd = _downsample_xy(t, y, max_points=100)
    assert len(td) == len(yd)
    assert np.isfinite(yd[:-1]).any()

    y2 = np.vstack([np.sin(t), np.cos(t)])
    y2[:, 200:230] = np.nan
    td2, yd2 = _downsample_xy(t, y2, max_points=100)
    assert yd2.shape[1] == len(td2)
    assert np.isfinite(yd2[:, :-1]).any()


def test_simulation_controller_stochastic_does_not_mutate_input_config(monkeypatch):
    from types import SimpleNamespace

    from core.models import FullModelConfig
    from gui.simulation_controller import SimulationController
    import core.analysis
    import core.morphology
    import core.solver

    seen = {}

    class InlinePool:
        def setMaxThreadCount(self, _n):
            pass

        def start(self, worker):
            worker.run()

    class DummySolver:
        def __init__(self, cfg):
            seen["cfg"] = cfg

        def run_single(self):
            return SimpleNamespace(config=seen["cfg"])

        def _post_process_physics(self, _result, _morph):
            pass

    monkeypatch.setattr(core.solver, "NeuronSolver", DummySolver)
    monkeypatch.setattr(core.morphology.MorphologyBuilder, "build", lambda cfg: {"cfg": cfg})
    monkeypatch.setattr(core.analysis, "full_analysis", lambda result, compute_lyapunov=False: {"ok": True})

    cfg = FullModelConfig()
    cfg.stim.stoch_gating = False
    controller = SimulationController()
    controller.thread_pool = InlinePool()
    payloads = []

    controller.run_stochastic(cfg, 1, on_success=payloads.append)

    assert cfg.stim.stoch_gating is False
    assert seen["cfg"] is not cfg
    assert seen["cfg"].stim.stoch_gating is True
    assert payloads and payloads[0]["stats"] == {"ok": True}
