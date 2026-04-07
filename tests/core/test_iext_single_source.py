import numpy as np

from core.models import FullModelConfig, SimulationParams


def test_iext_absolute_is_derived_on_full_config():
    cfg = FullModelConfig()
    cfg.morphology.d_soma = 20e-4
    cfg.stim.Iext = 10.0

    area = np.pi * float(cfg.morphology.d_soma) ** 2
    expected = 10.0 * area * 1000.0
    assert abs(cfg.Iext_absolute_nA - expected) < 1e-12


def test_simulation_params_no_mutable_iext_absolute_field():
    assert "Iext_absolute_nA" not in SimulationParams.model_fields
