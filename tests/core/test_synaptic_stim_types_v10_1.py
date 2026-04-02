import numpy as np

from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver


def _peak_abs(v: np.ndarray) -> float:
    return float(np.max(np.abs(v)))


def test_synaptic_stim_types_run_without_crash():
    """All new synaptic stim types should run in solver."""
    cfg = FullModelConfig()
    apply_preset(cfg, "B: Pyramidal L5 (Mainen 1996)")
    cfg.stim_location.location = "soma"
    cfg.stim.pulse_start = 10.0

    for stim_type in ["AMPA", "NMDA", "GABAA", "GABAB", "Kainate", "Nicotinic"]:
        cfg.stim.stim_type = stim_type
        cfg.stim.Iext = 20.0
        result = NeuronSolver(cfg).run_single()
        assert len(result.t) > 10
        assert np.isfinite(_peak_abs(result.v_soma))


def test_inhibitory_vs_excitatory_sign_effect():
    """
    Inhibitory GABA waveforms should push soma less depolarized than AMPA/NMDA
    when using the same magnitude.
    """
    cfg = FullModelConfig()
    apply_preset(cfg, "B: Pyramidal L5 (Mainen 1996)")
    cfg.stim_location.location = "soma"
    cfg.stim.pulse_start = 10.0
    cfg.stim.Iext = 20.0

    cfg.stim.stim_type = "AMPA"
    v_ampa = NeuronSolver(cfg).run_single().v_soma

    cfg.stim.stim_type = "GABAA"
    v_gabaa = NeuronSolver(cfg).run_single().v_soma

    assert float(np.max(v_ampa)) > float(np.max(v_gabaa))

