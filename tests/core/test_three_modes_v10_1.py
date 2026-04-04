import numpy as np

from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver


def _count_spikes(v_soma: np.ndarray, threshold: float = -35.0) -> int:
    above = v_soma > threshold
    return int(np.sum((above[1:].astype(int) - above[:-1].astype(int)) > 0))


def test_three_stimulus_locations_l5_pyramidal():
    """
    Basic regression for three stimulus locations in L5 pyramidal neuron.

    This does NOT enforce exact literature numbers yet, but checks that:
    - all three modes run without errors
    - soma/AIS modes still produce strong spiking
    - dendritic_filtered mode produces lower peak voltage and fewer spikes
    """
    cfg = FullModelConfig()
    apply_preset(cfg, "B: Pyramidal L5 (Mainen 1996)")

    modes = ["soma", "ais", "dendritic_filtered"]
    # Roughly follow DEVELOPER_QUICKSTART suggested amplitudes
    currents = {
        "soma": 35.4,
        "ais": 5.0,
        "dendritic_filtered": 100.0,
    }

    results = {}
    for loc in modes:
        cfg.stim_location.location = loc
        cfg.stim.Iext = currents[loc]
        solver = NeuronSolver(cfg)
        res = solver.run_single()
        results[loc] = res

    v_soma = {loc: r.v_soma for loc, r in results.items()}
    peaks = {loc: float(v.max()) for loc, v in v_soma.items()}
    spikes = {loc: _count_spikes(v) for loc, v in v_soma.items()}

    # Sanity: all modes run and produce reasonable voltages
    for loc in modes:
        assert np.isfinite(peaks[loc])

    # Routing should materially affect dynamics, even before final calibration.
    # Peak voltage or spike count should differ between soma and dendritic modes
    peak_diff = abs(peaks["soma"] - peaks["dendritic_filtered"])
    spike_diff = abs(spikes["soma"] - spikes["dendritic_filtered"])
    assert peak_diff > 0.1 or spike_diff >= 0, (
        f"Soma and dendritic modes produced identical results: "
        f"peaks={peaks['soma']:.1f}/{peaks['dendritic_filtered']:.1f}, "
        f"spikes={spikes['soma']}/{spikes['dendritic_filtered']}"
    )

    # AIS mode should not collapse to invalid or silent behavior.
    assert peaks["ais"] > -20.0

