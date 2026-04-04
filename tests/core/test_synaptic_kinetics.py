"""Test: Synaptic stimulation types produce expected responses."""

import pytest
import numpy as np
from core.models import FullModelConfig
from core.presets import apply_preset, apply_synaptic_stimulus
from core.solver import NeuronSolver


EXCITATORY_SYNAPSES = [
    "SYN: AMPA-receptor (Fast Excitation, 1-3 ms)",
    "SYN: NMDA-receptor (Slow Excitation, 50-100 ms)",
    "SYN: Nicotinic ACh (Fast Excitation, 5-10 ms)",
]

INHIBITORY_SYNAPSES = [
    "SYN: GABA-A receptor (Fast Inhibition, 3-5 ms)",
    "SYN: GABA-B receptor (Slow Inhibition, 100-300 ms)",
]


@pytest.mark.parametrize("syn_name", EXCITATORY_SYNAPSES)
def test_excitatory_synapse(syn_name):
    """Excitatory synapses on L5 should depolarize the membrane."""
    cfg = FullModelConfig()
    apply_preset(cfg, "B: Pyramidal L5 (Mainen 1996)")
    apply_synaptic_stimulus(cfg, syn_name)
    cfg.stim.t_sim = 50.0

    solver = NeuronSolver(cfg)
    result = solver.run_single()

    v_max = result.v_soma.max()
    assert v_max > -70, f"{syn_name}: expected depolarization, got V_max={v_max:.1f} mV"


@pytest.mark.parametrize("syn_name", INHIBITORY_SYNAPSES)
def test_inhibitory_synapse_reduces_excitability(syn_name):
    """Inhibitory synapses should produce less depolarization than excitatory."""
    # Baseline: L5 with AMPA
    cfg_ampa = FullModelConfig()
    apply_preset(cfg_ampa, "B: Pyramidal L5 (Mainen 1996)")
    apply_synaptic_stimulus(cfg_ampa, "SYN: AMPA-receptor (Fast Excitation, 1-3 ms)")
    cfg_ampa.stim.t_sim = 50.0
    r_ampa = NeuronSolver(cfg_ampa).run_single()

    # Test: L5 with inhibitory synapse (no Iext)
    cfg = FullModelConfig()
    apply_preset(cfg, "B: Pyramidal L5 (Mainen 1996)")
    apply_synaptic_stimulus(cfg, syn_name)
    cfg.stim.t_sim = 50.0
    cfg.stim.Iext = 0.0

    result = NeuronSolver(cfg).run_single()

    assert np.all(np.isfinite(result.v_soma)), f"{syn_name}: NaN/Inf in output"
    # Inhibitory peak should be lower than excitatory peak
    assert result.v_soma.max() <= r_ampa.v_soma.max(), (
        f"{syn_name}: inhibitory V_max={result.v_soma.max():.1f} > "
        f"AMPA V_max={r_ampa.v_soma.max():.1f}"
    )
