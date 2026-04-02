#!/usr/bin/env python3
"""Test synaptic stimulation: AMPA, NMDA, GABA-A, GABA-B"""

import sys
sys.path.insert(0, r'c:\NeuroModelPort')

from core.models import FullModelConfig
from core.presets import apply_preset, apply_synaptic_stimulus
from core.solver import NeuronSolver
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Start with L5 Pyramidal as base
synapses = [
    "SYN: AMPA-receptor (Fast Excitation, 1-3 ms)",
    "SYN: NMDA-receptor (Slow Excitation, 50-100 ms)",
    "SYN: GABA-A receptor (Fast Inhibition, 3-5 ms)",
    "SYN: GABA-B receptor (Slow Inhibition, 100-300 ms)",
    "SYN: Nicotinic ACh (Fast Excitation, 5-10 ms)",
]

print("=" * 80)
print("SYNAPTIC STIMULATION TEST (Applied to L5 Pyramidal)")
print("=" * 80)

for syn_name in synapses:
    cfg = FullModelConfig()
    apply_preset(cfg, "B: Pyramidal L5 (Mainen 1996)")
    apply_synaptic_stimulus(cfg, syn_name)

    try:
        solver = NeuronSolver(cfg)
        result = solver.run_single()

        V_soma = result.v_soma
        V_max = np.max(V_soma)
        V_min = np.min(V_soma)

        # Count peaks above threshold
        spike_threshold = -20
        spike_count = np.sum(V_soma > spike_threshold)

        # Estimate response type
        if spike_count > 10:
            response = "Strong spiking"
        elif spike_count > 0:
            response = "Weak/subthreshold"
        else:
            response = "No response"

        print(f"{syn_name:50s}")
        print(f"  V range: [{V_min:7.1f}, {V_max:7.1f}] mV | {response}")
        print()

    except Exception as e:
        print(f"{syn_name:50s}")
        print(f"  ERROR: {str(e)[:60]}")
        print()

print("=" * 80)
print("Note: GABA inputs show hyperpolarization (negative peak)")
print("      AMPA/NMDA show depolarization/spiking")
