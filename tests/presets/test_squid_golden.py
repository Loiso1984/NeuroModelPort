#!/usr/bin/env python3
"""Quick spike test for Squid"""

import sys
sys.path.insert(0, r'c:\NeuroModelPort')

from core.models import FullModelConfig
from core.presets import apply_preset
from core.morphology import MorphologyBuilder
from core.solver import NeuronSolver
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Test Squid preset
cfg = FullModelConfig()
apply_preset(cfg, "A: Squid Giant Axon (HH 1952)")
# Switch to const stimulus for clarity
cfg.stim.stim_type = 'const'

print("=" * 60)
print("Testing: Squid Giant Axon (HH 1952)")
print("=" * 60)
print(f"Soma diameter: {cfg.morphology.d_soma*1e4:.2f} um")
print(f"gNa_max: {cfg.channels.gNa_max} mS/cm2")
print(f"gK_max: {cfg.channels.gK_max} mS/cm2")
print(f"gL: {cfg.channels.gL} mS/cm2")
print(f"Iext: {cfg.stim.Iext} uA/cm2")
print(f"Stim type: {cfg.stim.stim_type}")
print()

# Build morphology
morph = MorphologyBuilder.build(cfg)
print(f"N_comp: {morph['N_comp']}")
print(f"Soma area: {morph['areas'][0]:.2e} cm2")
print()

# Run simulation
try:
    solver = NeuronSolver(cfg)
    result = solver.run_single()
    print(f"Simulation completed: {len(result.t)} timepoints ({result.t[-1]:.1f} ms)")
    print(f"V range: [{np.min(result.v_soma):.1f}, {np.max(result.v_soma):.1f}] mV")
    print(f"V[soma, 0]: {result.v_soma[0]:.2f} mV")
    print(f"V[soma, -1]: {result.v_soma[-1]:.2f} mV")

    # Check for spikes
    V_soma = result.v_soma
    spike_threshold = -20
    spikes = np.sum(V_soma > spike_threshold)
    if spikes > 0:
        print(f"\n[OK] SUCCESS: Generated {spikes} samples above {spike_threshold} mV")
    else:
        print(f"\n[FAIL] No spikes generated (max V = {np.max(V_soma):.1f} mV)")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
