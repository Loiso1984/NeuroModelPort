#!/usr/bin/env python3
"""Test key presets: Squid, L5, FS, Purkinje"""

import sys
sys.path.insert(0, r'c:\NeuroModelPort')

from core.models import FullModelConfig
from core.presets import get_preset_names, apply_preset
from core.solver import NeuronSolver
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Test only key presets
key_presets = [
    "A: Squid Giant Axon (HH 1952)",
    "B: Pyramidal L5 (Mainen 1996)",
    "C: FS Interneuron (Wang-Buzsaki)",
    "E: Cerebellar Purkinje (De Schutter)",
    "K: Thalamic Relay (Ih + ICa + Burst)",
]

print("=" * 80)
print("KEY PRESET VALIDATION TEST")
print("=" * 80)

for preset_name in key_presets:
    cfg = FullModelConfig()
    apply_preset(cfg, preset_name)

    # Switch to const for testing
    cfg.stim.stim_type = 'const'

    try:
        solver = NeuronSolver(cfg)
        result = solver.run_single()

        V_soma = result.v_soma
        V_max = np.max(V_soma)
        V_min = np.min(V_soma)
        spike_threshold = -20
        spike_count = np.sum(V_soma > spike_threshold)

        status = "PASS" if spike_count > 5 else "NEEDS_TUNING"
        freq_hz = spike_count / (result.t[-1] / 1000.0)

        print(f"[{status:12s}] {preset_name:40s} | V: [{V_min:7.1f}, {V_max:7.1f}] | Freq: {freq_hz:5.1f} Hz")

    except Exception as e:
        print(f"[ERROR        ] {preset_name:40s} | {str(e)[:60]}")

print("=" * 80)
