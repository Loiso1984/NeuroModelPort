#!/usr/bin/env python3
"""Test corrected Purkinje and Thalamic"""

import sys
sys.path.insert(0, r'c:\NeuroModelPort')

from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver
import numpy as np
import warnings
warnings.filterwarnings('ignore')

tests = [
    ("E: Cerebellar Purkinje (De Schutter)", "50-100 Hz expected"),
    ("K: Thalamic Relay (Ih + ICa + Burst)", "1-50 Hz expected (relay mode)"),
]

for preset_name, note in tests:
    cfg = FullModelConfig()
    apply_preset(cfg, preset_name)
    cfg.stim.stim_type = 'const'

    solver = NeuronSolver(cfg)
    result = solver.run_single()

    V_soma = result.v_soma
    V_max = np.max(V_soma)
    V_min = np.min(V_soma)
    spike_count = np.sum(V_soma > -20)
    freq_hz = spike_count / (result.t[-1] / 1000.0)

    print(f"{preset_name}")
    print(f"  V: [{V_min:7.1f}, {V_max:7.1f}] mV | Freq: {freq_hz:6.1f} Hz | {note}")
    print()
