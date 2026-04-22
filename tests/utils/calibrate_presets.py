#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Automatic preset calibrator - finds optimal Iext for target spike frequency"""

import sys
sys.path.insert(0, r'c:\NeuroModelPort')

from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def calibrate_iext(preset_name, target_freq_hz, iext_min=0.1, iext_max=100.0, tolerance=5.0):
    """Binary search for Iext that gives target spike frequency"""

    for iteration in range(20):  # Max 20 iterations
        iext_test = (iext_min + iext_max) / 2.0

        cfg = FullModelConfig()
        apply_preset(cfg, preset_name)
        cfg.stim.Iext = iext_test
        cfg.stim.stim_type = 'const'

        try:
            solver = NeuronSolver(cfg)
            result = solver.run_single()

            V_soma = result.v_soma
            spike_count = np.sum(V_soma > -20)
            actual_freq = spike_count / (result.t[-1] / 1000.0)

            error = abs(actual_freq - target_freq_hz)

            print(f"  Iter {iteration+1:2d}: Iext={iext_test:6.3f} → Freq={actual_freq:7.1f} Hz (target={target_freq_hz} Hz, error={error:6.1f})")

            if error < tolerance:
                print(f"  [OK] CONVERGED at Iext={iext_test:.3f} uA/cm²")
                return iext_test

            # Adjust bounds
            if actual_freq < target_freq_hz:
                iext_min = iext_test
            else:
                iext_max = iext_test

        except Exception as e:
            print(f"  [ERROR] at Iext={iext_test}: {str(e)[:40]}")
            return None

    print(f"  [WARNING] Did not converge (error={error:.1f} Hz)")
    return iext_test

# Calibration targets (from BIOPHYSICAL_REFERENCE.md)
targets = [
    ("A: Squid Giant Axon (HH 1952)", 15),
    ("B: Pyramidal L5 (Mainen 1996)", 15),
    ("C: FS Interneuron (Wang-Buzsaki)", 50),
    ("E: Cerebellar Purkinje (De Schutter)", 75),
    ("K: Thalamic Relay (Ih + ITCa + Burst)", 10),
]

print("=" * 80)
print("AUTOMATIC PRESET CALIBRATOR")
print("=" * 80)

for preset_name, target_freq in targets:
    print(f"\n{preset_name}")
    print(f"Target: {target_freq} Hz")
    iext_optimal = calibrate_iext(preset_name, target_freq, tolerance=5.0)
    if iext_optimal is not None:
        print(f"→ Recommended: cfg.stim.Iext = {iext_optimal:.2f}")

print("\n" + "=" * 80)
