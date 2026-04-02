#!/usr/bin/env python3
"""Test all 15 presets for spike generation"""

import sys
sys.path.insert(0, r'c:\NeuroModelPort')

from core.models import FullModelConfig
from core.presets import get_preset_names, apply_preset
from core.solver import NeuronSolver
import numpy as np
import warnings
warnings.filterwarnings('ignore')

preset_names = get_preset_names()
results = []

print("=" * 80)
print("PRESET VALIDATION TEST SUITE")
print("=" * 80)

for preset_name in preset_names:
    cfg = FullModelConfig()
    apply_preset(cfg, preset_name)

    # Ensure const stim for consistent testing
    cfg.stim.stim_type = 'const'

    try:
        solver = NeuronSolver(cfg)
        result = solver.run_single()

        V_soma = result.v_soma
        V_max = np.max(V_soma)
        V_min = np.min(V_soma)
        spike_threshold = -20
        spike_count = np.sum(V_soma > spike_threshold)

        status = "PASS" if spike_count > 0 else "FAIL"
        results.append({
            'name': preset_name,
            'status': status,
            'V_max': V_max,
            'V_min': V_min,
            'spikes': spike_count
        })

        print(f"[{status}] {preset_name:40s} | V: [{V_min:7.1f}, {V_max:7.1f}] mV | Spikes: {spike_count:3d}")

    except Exception as e:
        results.append({
            'name': preset_name,
            'status': 'ERROR',
            'error': str(e)
        })
        print(f"[ERR] {preset_name:40s} | {str(e)[:50]}")

print("=" * 80)
pass_count = sum(1 for r in results if r['status'] == 'PASS')
fail_count = sum(1 for r in results if r['status'] == 'FAIL')
error_count = sum(1 for r in results if r['status'] == 'ERROR')
print(f"SUMMARY: {pass_count} PASS, {fail_count} FAIL, {error_count} ERROR")
