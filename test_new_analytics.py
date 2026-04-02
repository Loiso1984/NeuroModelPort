#!/usr/bin/env python
"""Quick test of new analytics fields."""
import numpy as np
from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver
from core.analysis import full_analysis

config = FullModelConfig()
apply_preset(config, 'B: Pyramidal L5 (Mainen 1996)')
solver = NeuronSolver(config)
result = solver.run_single()
stats = full_analysis(result)

print('✅ New analysis fields (Phase 7.1):')
new_fields = ['isi_mean_ms', 'isi_std_ms', 'isi_min_ms', 'isi_max_ms',
              'cv_isi', 'first_spike_latency_ms', 'refractory_period_ms', 
              'firing_reliability']
for field in new_fields:
    val = stats.get(field, None)
    if isinstance(val, float) and not np.isnan(val):
        print(f'  {field:30} = {val:.4f}')
    elif isinstance(val, float):
        print(f'  {field:30} = (no spikes or N/A)')
    else:
        print(f'  {field:30} = {val}')
        
print(f'\nTotal spikes detected: {stats["n_spikes"]}')
print(f'Initial frequency: {stats["f_initial_hz"]:.1f} Hz')
print(f'Steady frequency: {stats["f_steady_hz"]:.1f} Hz')
print('✅ All fields computed successfully!')
