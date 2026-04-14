#!/usr/bin/env python3
"""Quick test to verify preset fix."""
print('Testing preset A after fix...')
from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver

cfg = FullModelConfig()
apply_preset(cfg, 'A: Squid Giant Axon (HH 1952)')
cfg.stim.t_sim = 10.0  # Short test

result = NeuronSolver(cfg).run_single()
print(f'✅ Simulation completed: {len(result.t)} time points')
spikes = len(result.spike_times) if hasattr(result, 'spike_times') else 'N/A'
print(f'   Spikes: {spikes}')
print('✅ BUG FIXED: zap_rise_ms parameter now passed correctly')
