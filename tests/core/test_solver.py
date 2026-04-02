#!/usr/bin/env python3
"""Test script: Load L5 config and run minimal simulation."""

import numpy as np
from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver

# Create base config and apply L5 preset
cfg = FullModelConfig()
apply_preset(cfg, "Pyramidal L5 (Mainen 1996)")
print(f"Loaded L5 config")
print(f"  Simulation time: {cfg.stim.t_sim} ms")
print(f"  Temperature: {cfg.env.T_celsius}°C")
print(f"  Soma diameter: {cfg.morphology.d_soma*1e4:.1f} µm")

# Create solver and run
solver = NeuronSolver(cfg)
print("Solver created")

try:
    print("Running simulation...")
    result = solver.run_single()
    print(f"✓ Simulation completed successfully!")
    print(f"  Time points: {len(result.t)}")
    print(f"  V_soma range: {result.v_soma.min():.2f} to {result.v_soma.max():.2f} mV")
    if result.ca_i is not None:
        print(f"  [Ca]_i range: {result.ca_i.min():.2e} to {result.ca_i.max():.2e} M")
except Exception as e:
    print(f"✗ Simulation failed: {e}")
    import traceback
    traceback.print_exc()
