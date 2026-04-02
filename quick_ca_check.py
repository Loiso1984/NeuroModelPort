# Quick calcium current check
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver

cfg = FullModelConfig()
apply_preset(cfg, "K: Thalamic Relay (Ih + ICa + Burst)")
solver = NeuronSolver(cfg)
result = solver.run_single()

print(f"V_max: {result.v_soma.max():.1f} mV")
print(f"Ca range: {result.ca_i[0,:].min():.2e} to {result.ca_i[0,:].max():.2e} M")
print(f"Ca change: {(result.ca_i[0,:].max() - result.ca_i[0,:].min()):.2e} M")
