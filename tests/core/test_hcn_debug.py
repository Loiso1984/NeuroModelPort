"""
HCN Debugging - Step by step
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("Step 1: Importing modules...")
from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver

print("Step 2: Creating config...")
cfg = FullModelConfig()

print("Step 3: Applying preset K...")
apply_preset(cfg, "K: Thalamic Relay (Ih + ICa + Burst)")

print("Step 4: Checking parameters...")
print(f"  enable_Ih: {cfg.channels.enable_Ih}")
print(f"  enable_ICa: {cfg.channels.enable_ICa}")
print(f"  gIh_max: {cfg.channels.gIh_max}")
print(f"  gCa_max: {cfg.channels.gCa_max}")

print("Step 5: Disabling ICa...")
cfg.channels.enable_ICa = False

print("Step 6: Setting zero stimulus...")
cfg.stim.Iext = 0.0
cfg.stim.stim_type = 'const'

print("Step 7: Creating solver...")
solver = NeuronSolver(cfg)

print("Step 8: Starting simulation...")
print("  (This may take a while...)")
sys.stdout.flush()

try:
    result = solver.run_single()
    print("Step 9: ✅ Simulation completed!")
    
    v_soma = result.v_soma
    print(f"  min V: {v_soma.min():.2f} mV")
    print(f"  max V: {v_soma.max():.2f} mV")
    print(f"  mean V (last 100): {v_soma[-100:].mean():.2f} mV")
    
except Exception as e:
    print(f"Step 9: ❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
