"""
Quick IA Validation Summary
Verify all IA-enabled presets work correctly after changes
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset, get_preset_names
from core.solver import NeuronSolver

print("=" * 80)
print("IA CHANNEL VALIDATION SUMMARY (Post-Update)")
print("=" * 80)

ia_presets = []
for preset_name in get_preset_names():
    cfg = FullModelConfig()
    apply_preset(cfg, preset_name)
    if cfg.channels.enable_IA:
        ia_presets.append((preset_name, cfg.channels.gA_max))

print(f"\nTotal presets with IA enabled: {len(ia_presets)}")
print()
print(f"{'Preset Name':45} | {'gA_max':10} | {'Excitable?':15} | {'Status':10}")
print("-" * 85)

success_count = 0
for preset_name, gA_max in ia_presets:
    cfg = FullModelConfig()
    apply_preset(cfg, preset_name)
    
    try:
        solver = NeuronSolver(cfg)
        result = solver.run_single()
        
        v_soma = result.v_soma
        rest = v_soma[0]
        peak = v_soma.max()
        spike_threshold = rest + 30
        spikes = (v_soma > spike_threshold).sum()
        
        is_excitable = "✓ YES" if spikes > 0 else "✗ NO"
        status = "✓ OK" if spikes > 0 else "⚠ CHECK"
        
        print(f"{preset_name:45} | {gA_max:10.2f} | {is_excitable:15} | {status:10}")
        if spikes > 0:
            success_count += 1
        
    except Exception as e:
        print(f"{preset_name:45} | {gA_max:10.2f} | {'ERROR':15} | {'✗':10}")

print()
print(f"Result: {success_count}/{len(ia_presets)} IA-enabled presets working correctly")

if success_count == len(ia_presets):
    print("\n✓ ALL IA PRESETS VALIDATED SUCCESSFULLY")
else:
    print(f"\n⚠ {len(ia_presets) - success_count} presets need attention")
