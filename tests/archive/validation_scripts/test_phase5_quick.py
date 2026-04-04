"""
PHASE 5-QUICK: Fast Multi-Channel Validation
Key presets only, essential tests only
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver

print("=" * 90)
print("PHASE 5-QUICK: Fast Channel Integration Test")
print("=" * 90)

test_cases = [
    # (preset_name, channels_to_test, stimulus_param)
    ('A: Squid Giant Axon (HH 1952)', 'baseline', 10.0),
    ('C: FS Interneuron (Wang-Buzsaki)', 'IA enabled', 40.0),
    ('E: Cerebellar Purkinje (De Schutter)', 'IA+ICa', 30.0),
    ('K: Thalamic Relay (Ih + ICa + Burst)', 'Ih+ICa', 30.0),
]

print(f"\n{'Preset':45} | {'Config':20} | {'Rest (mV)':12} | {'Peak (mV)':12} | {'Spikes':8} | {'Status':8}")
print("-" * 100)

passed = 0
total = len(test_cases)

for preset_name, config_desc, default_iext in test_cases:
    cfg = FullModelConfig()
    apply_preset(cfg, preset_name)
    cfg.stim.Iext = default_iext
    cfg.stim.stim_type = 'const'
    
    try:
        solver = NeuronSolver(cfg)
        result = solver.run_single()
        
        v_soma = result.v_soma
        rest = v_soma[0]
        peak = v_soma.max()
        spike_threshold = rest + 30
        spikes = (v_soma > spike_threshold).sum()
        
        # Validation
        if -120 < rest < -30 and -100 < peak < 80 and spikes > 0:
            status = "✓ OK"
            passed += 1
        else:
            status = "⚠ WARN"
        
        print(f"{preset_name:45} | {config_desc:20} | {rest:12.1f} | {peak:12.1f} | {spikes:8d} | {status:8}")
        
    except Exception as e:
        print(f"{preset_name:45} | {config_desc:20} | {'ERROR':>12} | {'ERROR':>12} | {'ERR':>8} | {'✗':8}")

print()
print(f"Quick Validation: {passed}/{total} essential tests passed")

if passed == total:
    print("\n✓ READY FOR NEXT PHASE (Analysis & Visualization)")
    print("\nCompleted phases:")
    print("  ✓ Phase 1: Calcium calibration")
    print("  ✓ Phase 2: HCN validation")  
    print("  ✓ Phase 3: IA channel validation")
    print("  ✓ Phase 4: Preset stress testing (config audit complete)")
    print("  ✓ Phase 5-Quick: Multi-channel integration verified")
    print("\nNext: Phase 6 - Analysis metrics, visualization, GUI improvements")
else:
    print(f"\n⚠ Issues found - {total - passed} tests failed, needs investigation")
