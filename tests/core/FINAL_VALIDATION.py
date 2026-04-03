"""
FINAL VALIDATION: All Core Phases Complete
Quick comprehensive check before cleanup
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset, get_preset_names
from core.solver import NeuronSolver

print("=" * 90)
print("FINAL VALIDATION: Phases 1-5 Complete")
print("=" * 90)

# Track results
results = {
    'phase1_calcium': None,
    'phase2_hcn': None,
    'phase3_ia': None,
    'phase4_all_presets': None,
    'phase5_multichannel': None,
}

# ============================================================================
# PHASE 1: Calcium Validation (Quick Check)
# ============================================================================
print("\n[PHASE 1] Calcium Channel Validation")
p1_pass = True

try:
    cfg = FullModelConfig()
    apply_preset(cfg, 'E: Cerebellar Purkinje (De Schutter)')
    
    # E preset has ICa enabled
    if cfg.channels.enable_ICa and cfg.calcium.dynamic_Ca:
        print("  ✓ Calcium setup: E preset has ICa + dynamic Ca")
        print(f"    gCa_max={cfg.channels.gCa_max}, tau_Ca={cfg.calcium.tau_Ca} ms")
        results['phase1_calcium'] = "PASS"
    else:
        print("  ⚠️  Calcium setup issue")
        p1_pass = False
except Exception as e:
    print(f"  ❌ ERROR: {e}")
    p1_pass = False

# ============================================================================
# PHASE 2: HCN Validation (Quick Check)
# ============================================================================
print("\n[PHASE 2] HCN Channel Validation")
p2_pass = True

try:
    cfg = FullModelConfig()
    apply_preset(cfg, 'K: Thalamic Relay (Ih + ICa + Burst)')
    
    if cfg.channels.enable_Ih:
        print("  ✓ HCN setup: K preset has Ih enabled")
        print(f"    gIh_max={cfg.channels.gIh_max} mS/cm², E_Ih={cfg.channels.E_Ih} mV")
        results['phase2_hcn'] = "PASS"
    else:
        print("  ⚠️  HCN not enabled on K preset")
        p2_pass = False
except Exception as e:
    print(f"  ❌ ERROR: {e}")
    p2_pass = False

# ============================================================================
# PHASE 3: IA Validation (Quick Test)
# ============================================================================
print("\n[PHASE 3] IA Channel Validation")
p3_pass = True

ia_presets = []
for p_name in get_preset_names():
    cfg = FullModelConfig()
    apply_preset(cfg, p_name)
    if cfg.channels.enable_IA:
        ia_presets.append((p_name, cfg.channels.gA_max))

if len(ia_presets) > 0:
    print(f"  ✓ IA enabled in {len(ia_presets)} presets:")
    for p_name, gA in ia_presets[:3]:  # Show first 3
        print(f"    - {p_name[:35]}: gA_max={gA}")
    if len(ia_presets) > 3:
        print(f"    ... and {len(ia_presets) - 3} more")
    
    # Quick functionality test
    try:
        cfg = FullModelConfig()
        apply_preset(cfg, ia_presets[0][0])  # Test first IA preset
        cfg.stim.Iext = 20.0
        cfg.stim.stim_type = 'const'
        
        solver = NeuronSolver(cfg)
        result = solver.run_single()
        if len(result.v_soma) > 0:
            print(f"  ✓ IA simulation: OK")
            results['phase3_ia'] = "PASS"
        else:
            p3_pass = False
    except Exception as e:
        print(f"  ❌ IA simulation error: {str(e)[:50]}")
        p3_pass = False
else:
    print("  ❌ No IA presets found")
    p3_pass = False

# ============================================================================
# PHASE 4: All Presets Functional
# ============================================================================
print("\n[PHASE 4] Preset Stress Test (Sample)")
p4_pass = True
p4_count = 0
p4_errors = []

sample_presets = ['A: Squid Giant Axon (HH 1952)', 
                  'C: FS Interneuron (Wang-Buzsaki)',
                  'E: Cerebellar Purkinje (De Schutter)',
                  'K: Thalamic Relay (Ih + ICa + Burst)']

for p_name in sample_presets:
    try:
        cfg = FullModelConfig()
        apply_preset(cfg, p_name)
        cfg.stim.stim_type = 'const'
        
        solver = NeuronSolver(cfg)
        result = solver.run_single()
        
        v_soma = result.v_soma
        rest = v_soma[0]
        peak = v_soma.max()
        
        if -120 < rest < -30 and -100 < peak < 80:
            p4_count += 1
        else:
            p4_errors.append(f"{p_name}: V_rest={rest:.1f}, V_peak={peak:.1f}")
    except Exception as e:
        p4_errors.append(f"{p_name}: {str(e)[:40]}")

print(f"  ✓ {p4_count}/{len(sample_presets)} sample presets functional")
if p4_errors:
    print("  ⚠️  Issues:")
    for err in p4_errors:
        print(f"    - {err}")
    p4_pass = False
else:
    p4_pass = True
    results['phase4_all_presets'] = "PASS"

# ============================================================================
# PHASE 5: Multi-Channel Integration
# ============================================================================
print("\n[PHASE 5] Multi-Channel Integration (Sample)")
p5_pass = True

try:
    cfg = FullModelConfig()
    apply_preset(cfg, 'K: Thalamic Relay (Ih + ICa + Burst)')
    
    channels = {
        'enable_IA': cfg.channels.enable_IA,
        'enable_ICa': cfg.channels.enable_ICa,
        'enable_Ih': cfg.channels.enable_Ih,
    }
    
    enabled_channels = [k.replace('enable_', '').upper() for k, v in channels.items() if v]
    
    print(f"  ✓ Multi-channel configuration: {'+'.join(enabled_channels)}")
    
    # Test it runs
    solver = NeuronSolver(cfg)
    result = solver.run_single()
    
    v_soma = result.v_soma
    if len(v_soma) > 0 and -120 < v_soma[0] < -30:
        print(f"  ✓ Multi-channel simulation: OK")
        results['phase5_multichannel'] = "PASS"
        p5_pass = True
    else:
        p5_pass = False
except Exception as e:
    print(f"  ❌ Multi-channel error: {str(e)[:50]}")
    p5_pass = False

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 90)
print("VALIDATION SUMMARY")
print("=" * 90)

phase_results = [
    ("Phase 1: Calcium Calibration", p1_pass),
    ("Phase 2: HCN Validation", p2_pass),
    ("Phase 3: IA Channel Validation", p3_pass),
    ("Phase 4: Preset Stress Testing", p4_pass),
    ("Phase 5: Multi-Channel Integration", p5_pass),
]

print()
for phase_name, passed in phase_results:
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {phase_name:40} | {status}")

total_pass = sum(1 for _, p in phase_results if p)
print()
print(f"OVERALL: {total_pass}/5 phases validated successfully")

if total_pass == 5:
    print("\n🎉 ALL PHASES READY FOR CLEANUP & NEXT STEPS")
    print("\nNext actions:")
    print("  1. Organize test files into /tests/phases/ structure")
    print("  2. Delete duplicate/incomplete test files")
    print("  3. Begin Phase 6: Enhanced Analysis Metrics")
else:
    print(f"\n⚠️  {5 - total_pass} phase(s) need attention before cleanup")
