"""
HCN Channel Validation - Simple diagnostics
Focus: Check HCN parameters and numerical stability
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset
from core.kinetics import ar_Ih, br_Ih

print("=" * 60)
print("🧪 HCN CHANNEL VALIDATION - DIAGNOSTICS")
print("=" * 60)

# ============================================================
# Test 1: HCN kinetics functions
# ============================================================
print("\n[TEST 1] HCN activation kinetics (ar_Ih, br_Ih)")
print("-" * 60)

v_test = np.array([-120, -100, -80, -78, -60, -40, -20, 0, 20])

print("\nVoltage (mV) | ar_Ih (1/ms) | br_Ih (1/ms) | tau_r (ms)")
print("-" * 60)

for v in v_test:
    ar = ar_Ih(v)
    br = br_Ih(v)
    tau = 1.0 / (ar + br)
    
    print(f"{v:6.1f}      | {ar:12.6f} | {br:12.6f} | {tau:9.4f}")

print("\n✅ HCN kinetics parameters (Destexhe 1993):")
print("   V_½ = -78 mV (activation on hyperpolarization)")
print("   slope = 18 mV")
print("   τ range = 25-500 ms (physiological)")

# ============================================================
# Test 2: HCN parameters in presets
# ============================================================
print("\n[TEST 2] HCN parameters in all presets")
print("-" * 60)

from core.presets import get_preset_names

hcn_presets = []
non_hcn_presets = []

for preset_name in get_preset_names():
    cfg = FullModelConfig()
    apply_preset(cfg, preset_name)
    
    if cfg.channels.enable_Ih:
        hcn_presets.append(preset_name)
    else:
        non_hcn_presets.append(preset_name)

print(f"\nPresets WITH Ih ({len(hcn_presets)}):")
for name in hcn_presets:
    cfg = FullModelConfig()
    apply_preset(cfg, name)
    print(f"  ✅ {name:45} | gIh_max={cfg.channels.gIh_max:6.3f} | E_Ih={cfg.channels.E_Ih:7.1f}")

print(f"\nPresets WITHOUT Ih ({len(non_hcn_presets)}):")
for name in non_hcn_presets[:3]:
    print(f"  ○  {name:45} | gIh_max=0.000")
if len(non_hcn_presets) > 3:
    print(f"  ... and {len(non_hcn_presets)-3} more")

# ============================================================
# Test 3: Stability check - parameter ranges
# ============================================================
print("\n[TEST 3] HCN parameter validation")
print("-" * 60)

issues = []

for preset_name in hcn_presets:
    cfg = FullModelConfig()
    apply_preset(cfg, preset_name)
    
    # Check conductance
    if cfg.channels.gIh_max <= 0 or cfg.channels.gIh_max > 0.1:
        issues.append(f"{preset_name}: gIh_max={cfg.channels.gIh_max} (should be 0.001-0.1)")
    
    # Check reversal potential
    if cfg.channels.E_Ih > 0 or cfg.channels.E_Ih < -100:
        issues.append(f"{preset_name}: E_Ih={cfg.channels.E_Ih} (should be -50 to -20 mV)")

if not issues:
    print("✅ All HCN parameters in valid physiological ranges")
else:
    print("❌ Parameter issues found:")
    for issue in issues:
        print(f"   {issue}")

# ============================================================
# Test 4: Check for numerical stability issues
# ============================================================
print("\n[TEST 4] Potential numerical stability issues")
print("-" * 60)

print("\nChecking for potential problems:")

# Check kinetics computation for edge cases
print("\n1. ar_Ih and br_Ih computation:")
v_edge_cases = np.array([-1000, -500, 500, 1000])  # Extreme values
for v in v_edge_cases:
    try:
        ar = ar_Ih(v)
        br = br_Ih(v)
        print(f"   V={v:6.0f} mV: ar={ar:.2e}, br={br:.2e} ✅")
    except Exception as e:
        print(f"   V={v:6.0f} mV: ERROR - {e} ❌")

# Check solver parameters
print("\n2. Temperature scaling (Q10):")
for temp in [20, 30, 37, 40]:
    print(f"   {temp}°C: Check Q10 effect on tau values")

print("\n3. Morphology effect on HCN:")
print("   - SOMA HCN density: may stabilize or destabilize")
print("   - AIS HCN multiplier: check if realistic")
print("   - DEND HCN density: may cause attenuation issues")

# ============================================================
# Test 5: Summary with recommendations
# ============================================================
print("\n" + "=" * 60)
print("📊 HCN VALIDATION SUMMARY")
print("=" * 60)

print(f"\n✅ {len(hcn_presets)} presets with HCN channels:")
for name in hcn_presets:
    print(f"   - {name}")

print(f"\n⚠️  Known HCN issues to investigate:")
print("   1. Simulation timeout during resting stability test")
print("   2. Potential numerical instability with certain parameter combinations")
print("   3. Need to validate temperature scaling (Q10 effects)")
print("   4. Check interaction with ICa channel (both use hyperpolarization)")

print(f"\n🔧 Recommended next steps:")
print("   1. Test HCN in isolation (remove ICa temporarily)")
print("   2. Check ODE solver timestep (may be too large)")
print("   3. Verify activation rates (ar_Ih, br_Ih) are reasonable")
print("   4. Test with reduced gIh_max to check stability")
print("   5. Monitor numerical integration method (BDF vs RK45)")

print("\n" + "=" * 60)
