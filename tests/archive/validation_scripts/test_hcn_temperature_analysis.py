"""
HCN Temperature Compensation Analysis
Check phi_T effects on kinetics timescales
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset
from core.kinetics import ar_Ih, br_Ih

print("=" * 70)
print("🧪 HCN KINETICS WITH TEMPERATURE COMPENSATION")
print("=" * 70)

# Test temperature compensation
cfg = FullModelConfig()
apply_preset(cfg, "K: Thalamic Relay (Ih + ICa + Burst)")

T_celsius = cfg.env.T_celsius
T_ref = cfg.env.T_ref if hasattr(cfg.env, 'T_ref') else 37.0

# Q10 for HCN (typical value around 3.0)
Q10 = 3.0
phi_T = Q10 ** ((T_celsius - T_ref) / 10.0)

print(f"\nTemperature settings:")
print(f"  T_celsius: {T_celsius}°C")
print(f"  T_ref: {T_ref}°C")
print(f"  Q10: {Q10}")
print(f"  phi_T: {phi_T:.4f}")

print(f"\nHCN tau values WITHOUT temperature compensation:")
print(f"  V (mV) | ar_Ih (1/ms) | br_Ih (1/ms) | tau_r (ms)")
print(f"  {'-'*55}")

v_test = [-120, -100, -80, -60, -40, -20, -10, 0]

tau_values = []
for v in v_test:
    ar = ar_Ih(v)
    br = br_Ih(v)
    tau = 1.0 / (ar + br)
    tau_values.append((v, tau))
    print(f"  {v:6.0f} | {ar:12.6f} | {br:12.6f} | {tau:10.2f}")

print(f"\nHCN tau values WITH temperature compensation (φ_T={phi_T:.4f}):")
print(f"  V (mV) | tau_r (ms)   | tau_r/phi_T (ms)")
print(f"  {'-'*55}")

for v, tau in tau_values:
    tau_compensated = tau / phi_T
    print(f"  {v:6.0f} | {tau:12.2f} | {tau_compensated:15.2f}")

print(f"\n💡 Analysis:")
print(f"  - At room temp (23°C), tau is very large (>100ms)")
print(f"  - With phi_T={phi_T:.4f}, tau is reduced to manageable levels")
print(f"  - Fastest tau_r at V=-120mV: {min([t[1]/phi_T for t in tau_values]):.2f} ms")
print(f"  - Slowest tau_r at V=-10mV: {max([t[1]/phi_T for t in tau_values]):.2f} ms")

# Estimate required timestep
print(f"\n⚠️  ODE Solver considerations:")
tau_min = min([t[1]/phi_T for t in tau_values])
dt_recommended = tau_min / 10.0  # Standard 10x oversample
print(f"  - Minimum tau: {tau_min:.3f} ms")
print(f"  - Recommended dt: {dt_recommended:.4f} ms")
print(f"  - For 1 second sim with dt={dt_recommended:.4f}: {int(1000/dt_recommended):,} steps")
print(f"  - This is numerically challenging!")

print(f"\n🔧 Potential solutions:")
print(f"  1. Increase temperature to reduce tau (use 37°C)")
print(f"  2. Reduce gIh_max to ~0.01 (less driving force needed)")
print(f"  3. Use adaptive timestep ODE solver")
print(f"  4. Review Destexhe 1993 HCN kinetics (may be too slow)")
print(f"  5. Use temperature-adjusted kinetics constants")

print("\n" + "=" * 70)
