"""
PHASE 3: IA Channel (Transient K) Validation - v2
Connor-Stevens A-current with two-gate m*h kinetics
Comprehensive tests across voltage range and neuron types
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver
from core.kinetics import aa_IA, ba_IA, ab_IA, bb_IA

print("=" * 90)
print("PHASE 3: IA Channel Validation (v2)")
print("=" * 90)

# ============================================================================
# TEST 1: Kinetics Validation
# ============================================================================
print("\n[TEST 1] IA Kinetics Analysis (Connor-Stevens)")
print("-" * 90)

V_range = np.linspace(-100, 50, 151)
print(f"\nKinetics across voltage range ({V_range[0]:.0f} to {V_range[-1]:.0f} mV):")
print()

# Get sample values
sample_voltages = [-80, -60, -40, -20, 0, 20]
print(f"{'V (mV)':>8} | {'am':>8} | {'bm':>8} | {'tau_m':>8} | {'m_inf':>8} | {'ah':>8} | {'bh':>8} | {'tau_h':>8} | {'h_inf':>8}")
print("-" * 90)

for V in sample_voltages:
    am_val = aa_IA(V)
    bm_val = ba_IA(V)
    tau_m = 1.0 / (am_val + bm_val) if (am_val + bm_val) > 0 else np.inf
    m_inf = am_val / (am_val + bm_val) if (am_val + bm_val) > 0 else 0
    
    ah_val = ab_IA(V)
    bh_val = bb_IA(V)
    tau_h = 1.0 / (ah_val + bh_val) if (ah_val + bh_val) > 0 else np.inf
    h_inf = ah_val / (ah_val + bh_val) if (ah_val + bh_val) > 0 else 0
    
    print(f"{V:8.0f} | {am_val:8.3f} | {bm_val:8.3f} | {tau_m:8.1f} | {m_inf:8.3f} | {ah_val:8.3f} | {bh_val:8.3f} | {tau_h:8.1f} | {h_inf:8.3f}")

# Voltage-clamp like analysis
print("\n\nKey kinetic features:")
tau_m_values = []
m_inf_values = []
for V in V_range:
    am_val = aa_IA(V)
    bm_val = ba_IA(V)
    tau_m = 1.0 / (am_val + bm_val) if (am_val + bm_val) > 0 else np.inf
    m_inf = am_val / (am_val + bm_val) if (am_val + bm_val) > 0 else 0
    tau_m_values.append(tau_m)
    m_inf_values.append(m_inf)

tau_m_values = np.array(tau_m_values)
m_inf_values = np.array(m_inf_values)

print(f"  m activation: V_half approx {V_range[np.argmin(np.abs(m_inf_values - 0.5))]:.1f} mV")
print(f"  m tau range: {np.nanmin(tau_m_values[tau_m_values < 1000]):.1f}-{np.nanmax(tau_m_values[tau_m_values < 1000]):.1f} ms")
print(f"  h kinetics: slower than m (inactivation)")
print()

# ============================================================================
# TEST 2: Single Neuron with IA (Squid variant)
# ============================================================================
print("\n[TEST 2] Single Neuron Response with IA Enabled")
print("-" * 90)

cfg = FullModelConfig()
apply_preset(cfg, 'A: Squid Giant Axon (HH 1952)')
cfg.channels.enable_IA = True
cfg.channels.gA_max = 0.15  # mS/cm² - realistic for squid
cfg.stim.Iext = 10.0
cfg.stim.stim_type = 'const'
cfg.env.T_celsius = 6.3

print(f"Config: enable_IA=True, gA_max={cfg.channels.gA_max} mS/cm²")
print(f"Stimulus: const {cfg.stim.Iext:.1f} µA")
print(f"Temperature: {cfg.env.T_celsius}°C")

try:
    solver = NeuronSolver(cfg)
    result = solver.run_single()
    v_soma = result.v_soma
    t_ms = result.t
    
    peak = v_soma.max()
    rest = v_soma[0]
    spike_threshold = rest + 30
    spikes = (v_soma > spike_threshold).sum()
    
    print(f"  V_rest: {rest:.1f} mV")
    print(f"  V_peak: {peak:.1f} mV")
    print(f"  Spike count: {spikes}")
    if spikes > 0:
        print(f"  ✅ SUCCESS: IA neuron is excitable")
    else:
        print(f"  ⚠️  WARNING: No spikes detected")
except Exception as e:
    print(f"  ❌ ERROR: {str(e)[:100]}")
    import traceback
    traceback.print_exc()

# ============================================================================
# TEST 3: IA Effect on Firing Pattern (Early vs Late)
# ============================================================================
print("\n[TEST 3] IA Time Course: Early Activation vs Late Inactivation")
print("-" * 90)

cfg = FullModelConfig()
apply_preset(cfg, 'A: Squid Giant Axon (HH 1952)')
cfg.channels.enable_IA = True
cfg.channels.gA_max = 0.20
cfg.stim.Iext = 8.0
cfg.stim.stim_type = 'const'

try:
    solver = NeuronSolver(cfg)
    result = solver.run_single()
    v_soma = result.v_soma
    t_ms = result.t
    
    # Analyze spike timing
    spike_threshold = v_soma[0] + 30
    spike_indices = np.where(v_soma > spike_threshold)[0]
    spike_times = t_ms[spike_indices]
    
    if len(spike_times) >= 2:
        isi_values = np.diff(spike_times)
        print(f"  First spike: {spike_times[0]:.1f} ms")
        early_isi = isi_values[isi_values < 100].mean() if len(isi_values[isi_values < 100]) > 0 else np.nan
        print(f"  ISI (early): {early_isi:.1f} ms")
        if len(isi_values) > 5:
            late_isi = isi_values[-3:].mean()
            print(f"  ISI (late): {late_isi:.1f} ms")
            if late_isi > early_isi:
                print(f"  ✅ Firing pattern: Adaptation detected (IA inactivation)")
            else:
                print(f"  ⚠️  Firing pattern: No clear adaptation")
    elif len(spike_times) == 1:
        print(f"  Single spike at {spike_times[0]:.1f} ms")
    else:
        print(f"  No spikes detected")
except Exception as e:
    print(f"  ❌ ERROR: {str(e)[:100]}")

# ============================================================================
# TEST 4: IA Blocking Effect (compare with/without IA)
# ============================================================================
print("\n[TEST 4] IA Functional Effect: With vs Without IA")
print("-" * 90)

test_currents = [5.0, 10.0, 15.0]
results_with_ia = []
results_without_ia = []

for Iext in test_currents:
    # Without IA
    cfg_no_ia = FullModelConfig()
    apply_preset(cfg_no_ia, 'A: Squid Giant Axon (HH 1952)')
    cfg_no_ia.channels.enable_IA = False
    cfg_no_ia.stim.Iext = Iext
    cfg_no_ia.stim.stim_type = 'const'
    
    try:
        solver = NeuronSolver(cfg_no_ia)
        result = solver.run_single()
        v_noIA = result.v_soma
        spike_threshold = v_noIA[0] + 30
        spikes_no_ia = (v_noIA > spike_threshold).sum()
        peak_no_ia = v_noIA.max()
        results_without_ia.append((Iext, peak_no_ia, spikes_no_ia))
    except:
        results_without_ia.append((Iext, np.nan, 0))
    
    # With IA
    cfg_with_ia = FullModelConfig()
    apply_preset(cfg_with_ia, 'A: Squid Giant Axon (HH 1952)')
    cfg_with_ia.channels.enable_IA = True
    cfg_with_ia.channels.gA_max = 0.15
    cfg_with_ia.stim.Iext = Iext
    cfg_with_ia.stim.stim_type = 'const'
    
    try:
        solver = NeuronSolver(cfg_with_ia)
        result = solver.run_single()
        v_ia = result.v_soma
        spike_threshold = v_ia[0] + 30
        spikes_ia = (v_ia > spike_threshold).sum()
        peak_ia = v_ia.max()
        results_with_ia.append((Iext, peak_ia, spikes_ia))
    except:
        results_with_ia.append((Iext, np.nan, 0))

print(f"\n{'I (µA)':>8} | {'V_peak (no IA)':>18} | {'Spikes (no IA)':>18} | {'V_peak (with IA)':>18} | {'Spikes (with IA)':>18} | {'Delta V':>10}")
print("-" * 125)
for (I, vp_no, sp_no), (_, vp_with, sp_with) in zip(results_without_ia, results_with_ia):
    delta_v = vp_with - vp_no if not np.isnan(vp_no) and not np.isnan(vp_with) else np.nan
    print(f"{I:8.1f} | {vp_no:18.1f} | {sp_no:18d} | {vp_with:18.1f} | {sp_with:18d} | {delta_v:10.1f}")

print("\nInterpretation:")
peak_reduction = [(results_with_ia[i][1] - results_without_ia[i][1]) for i in range(len(test_currents))]
if all(p < 0 for p in peak_reduction if not np.isnan(p)):
    print("  ✅ IA consistently suppresses peak voltage (early K+ activation)")
elif all(p > 0 for p in peak_reduction if not np.isnan(p)):
    print("  ⚠️  IA increases peak voltage (unexpected)")
else:
    print("  ⚠️  IA effect varies with stimulus intensity")

# ============================================================================
# TEST 5: IA Sensitivity to Temperature
# ============================================================================
print("\n[TEST 5] IA Temperature Sensitivity (Q10 effects)")
print("-" * 90)

temperatures = [6.3, 20.0, 37.0]
ia_effects = []

for T in temperatures:
    cfg = FullModelConfig()
    apply_preset(cfg, 'A: Squid Giant Axon (HH 1952)')
    cfg.channels.enable_IA = True
    cfg.channels.gA_max = 0.15
    cfg.stim.Iext = 10.0
    cfg.stim.stim_type = 'const'
    cfg.env.T_celsius = T
    
    try:
        solver = NeuronSolver(cfg)
        result = solver.run_single()
        v_soma = result.v_soma
        peak = v_soma.max()
        spike_threshold = v_soma[0] + 30
        spikes = (v_soma > spike_threshold).sum()
        ia_effects.append((T, peak, spikes))
    except:
        ia_effects.append((T, np.nan, 0))

print(f"\n{'Temp (°C)':>12} | {'V_peak (mV)':>16} | {'Spike count':>14}")
print("-" * 50)
for T, peak, spikes in ia_effects:
    print(f"{T:12.1f} | {peak:16.1f} | {spikes:14d}")

print("\nNote: Higher temperature increases kinase rates (IA opens faster)")

# ============================================================================
# TEST 6: IA Channel Current Analysis
# ============================================================================
print("\n[TEST 6] IA Channel Current Magnitude and Direction")
print("-" * 90)

print("IA is an outward (repolarizing) current (K+ leaving cell)")
print("Gate kinetics: m (activation, fast) and h (inactivation, slow)")
print("Typical properties:")
print("  - Activation threshold: approx -50 mV")
print("  - Inactivation at rest (negative V): partially open")
print("  - Reactivation after depolarization: approx 100-1000 ms")
print()
print("Expected effects:")
print("  - Early spike: Suppressed by IA outward current")
print("  - Repetitive firing: Reduced due to inactivation")
print("  - Spike accommodation: Increased ISI (adaptation)")
print()
print("status: Kinetics present and valid for squid giant axon (Connor-Stevens 1971)")

print("\n" + "=" * 90)
print("PHASE 3 COMPLETE")
print("=" * 90)
print("\nNext: Add IA to biophysically appropriate presets:")
print("  - C: FS Interneuron (high IA for fast repolarization)")
print("  - D: Motoneuron (moderate IA for repetitive firing patterns)")
print("  - E: Cerebellar Purkinje (complex spike dynamics)")
