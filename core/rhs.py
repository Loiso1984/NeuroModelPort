import numpy as np
from numba import njit, float64, int32
from .kinetics import *
from .dual_stimulation import (
    distributed_stimulus_current_for_comp,
)
from .physics_params import (
    PhysicsParams,
    unpack_conductances,
    unpack_env_params,
    unpack_temperature_scaling,
)

# ĐšĐľĐ˝ŃŃ‚Đ°Đ˝Ń‚Ń‹ | Constants
F_CONST = 96485.33
R_GAS = 8.314
TEMP_ZERO = 273.15
CA_I_MIN_M_M = 1e-9
CA_I_MAX_M_M = 10.0
NA_I_MIN_M_M = 1.0
NA_I_MAX_M_M = 80.0
K_O_MIN_M_M = 1.0
K_O_MAX_M_M = 30.0
CA_MICRODOMAIN_SCALAR = 10.0

# ATP metabolism constants
ATP_MIN_M_M = 0.0
ATP_MAX_M_M = 10.0
ATP_ISCHEMIC_THRESHOLD = 0.5  # mM - K_ATP opens below this threshold
# Note: ATP_PUMP_FAILURE_THRESHOLD removed v11.6 - MM kinetics uses smooth ATP dependence

# Dimensional analysis for µA/cm² → mM/ms conversion:
#   d[C]/dt = I_density / (z · F · d_shell)  [mol/(cm³·s)]
#   × 1e6 (mol/cm³→mM) × 1e-3 (s→ms) = 1e-3 / (z·F·d_shell)
# d_shell = 1µm submembrane metabolic shell (Kager et al. 2000, J Neurophysiol 84:495)
_METABOLIC_DEPTH_CM = 1.0e-4  # 1.0 µm
_NA_PUMP_FACTOR = 1e-3 / (3.0 * F_CONST * _METABOLIC_DEPTH_CM)  # 1 ATP per 3 Na+
_CA_PUMP_FACTOR = 1e-3 / (2.0 * F_CONST * _METABOLIC_DEPTH_CM)  # 1 ATP per Ca²⁺
# Noise parameters
OU_NOISE_AMPLITUDE = 0.1  # 10% noise amplitude
OU_NOISE_FREQUENCY = 0.1  # Hz (period = 10 seconds)

# Synaptic time constants
TAU_RISE_RATIO = 5.0  # tau_rise = tau_decay / TAU_RISE_RATIO
# Reference: Destexhe et al. 1994, J Neurophysiol 72: 689-703

# Validate constants at module load
if TAU_RISE_RATIO <= 0.0:
    raise ValueError("TAU_RISE_RATIO must be positive")
if OU_NOISE_AMPLITUDE < 0.0:
    raise ValueError("OU_NOISE_AMPLITUDE must be non-negative")
if OU_NOISE_FREQUENCY <= 0.0:
    raise ValueError("OU_NOISE_FREQUENCY must be positive")

# Synaptic reversal potentials (mV)
E_GABA_A = -75.0  # GABA-A (Cl-), Bormann 1988, J Physiol 406: 331-350
E_GABA_B = -95.0  # GABA-B (K+ via GIRK), LĂĽscher 1997, Science 275: 1097-1100
E_EXCITATORY = 0.0  # AMPA, NMDA, Kainate, Nicotinic (cation), ~0 mV
# Reference: Johnston & Wu 1995, Foundations of Cellular Neurophysiology

# NMDA Mg2+ block constants
MG_BLOCK_CONST_DEFAULT = 3.57  # mM, default affinity (now user-configurable via EnvironmentParams)
MG_BLOCK_VOLTAGE_FACTOR = -0.062  # mV^-1, voltage sensitivity
# Reference: Jahr & Stevens 1990, J Neurosci 10: 1830-1837
# B(V) = 1 / (1 + [Mg2+]/mg_block_mM * exp(-0.062 * V))

@njit(float64(float64, float64, float64), cache=True)
def nernst_ca_ion(ca_i, ca_ext, t_kelvin):
    """Đ”Đ¸Đ˝Đ°ĐĽĐ¸Ń‡ĐµŃĐşĐ¸Đą ĐżĐľŃ‚ĐµĐ˝Ń†Đ¸Đ°Đ» ĐťĐµŃ€Đ˝ŃŃ‚Đ° Đ´Đ»ŃŹ ĐšĐ°Đ»ŃŚŃ†Đ¸ŃŹ (z=2). | Dynamic Nernst potential for Calcium (z=2)."""
    ca_i_safe = min(max(ca_i, CA_I_MIN_M_M), CA_I_MAX_M_M)
    return (R_GAS * t_kelvin / (2.0 * F_CONST)) * np.log(ca_ext / ca_i_safe) * 1000.0


@njit(float64(float64, float64, float64, float64), cache=True)
def nernst_mono_ion(c_in, c_out, z_val, t_kelvin):
    """Generic Nernst helper for monovalent ions when z_val=1 or -1."""
    c_in_safe = max(c_in, 1e-9)
    c_out_safe = max(c_out, 1e-9)
    return (R_GAS * t_kelvin / (z_val * F_CONST)) * np.log(c_out_safe / c_in_safe) * 1000.0


@njit(float64(float64, float64, float64), cache=True)
def nernst_na_ion(na_i, na_ext, t_kelvin):
    na_i_safe = min(max(na_i, NA_I_MIN_M_M), NA_I_MAX_M_M)
    return nernst_mono_ion(na_i_safe, na_ext, 1.0, t_kelvin)


@njit(float64(float64, float64, float64), cache=True)
def nernst_k_ion(k_i, k_o, t_kelvin):
    k_o_safe = min(max(k_o, K_O_MIN_M_M), K_O_MAX_M_M)
    return nernst_mono_ion(k_i, k_o_safe, 1.0, t_kelvin)

@njit(float64(float64, float64, float64, float64), cache=True)
def _biexp_waveform(t, t0, tau_rise, tau_decay):
    """Normalised dual-exponential waveform, peak = 1.0 at t_peak."""
    if t < t0:
        return 0.0
    # Guard against tau_decay == tau_rise (would cause division by zero)
    if abs(tau_decay - tau_rise) < 1e-12:
        # Fallback to simple exponential
        dt = t - t0
        tau_safe = max(tau_decay, 1e-12)
        return np.exp(-dt / tau_safe)
    dt = t - t0
    t_peak = tau_rise * tau_decay / (tau_decay - tau_rise) * np.log(tau_decay / tau_rise)
    norm = np.exp(-t_peak / tau_decay) - np.exp(-t_peak / tau_rise)
    # Guard against norm == 0
    if abs(norm) < 1e-12:
        return 0.0
    return (np.exp(-dt / tau_decay) - np.exp(-dt / tau_rise)) / norm

@njit(float64(float64), cache=True)
def _validate_time_parameter(t: float64) -> float64:
    """Validate time parameter is finite and non-negative."""
    if not np.isfinite(t) or t < 0.0:
        return 0.0
    return t

@njit(float64(float64), cache=True)
def _validate_conductance(g: float64) -> float64:
    """Validate conductance is finite and non-negative."""
    if not np.isfinite(g) or g < 0.0:
        return 0.0
    return g


@njit(float64(float64, float64, float64), cache=True)
def compute_na_k_pump_current(na_i_mM, k_o_mM, atp_mM):
    """
    Electrogenic Na+/K+ pump current using Michaelis-Menten kinetics.
    
    Replaces legacy 5% fraction model with physiological MM kinetics.
    Pump saturates at high [Na+]i and [K+]o with Hill coefficients (3 and 2).
    ATP dependence: smooth hyperbolic, not piecewise linear.
    
    Parameters
    ----------
    na_i_mM : float
        Intracellular sodium [mM]
    k_o_mM : float
        Extracellular potassium [mM]  
    atp_mM : float
        ATP concentration [mM] (2.0 = healthy)
        
    Returns
    -------
    float
        Pump current density [µA/cm²], positive = outward
    """
    return pump_current(na_i_mM, k_o_mM, atp_mM)


@njit(float64(float64, float64), cache=True)
def effective_sk_calcium(ca_val: float64, ca_rest: float64) -> float64:
    """Proxy SK microdomain calcium while keeping values physiologically bounded."""
    if not np.isfinite(ca_val) or ca_val <= 0.0:
        ca_val = ca_rest
    delta = max(0.0, ca_val - ca_rest)
    ca_eff = ca_rest + CA_MICRODOMAIN_SCALAR * delta
    return min(max(ca_eff, CA_I_MIN_M_M), 1.0)


@njit(float64(float64, float64), cache=True)
def clamp_calcium_derivative(ca_val: float64, dca: float64) -> float64:
    """Clamp calcium drift at physiological bounds without injecting artificial mass."""
    if (ca_val <= CA_I_MIN_M_M and dca < 0.0) or (ca_val >= CA_I_MAX_M_M and dca > 0.0):
        return 0.0
    return dca

@njit(cache=True)
def compute_metabolism_and_pump(
    vi, mi, hi, ni, xi, yi, ji, ai, bi, zi, wi,
    gna, gk, ga, gsk, gim, gnap, gnar,
    ena_i, ek_i,
    en_nap, en_nar, en_ia, en_sk, en_im, dyn_ca,
    g_katp_max, katp_kd_atp_mM,
    atp_i_val, atp_synthesis_rate,
    na_i_val, k_o_val, k_o_rest_mM,
    ion_drift_gain, k_o_clearance_tau_ms,
    i_ca_influx,
):
    """Unified metabolism + pump computation for a single compartment.

    Computes total Na/K currents, Na/K pump, K_ATP channel, and metabolic
    derivatives (d[ATP]/dt, d[Na_i]/dt, d[K_o]/dt).

    Called from both rhs_multicompartment (SciPy BDF wrapper) and
    run_native_loop (Backward-Euler Hines) to guarantee identical physics.

    Returns: (i_pump, i_katp, datp, dnai, dko)
    """
    # ── Total Na current (transient + persistent + resurgent) ──
    i_na_total = gna * (mi ** 3) * hi * (vi - ena_i)
    if en_nap:
        i_na_total += gnap * xi * (vi - ena_i)
    if en_nar:
        i_na_total += gnar * yi * ji * (vi - ena_i)

    # ── Total K current (DR + IA + SK + IM) ──
    i_k_total = gk * (ni ** 4) * (vi - ek_i)
    if en_ia:
        i_k_total += ga * ai * bi * (vi - ek_i)
    if en_sk:
        i_k_total += gsk * zi * (vi - ek_i)
    if en_im:
        i_k_total += gim * wi * (vi - ek_i)

    # ── K_ATP channel: g = g_max / (1 + ([ATP]/Kd)^2) ──
    atp_ratio = atp_i_val / max(katp_kd_atp_mM, 1e-12)
    g_katp = g_katp_max / (1.0 + atp_ratio * atp_ratio)
    i_katp = g_katp * (vi - ek_i)
    i_k_total += i_katp

    # ── Na/K pump current (Michaelis-Menten kinetics, C1-continuous) ──
    # MM kinetics: saturates at high [Na+]i, [K+]o; smooth ATP dependence
    i_pump = compute_na_k_pump_current(na_i_val, k_o_val, atp_i_val)
    
    # Pump consumption: 1 ATP per 3 Na+ pumped (Skou 1957)
    # Derived from pump_current: consumption = pump_activity / (3*F*depth)
    pump_activity = max(0.0, i_pump)  # Only outward pump current consumes ATP
    pump_consumption = pump_activity * _NA_PUMP_FACTOR
    if dyn_ca:
        pump_consumption += max(0.0, i_ca_influx) * _CA_PUMP_FACTOR

    # ── ATP SOFT BOUNDARY (v11.6) — C1-continuous for Lyapunov stability ──
    # Substrate-limited synthesis: ADP availability = (ATP_max - ATP_current)
    # Physiology: Mitochondria slow down as ATP pool fills (ADP becomes scarce)
    adp_avail = max(0.0, ATP_MAX_M_M - atp_i_val)  # ADP proxy
    synthesis_eff = atp_synthesis_rate * (adp_avail / ATP_MAX_M_M)  # Michaelis-style
    datp = synthesis_eff * 0.001 - pump_consumption
    
    # Soft boundary: prevent negative ATP without hard cut-off
    # Linear ramp to zero derivative at boundary (prevents numerical chatter)
    if atp_i_val < 0.02 and datp < 0:  # critically low, draining
        soft_factor = max(0.0, atp_i_val / 0.02)
        datp *= soft_factor
    
    # Hard ceiling at ATP_MAX - homeostatic regulation
    if atp_i_val >= ATP_MAX_M_M and datp > 0.0:
        datp = 0.0

    # ── Na_i drift: inward Na load - pump efflux (3 Na+/cycle) ──
    dnai = ion_drift_gain * (max(0.0, -i_na_total) - 3.0 * max(0.0, i_pump))
    if na_i_val <= NA_I_MIN_M_M and dnai < 0.0:
        dnai = abs(dnai) * 0.5
    elif na_i_val >= NA_I_MAX_M_M and dnai > 0.0:
        dnai = -abs(dnai) * 0.5

    # ── K_o drift: outward K flux - pump influx (2 K+/cycle) + clearance ──
    dko = ion_drift_gain * (max(0.0, i_k_total) - 2.0 * max(0.0, i_pump))
    dko -= (k_o_val - k_o_rest_mM) / max(k_o_clearance_tau_ms, 1e-12)
    if k_o_val <= K_O_MIN_M_M and dko < 0.0:
        dko = abs(dko) * 0.5
    elif k_o_val >= K_O_MAX_M_M and dko > 0.0:
        dko = -abs(dko) * 0.5

    return i_pump, i_katp, datp, dnai, dko


@njit(cache=True)
def _calculate_syn_tau(stype, atau_mult):
    """
    Map a conductance-based synapse type code to its literature baseline rise and decay time constants, scaled by a multiplier.
    
    Parameters:
        stype (int): Synapse type code (4=AMPA, 5=NMDA, 6=GABA-A, 7=GABA-B, 8=Kainate, 9=Nicotinic).
        atau_mult (float): Multiplier applied to literature time constants; values less than 0.01 are treated as 0.01.
    
    Returns:
        (tau_rise, tau_decay) (tuple of float): Rise and decay time constants (same units as literature values) scaled by `atau_mult`; returns (0.0, 0.0) for unsupported `stype`.
    """
    # Prevent zero or negative multipliers
    mult = max(0.01, atau_mult)
    
    if stype == 4:      # AMPA (Literature: r=0.2, d=2.0)
        return 0.2 * mult, 2.0 * mult
    elif stype == 5:    # NMDA (Literature: r=5.0, d=120.0)
        return 5.0 * mult, 120.0 * mult
    elif stype == 6:    # GABA-A (Literature: r=0.5, d=6.0)
        return 0.5 * mult, 6.0 * mult
    elif stype == 7:    # GABA-B (Literature: r=30.0, d=250.0)
        return 30.0 * mult, 250.0 * mult
    elif stype == 8:    # Kainate (Literature: r=1.0, d=10.0)
        return 1.0 * mult, 10.0 * mult
    elif stype == 9:    # Nicotinic (Literature: r=2.0, d=20.0)
        return 2.0 * mult, 20.0 * mult
    else:
        return 0.0, 0.0


@njit(float64(float64, int32, float64, float64, float64, float64, float64, float64, float64), cache=True)
def get_stim_current(t, stype, iext, t0, td, atau, zap_f0_hz, zap_f1_hz, zap_rise_ms):
    """Return stimulus current at time t for various stimulus types.

    For stype >= 4, iext is a conductance amplitude [mS/cmÂ˛] and must be
    non-negative. Current direction is determined later by (V - E_syn).
    """
    # Input validation
    t = _validate_time_parameter(t)
    t0 = _validate_time_parameter(t0)
    td = _validate_time_parameter(td)
    iext = iext if np.isfinite(iext) else 0.0

    # Validate atau to prevent division by zero
    if atau <= 0.0:
        return 0.0

    if stype == 0:  # const / tonic current clamp
        return iext if t >= t0 else 0.0
    elif stype == 1:  # pulse
        return iext if t0 <= t <= t0 + td else 0.0
    elif stype == 2:  # alpha (EPSC) â€” current-based
        if t < t0:
            return 0.0
        dt = (t - t0) / atau
        return iext * dt * np.exp(1.0 - dt)
    elif stype == 3:  # OU_noise (Ornstein-Uhlenbeck noise)
        # Simplified OU noise that works with Numba without random seeding
        if t >= t0:
            # Use deterministic function for reproducibility
            phase = 2.0 * np.pi * OU_NOISE_FREQUENCY * t / 1000.0
            noise = np.sin(phase + t * 0.01)  # Slowly varying noise
            ou_factor = np.exp(-(t - t0) / 100.0) if t > t0 else 1.0
            noise_factor = 1.0 + OU_NOISE_AMPLITUDE * noise * ou_factor
            return iext * noise_factor
        else:
            return 0.0
    elif 4 <= stype <= 9:  # Conductance-based synaptic types (AMPA/NMDA/GABA-A/GABA-B/Kainate/Nicotinic)
        tau_r, tau_d = _calculate_syn_tau(stype, atau)
        iext_eff = iext
        if stype == 5:  # NMDA boost to compensate for Mg2+ block at resting potentials
            iext_eff *= 5.0
        return abs(iext_eff) * _biexp_waveform(t, t0, tau_r, tau_d)
    elif stype == 10:  # ZAP/Chirp current (frequency sweep)
        if td <= 0.0 or t < t0 or t > (t0 + td):
            return 0.0
        dt = t - t0  # in ms
        dt_sec = dt / 1000.0  # Convert to seconds for frequency math
        td_sec = td / 1000.0
        
        # Tukey window (cosine-tapered) to eliminate discontinuity at start/end
        # Reduces spectral leakage and numerical artifacts from abrupt current changes
        # Use configured rise time, fallback to 5% of duration if not specified
        rise_ms = max(0.0, zap_rise_ms) if zap_rise_ms > 0.0 else 0.05 * td
        if rise_ms > 0.0 and dt < rise_ms:
            # Rising phase: 0.5 * (1 - cos(π * t / t_rise))
            window = 0.5 * (1.0 - np.cos(np.pi * dt / rise_ms))
        elif rise_ms > 0.0 and dt > (td - rise_ms):
            # Falling phase: symmetric
            fall_dt = td - dt
            window = 0.5 * (1.0 - np.cos(np.pi * fall_dt / rise_ms))
        else:
            window = 1.0
        
        # Phase = 2*pi * integral(f0 + k*t) dt = 2*pi * (f0*t + 0.5*k*t^2)
        k_hz_per_sec = (zap_f1_hz - zap_f0_hz) / td_sec
        phase = 2.0 * np.pi * (zap_f0_hz * dt_sec + 0.5 * k_hz_per_sec * dt_sec * dt_sec)
        return iext * np.sin(phase) * window
    # Default: const (stype == 0)
    return iext

@njit(float64(float64, float64, float64), cache=True)
def nmda_mg_block(V, Mg_ext, mg_block_mM=3.57):
    """Voltage-dependent MgÂ˛âş block of NMDA receptors.

    B(V) = 1 / (1 + [MgÂ˛âş]/MG_BLOCK_CONST * exp(MG_BLOCK_VOLTAGE_FACTOR * V))
    Reference: Jahr & Stevens 1990, J Neurosci 10:1830
    """
    return 1.0 / (1.0 + (Mg_ext / max(mg_block_mM, 1e-12)) * np.exp(MG_BLOCK_VOLTAGE_FACTOR * V))


@njit(float64(float64, float64, float64, float64, float64), cache=True)
def ghk_current(v_mV, ci_mM, co_mM, z, T_kelvin):
    """
    Goldman-Hodgkin-Katz (GHK) equation for ionic current through membrane.
    
    Physics: Unlike Ohmic I=g(V-E), GHK accounts for rectification at extreme
    concentration gradients (e.g., Ca2+ with 40,000x gradient).
    
    Formula:
        I = P * z^2 * (V*F^2/RT) * (ci - co*exp(-zVF/RT)) / (1 - exp(-zVF/RT))
    
    Parameters
    ----------
    v_mV : float
        Membrane potential (mV)
    ci_mM : float
        Intracellular concentration (mM)
    co_mM : float
        Extracellular concentration (mM)
    z : float
        Ion valence (e.g., +2 for Ca2+, -1 for Cl-)
    T_kelvin : float
        Temperature (Kelvin)
    
    Returns
    -------
    float
        Current density (µA/cm²) - positive = outward
    """
    # Use global constants from module (avoid local redefinition for Numba cache efficiency)
    # F_CONST = 96485.33, R_GAS = 8.314 defined at module level
    
    # Convert mV to V for dimensional consistency
    V = v_mV * 1e-3
    
    # Precompute common terms
    RT = R_GAS * T_kelvin
    zF = z * F_CONST
    zF_V_RT = zF * V / RT
    
    # Handle V → 0 limit using Taylor expansion to avoid numerical issues
    # At V=0: GHK reduces to I = P * z^2 * F^2 * (ci - co) / RT
    eps = 1e-10
    if abs(V) < eps:
        # Linear approximation near zero: derivative of exp(-x) at x=0 is -1
        # I ≈ P * z^2 * F * (ci - co) * V / RT when V→0
        # But we need consistent units: convert back
        # At V→0: GHK simplifies to I = P*z^2*F^2*(ci-co)*V/RT
        # Factor already computed: zF = z*F_CONST
        return zF * (ci_mM - co_mM) * 1e-3 * (V / RT)  # mM·V→µA/cm² scaling
    
    # Full GHK equation
    exp_term = np.exp(-zF_V_RT)
    denominator = 1.0 - exp_term
    
    # Numerator: ci - co * exp(-zFV/RT)
    numerator = ci_mM - co_mM * exp_term
    
    # GHK current (permeability factor P applied externally for unit consistency)
    # Units: [P in cm/s] * [z^2] * [F^2/RT in C^2/(J·mol) = C/V·mol] * [V in V] * [concentration in mM]
    # = cm/s * C/V·mol * V * mol/L = C/(s·cm²) * 10 = µA/cm² (with appropriate scaling)
    ghk_factor = zF * zF / RT * V  # z^2 * F^2 / (RT) * V
    
    # Return with sign convention: positive = outward current
    return ghk_factor * numerator / denominator


@njit(cache=True)
def compute_ionic_currents_scalar(
    vi, mi, hi, ni,
    ri, si, ui, ai, bi, pi, qi, wi, xi, yi, ji, sk_gate,
    en_ih, en_ica, en_ia, en_sk, en_itca, en_im, en_nap, en_nar, dyn_ca,
    gna, gk, gl, gih, gca, ga, gsk, gtca, gim, gnap, gnar,
    ena, ek, el, eih,
    ca_i_val, ca_ext, ca_rest, t_kelvin,
):
    """
    Compute total ionic current and calcium influx contribution for a single compartment.
    
    Calculates leak, sodium, potassium, and optional channel currents (Ih, high-voltage Ca, A-type K, T-type Ca, SK, M-current, persistent and resurgent Na) from the provided gating variables, conductances, and reversal potentials. When `dyn_ca` is True, the Ca reversal potential is computed from `ca_i_val` and `ca_ext` and any inward Ca current (negative current) is accumulated into the calcium influx return value.
    
    GHK NOTE: When `dyn_ca=True`, calcium currents (ICa, ITCa) use the Goldman-Hodgkin-Katz
    (GHK) equation instead of Ohmic I=g(V-E). In this mode, `gca` and `gtca` parameters are
    interpreted as "Permeability Factors" [cm/s], not conductances [mS/cm²]. This is
    physically correct: GHK current depends on permeability P, not conductance g.
    The conversion factor P = g_max * RT/(z²F²) is applied internally for unit consistency.
    
    Parameters:
        vi (float): Membrane voltage in millivolts.
        sk_gate (float): SK-channel activation variable (unitless, typically 0â€“1).
        dyn_ca (bool): If True, use GHK calcium currents (gca as permeability factor).
        ca_i_val (float): Intracellular Ca concentration in mM; clamped to physiological bounds when used.
        ca_ext (float): Extracellular Ca concentration in mM (used for Nernst/GHK calculation).
        t_kelvin (float): Absolute temperature in Kelvin (used for Nernst/GHK calculation).
        gca (float): L-type calcium conductance [mS/cm²] or permeability [cm/s] if dyn_ca=True.
        gtca (float): T-type calcium conductance [mS/cm²] or permeability [cm/s] if dyn_ca=True.
    
    Returns:
        tuple: `(i_ion, i_ca_influx)` where `i_ion` is the net ionic current (signed, µA/cm²) 
        and `i_ca_influx` is the positive magnitude of inward Ca current (0.0 if none).
    """
    i_ion = gl * (vi - el)
    i_ion += gna * (mi * mi * mi) * hi * (vi - ena)
    i_ion += gk * (ni * ni * ni * ni) * (vi - ek)

    i_ca_influx = 0.0
    if dyn_ca:
        ca_i_safe = min(max(ca_i_val, CA_I_MIN_M_M), CA_I_MAX_M_M)
        eca_i = nernst_ca_ion(ca_i_safe, ca_ext, t_kelvin)
    else:
        eca_i = 120.0

    if en_ih:
        i_ion += gih * ri * (vi - eih)

    # ── Calcium currents (ICa, ITCa) with GHK when dynamic calcium enabled ──
    # Precompute common permeability factor for unit consistency: z=2 for Ca2+
    # P = g_max / (z^2 * F^2 / RT) = g_max * RT / (z^2 * F^2)
    use_ghk = dyn_ca and ca_i_val > CA_I_MIN_M_M and t_kelvin > 0
    _RT_over_z2F2 = (R_GAS * t_kelvin) / (4.0 * F_CONST * F_CONST) if use_ghk else 0.0
    
    if en_ica:
        if use_ghk:
            i_ca_current = gca * _RT_over_z2F2 * (si * si) * ui * ghk_current(vi, ca_i_val, ca_ext, 2.0, t_kelvin)
        else:
            # Fallback to Ohmic when dynamic calcium disabled or [Ca] invalid
            i_ca_current = gca * (si * si) * ui * (vi - eca_i)
        i_ion += i_ca_current
        if i_ca_current < 0.0:
            i_ca_influx += -i_ca_current

    if en_ia:
        i_ion += ga * ai * bi * (vi - ek)  # IA is a K+ channel, use ek

    if en_itca:
        if use_ghk:
            i_tca = gtca * _RT_over_z2F2 * (pi * pi) * qi * ghk_current(vi, ca_i_val, ca_ext, 2.0, t_kelvin)
        else:
            i_tca = gtca * (pi * pi) * qi * (vi - eca_i)
        i_ion += i_tca
        if i_tca < 0.0:
            i_ca_influx += -i_tca

    if en_sk:
        i_ion += gsk * sk_gate * (vi - ek)

    if en_im:
        i_ion += gim * wi * (vi - ek)

    if en_nap:
        i_ion += gnap * xi * (vi - ena)

    if en_nar:
        i_ion += gnar * yi * ji * (vi - ena)

    return i_ion, i_ca_influx


@njit(cache=True)
def compute_ionic_conductances_scalar(
    vi, mi, hi, ni,
    ri, si, ui, ai, bi, pi, qi, wi, xi, yi, ji, zi,
    en_ih, en_ica, en_ia, en_sk, en_itca, en_im, en_nap, en_nar, dyn_ca,
    gna, gk, gl, gih, gca, ga, gsk, gtca, gim, gnap, gnar,
    ena, ek, el, eih,
    ca_i_val, ca_ext, ca_rest, t_kelvin,
):
    """
    Unified scalar ionic conductance calculator for Hines solver.
    
    Computes total membrane conductance (g_total), conductance-weighted
    effective reversal (e_eff), and calcium influx for a single compartment.
    This is the ONLY place where I=g(V-E) logic lives for the Hines path.
    
    Parameters match compute_ionic_currents_scalar for consistency.
    
    Returns:
        tuple: (g_total, e_eff, i_ca_influx) where:
        - g_total: total ionic conductance [mS/cm²]
        - e_eff: weighted reversal potential [mV] = sum(g*E)/g_total
        - i_ca_influx: positive inward Ca current [µA/cm²]
    """
    # ── Base conductances ──
    g_tot = gl
    e_eff_i = gl * el
    
    # Na current
    g_na = gna * (mi ** 3) * hi
    g_tot += g_na
    e_eff_i += g_na * ena
    
    # K-DR current
    g_k = gk * (ni ** 4)
    g_tot += g_k
    e_eff_i += g_k * ek
    
    i_ca_influx = 0.0
    
    # Calcium reversal (for non-GHK or Ohmic fallback)
    if dyn_ca:
        ca_safe = min(max(ca_i_val, CA_I_MIN_M_M), CA_I_MAX_M_M)
        eca_i = nernst_ca_ion(ca_safe, ca_ext, t_kelvin)
    else:
        eca_i = 120.0
    
    # GHK pre-factor (same as compute_ionic_currents_scalar)
    use_ghk = dyn_ca and ca_i_val > CA_I_MIN_M_M and t_kelvin > 0
    _RT_over_z2F2 = (R_GAS * t_kelvin) / (4.0 * F_CONST * F_CONST) if use_ghk else 0.0
    
    if en_ih:
        g_ih = gih * ri
        g_tot += g_ih
        e_eff_i += g_ih * eih
    
    if en_ica:
        g_ca = gca * (si ** 2) * ui
        g_tot += g_ca
        if use_ghk:
            # For GHK: current is NOT ohmic, so we estimate effective E
            # GHK current = P * GHK_factor; approximate g_eff = I/(V-E)
            i_ca = gca * _RT_over_z2F2 * (si ** 2) * ui * ghk_current(vi, ca_i_val, ca_ext, 2.0, t_kelvin)
            # Approximate reversal contribution: if I=0 at E=ECa, use ECa
            e_eff_i += g_ca * eca_i  # Best Ohmic approximation for Hines
            if i_ca < 0.0:
                i_ca_influx += -i_ca
        else:
            e_eff_i += g_ca * eca_i
            i_ca = g_ca * (vi - eca_i)
            if i_ca < 0.0:
                i_ca_influx += -i_ca
    
    if en_ia:
        g_ia = ga * ai * bi
        g_tot += g_ia
        e_eff_i += g_ia * ek  # IA is K+ channel
    
    if en_itca:
        g_tca = gtca * (pi ** 2) * qi
        g_tot += g_tca
        if use_ghk:
            i_tca = gtca * _RT_over_z2F2 * (pi ** 2) * qi * ghk_current(vi, ca_i_val, ca_ext, 2.0, t_kelvin)
            e_eff_i += g_tca * eca_i
            if i_tca < 0.0:
                i_ca_influx += -i_tca
        else:
            e_eff_i += g_tca * eca_i
            i_tca = g_tca * (vi - eca_i)
            if i_tca < 0.0:
                i_ca_influx += -i_tca
    
    if en_sk:
        g_sk = gsk * zi
        g_tot += g_sk
        e_eff_i += g_sk * ek
    
    if en_im:
        g_im = gim * wi
        g_tot += g_im
        e_eff_i += g_im * ek
    
    if en_nap:
        g_nap = gnap * xi
        g_tot += g_nap
        e_eff_i += g_nap * ena
    
    if en_nar:
        g_nar = gnar * yi * ji
        g_tot += g_nar
        e_eff_i += g_nar * ena
    
    return g_tot, e_eff_i, i_ca_influx


@njit(float64(int32, float64, float64), cache=True)
def _get_syn_reversal(stype, e_rev_primary, e_rev_secondary):
    """
    Select the synaptic reversal potential (mV) for a conductance-based synapse type.
    
    Args:
        stype: Synapse type (4=AMPA, 5=NMDA, 6=GABAA, 7=GABAB, 8=Kainate, 9=Nicotinic)
        e_rev_primary: Reversal for excitatory synapses (AMPA/NMDA/Kainate/Nicotinic)
        e_rev_secondary: Reversal for inhibitory GABA-A synapses (pathology-configurable)
    
    Returns:
        Reversal potential in millivolts (mV) for the specified synapse type.
    """
    # Excitatory: AMPA(4), NMDA(5), Kainate(8), Nicotinic(9)
    if stype == 4 or stype == 5 or stype == 8 or stype == 9:
        return e_rev_primary
    # Inhibitory GABA-A (Cl-), pathology-configurable
    if stype == 6:
        return e_rev_secondary
    # Inhibitory GABA-B (K+ via GIRK), fixed biophysical reversal
    if stype == 7:
        return E_GABA_B
    # Fallback for unknown synaptic code
    return e_rev_primary


@njit(float64(float64, int32, float64, float64[:], int32, float64), cache=True)
def get_event_driven_conductance(t: float64, stype: int32, iext: float64,
                               event_times: float64[:], n_events: int32, atau: float64) -> float64:
    """
    Compute total synaptic conductance at time t from an event-time queue using a normalized biexponential kernel.
    
    OPTIMIZATION: Scalar version optimized for ODE integrator (called thousands of times).
    For batch analytics, use get_event_driven_conductance_batch.
    
    Each event contributes a normalized dual-exponential waveform (peak = 1) whose rise/decay 
    time constants are determined by the synapse type and the `atau` multiplier. Events with 
    times < 0 or > t + 1000 are ignored. If `n_events == 0`, `atau <= 0`, or the synapse type 
    yields zero time constants, the function returns 0.0.
    
    Parameters:
        event_times (float64[:]): Array of synaptic event timestamps (ms).
        n_events (int32): Number of event entries from `event_times` to consider.
    
    Returns:
        float64: Total conductance (mS/cm²) at time t, computed as `abs(iext)` times the sum of per-event normalized waveforms.
    """
    if n_events == 0 or atau <= 0.0:
        return 0.0
    
    # Validate event_times bounds
    if n_events > len(event_times):
        return 0.0
    
    # Use shared tau helper — keeps get_stim_current and event-driven paths consistent
    tau_r, tau_d = _calculate_syn_tau(stype, atau)
    if tau_r == 0.0 and tau_d == 0.0:
        return 0.0
    
    g = 0.0
    window = 5.0 * tau_d
    # OPTIMIZATION: Early termination for sorted event times (common case)
    # Events are typically sorted chronologically; we can break early when dt < 0
    for k in range(n_events):
        event_time = event_times[k]
        dt_event = t - event_time
        if dt_event < 0.0:
            # For sorted events, all subsequent events are in the future
            break  # Early termination ~50% faster for sparse events
        if dt_event > window:
            continue
        g += _biexp_waveform(t, event_time, tau_r, tau_d)
    return abs(iext) * g


def get_event_driven_conductance_batch(
    t_array: np.ndarray, 
    stype: int, 
    iext: float, 
    event_times: np.ndarray, 
    atau: float
) -> np.ndarray:
    """
    Vectorized batch computation of synaptic conductance for all time points.
    
    PERFORMANCE: Uses NumPy broadcasting and vectorization instead of Python loops.
    ~10-100x faster than scalar loop for analytics/post-processing.
    Not suitable for ODE integration (requires precomputed time array).
    
    Parameters:
        t_array (np.ndarray): 1D array of time points (ms)
        stype (int): Synapse type code
        iext (float): Conductance amplitude (mS/cm²)
        event_times (np.ndarray): 1D array of synaptic event timestamps (ms)
        atau (float): Time constant multiplier
    
    Returns:
        np.ndarray: Conductance values (mS/cm²) for each time point, shape matches t_array
    """
    if len(event_times) == 0 or atau <= 0.0:
        return np.zeros_like(t_array, dtype=float)
    
    tau_r, tau_d = _calculate_syn_tau(stype, atau)
    if tau_r == 0.0 and tau_d == 0.0:
        return np.zeros_like(t_array, dtype=float)
    
    # Vectorized computation: (n_times, n_events) -> sum over events
    # dt[i, j] = t[i] - event_times[j]
    dt = t_array[:, np.newaxis] - event_times[np.newaxis, :]
    
    # Mask valid contributions (dt >= 0 and within window)
    window = 5.0 * tau_d
    valid = (dt >= 0) & (dt <= window)
    
    # Compute biexponential waveform for all valid combinations
    # Using the analytical form from _biexp_waveform
    if abs(tau_d - tau_r) < 1e-12:
        # Single exponential fallback
        wave = np.exp(-dt / tau_d)
    else:
        t_peak = tau_r * tau_d / (tau_d - tau_r) * np.log(tau_d / tau_r)
        norm = np.exp(-t_peak / tau_d) - np.exp(-t_peak / tau_r)
        if abs(norm) < 1e-12:
            wave = np.zeros_like(dt)
        else:
            wave = (np.exp(-dt / tau_d) - np.exp(-dt / tau_r)) / norm
    
    # Zero out invalid contributions
    wave = np.where(valid, wave, 0.0)
    
    # Sum over events for each time point
    g = np.sum(wave, axis=1)
    
    return abs(iext) * g


@njit(cache=True)
def rhs_multicompartment(
    t, y, physics_params, dydt
):
    """
    Compute time derivatives for a multi-compartment Hodgkinâ€“Huxleyâ€“type neuron model.
    
    This function evaluates membrane voltage and gating-variable ODEs for all compartments, including:
    - intrinsic ionic currents (Na, K, leak, Ih, Ca, A, Tâ€‘type Ca, M, SK, persistent/resurgent Na as configured),
    - synaptic/externally applied stimuli (supporting current- and conductance-based modes, event-driven conductances, NMDA Mg2+ block, and optional dual stimulation),
    - sparse axial coupling via a Laplacian representation,
    - optional calcium and ATP dynamics with physiologically bounded handling,
    - optional single-compartment dendritic filtering and per-compartment temperature/Q10 scaling.
    
    dydt is a preallocated output array that is zeroed and then filled in-place; the same array is returned.
    """
    
    # --- Unpack physics parameters ---
    n_comp = physics_params.n_comp
    en_ih = physics_params.en_ih
    en_ica = physics_params.en_ica
    en_ia = physics_params.en_ia
    en_sk = physics_params.en_sk
    dyn_ca = physics_params.dyn_ca
    en_itca = physics_params.en_itca
    en_im = physics_params.en_im
    en_nap = physics_params.en_nap
    en_nar = physics_params.en_nar
    dyn_atp = physics_params.dyn_atp
    
    # Unpack conductance matrix
    (gna_v, gk_v, gl_v, gih_v, gca_v, ga_v, gsk_v, 
     gtca_v, gim_v, gnap_v, gnar_v) = unpack_conductances(physics_params.gbar_mat, n_comp)
    
    # Reversal potentials
    ena = physics_params.ena
    ek = physics_params.ek
    el = physics_params.el
    eih = physics_params.eih
    
    # Morphology and axial coupling
    cm_v = physics_params.cm_v
    l_data = physics_params.l_data
    l_indices = physics_params.l_indices
    l_indptr = physics_params.l_indptr
    
    # Unpack temperature scaling
    (phi_na, phi_k, phi_ih, phi_ca, phi_ia, phi_tca, 
     phi_im, phi_nap, phi_nar) = unpack_temperature_scaling(physics_params.phi_mat, n_comp)
    
    # Environment and ATP/metabolism scalars (packed for index-stable native/Jacobian use)
    (
        t_kelvin,
        ca_ext,
        ca_rest,
        tau_ca,
        mg_ext,
        tau_sk,
        im_speed_multiplier,
        g_katp_max,
        katp_kd_atp_mM,
        atp_max_mM,
        atp_synthesis_rate,
        na_i_rest_mM,
        na_ext_mM,
        k_i_mM,
        k_o_rest_mM,
        ion_drift_gain,
        k_o_clearance_tau_ms,
    ) = unpack_env_params(physics_params.env_params)
    b_ca = physics_params.b_ca
    nmda_mg_block_mM = physics_params.nmda_mg_block_mM

    # Primary stimulation
    stype = physics_params.stype
    iext = physics_params.iext
    t0 = physics_params.t0
    td = physics_params.td
    atau = physics_params.atau
    zap_f0_hz = physics_params.zap_f0_hz
    zap_f1_hz = physics_params.zap_f1_hz
    zap_rise_ms = physics_params.zap_rise_ms
    event_times_arr = physics_params.event_times_arr
    n_events = physics_params.n_events
    stim_comp = physics_params.stim_comp
    stim_mode = physics_params.stim_mode
    use_dfilter_primary = physics_params.use_dfilter_primary
    dfilter_attenuation = physics_params.dfilter_attenuation
    dfilter_tau_ms = physics_params.dfilter_tau_ms
    
    # Secondary stimulation
    dual_stim_enabled = physics_params.dual_stim_enabled
    stype_2 = physics_params.stype_2
    iext_2 = physics_params.iext_2
    t0_2 = physics_params.t0_2
    td_2 = physics_params.td_2
    atau_2 = physics_params.atau_2
    zap_f0_hz_2 = physics_params.zap_f0_hz_2
    zap_f1_hz_2 = physics_params.zap_f1_hz_2
    zap_rise_ms_2 = physics_params.zap_rise_ms_2
    event_times_arr_2 = physics_params.event_times_arr_2
    n_events_2 = physics_params.n_events_2
    stim_comp_2 = physics_params.stim_comp_2
    stim_mode_2 = physics_params.stim_mode_2
    use_dfilter_secondary = physics_params.use_dfilter_secondary
    dfilter_attenuation_2 = physics_params.dfilter_attenuation_2
    dfilter_tau_ms_2 = physics_params.dfilter_tau_ms_2


    
    offsets = physics_params.state_offsets
    off_v = int(offsets.off_v)
    off_m = int(offsets.off_m)
    off_h = int(offsets.off_h)
    off_n = int(offsets.off_n)
    off_r = int(offsets.off_r)
    off_s = int(offsets.off_s)
    off_u = int(offsets.off_u)
    off_a = int(offsets.off_a)
    off_b = int(offsets.off_b)
    off_p = int(offsets.off_p)
    off_q = int(offsets.off_q)
    off_w = int(offsets.off_w)
    off_x = int(offsets.off_x)
    off_y = int(offsets.off_y)
    off_j = int(offsets.off_j)
    off_zsk = int(offsets.off_zsk)
    off_ca = int(offsets.off_ca)
    off_atp = int(offsets.off_atp)
    off_na_i = int(offsets.off_na_i)
    off_k_o = int(offsets.off_k_o)
    off_ifilt_primary = int(offsets.off_ifilt_primary)
    off_ifilt_secondary = int(offsets.off_ifilt_secondary)

    # --- Stimulus: compute once, apply to target compartments ---
    # For stype >= 4 (synaptic): base_current is conductance g(t) [mS/cmÂ˛]
    # and will be multiplied by (V - E_syn) per-compartment in the main loop.
    # For stype < 4 (const/pulse/alpha): base_current is raw current [ÂµA/cmÂ˛].
    is_conductance_based = (stype >= 4)
    # Stage 6.3: event-driven mode if queue is non-empty and stim is synaptic
    if n_events > 0 and is_conductance_based:
        base_current = get_event_driven_conductance(t, stype, iext, event_times_arr, n_events, atau)
    else:
        base_current = get_stim_current(t, stype, iext, t0, td, atau, zap_f0_hz, zap_f1_hz, zap_rise_ms)
    e_syn = _get_syn_reversal(stype, physics_params.e_rev_syn_primary, physics_params.e_rev_syn_secondary) if is_conductance_based else 0.0
    is_nmda = (stype == 5)

    i_filtered_primary = 0.0
    if use_dfilter_primary == 1:
        i_filtered_primary = y[off_ifilt_primary]

    i_filtered_secondary = 0.0
    if use_dfilter_secondary == 1:
        i_filtered_secondary = y[off_ifilt_secondary]

    d_ifiltered_dt_primary = 0.0
    # Validate filter parameters before use
    if stim_mode == 2 and use_dfilter_primary == 1 and dfilter_tau_ms > 1e-12:
        attenuated_current = dfilter_attenuation * base_current
        d_ifiltered_dt_primary = (attenuated_current - i_filtered_primary) / dfilter_tau_ms
    # Disable filter if tau is invalid (below epsilon threshold)
    use_dfilter_primary_eff = 1 if (stim_mode == 2 and use_dfilter_primary == 1 and dfilter_tau_ms > 1e-12) else 0

    is_cond_2 = False
    e_syn_2 = 0.0
    is_nmda_2 = False
    d_ifiltered_dt_secondary = 0.0
    base_current_2 = 0.0
    if dual_stim_enabled == 1:
        is_cond_2 = (stype_2 >= 4)
        e_syn_2 = _get_syn_reversal(stype_2, physics_params.e_rev_syn_primary, physics_params.e_rev_syn_secondary) if is_cond_2 else 0.0
        is_nmda_2 = (stype_2 == 5)
        # Stage 6.3 symmetry: use event-driven conductance for secondary stim if queue is non-empty
        if n_events_2 > 0 and is_cond_2:
            base_current_2 = get_event_driven_conductance(t, stype_2, iext_2, event_times_arr_2, n_events_2, atau_2)
        else:
            base_current_2 = get_stim_current(t, stype_2, iext_2, t0_2, td_2, atau_2, zap_f0_hz_2, zap_f1_hz_2, zap_rise_ms_2)
        if stim_mode_2 == 2 and use_dfilter_secondary == 1 and dfilter_tau_ms_2 > 1e-12:
            attenuated_current_2 = dfilter_attenuation_2 * base_current_2
            d_ifiltered_dt_secondary = (attenuated_current_2 - i_filtered_secondary) / dfilter_tau_ms_2
    # Disable filter if tau is invalid or stim_mode is not 2 (below epsilon threshold)
    use_dfilter_secondary_eff = 1 if (dual_stim_enabled == 1 and stim_mode_2 == 2 and use_dfilter_secondary == 1 and dfilter_tau_ms_2 > 1e-12) else 0

    # --- Zero the pre-allocated output buffer (Numba-native; no heap alloc) ---
    for _k in range(len(dydt)):
        dydt[_k] = 0.0

    # --- Main C-style loop: all currents as scalars ---
    for i in range(n_comp):
        vi = y[off_v + i]
        mi = y[off_m + i]
        hi = y[off_h + i]
        ni = y[off_n + i]

        ri = y[off_r + i] if en_ih else 0.0
        si = y[off_s + i] if en_ica else 0.0
        ui = y[off_u + i] if en_ica else 0.0
        ai = y[off_a + i] if en_ia else 0.0
        bi = y[off_b + i] if en_ia else 0.0
        pi = y[off_p + i] if en_itca else 0.0
        qi = y[off_q + i] if en_itca else 0.0
        wi = y[off_w + i] if en_im else 0.0
        xi = y[off_x + i] if en_nap else 0.0
        yi = y[off_y + i] if en_nar else 0.0
        ji = y[off_j + i] if en_nar else 0.0
        zi = y[off_zsk + i] if en_sk else 0.0
        ca_i_val = y[off_ca + i] if dyn_ca else ca_rest
        atp_i_val = y[off_atp + i] if dyn_atp else atp_max_mM
        na_i_val = y[off_na_i + i] if off_na_i >= 0 else na_i_rest_mM
        k_o_val = y[off_k_o + i] if off_k_o >= 0 else k_o_rest_mM
        ena_i = nernst_na_ion(na_i_val, na_ext_mM, t_kelvin) if dyn_atp else ena
        ek_i = nernst_k_ion(k_i_mM, k_o_val, t_kelvin) if dyn_atp else ek

        i_ion, i_ca_influx = compute_ionic_currents_scalar(
            vi, mi, hi, ni,
            ri, si, ui, ai, bi, pi, qi, wi, xi, yi, ji, zi,
            en_ih, en_ica, en_ia, en_sk, en_itca, en_im, en_nap, en_nar, dyn_ca,
            gna_v[i], gk_v[i], gl_v[i], gih_v[i], gca_v[i], ga_v[i], gsk_v[i], gtca_v[i], gim_v[i], gnap_v[i], gnar_v[i],
            ena_i, ek_i, el, eih,
            ca_i_val, ca_ext, ca_rest, t_kelvin,
        )

        # Unified metabolism: pump current, K_ATP, and metabolic derivatives
        i_pump, i_katp, datp, dnai, dko = compute_metabolism_and_pump(
            vi, mi, hi, ni, xi, yi, ji, ai, bi, zi, wi,
            gna_v[i], gk_v[i], ga_v[i], gsk_v[i], gim_v[i], gnap_v[i], gnar_v[i],
            ena_i, ek_i,
            en_nap, en_nar, en_ia, en_sk, en_im, dyn_ca,
            g_katp_max, katp_kd_atp_mM,
            atp_i_val, atp_synthesis_rate,
            na_i_val, k_o_val, k_o_rest_mM,
            ion_drift_gain, k_o_clearance_tau_ms,
            i_ca_influx,
        )
        if dyn_atp:
            i_ion += i_katp

        # Axial coupling (sparse Laplacian row i)
        i_ax = 0.0
        for j_idx in range(l_indptr[i], l_indptr[i + 1]):
            col = l_indices[j_idx]
            i_ax += l_data[j_idx] * y[off_v + col]

        # Stimulus current: evaluate distributed contribution per-compartment.
        i_stim_primary = distributed_stimulus_current_for_comp(
            i, n_comp, base_current, stim_comp, stim_mode,
            use_dfilter_primary_eff, dfilter_attenuation, dfilter_tau_ms, i_filtered_primary,
        )
        if is_conductance_based:
            g_syn = i_stim_primary  # distributed conductance [mS/cmÂ˛]
            if is_nmda:
                g_syn *= nmda_mg_block(vi, mg_ext, nmda_mg_block_mM)
            i_stim_eff = -(g_syn * (vi - e_syn))
        else:
            i_stim_eff = i_stim_primary

        # Dual stim contribution
        if dual_stim_enabled == 1:
            i_stim_secondary = distributed_stimulus_current_for_comp(
                i, n_comp, base_current_2, stim_comp_2, stim_mode_2,
                use_dfilter_secondary_eff, dfilter_attenuation_2, dfilter_tau_ms_2, i_filtered_secondary,
            )
            if is_cond_2:
                g2 = i_stim_secondary
                if is_nmda_2:
                    g2 *= nmda_mg_block(vi, mg_ext, nmda_mg_block_mM)
                i_stim_eff -= g2 * (vi - e_syn_2)
            else:
                i_stim_eff += i_stim_secondary

        # dV/dt
        dydt[off_v + i] = (i_stim_eff - i_ion - i_pump + i_ax) / max(cm_v[i], 1e-12)

        # Gate derivatives (HH core) â€” per-compartment Q10 (Stage 6.2: thermal gradient)
        dydt[off_m + i] = phi_na[i] * (am(vi) * (1.0 - mi) - bm(vi) * mi)
        dydt[off_h + i] = phi_na[i] * (ah(vi) * (1.0 - hi) - bh(vi) * hi)
        dydt[off_n + i] = phi_k[i] * (an(vi) * (1.0 - ni) - bn(vi) * ni)

        # Optional gate derivatives
        if en_ih:
            ri = y[off_r + i]
            dydt[off_r + i] = phi_ih[i] * (ar_Ih(vi) * (1.0 - ri) - br_Ih(vi) * ri)

        if en_ica:
            si = y[off_s + i]
            ui = y[off_u + i]
            dydt[off_s + i] = phi_ca[i] * (as_Ca(vi) * (1.0 - si) - bs_Ca(vi) * si)
            dydt[off_u + i] = phi_ca[i] * (au_Ca(vi) * (1.0 - ui) - bu_Ca(vi) * ui)

        if en_ia:
            ai = y[off_a + i]
            bi = y[off_b + i]
            # A-current is a K channel â€” scale with phi_k (not phi_ia)
            dydt[off_a + i] = phi_k[i] * (aa_IA(vi) * (1.0 - ai) - ba_IA(vi) * ai)
            dydt[off_b + i] = phi_k[i] * (ab_IA(vi) * (1.0 - bi) - bb_IA(vi) * bi)

        if en_itca:
            pi = y[off_p + i]
            qi = y[off_q + i]
            # T-type Ca is a Ca channel â€” scale with phi_ca (not phi_tca)
            dydt[off_p + i] = phi_ca[i] * (am_TCa(vi) * (1.0 - pi) - bm_TCa(vi) * pi)
            dydt[off_q + i] = phi_ca[i] * (ah_TCa(vi) * (1.0 - qi) - bh_TCa(vi) * qi)

        if en_im:
            wi = y[off_w + i]
            # M-type K is a K channel â€” scale with phi_k (not phi_im)
            dydt[off_w + i] = (
                phi_k[i] * im_speed_multiplier * (aw_IM(vi) * (1.0 - wi) - bw_IM(vi) * wi)
            )

        if en_nap:
            xi = y[off_x + i]
            dydt[off_x + i] = phi_nap[i] * (ax_NaP(vi) * (1.0 - xi) - bx_NaP(vi) * xi)

        if en_nar:
            yi = y[off_y + i]
            ji = y[off_j + i]
            dydt[off_y + i] = phi_nar[i] * (ay_NaR(vi) * (1.0 - yi) - by_NaR(vi) * yi)
            dydt[off_j + i] = phi_nar[i] * (aj_NaR(vi) * (1.0 - ji) - bj_NaR(vi) * ji)

        # SK gate: dz/dt = (z_inf(Ca) - z) / tau_eff,  tau_eff = tau_SK / phi_k
        # Unified with hines.py analytic: z_new = z_inf + (z-z_inf)*exp(-dt/tau_eff)
        # ODE form here because BDF needs continuous derivatives.
        # Hirschberg et al. 1998, J Gen Physiol 111:565
        if en_sk:
            zi = y[off_zsk + i]
            if dyn_ca:
                ca_sk = effective_sk_calcium(y[off_ca + i], ca_rest)
            else:
                ca_sk = ca_rest
            z_inf = z_inf_SK(ca_sk)
            tau_eff = max(tau_sk, 1e-12) / max(phi_k[i], 1e-12)
            dydt[off_zsk + i] = (z_inf - zi) / tau_eff

        # Calcium dynamics
        if dyn_ca:
            ca_i_val = y[off_ca + i]
            dca = b_ca[i] * i_ca_influx - (ca_i_val - ca_rest) / tau_ca
            dydt[off_ca + i] = clamp_calcium_derivative(ca_i_val, dca)

        # Store metabolism derivatives (already computed by compute_metabolism_and_pump)
        if dyn_atp:
            dydt[off_atp + i] = datp
            if off_na_i >= 0:
                dydt[off_na_i + i] = dnai
            if off_k_o >= 0:
                dydt[off_k_o + i] = dko

    # Dendritic filter ODEs (outside main loop â€” not per-compartment)
    if use_dfilter_primary == 1:
        dydt[off_ifilt_primary] = d_ifiltered_dt_primary
    if use_dfilter_secondary == 1:
        dydt[off_ifilt_secondary] = d_ifiltered_dt_secondary

    return dydt

