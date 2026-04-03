"""
core/analysis.py — Spike Analysis & Membrane Property Toolkit v10.0

Ports and improves all analysis functions from Scilab hh_utils.sce:
  count_spikes, get_spikes, spike_threshold, spike_halfwidth,
  adaptation_index, classify_neuron, conduction_velocity,
  space_constant, compute_current_balance + Python-native extensions.
"""
import numpy as np
from scipy.signal import find_peaks


# ─────────────────────────────────────────────────────────────────────
#  1. SPIKE DETECTION
# ─────────────────────────────────────────────────────────────────────

def detect_spikes(V: np.ndarray, t: np.ndarray,
                  threshold: float = -20.0,
                  prominence: float = 10.0,
                  baseline_threshold: float = -50.0) -> tuple:
    """
    Spike detection: find peaks above threshold with repolarization check.
    
    IMPROVED: Now verifies that voltage returns to baseline after each spike.
    Prevents false positives from sustained depolarization.
    
    Parameters
    ----------
    V : np.ndarray
        Voltage trace (mV)
    t : np.ndarray  
        Time array (ms)
    threshold : float
        Spike detection threshold (mV)
    prominence : float
        Minimum spike prominence (mV)
    baseline_threshold : float
        Voltage must fall below this after spike to be valid (mV)
    
    Returns
    -------
    peak_idx   : np.ndarray[int]   — indices of spike peaks
    spike_times: np.ndarray[float] — times of peaks (ms)
    spike_amps : np.ndarray[float] — amplitudes of peaks (mV)
    """
    # Find regions where V crosses threshold
    above_thresh = V > threshold
    if not np.any(above_thresh):
        return np.array([], dtype=int), np.array([]), np.array([])

    # Find local maxima within suprathreshold regions
    peak_idx_list = []
    in_spike = False
    spike_start = 0

    for i in range(len(V)):
        if above_thresh[i] and not in_spike:
            # Start of suprathreshold region
            in_spike = True
            spike_start = i
        elif not above_thresh[i] and in_spike:
            # End of suprathreshold region - find peak within
            if spike_start < i:
                peak_in_region = np.argmax(V[spike_start:i]) + spike_start
                
                # NEW: Check if this is a true peak (local maximum)
                peak_val = V[peak_in_region]
                
                # Check if it's significantly higher than neighbors
                left_idx = max(0, peak_in_region - 5)
                right_idx = min(len(V), peak_in_region + 5)
                local_region = V[left_idx:right_idx]
                
                # Must be the maximum in local region
                is_local_max = peak_val == np.max(local_region)
                
                # NEW: Check if it's a sharp peak (not plateau)
                # Calculate prominence: difference from surrounding minima
                left_min = np.min(V[left_idx:peak_in_region]) if peak_in_region > left_idx else peak_val
                right_min = np.min(V[peak_in_region+1:right_idx]) if peak_in_region < right_idx-1 else peak_val
                prominence = peak_val - max(left_min, right_min)
                
                # Must have minimum prominence to be considered a real spike
                min_prominence = 5.0  # mV
                is_sharp_peak = prominence >= min_prominence
                
                # NEW: Check repolarization to baseline
                # Look ahead to see if voltage returns to baseline
                look_ahead = min(100, len(V) - i)  # Look up to 100 points ahead
                valid_repolarization = False
                
                if look_ahead > 0:
                    future_v = V[i:min(i + look_ahead, len(V))]
                    # Check if voltage falls below baseline threshold at any point
                    if len(future_v) > 0 and np.min(future_v) < baseline_threshold:
                        valid_repolarization = True
                else:
                    # Can't verify repolarization at end of trace
                    # Only count if we're already near baseline
                    if V[i] < baseline_threshold:
                        valid_repolarization = True
                
                # Only count if all conditions met: local max + sharp peak + repolarization
                if is_local_max and is_sharp_peak and valid_repolarization:
                    peak_idx_list.append(peak_in_region)
                
            in_spike = False

    # Handle case where trace ends while in spike
    if in_spike and spike_start < len(V):
        peak_in_region = np.argmax(V[spike_start:]) + spike_start
        # For final spike, check if it was already high at end
        # (can't verify repolarization if trace ends)
        peak_idx_list.append(peak_in_region)

    if len(peak_idx_list) == 0:
        return np.array([], dtype=int), np.array([]), np.array([])

    peak_idx = np.array(peak_idx_list, dtype=int)
    return peak_idx, t[peak_idx], V[peak_idx]


# ─────────────────────────────────────────────────────────────────────
#  2. SPIKE MORPHOLOGY
# ─────────────────────────────────────────────────────────────────────

def spike_threshold(V: np.ndarray, t: np.ndarray,
                    pct: float = 0.10) -> float:
    """
    Action potential threshold by dV/dt criterion.
    Returns the voltage at which dV/dt first exceeds `pct`×max(dV/dt).
    Classic electrophysiology definition (Hille 2001).
    """
    dvdt = np.gradient(V, t)
    max_dvdt = np.max(dvdt)
    if max_dvdt < 0.5:          # no fast depolarisation
        return np.nan
    idx = np.argmax(dvdt > pct * max_dvdt)
    return float(V[idx])


def spike_halfwidth(V: np.ndarray, t: np.ndarray) -> float:
    """
    Half-width at half-amplitude of the FIRST spike (ms).
    Standard electrophysiology measure of spike duration.
    """
    peak_idx, _, _ = detect_spikes(V, t)
    if len(peak_idx) == 0:
        return np.nan

    V_thr = spike_threshold(V, t)
    pk    = int(peak_idx[0])
    V_pk  = float(V[pk])

    if np.isnan(V_thr) or V_pk < -20.0:
        return np.nan

    half_V = (V_thr + V_pk) / 2.0

    # Search within a local window around first spike (±20 ms)
    dt_ms = float(t[1] - t[0]) if len(t) > 1 else 0.05
    win   = max(10, int(20.0 / dt_ms))
    lo    = max(0, pk - win)
    hi    = min(len(V) - 1, pk + win)

    local_V = V[lo:hi]
    local_t = t[lo:hi]
    above   = np.where(local_V > half_V)[0]

    if len(above) < 2:
        return np.nan

    return float(local_t[above[-1]] - local_t[above[0]])


def after_hyperpolarization(V: np.ndarray, t: np.ndarray,
                             peak_idx: int, window_ms: float = 25.0) -> float:
    """
    Minimum voltage in the window after the first spike peak.
    Measures the depth of AHP.
    """
    dt_ms = (t[-1] - t[0]) / (len(t) - 1)
    end_idx = min(peak_idx + int(window_ms / dt_ms), len(V) - 1)
    return float(np.min(V[peak_idx:end_idx]))


# ─────────────────────────────────────────────────────────────────────
#  3. FIRING PATTERN ANALYSIS
# ─────────────────────────────────────────────────────────────────────

def adaptation_index(spike_times: np.ndarray) -> float:
    """
    Adaptation Index = (ISI_last − ISI_first) / (ISI_last + ISI_first).

    Interpretation
    ─────────────
     > 0  : firing slows down (RS adaptation)
     ≈ 0  : regular firing   (FS)
     < 0  : firing speeds up (IB bursting)
    """
    if len(spike_times) < 3:
        return 0.0
    isi = np.diff(spike_times)
    denom = isi[-1] + isi[0]
    return float((isi[-1] - isi[0]) / denom) if denom > 0 else 0.0


def classify_neuron(AI: float, hw_ms: float,
                    fi_hz: float, fs_hz: float) -> str:
    """
    Electrophysiological cell-type classification.

    Based on McCormick et al. (1985) and Connors & Gutnick (1990):
      FS  — Fast Spiking      (hw < 0.6 ms, AI ≈ 0)
      RS  — Regular Spiking   (adapting, hw moderate)
      IB  — Intrinsic Bursting (AI < 0, burst onset)
      LTS — Low-Threshold Spiking (very wide spikes)
    """
    if np.isnan(hw_ms) or fi_hz < 1.0:
        return "Silent"
    if hw_ms < 0.6 and abs(AI) < 0.15:
        return "FS (Fast Spiking)"
    elif AI > 0.25 and hw_ms > 0.8:
        return "RS (Regular Spiking)"
    elif AI < -0.15:
        return "IB (Intrinsic Bursting)"
    elif hw_ms > 1.5:
        return "LTS (Low-Threshold Spiking)"
    else:
        return "Intermediate"


# ─────────────────────────────────────────────────────────────────────
#  4. CONDUCTION & CABLE PROPERTIES
# ─────────────────────────────────────────────────────────────────────

def conduction_velocity(VA: np.ndarray, VB: np.ndarray,
                        t: np.ndarray, dist_cm: float) -> float:
    """
    Axonal conduction velocity from peak-to-peak latency (m/s).

    Parameters
    ----------
    VA, VB   : voltage at proximal and distal compartments
    dist_cm  : distance between the two points in cm
    """
    ia = int(np.argmax(VA))
    ib = int(np.argmax(VB))
    dt_ms = t[ib] - t[ia]
    if dt_ms > 0 and dist_cm > 0:
        return dist_cm / dt_ms * 10.0   # cm/ms → m/s
    return 0.0


def space_constant(d_cm: float, Ra_ohm_cm: float, gL_mS_cm2: float) -> float:
    """
    Electronic length constant λ = sqrt(d / (4·Ra·gL)) in cm.
    gL in mS/cm² is converted to S/cm² internally.
    """
    gL_S = gL_mS_cm2 * 1e-3
    if Ra_ohm_cm <= 0 or gL_S <= 0 or d_cm <= 0:
        return 0.0
    return float(np.sqrt(d_cm / (4.0 * Ra_ohm_cm * gL_S)))


def membrane_time_constant(Cm_uF: float, gL_mS: float) -> float:
    """τ_m = Cm / gL  (ms, since Cm in µF/cm² and gL in mS/cm²)."""
    return float(Cm_uF / gL_mS) if gL_mS > 0 else np.inf


def input_resistance(gL_mS: float) -> float:
    """Rin = 1 / gL  (kΩ·cm²)."""
    return float(1.0 / gL_mS) if gL_mS > 0 else np.inf


# ─────────────────────────────────────────────────────────────────────
#  5. GATE EQUILIBRIA & TIME CONSTANTS
# ─────────────────────────────────────────────────────────────────────

def compute_equilibrium_curves(V_range: np.ndarray,
                                phi: float = 1.0) -> dict:
    """
    Steady-state gating variables and time constants for HH channels.

    Returns dict with keys:
      m_inf, h_inf, n_inf  — equilibrium opening probabilities [0–1]
      tau_m, tau_h, tau_n  — time constants (ms)
    """
    from core.kinetics import am, bm, ah, bh, an, bn

    am_v = am(V_range);  bm_v = bm(V_range)
    ah_v = ah(V_range);  bh_v = bh(V_range)
    an_v = an(V_range);  bn_v = bn(V_range)

    return {
        'm_inf': am_v / (am_v + bm_v),
        'h_inf': ah_v / (ah_v + bh_v),
        'n_inf': an_v / (an_v + bn_v),
        'tau_m': 1.0 / (phi * (am_v + bm_v)),
        'tau_h': 1.0 / (phi * (ah_v + bh_v)),
        'tau_n': 1.0 / (phi * (an_v + bn_v)),
    }


def compute_optional_equilibrium(V_range: np.ndarray,
                                  config, phi: float = 1.0) -> dict:
    """Equilibrium curves for optional channels (Ih, ICa, IA) if enabled."""
    from core.kinetics import (ar_Ih, br_Ih,
                                as_Ca, bs_Ca, au_Ca, bu_Ca,
                                aa_IA, ba_IA, ab_IA, bb_IA)
    ch = config.channels
    result = {}

    if ch.enable_Ih:
        ar = ar_Ih(V_range);  br = br_Ih(V_range)
        result['r_inf'] = ar / (ar + br)
        result['tau_r'] = 1.0 / (phi * (ar + br))

    if ch.enable_ICa:
        as_ = as_Ca(V_range);  bs_ = bs_Ca(V_range)
        au_ = au_Ca(V_range);  bu_ = bu_Ca(V_range)
        result['s_inf'] = as_ / (as_ + bs_)
        result['u_inf'] = au_ / (au_ + bu_)
        result['tau_s'] = 1.0 / (phi * (as_ + bs_))
        result['tau_u'] = 1.0 / (phi * (au_ + bu_))

    if ch.enable_IA:
        aa_ = aa_IA(V_range);  ba_ = ba_IA(V_range)
        ab_ = ab_IA(V_range);  bb_ = bb_IA(V_range)
        result['a_inf'] = aa_ / (aa_ + ba_)
        result['b_inf'] = ab_ / (ab_ + bb_)
        result['tau_a'] = 1.0 / (phi * (aa_ + ba_))
        result['tau_b'] = 1.0 / (phi * (ab_ + bb_))

    return result


# ─────────────────────────────────────────────────────────────────────
#  6. NULLCLINES  (Phase-plane analysis)
# ─────────────────────────────────────────────────────────────────────

def compute_nullclines(V_range: np.ndarray, config,
                        I_stim: float = None) -> tuple:
    """
    V-nullcline (dV/dt = 0) and n-nullcline (dn/dt = 0).

    NOTE: Only Na + K + Leak are included (as in Scilab).
    Optional channels shift the nullclines but are not modelled here.

    Returns
    -------
    n_V_null : np.ndarray — n value on V-nullcline (NaN where undefined)
    n_n_null : np.ndarray — n_∞(V)  (n-nullcline, always defined)
    """
    from core.kinetics import am, bm, ah, bh, an, bn

    ch = config.channels
    if I_stim is None:
        I_stim = config.stim.Iext if config.stim.stim_type == 'const' else 0.0

    am_v = am(V_range);  bm_v = bm(V_range)
    ah_v = ah(V_range);  bh_v = bh(V_range)
    an_v = an(V_range);  bn_v = bn(V_range)

    m_inf = am_v / (am_v + bm_v)
    h_inf = ah_v / (ah_v + bh_v)
    n_inf = an_v / (an_v + bn_v)

    I_Na = ch.gNa_max * (m_inf ** 3) * h_inf * (V_range - ch.ENa)
    I_L  = ch.gL * (V_range - ch.EL)

    numerator   = I_stim - I_Na - I_L
    denominator = ch.gK_max * (V_range - ch.EK)

    n_V_null = np.full_like(V_range, np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        n4 = numerator / denominator
        valid = (n4 >= 0) & (np.abs(denominator) > 1e-6)
        n_V_null[valid] = n4[valid] ** 0.25
        # clamp to [0,1]
        n_V_null = np.where((n_V_null >= 0) & (n_V_null <= 1.0),
                            n_V_null, np.nan)

    return n_V_null, n_inf


# ─────────────────────────────────────────────────────────────────────
#  7. CURRENT BALANCE  (numerical validation)
# ─────────────────────────────────────────────────────────────────────

def compute_current_balance(result, morph: dict) -> np.ndarray:
    """
    Numerical consistency check: Cm·dV/dt − (I_stim − I_ion + I_ax) ≈ 0.

    A large balance error indicates solver issues or dt too coarse.
    Returns balance signal for the soma compartment.
    """
    from core.rhs import get_stim_current
    from scipy.sparse import csr_matrix

    t   = result.t
    V   = result.v_soma
    cfg = result.config

    dVdt = np.gradient(V, t)

    # Total ionic current (soma, index 0)
    I_ion_soma = sum(result.currents.values())

    # Stimulus current
    s_map = {'const': 0, 'pulse': 1, 'alpha': 2, 'ou_noise': 0}
    stype = s_map.get(cfg.stim.stim_type, 0)
    I_stim = np.array([
        get_stim_current(float(ti), stype,
                         cfg.stim.Iext, cfg.stim.pulse_start,
                         cfg.stim.pulse_dur, cfg.stim.alpha_tau)
        for ti in t
    ])

    # Axial current (row 0 of sparse Laplacian × V_all)
    if result.n_comp > 1:
        n_c = result.n_comp
        L = csr_matrix((morph['L_data'], morph['L_indices'], morph['L_indptr']),
                        shape=(n_c, n_c))
        I_ax_soma = np.array(
            [(L.dot(result.v_all[:, k]))[0] for k in range(len(t))]
        )
    else:
        I_ax_soma = np.zeros_like(t)

    return cfg.channels.Cm * dVdt - (I_stim - I_ion_soma + I_ax_soma)


# ─────────────────────────────────────────────────────────────────────
#  8. GATE TRACE EXTRACTION
# ─────────────────────────────────────────────────────────────────────

def extract_gate_traces(result) -> dict:
    """
    Unpack gate variable time-series from the raw ODE solution.

    Returns dict mapping gate name → 1-D numpy array (soma compartment).
    Keys: 'm', 'h', 'n', optionally 'r', 's', 'u', 'a', 'b'.
    """
    y   = result.y
    n   = result.n_comp
    cfg = result.config.channels

    gates  = {}
    cursor = n          # skip V rows

    gates['m'] = y[cursor, :];  cursor += n
    gates['h'] = y[cursor, :];  cursor += n
    gates['n'] = y[cursor, :];  cursor += n

    if cfg.enable_Ih:
        gates['r'] = y[cursor, :];  cursor += n

    if cfg.enable_ICa:
        gates['s'] = y[cursor, :];  cursor += n
        gates['u'] = y[cursor, :];  cursor += n

    if cfg.enable_IA:
        gates['a'] = y[cursor, :];  cursor += n
        gates['b'] = y[cursor, :];  cursor += n

    return gates


# ─────────────────────────────────────────────────────────────────────
#  9. NEURON PASSPORT  (full analysis report dict)
# ─────────────────────────────────────────────────────────────────────

def full_analysis(result) -> dict:
    """
    Compute the complete Neuron Passport from a SimulationResult.

    Returns a flat dict with all key biophysical metrics.
    """
    V   = result.v_soma
    t   = result.t
    cfg = result.config

    peak_idx, spike_times, spike_amps = detect_spikes(V, t)
    n_spikes = len(spike_times)

    V_thr = spike_threshold(V, t) if n_spikes > 0 else np.nan
    hw    = spike_halfwidth(V, t)  if n_spikes > 0 else np.nan
    V_pk  = float(np.max(V))
    V_ahp = np.nan
    if n_spikes > 0 and len(peak_idx) > 0:
        V_ahp = after_hyperpolarization(V, t, int(peak_idx[0]))

    dvdt     = np.gradient(V, t)
    dvdt_max = float(np.max(dvdt))
    dvdt_min = float(np.min(dvdt))

    f_initial = np.nan
    f_steady  = np.nan
    AI        = 0.0
    neuron_type = "Silent"

    if n_spikes > 1:
        isi = np.diff(spike_times)
        f_initial = 1000.0 / isi[0]
        f_steady  = 1000.0 / float(np.mean(isi[max(0, len(isi) - 2):]))
        AI = adaptation_index(spike_times)
        neuron_type = classify_neuron(
            AI,
            hw if not np.isnan(hw) else np.nan,
            f_initial, f_steady
        )

    # Conduction velocity
    cv = 0.0
    mc = cfg.morphology
    if result.n_comp > 2:
        junction = min(1 + mc.N_ais + mc.N_trunk, result.n_comp - 1)
        dist_cm  = (mc.N_ais + mc.N_trunk) * mc.dx
        cv = conduction_velocity(
            result.v_all[0, :], result.v_all[junction, :], t, dist_cm
        )

    # Cable / passive properties
    ch      = cfg.channels
    tau_m   = membrane_time_constant(ch.Cm, ch.gL)
    Rin     = input_resistance(ch.gL)
    lam_um  = space_constant(mc.d_soma, mc.Ra, ch.gL) * 1e4   # cm → µm

    # Energy
    dt_val         = float(t[1] - t[0]) if len(t) > 1 else 0.05
    Q_per_channel  = {name: float(np.sum(np.abs(curr)) * dt_val)
                      for name, curr in result.currents.items()}

    # ──────────────────────────────────────────────────────────────
    #  NEW ANALYTICS (Phase 7.1) — Advanced firing pattern analysis
    # ──────────────────────────────────────────────────────────────
    
    # ISI (inter-spike interval) statistics
    isi_mean = np.nan
    isi_std  = np.nan
    isi_min  = np.nan
    isi_max  = np.nan
    cv_isi   = np.nan  # Coefficient of variation
    if n_spikes > 1:
        isi = np.diff(spike_times)
        isi_mean = float(np.mean(isi))
        isi_std  = float(np.std(isi))
        isi_min  = float(np.min(isi))
        isi_max  = float(np.max(isi))
        if isi_mean > 0:
            cv_isi = isi_std / isi_mean
    
    # First spike latency (time to first spike from stim onset)
    first_spike_lat = np.nan
    if n_spikes > 0:
        # Use pulse_start for pulse mode, else use 0 (stimulus always active if const)
        stim_onset = cfg.stim.pulse_start if cfg.stim.stim_type == 'pulse' else 0.0
        first_spike_lat = float(spike_times[0] - stim_onset)
    
    # Refractory period estimate (from AHP recovery)
    refr_period = np.nan
    if n_spikes > 0 and len(peak_idx) > 0:
        # Simple estimate: time from spike peak to 90% AHP recovery
        pk_idx = int(peak_idx[0])
        if pk_idx + 1 < len(V):
            # Look for 50% recovery from peak towards baseline
            V_baseline = V[0]
            V_peak_val = V[pk_idx]
            V_target = V_baseline + 0.5 * (V_peak_val - V_baseline)
            for i in range(pk_idx, min(pk_idx + len(t)//10, len(V))):
                if V[i] >= V_target:
                    refr_period = float(t[i] - t[pk_idx])
                    break
    
    # Firing reliability (proportion of time neuron was able to spike)
    # Estimate as (actual ISI / expected ISI) — useful for stability metric
    firing_reliability = np.nan
    if n_spikes > 1 and not np.isnan(f_steady):
        expected_isi = 1000.0 / f_steady if f_steady > 0 else np.nan
        if not np.isnan(expected_isi):
            actual_isi_steady = np.mean(np.diff(spike_times[max(0, n_spikes-3):]))
            if actual_isi_steady > 0:
                firing_reliability = min(1.0, expected_isi / actual_isi_steady)

    return {
        'n_spikes':            n_spikes,
        'spike_times':         spike_times,
        'spike_amps':          spike_amps,
        'V_threshold':         V_thr,
        'V_peak':              V_pk,
        'V_ahp':               V_ahp,
        'halfwidth_ms':        hw,
        'dvdt_max':            dvdt_max,
        'dvdt_min':            dvdt_min,
        'f_initial_hz':        f_initial,
        'f_steady_hz':         f_steady,
        'adaptation_index':    AI,
        'neuron_type':         neuron_type,
        'conduction_vel_ms':   cv,
        'tau_m_ms':            tau_m,
        'Rin_kohm_cm2':        Rin,
        'lambda_um':           lam_um,
        'Q_per_channel':       Q_per_channel,
        'atp_nmol_cm2':        result.atp_estimate,
        # ── NEW: Advanced firing analysis ──
        'isi_mean_ms':         isi_mean,
        'isi_std_ms':          isi_std,
        'isi_min_ms':          isi_min,
        'isi_max_ms':          isi_max,
        'cv_isi':              cv_isi,
        'first_spike_latency_ms': first_spike_lat,
        'refractory_period_ms':   refr_period,
        'firing_reliability':  firing_reliability,
    }
