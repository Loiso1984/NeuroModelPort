"""
core/analysis.py — Spike Analysis & Membrane Property Toolkit v10.0

Ports and improves all analysis functions from Scilab hh_utils.sce:
  count_spikes, get_spikes, spike_threshold, spike_halfwidth,
  adaptation_index, classify_neuron, conduction_velocity,
  space_constant, compute_current_balance + Python-native extensions.
"""
import numpy as np
from numba import njit
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt
from scipy.spatial import cKDTree


# ─────────────────────────────────────────────────────────────────────
#  1. SPIKE DETECTION
# ─────────────────────────────────────────────────────────────────────

# FSM states (Stage 3.1 — AIDER_PLAN)
_FSM_RESTING = 0
_FSM_DEPOLARIZING = 1
_FSM_REFRACTORY = 2


@njit(cache=True)
def _fsm_detect_spikes(V, t, threshold, baseline_threshold, refractory_ms):
    """
    Numba-jitted Finite State Machine spike detector.

    States
    ------
    RESTING       → voltage below threshold; waiting for upswing.
    DEPOLARIZING  → voltage crossed threshold; tracking peak.
    REFRACTORY    → spike confirmed (repolarised below baseline);
                    waiting out refractory period before returning to RESTING.

    A spike is counted only when voltage:
      1. crosses *threshold* from below  (RESTING → DEPOLARIZING)
      2. reaches a local peak            (tracked in DEPOLARIZING)
      3. strictly falls below *baseline_threshold*  (→ REFRACTORY, spike recorded)
    """
    n = len(V)
    # Pre-allocate output arrays (max possible = n)
    peak_indices = np.empty(n, dtype=np.int64)
    count = 0

    state = _FSM_RESTING
    peak_idx = 0
    peak_val = -1e9
    refract_end_t = -1e9

    for i in range(n):
        vi = V[i]
        ti = t[i]

        if state == _FSM_RESTING:
            if vi >= threshold:
                state = _FSM_DEPOLARIZING
                peak_val = vi
                peak_idx = i

        elif state == _FSM_DEPOLARIZING:
            if vi > peak_val:
                peak_val = vi
                peak_idx = i
            if vi < baseline_threshold:
                # Spike confirmed — record peak
                peak_indices[count] = peak_idx
                count += 1
                state = _FSM_REFRACTORY
                refract_end_t = ti + refractory_ms

        elif state == _FSM_REFRACTORY:
            if ti >= refract_end_t and vi < threshold:
                state = _FSM_RESTING

    return peak_indices[:count]

def detect_spikes(V: np.ndarray, t: np.ndarray,
                  threshold: float = -20.0,
                  prominence: float = 10.0,
                  baseline_threshold: float = -50.0,
                  repolarization_window_ms: float = 20.0,
                  refractory_ms: float = 1.0,
                  algorithm: str = "peak_repolarization") -> tuple:
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
    if len(V) == 0 or len(t) == 0:
        return np.array([], dtype=int), np.array([]), np.array([])

    if len(V) != len(t):
        import logging
        length_diff = abs(len(V) - len(t))
        if length_diff > 10:  # Significant mismatch
            logging.error(f"Significant array length mismatch: V={len(V)}, t={len(t)} (diff={length_diff}). "
                         "This indicates serious data integrity issues.")
            raise ValueError(f"Array length mismatch too large ({length_diff} points). "
                           f"Voltage: {len(V)}, Time: {len(t)}")
        else:
            logging.warning(f"Minor array length mismatch: V={len(V)}, t={len(t)} (diff={length_diff}). "
                           "Truncating to minimum length.")
        # Truncate to minimum length but warn user
        n = min(len(V), len(t))
        V = V[:n]
        t = t[:n]
        if n < 10:
            logging.error("Array length too short after truncation (< 10 points). Results may be unreliable.")
            raise ValueError("Arrays too short after truncation for reliable analysis")

    dt_ms = float(np.median(np.diff(t))) if len(t) > 1 else 0.05
    dt_ms = max(dt_ms, 1e-6)

    min_distance = max(1, int(round(refractory_ms / dt_ms)))
    repol_window_pts = max(1, int(round(repolarization_window_ms / dt_ms)))

    # ── FSM algorithm: Numba-jitted state machine (Stage 3.1) ──
    if algorithm == "fsm":
        V_c = np.ascontiguousarray(V, dtype=np.float64)
        t_c = np.ascontiguousarray(t, dtype=np.float64)
        valid_idx = _fsm_detect_spikes(
            V_c, t_c, threshold, baseline_threshold, refractory_ms
        )
        if len(valid_idx) == 0:
            return np.array([], dtype=int), np.array([]), np.array([])
        return valid_idx, t[valid_idx], V[valid_idx]

    if algorithm == "threshold_crossing":
        above = V >= threshold
        crossings = np.where((~above[:-1]) & (above[1:]))[0] + 1
        if len(crossings) == 0:
            return np.array([], dtype=int), np.array([]), np.array([])

        keep = [int(crossings[0])]
        for i in range(1, len(crossings)):
            if t[crossings[i]] - t[keep[-1]] >= refractory_ms:
                keep.append(int(crossings[i]))
        candidate_idx = np.array(keep, dtype=int)
    else:
        candidate_idx, _ = find_peaks(
            V,
            height=threshold,
            prominence=max(prominence, 0.0),
            distance=min_distance,
        )

    if len(candidate_idx) == 0:
        return np.array([], dtype=int), np.array([]), np.array([])

    valid = []
    for candidate in candidate_idx:
        # For threshold-crossing mode convert crossing to local peak index.
        if algorithm == "threshold_crossing":
            local_end = min(len(V), candidate + repol_window_pts + 1)
            if local_end <= candidate:
                continue
            pk = int(candidate + np.argmax(V[candidate:local_end]))
        else:
            pk = int(candidate)
        end = min(len(V), pk + repol_window_pts + 1)
        if np.min(V[pk:end]) < baseline_threshold:
            valid.append(int(pk))
            continue
        # Edge case: near trace end we may not have full window.
        if end == len(V) and V[-1] < threshold:
            valid.append(int(pk))

    if len(valid) == 0:
        return np.array([], dtype=int), np.array([]), np.array([])

    valid_idx = np.array(valid, dtype=int)
    return valid_idx, t[valid_idx], V[valid_idx]


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


def spike_halfwidth(
    V: np.ndarray,
    t: np.ndarray,
    detect_threshold: float = -20.0,
    detect_prominence: float = 10.0,
    detect_baseline_threshold: float = -50.0,
    detect_repolarization_window_ms: float = 20.0,
    detect_refractory_ms: float = 1.0,
    detect_algorithm: str = "peak_repolarization",
) -> float:
    """
    Half-width at half-amplitude of the FIRST spike (ms).
    Standard electrophysiology measure of spike duration.
    """
    peak_idx, _, _ = detect_spikes(
        V,
        t,
        threshold=detect_threshold,
        prominence=detect_prominence,
        baseline_threshold=detect_baseline_threshold,
        repolarization_window_ms=detect_repolarization_window_ms,
        refractory_ms=detect_refractory_ms,
        algorithm=detect_algorithm,
    )
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


_NEURON_ML_PROTOTYPES = {
    # feature units:
    # fi_hz, fs_hz, ai, hw_ms, cv_isi
    "FS":  {"fi_hz": 180.0, "fs_hz": 160.0, "ai": 0.05, "hw_ms": 0.40, "cv_isi": 0.10},
    "RS":  {"fi_hz": 35.0,  "fs_hz": 15.0,  "ai": 0.40, "hw_ms": 1.00, "cv_isi": 0.25},
    "IB":  {"fi_hz": 80.0,  "fs_hz": 30.0,  "ai": -0.30, "hw_ms": 0.90, "cv_isi": 0.45},
    "LTS": {"fi_hz": 20.0,  "fs_hz": 8.0,   "ai": 0.20, "hw_ms": 1.80, "cv_isi": 0.30},
}

_NEURON_ML_SCALES = {
    "fi_hz": 120.0,
    "fs_hz": 100.0,
    "ai": 0.60,
    "hw_ms": 1.20,
    "cv_isi": 0.50,
}


def _rule_label_to_code(rule_label: str) -> str:
    if rule_label.startswith("FS"):
        return "FS"
    if rule_label.startswith("RS"):
        return "RS"
    if rule_label.startswith("IB"):
        return "IB"
    if rule_label.startswith("LTS"):
        return "LTS"
    if rule_label.startswith("Silent"):
        return "Silent"
    return "Intermediate"


def classify_neuron_ml(
    fi_hz: float,
    fs_hz: float,
    ai: float,
    hw_ms: float,
    cv_isi: float,
) -> tuple[str, float]:
    """
    Lightweight prototype-based classifier.

    Returns (label_code, confidence_0_1).
    """
    feat = {
        "fi_hz": fi_hz,
        "fs_hz": fs_hz,
        "ai": ai,
        "hw_ms": hw_ms,
        "cv_isi": cv_isi,
    }
    valid_keys = [k for k, v in feat.items() if np.isfinite(v)]
    if len(valid_keys) < 3:
        return "Intermediate", 0.0

    dists = {}
    for cls, proto in _NEURON_ML_PROTOTYPES.items():
        d2 = 0.0
        for k in valid_keys:
            scale = _NEURON_ML_SCALES[k]
            dv = (feat[k] - proto[k]) / scale
            d2 += dv * dv
        dists[cls] = np.sqrt(d2 / len(valid_keys))

    ranking = sorted(dists.items(), key=lambda kv: kv[1])
    best_cls, best_d = ranking[0]
    second_d = ranking[1][1] if len(ranking) > 1 else best_d + 1.0
    # Confidence from margin in normalized feature space.
    conf = 1.0 / (1.0 + max(0.0, best_d))
    margin = max(0.0, second_d - best_d)
    conf = float(np.clip(conf * (0.7 + 0.3 * np.tanh(2.0 * margin)), 0.0, 1.0))
    return best_cls, conf


def classify_neuron_hybrid(
    rule_label: str,
    fi_hz: float,
    fs_hz: float,
    ai: float,
    hw_ms: float,
    cv_isi: float,
) -> tuple[str, str, float]:
    """
    Hybrid classifier = rules + prototype-ML.

    Returns:
    - hybrid label (human-readable)
    - source tag (rule_only / ml_only / rule+ml / rule_priority)
    - confidence in [0, 1]
    """
    rule_code = _rule_label_to_code(rule_label)
    if rule_code == "Silent":
        return "Silent", "rule_only", 1.0

    ml_code, ml_conf = classify_neuron_ml(fi_hz, fs_hz, ai, hw_ms, cv_isi)
    if ml_conf <= 0.0:
        return rule_label, "rule_only", 0.50

    if rule_code in {"FS", "RS", "IB", "LTS"} and ml_code == rule_code:
        return rule_label, "rule+ml", float(np.clip(0.65 + 0.35 * ml_conf, 0.0, 1.0))

    if rule_code == "Intermediate" and ml_conf >= 0.65:
        return f"{ml_code} (ML)", "ml_only", ml_conf

    if ml_conf >= 0.85:
        return f"{ml_code} (ML override)", "ml_only", ml_conf

    return rule_label, "rule_priority", float(np.clip(0.55 + 0.25 * ml_conf, 0.0, 1.0))


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
    s_map = {'const': 0, 'pulse': 1, 'alpha': 2, 'ou_noise': 0, 'zap': 10}
    stype = s_map.get(cfg.stim.stim_type, 0)
    I_stim = np.array([
        get_stim_current(
            float(ti), stype,
            cfg.stim.Iext, cfg.stim.pulse_start,
            cfg.stim.pulse_dur, cfg.stim.alpha_tau,
            float(getattr(cfg.stim, "zap_f0_hz", 0.5)),
            float(getattr(cfg.stim, "zap_f1_hz", 40.0)),
        )
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


def classify_lyapunov(ftle_per_ms: float, tol_per_ms: float = 1e-3) -> str:
    """Qualitative regime label from FTLE/LLE estimate."""
    if np.isnan(ftle_per_ms):
        return "unknown"
    if ftle_per_ms > tol_per_ms:
        return "unstable_or_chaotic"
    if ftle_per_ms < -tol_per_ms:
        return "stable"
    return "limit_cycle_like"


def estimate_ftle_lle(
    x: np.ndarray,
    t: np.ndarray,
    *,
    embedding_dim: int = 3,
    lag_steps: int = 2,
    min_separation_ms: float = 10.0,
    fit_start_ms: float = 5.0,
    fit_end_ms: float = 40.0,
) -> dict:
    """
    Estimate FTLE/LLE from scalar time-series using Rosenstein-style divergence.
    """
    out = {
        "lle_per_ms": np.nan,
        "lle_per_s": np.nan,
        "fit_window_ms": (fit_start_ms, fit_end_ms),
        "valid_pairs": 0,
        "ftle_time_ms": np.array([]),
        "ftle_log_divergence": np.array([]),
    }

    if len(x) < 50 or len(t) < 50:
        return out

    n = min(len(x), len(t))
    x = np.asarray(x[:n], dtype=float)
    t = np.asarray(t[:n], dtype=float)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(t)):
        return out

    # Stage 3.3 — LLE protection: require t_sim > 1000 ms and discard
    # first 200 ms of transients before attractor reconstruction.
    t_sim_ms = float(t[-1] - t[0])
    if t_sim_ms < 1000.0:
        return out

    transient_ms = 200.0
    transient_mask = t >= (t[0] + transient_ms)
    if np.sum(transient_mask) < 50:
        return out
    x = x[transient_mask]
    t = t[transient_mask]
    n = len(x)

    dt_ms = float(np.median(np.diff(t))) if len(t) > 1 else 0.1
    dt_ms = max(dt_ms, 1e-9)

    embedding_dim = max(2, int(embedding_dim))
    lag_steps = max(1, int(lag_steps))
    max_start = n - (embedding_dim - 1) * lag_steps
    if max_start <= 30:
        return out

    x_std = float(np.std(x))
    if x_std < 1e-12:
        return out
    xn = (x - float(np.mean(x))) / x_std

    emb = np.empty((max_start, embedding_dim), dtype=float)
    for d in range(embedding_dim):
        emb[:, d] = xn[d * lag_steps : d * lag_steps + max_start]

    tree = cKDTree(emb)
    min_sep_steps = int(round(min_separation_ms / dt_ms))

    max_k = min(250, max_start // 3)
    if max_k < 8:
        return out

    pair_i = []
    pair_j = []
    for i in range(max_start - max_k):
        dists, idxs = tree.query(emb[i], k=min(20, max_start))
        del dists
        if np.isscalar(idxs):
            continue
        chosen = None
        for cand in idxs[1:]:
            if abs(int(cand) - i) >= min_sep_steps:
                chosen = int(cand)
                break
        if chosen is None or chosen + max_k >= max_start:
            continue
        pair_i.append(i)
        pair_j.append(chosen)

    if len(pair_i) < 20:
        return out

    pair_i = np.asarray(pair_i, dtype=int)
    pair_j = np.asarray(pair_j, dtype=int)

    mean_log_div = np.full(max_k, np.nan, dtype=float)
    for k in range(max_k):
        diff = emb[pair_i + k] - emb[pair_j + k]
        dist = np.linalg.norm(diff, axis=1)
        dist = np.maximum(dist, 1e-12)
        mean_log_div[k] = float(np.mean(np.log(dist)))

    times_ms = np.arange(max_k, dtype=float) * dt_ms
    fit_mask = (times_ms >= fit_start_ms) & (times_ms <= fit_end_ms) & np.isfinite(mean_log_div)
    if np.sum(fit_mask) < 6:
        return out

    slope, intercept = np.polyfit(times_ms[fit_mask], mean_log_div[fit_mask], 1)
    del intercept

    out["lle_per_ms"] = float(slope)
    out["lle_per_s"] = float(slope * 1000.0)
    out["valid_pairs"] = int(len(pair_i))
    out["ftle_time_ms"] = times_ms
    out["ftle_log_divergence"] = mean_log_div
    return out


# ─────────────────────────────────────────────────────────────────────
#  9. NEURON PASSPORT  (full analysis report dict)
# ─────────────────────────────────────────────────────────────────────

def _stim_type_to_code(stim_type: str) -> int:
    """Map textual stimulus type to the RHS integer code."""
    return {
        "const": 0,
        "pulse": 1,
        "alpha": 2,
        "ou_noise": 3,
        "AMPA": 4,
        "NMDA": 5,
        "GABAA": 6,
        "GABAB": 7,
        "Kainate": 8,
        "Nicotinic": 9,
        "zap": 10,
    }.get(stim_type, 0)


def _reconstruct_stimulus_proxy(result) -> np.ndarray:
    """
    Build a deterministic low-frequency stimulus proxy for modulation analysis.

    Priority:
    1) If dendritic filtering state exists in solution, use that directly.
    2) Otherwise reconstruct from configured stimulus equations including event_times.
    """
    from core.rhs import get_stim_current, get_event_driven_conductance
    from core.solver import generate_effective_event_times

    t = np.asarray(result.t, dtype=float)
    n = len(t)
    stim = np.zeros(n, dtype=float)
    if n == 0:
        return stim

    cfg = result.config
    mode = getattr(cfg.stim_location, "location", "soma")
    stype = _stim_type_to_code(cfg.stim.stim_type)

    if (
        mode == "dendritic_filtered"
        and cfg.dendritic_filter.enabled
        and getattr(result, "v_dendritic_filtered", None) is not None
    ):
        vd = np.asarray(result.v_dendritic_filtered, dtype=float)
        stim += vd[:n]
    else:
        attenuation = 1.0
        if mode == "dendritic_filtered" and cfg.dendritic_filter.space_constant_um > 0.0:
            attenuation = float(
                np.exp(-cfg.dendritic_filter.distance_um / cfg.dendritic_filter.space_constant_um)
            )
        
        # Get event_times - only use train generator for synaptic types (4-9)
        if 4 <= stype <= 9:
            event_times_arr = generate_effective_event_times(
                getattr(cfg.stim, "synaptic_train_type", "none"),
                getattr(cfg.stim, "synaptic_train_freq_hz", 40.0),
                getattr(cfg.stim, "synaptic_train_duration_ms", 200.0),
                cfg.stim.pulse_start,
                getattr(cfg.stim, "event_times", [])
            )
        else:
            # For non-synaptic types, just use manual event_times directly
            event_times_arr = np.array(getattr(cfg.stim, "event_times", []), dtype=np.float64)
        n_events = len(event_times_arr)
        
        for i, ti in enumerate(t):
            # For synaptic types (4-9), use event-driven conductance if event_times are provided
            if 4 <= stype <= 9:
                if n_events > 0:
                    base = get_event_driven_conductance(
                        float(ti),
                        stype,
                        float(cfg.stim.Iext),
                        event_times_arr,
                        n_events,
                        float(cfg.stim.alpha_tau),
                    )
                    # Convert conductance to current (simplified for preview)
                    # Use reversal potential based on type
                    if stype == 6:  # GABA-A
                        e_rev = -70.0  # mV
                    elif stype == 7:  # GABA-B
                        e_rev = -95.0  # mV
                    else:  # Excitatory
                        e_rev = 0.0  # mV
                    # Assume V ~ -65 mV for preview
                    stim[i] += attenuation * abs(float(base)) * (e_rev - (-65.0))
                # If synaptic type but no event times, produce no stimulation (skip)
            else:
                # Non-synaptic types use regular stimulus current
                base = get_stim_current(
                    float(ti),
                    stype,
                    float(cfg.stim.Iext),
                    float(cfg.stim.pulse_start),
                    float(cfg.stim.pulse_dur),
                    float(cfg.stim.alpha_tau),
                    float(getattr(cfg.stim, "zap_f0_hz", 0.5)),
                    float(getattr(cfg.stim, "zap_f1_hz", 40.0)),
                )
                stim[i] += attenuation * float(base)

    dual_cfg = getattr(cfg, "dual_stimulation", None)
    if dual_cfg is not None and getattr(dual_cfg, "enabled", False):
        stype_2 = _stim_type_to_code(getattr(dual_cfg, "secondary_stim_type", "const"))
        mode_2 = getattr(dual_cfg, "secondary_location", "soma")
        attenuation_2 = 1.0
        if mode_2 == "dendritic_filtered":
            space_const = float(getattr(dual_cfg, "secondary_space_constant_um", 0.0))
            dist = float(getattr(dual_cfg, "secondary_distance_um", 0.0))
            if space_const > 0.0:
                attenuation_2 = float(np.exp(-dist / space_const))
        
        # Get event_times for secondary - only use train generator for synaptic types (4-9)
        if 4 <= stype_2 <= 9:
            event_times_arr_2 = generate_effective_event_times(
                getattr(dual_cfg, 'secondary_train_type', 'none'),
                getattr(dual_cfg, 'secondary_train_freq_hz', 40.0),
                getattr(dual_cfg, 'secondary_train_duration_ms', 200.0),
                getattr(dual_cfg, 'secondary_start', 0.0),
                getattr(dual_cfg, 'secondary_event_times', [])
            )
        else:
            # For non-synaptic types, just use manual event_times directly
            event_times_arr_2 = np.array(getattr(dual_cfg, 'secondary_event_times', []), dtype=np.float64)
        n_events_2 = len(event_times_arr_2)
        
        for i, ti in enumerate(t):
            # For synaptic types (4-9), use event-driven conductance if event_times are provided
            if 4 <= stype_2 <= 9:
                if n_events_2 > 0:
                    base_2 = get_event_driven_conductance(
                        float(ti),
                        stype_2,
                        float(getattr(dual_cfg, "secondary_Iext", 0.0)),
                        event_times_arr_2,
                        n_events_2,
                        float(getattr(dual_cfg, "secondary_alpha_tau", 2.0)),
                    )
                    # Convert conductance to current (simplified for preview)
                    # Use reversal potential based on type
                    if stype_2 == 6:  # GABA-A
                        e_rev_2 = -70.0  # mV
                    elif stype_2 == 7:  # GABA-B
                        e_rev_2 = -95.0  # mV
                    else:  # Excitatory
                        e_rev_2 = 0.0  # mV
                    # Assume V ~ -65 mV for preview
                    stim[i] += attenuation_2 * abs(float(base_2)) * (e_rev_2 - (-65.0))
                # If synaptic type but no event times, produce no stimulation (skip)
            else:
                # Non-synaptic types use regular stimulus current
                base_2 = get_stim_current(
                    float(ti),
                    stype_2,
                    float(getattr(dual_cfg, "secondary_Iext", 0.0)),
                    float(getattr(dual_cfg, "secondary_start", 0.0)),
                    float(getattr(dual_cfg, "secondary_duration", 0.0)),
                    float(getattr(dual_cfg, "secondary_alpha_tau", 2.0)),
                    float(getattr(dual_cfg, "secondary_zap_f0_hz", getattr(cfg.stim, "zap_f0_hz", 0.5))),
                    float(getattr(dual_cfg, "secondary_zap_f1_hz", getattr(cfg.stim, "zap_f1_hz", 40.0))),
                )
                stim[i] += attenuation_2 * float(base_2)

    return stim


def reconstruct_stimulus_trace(result) -> np.ndarray:
    """
    Public wrapper for deterministic stimulus reconstruction from a run result.

    Intended for GUI/analytics overlays where a direct stimulus trace is needed
    without rerunning the solver.
    """
    return _reconstruct_stimulus_proxy(result)


def estimate_spike_modulation(
    spike_times_ms: np.ndarray,
    t_ms: np.ndarray,
    mod_signal: np.ndarray,
    *,
    low_hz: float = 4.0,
    high_hz: float = 12.0,
    phase_bins: int = 18,
    surrogate_count: int = 60,
) -> dict:
    """
    Estimate phase-locking between spikes and a slow modulatory rhythm.

    This is a non-FFT readout from spike timing:
    - PLV / preferred phase
    - phase-conditioned firing rates
    - deterministic surrogate p-value from resampled spike-time null
    """
    out = {
        "valid": False,
        "plv": np.nan,
        "preferred_phase_rad": np.nan,
        "preferred_phase_deg": np.nan,
        "modulation_depth": np.nan,
        "modulation_index": np.nan,
        "surrogate_p_value": np.nan,
        "surrogate_z_score": np.nan,
        "spikes_used": 0,
        "band_low_hz": float(low_hz),
        "band_high_hz": float(high_hz),
        "phase_bin_centers_rad": np.array([]),
        "phase_rate_hz": np.array([]),
    }

    n = min(len(t_ms), len(mod_signal))
    if n < 32 or len(spike_times_ms) < 3:
        return out

    t = np.asarray(t_ms[:n], dtype=float)
    x = np.asarray(mod_signal[:n], dtype=float)
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(x)):
        return out
    if np.max(t) <= np.min(t):
        return out

    # Stage 3.2 — Phase-locking protection: require at least 3 full periods
    # of the lowest cutoff frequency. Without this, Butterworth filter edge
    # artifacts produce meaningless PLV estimates.
    t_sim_ms = float(t[-1] - t[0])
    min_periods = 3.0
    min_duration_ms = min_periods * (1000.0 / max(float(low_hz), 1e-6))
    if t_sim_ms < min_duration_ms:
        return out

    dt_ms = float(np.median(np.diff(t))) if len(t) > 1 else 0.1
    dt_ms = max(dt_ms, 1e-9)
    fs_hz = 1000.0 / dt_ms
    nyq_hz = 0.5 * fs_hz

    low = max(float(low_hz), 0.05)
    high = min(float(high_hz), nyq_hz * 0.95)
    if high <= low:
        return out

    x_demean = x - float(np.mean(x))
    if float(np.std(x_demean)) < 1e-12:
        return out

    try:
        sos = butter(2, [low / nyq_hz, high / nyq_hz], btype="band", output="sos")
        x_band = sosfiltfilt(sos, x_demean)
        phase = np.angle(hilbert(x_band))
    except Exception:
        return out

    st = np.asarray(spike_times_ms, dtype=float)
    st = st[np.isfinite(st)]
    st = st[(st >= float(t[0])) & (st <= float(t[-1]))]
    if len(st) < 3:
        return out

    unit = np.exp(1j * phase)
    re_sp = np.interp(st, t, np.real(unit))
    im_sp = np.interp(st, t, np.imag(unit))
    phase_sp = np.angle(re_sp + 1j * im_sp)
    if len(phase_sp) < 3:
        return out

    vec = np.mean(np.exp(1j * phase_sp))
    plv = float(np.abs(vec))
    pref = float(np.angle(vec))

    bins = max(8, int(phase_bins))
    edges = np.linspace(-np.pi, np.pi, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    occ_counts, _ = np.histogram(phase, bins=edges)
    spike_counts, _ = np.histogram(phase_sp, bins=edges)
    occ_time_s = occ_counts.astype(float) * (dt_ms / 1000.0)
    rate_hz = np.divide(
        spike_counts.astype(float),
        occ_time_s,
        out=np.zeros_like(occ_time_s, dtype=float),
        where=occ_time_s > 0,
    )
    mean_rate = float(np.mean(rate_hz)) if len(rate_hz) > 0 else 0.0
    mod_depth = (
        float((np.max(rate_hz) - np.min(rate_hz)) / mean_rate) if mean_rate > 0.0 else np.nan
    )

    total_spikes = int(np.sum(spike_counts))
    if total_spikes > 0:
        p = spike_counts.astype(float) / float(total_spikes)
        p = np.maximum(p, 1e-12)
        uniform = 1.0 / float(len(p))
        kl = float(np.sum(p * np.log(p / uniform)))
        mod_index = float(kl / np.log(len(p))) if len(p) > 1 else np.nan
    else:
        mod_index = np.nan

    p_value = np.nan
    z_score = np.nan
    if surrogate_count > 0:
        s_count = int(surrogate_count)
        if s_count > 0:
            from core.stochastic_rng import get_rng
            rng = get_rng()
            plv_surr = np.empty(s_count, dtype=float)
            for i in range(s_count):
                st_surr = np.sort(rng.choice(t, size=len(st), replace=True))
                re_s = np.interp(st_surr, t, np.real(unit))
                im_s = np.interp(st_surr, t, np.imag(unit))
                phase_s = np.angle(re_s + 1j * im_s)
                plv_surr[i] = float(np.abs(np.mean(np.exp(1j * phase_s))))
            p_value = float((1.0 + np.sum(plv_surr >= plv)) / (len(plv_surr) + 1.0))
            s_std = float(np.std(plv_surr))
            if s_std > 1e-12:
                z_score = float((plv - float(np.mean(plv_surr))) / s_std)

    out.update(
        {
            "valid": True,
            "plv": plv,
            "preferred_phase_rad": pref,
            "preferred_phase_deg": float(np.degrees(pref)),
            "modulation_depth": mod_depth,
            "modulation_index": mod_index,
            "surrogate_p_value": p_value,
            "surrogate_z_score": z_score,
            "spikes_used": int(len(phase_sp)),
            "band_low_hz": low,
            "band_high_hz": high,
            "phase_bin_centers_rad": centers,
            "phase_rate_hz": rate_hz,
        }
    )
    return out


def compute_membrane_impedance(
    t_ms: np.ndarray,
    v_mV: np.ndarray,
    i_stim_uA_cm2: np.ndarray,
    fmin_hz: float = 0.5,
    fmax_hz: float = 200.0,
) -> dict:
    """Estimate membrane impedance Z(f)=V(f)/I(f) from time-domain traces.

    Returns magnitude in kOhm*cm^2 because input units are mV and uA/cm^2.
    """
    t_ms = np.asarray(t_ms, dtype=float)
    v_mV = np.asarray(v_mV, dtype=float)
    i_stim_uA_cm2 = np.asarray(i_stim_uA_cm2, dtype=float)

    out = {
        "valid": False,
        "freq_hz": np.array([]),
        "z_mag_kohm_cm2": np.array([]),
        "z_phase_deg": np.array([]),
        "f_res_hz": np.nan,
        "z_res_kohm_cm2": np.nan,
    }

    if len(t_ms) < 16 or len(v_mV) != len(t_ms) or len(i_stim_uA_cm2) != len(t_ms):
        return out

    dt_ms = float(np.mean(np.diff(t_ms)))
    if not np.isfinite(dt_ms) or dt_ms <= 0.0:
        return out

    fs_hz = 1000.0 / dt_ms
    n = len(t_ms)

    v = v_mV - float(np.mean(v_mV))
    i = i_stim_uA_cm2 - float(np.mean(i_stim_uA_cm2))

    vf = np.fft.rfft(v)
    inf = np.fft.rfft(i)
    freq = np.fft.rfftfreq(n, d=dt_ms / 1000.0)

    band = (freq >= float(fmin_hz)) & (freq <= float(fmax_hz))
    if not np.any(band):
        return out

    eps = 1e-12
    i_abs = np.abs(inf)
    drive_mask = i_abs > (0.01 * np.max(i_abs) + eps)
    mask = band & drive_mask
    if not np.any(mask):
        mask = band

    z = np.zeros_like(vf, dtype=np.complex128)
    z[mask] = vf[mask] / (inf[mask] + eps)

    freq_b = freq[mask]
    z_mag = np.abs(z[mask])
    z_phase = np.angle(z[mask], deg=True)

    if len(freq_b) == 0:
        return out

    k = int(np.argmax(z_mag))
    out.update({
        "valid": True,
        "freq_hz": freq_b,
        "z_mag_kohm_cm2": z_mag,
        "z_phase_deg": z_phase,
        "f_res_hz": float(freq_b[k]),
        "z_res_kohm_cm2": float(z_mag[k]),
    })
    return out


def full_analysis(result, compute_lyapunov: bool | None = None) -> dict:
    """
    Compute the complete Neuron Passport from a SimulationResult.

    Returns a flat dict with all key biophysical metrics.
    """
    V   = result.v_soma
    t   = result.t
    cfg = result.config

    ana = cfg.analysis
    spike_detect_algorithm = getattr(ana, "spike_detect_algorithm", "peak_repolarization")
    spike_detect_threshold = float(getattr(ana, "spike_detect_threshold", -20.0))
    spike_detect_prominence = float(getattr(ana, "spike_detect_prominence", 10.0))
    spike_detect_baseline_threshold = float(
        getattr(ana, "spike_detect_baseline_threshold", -50.0)
    )
    spike_detect_repolarization_window_ms = float(
        getattr(ana, "spike_detect_repolarization_window_ms", 20.0)
    )
    spike_detect_refractory_ms = float(getattr(ana, "spike_detect_refractory_ms", 1.0))

    peak_idx, spike_times, spike_amps = detect_spikes(
        V,
        t,
        threshold=spike_detect_threshold,
        prominence=spike_detect_prominence,
        baseline_threshold=spike_detect_baseline_threshold,
        repolarization_window_ms=spike_detect_repolarization_window_ms,
        refractory_ms=spike_detect_refractory_ms,
        algorithm=spike_detect_algorithm,
    )
    n_spikes = len(spike_times)

    V_thr = spike_threshold(V, t) if n_spikes > 0 else np.nan
    hw    = spike_halfwidth(
        V,
        t,
        detect_threshold=spike_detect_threshold,
        detect_prominence=spike_detect_prominence,
        detect_baseline_threshold=spike_detect_baseline_threshold,
        detect_repolarization_window_ms=spike_detect_repolarization_window_ms,
        detect_refractory_ms=spike_detect_refractory_ms,
        detect_algorithm=spike_detect_algorithm,
    ) if n_spikes > 0 else np.nan
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
        if mc.N_trunk > 0:
            junction = min(1 + mc.N_ais + mc.N_trunk - 1, result.n_comp - 1)
        elif mc.N_ais > 0:
            junction = min(mc.N_ais, result.n_comp - 1)
        else:
            junction = min(1, result.n_comp - 1)
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

    neuron_type_rule = neuron_type
    neuron_type_ml = "Intermediate"
    neuron_type_ml_conf = 0.0
    neuron_type_hybrid = neuron_type_rule
    neuron_type_source = "rule_only"
    if n_spikes > 1:
        neuron_type_ml, neuron_type_ml_conf = classify_neuron_ml(
            f_initial,
            f_steady,
            AI,
            hw if not np.isnan(hw) else np.nan,
            cv_isi,
        )
        neuron_type_hybrid, neuron_type_source, hybrid_conf = classify_neuron_hybrid(
            neuron_type_rule,
            f_initial,
            f_steady,
            AI,
            hw if not np.isnan(hw) else np.nan,
            cv_isi,
        )
    else:
        hybrid_conf = 1.0 if neuron_type_rule == "Silent" else 0.5

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

    # Optional FTLE/LLE stability analysis (default OFF).
    lyap = {
        "lle_per_ms": np.nan,
        "lle_per_s": np.nan,
        "lyapunov_class": "disabled",
        "lyapunov_valid_pairs": 0,
        "ftle_time_ms": np.array([]),
        "ftle_log_divergence": np.array([]),
    }
    should_compute_lyapunov = bool(compute_lyapunov) if compute_lyapunov is not None else False
    if should_compute_lyapunov:
        lyap_raw = estimate_ftle_lle(
            V,
            t,
            embedding_dim=getattr(ana, "lyapunov_embedding_dim", 3),
            lag_steps=getattr(ana, "lyapunov_lag_steps", 2),
            min_separation_ms=getattr(ana, "lyapunov_min_separation_ms", 10.0),
            fit_start_ms=getattr(ana, "lyapunov_fit_start_ms", 5.0),
            fit_end_ms=getattr(ana, "lyapunov_fit_end_ms", 40.0),
        )
        lyap = {
            "lle_per_ms": float(lyap_raw["lle_per_ms"]) if np.isfinite(lyap_raw["lle_per_ms"]) else np.nan,
            "lle_per_s": float(lyap_raw["lle_per_s"]) if np.isfinite(lyap_raw["lle_per_s"]) else np.nan,
            "lyapunov_class": classify_lyapunov(lyap_raw["lle_per_ms"]),
            "lyapunov_valid_pairs": int(lyap_raw["valid_pairs"]),
            "ftle_time_ms": lyap_raw.get("ftle_time_ms", np.array([])),
            "ftle_log_divergence": lyap_raw.get("ftle_log_divergence", np.array([])),
        }

    # Optional non-FFT modulation decomposition (default OFF).
    modulation = {
        "valid": False,
        "source": getattr(ana, "modulation_source", "voltage"),
        "plv": np.nan,
        "preferred_phase_rad": np.nan,
        "preferred_phase_deg": np.nan,
        "modulation_depth": np.nan,
        "modulation_index": np.nan,
        "surrogate_p_value": np.nan,
        "surrogate_z_score": np.nan,
        "spikes_used": 0,
        "band_low_hz": np.nan,
        "band_high_hz": np.nan,
        "phase_bin_centers_rad": np.array([]),
        "phase_rate_hz": np.array([]),
    }
    if getattr(ana, "enable_modulation_decomposition", False):
        source = getattr(ana, "modulation_source", "voltage")
        mod_signal = V if source == "voltage" else _reconstruct_stimulus_proxy(result)
        mod_raw = estimate_spike_modulation(
            spike_times,
            t,
            mod_signal,
            low_hz=getattr(ana, "modulation_low_hz", 4.0),
            high_hz=getattr(ana, "modulation_high_hz", 12.0),
            phase_bins=getattr(ana, "modulation_phase_bins", 18),
            surrogate_count=getattr(ana, "modulation_surrogates", 60),
        )
        modulation.update(mod_raw)
        modulation["source"] = source

    return {
        'n_spikes':            n_spikes,
        'spike_times':         spike_times,
        'spike_amps':          spike_amps,
        'spike_detect_algorithm': spike_detect_algorithm,
        'spike_detect_threshold': spike_detect_threshold,
        'spike_detect_prominence': spike_detect_prominence,
        'spike_detect_baseline_threshold': spike_detect_baseline_threshold,
        'spike_detect_repolarization_window_ms': spike_detect_repolarization_window_ms,
        'spike_detect_refractory_ms': spike_detect_refractory_ms,
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
        'neuron_type_rule':    neuron_type_rule,
        'neuron_type_ml':      neuron_type_ml,
        'neuron_type_ml_confidence': neuron_type_ml_conf,
        'neuron_type_hybrid':  neuron_type_hybrid,
        'neuron_type_hybrid_source': neuron_type_source,
        'neuron_type_hybrid_confidence': hybrid_conf,
        'conduction_vel_ms':   cv,
        'tau_m_ms':            tau_m,
        'Rin_kohm_cm2':        Rin,
        'lambda_um':           lam_um,
        'Q_per_channel':       Q_per_channel,
        'atp_nmol_cm2':        result.atp_estimate,
        'atp_breakdown':       result.atp_breakdown,
        # ── NEW: Advanced firing analysis ──
        'isi_mean_ms':         isi_mean,
        'isi_std_ms':          isi_std,
        'isi_min_ms':          isi_min,
        'isi_max_ms':          isi_max,
        'cv_isi':              cv_isi,
        'first_spike_latency_ms': first_spike_lat,
        'refractory_period_ms':   refr_period,
        'firing_reliability':  firing_reliability,
        'lle_per_ms':           lyap["lle_per_ms"],
        'lle_per_s':            lyap["lle_per_s"],
        'lyapunov_class':       lyap["lyapunov_class"],
        'lyapunov_valid_pairs': lyap["lyapunov_valid_pairs"],
        'ftle_time_ms':         lyap["ftle_time_ms"],
        'ftle_log_divergence':  lyap["ftle_log_divergence"],
        'modulation_valid':    modulation["valid"],
        'modulation_source':   modulation["source"],
        'modulation_plv':      modulation["plv"],
        'modulation_preferred_phase_rad': modulation["preferred_phase_rad"],
        'modulation_preferred_phase_deg': modulation["preferred_phase_deg"],
        'modulation_depth':    modulation["modulation_depth"],
        'modulation_index':    modulation["modulation_index"],
        'modulation_p_value':  modulation["surrogate_p_value"],
        'modulation_z_score':  modulation["surrogate_z_score"],
        'modulation_spikes_used': modulation["spikes_used"],
        'modulation_band_low_hz': modulation["band_low_hz"],
        'modulation_band_high_hz': modulation["band_high_hz"],
        'modulation_phase_bin_centers_rad': modulation["phase_bin_centers_rad"],
        'modulation_phase_rate_hz': modulation["phase_rate_hz"],
    }
