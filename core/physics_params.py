"""Physics parameter container for RHS unification.

This module provides a Numba-compatible structured container for all
static physics parameters, eliminating the "argument explosion" problem
and reducing silent-failure risk during signature updates.
"""

from numba import njit
from numba.types import float64, int32, boolean
import numpy as np
from typing import NamedTuple, Optional


ENV_T_KELVIN = 0
ENV_CA_EXT = 1
ENV_CA_REST = 2
ENV_TAU_CA = 3
ENV_MG_EXT = 4
ENV_TAU_SK = 5
ENV_IM_SPEED = 6
ENV_G_KATP_MAX = 7
ENV_KATP_KD = 8
ENV_ATP_MAX = 9
ENV_ATP_SYNTH = 10
ENV_NA_I_REST = 11
ENV_NA_EXT = 12
ENV_K_I = 13
ENV_K_O_REST = 14
ENV_ION_DRIFT_GAIN = 15
ENV_K_O_CLEARANCE_TAU = 16
ENV_PARAM_COUNT = 17


class StateOffsets(NamedTuple):
    """Shared state-vector layout for solver, native loop, Jacobian, and analysis."""
    off_v: int32
    off_m: int32
    off_h: int32
    off_n: int32
    off_r: int32
    off_s: int32
    off_u: int32
    off_a: int32
    off_b: int32
    off_p: int32
    off_q: int32
    off_w: int32
    off_x: int32
    off_y: int32
    off_j: int32
    off_zsk: int32
    off_ca: int32
    off_atp: int32
    off_na_i: int32
    off_k_o: int32
    off_ifilt_primary: int32
    off_ifilt_secondary: int32
    n_state: int32


def build_env_params(
    t_kelvin: float,
    ca_ext: float,
    ca_rest: float,
    tau_ca: float,
    mg_ext: float,
    tau_sk: float,
    im_speed_multiplier: float,
    g_katp_max: float,
    katp_kd_atp_mM: float,
    atp_max_mM: float,
    atp_synthesis_rate: float,
    na_i_rest_mM: float,
    na_ext_mM: float,
    k_i_mM: float,
    k_o_rest_mM: float,
    ion_drift_gain: float,
    k_o_clearance_tau_ms: float,
) -> np.ndarray:
    env = np.zeros(ENV_PARAM_COUNT, dtype=np.float64)
    env[ENV_T_KELVIN] = t_kelvin
    env[ENV_CA_EXT] = ca_ext
    env[ENV_CA_REST] = ca_rest
    env[ENV_TAU_CA] = tau_ca
    env[ENV_MG_EXT] = mg_ext
    env[ENV_TAU_SK] = tau_sk
    env[ENV_IM_SPEED] = im_speed_multiplier
    env[ENV_G_KATP_MAX] = g_katp_max
    env[ENV_KATP_KD] = katp_kd_atp_mM
    env[ENV_ATP_MAX] = atp_max_mM
    env[ENV_ATP_SYNTH] = atp_synthesis_rate
    env[ENV_NA_I_REST] = na_i_rest_mM
    env[ENV_NA_EXT] = na_ext_mM
    env[ENV_K_I] = k_i_mM
    env[ENV_K_O_REST] = k_o_rest_mM
    env[ENV_ION_DRIFT_GAIN] = ion_drift_gain
    env[ENV_K_O_CLEARANCE_TAU] = k_o_clearance_tau_ms
    return env


def build_state_offsets(
    n_comp: int,
    *,
    en_ih: bool,
    en_ica: bool,
    en_ia: bool,
    en_sk: bool,
    dyn_ca: bool,
    en_itca: bool,
    en_im: bool,
    en_nap: bool,
    en_nar: bool,
    dyn_atp: bool,
    use_dfilter_primary: int,
    use_dfilter_secondary: int,
) -> StateOffsets:
    cursor = 0
    off_v = cursor
    cursor += n_comp
    off_m = cursor
    cursor += n_comp
    off_h = cursor
    cursor += n_comp
    off_n = cursor
    cursor += n_comp

    off_r = cursor if en_ih else -1
    if en_ih:
        cursor += n_comp

    off_s = cursor if en_ica else -1
    if en_ica:
        cursor += n_comp
    off_u = cursor if en_ica else -1
    if en_ica:
        cursor += n_comp

    off_a = cursor if en_ia else -1
    if en_ia:
        cursor += n_comp
    off_b = cursor if en_ia else -1
    if en_ia:
        cursor += n_comp

    off_p = cursor if en_itca else -1
    if en_itca:
        cursor += n_comp
    off_q = cursor if en_itca else -1
    if en_itca:
        cursor += n_comp

    off_w = cursor if en_im else -1
    if en_im:
        cursor += n_comp

    off_x = cursor if en_nap else -1
    if en_nap:
        cursor += n_comp

    off_y = cursor if en_nar else -1
    if en_nar:
        cursor += n_comp
    off_j = cursor if en_nar else -1
    if en_nar:
        cursor += n_comp

    off_zsk = cursor if en_sk else -1
    if en_sk:
        cursor += n_comp

    off_ca = cursor if dyn_ca else -1
    if dyn_ca:
        cursor += n_comp

    off_atp = cursor if dyn_atp else -1
    if dyn_atp:
        cursor += n_comp

    off_na_i = cursor if dyn_atp else -1
    if dyn_atp:
        cursor += n_comp

    off_k_o = cursor if dyn_atp else -1
    if dyn_atp:
        cursor += n_comp

    off_ifilt_primary = cursor if use_dfilter_primary == 1 else -1
    if use_dfilter_primary == 1:
        cursor += 1

    off_ifilt_secondary = cursor if use_dfilter_secondary == 1 else -1
    if use_dfilter_secondary == 1:
        cursor += 1

    return StateOffsets(
        np.int32(off_v),
        np.int32(off_m),
        np.int32(off_h),
        np.int32(off_n),
        np.int32(off_r),
        np.int32(off_s),
        np.int32(off_u),
        np.int32(off_a),
        np.int32(off_b),
        np.int32(off_p),
        np.int32(off_q),
        np.int32(off_w),
        np.int32(off_x),
        np.int32(off_y),
        np.int32(off_j),
        np.int32(off_zsk),
        np.int32(off_ca),
        np.int32(off_atp),
        np.int32(off_na_i),
        np.int32(off_k_o),
        np.int32(off_ifilt_primary),
        np.int32(off_ifilt_secondary),
        np.int32(cursor),
    )


def state_slices_from_offsets(offsets: StateOffsets, n_comp: int) -> dict[str, slice | int | None]:
    def _slice(start: int) -> slice | None:
        return None if int(start) < 0 else slice(int(start), int(start) + n_comp)

    def _scalar(start: int) -> int | None:
        return None if int(start) < 0 else int(start)

    return {
        "v": slice(int(offsets.off_v), int(offsets.off_v) + n_comp),
        "m": slice(int(offsets.off_m), int(offsets.off_m) + n_comp),
        "h": slice(int(offsets.off_h), int(offsets.off_h) + n_comp),
        "n": slice(int(offsets.off_n), int(offsets.off_n) + n_comp),
        "r": _slice(offsets.off_r),
        "s": _slice(offsets.off_s),
        "u": _slice(offsets.off_u),
        "a": _slice(offsets.off_a),
        "b": _slice(offsets.off_b),
        "p": _slice(offsets.off_p),
        "q": _slice(offsets.off_q),
        "w": _slice(offsets.off_w),
        "x": _slice(offsets.off_x),
        "y_nr": _slice(offsets.off_y),
        "j_nr": _slice(offsets.off_j),
        "z_sk": _slice(offsets.off_zsk),
        "ca": _slice(offsets.off_ca),
        "atp": _slice(offsets.off_atp),
        "na_i": _slice(offsets.off_na_i),
        "k_o": _slice(offsets.off_k_o),
        "dfilter_primary": _scalar(offsets.off_ifilt_primary),
        "dfilter_secondary": _scalar(offsets.off_ifilt_secondary),
    }


class PhysicsParams(NamedTuple):
    """Structured container for all static physics parameters.
    
    This replaces 50+ positional arguments with a single, well-structured
    object that Numba can efficiently unpack without heap allocations.
    """
    # Model size and feature flags
    n_comp: int32
    en_ih: boolean
    en_ica: boolean
    en_ia: boolean
    en_sk: boolean
    dyn_ca: boolean
    en_itca: boolean
    en_im: boolean
    en_nap: boolean
    en_nar: boolean
    dyn_atp: boolean
    
    # Conductance matrix [11 x n_comp]
    # Rows: [gna, gk, gl, gih, gca, ga, gsk, gtca, gim, gnap, gnar]
    gbar_mat: np.ndarray
    
    # Reversal potentials
    ena: float64
    ek: float64
    el: float64
    eih: float64
    ea: float64
    e_rev_syn_primary: float64   # Primary stimulus synaptic reversal (for pathology)
    e_rev_syn_secondary: float64  # Secondary stimulus synaptic reversal (for pathology)
    
    # Morphology and axial coupling
    cm_v: np.ndarray
    l_data: np.ndarray
    l_indices: np.ndarray
    l_indptr: np.ndarray
    
    # Temperature scaling matrix [9 x n_comp]
    # Rows: [phi_na, phi_k, phi_ih, phi_ca, phi_ia, phi_tca, phi_im, phi_nap, phi_nar]
    phi_mat: np.ndarray
    
    # Environment and calcium dynamics
    t_kelvin: float64
    ca_ext: float64
    ca_rest: float64
    tau_ca: float64
    b_ca: np.ndarray
    mg_ext: float64
    nmda_mg_block_mM: float64
    tau_sk: float64
    im_speed_multiplier: float64
    env_params: np.ndarray

    # ATP metabolism parameters
    g_katp_max: float64
    katp_kd_atp_mM: float64
    atp_max_mM: float64
    atp_synthesis_rate: float64
    na_i_rest_mM: float64
    na_ext_mM: float64
    k_i_mM: float64
    k_o_rest_mM: float64
    ion_drift_gain: float64
    k_o_clearance_tau_ms: float64
    
    # Primary stimulation parameters
    stype: int32
    iext: float64
    t0: float64
    td: float64
    atau: float64
    zap_f0_hz: float64
    zap_f1_hz: float64
    zap_rise_ms: float64  # Tukey window rise/fall time for ZAP (0 = no window, abrupt)
    # Precomputed Tukey window lookup table (primary ZAP)
    zap_win_t: np.ndarray    # float64[:] time offsets from stimulus start (ms)
    zap_win_g: np.ndarray    # float64[:] window gain values [0, 1]
    zap_win_size: int32      # table size (0 = fall back to direct cos computation)
    event_times_arr: np.ndarray
    n_events: int32
    event_times_arr_2: np.ndarray
    n_events_2: int32
    stim_comp: int32
    stim_mode: int32
    use_dfilter_primary: int32
    # Dynamic AC attenuation parameters (v10.3) - for real-time frequency-dependent calculation
    # Distance: 0 µm = soma (no attenuation), 150-300 µm = distal synapses (high attenuation)
    # Physics: Used in AC attenuation formula |A| = exp(-x/λ · Re(√(1+jωτ)))
    # Pathology: Ischemia causes dendritic beading → increased attenuation (smaller effective λ)
    dfilter_distance_um: float64
    # Space constant λ: 150 µm typical for pyramidal L5, 50-100 µm for thin dendrites
    # Physics: λ = √(a/(4·ρ_m·g_m)), where a=radius, ρ_m=membrane resistivity
    # Pathology: Channelopathies, edema, or demyelination alter λ
    dfilter_lambda_um: float64
    # Membrane time constant τ: 10-20 ms typical for pyramidal neurons
    # Physics: τ = C_m / g_m, determines high-frequency attenuation strength
    dfilter_tau_ms: float64
    # Input frequency: 100 Hz typical for AMPA synaptic inputs, 5-50 Hz for dendritic spikes
    # Physics: Higher f → stronger AC attenuation (membrane capacitance shunts high frequencies)
    dfilter_input_freq_hz: float64
    # Filter mode: 0=DC (classic exponential, steady-state), 1=AC (frequency-dependent)
    # Physics: AC mode captures frequency-dependent attenuation; DC mode is steady-state limit
    # Usage: AC mode for fast synaptic inputs (AMPA), DC mode for slow currents (NMDA, Ca²⁺)
    dfilter_filter_mode: int32
    # Legacy: pre-computed attenuation (used as fallback when real-time calc disabled)
    dfilter_attenuation: float64
    
    # Secondary stimulation (dual)
    dual_stim_enabled: int32
    stype_2: int32
    iext_2: float64
    t0_2: float64
    td_2: float64
    atau_2: float64
    zap_f0_hz_2: float64
    zap_f1_hz_2: float64
    zap_rise_ms_2: float64
    # Precomputed Tukey window lookup table (secondary ZAP)
    zap_win_t_2: np.ndarray
    zap_win_g_2: np.ndarray
    zap_win_size_2: int32
    stim_comp_2: int32
    stim_mode_2: int32
    use_dfilter_secondary: int32
    # Dynamic AC attenuation parameters for secondary stimulus (dual)
    # Same physics as primary, allows independent control of dual stimulation
    # Usage: Model convergent inputs with different dendritic locations/frequencies
    dfilter_distance_um_2: float64
    # Space constant λ for secondary pathway - independent of primary
    dfilter_lambda_um_2: float64
    # Time constant τ for secondary - can differ if different dendrite type
    dfilter_tau_ms_2: float64
    # Input frequency for secondary - allows modeling different synaptic types
    # e.g., primary=AMPA (100Hz), secondary=NMDA (5Hz) with different attenuation
    dfilter_input_freq_hz_2: float64
    # Filter mode for secondary: 0=DC, 1=AC (independent control)
    dfilter_filter_mode_2: int32
    # Legacy pre-computed attenuation for secondary
    dfilter_attenuation_2: float64
    
    # Stochastic parameters
    stoch_gating: boolean  # Enable Langevin gate noise
    noise_sigma: float64    # Additive membrane current noise
    gna_max: float64        # Max Na conductance for channel count scaling
    gk_max: float64         # Max K conductance for channel count scaling
    rng_state: Optional[np.ndarray]  # RNG state for reproducibility
    state_offsets: StateOffsets


def create_physics_params(**kwargs) -> PhysicsParams:
    """
    Create a PhysicsParams instance from keyword arguments, supplying sensible defaults for optional fields.
    
    Parameters:
        kwargs (mapping): Keyword arguments corresponding to PhysicsParams fields. Any omitted optional fields
        (stochastic settings, ATP/metabolism fields, secondary-stimulus fields, synaptic reversal potentials,
        and dynamic AC attenuation parameters `dfilter_distance_um`, `dfilter_lambda_um`, etc.) are filled
        with library defaults.
    
    Returns:
        PhysicsParams: A fully populated PhysicsParams NamedTuple built from the provided and defaulted values.
    """
    # Set defaults for optional fields if not provided
    defaults = {
        'stoch_gating': False,
        'noise_sigma': 0.0,
        'gna_max': 120.0,
        'gk_max': 36.0,
        'rng_state': None,
        'event_times_arr_2': np.zeros(0, dtype=np.float64),
        'n_events_2': np.int32(0),
        'dyn_atp': False,
        'g_katp_max': 0.0,
        'katp_kd_atp_mM': 0.5,
        'atp_max_mM': 2.0,
        'atp_synthesis_rate': 0.6,
        'na_i_rest_mM': 12.0,
        'na_ext_mM': 145.0,
        'k_i_mM': 140.0,
        'k_o_rest_mM': 3.5,
        'ion_drift_gain': 0.0,
        'k_o_clearance_tau_ms': 800.0,
        'e_rev_syn_primary': 0.0,      # Default: 0 mV (excitatory)
        'e_rev_syn_secondary': -75.0,  # Default: -75 mV (inhibitory)
        'im_speed_multiplier': 1.0,
        # Dynamic AC attenuation defaults (v10.3) - for frequency-dependent dendritic filtering
        # Distance: 0 µm = soma (no attenuation), 150-300 µm = distal synapses (high attenuation)
        # Used in: AC attenuation formula |A| = exp(-x/λ · Re(√(1+jωτ)))
        'dfilter_distance_um': 0.0,
        # Space constant λ: 150 µm typical for pyramidal L5, 50-100 µm for thin dendrites
        # Pathology: Shrinkage in ischemia increases attenuation (smaller λ)
        'dfilter_lambda_um': 150.0,
        # Input frequency: 100 Hz typical for synaptic inputs, 5-50 Hz for dendritic spikes
        # Higher frequencies → stronger attenuation due to membrane capacitance
        'dfilter_input_freq_hz': 100.0,
        # Filter mode: 0=DC (classic exponential, steady-state), 1=AC (frequency-dependent)
        # AC mode: More accurate for fast synaptic inputs (AMPA), preserves frequency content
        # DC mode: Valid for slow currents (NMDA, calcium), computationally simpler
        'dfilter_filter_mode': 0,
        # ZAP stimulus Tukey window defaults - smooth ramp reduces spectral leakage
        # Rise time: 5ms (5% of 100ms pulse), 0 = abrupt (no window)
        # Physics: Cosine-tapered window eliminates discontinuities at stimulus edges
        'zap_rise_ms': 5.0,
        'zap_win_t': np.zeros(0, dtype=np.float64),
        'zap_win_g': np.zeros(0, dtype=np.float64),
        'zap_win_size': np.int32(0),
        'zap_rise_ms_2': 5.0,
        'zap_win_t_2': np.zeros(0, dtype=np.float64),
        'zap_win_g_2': np.zeros(0, dtype=np.float64),
        'zap_win_size_2': np.int32(0),
        # Secondary (dual) stimulus - same parameters for independent control
        'dfilter_distance_um_2': 0.0,
        'dfilter_lambda_um_2': 150.0,
        'dfilter_input_freq_hz_2': 100.0,
        'dfilter_filter_mode_2': 0,
    }
    for k, v in defaults.items():
        if k not in kwargs:
            kwargs[k] = v

    if 'env_params' not in kwargs:
        kwargs['env_params'] = build_env_params(
            kwargs['t_kelvin'],
            kwargs['ca_ext'],
            kwargs['ca_rest'],
            kwargs['tau_ca'],
            kwargs['mg_ext'],
            kwargs['tau_sk'],
            kwargs['im_speed_multiplier'],
            kwargs['g_katp_max'],
            kwargs['katp_kd_atp_mM'],
            kwargs['atp_max_mM'],
            kwargs['atp_synthesis_rate'],
            kwargs['na_i_rest_mM'],
            kwargs['na_ext_mM'],
            kwargs['k_i_mM'],
            kwargs['k_o_rest_mM'],
            kwargs['ion_drift_gain'],
            kwargs['k_o_clearance_tau_ms'],
        )

    if 'state_offsets' not in kwargs:
        kwargs['state_offsets'] = build_state_offsets(
            int(kwargs['n_comp']),
            en_ih=bool(kwargs['en_ih']),
            en_ica=bool(kwargs['en_ica']),
            en_ia=bool(kwargs['en_ia']),
            en_sk=bool(kwargs['en_sk']),
            dyn_ca=bool(kwargs['dyn_ca']),
            en_itca=bool(kwargs['en_itca']),
            en_im=bool(kwargs['en_im']),
            en_nap=bool(kwargs['en_nap']),
            en_nar=bool(kwargs['en_nar']),
            dyn_atp=bool(kwargs['dyn_atp']),
            use_dfilter_primary=int(kwargs['use_dfilter_primary']),
            use_dfilter_secondary=int(kwargs['use_dfilter_secondary']),
        )

    return PhysicsParams(**kwargs)


@njit
def unpack_env_params(env_params):
    """Unpack packed environment/metabolism scalars from env_params."""
    return (
        env_params[ENV_T_KELVIN],
        env_params[ENV_CA_EXT],
        env_params[ENV_CA_REST],
        env_params[ENV_TAU_CA],
        env_params[ENV_MG_EXT],
        env_params[ENV_TAU_SK],
        env_params[ENV_IM_SPEED],
        env_params[ENV_G_KATP_MAX],
        env_params[ENV_KATP_KD],
        env_params[ENV_ATP_MAX],
        env_params[ENV_ATP_SYNTH],
        env_params[ENV_NA_I_REST],
        env_params[ENV_NA_EXT],
        env_params[ENV_K_I],
        env_params[ENV_K_O_REST],
        env_params[ENV_ION_DRIFT_GAIN],
        env_params[ENV_K_O_CLEARANCE_TAU],
    )


@njit
def unpack_conductances(gbar_mat, n_comp):
    """Unpack conductance matrix into individual vectors.
    
    This function extracts individual conductance vectors from the
    packed matrix, avoiding repeated indexing in the main loop.
    """
    gna_v = gbar_mat[0]
    gk_v = gbar_mat[1]
    gl_v = gbar_mat[2]
    gih_v = gbar_mat[3]
    gca_v = gbar_mat[4]
    ga_v = gbar_mat[5]
    gsk_v = gbar_mat[6]
    gtca_v = gbar_mat[7]
    gim_v = gbar_mat[8]
    gnap_v = gbar_mat[9]
    gnar_v = gbar_mat[10]
    
    return (gna_v, gk_v, gl_v, gih_v, gca_v, ga_v, gsk_v, 
            gtca_v, gim_v, gnap_v, gnar_v)


@njit
def unpack_temperature_scaling(phi_mat, n_comp):
    """Unpack temperature scaling matrix into individual vectors."""
    phi_na = phi_mat[0]
    phi_k = phi_mat[1]
    phi_ih = phi_mat[2]
    phi_ca = phi_mat[3]
    phi_ia = phi_mat[4]
    phi_tca = phi_mat[5]
    phi_im = phi_mat[6]
    phi_nap = phi_mat[7]
    phi_nar = phi_mat[8]
    
    return (phi_na, phi_k, phi_ih, phi_ca, phi_ia, phi_tca, 
            phi_im, phi_nap, phi_nar)
