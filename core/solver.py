import numpy as np
import copy
import logging
import hashlib
import os
from scipy.integrate import solve_ivp, trapezoid
from concurrent.futures import ThreadPoolExecutor

from core.models import FullModelConfig
from core.morphology import MorphologyBuilder
from core.channels import ChannelRegistry, derive_dynamic_atp_rest_ions
from core.jacobian import analytic_sparse_jacobian, build_jacobian_sparsity, make_analytic_jacobian
from core.rhs import (
    rhs_multicompartment, _get_syn_reversal, F_CONST, R_GAS,
    ATP_ISCHEMIC_THRESHOLD, NA_I_MIN_M_M, NA_I_MAX_M_M, K_O_MIN_M_M, K_O_MAX_M_M,
    compute_na_k_pump_current, get_pump_current_array,
)
from core.dendritic_filter import get_ac_attenuation
from core.physics_params import create_physics_params, build_state_offsets
from core.kinetics import z_inf_SK
from core.validation import estimate_simulation_runtime, validate_simulation_config


logger = logging.getLogger(__name__)

# LLE subspace mode mapping for native solver (Numba-compatible ints)
# 0=v_only, 1=v_and_gates, 2=full_state, 3=custom
_LLE_MODE_MAP = {"v_only": 0, "v_and_gates": 1, "full_state": 2, "custom": 3}


def _precompute_zap_window(td_ms: float, rise_ms: float, res_ms: float = 0.1):
    """Precompute a Tukey (cosine-tapered) window lookup table for ZAP stimulus.

    Returns (win_t, win_g, win_size) — arrays of time-offsets and gain values.
    When rise_ms <= 0 or td_ms <= 0, returns empty arrays (rectangular window;
    get_stim_current falls back to direct computation).
    """
    if td_ms <= 0.0 or rise_ms <= 0.0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64), 0
    n_pts = max(int(td_ms / res_ms) + 1, 3)
    win_t = np.linspace(0.0, td_ms, n_pts)
    win_g = np.ones(n_pts, dtype=np.float64)
    for i in range(n_pts):
        dt = win_t[i]
        if dt < rise_ms:
            win_g[i] = 0.5 * (1.0 - np.cos(np.pi * dt / rise_ms))
        elif dt > (td_ms - rise_ms):
            fall_dt = td_ms - dt
            win_g[i] = 0.5 * (1.0 - np.cos(np.pi * fall_dt / rise_ms))
    return win_t, win_g, n_pts


def _stable_seed_from_values(*values) -> int:
    """Deterministic 32-bit seed from value tuple (stable across Python sessions)."""
    payload = "|".join(str(v) for v in values).encode("utf-8")
    digest = hashlib.blake2s(payload, digest_size=4).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _resolve_stochastic_seed(cfg, noise_sigma: float, stoch_gating: bool) -> int:
    """Prefer explicit global RNG seeds; fall back to deterministic config hashing."""
    try:
        from core.stochastic_rng import get_rng
        seeded_rng = get_rng()
        if getattr(seeded_rng, "seed", None) is not None:
            return int(seeded_rng.seed)
    except Exception:
        pass
    return _stable_seed_from_values(
        cfg.stim.t_sim,
        cfg.stim.Iext,
        cfg.stim.pulse_start,
        cfg.stim.pulse_dur,
        cfg.stim.stim_type,
        noise_sigma,
        stoch_gating,
    )


def _resolve_dynamic_atp_rest_values(cfg) -> tuple[float, float]:
    """Use preset-consistent Na_i/K_o rest values whenever dynamic ATP is enabled."""
    if bool(getattr(cfg.metabolism, "enable_dynamic_atp", False)):
        return derive_dynamic_atp_rest_ions(cfg)
    return float(cfg.metabolism.na_i_rest_mM), float(cfg.metabolism.k_o_rest_mM)


def _set_nested_attr(obj, path: str, val: float):
    """Set nested attribute by dot notation (e.g. 'channels.gNa_max')."""
    path = str(path)
    if "." not in path:
        for section_name in (
            "stim", "channels", "env", "morphology",
            "calcium", "analysis", "metabolism", "preset_modes",
        ):
            section = getattr(obj, section_name, None)
            if section is not None and hasattr(section, path):
                setattr(section, path, float(val))
                return
    parts = path.split(".")
    target = obj
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], float(val))


class SimulationResult:
    """Complete scientific dataset after a simulation run."""

    def __init__(self, t, y, n_comp, config: FullModelConfig):
        """
        Initialize a SimulationResult container and extract commonly used state projections.
        
        Parameters:
            t (ndarray): Time vector for the simulation.
            y (ndarray): State matrix with shape (n_states, n_timepoints).
            n_comp (int): Number of compartments in the morphology.
            config (FullModelConfig): Simulation configuration that determines state layout and feature flags.
        
        Notes:
            - Sets attributes: t, y, config, n_comp, and diverged (initialized False).
            - Extracts membrane voltages:
                - v_all: first `n_comp` rows of `y`.
                - v_soma: soma voltage (row 0 of v_all).
            - Detects presence of dendritic-filter state(s) from `config.stim_location`, `config.dendritic_filter`,
              and `config.dual_stimulation` and slices `y` to populate:
                - v_dendritic_filtered (primary) and v_dendritic_filtered_secondary (secondary) when present.
            - Extracts intracellular calcium `ca_i` when `config.calcium.dynamic_Ca`:
                - If filter state(s) exist, reads the `n_comp` rows immediately preceding the filter states.
                - Otherwise reads the final `n_comp` rows of `y`.
            - Initializes:
                - currents (dict) for reconstructed ion currents,
                - atp_estimate (float) and atp_breakdown (dict) for metabolic bookkeeping,
                - morph (dict) reserved for morphology-derived parameters used in post-processing.
        """
        self.t      = t
        self.y      = y
        self.config = config
        self.n_comp = n_comp
        self.diverged = False  # v12.0: Flag for simulation divergence due to non-physical parameters

        self.v_all  = y[0:n_comp, :]
        self.v_soma = self.v_all[0, :]
        primary_loc = config.stim_location.location
        use_dfilter_primary = (
            primary_loc == "dendritic_filtered"
            and config.dendritic_filter.enabled
            and config.dendritic_filter.tau_dendritic_ms > 0.0
        )
        dual = getattr(config, "dual_stimulation", None)
        dual_enabled = bool(dual is not None and getattr(dual, "enabled", False))
        use_dfilter_secondary = (
            dual_enabled
            and getattr(dual, "secondary_location", "soma") == "dendritic_filtered"
            and getattr(dual, "secondary_tau_dendritic_ms", 0.0) > 0.0
        )
        ch = config.channels
        offsets = build_state_offsets(
            n_comp,
            en_ih=ch.enable_Ih,
            en_ica=ch.enable_ICa,
            en_ia=ch.enable_IA,
            en_sk=ch.enable_SK,
            dyn_ca=config.calcium.dynamic_Ca,
            en_itca=ch.enable_ITCa,
            en_im=ch.enable_IM,
            en_nap=ch.enable_NaP,
            en_nar=ch.enable_NaR,
            dyn_atp=config.metabolism.enable_dynamic_atp,
            use_dfilter_primary=1 if use_dfilter_primary else 0,
            use_dfilter_secondary=1 if use_dfilter_secondary else 0,
        )

        self.ca_i = None
        if int(offsets.off_ca) >= 0:
            self.ca_i = y[int(offsets.off_ca):int(offsets.off_ca) + n_comp, :]

        self.atp_level = None
        if int(offsets.off_atp) >= 0:
            self.atp_level = y[int(offsets.off_atp):int(offsets.off_atp) + n_comp, :]
        self.atp = self.atp_level

        self.na_i = None
        if int(offsets.off_na_i) >= 0:
            self.na_i = y[int(offsets.off_na_i):int(offsets.off_na_i) + n_comp, :]

        self.k_o = None
        if int(offsets.off_k_o) >= 0:
            self.k_o = y[int(offsets.off_k_o):int(offsets.off_k_o) + n_comp, :]

        self.v_dendritic_filtered = None
        self.v_dendritic_filtered_secondary = None
        if int(offsets.off_ifilt_primary) >= 0:
            self.v_dendritic_filtered = y[int(offsets.off_ifilt_primary), :]
        if int(offsets.off_ifilt_secondary) >= 0:
            self.v_dendritic_filtered_secondary = y[int(offsets.off_ifilt_secondary), :]

        self.currents:    dict  = {}
        self.atp_estimate: float = 0.0
        self.atp_breakdown: dict = {}  # {Na_pump, Ca_pump, baseline, total}

        # Morphology dict stored for post-analysis (current balance, etc.)
        self.morph: dict = {}

        # LLE convergence array (populated when calc_lle=True in run_native)
        self.lle_convergence: np.ndarray | None = None

    def _finalize_current_shapes(self):
        """Collapse single-compartment current matrices to 1-D time series."""
        for key, arr in list(self.currents.items()):
            arr_np = np.asarray(arr)
            if arr_np.ndim == 2 and arr_np.shape[0] == 1:
                self.currents[key] = arr_np[0, :]


def generate_effective_event_times(train_type: str, freq_hz: float, duration_ms: float, t_start: float, manual_times: list, seed_hash: int = None) -> np.ndarray:
    """
    Create an array of synaptic event times (in milliseconds) for a stimulus train without modifying config.
    
    Parameters:
        train_type (str): One of 'none', 'regular', or 'poisson'. Unknown values are treated as 'none'.
        freq_hz (float): Event frequency in hertz used for generated trains; ignored when `train_type` is 'none' or manual times are used.
        duration_ms (float): Duration of the generated train in milliseconds; ignored when `train_type` is 'none' or manual times are used.
        t_start (float): Start time of the train in milliseconds.
        manual_times (list): Explicit event times to return when `train_type` is 'none'.
        seed_hash (int, optional): Deterministic seed for Poisson interval sampling; when provided, identical inputs produce identical Poisson trains.
    
    Returns:
        event_times (np.ndarray): 1-D array of event times in milliseconds (dtype float64). For 'none' this contains the provided manual times (or is empty); for 'regular' contains evenly spaced times; for 'poisson' contains exponentially spaced times truncated to the specified duration.
    """
    import numpy as np
    
    # Input validation
    if train_type not in ('none', 'regular', 'poisson'):
        logger.warning(f"Unknown train_type '{train_type}', defaulting to 'none'")
        train_type = 'none'
    
    if train_type == 'none':
        return np.array(manual_times or [], dtype=np.float64)
    
    if freq_hz <= 0:
        logger.warning(f"Invalid freq_hz {freq_hz}, must be > 0. Defaulting to 40 Hz")
        freq_hz = 40.0
    
    if duration_ms <= 0:
        logger.warning(f"Invalid duration_ms {duration_ms}, must be > 0. Defaulting to 200 ms")
        duration_ms = 200.0

    from core.stochastic_rng import get_rng, StochasticRNG

    if train_type == 'regular':
        if freq_hz <= 0:
            return np.array([t_start], dtype=np.float64)  # Fallback: single event
        isi_ms = 1000.0 / freq_hz
        return np.arange(t_start, t_start + duration_ms, isi_ms, dtype=np.float64)
    elif train_type == 'poisson':
        if freq_hz <= 0:
            return np.array([t_start], dtype=np.float64)  # Fallback: single event
        rate_ms = freq_hz / 1000.0
        if rate_ms <= 0:
            return np.array([t_start], dtype=np.float64)  # Fallback: single event
        expected_spikes = int(duration_ms * rate_ms * 1.5)
        
        # Use a temporary RNG with a fixed seed based on the parameters
        # This ensures the previewer and the solver get the EXACT same train for the same settings
        if seed_hash is not None:
            temp_rng = StochasticRNG(seed_hash)
            intervals = temp_rng.exponential(1.0 / rate_ms, expected_spikes)
        else:
            rng = get_rng()
            intervals = rng.exponential(1.0 / rate_ms, expected_spikes)
        times = t_start + np.cumsum(intervals)
        return times[times < (t_start + duration_ms)].astype(np.float64)
    return np.array([], dtype=np.float64)


class NeuronSolver:
    """Multi-threaded simulation engine v10.0."""

    def __init__(self, config: FullModelConfig):
        self.config   = config
        self.registry = ChannelRegistry()

    @staticmethod
    def _build_b_ca_vector(cfg, morph) -> np.ndarray:
        """Compute per-compartment B_Ca from surface/volume ratio (Stage 3.4).

        Thin compartments get larger B_Ca because their surface-to-volume
        ratio is higher, producing larger Ca²⁺ transients — matching the
        physiology of thin dendrites vs. large somata.

        The user's cfg.calcium.B_Ca is treated as the soma value;
        other compartments are scaled by (A/V)_i / (A/V)_soma.
        """
        n_comp = morph['N_comp']
        diameters = morph['diameters']
        dx = cfg.morphology.dx
        b_ca_base = cfg.calcium.B_Ca

        # Surface-to-volume ratio:
        #   Soma (sphere):   A/V = 6/d
        #   Cylinder:        A/V = 4/d
        d_soma = diameters[0]
        av_soma = 6.0 / max(d_soma, 1e-12)  # sphere, guard against zero diameter

        b_ca_v = np.empty(n_comp, dtype=np.float64)
        b_ca_v[0] = b_ca_base  # soma = user value

        for i in range(1, n_comp):
            av_i = 4.0 / max(diameters[i], 1e-12)  # cylinder, guard against zero diameter
            b_ca_v[i] = b_ca_base * (av_i / av_soma)

        return b_ca_v

    # ─────────────────────────────────────────────────────────────────
    def run_single(self, custom_config: FullModelConfig = None) -> SimulationResult:
        """
        Run a single deterministic simulation using the current (or provided) configuration.
        
        Parameters:
            custom_config (FullModelConfig | None): Optional config to use for this run; if omitted, uses the solver's stored config.
        
        Returns:
            SimulationResult: Container with time vector, state matrix, compartment count, and derived outputs.
        
        Raises:
            RuntimeError: If both primary and fallback ODE integrators fail to produce a successful solution.
        """
        cfg = custom_config or self.config
        if cfg.stim.jacobian_mode == "native_hines":
            return self.run_native(cfg)
        for msg in validate_simulation_config(cfg):
            logger.warning("Simulation config warning: %s", msg)
        
        # ── Simulation complexity estimation ──
        rt = estimate_simulation_runtime(cfg)
        if rt["estimated_seconds"] > 30.0:
            logger.warning(
                "Heavy simulation detected: steps=%s, compartments=%s, channel-gates=%s, est=%.1fs, jac=%s",
                f"{int(rt['n_steps']):,}",
                int(rt["n_comp"]),
                int(rt["n_channels"]),
                rt["estimated_seconds"],
                cfg.stim.jacobian_mode,
            )
        
        import time
        t_start = time.time()

        morph  = MorphologyBuilder.build(cfg)
        n_comp = morph['N_comp']

        y0 = self.registry.compute_initial_states(cfg.channels.EL, cfg)

        s_map = {
            'const': 0, 'pulse': 1, 'alpha': 2, 'ou_noise': 3,
            'AMPA': 4, 'NMDA': 5, 'GABAA': 6, 'GABAB': 7,
            'Kainate': 8, 'Nicotinic': 9, 'zap': 10,
        }
        t_kelvin = cfg.env.T_celsius + 273.15

        stim_mode_map = {'soma': 0, 'ais': 1, 'dendritic_filtered': 2}
        # --- Primary stimulus source ---
        primary_stim_type = cfg.stim.stim_type
        primary_iext = cfg.stim.Iext
        primary_t0 = cfg.stim.pulse_start
        primary_td = cfg.stim.pulse_dur
        primary_atau = cfg.stim.alpha_tau
        primary_zap_f0 = cfg.stim.zap_f0_hz
        primary_zap_f1 = cfg.stim.zap_f1_hz
        primary_zap_rise = getattr(cfg.stim, 'zap_rise_ms', 5.0)  # Tukey window rise time
        primary_stim_comp = cfg.stim.stim_comp
        primary_location = cfg.stim_location.location

        # --- Dual stimulation detection and parameter preparation ---
        dual_stim_enabled = 0
        dual_cfg = getattr(cfg, 'dual_stimulation', None)
        if dual_cfg is not None and hasattr(dual_cfg, 'enabled') and dual_cfg.enabled:
            dual_stim_enabled = 1
            # Primary stimulus always comes from cfg.stim, not dual config

        stype = s_map.get(primary_stim_type, 0)
        stim_mode = stim_mode_map.get(primary_location, 0)
        use_dfilter_primary = int(
            stim_mode == 2
            and cfg.dendritic_filter.enabled
            and cfg.dendritic_filter.tau_dendritic_ms > 0.0
        )
        if use_dfilter_primary == 1:
            y0 = np.concatenate([y0, np.array([0.0])])

        # Dynamic AC attenuation parameters (v10.3) - for real-time frequency-dependent calculation
        dfilter_distance_um = cfg.dendritic_filter.distance_um if stim_mode == 2 else 0.0
        dfilter_lambda_um = cfg.dendritic_filter.space_constant_um if stim_mode == 2 else 1.0
        dfilter_tau_ms = cfg.dendritic_filter.tau_dendritic_ms
        dfilter_input_freq_hz = getattr(cfg.dendritic_filter, 'input_frequency', 100.0)
        # filter_mode: 0=DC (classic), 1=AC (physiological)
        dfilter_filter_mode = 1 if (stim_mode == 2 and 
                                   getattr(cfg.dendritic_filter, 'filter_mode', 'Classic (DC)') == "Physiological (AC)") else 0
        # Dynamic AC attenuation (v11.7): Use frequency-dependent attenuation in AC mode
        dfilter_attenuation = 1.0
        if dfilter_filter_mode == 1 and dfilter_lambda_um > 0:
            dfilter_attenuation = get_ac_attenuation(
                dfilter_distance_um, dfilter_lambda_um, dfilter_tau_ms, dfilter_input_freq_hz
            )
        elif stim_mode == 2 and dfilter_lambda_um > 0:
            # DC fallback: classic exponential attenuation
            dfilter_attenuation = np.exp(-dfilter_distance_um / dfilter_lambda_um)

        # Secondary stimulus defaults.
        # Keep secondary defaults physically valid even when dual stimulation is disabled.
        # This protects against stricter external validators and stale serialized configs.
        stype_2, iext_2, t0_2, td_2, atau_2 = 0, 0.0, 0.0, 0.0, 1.0
        zap_f0_2, zap_f1_2, zap_rise_2 = primary_zap_f0, primary_zap_f1, primary_zap_rise
        stim_comp_2, stim_mode_2 = 0, 0
        use_dfilter_secondary = 0
        # Dynamic AC attenuation for secondary (dual) - defaults
        dfilter_distance_um_2, dfilter_lambda_um_2 = 0.0, 1.0
        dfilter_tau_ms_2, dfilter_input_freq_hz_2 = 0.0, 100.0
        dfilter_filter_mode_2, dfilter_attenuation_2 = 0, 1.0

        if dual_stim_enabled == 1:
            stype_2 = s_map.get(dual_cfg.secondary_stim_type, 0)
            iext_2 = dual_cfg.secondary_Iext
            t0_2 = dual_cfg.secondary_start
            td_2 = dual_cfg.secondary_duration
            atau_2 = dual_cfg.secondary_alpha_tau
            stim_comp_2 = 0
            stim_mode_2 = stim_mode_map.get(dual_cfg.secondary_location, 0)
            zap_f0_2 = getattr(dual_cfg, "secondary_zap_f0_hz", zap_f0_2)
            zap_f1_2 = getattr(dual_cfg, "secondary_zap_f1_hz", zap_f1_2)
            zap_rise_2 = getattr(dual_cfg, "secondary_zap_rise_ms", zap_rise_2)
            dfilter_tau_ms_2 = dual_cfg.secondary_tau_dendritic_ms
            use_dfilter_secondary = int(stim_mode_2 == 2 and dfilter_tau_ms_2 > 0.0)
            if use_dfilter_secondary == 1:
                y0 = np.concatenate([y0, np.array([0.0])])
            # Dynamic AC parameters for secondary stimulus
            dfilter_distance_um_2 = dual_cfg.secondary_distance_um if stim_mode_2 == 2 else 0.0
            dfilter_lambda_um_2 = dual_cfg.secondary_space_constant_um if stim_mode_2 == 2 else 1.0
            dfilter_input_freq_hz_2 = getattr(dual_cfg, 'secondary_input_frequency', 100.0)
            dfilter_filter_mode_2 = 1 if (stim_mode_2 == 2 and 
                                          getattr(dual_cfg, 'secondary_filter_mode', 'Classic (DC)') == "Physiological (AC)") else 0
            # Dynamic AC attenuation for secondary (v11.7)
            if dfilter_filter_mode_2 == 1 and dfilter_lambda_um_2 > 0:
                dfilter_attenuation_2 = get_ac_attenuation(
                    dfilter_distance_um_2, dfilter_lambda_um_2, dfilter_tau_ms_2, dfilter_input_freq_hz_2
                )
            elif stim_mode_2 == 2 and dfilter_lambda_um_2 > 0:
                # DC fallback
                dfilter_attenuation_2 = np.exp(-dfilter_distance_um_2 / dfilter_lambda_um_2)

        # Generate ephemeral primary train
        # Create a stable seed based on the parameters
        seed_hash = _stable_seed_from_values(
            cfg.stim.synaptic_train_freq_hz,
            cfg.stim.synaptic_train_duration_ms,
            cfg.stim.pulse_start,
            cfg.stim.synaptic_train_type,
        )
        eff_event_times_1 = generate_effective_event_times(
            cfg.stim.synaptic_train_type, cfg.stim.synaptic_train_freq_hz,
            cfg.stim.synaptic_train_duration_ms, cfg.stim.pulse_start, cfg.stim.event_times, seed_hash=seed_hash
        )
        # CRITICAL: Sort event times for early termination optimization in get_event_driven_conductance
        # The break-on-future-event optimization requires strictly ascending order
        eff_event_times_1 = np.sort(eff_event_times_1)
        
        # Generate ephemeral secondary train
        eff_event_times_2 = np.zeros(0, dtype=np.float64)
        if dual_stim_enabled == 1 and dual_cfg is not None:
            seed_hash_2 = _stable_seed_from_values(
                getattr(dual_cfg, 'secondary_train_freq_hz', 40.0),
                getattr(dual_cfg, 'secondary_train_duration_ms', 200.0),
                getattr(dual_cfg, 'secondary_start', 0.0),
                getattr(dual_cfg, 'secondary_train_type', 'none'),
            )
            eff_event_times_2 = generate_effective_event_times(
                getattr(dual_cfg, 'secondary_train_type', 'none'),
                getattr(dual_cfg, 'secondary_train_freq_hz', 40.0),
                getattr(dual_cfg, 'secondary_train_duration_ms', 200.0),
                dual_cfg.secondary_start,
                dual_cfg.secondary_event_times,
                seed_hash=seed_hash_2
            )
            # Sort secondary events as well
            eff_event_times_2 = np.sort(eff_event_times_2)

        # ── Initialize RNG state for reproducibility ──
        from core.stochastic_rng import get_rng
        rng = get_rng()
        rng_state = rng.get_state()['state'] if (cfg.stim.stoch_gating or cfg.stim.noise_sigma > 0) else None
        na_i_rest_mM, k_o_rest_mM = _resolve_dynamic_atp_rest_values(cfg)

        # Precompute ZAP Tukey windows (primary & secondary)
        _zw1_t, _zw1_g, _zw1_n = _precompute_zap_window(primary_td, primary_zap_rise) if stype == 10 else (np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64), 0)
        _zw2_t, _zw2_g, _zw2_n = _precompute_zap_window(td_2, zap_rise_2) if stype_2 == 10 else (np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64), 0)

        rhs_values = {
            "n_comp": n_comp,
            "en_ih": cfg.channels.enable_Ih,
            "en_ica": cfg.channels.enable_ICa,
            "en_ia": cfg.channels.enable_IA,
            "en_sk": cfg.channels.enable_SK,
            "dyn_ca": cfg.calcium.dynamic_Ca,
            "en_itca": cfg.channels.enable_ITCa,
            "en_im": cfg.channels.enable_IM,
            "en_nap": cfg.channels.enable_NaP,
            "en_nar": cfg.channels.enable_NaR,
            "dyn_atp": cfg.metabolism.enable_dynamic_atp,
            "gbar_mat": np.vstack([
                morph['gNa_v'],
                morph['gK_v'],
                morph['gL_v'],
                morph['gIh_v'],
                morph['gCa_v'],
                morph['gA_v'],
                morph['gSK_v'],
                morph['gTCa_v'],
                morph['gIM_v'],
                morph['gNaP_v'],
                morph['gNaR_v'],
            ]),
            "ena": cfg.channels.ENa,
            "ek": cfg.channels.EK,
            "el": cfg.channels.EL,
            "eih": cfg.channels.E_Ih,
            "ea": cfg.channels.EK,  # A-current uses K reversal potential
            "cm_v": morph['Cm_v'],
            "l_data": morph['L_data'],
            "l_indices": morph['L_indices'],
            "l_indptr": morph['L_indptr'],
            "phi_mat": np.vstack([
                cfg.env.build_phi_vector(cfg.env.Q10_Na, n_comp),
                cfg.env.build_phi_vector(cfg.env.Q10_K, n_comp),
                cfg.env.build_phi_vector(cfg.env.Q10_Ih, n_comp),
                cfg.env.build_phi_vector(cfg.env.Q10_Ca, n_comp),
                cfg.env.build_phi_vector(cfg.env.Q10_IA, n_comp),
                cfg.env.build_phi_vector(cfg.env.Q10_TCa, n_comp),
                cfg.env.build_phi_vector(cfg.env.Q10_IM, n_comp),
                cfg.env.build_phi_vector(cfg.env.Q10_NaP, n_comp),
                cfg.env.build_phi_vector(cfg.env.Q10_NaR, n_comp),
            ]),
            "t_kelvin": t_kelvin,
            "ca_ext": cfg.calcium.Ca_ext,
            "ca_rest": cfg.calcium.Ca_rest,
            "tau_ca": cfg.calcium.tau_Ca,
            "mg_ext": cfg.env.Mg_ext,
            "nmda_mg_block_mM": cfg.env.nmda_mg_block_mM,
            "tau_sk": cfg.channels.tau_SK,
            "im_speed_multiplier": cfg.channels.im_speed_multiplier,
            "b_ca": self._build_b_ca_vector(cfg, morph),
            "g_katp_max": cfg.metabolism.g_katp_max,
            "katp_kd_atp_mM": cfg.metabolism.katp_kd_atp_mM,
            "atp_max_mM": cfg.metabolism.atp_max_mM,
            "atp_synthesis_rate": cfg.metabolism.atp_synthesis_rate,
            "na_i_rest_mM": na_i_rest_mM,
            "na_ext_mM": cfg.metabolism.na_ext_mM,
            "k_i_mM": cfg.metabolism.k_i_mM,
            "k_o_rest_mM": k_o_rest_mM,
            "ion_drift_gain": cfg.metabolism.ion_drift_gain,
            "k_o_clearance_tau_ms": cfg.metabolism.k_o_clearance_tau_ms,
            "pump_max_capacity": getattr(cfg.metabolism, 'pump_max_capacity', 0.25),
            "km_na": getattr(cfg.metabolism, 'km_na', 15.0),
            "stype": stype,
            "iext": primary_iext,
            "t0": primary_t0,
            "td": primary_td,
            "atau": primary_atau,
            "zap_f0_hz": primary_zap_f0,
            "zap_f1_hz": primary_zap_f1,
            "zap_rise_ms": primary_zap_rise,
            "zap_win_t": _zw1_t,
            "zap_win_g": _zw1_g,
            "zap_win_size": np.int32(_zw1_n),
            "stim_comp": primary_stim_comp,
            "stim_mode": stim_mode,
            "use_dfilter_primary": use_dfilter_primary,
            "dfilter_distance_um": dfilter_distance_um,
            "dfilter_lambda_um": dfilter_lambda_um,
            "dfilter_tau_ms": dfilter_tau_ms,
            "dfilter_input_freq_hz": dfilter_input_freq_hz,
            "dfilter_filter_mode": dfilter_filter_mode,
            "dfilter_attenuation": dfilter_attenuation,
            "event_times_arr": eff_event_times_1,
            "n_events": int(len(eff_event_times_1)),
            "event_times_arr_2": eff_event_times_2,
            "n_events_2": int(len(eff_event_times_2)),
            "dual_stim_enabled": dual_stim_enabled,
            "gna_max": cfg.channels.gNa_max,
            "gk_max": cfg.channels.gK_max,
            "e_rev_syn_primary": cfg.channels.e_rev_syn_primary,
            "e_rev_syn_secondary": cfg.channels.e_rev_syn_secondary,
            "stype_2": stype_2,
            "iext_2": iext_2,
            "t0_2": t0_2,
            "td_2": td_2,
            "atau_2": atau_2,
            "zap_f0_hz_2": zap_f0_2,
            "zap_f1_hz_2": zap_f1_2,
            "zap_rise_ms_2": zap_rise_2,
            "zap_win_t_2": _zw2_t,
            "zap_win_g_2": _zw2_g,
            "zap_win_size_2": np.int32(_zw2_n),
            "stim_comp_2": stim_comp_2,
            "stim_mode_2": stim_mode_2,
            "use_dfilter_secondary": use_dfilter_secondary,
            "dfilter_distance_um_2": dfilter_distance_um_2,
            "dfilter_lambda_um_2": dfilter_lambda_um_2,
            "dfilter_tau_ms_2": dfilter_tau_ms_2,
            "dfilter_input_freq_hz_2": dfilter_input_freq_hz_2,
            "dfilter_filter_mode_2": dfilter_filter_mode_2,
            "dfilter_attenuation_2": dfilter_attenuation_2,
        }
        # Create structured PhysicsParams container
        physics_params = create_physics_params(**rhs_values)

        # Pre-allocate RHS output buffer once (reused across integration steps)
        _dydt_buf = np.empty(len(y0), dtype=np.float64)

        def _rhs_fn(t, y):
            # Fill pre-allocated scratch buffer in-place (zero-alloc RHS eval),
            # then return a COPY so SciPy BDF can safely retain references to
            # previous evaluations in its internal history buffer.
            # Native Hines path is fully zero-allocation and never reaches here.
            _dydt_buf.fill(0.0)
            rhs_multicompartment(t, y, physics_params, _dydt_buf)
            return _dydt_buf.copy()

        t_eval = np.arange(0.0, cfg.stim.t_sim, cfg.stim.dt_eval)

        # ── Optimization settings ──
        # max_step prevents integrator from taking too large steps
        # which can cause instability in stiff systems
        max_step = min(cfg.stim.dt_eval * 5, 1.0)  # Max 1ms or 5x evaluation step

        jacobian_mode = cfg.stim.jacobian_mode
        jacobian_options = {}
        _sparsity_kwargs = dict(
            n_comp=n_comp,
            en_ih=cfg.channels.enable_Ih,
            en_ica=cfg.channels.enable_ICa,
            en_ia=cfg.channels.enable_IA,
            en_sk=cfg.channels.enable_SK,
            dyn_ca=cfg.calcium.dynamic_Ca,
            dyn_atp=cfg.metabolism.enable_dynamic_atp,
            l_indices=morph["L_indices"],
            l_indptr=morph["L_indptr"],
            use_dfilter_primary=use_dfilter_primary,
            use_dfilter_secondary=use_dfilter_secondary,
            en_itca=cfg.channels.enable_ITCa,
            en_im=cfg.channels.enable_IM,
            en_nap=cfg.channels.enable_NaP,
            en_nar=cfg.channels.enable_NaR,
            # Pass per-compartment gNa so the sparsity builder can omit
            # m/h→V connections for near-zero Na compartments (demyelinated
            # trunk).  Prevents SuperLU "Factor is exactly singular" when
            # gNa_trunk_mult << 1 creates 35 nearly-identical passive rows.
            gNa_v=morph.get("gNa_v"),
        )
        if jacobian_mode == "sparse_fd":
            jacobian_options["jac_sparsity"] = build_jacobian_sparsity(**_sparsity_kwargs)
        elif jacobian_mode == "analytic_sparse":
            sparsity = build_jacobian_sparsity(**_sparsity_kwargs)
            _jac_raw = make_analytic_jacobian(sparsity)
            # Wrap as closure: jacobian uses physics params only (no dydt buffer).
            def _jac_fn(t, y):
                return _jac_raw(t, y, physics_params)
            jacobian_options["jac"] = _jac_fn
        elif jacobian_mode != "dense_fd":
            raise ValueError(f"Unsupported jacobian_mode={jacobian_mode}")

        def _solve(method: str, *, rtol: float, atol: float, use_jacobian: bool):
            opts = jacobian_options if use_jacobian else {}
            return solve_ivp(
                _rhs_fn,
                (0.0, cfg.stim.t_sim),
                y0,
                # args= omitted: _rhs_fn captures args and _dydt_buf via closure
                method=method,
                t_eval=t_eval,
                rtol=rtol,
                atol=atol,
                max_step=max_step,
                dense_output=False,  # Save memory
                **opts,
            )

        # Build analytic-sparse Jacobian closure once (used as fast fallback
        # when sparse/dense FD produces a singular LU factor).
        _analytic_jac_opts: dict = {}
        if jacobian_mode in ("sparse_fd", "dense_fd"):
            try:
                _as_sparsity = build_jacobian_sparsity(**_sparsity_kwargs)
                _as_raw = make_analytic_jacobian(_as_sparsity)
                def _as_jac_fn(t, y):
                    return _as_raw(t, y, physics_params)
                _analytic_jac_opts = {"jac": _as_jac_fn}
            except Exception:
                _analytic_jac_opts = {}  # analytic fallback unavailable

        def _solve_analytic_bdf(rtol: float, atol: float):
            """BDF with exact analytic Jacobian — fast fallback for FD singularity."""
            return solve_ivp(
                _rhs_fn,
                (0.0, cfg.stim.t_sim),
                y0,
                method="BDF",
                t_eval=t_eval,
                rtol=rtol,
                atol=atol,
                max_step=max_step,
                dense_output=False,
                **_analytic_jac_opts,
            )

        try:
            sol = _solve("BDF", rtol=1e-5, atol=1e-7, use_jacobian=True)
            bdf_exception_message = None
        except Exception as exc:
            # Some sparse factorization failures (e.g., exactly singular Jacobian)
            # can propagate from SciPy internals before `sol.success` is populated.
            # Treat this path as a BDF failure and fall back to LSODA.
            bdf_exception_message = f"{type(exc).__name__}: {exc}"
            logger.warning("Primary integrator crashed during BDF step: %s", bdf_exception_message)
            sol = _solve("LSODA", rtol=3e-5, atol=3e-7, use_jacobian=False)
            if not sol.success:
                raise RuntimeError(
                    "Integrator failed after fallback. "
                    f"BDF exception='{bdf_exception_message}'; LSODA='{sol.message}'"
                ) from exc
        
        # Report actual simulation time
        t_elapsed = time.time() - t_start
        if rt["estimated_seconds"] > 30.0 or t_elapsed > 10:
            logger.info("   Completed in %.1fs", t_elapsed)

        if not sol.success:
            first_message = str(sol.message)
            logger.warning("Primary integrator failed (BDF): %s", first_message)
            # Fallback for stiff/ill-conditioned episodes where BDF can fail with
            # "Required step size is less than spacing between numbers."
            # LSODA often recovers these trajectories by switching methods internally.
            sol = _solve("LSODA", rtol=3e-5, atol=3e-7, use_jacobian=False)
            if not sol.success:
                raise RuntimeError(
                    "Integrator failed after fallback. "
                    f"BDF='{first_message}'; LSODA='{sol.message}'"
                )

        res = SimulationResult(sol.t, sol.y, n_comp, cfg)
        self._post_process_physics(res, morph)
        res.morph = morph        # store for current-balance analysis
        return res

    # ─────────────────────────────────────────────────────────────────
    def _post_process_physics(self, res: SimulationResult, morph: dict):
        """
        Reconstruct ion-channel current density arrays for every compartment and compute a compartment-normalized ATP consumption estimate.
        
        This populates SimulationResult.currents with per-channel current density arrays (keys include 'Na', 'K', 'Leak' and any enabled channels such as 'Ih', 'ICa', 'IA', 'ITCa', 'IM', 'NaP', 'NaR', 'SK', and 'KATP' when applicable) and fills SimulationResult.atp_estimate and SimulationResult.atp_breakdown with nmol/cm² estimates for Na-pump, Ca-pump, baseline, and total ATP consumption over the simulated interval. When dynamic calcium is enabled and intracellular Ca is available, calcium reversal potentials are computed from Nernst; KATP is computed when metabolism.dynamic ATP is enabled using the ATP state from the end of the state vector. Warnings are emitted if channels are enabled in the config but required morph keys are missing.
        
        Parameters:
            res (SimulationResult): Simulation output container whose state vector and config are used; mutated to add reconstructed currents and ATP estimates.
            morph (dict): Morphology-derived per-compartment conductance vectors required for current reconstruction (must contain 'gNa_v', 'gK_v', 'gL_v'; additional keys used if corresponding channels are enabled).
        """
        y, n, cfg = res.y, res.n_comp, res.config
        v  = y[0:n, :]
        m  = y[n   :2*n, :]
        h  = y[2*n :3*n, :]
        nk = y[3*n :4*n, :]

        # Safety check for required morph keys
        required_keys = ['gNa_v', 'gK_v', 'gL_v']
        for key in required_keys:
            if key not in morph:
                logger.error(f"Missing morph key: {key}")
                return
        
        t_kelvin = cfg.env.T_celsius + 273.15
        ena_eff = cfg.channels.ENa
        ek_eff = cfg.channels.EK
        if cfg.metabolism.enable_dynamic_atp and res.na_i is not None:
            na_i = np.clip(np.asarray(res.na_i, dtype=float), NA_I_MIN_M_M, NA_I_MAX_M_M)
            ena_eff = (R_GAS * t_kelvin / F_CONST) * np.log(cfg.metabolism.na_ext_mM / na_i) * 1000.0
        if cfg.metabolism.enable_dynamic_atp and res.k_o is not None:
            k_o = np.clip(np.asarray(res.k_o, dtype=float), K_O_MIN_M_M, K_O_MAX_M_M)
            ek_eff = (R_GAS * t_kelvin / F_CONST) * np.log(k_o / cfg.metabolism.k_i_mM) * 1000.0

        # Broadcast conductances to shape (n_comp, 1) for 2D current calculation
        g_na = morph['gNa_v'][:, np.newaxis]
        g_k  = morph['gK_v'][:, np.newaxis]
        g_l  = morph['gL_v'][:, np.newaxis]
        
        res.currents['Na']   = g_na * (m ** 3) * h * (v - ena_eff)
        res.currents['K']    = g_k  * (nk ** 4) * (v - ek_eff)
        res.currents['Leak'] = g_l  * (v - cfg.channels.EL)

        offsets = build_state_offsets(
            n,
            en_ih=cfg.channels.enable_Ih,
            en_ica=cfg.channels.enable_ICa,
            en_ia=cfg.channels.enable_IA,
            en_sk=cfg.channels.enable_SK,
            dyn_ca=cfg.calcium.dynamic_Ca,
            en_itca=cfg.channels.enable_ITCa,
            en_im=cfg.channels.enable_IM,
            en_nap=cfg.channels.enable_NaP,
            en_nar=cfg.channels.enable_NaR,
            dyn_atp=cfg.metabolism.enable_dynamic_atp,
            use_dfilter_primary=1 if res.v_dendritic_filtered is not None else 0,
            use_dfilter_secondary=1 if res.v_dendritic_filtered_secondary is not None else 0,
        )

        if cfg.channels.enable_Ih:
            if 'gIh_v' in morph:
                r = y[int(offsets.off_r):int(offsets.off_r) + n, :]
                g_ih = morph['gIh_v'][:, np.newaxis]
                res.currents['Ih'] = g_ih * r * (v - cfg.channels.E_Ih)
            else:
                logger.warning("Ih channel enabled but gIh_v missing from morph")

        if cfg.channels.enable_ICa:
            if 'gCa_v' in morph:
                s = y[int(offsets.off_s):int(offsets.off_s) + n, :]
                u = y[int(offsets.off_u):int(offsets.off_u) + n, :]
                g_ca = morph['gCa_v'][:, np.newaxis]
                if cfg.calcium.dynamic_Ca and res.ca_i is not None:
                    ca_i = np.maximum(res.ca_i, 1e-9)
                    e_ca = (R_GAS * t_kelvin / (2.0 * F_CONST)) * np.log(cfg.calcium.Ca_ext / ca_i) * 1000.0
                else:
                    e_ca = 120.0
                res.currents['ICa'] = g_ca * (s ** 2) * u * (v - e_ca)
            else:
                logger.warning("ICa channel enabled but gCa_v missing from morph")

        if cfg.channels.enable_IA:
            if 'gA_v' in morph:
                a = y[int(offsets.off_a):int(offsets.off_a) + n, :]
                b = y[int(offsets.off_b):int(offsets.off_b) + n, :]
                g_a = morph['gA_v'][:, np.newaxis]
                res.currents['IA'] = g_a * a * b * (v - ek_eff)  # A-current uses EK
            else:
                logger.warning("IA channel enabled but gA_v missing from morph")

        if cfg.channels.enable_ITCa:
            if 'gTCa_v' in morph:
                p_t = y[int(offsets.off_p):int(offsets.off_p) + n, :]
                q_t = y[int(offsets.off_q):int(offsets.off_q) + n, :]
                g_tca = morph['gTCa_v'][:, np.newaxis]
                if not cfg.channels.enable_ICa:
                    # e_ca not yet computed - compute here
                    if cfg.calcium.dynamic_Ca and res.ca_i is not None:
                        t_kelvin = cfg.env.T_celsius + 273.15
                        ca_i = np.maximum(res.ca_i, 1e-9)
                        e_ca = (R_GAS * t_kelvin / (2.0 * F_CONST)) * np.log(cfg.calcium.Ca_ext / ca_i) * 1000.0
                    else:
                        e_ca = 120.0
                res.currents['ITCa'] = g_tca * (p_t ** 2) * q_t * (v - e_ca)
            else:
                logger.warning("ITCa channel enabled but gTCa_v missing from morph")

        if cfg.channels.enable_IM:
            if 'gIM_v' in morph:
                w_m = y[int(offsets.off_w):int(offsets.off_w) + n, :]
                g_im = morph['gIM_v'][:, np.newaxis]
                res.currents['IM'] = g_im * w_m * (v - ek_eff)
            else:
                logger.warning("IM channel enabled but gIM_v missing from morph")

        if cfg.channels.enable_NaP:
            if 'gNaP_v' in morph:
                x_p = y[int(offsets.off_x):int(offsets.off_x) + n, :]
                g_nap = morph['gNaP_v'][:, np.newaxis]
                res.currents['NaP'] = g_nap * x_p * (v - ena_eff)
            else:
                logger.warning("NaP channel enabled but gNaP_v missing from morph")

        if cfg.channels.enable_NaR:
            if 'gNaR_v' in morph:
                y_r = y[int(offsets.off_y):int(offsets.off_y) + n, :]
                j_r = y[int(offsets.off_j):int(offsets.off_j) + n, :]
                g_nar = morph['gNaR_v'][:, np.newaxis]
                res.currents['NaR'] = g_nar * y_r * j_r * (v - ena_eff)
            else:
                logger.warning("NaR channel enabled but gNaR_v missing from morph")

        if cfg.channels.enable_SK:
            if 'gSK_v' in morph:
                z_sk = y[int(offsets.off_zsk):int(offsets.off_zsk) + n, :]
                g_sk = morph['gSK_v'][:, np.newaxis]
                res.currents['SK'] = g_sk * z_sk * (v - ek_eff)
            else:
                logger.warning("SK channel enabled but gSK_v missing from morph")

        # Extract K_ATP current if metabolism is enabled
        if cfg.metabolism.enable_dynamic_atp and res.atp_level is not None:
            atp_ratio = res.atp_level / cfg.metabolism.katp_kd_atp_mM
            g_katp = cfg.metabolism.g_katp_max / (1.0 + atp_ratio ** 2)
            res.currents['KATP'] = g_katp * (v - ek_eff)

        # Electrogenic Na/K pump current using Michaelis-Menten kinetics
        # Honest calculation from [Na+]i, [K+]o, [ATP]i - matches native_loop.py physics
        pump_drive = None  # Will hold pump current for ATP consumption calculation below
        if cfg.metabolism.enable_dynamic_atp and res.na_i is not None and res.k_o is not None:
            # Extract arrays and determine time dimension
            na_i_arr = np.clip(np.asarray(res.na_i, dtype=float), NA_I_MIN_M_M, NA_I_MAX_M_M)
            k_o_arr = np.clip(np.asarray(res.k_o, dtype=float), K_O_MIN_M_M, K_O_MAX_M_M)
            n_t = na_i_arr.shape[1] if na_i_arr.ndim == 2 else na_i_arr.shape[0]
            atp_arr = np.asarray(res.atp_level, dtype=float) if res.atp_level is not None else np.full(n_t, 2.0)

            # Helper to extract soma value (compartment 0) at time t
            _get = lambda arr, t: arr[0, t] if arr.ndim == 2 else arr[t]

            # Vectorized computation of pump current for all timepoints (v11.7)
            # Uses Numba-jitted helper instead of slow Python list-comprehension
            _pump_max = getattr(cfg.metabolism, 'pump_max_capacity', 0.25)
            _km_na = getattr(cfg.metabolism, 'km_na', 15.0)
            # Extract 1D arrays (soma compartment 0)
            na_1d = na_i_arr[0, :] if na_i_arr.ndim == 2 else na_i_arr
            ko_1d = k_o_arr[0, :] if k_o_arr.ndim == 2 else k_o_arr
            atp_1d = atp_arr[0, :] if atp_arr.ndim == 2 else atp_arr
            pump_current = get_pump_current_array(na_1d, ko_1d, atp_1d, _pump_max, _km_na)
            # Reshape to (1, n_t) to match other currents broadcasting in compute_current_balance
            pump_drive = pump_current.reshape(1, -1)
            res.currents['PumpNaK'] = pump_drive
        else:
            # Fallback: static ATP - use honest Michaelis-Menten with fixed resting ion concentrations
            # This eliminates systematic error from 5% heuristic while maintaining compatibility
            na_i_fixed = float(cfg.metabolism.na_i_rest_mM)
            k_o_fixed = float(cfg.metabolism.k_o_rest_mM)
            atp_fixed = float(cfg.metabolism.atp_max_mM)

            # Compute single pump value (constant in time for static ATP)
            _pump_max = getattr(cfg.metabolism, 'pump_max_capacity', 0.25)
            _km_na = getattr(cfg.metabolism, 'km_na', 15.0)
            pump_val = compute_na_k_pump_current(na_i_fixed, k_o_fixed, atp_fixed, _pump_max, _km_na)
            n_t = len(res.t)

            # Broadcast to (1, n_t) for consistency with dynamic ATP branch
            pump_drive = np.full((1, n_t), pump_val, dtype=float)
            res.currents['PumpNaK'] = pump_drive

        res._finalize_current_shapes()

        # ── ATP consumption estimate ──────────────────────────────────
        # Na+/K+-ATPase: 3 Na+ pumped per ATP hydrolysis (Skou 1957).
        # Ca²+-ATPase (PMCA): 1 Ca²+ pumped per ATP (Bhatt et al. 2005).
        # Baseline pump: resting Na+/K+-ATPase ≈ 50% of neuronal ATP even
        # without spiking (Attwell & Laughlin 2001, J Cereb Blood Flow Metab).
        F = 96485.33  # Faraday constant [C/mol]

        # 1. Na+ pump cost: |Q_Na| / (3·F)  [µC/cm²·ms → mol → nmol ATP/cm²]
        t_arr = np.asarray(res.t, dtype=float)
        # Safety: pump_drive could be None if dynamic ATP enabled but na_i/k_o unavailable
        i_na_pump = pump_drive if pump_drive is not None else np.zeros_like(t_arr)
        if np.ndim(i_na_pump) == 2:
            i_na_spatial_sum = np.sum(i_na_pump, axis=0)
        else:
            i_na_spatial_sum = i_na_pump
        q_na = float(trapezoid(i_na_spatial_sum, x=t_arr))
        atp_na = (q_na * 1e-9) / (3.0 * F) * 1e9  # nmol/cm²

        # 2. Ca²+ pump cost: |Q_Ca| / (1·2F)  [divalent ion, z=2]
        atp_ca = 0.0
        for ca_key in ('ICa', 'ITCa'):
            if ca_key in res.currents:
                i_ca_curr = np.abs(res.currents[ca_key])
                i_ca_spatial_sum = np.sum(i_ca_curr, axis=0) if i_ca_curr.ndim == 2 else i_ca_curr
                q_ca = float(trapezoid(i_ca_spatial_sum, x=t_arr))
                atp_ca += (q_ca * 1e-9) / (1.0 * 2.0 * F) * 1e9

        # 3. Baseline Na+/K+-ATPase resting cost
        # ~0.4 nmol ATP/(cm²·s) for a resting neuron (Attwell & Laughlin 2001)
        t_sim_s = cfg.stim.t_sim * 1e-3
        atp_baseline = 0.4 * t_sim_s  # nmol/cm²

        atp_total = atp_na + atp_ca + atp_baseline
        res.atp_estimate = atp_total
        res.atp_breakdown = {
            'Na_pump': atp_na,
            'Ca_pump': atp_ca,
            'baseline': atp_baseline,
            'total': atp_total,
        }

    # ─────────────────────────────────────────────────────────────────
    def run_native(self, custom_config: "FullModelConfig | None" = None,
                    calc_lle: bool = False, lle_delta: float = 1e-6,
                    lle_t_evolve: float = 1.0,
                    lle_subspace_mode: str | int = "v_only",
                    lle_custom_mask = None,
                    lle_weights = None) -> SimulationResult:
        """
        Run the model simulation using the native Hines fixed-step solver.
        
        Performs a simulation using the O(N) Hines tree solver (Backward-Euler integration)
        and returns a SimulationResult compatible with the rest of the analysis pipeline.
        Requires the solver mode to be configured for native Hines execution.
        
        Returns:
            SimulationResult: Container with time vector, state matrix, extracted voltages,
            reconstructed currents, ATP estimates, and morphology. If the native solver
            detects numerical divergence the returned result contains the partial output
            and has `res.diverged = True`.
        """
        from core.native_loop import run_native_loop, set_numba_random_seed

        cfg = custom_config or self.config

        morph  = MorphologyBuilder.build(cfg)
        n_comp = morph["N_comp"]

        y0 = self.registry.compute_initial_states(cfg.channels.EL, cfg)

        # ── Stimulus maps (mirrors run_single) ──
        s_map = {
            "const": 0, "pulse": 1, "alpha": 2, "ou_noise": 3,
            "AMPA": 4, "NMDA": 5, "GABAA": 6, "GABAB": 7,
            "Kainate": 8, "Nicotinic": 9, "zap": 10,
        }
        stim_mode_map = {"soma": 0, "ais": 1, "dendritic_filtered": 2}
        t_kelvin = cfg.env.T_celsius + 273.15

        # --- Dual stimulation detection and parameter preparation ---
        dual_stim_enabled = 0
        dual_cfg = getattr(cfg, "dual_stimulation", None)
        if dual_cfg is not None and hasattr(dual_cfg, 'enabled') and dual_cfg.enabled:
            dual_stim_enabled = 1
        
        stype = s_map.get(cfg.stim.stim_type, 0)
        iext = cfg.stim.Iext
        t0 = cfg.stim.pulse_start
        td = cfg.stim.pulse_dur
        atau = cfg.stim.alpha_tau
        stim_mode = stim_mode_map.get(cfg.stim_location.location, 0)
        stim_comp = cfg.stim.stim_comp
        zap_f0 = cfg.stim.zap_f0_hz
        zap_f1 = cfg.stim.zap_f1_hz
        zap_rise = getattr(cfg.stim, 'zap_rise_ms', 5.0)  # Tukey window rise time
        use_dfilter_primary = int(
            stim_mode == 2
            and cfg.dendritic_filter.enabled
            and cfg.dendritic_filter.tau_dendritic_ms > 0.0
        )
        if use_dfilter_primary == 1:
            y0 = np.concatenate([y0, np.array([0.0])])
        # Dynamic AC attenuation parameters for native solver (v10.3)
        dfilter_distance_um = cfg.dendritic_filter.distance_um if stim_mode == 2 else 0.0
        dfilter_lambda_um = cfg.dendritic_filter.space_constant_um if stim_mode == 2 else 150.0
        dfilter_tau_ms = cfg.dendritic_filter.tau_dendritic_ms
        dfilter_input_freq_hz = getattr(cfg.dendritic_filter, 'input_frequency', 100.0)
        dfilter_filter_mode = 1 if (stim_mode == 2 and 
                                    getattr(cfg.dendritic_filter, 'filter_mode', 'Classic (DC)') == "Physiological (AC)") else 0
        
        dfilter_attenuation = 1.0
        if dfilter_filter_mode == 1 and dfilter_lambda_um > 0:
            dfilter_attenuation = get_ac_attenuation(
                dfilter_distance_um, dfilter_lambda_um, dfilter_tau_ms, dfilter_input_freq_hz
            )
        elif stim_mode == 2 and dfilter_lambda_um > 0:
            dfilter_attenuation = np.exp(-dfilter_distance_um / dfilter_lambda_um)

        stype_2, iext_2, t0_2, td_2, atau_2 = 0, 0.0, 0.0, 0.0, 1.0
        zap_f0_2, zap_f1_2, zap_rise_2 = zap_f0, zap_f1, zap_rise
        stim_comp_2, stim_mode_2 = 0, 0
        use_dfilter_secondary = 0
        # Secondary AC filter defaults (disabled when no dual stim)
        dfilter_distance_um_2, dfilter_lambda_um_2 = 0.0, 150.0
        dfilter_tau_ms_2, dfilter_input_freq_hz_2 = 0.0, 100.0
        dfilter_filter_mode_2, dfilter_attenuation_2 = 0, 1.0
        if dual_stim_enabled == 1:
            stype_2 = s_map.get(dual_cfg.secondary_stim_type, 0)
            iext_2 = dual_cfg.secondary_Iext
            t0_2 = dual_cfg.secondary_start
            td_2 = dual_cfg.secondary_duration
            atau_2 = dual_cfg.secondary_alpha_tau
            stim_mode_2 = stim_mode_map.get(dual_cfg.secondary_location, 0)
            zap_f0_2 = getattr(dual_cfg, "secondary_zap_f0_hz", zap_f0_2)
            zap_f1_2 = getattr(dual_cfg, "secondary_zap_f1_hz", zap_f1_2)
            zap_rise_2 = getattr(dual_cfg, "secondary_zap_rise_ms", zap_rise_2)
            # Secondary AC filter parameters (dual stimulus)
            dfilter_distance_um_2 = dual_cfg.secondary_distance_um if stim_mode_2 == 2 else 0.0
            dfilter_lambda_um_2 = dual_cfg.secondary_space_constant_um if stim_mode_2 == 2 else 150.0
            dfilter_tau_ms_2 = dual_cfg.secondary_tau_dendritic_ms
            dfilter_input_freq_hz_2 = getattr(dual_cfg, 'secondary_input_frequency', 100.0)
            dfilter_filter_mode_2 = 1 if (stim_mode_2 == 2 and 
                                        getattr(dual_cfg, 'secondary_filter_mode', 'Classic (DC)') == "Physiological (AC)") else 0
            use_dfilter_secondary = int(stim_mode_2 == 2 and dfilter_tau_ms_2 > 0.0)
            if use_dfilter_secondary == 1:
                y0 = np.concatenate([y0, np.array([0.0])])
            # Dynamic AC attenuation for secondary (mirrors primary logic)
            if dfilter_filter_mode_2 == 1 and dfilter_lambda_um_2 > 0:
                dfilter_attenuation_2 = get_ac_attenuation(
                    dfilter_distance_um_2, dfilter_lambda_um_2, dfilter_tau_ms_2, dfilter_input_freq_hz_2
                )
            elif stim_mode_2 == 2 and dfilter_lambda_um_2 > 0:
                dfilter_attenuation_2 = np.exp(-dfilter_distance_um_2 / dfilter_lambda_um_2)
            # Generate event times for secondary stimulus (synaptic train)
            event_times_arr_2 = np.zeros(0, dtype=np.float64)
            if stype_2 >= 4:  # Conductance-based synapse
                seed_hash_2 = _stable_seed_from_values(
                    getattr(dual_cfg, 'secondary_train_freq_hz', 40.0),
                    getattr(dual_cfg, 'secondary_train_duration_ms', 200.0),
                    getattr(dual_cfg, 'secondary_start', 0.0),
                    getattr(dual_cfg, 'secondary_train_type', 'none'),
                )
                event_times_arr_2 = generate_effective_event_times(
                    getattr(dual_cfg, 'secondary_train_type', 'none'),
                    getattr(dual_cfg, 'secondary_train_freq_hz', 40.0),
                    getattr(dual_cfg, 'secondary_train_duration_ms', 200.0),
                    dual_cfg.secondary_start,
                    dual_cfg.secondary_event_times,
                    seed_hash=seed_hash_2
                )

        # ── Reconstruct state offsets (must match rhs.py exactly) ──
        cc = cfg.channels
        offsets = build_state_offsets(
            n_comp,
            en_ih=cc.enable_Ih,
            en_ica=cc.enable_ICa,
            en_ia=cc.enable_IA,
            en_sk=cc.enable_SK,
            dyn_ca=cfg.calcium.dynamic_Ca,
            en_itca=cc.enable_ITCa,
            en_im=cc.enable_IM,
            en_nap=cc.enable_NaP,
            en_nar=cc.enable_NaR,
            dyn_atp=cfg.metabolism.enable_dynamic_atp,
            use_dfilter_primary=use_dfilter_primary,
            use_dfilter_secondary=use_dfilter_secondary,
        )
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
        off_vfilt_primary = int(offsets.off_ifilt_primary)
        off_vfilt_secondary = int(offsets.off_ifilt_secondary)

        phi_na  = cfg.env.build_phi_vector(cfg.env.Q10_Na,  n_comp)
        phi_k   = cfg.env.build_phi_vector(cfg.env.Q10_K,   n_comp)
        phi_ih  = cfg.env.build_phi_vector(cfg.env.Q10_Ih,  n_comp)
        phi_ca  = cfg.env.build_phi_vector(cfg.env.Q10_Ca,  n_comp)
        phi_nap = cfg.env.build_phi_vector(cfg.env.Q10_NaP, n_comp)
        phi_nar = cfg.env.build_phi_vector(cfg.env.Q10_NaR, n_comp)

        # ── Laplacian diagonal ──
        import scipy.sparse
        l_sparse = scipy.sparse.csr_matrix(
            (morph["L_data"], morph["L_indices"], morph["L_indptr"]),
            shape=(n_comp, n_comp),
        )
        l_diag = np.array(l_sparse.diagonal(), dtype=np.float64)  # all < 0

        b_ca_v = self._build_b_ca_vector(cfg, morph)

        # ── Fixed dt for native loop ──
        dt_eval_f    = float(cfg.stim.dt_eval)
        dt_internal  = min(0.025, dt_eval_f / 4.0)   # sub-step for accuracy

        # ── Pack gbar_mat: rows = [gNa, gK, gL, gIh, gCa, gA, gSK, gTCa, gIM, gNaP, gNaR] ──
        gbar_mat = np.array([
            morph["gNa_v"], morph["gK_v"],  morph["gL_v"],
            morph["gIh_v"], morph["gCa_v"], morph["gA_v"],
            morph["gSK_v"], morph["gTCa_v"],morph["gIM_v"],
            morph["gNaP_v"],morph["gNaR_v"],
        ], dtype=np.float64)   # shape (11, n_comp)

        # ── Pack phi_mat: rows = [na, k, ih, ca, ia, tca, im, nap, nar] ──
        # IA and IM follow K-channel Q10; T-type Ca follows Ca Q10;
        # persistent/resurgent Na follow Na Q10.
        phi_mat = np.array([
            phi_na,  phi_k,   phi_ih,
            phi_ca,
            phi_k,   # phi_ia  — IA uses K-family Q10 (Sah & Bhatt 1999)
            phi_ca,  # phi_tca — T-type Ca same Q10 as L-type Ca
            phi_k,   # phi_im  — M-type uses K-family Q10 (Yamada 1989)
            phi_nap, phi_nar,
        ], dtype=np.float64)   # shape (9, n_comp)

        # ── Stochastic parameters ──
        stoch_gating = getattr(cfg.stim, 'stoch_gating', False)
        noise_sigma = getattr(cfg.stim, 'noise_sigma', 0.0)
        gna_max = cfg.channels.gNa_max
        gk_max = cfg.channels.gK_max
        
        # ── Initialize RNG state for reproducibility ──
        from core.stochastic_rng import get_rng
        rng = get_rng()
        rng_state = rng.get_state()['state'] if (stoch_gating or noise_sigma > 0) else None

        # Generate ephemeral primary train
        seed_hash = _stable_seed_from_values(
            cfg.stim.synaptic_train_freq_hz,
            cfg.stim.synaptic_train_duration_ms,
            cfg.stim.pulse_start,
            cfg.stim.synaptic_train_type,
        )
        event_times_arr = generate_effective_event_times(
            cfg.stim.synaptic_train_type, cfg.stim.synaptic_train_freq_hz,
            cfg.stim.synaptic_train_duration_ms, cfg.stim.pulse_start, cfg.stim.event_times, seed_hash=seed_hash
        )
        # CRITICAL: Sort event times for early termination optimization in get_event_driven_conductance
        # The break-on-future-event optimization requires strictly ascending order
        event_times_arr = np.sort(event_times_arr)
        
        # Generate ephemeral secondary train
        event_times_arr_2 = np.zeros(0, dtype=np.float64)
        if dual_stim_enabled == 1 and dual_cfg is not None:
            seed_hash_2 = _stable_seed_from_values(
                getattr(dual_cfg, 'secondary_train_freq_hz', 40.0),
                getattr(dual_cfg, 'secondary_train_duration_ms', 200.0),
                getattr(dual_cfg, 'secondary_start', 0.0),
                getattr(dual_cfg, 'secondary_train_type', 'none'),
            )
            event_times_arr_2 = generate_effective_event_times(
                getattr(dual_cfg, 'secondary_train_type', 'none'),
                getattr(dual_cfg, 'secondary_train_freq_hz', 40.0),
                getattr(dual_cfg, 'secondary_train_duration_ms', 200.0),
                dual_cfg.secondary_start,
                dual_cfg.secondary_event_times,
                seed_hash=seed_hash_2
            )
            # Sort secondary events as well
            event_times_arr_2 = np.sort(event_times_arr_2)

        na_i_rest_mM, k_o_rest_mM = _resolve_dynamic_atp_rest_values(cfg)

        # Precompute ZAP Tukey windows (primary & secondary)
        _zw1_t, _zw1_g, _zw1_n = _precompute_zap_window(td, zap_rise) if stype == 10 else (np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64), 0)
        _zw2_t, _zw2_g, _zw2_n = _precompute_zap_window(td_2, zap_rise_2) if stype_2 == 10 else (np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64), 0)

        physics = create_physics_params(
            n_comp              = np.int32(n_comp),
            en_ih               = bool(cc.enable_Ih),
            en_ica              = bool(cc.enable_ICa),
            en_ia               = bool(cc.enable_IA),
            en_sk               = bool(cc.enable_SK),
            dyn_ca              = bool(cfg.calcium.dynamic_Ca),
            en_itca             = bool(cc.enable_ITCa),
            en_im               = bool(cc.enable_IM),
            en_nap              = bool(cc.enable_NaP),
            en_nar              = bool(cc.enable_NaR),
            dyn_atp             = bool(cfg.metabolism.enable_dynamic_atp),
            gbar_mat            = gbar_mat,
            ena                 = float(cc.ENa),
            ek                  = float(cc.EK),
            el                  = float(cc.EL),
            eih                 = float(cc.E_Ih),
            ea                  = float(cc.EK),  # A-current uses K reversal potential
            e_rev_syn_primary   = float(cc.e_rev_syn_primary),
            e_rev_syn_secondary = float(cc.e_rev_syn_secondary),
            cm_v                = morph["Cm_v"].astype(np.float64),
            l_data              = morph["L_data"].astype(np.float64),
            l_indices           = morph["L_indices"].astype(np.int32),
            l_indptr            = morph["L_indptr"].astype(np.int32),
            phi_mat             = phi_mat,
            t_kelvin            = float(t_kelvin),
            ca_ext              = float(cfg.calcium.Ca_ext),
            ca_rest             = float(cfg.calcium.Ca_rest),
            tau_ca              = float(cfg.calcium.tau_Ca),
            b_ca                = b_ca_v,
            mg_ext              = float(getattr(cfg.env, "Mg_ext", 1.0)),
            nmda_mg_block_mM    = float(getattr(cfg.env, "nmda_mg_block_mM", 3.57)),
            tau_sk              = float(getattr(cc, "tau_SK", 15.0)),
            im_speed_multiplier = float(getattr(cc, "im_speed_multiplier", 1.0)),
            g_katp_max          = float(cfg.metabolism.g_katp_max),
            katp_kd_atp_mM      = float(cfg.metabolism.katp_kd_atp_mM),
            atp_max_mM          = float(cfg.metabolism.atp_max_mM),
            atp_synthesis_rate  = float(cfg.metabolism.atp_synthesis_rate),
            na_i_rest_mM        = float(na_i_rest_mM),
            na_ext_mM           = float(cfg.metabolism.na_ext_mM),
            k_i_mM              = float(cfg.metabolism.k_i_mM),
            k_o_rest_mM         = float(k_o_rest_mM),
            ion_drift_gain      = float(cfg.metabolism.ion_drift_gain),
            k_o_clearance_tau_ms = float(cfg.metabolism.k_o_clearance_tau_ms),
            stype               = np.int32(stype),
            iext                = float(iext),
            t0                  = float(t0),
            td                  = float(td),
            atau                = float(atau),
            zap_f0_hz           = float(zap_f0),
            zap_f1_hz           = float(zap_f1),
            zap_rise_ms         = float(zap_rise),
            zap_win_t           = _zw1_t,
            zap_win_g           = _zw1_g,
            zap_win_size        = np.int32(_zw1_n),
            event_times_arr     = event_times_arr,
            n_events            = np.int32(len(event_times_arr)),
            event_times_arr_2   = event_times_arr_2,
            n_events_2          = np.int32(len(event_times_arr_2)),
            stim_comp           = np.int32(stim_comp),
            stim_mode           = np.int32(stim_mode),
            use_dfilter_primary   = np.int32(use_dfilter_primary),
            dfilter_distance_um   = float(dfilter_distance_um),
            dfilter_lambda_um     = float(dfilter_lambda_um),
            dfilter_tau_ms        = float(dfilter_tau_ms),
            dfilter_input_freq_hz = float(dfilter_input_freq_hz),
            dfilter_filter_mode   = np.int32(dfilter_filter_mode),
            dfilter_attenuation   = float(dfilter_attenuation),
            dual_stim_enabled   = np.int32(dual_stim_enabled),
            stype_2             = np.int32(stype_2),
            iext_2              = float(iext_2),
            t0_2                = float(t0_2),
            td_2                = float(td_2),
            atau_2              = float(atau_2),
            zap_f0_hz_2         = float(zap_f0_2),
            zap_f1_hz_2         = float(zap_f1_2),
            zap_rise_ms_2       = float(zap_rise_2),
            zap_win_t_2         = _zw2_t,
            zap_win_g_2         = _zw2_g,
            zap_win_size_2      = np.int32(_zw2_n),
            stim_comp_2         = np.int32(stim_comp_2),
            stim_mode_2         = np.int32(stim_mode_2),
            use_dfilter_secondary     = np.int32(use_dfilter_secondary),
            dfilter_distance_um_2     = float(dfilter_distance_um_2),
            dfilter_lambda_um_2       = float(dfilter_lambda_um_2),
            dfilter_tau_ms_2          = float(dfilter_tau_ms_2),
            dfilter_input_freq_hz_2   = float(dfilter_input_freq_hz_2),
            dfilter_filter_mode_2     = np.int32(dfilter_filter_mode_2),
            dfilter_attenuation_2     = float(dfilter_attenuation_2),
            stoch_gating       = stoch_gating,
            noise_sigma       = noise_sigma,
            gna_max           = gna_max,
            gk_max            = gk_max,
            rng_state         = None,
        )

        if stoch_gating or noise_sigma > 0:
            seed = _resolve_stochastic_seed(cfg, noise_sigma, stoch_gating)
            np.random.seed(seed)
            set_numba_random_seed(seed)

        # Convert string mode to int for Numba compatibility
        # 0=v_only, 1=v_and_gates, 2=full_state, 3=custom
        if isinstance(lle_subspace_mode, str):
            lle_subspace_mode_int = _LLE_MODE_MAP.get(lle_subspace_mode, 0)
        else:
            lle_subspace_mode_int = int(lle_subspace_mode)

        t_out, y_out, diverged, lle_arr = run_native_loop(
            y0.astype(np.float64),
            float(cfg.stim.t_sim),
            dt_internal,
            dt_eval_f,
            physics,
            l_diag,
            morph["parent_idx"].astype(np.int32),
            morph["hines_order"].astype(np.int32),
            morph["g_axial_to_parent"].astype(np.float64),
            morph["g_axial_parent_to_child"].astype(np.float64),
            calc_lle,
            lle_delta,
            lle_t_evolve,
            lle_subspace_mode_int,
            lle_custom_mask,
            lle_weights,
        )

        if diverged:
            # Return partial result with warning flag
            res = SimulationResult(t_out, y_out, n_comp, cfg)
            self._post_process_physics(res, morph)
            res.morph = morph
            if calc_lle:
                res.lle_convergence = lle_arr
            res.diverged = True
            return res

        res = SimulationResult(t_out, y_out, n_comp, cfg)
        self._post_process_physics(res, morph)
        res.morph = morph
        if calc_lle:
            res.lle_convergence = lle_arr
        return res

    # ─────────────────────────────────────────────────────────────────
    def _run_with_backend(self, cfg) -> SimulationResult:
        """Execute single run using appropriate backend based on configuration.
        
        If jacobian_mode == 'native_hines', uses fast native Hines solver.
        Falls back to SciPy BDF if native solver fails (e.g., Numba unavailable).
        
        Parameters
        ----------
        cfg : FullModelConfig
            Configuration for this run
            
        Returns
        -------
        SimulationResult
            Simulation output from chosen backend
        """
        if cfg.stim.jacobian_mode == 'native_hines':
            try:
                return self.run_native(cfg)
            except (ImportError, RuntimeError, Exception) as e:
                # Graceful fallback to SciPy if native solver unavailable/fails
                import warnings
                warnings.warn(f"Native Hines solver failed ({e}), falling back to SciPy BDF")
                return self.run_single(cfg)
        else:
            return self.run_single(cfg)

    def run_mc(self, n_trials: int = 10, param_var: float = 0.05,
                progress_cb=None) -> list[SimulationResult]:
        """
        Monte-Carlo parameter sweep with stochastic reproducibility.
        
        Uses native_hines backend when available for 10-100× speedup on multi-run analysis.
        
        Parameters
        ----------
        n_trials : int
            Number of Monte-Carlo trials
        param_var : float
            Parameter variation (±%) for gNa/gK randomization
        progress_cb : callable, optional
            Progress callback function
            
        Returns
        -------
        list[SimulationResult]
            Results from all Monte-Carlo trials
        """
        from core.stochastic_rng import get_rng
        
        configs  = []
        rng = get_rng()  # Use centralized RNG
        
        for _ in range(n_trials):
            m_cfg = copy.deepcopy(self.config)
            # Use centralized RNG for parameter variations
            m_cfg.channels.gNa_max *= rng.normal(1.0, param_var)
            m_cfg.channels.gK_max  *= rng.normal(1.0, param_var)
            configs.append(m_cfg)

        with ThreadPoolExecutor(max_workers=min(n_trials, max(1, os.cpu_count() or 1))) as executor:
            futures = []
            for i, cfg in enumerate(configs):
                if progress_cb:
                    progress_cb(i, n_trials, None)
                # Use backend-agnostic execution for speed
                futures.append(executor.submit(NeuronSolver(cfg)._run_with_backend, cfg))
            
            results = []
            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    results.append(result)
                    if progress_cb:
                        progress_cb(i, n_trials, result)
                except Exception as e:
                    print(f"Monte-Carlo trial {i} failed: {e}")
                    results.append(None)
            
            return results

    # ─────────────────────────────────────────────────────────────────
    def run_bifurcation(self) -> list:
        """Bifurcation diagram: V_min/V_max vs parameter in steady state.
        
        Uses native_hines backend when available for 10-100× speedup.
        """
        ana = self.config.analysis
        param_values = np.linspace(ana.bif_min, ana.bif_max, ana.bif_steps)
        results = []

        for val in param_values:
            tmp = copy.deepcopy(self.config)
            _set_nested_attr(tmp, ana.bif_param, val)
            tmp.stim.t_sim = 300.0
            # Use backend-agnostic execution for speed
            res = self._run_with_backend(tmp)
            half = len(res.t) // 2
            v_ss = res.v_soma[half:]

            # Collect actual spike peaks for Poincaré plot
            from core.analysis import detect_spikes
            pks, sp_t, sp_amp = detect_spikes(v_ss, res.t[half:])

            results.append({
                'val':    val,
                'min':    float(np.min(v_ss)),
                'max':    float(np.max(v_ss)),
                'peaks':  sp_amp.tolist(),
                'freq':   float(1000.0 / np.mean(np.diff(sp_t))) if len(sp_t) > 1 else 0.0,
                'n_sp':   len(pks),
            })
        return results

    # (ghost run_native removed — authoritative version at line 615)


def worker_task(config: FullModelConfig) -> SimulationResult:
    """Picklable worker for ProcessPoolExecutor."""
    return NeuronSolver(config).run_single()
