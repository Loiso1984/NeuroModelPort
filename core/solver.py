import numpy as np
import copy
import logging
from scipy.integrate import solve_ivp
from concurrent.futures import ProcessPoolExecutor

from core.models import FullModelConfig
from core.morphology import MorphologyBuilder
from core.channels import ChannelRegistry
from core.jacobian import analytic_sparse_jacobian, build_jacobian_sparsity, make_analytic_jacobian
from core.rhs import rhs_multicompartment, _get_syn_reversal, F_CONST, R_GAS
from core.physics_params import create_physics_params
from core.kinetics import z_inf_SK
from core.validation import estimate_simulation_runtime, validate_simulation_config


logger = logging.getLogger(__name__)


class SimulationResult:
    """Complete scientific dataset after a simulation run."""

    def __init__(self, t, y, n_comp, config: FullModelConfig):
        self.t      = t
        self.y      = y
        self.config = config
        self.n_comp = n_comp
        self.diverged = False  # v12.0: Flag for simulation divergence due to non-physical parameters

        self.v_all  = y[0:n_comp, :]
        self.v_soma = self.v_all[0, :]

        dual_cfg = getattr(config, "dual_stimulation", None)
        dual_enabled = bool(dual_cfg is not None and getattr(dual_cfg, "enabled", False))

        # Primary stimulus always comes from cfg.stim_location and cfg.stim
        primary_location = config.stim_location.location
        has_primary_dfilter_state = (
            primary_location == "dendritic_filtered"
            and config.dendritic_filter.enabled
            and config.dendritic_filter.tau_dendritic_ms > 0.0
        )
        has_secondary_dfilter_state = (
            dual_enabled
            and getattr(dual_cfg, "secondary_location", "soma") == "dendritic_filtered"
            and getattr(dual_cfg, "secondary_tau_dendritic_ms", 0.0) > 0.0
        )
        filter_state_count = int(has_primary_dfilter_state) + int(has_secondary_dfilter_state)

        self.ca_i = None
        if config.calcium.dynamic_Ca:
            if filter_state_count > 0:
                self.ca_i = y[-(n_comp + filter_state_count):-filter_state_count, :]
            else:
                self.ca_i = y[-n_comp:, :]

        self.v_dendritic_filtered = None
        self.v_dendritic_filtered_secondary = None
        if filter_state_count > 0:
            tail_start = y.shape[0] - filter_state_count
            cursor = tail_start
            if has_primary_dfilter_state:
                self.v_dendritic_filtered = y[cursor, :]
                cursor += 1
            if has_secondary_dfilter_state:
                self.v_dendritic_filtered_secondary = y[cursor, :]

        self.currents:    dict  = {}
        self.atp_estimate: float = 0.0
        self.atp_breakdown: dict = {}  # {Na_pump, Ca_pump, baseline, total}

        # Morphology dict stored for post-analysis (current balance, etc.)
        self.morph: dict = {}


def generate_effective_event_times(train_type: str, freq_hz: float, duration_ms: float, t_start: float, manual_times: list, seed_hash: int = None) -> np.ndarray:
    """Generates an ephemeral array of event times without mutating the base config.
    
    Args:
        train_type: Type of train ('none', 'regular', 'poisson')
        freq_hz: Frequency in Hz for auto-generated trains
        duration_ms: Duration in ms for auto-generated trains
        t_start: Start time in ms
        manual_times: Manual event times list
        seed_hash: Optional seed hash for deterministic Poisson generation (for previewer stability)
        
    Returns:
        Array of event times
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
        """Run a single deterministic simulation.
        
        Dispatches to run_native() when jacobian_mode='native_hines',
        otherwise uses SciPy BDF integrator.
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

        y0 = self.registry.compute_initial_states(-65.0, cfg)

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

        dfilter_attenuation = 1.0
        if stim_mode == 2 and cfg.dendritic_filter.space_constant_um > 0:
            dfilter_attenuation = np.exp(
                -cfg.dendritic_filter.distance_um / cfg.dendritic_filter.space_constant_um
            )
        dfilter_tau_ms = cfg.dendritic_filter.tau_dendritic_ms

        # Secondary stimulus defaults.
        # Keep secondary defaults physically valid even when dual stimulation is disabled.
        # This protects against stricter external validators and stale serialized configs.
        stype_2, iext_2, t0_2, td_2, atau_2 = 0, 0.0, 0.0, 0.0, 1.0
        zap_f0_2, zap_f1_2 = primary_zap_f0, primary_zap_f1
        stim_comp_2, stim_mode_2 = 0, 0
        use_dfilter_secondary = 0
        dfilter_attenuation_2, dfilter_tau_ms_2 = 1.0, 0.0

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
            dfilter_tau_ms_2 = dual_cfg.secondary_tau_dendritic_ms
            use_dfilter_secondary = int(stim_mode_2 == 2 and dfilter_tau_ms_2 > 0.0)
            if use_dfilter_secondary == 1:
                y0 = np.concatenate([y0, np.array([0.0])])
            if stim_mode_2 == 2 and dual_cfg.secondary_space_constant_um > 0:
                dfilter_attenuation_2 = np.exp(
                    -dual_cfg.secondary_distance_um / dual_cfg.secondary_space_constant_um
                )

        # Generate ephemeral primary train
        # Create a stable seed based on the parameters
        seed_hash = hash((cfg.stim.synaptic_train_freq_hz, cfg.stim.synaptic_train_duration_ms, cfg.stim.pulse_start)) % (2**32 - 1)
        eff_event_times_1 = generate_effective_event_times(
            cfg.stim.synaptic_train_type, cfg.stim.synaptic_train_freq_hz,
            cfg.stim.synaptic_train_duration_ms, cfg.stim.pulse_start, cfg.stim.event_times, seed_hash=seed_hash
        )
        
        # Generate ephemeral secondary train
        eff_event_times_2 = np.zeros(0, dtype=np.float64)
        if dual_stim_enabled == 1 and dual_cfg is not None:
            seed_hash_2 = hash((getattr(dual_cfg, 'secondary_train_freq_hz', 40.0),
                               getattr(dual_cfg, 'secondary_train_duration_ms', 200.0),
                               getattr(dual_cfg, 'secondary_start', 0.0))) % (2**32 - 1)
            eff_event_times_2 = generate_effective_event_times(
                getattr(dual_cfg, 'secondary_train_type', 'none'),
                getattr(dual_cfg, 'secondary_train_freq_hz', 40.0),
                getattr(dual_cfg, 'secondary_train_duration_ms', 200.0),
                dual_cfg.secondary_start,
                dual_cfg.secondary_event_times,
                seed_hash=seed_hash_2
            )

        # ── Initialize RNG state for reproducibility ──
        from core.stochastic_rng import get_rng
        rng = get_rng()
        rng_state = rng.get_state()['state'] if (cfg.stim.stoch_gating or cfg.stim.noise_sigma > 0) else None

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
            "tau_sk": cfg.channels.tau_SK,
            "b_ca": self._build_b_ca_vector(cfg, morph),
            "g_katp_max": cfg.metabolism.g_katp_max,
            "katp_kd_atp_mM": cfg.metabolism.katp_kd_atp_mM,
            "atp_max_mM": cfg.metabolism.atp_max_mM,
            "atp_synthesis_rate": cfg.metabolism.atp_synthesis_rate,
            "stype": stype,
            "iext": primary_iext,
            "t0": primary_t0,
            "td": primary_td,
            "atau": primary_atau,
            "zap_f0_hz": primary_zap_f0,
            "zap_f1_hz": primary_zap_f1,
            "stim_comp": primary_stim_comp,
            "stim_mode": stim_mode,
            "use_dfilter_primary": use_dfilter_primary,
            "dfilter_attenuation": dfilter_attenuation,
            "dfilter_tau_ms": dfilter_tau_ms,
            "event_times_arr": eff_event_times_1,
            "n_events": int(len(eff_event_times_1)),
            "event_times_arr_2": eff_event_times_2,
            "n_events_2": int(len(eff_event_times_2)),
            "dual_stim_enabled": dual_stim_enabled,
            "gna_max": cfg.channels.gNa_max,
            "gk_max": cfg.channels.gK_max,
            "e_rev_syn_primary": getattr(cfg.channels, 'e_rev_syn_primary', 0.0),
            "e_rev_syn_secondary": getattr(cfg.channels, 'e_rev_syn_secondary', -75.0),
            "stype_2": stype_2,
            "iext_2": iext_2,
            "t0_2": t0_2,
            "td_2": td_2,
            "atau_2": atau_2,
            "zap_f0_hz_2": zap_f0_2,
            "zap_f1_hz_2": zap_f1_2,
            "stim_comp_2": stim_comp_2,
            "stim_mode_2": stim_mode_2,
            "use_dfilter_secondary": use_dfilter_secondary,
            "dfilter_attenuation_2": dfilter_attenuation_2,
            "dfilter_tau_ms_2": dfilter_tau_ms_2,
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
        """Reconstruct ion-channel current densities and ATP estimate for ALL compartments."""
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
        
        # Broadcast conductances to shape (n_comp, 1) for 2D current calculation
        g_na = morph['gNa_v'][:, np.newaxis]
        g_k  = morph['gK_v'][:, np.newaxis]
        g_l  = morph['gL_v'][:, np.newaxis]
        
        res.currents['Na']   = g_na * (m ** 3) * h * (v - cfg.channels.ENa)
        res.currents['K']    = g_k  * (nk ** 4) * (v - cfg.channels.EK)
        res.currents['Leak'] = g_l  * (v - cfg.channels.EL)

        cursor = 4 * n

        if cfg.channels.enable_Ih:
            if 'gIh_v' in morph:
                r = y[cursor:cursor + n, :]
                g_ih = morph['gIh_v'][:, np.newaxis]
                res.currents['Ih'] = g_ih * r * (v - cfg.channels.E_Ih)
                cursor += n
            else:
                logger.warning("Ih channel enabled but gIh_v missing from morph")

        if cfg.channels.enable_ICa:
            if 'gCa_v' in morph:
                s = y[cursor:cursor + n, :]
                u = y[cursor + n:cursor + 2*n, :]
                g_ca = morph['gCa_v'][:, np.newaxis]
                if cfg.calcium.dynamic_Ca and res.ca_i is not None:
                    t_kelvin = cfg.env.T_celsius + 273.15
                    ca_i = np.maximum(res.ca_i, 1e-9)
                    e_ca = (R_GAS * t_kelvin / (2.0 * F_CONST)) * np.log(cfg.calcium.Ca_ext / ca_i) * 1000.0
                else:
                    e_ca = 120.0
                res.currents['ICa'] = g_ca * (s ** 2) * u * (v - e_ca)
                cursor += 2 * n
            else:
                logger.warning("ICa channel enabled but gCa_v missing from morph")

        if cfg.channels.enable_IA:
            if 'gA_v' in morph:
                a = y[cursor:cursor + n, :]
                b = y[cursor + n:cursor + 2*n, :]
                g_a = morph['gA_v'][:, np.newaxis]
                res.currents['IA'] = g_a * a * b * (v - cfg.channels.EK)  # A-current uses EK
                cursor += 2 * n
            else:
                logger.warning("IA channel enabled but gA_v missing from morph")

        if cfg.channels.enable_ITCa:
            if 'gTCa_v' in morph:
                p_t = y[cursor:cursor + n, :]
                q_t = y[cursor + n:cursor + 2*n, :]
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
                cursor += 2 * n
            else:
                logger.warning("ITCa channel enabled but gTCa_v missing from morph")

        if cfg.channels.enable_IM:
            if 'gIM_v' in morph:
                w_m = y[cursor:cursor + n, :]
                g_im = morph['gIM_v'][:, np.newaxis]
                res.currents['IM'] = g_im * w_m * (v - cfg.channels.EK)
                cursor += n
            else:
                logger.warning("IM channel enabled but gIM_v missing from morph")

        if cfg.channels.enable_NaP:
            if 'gNaP_v' in morph:
                x_p = y[cursor:cursor + n, :]
                g_nap = morph['gNaP_v'][:, np.newaxis]
                res.currents['NaP'] = g_nap * x_p * (v - cfg.channels.ENa)
                cursor += n
            else:
                logger.warning("NaP channel enabled but gNaP_v missing from morph")

        if cfg.channels.enable_NaR:
            if 'gNaR_v' in morph:
                y_r = y[cursor:cursor + n, :]
                j_r = y[cursor + n:cursor + 2*n, :]
                g_nar = morph['gNaR_v'][:, np.newaxis]
                res.currents['NaR'] = g_nar * y_r * j_r * (v - cfg.channels.ENa)
                cursor += 2 * n
            else:
                logger.warning("NaR channel enabled but gNaR_v missing from morph")

        if cfg.channels.enable_SK:
            if 'gSK_v' in morph:
                z_sk = y[cursor:cursor + n, :]
                g_sk = morph['gSK_v'][:, np.newaxis]
                res.currents['SK'] = g_sk * z_sk * (v - cfg.channels.EK)
                cursor += n
            else:
                logger.warning("SK channel enabled but gSK_v missing from morph")

        # Extract K_ATP current if metabolism is enabled
        if cfg.metabolism.enable_dynamic_atp:
            # ATP state is at the end of the state vector
            atp_state = y[-n:, :]
            atp_ratio = atp_state / cfg.metabolism.katp_kd_atp_mM
            g_katp = cfg.metabolism.g_katp_max / (1.0 + atp_ratio ** 2)
            res.currents['KATP'] = g_katp * (v - cfg.channels.EK)

        # ── ATP consumption estimate ──────────────────────────────────
        # Na+/K+-ATPase: 3 Na+ pumped per ATP hydrolysis (Skou 1957).
        # Ca²+-ATPase (PMCA): 1 Ca²+ pumped per ATP (Bhatt et al. 2005).
        # Baseline pump: resting Na+/K+-ATPase ≈ 50% of neuronal ATP even
        # without spiking (Attwell & Laughlin 2001, J Cereb Blood Flow Metab).
        dt = cfg.stim.dt_eval
        F = 96485.33  # Faraday constant [C/mol]

        # 1. Na+ pump cost: |Q_Na| / (3·F)  [µC/cm²·ms → mol → nmol ATP/cm²]
        q_na = float(np.sum(np.abs(res.currents['Na']))) * dt
        if 'NaP' in res.currents:
            q_na += float(np.sum(np.abs(res.currents['NaP']))) * dt
        if 'NaR' in res.currents:
            q_na += float(np.sum(np.abs(res.currents['NaR']))) * dt
        atp_na = (q_na * 1e-9) / (3.0 * F) * 1e9  # nmol/cm²

        # 2. Ca²+ pump cost: |Q_Ca| / (1·2F)  [divalent ion, z=2]
        atp_ca = 0.0
        for ca_key in ('ICa', 'ITCa'):
            if ca_key in res.currents:
                q_ca = float(np.sum(np.abs(res.currents[ca_key]))) * dt
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
    def run_native(self, custom_config: "FullModelConfig | None" = None) -> SimulationResult:
        """Run simulation with the native Hines solver (v11.0).

        Uses a fixed-step Backward-Euler integrator with the O(N) Hines
        tree solver instead of SciPy BDF.  Requires jacobian_mode='native_hines'.
        Returns a standard SimulationResult compatible with the GUI and analytics.
        """
        from core.native_loop import run_native_loop

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
        use_dfilter_primary = int(
            stim_mode == 2
            and cfg.dendritic_filter.enabled
            and cfg.dendritic_filter.tau_dendritic_ms > 0.0
        )
        
        stype = s_map.get(cfg.stim.stim_type, 0)
        iext = cfg.stim.Iext
        t0 = cfg.stim.pulse_start
        td = cfg.stim.pulse_dur
        atau = cfg.stim.alpha_tau
        stim_mode = stim_mode_map.get(cfg.stim_location.location, 0)
        stim_comp = cfg.stim.stim_comp
        zap_f0 = cfg.stim.zap_f0_hz
        zap_f1 = cfg.stim.zap_f1_hz
        use_dfilter_primary = int(
            stim_mode == 2
            and cfg.dendritic_filter.enabled
            and cfg.dendritic_filter.tau_dendritic_ms > 0.0
        )
        if use_dfilter_primary == 1:
            y0 = np.concatenate([y0, np.array([0.0])])
        dfilter_attenuation = 1.0
        if stim_mode == 2 and cfg.dendritic_filter.space_constant_um > 0:
            dfilter_attenuation = np.exp(
                -cfg.dendritic_filter.distance_um / cfg.dendritic_filter.space_constant_um
            )
        dfilter_tau_ms = cfg.dendritic_filter.tau_dendritic_ms

        stype_2, iext_2, t0_2, td_2, atau_2 = 0, 0.0, 0.0, 0.0, 1.0
        zap_f0_2, zap_f1_2 = zap_f0, zap_f1
        stim_comp_2, stim_mode_2 = 0, 0
        use_dfilter_secondary = 0
        dfilter_attenuation_2, dfilter_tau_ms_2 = 1.0, 0.0
        if dual_stim_enabled == 1:
            stype_2 = s_map.get(dual_cfg.secondary_stim_type, 0)
            iext_2 = dual_cfg.secondary_Iext
            t0_2 = dual_cfg.secondary_start
            td_2 = dual_cfg.secondary_duration
            atau_2 = dual_cfg.secondary_alpha_tau
            stim_mode_2 = stim_mode_map.get(dual_cfg.secondary_location, 0)
            zap_f0_2 = getattr(dual_cfg, "secondary_zap_f0_hz", zap_f0_2)
            zap_f1_2 = getattr(dual_cfg, "secondary_zap_f1_hz", zap_f1_2)
            dfilter_tau_ms_2 = dual_cfg.secondary_tau_dendritic_ms
            use_dfilter_secondary = int(stim_mode_2 == 2 and dfilter_tau_ms_2 > 0.0)
            if use_dfilter_secondary == 1:
                y0 = np.concatenate([y0, np.array([0.0])])
            if stim_mode_2 == 2 and dual_cfg.secondary_space_constant_um > 0:
                dfilter_attenuation_2 = np.exp(
                    -dual_cfg.secondary_distance_um / dual_cfg.secondary_space_constant_um
                )
            # Generate event times for secondary stimulus (synaptic train)
            event_times_arr_2 = np.zeros(0, dtype=np.float64)
            if stype_2 >= 4:  # Conductance-based synapse
                seed_hash_2 = hash((getattr(dual_cfg, 'secondary_train_freq_hz', 40.0),
                                   getattr(dual_cfg, 'secondary_train_duration_ms', 200.0),
                                   getattr(dual_cfg, 'secondary_start', 0.0))) % (2**32 - 1)
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
        cursor = 4 * n_comp  # V, m, h, n_K are always first 4*n_comp
        off_m, off_h, off_n = n_comp, 2 * n_comp, 3 * n_comp

        off_r = cursor
        if cc.enable_Ih:
            cursor += n_comp
        off_s = off_u = cursor
        if cc.enable_ICa:
            off_s = cursor;  cursor += n_comp
            off_u = cursor;  cursor += n_comp
        off_a = off_b = cursor
        if cc.enable_IA:
            off_a = cursor;  cursor += n_comp
            off_b = cursor;  cursor += n_comp
        off_p = off_q = cursor
        if cc.enable_ITCa:
            off_p = cursor;  cursor += n_comp
            off_q = cursor;  cursor += n_comp
        off_w = cursor
        if cc.enable_IM:
            cursor += n_comp
        off_x = cursor
        if cc.enable_NaP:
            cursor += n_comp
        off_y = off_j = cursor
        if cc.enable_NaR:
            off_y = cursor;  cursor += n_comp
            off_j = cursor;  cursor += n_comp
        off_zsk = cursor
        if cc.enable_SK:
            cursor += n_comp
        off_ca = cursor
        if cfg.calcium.dynamic_Ca:
            cursor += n_comp
        off_vfilt_primary = cursor
        if use_dfilter_primary == 1:
            cursor += 1
        off_vfilt_secondary = cursor

        # ── Temperature scaling ──
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
        seed_hash = hash((cfg.stim.synaptic_train_freq_hz, cfg.stim.synaptic_train_duration_ms, cfg.stim.pulse_start)) % (2**32 - 1)
        event_times_arr = generate_effective_event_times(
            cfg.stim.synaptic_train_type, cfg.stim.synaptic_train_freq_hz,
            cfg.stim.synaptic_train_duration_ms, cfg.stim.pulse_start, cfg.stim.event_times, seed_hash=seed_hash
        )
        
        # Generate ephemeral secondary train
        event_times_arr_2 = np.zeros(0, dtype=np.float64)
        if dual_stim_enabled == 1 and dual_cfg is not None:
            seed_hash_2 = hash((getattr(dual_cfg, 'secondary_train_freq_hz', 40.0),
                               getattr(dual_cfg, 'secondary_train_duration_ms', 200.0),
                               getattr(dual_cfg, 'secondary_start', 0.0))) % (2**32 - 1)
            event_times_arr_2 = generate_effective_event_times(
                getattr(dual_cfg, 'secondary_train_type', 'none'),
                getattr(dual_cfg, 'secondary_train_freq_hz', 40.0),
                getattr(dual_cfg, 'secondary_train_duration_ms', 200.0),
                dual_cfg.secondary_start,
                dual_cfg.secondary_event_times,
                seed_hash=seed_hash_2
            )

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
            e_rev_syn_primary   = float(getattr(cc, 'e_rev_syn_primary', 0.0)),
            e_rev_syn_secondary = float(getattr(cc, 'e_rev_syn_secondary', -75.0)),
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
            tau_sk              = float(getattr(cc, "tau_SK", 15.0)),
            g_katp_max          = float(cfg.metabolism.g_katp_max),
            katp_kd_atp_mM      = float(cfg.metabolism.katp_kd_atp_mM),
            atp_max_mM          = float(cfg.metabolism.atp_max_mM),
            atp_synthesis_rate  = float(cfg.metabolism.atp_synthesis_rate),
            stype               = np.int32(stype),
            iext                = float(iext),
            t0                  = float(t0),
            td                  = float(td),
            atau                = float(atau),
            zap_f0_hz           = float(zap_f0),
            zap_f1_hz           = float(zap_f1),
            event_times_arr     = event_times_arr,
            n_events            = np.int32(len(event_times_arr)),
            event_times_arr_2   = event_times_arr_2,
            n_events_2          = np.int32(len(event_times_arr_2)),
            stim_comp           = np.int32(stim_comp),
            stim_mode           = np.int32(stim_mode),
            use_dfilter_primary = np.int32(use_dfilter_primary),
            dfilter_attenuation = float(dfilter_attenuation),
            dfilter_tau_ms      = float(dfilter_tau_ms),
            dual_stim_enabled   = np.int32(dual_stim_enabled),
            stype_2             = np.int32(stype_2),
            iext_2              = float(iext_2),
            t0_2                = float(t0_2),
            td_2                = float(td_2),
            atau_2              = float(atau_2),
            zap_f0_hz_2         = float(zap_f0_2),
            zap_f1_hz_2         = float(zap_f1_2),
            stim_comp_2         = np.int32(stim_comp_2),
            stim_mode_2         = np.int32(stim_mode_2),
            use_dfilter_secondary   = np.int32(use_dfilter_secondary),
            dfilter_attenuation_2   = float(dfilter_attenuation_2),
            dfilter_tau_ms_2        = float(dfilter_tau_ms_2),
            stoch_gating       = stoch_gating,
            noise_sigma       = noise_sigma,
            gna_max           = gna_max,
            gk_max            = gk_max,
            rng_state         = None,
        )

        t_out, y_out, diverged = run_native_loop(
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
        )

        if diverged:
            # Return partial result with warning flag
            res = SimulationResult(t_out, y_out, n_comp, cfg)
            self._post_process_physics(res, morph)
            res.morph = morph
            res.diverged = True
            return res

        res = SimulationResult(t_out, y_out, n_comp, cfg)
        self._post_process_physics(res, morph)
        res.morph = morph
        return res

    # ─────────────────────────────────────────────────────────────────
    def run_mc(self, n_trials: int = 10, param_var: float = 0.05,
                progress_cb=None) -> list[SimulationResult]:
        """
        Monte-Carlo parameter sweep with stochastic reproducibility.
        
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

        with ProcessPoolExecutor() as executor:
            futures = []
            for i, cfg in enumerate(configs):
                if progress_cb:
                    progress_cb(i, n_trials, None)
                futures.append(executor.submit(NeuronSolver(cfg).run_single))
            
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
        """Bifurcation diagram: V_min/V_max vs parameter in steady state."""
        ana = self.config.analysis
        param_values = np.linspace(ana.bif_min, ana.bif_max, ana.bif_steps)
        results = []

        for val in param_values:
            tmp = copy.deepcopy(self.config)
            setattr(tmp.stim, ana.bif_param, val)
            tmp.stim.t_sim = 300.0
            res = self.run_single(tmp)
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
