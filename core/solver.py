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

        self.v_all  = y[0:n_comp, :]
        self.v_soma = self.v_all[0, :]

        dual_cfg = getattr(config, "dual_stimulation", None)
        dual_enabled = bool(dual_cfg is not None and getattr(dual_cfg, "enabled", False))

        primary_location = (
            getattr(dual_cfg, "primary_location", config.stim_location.location)
            if dual_enabled
            else config.stim_location.location
        )
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
        av_soma = 6.0 / d_soma  # sphere

        b_ca_v = np.empty(n_comp, dtype=np.float64)
        b_ca_v[0] = b_ca_base  # soma = user value

        for i in range(1, n_comp):
            av_i = 4.0 / diameters[i]  # cylinder
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
            'Kainate': 8, 'Nicotinic': 9,
        }
        t_kelvin = cfg.env.T_celsius + 273.15

        stim_mode_map = {'soma': 0, 'ais': 1, 'dendritic_filtered': 2}
        # --- Primary stimulus source ---
        primary_stim_type = cfg.stim.stim_type
        primary_iext = cfg.stim.Iext
        primary_t0 = cfg.stim.pulse_start
        primary_td = cfg.stim.pulse_dur
        primary_atau = cfg.stim.alpha_tau
        primary_stim_comp = cfg.stim.stim_comp
        primary_location = cfg.stim_location.location

        # --- Dual stimulation detection and parameter preparation ---
        dual_stim_enabled = 0
        dual_cfg = getattr(cfg, 'dual_stimulation', None)
        if dual_cfg is not None and hasattr(dual_cfg, 'enabled') and dual_cfg.enabled:
            dual_stim_enabled = 1
            # When dual stimulation is enabled, primary parameters come from the dual config.
            primary_stim_type = getattr(dual_cfg, "primary_stim_type", primary_stim_type)
            primary_iext = getattr(dual_cfg, "primary_Iext", primary_iext)
            primary_t0 = getattr(dual_cfg, "primary_start", primary_t0)
            primary_td = getattr(dual_cfg, "primary_duration", primary_td)
            primary_atau = getattr(dual_cfg, "primary_alpha_tau", primary_atau)
            primary_location = getattr(dual_cfg, "primary_location", primary_location)
            primary_stim_comp = 0

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
        stype_2, iext_2, t0_2, td_2, atau_2 = 0, 0.0, 0.0, 0.0, 0.0
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
            dfilter_tau_ms_2 = dual_cfg.secondary_tau_dendritic_ms
            use_dfilter_secondary = int(stim_mode_2 == 2 and dfilter_tau_ms_2 > 0.0)
            if use_dfilter_secondary == 1:
                y0 = np.concatenate([y0, np.array([0.0])])
            if stim_mode_2 == 2 and dual_cfg.secondary_space_constant_um > 0:
                dfilter_attenuation_2 = np.exp(
                    -dual_cfg.secondary_distance_um / dual_cfg.secondary_space_constant_um
                )

        args = (
            n_comp,
            cfg.channels.enable_Ih, cfg.channels.enable_ICa,
            cfg.channels.enable_IA, cfg.channels.enable_SK,
            cfg.calcium.dynamic_Ca, cfg.channels.enable_ITCa,
            cfg.channels.enable_IM, cfg.channels.enable_NaP, cfg.channels.enable_NaR,
            morph['gNa_v'], morph['gK_v'], morph['gL_v'],
            morph['gIh_v'], morph['gCa_v'], morph['gA_v'], morph['gSK_v'], morph['gTCa_v'],
            morph['gIM_v'], morph['gNaP_v'], morph['gNaR_v'],
            cfg.channels.ENa, cfg.channels.EK, cfg.channels.EL,
            cfg.channels.E_Ih, cfg.channels.E_A,
            morph['Cm_v'],
            morph['L_data'], morph['L_indices'], morph['L_indptr'],
            cfg.env.phi_channel(cfg.env.Q10_Na),
            cfg.env.phi_channel(cfg.env.Q10_K),
            cfg.env.phi_channel(cfg.env.Q10_Ih),
            cfg.env.phi_channel(cfg.env.Q10_Ca),
            cfg.env.phi_channel(cfg.env.Q10_IA),
            cfg.env.phi_channel(cfg.env.Q10_TCa),
            cfg.env.phi_channel(cfg.env.Q10_IM),
            cfg.env.phi_channel(cfg.env.Q10_NaP),
            cfg.env.phi_channel(cfg.env.Q10_NaR),
            t_kelvin,
            cfg.calcium.Ca_ext, cfg.calcium.Ca_rest,
            cfg.calcium.tau_Ca,
            self._build_b_ca_vector(cfg, morph),
            cfg.env.Mg_ext,
            cfg.channels.tau_SK,
            stype, primary_iext,
            primary_t0, primary_td,
            primary_atau, primary_stim_comp, stim_mode,
            use_dfilter_primary, dfilter_attenuation, dfilter_tau_ms,
            # Dual stimulation parameters (optional)
            dual_stim_enabled,
            stype_2, iext_2, t0_2, td_2, atau_2, stim_comp_2, stim_mode_2,
            use_dfilter_secondary, dfilter_attenuation_2, dfilter_tau_ms_2,
        )

        t_eval = np.arange(0.0, cfg.stim.t_sim, cfg.stim.dt_eval)
        
        # ── Optimization settings ──
        # max_step prevents integrator from taking too large steps
        # which can cause instability in stiff systems
        max_step = min(cfg.stim.dt_eval * 5, 1.0)  # Max 1ms or 5x evaluation step

        jacobian_mode = cfg.stim.jacobian_mode
        jacobian_options = {}
        if jacobian_mode == "sparse_fd":
            jacobian_options["jac_sparsity"] = build_jacobian_sparsity(
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
            )
        elif jacobian_mode == "analytic_sparse":
            sparsity = build_jacobian_sparsity(
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
            )
            jacobian_options["jac"] = make_analytic_jacobian(sparsity)
        elif jacobian_mode != "dense_fd":
            raise ValueError(f"Unsupported jacobian_mode={jacobian_mode}")
        
        sol = solve_ivp(
            rhs_multicompartment,
            (0.0, cfg.stim.t_sim),
            y0,
            args=args,
            method='BDF',
            t_eval=t_eval,
            rtol=1e-5,
            atol=1e-7,
            max_step=max_step,
            dense_output=False,  # Save memory
            **jacobian_options,
        )
        
        # Report actual simulation time
        t_elapsed = time.time() - t_start
        if rt["estimated_seconds"] > 30.0 or t_elapsed > 10:
            logger.info("   Completed in %.1fs", t_elapsed)

        if not sol.success:
            raise RuntimeError(f"Integrator failed: {sol.message}")

        res = SimulationResult(sol.t, sol.y, n_comp, cfg)
        self._post_process_physics(res, morph)
        res.morph = morph        # store for current-balance analysis
        return res

    # ─────────────────────────────────────────────────────────────────
    def _post_process_physics(self, res: SimulationResult, morph: dict):
        """Reconstruct ion-channel current densities and ATP estimate."""
        y, n, cfg = res.y, res.n_comp, res.config
        v  = y[0:n, :]
        m  = y[n   :2*n, :]
        h  = y[2*n :3*n, :]
        nk = y[3*n :4*n, :]

        res.currents['Na']   = morph['gNa_v'][0] * (m[0, :] ** 3) * h[0, :] * (v[0, :] - cfg.channels.ENa)
        res.currents['K']    = morph['gK_v'][0]  * (nk[0, :] ** 4) * (v[0, :] - cfg.channels.EK)
        res.currents['Leak'] = morph['gL_v'][0]  * (v[0, :] - cfg.channels.EL)

        cursor = 4 * n

        if cfg.channels.enable_Ih:
            r = y[cursor:cursor + n, :]
            res.currents['Ih'] = morph['gIh_v'][0] * r[0, :] * (v[0, :] - cfg.channels.E_Ih)
            cursor += n

        if cfg.channels.enable_ICa:
            s = y[cursor:cursor + n, :]
            u = y[cursor + n:cursor + 2*n, :]
            if cfg.calcium.dynamic_Ca and res.ca_i is not None:
                t_kelvin = cfg.env.T_celsius + 273.15
                ca_soma = np.maximum(res.ca_i[0, :], 1e-9)
                e_ca = (R_GAS * t_kelvin / (2.0 * F_CONST)) * np.log(cfg.calcium.Ca_ext / ca_soma) * 1000.0
            else:
                e_ca = 120.0
            res.currents['ICa'] = morph['gCa_v'][0] * (s[0, :] ** 2) * u[0, :] * (v[0, :] - e_ca)
            cursor += 2 * n

        if cfg.channels.enable_IA:
            a = y[cursor:cursor + n, :]
            b = y[cursor + n:cursor + 2*n, :]
            res.currents['IA'] = morph['gA_v'][0] * a[0, :] * b[0, :] * (v[0, :] - cfg.channels.E_A)
            cursor += 2 * n

        if cfg.channels.enable_ITCa:
            p_t = y[cursor:cursor + n, :]
            q_t = y[cursor + n:cursor + 2*n, :]
            if not cfg.channels.enable_ICa:
                # e_ca not yet computed — compute here
                if cfg.calcium.dynamic_Ca and res.ca_i is not None:
                    t_kelvin = cfg.env.T_celsius + 273.15
                    ca_soma = np.maximum(res.ca_i[0, :], 1e-9)
                    e_ca = (R_GAS * t_kelvin / (2.0 * F_CONST)) * np.log(cfg.calcium.Ca_ext / ca_soma) * 1000.0
                else:
                    e_ca = 120.0
            res.currents['ITCa'] = morph['gTCa_v'][0] * (p_t[0, :] ** 2) * q_t[0, :] * (v[0, :] - e_ca)
            cursor += 2 * n

        if cfg.channels.enable_IM:
            w_m = y[cursor:cursor + n, :]
            res.currents['IM'] = morph['gIM_v'][0] * w_m[0, :] * (v[0, :] - cfg.channels.EK)
            cursor += n

        if cfg.channels.enable_NaP:
            x_p = y[cursor:cursor + n, :]
            res.currents['NaP'] = morph['gNaP_v'][0] * x_p[0, :] * (v[0, :] - cfg.channels.ENa)
            cursor += n

        if cfg.channels.enable_NaR:
            y_r = y[cursor:cursor + n, :]
            j_r = y[cursor + n:cursor + 2*n, :]
            res.currents['NaR'] = morph['gNaR_v'][0] * y_r[0, :] * j_r[0, :] * (v[0, :] - cfg.channels.ENa)
            cursor += 2 * n

        if cfg.channels.enable_SK:
            z_sk = y[cursor:cursor + n, :]
            res.currents['SK'] = morph['gSK_v'][0] * z_sk[0, :] * (v[0, :] - cfg.channels.EK)
            cursor += n

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
    def run_monte_carlo(self) -> list:
        """Parallel Monte-Carlo: biological variability ±5% on gNa, gK."""
        n_trials = self.config.analysis.mc_trials
        configs  = []
        for _ in range(n_trials):
            m_cfg = copy.deepcopy(self.config)
            m_cfg.channels.gNa_max *= np.random.normal(1.0, 0.05)
            m_cfg.channels.gK_max  *= np.random.normal(1.0, 0.05)
            configs.append(m_cfg)

        with ProcessPoolExecutor() as executor:
            results = list(executor.map(worker_task, configs))
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

    # ─────────────────────────────────────────────────────────────────
    def run_native(self, custom_config: "FullModelConfig | None" = None) -> SimulationResult:
        """Run simulation with the native Hines solver (v11.1).
        
        Fixed-step Backward-Euler + TRUE O(N) Hines tree solver.
        Returns a standard SimulationResult compatible with GUI and analytics.
        """
        from core.native_loop import run_native_loop
        import scipy.sparse
        import time

        cfg = custom_config or self.config
        
        logger.info("Using NATIVE HINES solver (O(N) tree solver, Exponential Euler)")
        t_start = time.time()

        morph  = MorphologyBuilder.build(cfg)
        n_comp = morph['N_comp']

        y0 = self.registry.compute_initial_states(-65.0, cfg)

        # ── Stimulus maps (mirrors run_single) ──
        s_map = {
            "const": 0, "pulse": 1, "alpha": 2, "ou_noise": 3,
            "AMPA": 4, "NMDA": 5, "GABAA": 6, "GABAB": 7,
            "Kainate": 8, "Nicotinic": 9, "zap": 10,
        }
        stim_mode_map = {"soma": 0, "ais": 1, "dendritic_filtered": 2}
        t_kelvin = cfg.env.T_celsius + 273.15

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
            # When dual stimulation is enabled, primary parameters come from dual config.
            primary_stim_type = getattr(dual_cfg, "primary_stim_type", primary_stim_type)
            primary_iext = getattr(dual_cfg, "primary_Iext", primary_iext)
            primary_t0 = getattr(dual_cfg, "primary_start", primary_t0)
            primary_td = getattr(dual_cfg, "primary_duration", primary_td)
            primary_atau = getattr(dual_cfg, "primary_alpha_tau", primary_atau)
            primary_location = getattr(dual_cfg, "primary_location", primary_location)
            primary_stim_comp = 0
            primary_zap_f0 = getattr(dual_cfg, "primary_zap_f0_hz", primary_zap_f0)
            primary_zap_f1 = getattr(dual_cfg, "primary_zap_f1_hz", primary_zap_f1)

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
        stype_2, iext_2, t0_2, td_2, atau_2 = 0, 0.0, 0.0, 0.0, 0.0
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

        # ── Event-driven stimulus preparation ──
        event_times = []
        if stype in [4, 5, 6, 7, 8, 9]:  # AMPA, NMDA, GABAA, GABAB, Kainate, Nicotinic
            event_times = np.array([primary_t0])
        n_events = len(event_times)

        # ── Conductance vs current mode ──
        is_conductance_based = stype in [4, 5, 6, 7, 8, 9]
        is_nmda = stype == 5
        is_cond_2 = stype_2 in [4, 5, 6, 7, 8, 9]
        is_nmda_2 = stype_2 == 5

        # ── Synaptic reversal potentials ──
        e_syn = _get_syn_reversal(stype)
        e_syn_2 = _get_syn_reversal(stype_2) if dual_stim_enabled else 0.0

        # ── State offsets ──
        off_s = 0
        off_m = n_comp
        off_h = off_m + n_comp
        off_n = off_h + n_comp
        off_r = off_n + n_comp
        off_w = off_r + n_comp
        off_w_im = off_w + n_comp
        off_ca = off_w_im + n_comp
        off_zsk = off_ca + n_comp
        off_itca = off_zsk + n_comp
        off_im = off_itca + n_comp
        off_nap = off_im + n_comp
        off_nar = off_nap + n_comp
        off_vfilt_primary = off_nar + n_comp
        off_vfilt_secondary = off_vfilt_primary + n_comp

        # ── Morphology arrays for Hines ──
        parent_idx = morph['parent_idx']
        hines_order = morph['hines_order']
        g_axial_to_parent = morph['g_axial_to_parent']
        g_axial_parent_to_child = morph['g_axial_parent_to_child']

        # ── Conductance arrays ──
        gna_v = morph['gNa_v']
        gk_v = morph['gK_v']
        gl_v = morph['gL_v']
        gih_v = morph['gIh_v']
        gca_v = morph['gCa_v']
        ga_v = morph['gA_v']
        gsk_v = morph['gSK_v']
        gtca_v = morph['gTCa_v']
        gim_v = morph['gIM_v']
        gnap_v = morph['gNaP_v']
        gnar_v = morph['gNaR_v']

        # ── Reversal potentials ──
        ena = cfg.channels.E_Na
        ek = cfg.channels.E_K
        el = cfg.channels.E_L
        eih = cfg.channels.E_Ih
        ea = cfg.channels.E_A
        en_ica = cfg.channels.E_Ca
        en_sk = cfg.channels.E_SK
        en_itca = cfg.channels.E_TCa
        en_im = cfg.channels.E_IM
        en_nap = cfg.channels.E_NaP
        en_nar = cfg.channels.E_NaR

        # ── Channel enable flags ──
        en_sk = cfg.channels.enable_SK
        en_itca = cfg.channels.enable_TCa
        en_im = cfg.channels.enable_IM
        en_nap = cfg.channels.enable_NaP
        en_nar = cfg.channels.enable_NaR
        dyn_ca = cfg.calcium.dynamic_Ca

        # ── Membrane capacitance ──
        cm_v = np.full(n_comp, cfg.channels.C_m)

        # ── Build B_Ca vector ──
        b_ca_v = self._build_b_ca_vector(cfg, morph)

        # ── Time stepping: internal vs output ──
        # Hodgkin-Huxley needs dt ≤ 0.025ms for stability
        dt_eval = float(cfg.stim.dt_eval)
        dt_internal = min(0.025, dt_eval / 4.0)  # At least 4 micro-steps per output point
        
        t_out, y_out = run_native_loop(
            y0.astype(np.float64),
            float(cfg.stim.t_sim), dt_internal, dt_eval,
            int(n_comp),
            parent_idx.astype(np.int64),
            hines_order.astype(np.int64),
            # Morphology arrays
            g_axial_to_parent.astype(np.float64),
            g_axial_parent_to_child.astype(np.float64),
            cm_v.astype(np.float64),
            # Conductance arrays
            gna_v.astype(np.float64), gk_v.astype(np.float64), gl_v.astype(np.float64),
            gih_v.astype(np.float64), gca_v.astype(np.float64), ga_v.astype(np.float64),
            gsk_v.astype(np.float64), gtca_v.astype(np.float64), gim_v.astype(np.float64),
            gnap_v.astype(np.float64), gnar_v.astype(np.float64),
            # Reversal potentials
            ena, ek, el, eih, ea, en_ica, en_sk, en_itca, en_im, en_nap, en_nar,
            # Environment
            t_kelvin, cfg.env.Ca_ext, cfg.calcium.Ca_rest,
            cfg.calcium.tau_Ca, cfg.env.Mg_ext, cfg.calcium.tau_SK,
            # Channel enable flags
            cfg.channels.enable_Ih, cfg.channels.enable_ICa, cfg.channels.enable_IA,
            cfg.channels.enable_SK, cfg.channels.enable_ITCa, cfg.channels.enable_IM,
            cfg.channels.enable_NaP, cfg.channels.enable_NaR, dyn_ca,
            # Primary stimulus
            stype, primary_iext, primary_t0, primary_td, primary_atau,
            primary_zap_f0, primary_zap_f1, primary_stim_comp, stim_mode,
            use_dfilter_primary, dfilter_attenuation, dfilter_tau_ms,
            # Secondary stimulus
            dual_stim_enabled, stype_2, iext_2, t0_2, td_2, atau_2,
            zap_f0_2, zap_f1_2, stim_comp_2, stim_mode_2,
            use_dfilter_secondary, dfilter_attenuation_2, dfilter_tau_ms_2,
            # Event-driven
            event_times.astype(np.float64), n_events,
            # Synaptic
            is_conductance_based, is_nmda,
            e_syn,
        )

        # ── Package result ──
        result = SimulationResult(t_out, y_out, n_comp, cfg)
        result.morph = morph
        result.currents = {
            'gna_v': gna_v, 'gk_v': gk_v, 'gl_v': gl_v, 'gih_v': gih_v,
            'gca_v': gca_v, 'ga_v': ga_v, 'gsk_v': gsk_v, 'gtca_v': gtca_v,
            'gim_v': gim_v, 'gnap_v': gnap_v, 'gnar_v': gnar_v,
        }
        result.atp_estimate = self._estimate_atp_consumption(result, cfg)
        
        t_elapsed = time.time() - t_start
        logger.info("Native Hines solver completed in %.2fs", t_elapsed)
        
        return result


def worker_task(config: FullModelConfig) -> SimulationResult:
    """Picklable worker for ProcessPoolExecutor."""
    return NeuronSolver(config).run_single()
