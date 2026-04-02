import numpy as np
import copy
from scipy.integrate import solve_ivp
from concurrent.futures import ProcessPoolExecutor

from core.models import FullModelConfig
from core.morphology import MorphologyBuilder
from core.channels import ChannelRegistry
from core.rhs import rhs_multicompartment
from core.kinetics import z_inf_SK


class SimulationResult:
    """Complete scientific dataset after a simulation run."""

    def __init__(self, t, y, n_comp, config: FullModelConfig):
        self.t      = t
        self.y      = y
        self.config = config
        self.n_comp = n_comp

        self.v_all  = y[0:n_comp, :]
        self.v_soma = self.v_all[0, :]

        has_dfilter_state = (
            config.stim_location.location == "dendritic_filtered" and
            config.dendritic_filter.enabled
        )

        self.ca_i = None
        if config.calcium.dynamic_Ca:
            if has_dfilter_state:
                self.ca_i = y[-(n_comp + 1):-1, :]
            else:
                self.ca_i = y[-n_comp:, :]

        self.v_dendritic_filtered = None
        if has_dfilter_state:
            self.v_dendritic_filtered = y[-1, :]

        self.currents:    dict  = {}
        self.atp_estimate: float = 0.0

        # Morphology dict stored for post-analysis (current balance, etc.)
        self.morph: dict = {}


class NeuronSolver:
    """Multi-threaded simulation engine v10.0."""

    def __init__(self, config: FullModelConfig):
        self.config   = config
        self.registry = ChannelRegistry()

    # ─────────────────────────────────────────────────────────────────
    def run_single(self, custom_config: FullModelConfig = None) -> SimulationResult:
        """Run a single deterministic simulation (BDF integrator)."""
        cfg = custom_config or self.config

        morph  = MorphologyBuilder.build(cfg)
        n_comp = morph['N_comp']

        y0 = self.registry.compute_initial_states(-65.0, cfg)

        s_map    = {
            'const': 0, 'pulse': 1, 'alpha': 2, 'ou_noise': 3,
            'AMPA': 4, 'NMDA': 5, 'GABAA': 6, 'GABAB': 7,
            'Kainate': 8, 'Nicotinic': 9,
        }
        stype    = s_map.get(cfg.stim.stim_type, 0)
        t_kelvin = cfg.env.T_celsius + 273.15

        stim_mode_map = {'soma': 0, 'ais': 1, 'dendritic_filtered': 2}
        stim_mode = stim_mode_map.get(cfg.stim_location.location, 0)
        use_dfilter = int(stim_mode == 2 and cfg.dendritic_filter.enabled)

        if use_dfilter == 1:
            y0 = np.concatenate([y0, np.array([0.0])])

        attenuation = 1.0
        if use_dfilter == 1 and cfg.dendritic_filter.space_constant_um > 0:
            attenuation = np.exp(
                -cfg.dendritic_filter.distance_um / cfg.dendritic_filter.space_constant_um
            )

        # --- Dual stimulation detection and parameter preparation ---
        dual_stim_enabled = 0
        stype_2, iext_2, t0_2, td_2, atau_2, stim_comp_2, stim_mode_2 = 0, 0.0, 0.0, 0.0, 0.0, 0, 0
        dfilter_attenuation_2, dfilter_tau_ms_2 = 1.0, 0.0
        
        if hasattr(cfg, 'dual_stimulation') and cfg.dual_stimulation is not None:
            dual_cfg = cfg.dual_stimulation
            if hasattr(dual_cfg, 'enabled') and dual_cfg.enabled:
                dual_stim_enabled = 1
                
                # Map secondary stimulus type
                stype_2 = s_map.get(dual_cfg.secondary_stim_type, 0)
                iext_2 = dual_cfg.secondary_Iext
                t0_2 = dual_cfg.secondary_start
                td_2 = dual_cfg.secondary_duration
                atau_2 = dual_cfg.secondary_alpha_tau
                stim_comp_2 = 0  # Default to soma
                
                # Map secondary stimulus location
                stim_mode_2 = stim_mode_map.get(dual_cfg.secondary_location, 0)
                
                # Dendritic filter for secondary stimulus (if dendritic_filtered)
                if stim_mode_2 == 2 and dual_cfg.secondary_space_constant_um > 0:
                    dfilter_attenuation_2 = np.exp(
                        -dual_cfg.secondary_distance_um / dual_cfg.secondary_space_constant_um
                    )
                    dfilter_tau_ms_2 = dual_cfg.secondary_tau_dendritic_ms

        args = (
            n_comp,
            cfg.channels.enable_Ih, cfg.channels.enable_ICa,
            cfg.channels.enable_IA, cfg.channels.enable_SK,
            cfg.calcium.dynamic_Ca,
            morph['gNa_v'], morph['gK_v'], morph['gL_v'],
            morph['gIh_v'], morph['gCa_v'], morph['gA_v'], morph['gSK_v'],
            cfg.channels.ENa, cfg.channels.EK, cfg.channels.EL,
            cfg.channels.E_Ih, cfg.channels.E_A,
            morph['Cm_v'],
            morph['L_data'], morph['L_indices'], morph['L_indptr'],
            cfg.env.phi, t_kelvin,
            cfg.calcium.Ca_ext, cfg.calcium.Ca_rest,
            cfg.calcium.tau_Ca, cfg.calcium.B_Ca,
            stype, cfg.stim.Iext,
            cfg.stim.pulse_start, cfg.stim.pulse_dur,
            cfg.stim.alpha_tau, cfg.stim.stim_comp, stim_mode,
            use_dfilter, attenuation,
            cfg.dendritic_filter.tau_dendritic_ms if use_dfilter == 1 else 0.0,
            # Dual stimulation parameters (optional)
            dual_stim_enabled,
            stype_2, iext_2, t0_2, td_2, atau_2, stim_comp_2, stim_mode_2,
            dfilter_attenuation_2, dfilter_tau_ms_2,
        )

        t_eval = np.arange(0.0, cfg.stim.t_sim, cfg.stim.dt_eval)
        sol = solve_ivp(
            rhs_multicompartment,
            (0.0, cfg.stim.t_sim),
            y0,
            args=args,
            method='BDF',
            t_eval=t_eval,
            rtol=1e-5,
            atol=1e-7,
        )

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
            res.currents['ICa'] = morph['gCa_v'][0] * (s[0, :] ** 2) * u[0, :] * (v[0, :] - 120.0)
            cursor += 2 * n

        if cfg.channels.enable_IA:
            a = y[cursor:cursor + n, :]
            b = y[cursor + n:cursor + 2*n, :]
            res.currents['IA'] = morph['gA_v'][0] * a[0, :] * b[0, :] * (v[0, :] - cfg.channels.E_A)
            cursor += 2 * n

        if cfg.channels.enable_SK and res.ca_i is not None:
            z_act = z_inf_SK(res.ca_i[0, :])
            res.currents['SK'] = morph['gSK_v'][0] * z_act * (v[0, :] - cfg.channels.EK)

        # ATP estimate: Q_Na [nC/cm²] → nmol ATP/cm²
        dt = cfg.stim.dt_eval
        q_na = float(np.sum(np.abs(res.currents['Na']))) * dt
        res.atp_estimate = (q_na * 1e-9) / (3.0 * 96485.33) * 1e9

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


def worker_task(config: FullModelConfig) -> SimulationResult:
    """Picklable worker for ProcessPoolExecutor."""
    return NeuronSolver(config).run_single()
