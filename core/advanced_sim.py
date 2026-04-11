"""
core/advanced_sim.py — Advanced Simulation Modes v10.0

Ports Scilab SWEEP, ANA_SD_CURVE, ANA_EXCMAP modes +
adds Euler-Maruyama stochastic integrator for Langevin gate noise.
"""
import numpy as np
import copy

from core.models import FullModelConfig
from core.morphology import MorphologyBuilder
from core.channels import ChannelRegistry
from core.rhs import compute_ionic_currents_scalar, get_stim_current
from core.solver import NeuronSolver, SimulationResult
from core.analysis import detect_spikes


# ─────────────────────────────────────────────────────────────────────
#  PARAMETER SWEEP
# ─────────────────────────────────────────────────────────────────────

SWEEP_PARAMS = [
    'Iext', 'gNa_max', 'gK_max', 'gL', 'ENa', 'EK', 'Cm',
    'T_celsius', 'pulse_dur', 'gIh_max', 'gCa_max', 'gA_max', 'gSK_max', 'Ra'
]


def _set_sweep_param(cfg: FullModelConfig, param: str, val: float):
    """Apply a single sweep parameter value to a config copy."""
    stim_fields    = {'Iext', 'pulse_start', 'pulse_dur', 'alpha_tau'}
    channel_fields = {'gNa_max', 'gK_max', 'gL', 'ENa', 'EK', 'EL', 'Cm',
                      'gIh_max', 'gCa_max', 'gA_max', 'gSK_max'}
    if param in stim_fields:
        setattr(cfg.stim, param, val)
    elif param in channel_fields:
        setattr(cfg.channels, param, val)
    elif param == 'T_celsius':
        cfg.env.T_celsius = val
    elif param == 'Ra':
        cfg.morphology.Ra = val


def run_sweep(config: FullModelConfig,
              param_name: str,
              param_values: np.ndarray,
              progress_cb=None) -> list:
    """
    Parametric sweep over a single parameter.

    Parameters
    ----------
    config       : base FullModelConfig
    param_name   : parameter to vary (see SWEEP_PARAMS)
    param_values : array of values to test
    progress_cb  : optional callable(i, n, val) for progress reporting

    Returns
    -------
    list of (param_value, SimulationResult | None)
    """
    results = []
    n = len(param_values)

    for i, val in enumerate(param_values):
        if progress_cb:
            progress_cb(i, n, val)

        cfg = copy.deepcopy(config)
        _set_sweep_param(cfg, param_name, float(val))

        try:
            res = NeuronSolver(cfg).run_single()
            results.append((float(val), res))
        except Exception as e:
            print(f"[SWEEP] {param_name}={val:.4g} → {e}")
            results.append((float(val), None))

    return results


# ─────────────────────────────────────────────────────────────────────
#  STRENGTH-DURATION CURVE
# ─────────────────────────────────────────────────────────────────────

_DEFAULT_DURATIONS = np.array([
    0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0
])


def run_sd_curve(config: FullModelConfig,
                 durations: np.ndarray = None,
                 progress_cb=None) -> dict:
    """
    Strength-Duration curve via binary search (Scilab ANA_SD_CURVE).

    For each pulse duration, finds the minimum current threshold I_thr
    using 16 iterations of bisection → precision ≈ 0.003 µA/cm².

    Returns
    -------
    dict with keys:
      durations, I_threshold, rheobase, chronaxie,
      weiss_fit, Q_threshold
    """
    if durations is None:
        durations = _DEFAULT_DURATIONS

    I_threshold = np.zeros(len(durations))
    n = len(durations)

    for di, dur in enumerate(durations):
        if progress_cb:
            progress_cb(di, n, dur)

        I_lo, I_hi = 0.1, 150.0

        for _ in range(18):     # 18 bisection steps → < 0.001 µA/cm²
            I_try = (I_lo + I_hi) / 2.0
            cfg = copy.deepcopy(config)
            cfg.stim.stim_type  = 'pulse'
            cfg.stim.Iext       = I_try
            cfg.stim.pulse_start = 2.0
            cfg.stim.pulse_dur  = dur
            cfg.stim.t_sim      = max(30.0, dur * 3 + 15.0)
            cfg.stim.dt_eval    = 0.05

            try:
                res = NeuronSolver(cfg).run_single()
                pks, _, _ = detect_spikes(res.v_soma, res.t)
                if len(pks) > 0:
                    I_hi = I_try
                else:
                    I_lo = I_try
            except Exception:
                I_lo = I_try

        I_threshold[di] = I_hi

    rheobase  = float(np.min(I_threshold))
    chronaxie = np.nan
    idx = np.where(I_threshold <= 2.0 * rheobase * 1.05)[0]
    if len(idx) > 0:
        chronaxie = float(durations[idx[0]])

    # Weiss' Law fit: I_thr = I_rh * (1 + t_ch / dur)
    if not np.isnan(chronaxie):
        weiss_fit = rheobase * (1.0 + chronaxie / durations)
    else:
        weiss_fit = None

    return {
        'durations':    durations,
        'I_threshold':  I_threshold,
        'rheobase':     rheobase,
        'chronaxie':    chronaxie,
        'weiss_fit':    weiss_fit,
        'Q_threshold':  I_threshold * durations,
    }


# ─────────────────────────────────────────────────────────────────────
#  2-D EXCITABILITY MAP
# ─────────────────────────────────────────────────────────────────────

def run_excitability_map(config: FullModelConfig,
                          I_range: np.ndarray = None,
                          dur_range: np.ndarray = None,
                          progress_cb=None) -> dict:
    """
    2-D excitability map: spike count & mean frequency as function of
    (I_ext, pulse_duration). Scilab ANA_EXCMAP.

    Returns
    -------
    dict with I_range, dur_range, spike_matrix, freq_matrix
    """
    ana = config.analysis

    if I_range is None:
        I_range = np.linspace(ana.excmap_I_min, ana.excmap_I_max, ana.excmap_NI)
    if dur_range is None:
        dur_range = np.linspace(ana.excmap_D_min, ana.excmap_D_max, ana.excmap_ND)

    NI, ND = len(I_range), len(dur_range)
    spike_matrix = np.zeros((NI, ND), dtype=np.int32)
    freq_matrix  = np.zeros((NI, ND))

    t_sim = float(max(50.0, dur_range[-1] * 3 + 30.0))
    total = NI * ND

    for ii, I_val in enumerate(I_range):
        for di, dur_val in enumerate(dur_range):
            if progress_cb:
                progress_cb(ii * ND + di, total, I_val)

            cfg = copy.deepcopy(config)
            cfg.stim.stim_type   = 'pulse'
            cfg.stim.Iext        = float(I_val)
            cfg.stim.pulse_start = 2.0
            cfg.stim.pulse_dur   = float(dur_val)
            cfg.stim.t_sim       = t_sim
            cfg.stim.dt_eval     = 0.05

            try:
                res = NeuronSolver(cfg).run_single()
                pks, sp_t, _ = detect_spikes(res.v_soma, res.t)
                n_sp = len(pks)
                spike_matrix[ii, di] = n_sp
                if n_sp > 1:
                    isi_mean = float(np.mean(np.diff(sp_t)))
                    freq_matrix[ii, di] = 1000.0 / isi_mean if isi_mean > 0 else 0.0
            except Exception:
                pass

    return {
        'I_range':      I_range,
        'dur_range':    dur_range,
        'spike_matrix': spike_matrix,
        'freq_matrix':  freq_matrix,
    }


# ─────────────────────────────────────────────────────────────────────
#  EULER-MARUYAMA STOCHASTIC INTEGRATOR
# ─────────────────────────────────────────────────────────────────────

def run_euler_maruyama(config: FullModelConfig,
                        progress_cb=None) -> SimulationResult:
    """
                        Run a stochastic neuron simulation using Euler–Maruyama integration with optional Langevin gate noise.
                        
                        Performs an SDE-based integration of the full neuron model configured by `config`, including optional stochastic gating and membrane current noise, and returns a downsampled SimulationResult evaluated at `config.stim.dt_eval`.
                        
                        Parameters:
                            config (FullModelConfig): Simulation configuration for morphology, channels, stimulation, calcium, and noise settings.
                            progress_cb (callable, optional): Progress callback invoked as progress_cb(step_index, n_steps, time) periodically during the integration.
                        
                        Returns:
                            SimulationResult: Time series of state variables (downsampled to `config.stim.dt_eval`), number of compartments, and the configuration used.
                        """
    from core.kinetics import (am, bm, ah, bh, an, bn,
                                ar_Ih, br_Ih,
                                as_Ca, bs_Ca, au_Ca, bu_Ca,
                                aa_IA, ba_IA, ab_IA, bb_IA,
                                z_inf_SK,
                                am_TCa, bm_TCa, ah_TCa, bh_TCa,
                                aw_IM, bw_IM,
                                ax_NaP, bx_NaP,
                                ay_NaR, by_NaR, aj_NaR, bj_NaR)
    from core.stochastic_rng import get_rng
    from scipy.sparse import csr_matrix

    cfg    = config
    morph  = MorphologyBuilder.build(cfg)
    reg    = ChannelRegistry()
    n_comp = morph['N_comp']
    ch     = cfg.channels

    y      = reg.compute_initial_states(ch.EL, cfg)
    
    # Debug: Check array sizes
    if morph['Cm_v'].shape[0] != n_comp:
        print(f"DEBUG: Cm_v shape mismatch: {morph['Cm_v'].shape[0]} vs n_comp={n_comp}")
        # Use the actual size from morphology arrays
        n_comp = morph['Cm_v'].shape[0]
    L      = csr_matrix((morph['L_data'], morph['L_indices'], morph['L_indptr']),
                        shape=(n_comp, n_comp))

    phi_na_v = cfg.env.build_phi_vector(cfg.env.Q10_Na, n_comp)
    phi_k_v = cfg.env.build_phi_vector(cfg.env.Q10_K, n_comp)
    phi_ih_v = cfg.env.build_phi_vector(cfg.env.Q10_Ih, n_comp)
    phi_ca_v = cfg.env.build_phi_vector(cfg.env.Q10_Ca, n_comp)
    # phi_ia_v removed: A-current (K channel) now uses phi_k_v (consistent with deterministic RHS)
    t_kelvin = cfg.env.T_celsius + 273.15

    # Per-compartment B_Ca (Stage 3.4 — volume-dependent calcium dynamics)
    from core.solver import NeuronSolver
    b_ca_v = NeuronSolver._build_b_ca_vector(cfg, morph)

    # Effective channel counts (proportional to conductance density)
    N_Na = max(50, int(1000 * ch.gNa_max / 120.0))
    N_K  = max(50, int(1000 * ch.gK_max  /  36.0))

    # Get centralized RNG
    rng = get_rng()

    # Integration time grid (fine step for EM)
    t_end   = cfg.stim.t_sim
    # Use configured dt_eval as base, but ensure minimum stability step
    base_dt = getattr(cfg.stim, 'dt_em', None) or cfg.stim.dt_eval
    dt      = min(base_dt / 5.0, 0.001)  # Default 1ms max, or 1/5 of eval dt
    dt      = max(dt, 0.0001)  # Minimum 0.1ms for numerical stability
    t_pts   = np.arange(0.0, t_end + dt, dt)
    n_steps = len(t_pts)
    sqrt_dt = np.sqrt(dt)

    s_map = {'const': 0, 'pulse': 1, 'alpha': 2, 'ou_noise': 0, 'zap': 10}
    stype = s_map.get(cfg.stim.stim_type, 0)

    # Pre-allocate using actual state size
    n_vars = len(y)
    sol_buf  = np.zeros((n_vars, n_steps))
    sol_buf[:, 0] = y.copy()

    stoch  = cfg.stim.stoch_gating
    noise  = cfg.stim.noise_sigma > 0
    dyn_ca = cfg.calcium.dynamic_Ca

    for k in range(1, n_steps):
        t = t_pts[k - 1]
        if progress_cb and k % 500 == 0:
            progress_cb(k, n_steps, t)

        # Unpack state
        cur = 0
        V  = y[cur:cur + n_comp];  cur += n_comp
        m  = y[cur:cur + n_comp];  cur += n_comp
        h  = y[cur:cur + n_comp];  cur += n_comp
        nk = y[cur:cur + n_comp];  cur += n_comp

        r = np.zeros(n_comp); s = np.zeros(n_comp)
        u = np.zeros(n_comp); a = np.zeros(n_comp); b = np.zeros(n_comp)
        # T-type Ca (ITCa): gates p (activation), q (inactivation)
        p = np.zeros(n_comp); q = np.ones(n_comp)
        # M-type K (IM): gate w
        w = np.zeros(n_comp)
        # Persistent Na (NaP): gate x
        x_nap = np.zeros(n_comp)
        # Resurgent Na (NaR): gates y_nr (activation), j_nr (inactivation)
        y_nr = np.zeros(n_comp); j_nr = np.ones(n_comp)
        # SK Ca-activated K: gate z_sk
        z_sk = np.zeros(n_comp)

        if ch.enable_Ih:
            r = y[cur:cur + n_comp];  cur += n_comp
        if ch.enable_ICa:
            s = y[cur:cur + n_comp];  cur += n_comp
            u = y[cur:cur + n_comp];  cur += n_comp
        if ch.enable_IA:
            a = y[cur:cur + n_comp];  cur += n_comp
            b = y[cur:cur + n_comp];  cur += n_comp
        if ch.enable_ITCa:
            p = y[cur:cur + n_comp];  cur += n_comp
            q = y[cur:cur + n_comp];  cur += n_comp
        if ch.enable_IM:
            w = y[cur:cur + n_comp];  cur += n_comp
        if ch.enable_NaP:
            x_nap = y[cur:cur + n_comp];  cur += n_comp
        if ch.enable_NaR:
            y_nr = y[cur:cur + n_comp];  cur += n_comp
            j_nr = y[cur:cur + n_comp];  cur += n_comp
        if ch.enable_SK:
            z_sk = y[cur:cur + n_comp];  cur += n_comp

        ca_i = np.full(n_comp, cfg.calcium.Ca_rest)
        if dyn_ca:
            ca_i = y[cur:cur + n_comp];  cur += n_comp

        # ── ionic currents (shared math with deterministic RHS) ────
        I_ion = np.zeros(n_comp)
        I_ca_total = np.zeros(n_comp)
        for i in range(n_comp):
            sk_gate_i = z_inf_SK(ca_i[i]) if ch.enable_SK else 0.0
            i_ion_i, i_ca_influx_i = compute_ionic_currents_scalar(
                V[i], m[i], h[i], nk[i],
                r[i] if ch.enable_Ih else 0.0,
                s[i] if ch.enable_ICa else 0.0,
                u[i] if ch.enable_ICa else 0.0,
                a[i] if ch.enable_IA else 0.0,
                b[i] if ch.enable_IA else 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                sk_gate_i,
                ch.enable_Ih, ch.enable_ICa, ch.enable_IA, ch.enable_SK,
                False, False, False, False, dyn_ca,
                morph['gNa_v'][i], morph['gK_v'][i], morph['gL_v'][i], morph['gIh_v'][i],
                morph['gCa_v'][i], morph['gA_v'][i], morph['gSK_v'][i], 0.0, 0.0, 0.0, 0.0,
                ch.ENa, ch.EK, ch.EL, ch.E_Ih, ch.EK,  # A-current uses EK
                ca_i[i], cfg.calcium.Ca_ext, cfg.calcium.Ca_rest, t_kelvin,
            )
            I_ion[i] = i_ion_i
            I_ca_total[i] = i_ca_influx_i

        I_ax    = L.dot(V)
        I_stim  = np.zeros(n_comp)
        sc      = cfg.stim.stim_comp
        if 0 <= sc < n_comp:
            I_stim[sc] = get_stim_current(
                t, stype, cfg.stim.Iext,
                cfg.stim.pulse_start,
                cfg.stim.pulse_dur,
                cfg.stim.alpha_tau,
                float(getattr(cfg.stim, "zap_f0_hz", 0.5)),
                float(getattr(cfg.stim, "zap_f1_hz", 40.0)),
            )

        # ── kinetics ────────────────────────────────────────────────
        am_v = am(V);  bm_v = bm(V)
        ah_v = ah(V);  bh_v = bh(V)
        an_v = an(V);  bn_v = bn(V)

        # ── deterministic drift vector ───────────────────────────────
        dy = np.zeros_like(y)
        cur = 0
        dy[cur:cur + n_comp] = (I_stim - I_ion + I_ax) / morph['Cm_v'];  cur += n_comp
        dy[cur:cur + n_comp] = phi_na_v * (am_v * (1 - m)  - bm_v * m);        cur += n_comp
        dy[cur:cur + n_comp] = phi_na_v * (ah_v * (1 - h)  - bh_v * h);        cur += n_comp
        dy[cur:cur + n_comp] = phi_k_v * (an_v * (1 - nk) - bn_v * nk);        cur += n_comp

        if ch.enable_Ih:
            ar_v = ar_Ih(V);  br_v = br_Ih(V)
            dy[cur:cur + n_comp] = phi_ih_v * (ar_v * (1 - r) - br_v * r);  cur += n_comp
        if ch.enable_ICa:
            as_v = as_Ca(V);  bs_v = bs_Ca(V)
            au_v = au_Ca(V);  bu_v = bu_Ca(V)
            dy[cur:cur + n_comp] = phi_ca_v * (as_v * (1 - s) - bs_v * s);  cur += n_comp
            dy[cur:cur + n_comp] = phi_ca_v * (au_v * (1 - u) - bu_v * u);  cur += n_comp
        if ch.enable_IA:
            aa_v = aa_IA(V);  ba_v = ba_IA(V)
            ab_v = ab_IA(V);  bb_v = bb_IA(V)
            # A-current is a K channel — scale with phi_k_v (not phi_ia_v)
            dy[cur:cur + n_comp] = phi_k_v * (aa_v * (1 - a) - ba_v * a);  cur += n_comp
            dy[cur:cur + n_comp] = phi_k_v * (ab_v * (1 - b) - bb_v * b);  cur += n_comp
        if dyn_ca:
            dca = (-b_ca_v * I_ca_total
                   - (ca_i - cfg.calcium.Ca_rest) / cfg.calcium.tau_Ca)
            dca = np.where((ca_i < 1e-9) & (dca < 0), 0.0, dca)
            dy[cur:cur + n_comp] = dca

        # ── Euler step ──────────────────────────────────────────────
        y_new = y + dt * dy

        # ── Langevin diffusion (gate noise) ──────────────────────────
        if stoch:
            g_start = n_comp
            def _add_noise(offset, alpha_v, beta_v, gate_v, N_ch, phi_gate_v):
                N_ch_safe = np.maximum(N_ch, 1.0)  # Guard against zero channel count
                sigma = np.sqrt(
                    np.maximum(0.0, alpha_v * (1.0 - gate_v) + beta_v * gate_v) / N_ch_safe
                ) * phi_gate_v
                y_new[offset:offset + n_comp] += sigma * np.random.randn(n_comp) * sqrt_dt

            _add_noise(g_start,           am_v, bm_v, m,  N_Na, phi_na_v)
            _add_noise(g_start + n_comp,   ah_v, bh_v, h,  N_Na, phi_na_v)
            _add_noise(g_start + 2*n_comp, an_v, bn_v, nk, N_K, phi_k_v)

        # ── Membrane current noise ───────────────────────────────────
        if noise:
            y_new[:n_comp] += (cfg.stim.noise_sigma * np.random.randn(n_comp)
                               * sqrt_dt / morph['Cm_v'])

        # ── Clamp gates [0, 1] ──────────────────────────────────────
        # Safely clip all gating variables (everything between V and Ca)
        gate_start = n_comp
        gate_end = n_comp * 4  # m, h, n
        if ch.enable_Ih: gate_end += n_comp
        if ch.enable_ICa: gate_end += 2 * n_comp
        if ch.enable_IA: gate_end += 2 * n_comp
        if ch.enable_ITCa: gate_end += 2 * n_comp
        if ch.enable_IM: gate_end += n_comp
        if ch.enable_NaP: gate_end += n_comp
        if ch.enable_NaR: gate_end += 2 * n_comp
        if ch.enable_SK: gate_end += n_comp

        y_new[gate_start:gate_end] = np.clip(y_new[gate_start:gate_end], 0.0, 1.0)
        if dyn_ca:
            y_new[gate_end:gate_end+n_comp] = np.maximum(y_new[gate_end:gate_end+n_comp], 1e-9)

        y = y_new
        sol_buf[:, k] = y

    # ── Downsample to dt_eval ────────────────────────────────────────
    t_eval   = np.arange(0.0, t_end, cfg.stim.dt_eval)
    indices  = np.clip(np.searchsorted(t_pts, t_eval), 0, n_steps - 1)
    sol_y    = sol_buf[:, indices]

    res = SimulationResult(t_eval, sol_y, n_comp, cfg)
    NeuronSolver(cfg)._post_process_physics(res, morph)
    return res
