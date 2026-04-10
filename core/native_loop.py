"""Native Hines time-loop — v11.2.

Pure @njit Backward-Euler time integration using the O(N) Hines solver.
Called by NeuronSolver.run_native(); returns (t_arr, y_arr) where
y_arr has shape (N_state, N_out) matching SimulationResult expectations.

Architecture change from v11.0:
  PhysicsParams NamedTuple carries all static physics, eliminating the
  80+ argument signature.  l_diag is passed separately (it is the
  Laplacian diagonal needed every step, and can't be extracted from the
  CSR sparse format inside @njit without extra work).

RHS notation (per compartment i):
  d[i]   = Cm_i/dt + g_total_ion_i - L_diag_i     (L_diag < 0)
  a[i]   = g_axial_to_parent[i]                    (positive, child→parent)
  b[i]   = g_axial_parent_to_child[i]              (positive, parent←child)
  rhs[i] = Cm_i/dt*V_n + e_eff_i + I_stim_eff_i

Hines system (solved by hines.py::hines_solve):
  d[i]*V[i] - a[i]*V[parent] - sum_c b[c]*V[c]  = rhs[i]
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .physics_params import PhysicsParams, unpack_conductances, unpack_temperature_scaling
from .rhs import (
    get_stim_current, get_event_driven_conductance,
    _get_syn_reversal, nernst_ca_ion, nmda_mg_block,
    CA_I_MIN_M_M, CA_I_MAX_M_M, CA_DAMPING_FACTOR,
    ATP_MIN_M_M, ATP_MAX_M_M, ATP_PUMP_FAILURE_THRESHOLD,
)
from .dual_stimulation import distributed_stimulus_current_for_comp
from .hines import hines_solve, update_gates_analytic
from .kinetics import am_lut, bm_lut, ah_lut, bh_lut, an_lut, bn_lut


@njit(fastmath=True, cache=True)
def _compute_ionic_currents_vectorized(
    v_arr, m_arr, h_arr, n_arr,
    r_arr, s_arr, u_arr, a_arr, b_arr, p_arr, q_arr, w_arr, x_arr, y_arr, j_arr, z_arr,
    ca_arr,
    gna_v, gk_v, gl_v, gih_v, gca_v, ga_v, gsk_v, gtca_v, gim_v, gnap_v, gnar_v,
    ena, ek, el, eih, ea,
    en_ih, en_ica, en_ia, en_sk, en_itca, en_im, en_nap, en_nar, dyn_ca,
    ca_ext, t_kelvin, ca_rest,
    n_comp,
):
    """
    Compute per-compartment total membrane conductance, conductance-weighted effective reversal, and calcium influx contributions.
    
    For each compartment, returns:
    - g_total: sum of all active and passive ionic conductances.
    - e_eff: conductance-weighted effective reversal potential (Σ g_ch * E_ch).
    - i_ca_influx: nonnegative calcium influx contributions (accumulated inward currents from calcium-permeable channels and ITCa).
    
    Notes:
    - When `dyn_ca` is enabled, the calcium reversal potential is computed from the Nernst equation using a clamped intracellular calcium concentration; otherwise a constant calcium reversal is used.
    - The IA current is treated as a potassium-like channel and contributes with the potassium reversal (`ek`).
    """
    g_total = np.empty(n_comp, dtype=np.float64)
    e_eff = np.empty(n_comp, dtype=np.float64)
    i_ca_influx = np.zeros(n_comp, dtype=np.float64)
    
    for i in range(n_comp):
        vi = v_arr[i]
        mi = m_arr[i]
        hi = h_arr[i]
        ni = n_arr[i]
        ri = r_arr[i] if en_ih else 0.0
        si = s_arr[i] if en_ica else 0.0
        ui = u_arr[i] if en_ica else 0.0
        ai = a_arr[i] if en_ia else 0.0
        bi = b_arr[i] if en_ia else 0.0
        pi = p_arr[i] if en_itca else 0.0
        qi = q_arr[i] if en_itca else 0.0
        wi = w_arr[i] if en_im else 0.0
        xi = x_arr[i] if en_nap else 0.0
        yi = y_arr[i] if en_nar else 0.0
        ji = j_arr[i] if en_nar else 0.0
        zi = z_arr[i] if en_sk else 0.0
        ca_i_val = ca_arr[i] if dyn_ca else ca_rest
        
        # Compute Nernst potential for calcium if dynamic
        if dyn_ca:
            ca_safe = min(max(ca_i_val, CA_I_MIN_M_M), CA_I_MAX_M_M)
            eca_i = nernst_ca_ion(ca_safe, ca_ext, t_kelvin)
        else:
            eca_i = 120.0
        
        # Conductance-weighted sum: g_total and e_eff = Σ g_ch * E_ch
        g_tot = gl_v[i]
        e_eff_i = gl_v[i] * el
        
        g_na = gna_v[i] * (mi ** 3) * hi
        g_tot += g_na
        e_eff_i += g_na * ena
        
        g_k_dr = gk_v[i] * (ni ** 4)
        g_tot += g_k_dr
        e_eff_i += g_k_dr * ek
        
        if en_ih:
            g_ih = gih_v[i] * ri
            g_tot += g_ih
            e_eff_i += g_ih * eih
        
        if en_ica:
            g_ca = gca_v[i] * (si ** 2) * ui
            g_tot += g_ca
            e_eff_i += g_ca * eca_i
            i_ca = g_ca * (vi - eca_i)
            if i_ca < 0.0:
                i_ca_influx[i] += -i_ca
        
        if en_ia:
            g_ia = ga_v[i] * ai * bi
            g_tot += g_ia
            e_eff_i += g_ia * ek  # IA is a K+ channel, use ek
        
        if en_itca:
            g_tca = gtca_v[i] * (pi ** 2) * qi
            g_tot += g_tca
            e_eff_i += g_tca * eca_i
            i_tca = g_tca * (vi - eca_i)
            if i_tca < 0.0:
                i_ca_influx[i] += -i_tca
        
        if en_sk:
            g_sk = gsk_v[i] * zi
            g_tot += g_sk
            e_eff_i += g_sk * ek
        
        if en_im:
            g_im = gim_v[i] * wi
            g_tot += g_im
            e_eff_i += g_im * ek
        
        if en_nap:
            g_nap = gnap_v[i] * xi
            g_tot += g_nap
            e_eff_i += g_nap * ena
        
        if en_nar:
            g_nar = gnar_v[i] * yi * ji
            g_tot += g_nar
            e_eff_i += g_nar * ena
        
        g_total[i] = g_tot
        e_eff[i] = e_eff_i
    
    return g_total, e_eff, i_ca_influx


@njit(fastmath=True, cache=True)
def run_native_loop(
    y0,       # float64[N_state]   — initial state vector
    t_sim,    # float64            — simulation duration (ms)
    dt,       # float64            — fixed time step (ms)
    dt_eval,  # float64            — output sample interval (ms)
    # ── Structured physics container ──
    physics,  # PhysicsParams
    # ── Laplacian diagonal (all values < 0) ──
    l_diag,   # float64[n_comp]
    # ── Hines topology ──
    parent_idx,              # int32[n_comp]
    hines_order,             # int32[n_comp]
    g_axial_to_parent,       # float64[n_comp]  — positive, child→parent
    g_axial_parent_to_child, # float64[n_comp]  — positive, parent←child
):
    """Fixed-step Backward-Euler Hines loop.  Returns (t_out, y_out).

    y_out shape: (N_state, N_steps_out) — columns are time-points,
    matching scipy solve_ivp.y convention used by SimulationResult.
    """
    n_comp  = physics.n_comp
    n_state = y0.shape[0]

    # ── Unpack conductance and temperature-scaling matrices ──
    (gna_v, gk_v, gl_v, gih_v, gca_v, ga_v,
     gsk_v, gtca_v, gim_v, gnap_v, gnar_v) = unpack_conductances(physics.gbar_mat, n_comp)

    (phi_na, phi_k, phi_ih, phi_ca,
     _phi_ia, _phi_tca, _phi_im, _phi_nap, _phi_nar) = unpack_temperature_scaling(
        physics.phi_mat, n_comp)
    # update_gates_analytic uses phi_k as phi_k2 for IA/IM, phi_na for NaP/NaR,
    # phi_ca for ITCa — those are already the correct physical scalings.

    # ── Channel flags ──
    en_ih   = physics.en_ih
    en_ica  = physics.en_ica
    en_ia   = physics.en_ia
    en_sk   = physics.en_sk
    en_itca = physics.en_itca
    en_im   = physics.en_im
    en_nap  = physics.en_nap
    en_nar  = physics.en_nar
    dyn_ca  = physics.dyn_ca
    dyn_atp = physics.dyn_atp

    # ── Channel counts for Langevin noise (approximate, proportional to conductance) ──
    N_Na = np.maximum(50.0, 1000.0 * gna_v / max(physics.gna_max, 1e-12))
    N_K  = np.maximum(50.0, 1000.0 * gk_v / max(physics.gk_max, 1e-12))
    sqrt_dt = np.sqrt(dt)

    # ── Reversal potentials ──
    ena = physics.ena
    ek  = physics.ek
    el  = physics.el
    eih = physics.eih
    ea  = physics.ea

    # ── Calcium / SK ──
    ca_rest  = physics.ca_rest
    ca_ext   = physics.ca_ext
    tau_ca   = physics.tau_ca
    b_ca     = physics.b_ca
    t_kelvin = physics.t_kelvin
    tau_sk   = physics.tau_sk
    mg_ext   = physics.mg_ext

    # ── ATP metabolism ──
    g_katp_max = physics.g_katp_max
    katp_kd_atp_mM = physics.katp_kd_atp_mM
    atp_max_mM = physics.atp_max_mM
    atp_synthesis_rate = physics.atp_synthesis_rate

    # ── State offsets (must mirror channels.py / rhs.py exactly) ──
    # Layout: V(n_comp), m(n_comp), h(n_comp), n_K(n_comp),
    #         [r], [s,u], [a,b], [p,q], [w], [x], [y,j], [zsk], [Ca],
    #         [vfilt_primary], [vfilt_secondary]
    off_m = n_comp
    off_h = 2 * n_comp
    off_n = 3 * n_comp
    cursor = 4 * n_comp

    off_r = cursor
    if en_ih:
        cursor += n_comp

    off_s = cursor
    off_u = cursor
    if en_ica:
        off_s = cursor;  cursor += n_comp
        off_u = cursor;  cursor += n_comp

    off_a = cursor
    off_b = cursor
    if en_ia:
        off_a = cursor;  cursor += n_comp
        off_b = cursor;  cursor += n_comp

    off_p = cursor
    off_q = cursor
    if en_itca:
        off_p = cursor;  cursor += n_comp
        off_q = cursor;  cursor += n_comp

    off_w = cursor
    if en_im:
        cursor += n_comp

    off_x = cursor
    if en_nap:
        cursor += n_comp

    off_y = cursor
    off_j = cursor
    if en_nar:
        off_y = cursor;  cursor += n_comp
        off_j = cursor;  cursor += n_comp

    off_zsk = cursor
    if en_sk:
        cursor += n_comp

    off_ca = cursor
    if dyn_ca:
        cursor += n_comp

    off_atp = cursor
    if dyn_atp:
        cursor += n_comp

    off_vfilt_primary = cursor
    if physics.use_dfilter_primary == 1:
        cursor += 1
    off_vfilt_secondary = cursor

    # ── Output sizing ──
    n_steps = int(t_sim / dt) + 1
    every   = max(1, int(dt_eval / dt + 0.5))
    n_out   = n_steps // every + 1

    t_out = np.empty(n_out, dtype=np.float64)
    y_out = np.empty((n_state, n_out), dtype=np.float64)  # (state, time) like solve_ivp.y

    # ── Working state copy ──
    y = y0.copy()

    # ── Hines system buffers (pre-allocated, reused every step) ──
    d     = np.empty(n_comp, dtype=np.float64)
    a_vec = np.empty(n_comp, dtype=np.float64)
    b_vec = np.empty(n_comp, dtype=np.float64)
    rhs   = np.empty(n_comp, dtype=np.float64)
    v_new = np.empty(n_comp, dtype=np.float64)
    i_ca_influx_v = np.zeros(n_comp, dtype=np.float64)
    
    # ── Gate array buffers for vectorized computation (pre-allocated, reused every step) ──
    r_arr_buf = np.zeros(n_comp, dtype=np.float64)
    s_arr_buf = np.zeros(n_comp, dtype=np.float64)
    u_arr_buf = np.zeros(n_comp, dtype=np.float64)
    a_arr_buf = np.zeros(n_comp, dtype=np.float64)
    b_arr_buf = np.zeros(n_comp, dtype=np.float64)
    p_arr_buf = np.zeros(n_comp, dtype=np.float64)
    q_arr_buf = np.zeros(n_comp, dtype=np.float64)
    w_arr_buf = np.zeros(n_comp, dtype=np.float64)
    x_arr_buf = np.zeros(n_comp, dtype=np.float64)
    y_arr_buf = np.zeros(n_comp, dtype=np.float64)
    j_arr_buf = np.zeros(n_comp, dtype=np.float64)
    z_arr_buf = np.zeros(n_comp, dtype=np.float64)
    ca_arr_buf = np.zeros(n_comp, dtype=np.float64)
    atp_arr_buf = np.zeros(n_comp, dtype=np.float64)

    # ── Stimulus flags (computed once) ──
    is_cond   = (physics.stype >= 4)
    is_cond_2 = (physics.stype_2 >= 4)
    e_syn     = _get_syn_reversal(physics.stype)   if is_cond   else 0.0
    e_syn_2   = _get_syn_reversal(physics.stype_2) if is_cond_2 else 0.0
    is_nmda   = (physics.stype == 5)
    is_nmda_2 = (physics.stype_2 == 5)

    out_idx = 0
    t = 0.0
    diverged = 0  # Initialize divergence flag

    for step in range(n_steps):
        # ── Record output ──
        if step % every == 0 and out_idx < n_out:
            t_out[out_idx] = t
            for s in range(n_state):
                y_out[s, out_idx] = y[s]
            out_idx += 1

        # ─────────────────────────────────────────────────────────────
        # 1. Compute Ca²⁺ influx at V_n  (needed by update_gates_analytic)
        # ─────────────────────────────────────────────────────────────
        # Extract gate arrays for vectorized computation (slicing creates views, no allocation)
        v_arr = y[:n_comp]
        m_arr = y[off_m:off_m + n_comp]
        h_arr = y[off_h:off_h + n_comp]
        n_arr = y[off_n:off_n + n_comp]
        
        # Fill pre-allocated buffers based on channel flags
        if en_ih:
            for i in range(n_comp):
                r_arr_buf[i] = y[off_r + i]
        if en_ica:
            for i in range(n_comp):
                s_arr_buf[i] = y[off_s + i]
                u_arr_buf[i] = y[off_u + i]
        if en_ia:
            for i in range(n_comp):
                a_arr_buf[i] = y[off_a + i]
                b_arr_buf[i] = y[off_b + i]
        if en_itca:
            for i in range(n_comp):
                p_arr_buf[i] = y[off_p + i]
                q_arr_buf[i] = y[off_q + i]
        if en_im:
            for i in range(n_comp):
                w_arr_buf[i] = y[off_w + i]
        if en_nap:
            for i in range(n_comp):
                x_arr_buf[i] = y[off_x + i]
        if en_nar:
            for i in range(n_comp):
                y_arr_buf[i] = y[off_y + i]
                j_arr_buf[i] = y[off_j + i]
        if en_sk:
            for i in range(n_comp):
                z_arr_buf[i] = y[off_zsk + i]
        if dyn_ca:
            for i in range(n_comp):
                ca_arr_buf[i] = y[off_ca + i]
        else:
            for i in range(n_comp):
                ca_arr_buf[i] = ca_rest
        if dyn_atp:
            for i in range(n_comp):
                atp_arr_buf[i] = y[off_atp + i]
        else:
            for i in range(n_comp):
                atp_arr_buf[i] = atp_max_mM

        # Compute calcium influx using vectorized function
        _, _, i_ca_influx_v = _compute_ionic_currents_vectorized(
            v_arr, m_arr, h_arr, n_arr,
            r_arr_buf, s_arr_buf, u_arr_buf, a_arr_buf, b_arr_buf, p_arr_buf, q_arr_buf,
            w_arr_buf, x_arr_buf, y_arr_buf, j_arr_buf, z_arr_buf,
            ca_arr_buf,
            gna_v, gk_v, gl_v, gih_v, gca_v, ga_v, gsk_v, gtca_v, gim_v, gnap_v, gnar_v,
            ena, ek, el, eih, ea,
            en_ih, en_ica, en_ia, en_sk, en_itca, en_im, en_nap, en_nar, dyn_ca,
            ca_ext, t_kelvin, ca_rest,
            n_comp,
        )

        # ─────────────────────────────────────────────────────────────
        # 2. Update gating variables analytically at V_n (exact exponential)
        # ─────────────────────────────────────────────────────────────
        update_gates_analytic(
            y, dt, n_comp,
            en_ih, en_ica, en_ia, en_sk, en_itca, en_im, en_nap, en_nar, dyn_ca,
            off_m, off_h, off_n,
            off_r, off_s, off_u, off_a, off_b,
            off_p, off_q, off_w, off_x, off_y, off_j, off_zsk, off_ca,
            phi_na, phi_k, phi_ih, phi_ca, _phi_ia,   # _phi_ia used as phi_k2 for IA/IM
            ca_rest, tau_ca, tau_sk, b_ca,
            i_ca_influx_v,
        )

        # ─────────────────────────────────────────────────────────────
        # 2.5. Add Langevin gate noise if enabled (after deterministic update)
        # NOTE: np.random.randn() in Numba-jitted code uses global NumPy RNG state.
        # For full reproducibility, set numpy seed before simulation: np.random.seed(seed)
        # ─────────────────────────────────────────────────────────────
        if physics.stoch_gating:
            for i in range(n_comp):
                vi = y[i]
                # Na m-gate noise
                am_v, bm_v = am_lut(vi), bm_lut(vi)
                m_val = y[off_m + i]
                var_m = max(0.0, am_v * (1.0 - m_val) + bm_v * m_val) / N_Na[i]
                y[off_m + i] += np.sqrt(var_m) * phi_na[i] * np.random.randn() * sqrt_dt
                
                # Na h-gate noise
                ah_v, bh_v = ah_lut(vi), bh_lut(vi)
                h_val = y[off_h + i]
                var_h = max(0.0, ah_v * (1.0 - h_val) + bh_v * h_val) / N_Na[i]
                y[off_h + i] += np.sqrt(var_h) * phi_na[i] * np.random.randn() * sqrt_dt
                
                # K n-gate noise
                an_v, bn_v = an_lut(vi), bn_lut(vi)
                n_val = y[off_n + i]
                var_n = max(0.0, an_v * (1.0 - n_val) + bn_v * n_val) / N_K[i]
                y[off_n + i] += np.sqrt(var_n) * phi_k[i] * np.random.randn() * sqrt_dt
                
                # Clamp gates to [0, 1]
                y[off_m + i] = max(0.0, min(1.0, y[off_m + i]))
                y[off_h + i] = max(0.0, min(1.0, y[off_h + i]))
                y[off_n + i] = max(0.0, min(1.0, y[off_n + i]))

        # ─────────────────────────────────────────────────────────────
        # 3. Compute stimulus currents at time t
        # ─────────────────────────────────────────────────────────────
        if physics.n_events > 0 and is_cond:
            base_current = get_event_driven_conductance(
                t, physics.stype, physics.iext,
                physics.event_times_arr, physics.n_events, physics.atau)
        else:
            base_current = get_stim_current(
                t, physics.stype, physics.iext,
                physics.t0, physics.td, physics.atau,
                physics.zap_f0_hz, physics.zap_f1_hz)

        base_current_2 = 0.0
        if physics.dual_stim_enabled == 1:
            if physics.n_events_2 > 0 and is_cond_2:
                base_current_2 = get_event_driven_conductance(
                    t, physics.stype_2, physics.iext_2,
                    physics.event_times_arr_2, physics.n_events_2, physics.atau_2)
            else:
                base_current_2 = get_stim_current(
                    t, physics.stype_2, physics.iext_2,
                    physics.t0_2, physics.td_2, physics.atau_2,
                    physics.zap_f0_hz_2, physics.zap_f1_hz_2)

        v_filt   = y[off_vfilt_primary]   if physics.use_dfilter_primary   == 1 else 0.0
        v_filt_2 = y[off_vfilt_secondary] if physics.use_dfilter_secondary == 1 else 0.0

        # ─────────────────────────────────────────────────────────────
        # 4. Build Hines linear system at V_n, gates at n+1
        #    Row i: d[i]*V[i] - a[i]*V[parent] - Σ b[c]*V[c] = rhs[i]
        # ─────────────────────────────────────────────────────────────
        # Re-fill gate buffers with updated gate values (after analytic update)
        if en_ih:
            for i in range(n_comp):
                r_arr_buf[i] = y[off_r + i]
        if en_ica:
            for i in range(n_comp):
                s_arr_buf[i] = y[off_s + i]
                u_arr_buf[i] = y[off_u + i]
        if en_ia:
            for i in range(n_comp):
                a_arr_buf[i] = y[off_a + i]
                b_arr_buf[i] = y[off_b + i]
        if en_itca:
            for i in range(n_comp):
                p_arr_buf[i] = y[off_p + i]
                q_arr_buf[i] = y[off_q + i]
        if en_im:
            for i in range(n_comp):
                w_arr_buf[i] = y[off_w + i]
        if en_nap:
            for i in range(n_comp):
                x_arr_buf[i] = y[off_x + i]
        if en_nar:
            for i in range(n_comp):
                y_arr_buf[i] = y[off_y + i]
                j_arr_buf[i] = y[off_j + i]
        if en_sk:
            for i in range(n_comp):
                z_arr_buf[i] = y[off_zsk + i]
        if dyn_ca:
            for i in range(n_comp):
                ca_arr_buf[i] = y[off_ca + i]
        else:
            for i in range(n_comp):
                ca_arr_buf[i] = ca_rest
        if dyn_atp:
            for i in range(n_comp):
                atp_arr_buf[i] = y[off_atp + i]
        else:
            for i in range(n_comp):
                atp_arr_buf[i] = atp_max_mM

        # Compute g_total and e_eff using vectorized function
        g_total_arr, e_eff_arr, _ = _compute_ionic_currents_vectorized(
            v_arr, m_arr, h_arr, n_arr,
            r_arr_buf, s_arr_buf, u_arr_buf, a_arr_buf, b_arr_buf, p_arr_buf, q_arr_buf,
            w_arr_buf, x_arr_buf, y_arr_buf, j_arr_buf, z_arr_buf,
            ca_arr_buf,
            gna_v, gk_v, gl_v, gih_v, gca_v, ga_v, gsk_v, gtca_v, gim_v, gnap_v, gnar_v,
            ena, ek, el, eih, ea,
            en_ih, en_ica, en_ia, en_sk, en_itca, en_im, en_nap, en_nar, dyn_ca,
            ca_ext, t_kelvin, ca_rest,
            n_comp,
        )
        
        # Build Hines system using pre-computed g_total and e_eff
        for i in range(n_comp):
            vi = y[i]   # V_n
            
            # Hines diagonal: Cm/dt + g_ion - L_diag[i]   (L_diag[i] < 0)
            cm_over_dt = physics.cm_v[i] / dt
            d[i]     = cm_over_dt + g_total_arr[i] - l_diag[i]
            a_vec[i] = g_axial_to_parent[i]        # positive: child→parent coupling
            b_vec[i] = g_axial_parent_to_child[i]  # positive: parent←child coupling
            
            # ── Primary stimulus contribution at compartment i ──
            i_stim_p = distributed_stimulus_current_for_comp(
                i, n_comp, base_current,
                physics.stim_comp, physics.stim_mode,
                physics.use_dfilter_primary, physics.dfilter_attenuation,
                physics.dfilter_tau_ms, v_filt,
            )
            # RHS sign: rhs[i] = Cm/dt*V_n + e_eff + I_stim_eff
            # For current injection (positive = depolarising):
            #   I_stim_eff = +I_ext (already correct from distributed_stimulus_current_for_comp)
            # For conductance-based (explicit at V_n):
            #   I_stim_eff = -g_syn*(V_n - E_syn)
            if is_cond:
                g_syn = i_stim_p
                if is_nmda:
                    g_syn *= nmda_mg_block(vi, mg_ext)
                i_stim_eff = -g_syn * (vi - e_syn)   # inward ⟹ positive contribution
                # Add g_syn to diagonal for fully implicit treatment of fast synaptic events
                d[i] += g_syn
            else:
                i_stim_eff = i_stim_p

            # ── Secondary stimulus ──
            if physics.dual_stim_enabled == 1:
                i_stim_s = distributed_stimulus_current_for_comp(
                    i, n_comp, base_current_2,
                    physics.stim_comp_2, physics.stim_mode_2,
                    physics.use_dfilter_secondary, physics.dfilter_attenuation_2,
                    physics.dfilter_tau_ms_2, v_filt_2,
                )
                if is_cond_2:
                    g2 = i_stim_s
                    if is_nmda_2:
                        g2 *= nmda_mg_block(vi, mg_ext)
                    i_stim_eff += -g2 * (vi - e_syn_2)
                    # Add g2 to diagonal for fully implicit treatment of fast synaptic events
                    d[i] += g2
                else:
                    i_stim_eff += i_stim_s

            rhs[i] = cm_over_dt * vi + e_eff_arr[i] + i_stim_eff

            # ── Additive membrane noise if enabled ──
            if physics.noise_sigma > 0.0:
                rhs[i] += (physics.noise_sigma * np.random.randn() * sqrt_dt) / dt

        # ─────────────────────────────────────────────────────────────
        # 5. Solve Hines tridiagonal system → V_{n+1}
        # ─────────────────────────────────────────────────────────────
        hines_solve(d, a_vec, b_vec, parent_idx, hines_order, rhs, v_new)

        for i in range(n_comp):
            # Check for divergence: NaN or extreme rate of change (>100mV/step)
            # Rate-based detection is more physiologically meaningful than absolute threshold
            if np.isnan(v_new[i]) or abs(v_new[i] - y[i]) > 100.0:
                diverged = 1
                break
            y[i] = v_new[i]

        if diverged == 1:
            break

        # ─────────────────────────────────────────────────────────────
        # 6. Dendritic filter states — Backward Euler
        # ─────────────────────────────────────────────────────────────
        if physics.use_dfilter_primary == 1 and physics.dfilter_tau_ms > 0.0:
            factor = dt / max(physics.dfilter_tau_ms, 1e-12)
            i_att  = base_current * physics.dfilter_attenuation
            y[off_vfilt_primary] = (y[off_vfilt_primary] + factor * i_att) / (1.0 + factor)

        if physics.use_dfilter_secondary == 1 and physics.dfilter_tau_ms_2 > 0.0:
            factor_2 = dt / max(physics.dfilter_tau_ms_2, 1e-12)
            i_att_2  = base_current_2 * physics.dfilter_attenuation_2
            y[off_vfilt_secondary] = (y[off_vfilt_secondary] + factor_2 * i_att_2) / (1.0 + factor_2)

        # ─────────────────────────────────────────────────────────────
        # 6.5. Calcium dynamics (Euler — bounded) — using i_ca_influx_v from Step 1
        # ─────────────────────────────────────────────────────────────
        if dyn_ca:
            for i in range(n_comp):
                ca_val = y[off_ca + i]
                tau_ca_safe = max(tau_ca, 1e-12)
                dca = b_ca[i] * i_ca_influx_v[i] - (ca_val - ca_rest) / tau_ca_safe
                ca_at_min = (ca_val <= CA_I_MIN_M_M and dca < 0.0)
                ca_at_max = (ca_val >= CA_I_MAX_M_M and dca > 0.0)
                if ca_at_min:
                    dca = abs(dca) * CA_DAMPING_FACTOR
                elif ca_at_max:
                    dca = -abs(dca) * CA_DAMPING_FACTOR
                y[off_ca + i] = ca_val + dt * dca

        # ATP dynamics (simple Euler integration)
        if dyn_atp:
            for i in range(n_comp):
                atp_val = y[off_atp + i]
                vi = y[i]
                mi = y[off_m + i]
                hi = y[off_h + i]

                # Pump consumption: proportional to Na+ influx
                # Na/K pump moves 3 Na+ per ATP, Faraday constant = 96485 C/mol
                # Convert µA/cm² to nmol/cm²/s: i_na (µA/cm²) * 1e-6 / (3*F) * 1e9 = i_na * 1e3 / (3*F)
                i_na = gna_v[i] * (mi ** 3) * hi * (vi - ena)
                
                # Add NaP (persistent sodium) current if enabled
                if physics.en_nap:
                    i_na += gnap_v[i] * y[off_x + i] * (vi - ena)
                
                # Add NaR (resurgent sodium) current if enabled
                if physics.en_nar:
                    i_na += gnar_v[i] * y[off_y + i] * y[off_j + i] * (vi - ena)
                
                pump_consumption = abs(i_na) * 1e3 / (3.0 * 96485.0)

                # Add calcium pump consumption if dynamic Ca is enabled
                # Ca pump moves 1 Ca2+ per ATP, z=2 for divalent ion
                if dyn_ca:
                    pump_consumption += abs(i_ca_influx_v[i]) * 1e3 / (2.0 * 96485.0)

                # ATP ODE: d[ATP]/dt = Synthesis - PumpConsumption
                datp = atp_synthesis_rate * 0.001 - pump_consumption

                # Metabolic feedback: reduce synthesis if ATP < 0.2 mM
                if atp_val < ATP_PUMP_FAILURE_THRESHOLD:
                    datp *= atp_val / ATP_PUMP_FAILURE_THRESHOLD

                # Clamp ATP to physiological bounds
                if atp_val <= ATP_MIN_M_M and datp < 0.0:
                    datp = abs(datp) * 0.5
                elif atp_val >= ATP_MAX_M_M and datp > 0.0:
                    datp = -abs(datp) * 0.5

                y[off_atp + i] += datp * dt

        t += dt

    # ── Final sample ──
    if out_idx < n_out:
        t_out[out_idx] = t
        for s in range(n_state):
            y_out[s, out_idx] = y[s]
        out_idx += 1

    return t_out[:out_idx], y_out[:, :out_idx], bool(diverged)
