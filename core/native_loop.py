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
)
from .dual_stimulation import distributed_stimulus_current_for_comp
from .hines import hines_solve, update_gates_analytic


@njit(cache=True)
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

    # ── Stimulus flags (computed once) ──
    is_cond   = (physics.stype >= 4)
    is_cond_2 = (physics.stype_2 >= 4)
    e_syn     = _get_syn_reversal(physics.stype)   if is_cond   else 0.0
    e_syn_2   = _get_syn_reversal(physics.stype_2) if is_cond_2 else 0.0
    is_nmda   = (physics.stype == 5)
    is_nmda_2 = (physics.stype_2 == 5)

    out_idx = 0
    t = 0.0

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
        for i in range(n_comp):
            vi       = y[i]
            si       = y[off_s + i] if en_ica  else 0.0
            ui       = y[off_u + i] if en_ica  else 0.0
            pi       = y[off_p + i] if en_itca else 0.0
            qi       = y[off_q + i] if en_itca else 0.0
            ca_i_val = y[off_ca + i] if dyn_ca else ca_rest

            influx = 0.0
            if dyn_ca:
                ca_safe = min(max(ca_i_val, CA_I_MIN_M_M), CA_I_MAX_M_M)
                eca_i   = nernst_ca_ion(ca_safe, ca_ext, t_kelvin)
            else:
                eca_i = 120.0

            if en_ica:
                i_ca = gca_v[i] * (si * si) * ui * (vi - eca_i)
                if i_ca < 0.0:
                    influx += -i_ca
            if en_itca:
                i_tca = gtca_v[i] * (pi * pi) * qi * (vi - eca_i)
                if i_tca < 0.0:
                    influx += -i_tca
            i_ca_influx_v[i] = influx

        # ─────────────────────────────────────────────────────────────
        # 2. Update gating variables analytically at V_n (exact exponential)
        # ─────────────────────────────────────────────────────────────
        update_gates_analytic(
            y, dt, n_comp,
            en_ih, en_ica, en_ia, en_sk, en_itca, en_im, en_nap, en_nar, dyn_ca,
            off_m, off_h, off_n,
            off_r, off_s, off_u, off_a, off_b,
            off_p, off_q, off_w, off_x, off_y, off_j, off_zsk, off_ca,
            phi_na, phi_k, phi_ih, phi_ca, phi_k,   # phi_k used as phi_k2 for IA/IM
            ca_rest, tau_ca, tau_sk, b_ca,
            i_ca_influx_v,
        )

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
        for i in range(n_comp):
            vi = y[i]   # V_n

            mi   = y[off_m   + i]
            hi   = y[off_h   + i]
            ni   = y[off_n   + i]
            ri   = y[off_r   + i] if en_ih   else 0.0
            si   = y[off_s   + i] if en_ica  else 0.0
            ui   = y[off_u   + i] if en_ica  else 0.0
            ai_g = y[off_a   + i] if en_ia   else 0.0
            bi_g = y[off_b   + i] if en_ia   else 0.0
            pi   = y[off_p   + i] if en_itca else 0.0
            qi   = y[off_q   + i] if en_itca else 0.0
            wi   = y[off_w   + i] if en_im   else 0.0
            xi   = y[off_x   + i] if en_nap  else 0.0
            yi   = y[off_y   + i] if en_nar  else 0.0
            ji   = y[off_j   + i] if en_nar  else 0.0
            zi   = y[off_zsk + i] if en_sk   else 0.0
            ca_i_val = y[off_ca + i] if dyn_ca else ca_rest

            # Conductance-weighted sum: g_total and e_eff = Σ g_ch * E_ch
            g_total = gl_v[i]
            e_eff   = gl_v[i] * el

            g_na = gna_v[i] * (mi ** 3) * hi
            g_total += g_na;  e_eff += g_na * ena

            g_k_dr = gk_v[i] * (ni ** 4)
            g_total += g_k_dr;  e_eff += g_k_dr * ek

            if en_ih:
                g_ih = gih_v[i] * ri
                g_total += g_ih;  e_eff += g_ih * eih

            if dyn_ca:
                ca_safe = min(max(ca_i_val, CA_I_MIN_M_M), CA_I_MAX_M_M)
                eca_i   = nernst_ca_ion(ca_safe, ca_ext, t_kelvin)
            else:
                eca_i = 120.0

            if en_ica:
                g_ca = gca_v[i] * (si ** 2) * ui
                g_total += g_ca;  e_eff += g_ca * eca_i

            if en_ia:
                g_ia = ga_v[i] * ai_g * bi_g
                g_total += g_ia;  e_eff += g_ia * ea

            if en_itca:
                g_tca = gtca_v[i] * (pi ** 2) * qi
                g_total += g_tca;  e_eff += g_tca * eca_i

            if en_sk:
                g_sk = gsk_v[i] * zi
                g_total += g_sk;  e_eff += g_sk * ek

            if en_im:
                g_im = gim_v[i] * wi
                g_total += g_im;  e_eff += g_im * ek

            if en_nap:
                g_nap = gnap_v[i] * xi
                g_total += g_nap;  e_eff += g_nap * ena

            if en_nar:
                g_nar = gnar_v[i] * yi * ji
                g_total += g_nar;  e_eff += g_nar * ena

            # Hines diagonal: Cm/dt + g_ion - L_diag[i]   (L_diag[i] < 0)
            cm_over_dt = physics.cm_v[i] / dt
            d[i]     = cm_over_dt + g_total - l_diag[i]
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
                else:
                    i_stim_eff += i_stim_s

            rhs[i] = cm_over_dt * vi + e_eff + i_stim_eff

        # ─────────────────────────────────────────────────────────────
        # 5. Solve Hines tridiagonal system → V_{n+1}
        # ─────────────────────────────────────────────────────────────
        hines_solve(d, a_vec, b_vec, parent_idx, hines_order, rhs, v_new)

        for i in range(n_comp):
            y[i] = v_new[i]

        # ─────────────────────────────────────────────────────────────
        # 6. Dendritic filter states — Backward Euler
        # ─────────────────────────────────────────────────────────────
        if physics.use_dfilter_primary == 1 and physics.dfilter_tau_ms > 0.0:
            factor = dt / physics.dfilter_tau_ms
            i_att  = base_current * physics.dfilter_attenuation
            y[off_vfilt_primary] = (y[off_vfilt_primary] + factor * i_att) / (1.0 + factor)

        if physics.use_dfilter_secondary == 1 and physics.dfilter_tau_ms_2 > 0.0:
            factor_2 = dt / physics.dfilter_tau_ms_2
            i_att_2  = base_current_2 * physics.dfilter_attenuation_2
            y[off_vfilt_secondary] = (y[off_vfilt_secondary] + factor_2 * i_att_2) / (1.0 + factor_2)

        t += dt

    # ── Final sample ──
    if out_idx < n_out:
        t_out[out_idx] = t
        for s in range(n_state):
            y_out[s, out_idx] = y[s]
        out_idx += 1

    return t_out[:out_idx], y_out[:, :out_idx]
