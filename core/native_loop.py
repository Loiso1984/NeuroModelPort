"""Native Hines time-loop — v11.0.

Pure @njit Backward-Euler time integration using the O(N) Hines solver.
Called by NeuronSolver.run_native(); returns (t_arr, y_arr) in the same
layout as SciPy solve_ivp.y  so SimulationResult is unchanged.

Layout of y vector (mirrors rhs.py / channels.py):
  [V(0..n-1), m, h, n_K, r?, s?,u?, a?,b?, p?,q?, w?, x?, y?,j?, zsk?, Ca?]
  Dendritic-filter state appended if use_dfilter_primary/secondary == 1.

RHS notation (per compartment i):
  d[i]  = Cm_i/dt + g_total_ion_i + |L_diag_i|
  a[i]  = g_axial_to_parent[i]        (own row, child→parent coupling)
  b[i]  = g_axial_parent_to_child[i]  (parent's row, parent←child coupling)
  rhs[i]= Cm_i/dt*V_n_i + g_eff_rhs_i + I_stim_i
"""
from __future__ import annotations

import numpy as np
from numba import njit, float64, int32

from .rhs import (
    get_stim_current, get_event_driven_conductance,
    _get_syn_reversal, compute_ionic_currents_scalar,
    nernst_ca_ion,
    CA_I_MIN_M_M, CA_I_MAX_M_M, CA_DAMPING_FACTOR,
)
from .hines import hines_solve, update_gates_analytic


@njit(cache=True)
def run_native_loop(
    y0,                    # float64[N_state]   — initial state vector
    t_sim,                 # float64            — simulation duration (ms)
    dt,                    # float64            — fixed time step (ms)
    dt_eval,               # float64            — output sample interval (ms)
    # ── Morphology ──
    n_comp,                # int32
    cm_v,                  # float64[n_comp]    — membrane capacitance µF/cm²
    # Laplacian diagonal (negative values)
    l_diag,                # float64[n_comp]    — L_csr diagonal (< 0)
    # Hines coupling arrays
    parent_idx,            # int32[n_comp]
    hines_order,           # int32[n_comp]
    g_axial_to_parent,     # float64[n_comp]    — a[i]
    g_axial_parent_to_child,  # float64[n_comp] — b[i]
    # ── Conductance vectors (shape [n_comp]) ──
    gna_v, gk_v, gl_v, gih_v, gca_v, ga_v, gsk_v, gtca_v, gim_v, gnap_v, gnar_v,
    # ── Reversal potentials ──
    ena, ek, el, eih, ea,
    # ── Temperature scaling vectors (shape [n_comp]) ──
    phi_na, phi_k, phi_ih, phi_ca, phi_nap, phi_nar,
    # ── Channel enable flags ──
    en_ih, en_ica, en_ia, en_sk, en_itca, en_im, en_nap, en_nar, dyn_ca,
    # ── State offsets ──
    off_m, off_h, off_n,
    off_r, off_s, off_u,
    off_a, off_b, off_p, off_q,
    off_w, off_x, off_y, off_j,
    off_zsk, off_ca,
    off_vfilt_primary, off_vfilt_secondary,
    use_dfilter_primary, use_dfilter_secondary,
    dfilter_attenuation, dfilter_tau_ms,
    dfilter_attenuation_2, dfilter_tau_ms_2,
    # ── Calcium ──
    ca_rest, ca_ext, tau_ca, b_ca, t_kelvin,
    # ── SK ──
    tau_sk,
    # ── Primary stimulus ──
    stype, iext, t0, td, atau, zap_f0_hz, zap_f1_hz,
    event_times_arr, n_events,
    stim_comp, stim_mode,
    # ── Secondary stimulus ──
    dual_stim_enabled,
    stype_2, iext_2, t0_2, td_2, atau_2, zap_f0_hz_2, zap_f1_hz_2,
    event_times_arr_2, n_events_2,
    stim_comp_2, stim_mode_2,
):
    """Fixed-step Backward-Euler Hines loop.  Returns (t_out, y_out).

    y_out shape: (N_state, N_steps_out).  Layout matches solve_ivp.y.
    """
    n_state = y0.shape[0]
    n_steps = int(t_sim / dt) + 1
    # Output decimation
    every = max(1, int(dt_eval / dt + 0.5))
    n_out = n_steps // every + 1

    # ── Pre-allocate output arrays ──
    t_out = np.empty(n_out, dtype=np.float64)
    y_out = np.empty((n_state, n_out), dtype=np.float64)

    # ── Working state copy ──
    y = y0.copy()

    # ── Hines system buffers (re-used each step — no alloc in loop) ──
    d    = np.empty(n_comp, dtype=np.float64)
    a    = np.empty(n_comp, dtype=np.float64)
    b    = np.empty(n_comp, dtype=np.float64)
    rhs  = np.empty(n_comp, dtype=np.float64)
    v_new = np.empty(n_comp, dtype=np.float64)
    i_ca_influx_v = np.zeros(n_comp, dtype=np.float64)

    # ── Stimulus flags ──
    is_conductance_based = (stype >= 4)
    is_cond_2            = (stype_2 >= 4)
    e_syn   = _get_syn_reversal(stype)   if is_conductance_based else 0.0
    e_syn_2 = _get_syn_reversal(stype_2) if is_cond_2 else 0.0
    is_nmda   = (stype == 5)
    is_nmda_2 = (stype_2 == 5)

    out_idx = 0
    t = 0.0

    for step in range(n_steps):
        # ── Record output ──
        if step % every == 0 and out_idx < n_out:
            t_out[out_idx] = t
            for s in range(n_state):
                y_out[s, out_idx] = y[s]
            out_idx += 1

        # ── 1. Compute stimulus current (scalar) ──
        if n_events > 0 and is_conductance_based:
            base_current = get_event_driven_conductance(
                t, stype, iext, event_times_arr, n_events, atau)
        else:
            base_current = get_stim_current(
                t, stype, iext, t0, td, atau, zap_f0_hz, zap_f1_hz)

        base_current_2 = 0.0
        if dual_stim_enabled == 1:
            if n_events_2 > 0 and is_cond_2:
                base_current_2 = get_event_driven_conductance(
                    t, stype_2, iext_2, event_times_arr_2, n_events_2, atau_2)
            else:
                base_current_2 = get_stim_current(
                    t, stype_2, iext_2, t0_2, td_2, atau_2, zap_f0_hz_2, zap_f1_hz_2)

        # ── 2. Compute i_ca_influx at V_n (for Ca²⁺ dynamics) ──
        for i in range(n_comp):
            vi = y[i]
            si = y[off_s + i] if en_ica else 0.0
            ui = y[off_u + i] if en_ica else 0.0
            pi = y[off_p + i] if en_itca else 0.0
            qi = y[off_q + i] if en_itca else 0.0
            ca_i_val = y[off_ca + i] if dyn_ca else ca_rest

            influx = 0.0
            if dyn_ca:
                ca_safe = min(max(ca_i_val, CA_I_MIN_M_M), CA_I_MAX_M_M)
                eca_i = nernst_ca_ion(ca_safe, ca_ext, t_kelvin)
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

        # ── 3. Update gating variables analytically at V_n ──
        update_gates_analytic(
            y, dt, n_comp,
            en_ih, en_ica, en_ia, en_sk, en_itca, en_im, en_nap, en_nar, dyn_ca,
            off_m, off_h, off_n,
            off_r, off_s, off_u, off_a, off_b,
            off_p, off_q, off_w, off_x, off_y, off_j, off_zsk, off_ca,
            phi_na, phi_k, phi_ih, phi_ca, phi_k,  # phi_k2 = phi_k
            ca_rest, tau_ca, tau_sk, b_ca,
            i_ca_influx_v,
        )

        # ── 4. Build Hines system at V_n, gates_n+1 ──
        for i in range(n_comp):
            vi = y[i]   # V_n

            # Read updated gates
            mi = y[off_m + i]
            hi = y[off_h + i]
            ni = y[off_n + i]
            ri = y[off_r + i] if en_ih else 0.0
            si = y[off_s + i] if en_ica else 0.0
            ui = y[off_u + i] if en_ica else 0.0
            ai_g = y[off_a + i] if en_ia else 0.0
            bi_g = y[off_b + i] if en_ia else 0.0
            pi = y[off_p + i] if en_itca else 0.0
            qi = y[off_q + i] if en_itca else 0.0
            wi = y[off_w + i] if en_im else 0.0
            xi = y[off_x + i] if en_nap else 0.0
            yi = y[off_y + i] if en_nar else 0.0
            ji = y[off_j + i] if en_nar else 0.0
            zi = y[off_zsk + i] if en_sk else 0.0
            ca_i_val = y[off_ca + i] if dyn_ca else ca_rest

            # Conductance-weighted sum g_total * V and g_total * E for linearization
            # I_ion = g_total * V - g_eff_rhs
            # => (Cm/dt + g_total - L_diag) * V_{n+1} = Cm/dt*V_n + g_eff_rhs + I_stim
            g_total = gl_v[i]
            e_eff   = gl_v[i] * el

            g_na = gna_v[i] * (mi**3) * hi
            g_total += g_na;  e_eff += g_na * ena

            g_k_dr = gk_v[i] * (ni**4)
            g_total += g_k_dr;  e_eff += g_k_dr * ek

            if en_ih:
                g_ih = gih_v[i] * ri
                g_total += g_ih;  e_eff += g_ih * eih

            if dyn_ca:
                ca_safe = min(max(ca_i_val, CA_I_MIN_M_M), CA_I_MAX_M_M)
                eca_i = nernst_ca_ion(ca_safe, ca_ext, t_kelvin)
            else:
                eca_i = 120.0

            if en_ica:
                g_ca = gca_v[i] * (si**2) * ui
                g_total += g_ca;  e_eff += g_ca * eca_i

            if en_ia:
                g_ia = ga_v[i] * ai_g * bi_g
                g_total += g_ia;  e_eff += g_ia * ea

            if en_itca:
                g_tca = gtca_v[i] * (pi**2) * qi
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

            # Axial diagonal contribution: -L_diag[i] > 0
            cm_over_dt = cm_v[i] / dt
            d[i] = cm_over_dt + g_total - l_diag[i]   # l_diag[i] < 0

            # Coupling off-diagonals (positive values from g_axial arrays)
            a[i] = g_axial_to_parent[i]
            b[i] = g_axial_parent_to_child[i]

            # Stimulus current for compartment i
            i_stim = 0.0
            if stim_mode == 0:              # soma injection
                if i == stim_comp:
                    if is_conductance_based:
                        e_s = e_syn
                        i_stim = base_current * (vi - e_s)
                    else:
                        i_stim = base_current
            else:                           # AIS / dendritic (broadcast)
                if is_conductance_based:
                    i_stim = base_current * (vi - e_syn)
                else:
                    i_stim = base_current

            if dual_stim_enabled == 1:
                if stim_mode_2 == 0:
                    if i == stim_comp_2:
                        if is_cond_2:
                            i_stim += base_current_2 * (vi - e_syn_2)
                        else:
                            i_stim += base_current_2
                else:
                    if is_cond_2:
                        i_stim += base_current_2 * (vi - e_syn_2)
                    else:
                        i_stim += base_current_2

            rhs[i] = cm_over_dt * vi + e_eff - i_stim

        # ── 5. Solve Hines system → V_{n+1} ──
        hines_solve(d, a, b, parent_idx, hines_order, rhs, v_new)

        # Write V_{n+1} back into state vector
        for i in range(n_comp):
            y[i] = v_new[i]

        t += dt

    # Final sample
    if out_idx < n_out:
        t_out[out_idx] = t
        for s in range(n_state):
            y_out[s, out_idx] = y[s]
        out_idx += 1

    return t_out[:out_idx], y_out[:, :out_idx]
