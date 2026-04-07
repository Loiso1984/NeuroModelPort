"""Native Hines solver loop — v11.2.

Fixed-step Backward-Euler integrator with TRUE O(N) Hines tree solver.
Handles branched morphologies correctly with Exponential Euler for gates.
"""
from __future__ import annotations
import numpy as np
from numba import njit
from numba.types import int64, float64

# Import Hines kernels
from .hines import hines_solve, _gate_step

# Physics constants
F_CONST = 96485.0
R_GAS = 8.314
TEMP_ZERO = 273.15
Q10_IA = 3.0


@njit(cache=True)
def _compute_ionic_currents(
    v, m, h, n, r, s, u, a, b, p, q, w, x, y_gate, j, z_sk, ca_i,
    en_ih, en_ica, en_ia, en_sk, en_itca, en_im, en_nap, en_nar,
    gna, gk, gl, gih, gca, ga, gsk, gtca, gim, gnap, gnar,
    ena, ek, el, eih, ea, e_ca, e_sk, e_tca, e_im, e_nap, e_nar
):
    """Compute ionic currents for a single compartment."""
    g_na = gna * (m ** 3) * h
    g_k = gk * (n ** 4)
    g_l = gl
    
    g_ih_eff = gih * r if en_ih else 0.0
    g_ca_eff = gca * (s ** 2) * u if en_ica else 0.0
    g_a_eff = ga * a * b if en_ia else 0.0
    g_tca_eff = gtca * (p ** 2) * q if en_itca else 0.0
    g_im_eff = gim * w if en_im else 0.0
    g_nap_eff = gnap * x if en_nap else 0.0
    g_nar_eff = gnar * y_gate * j if en_nar else 0.0
    g_sk_eff = gsk * z_sk if en_sk else 0.0
    
    i_ion = (g_na * (v - ena) + g_k * (v - ek) + g_l * (v - el) +
             g_ih_eff * (v - eih) + g_ca_eff * (v - e_ca) +
             g_a_eff * (v - ea) + g_tca_eff * (v - e_tca) +
             g_im_eff * (v - e_im) + g_nap_eff * (v - e_nap) +
             g_nar_eff * (v - e_nar) + g_sk_eff * (v - e_sk))
    
    i_ca = 0.0
    if en_ica:
        i_ca += g_ca_eff * (v - e_ca)
    if en_itca:
        i_ca += g_tca_eff * (v - e_tca)
    
    return i_ion, i_ca, g_na + g_k + g_l + g_ih_eff + g_ca_eff + g_a_eff + g_tca_eff + g_im_eff + g_nap_eff + g_nar_eff + g_sk_eff


@njit(cache=True)
def run_native_loop(
    y0: float64[:], t_final: float64, dt: float64, dt_out: float64,
    n_comp: int64, parent_idx: int64[:], hines_order: int64[:],
    # Morphology arrays
    g_axial_to_parent: float64[:], g_axial_parent_to_child: float64[:],
    cm_v: float64[:],
    # Conductance arrays
    gna_v: float64[:], gk_v: float64[:], gl_v: float64[:],
    gih_v: float64[:], gca_v: float64[:], ga_v: float64[:],
    gsk_v: float64[:], gtca_v: float64[:], gim_v: float64[:],
    gnap_v: float64[:], gnar_v: float64[:],
    # Reversal potentials
    ena: float64, ek: float64, el: float64, eih: float64,
    ea: float64, e_ca: float64, e_sk: float64,
    e_tca: float64, e_im: float64, e_nap: float64, e_nar: float64,
    # Environment
    t_kelvin: float64, ca_ext: float64, ca_rest: float64,
    tau_ca: float64, mg_ext: float64, tau_sk: float64,
    # Channel enable flags
    en_ih: int64, en_ica: int64, en_ia: int64, en_sk: int64,
    en_itca: int64, en_im: int64, en_nap: int64, en_nar: int64,
    dyn_ca: int64,
    # Primary stimulus
    stype: int64, iext: float64, t0: float64, td: float64, atau: float64,
    zap_f0_hz: float64, zap_f1_hz: float64, stim_comp: int64, stim_mode: int64,
    use_dfilter_primary: int64, dfilter_attenuation: float64, dfilter_tau_ms: float64,
    # Secondary stimulus
    dual_stim_enabled: int64, stype_2: int64, iext_2: float64, t0_2: float64,
    td_2: float64, atau_2: float64, zap_f0_hz_2: float64, zap_f1_hz_2: float64,
    stim_comp_2: int64, stim_mode_2: int64,
    use_dfilter_secondary: int64, dfilter_attenuation_2: float64, dfilter_tau_ms_2: float64,
    # Event-driven
    event_times_arr: float64[:], n_events: int64,
    # Synaptic
    is_conductance_based: int64, is_nmda: int64,
    e_syn: float64,
) -> tuple:
    """Run native Hines solver with fixed Backward-Euler integration."""
    # Time stepping
    n_steps = int(np.ceil(t_final / dt))
    n_out = max(1, int(np.ceil(t_final / dt_out)))
    every = max(1, int(np.ceil(dt_out / dt)))
    n_states = y0.shape[0]
    
    t_out = np.empty(n_out, dtype=np.float64)
    y_out = np.empty((n_out, n_states), dtype=np.float64)
    
    y = y0.copy()
    
    # Hines arrays
    d = np.empty(n_comp, dtype=np.float64)
    a = np.empty(n_comp, dtype=np.float64)
    b = np.empty(n_comp, dtype=np.float64)
    rhs = np.empty(n_comp, dtype=np.float64)
    v_new = np.empty(n_comp, dtype=np.float64)
    
    # State offsets
    off_v = 0
    off_m = n_comp
    off_h = off_m + n_comp
    off_n = off_h + n_comp
    curr_off = off_n + n_comp
    
    off_r = curr_off if en_ih else -1
    if en_ih: curr_off += n_comp
    
    off_s = curr_off if en_ica else -1
    off_u = curr_off + n_comp if en_ica else -1
    if en_ica: curr_off += 2 * n_comp
    
    off_a = curr_off if en_ia else -1
    off_b = curr_off + n_comp if en_ia else -1
    if en_ia: curr_off += 2 * n_comp
    
    off_zsk = curr_off if en_sk else -1
    if en_sk: curr_off += n_comp
    
    off_p = curr_off if en_itca else -1
    off_q = curr_off + n_comp if en_itca else -1
    if en_itca: curr_off += 2 * n_comp
    
    off_w_im = curr_off if en_im else -1
    if en_im: curr_off += n_comp
    
    off_x = curr_off if en_nap else -1
    if en_nap: curr_off += n_comp
    
    off_y = curr_off if en_nar else -1
    off_j = curr_off + n_comp if en_nar else -1
    if en_nar: curr_off += 2 * n_comp
    
    off_ca = curr_off if dyn_ca else -1
    if dyn_ca: curr_off += n_comp
    
    off_vfilt_primary = curr_off if use_dfilter_primary else -1
    if use_dfilter_primary: curr_off += 1
    
    off_vfilt_secondary = curr_off if use_dfilter_secondary else -1
    
    out_idx = 0
    t = 0.0
    step = 0
    
    # IA temperature scaling reference
    t_ref = TEMP_ZERO + 6.3
    phi_ia = Q10_IA ** ((t_kelvin - t_ref) / 10.0) if en_ia else 1.0
    
    while step < n_steps:
        # Record output
        if step % every == 0 and out_idx < n_out:
            t_out[out_idx] = t
            for s in range(n_states):
                y_out[out_idx, s] = y[s]
            out_idx += 1
        
        # Update gates
        for i in range(n_comp):
            vi = y[off_v + i]
            
            # m, h, n gates
            am = 0.32 * (vi + 54.0) / (1.0 - np.exp(-(vi + 54.0) / 4.0))
            bm = 0.28 * (vi + 27.0) / (np.exp((vi + 27.0) / 5.0) - 1.0)
            y[off_m + i] = _gate_step(y[off_m + i], dt, am, bm)
            
            ah = 0.128 * np.exp(-(vi + 50.0) / 18.0)
            bh = 4.0 / (1.0 + np.exp(-(vi + 27.0) / 5.0))
            y[off_h + i] = _gate_step(y[off_h + i], dt, ah, bh)
            
            an = 0.032 * (vi + 52.0) / (1.0 - np.exp(-(vi + 52.0) / 5.0))
            bn = 0.5 * np.exp(-(vi + 57.0) / 40.0)
            y[off_n + i] = _gate_step(y[off_n + i], dt, an, bn)
            
            if en_ih:
                ar = 1.0 / (1.0 + np.exp((vi + 80.0) / 8.0))
                br = 1.0 / (1.0 + np.exp(-(vi + 80.0) / 8.0))
                y[off_r + i] = _gate_step(y[off_r + i], dt, ar, br)
            
            if en_ica:
                as_ = 0.1 * (vi + 40.0) / (1.0 - np.exp(-(vi + 40.0) / 10.0))
                bs = 0.5 * np.exp(-(vi + 15.0) / 12.0)
                y[off_s + i] = _gate_step(y[off_s + i], dt, as_, bs)
                au = 0.01 * np.exp(-(vi + 50.0) / 27.0)
                bu = 0.5 / (1.0 + np.exp(-(vi + 15.0) / 12.0))
                y[off_u + i] = _gate_step(y[off_u + i], dt, au, bu)
            
            if en_ia:
                aa = 0.0667 * (vi + 31.0) / (1.0 - np.exp(-(vi + 31.0) / 5.0)) * phi_ia
                ba = 0.125 * np.exp(-(vi + 43.0) / 18.0) * phi_ia
                y[off_a + i] = _gate_step(y[off_a + i], dt, aa, ba)
                ab = 0.0028 / (np.exp((vi + 43.0) / 18.0) + 1.0)
                bb = 0.15 / (np.exp(-(vi + 68.0) / 7.0) + 1.0)
                y[off_b + i] = _gate_step(y[off_b + i], dt, ab, bb)
            
            if en_itca:
                ap = 0.1 * (vi + 56.0) / (1.0 - np.exp(-(vi + 56.0) / 10.0))
                bp = 0.5 * np.exp(-(vi + 15.0) / 12.0)
                y[off_p + i] = _gate_step(y[off_p + i], dt, ap, bp)
                aq = 0.01 * np.exp(-(vi + 58.0) / 27.0)
                bq = 0.5 / (1.0 + np.exp(-(vi + 15.0) / 12.0))
                y[off_q + i] = _gate_step(y[off_q + i], dt, aq, bq)
            
            if en_im:
                aw = 0.003 / (1.0 + np.exp(-(vi + 40.0) / 10.0))
                bw = 0.003 / (1.0 + np.exp((vi + 40.0) / 10.0))
                y[off_w_im + i] = _gate_step(y[off_w_im + i], dt, aw, bw)
            
            if en_nap:
                ax = 0.32 * (vi + 58.0) / (1.0 - np.exp(-(vi + 58.0) / 4.0))
                bx = 0.28 * (vi + 25.0) / (np.exp((vi + 25.0) / 5.0) - 1.0)
                y[off_x + i] = _gate_step(y[off_x + i], dt, ax, bx)
            
            if en_nar:
                ay = 0.5 * (vi + 40.0) / (1.0 - np.exp(-(vi + 40.0) / 5.0))
                by = 0.5 * np.exp(-(vi + 10.0) / 10.0)
                y[off_y + i] = _gate_step(y[off_y + i], dt, ay, by)
                aj = 0.01 * np.exp(-(vi + 60.0) / 20.0)
                bj = 0.5 / (1.0 + np.exp(-(vi + 10.0) / 10.0))
                y[off_j + i] = _gate_step(y[off_j + i], dt, aj, bj)
        
        # Build Hines system
        for i in range(n_comp):
            mi = y[off_m + i]
            hi = y[off_h + i]
            ni = y[off_n + i]
            ri = y[off_r + i] if en_ih else 0.0
            si = y[off_s + i] if en_ica else 0.0
            ui = y[off_u + i] if en_ica else 0.0
            ai = y[off_a + i] if en_ia else 0.0
            bi = y[off_b + i] if en_ia else 0.0
            pi = y[off_p + i] if en_itca else 0.0
            qi = y[off_q + i] if en_itca else 0.0
            wi = y[off_w_im + i] if en_im else 0.0
            xi = y[off_x + i] if en_nap else 0.0
            yi = y[off_y + i] if en_nar else 0.0
            ji = y[off_j + i] if en_nar else 0.0
            zi = y[off_zsk + i] if en_sk else 0.0
            
            i_ion, i_ca, g_total = _compute_ionic_currents(
                y[off_v + i], mi, hi, ni, ri, si, ui, ai, bi, pi, qi, wi, xi, yi, ji, zi, 0.0,
                en_ih, en_ica, en_ia, en_sk, en_itca, en_im, en_nap, en_nar,
                gna_v[i], gk_v[i], gl_v[i], gih_v[i], gca_v[i], ga_v[i],
                gsk_v[i], gtca_v[i], gim_v[i], gnap_v[i], gnar_v[i],
                ena, ek, el, eih, ea, e_ca, e_sk, e_tca, e_im, e_nap, e_nar
            )
            
            cm_over_dt = cm_v[i] / dt
            d[i] = cm_over_dt + g_total
            
            if parent_idx[i] >= 0:
                d[i] += g_axial_to_parent[i]
            d[i] += g_axial_parent_to_child[i]
            
            a[i] = -g_axial_to_parent[i] if parent_idx[i] >= 0 else 0.0
            b[i] = -g_axial_parent_to_child[i]
            
            vi = y[off_v + i]
            
            # Simple stimulus
            i_stim = 0.0
            if stype == 0:  # const
                if t >= t0 and t < t0 + td:
                    i_stim = iext
            elif stype == 1:  # pulse
                if t >= t0 and t < t0 + td:
                    i_stim = iext
            
            rhs[i] = cm_over_dt * vi - i_ion + i_stim
        
        # Solve
        hines_solve(d, a, b, parent_idx, hines_order, rhs, v_new)
        
        for i in range(n_comp):
            y[off_v + i] = v_new[i]
        
        # Calcium dynamics
        if dyn_ca:
            for i in range(n_comp):
                ca_old = y[off_ca + i]
                ca_new = ca_old + dt * ((ca_rest - ca_old) / tau_ca)
                y[off_ca + i] = max(ca_new, 1e-9)
        
        # Dendritic filters
        if use_dfilter_primary and dfilter_tau_ms > 0.0:
            f = dt / dfilter_tau_ms
            y[off_vfilt_primary] = (y[off_vfilt_primary] + f * iext * dfilter_attenuation) / (1.0 + f)
        
        if use_dfilter_secondary and dfilter_tau_ms_2 > 0.0:
            f = dt / dfilter_tau_ms_2
            y[off_vfilt_secondary] = (y[off_vfilt_secondary] + f * iext_2 * dfilter_attenuation_2) / (1.0 + f)
        
        t += dt
        step += 1
    
    return t_out, y_out
