"""Native Hines solver loop — v11.0.

Fixed-step Backward-Euler integrator with O(N) Hines tree solver.
All stimulus and RHS calculations are inline to avoid call overhead.
"""
from __future__ import annotations
import numpy as np
from numba import njit, prange
from numba.types import int64, float64

# Local calcium constants (avoid rhs.py dependency)
CA_I_MIN_M_M = 1e-9
CA_I_MAX_M_M = 10.0
CA_DAMPING_FACTOR = 0.5

# Import only what we actually need
from .rhs import (
    get_stim_current, get_event_driven_conductance,
    _get_syn_reversal, compute_ionic_currents_scalar,
    nernst_ca_ion, nmda_mg_block,
)
from .hines import hines_solve, update_gates_analytic
from .dual_stimulation import distributed_stimulus_current_for_comp


@njit(cache=True)
def run_native_loop(
    y0: float64[:], t_final: float64, dt: float64, dt_out: float64,
    n_comp: int64, parent_idx: int64[:], hines_order: int64[:],
    # Environment
    t_kelvin: float64, ca_ext: float64, ca_rest: float64,
    tau_ca: float64, mg_ext: float64, tau_sk: float64,
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
    e_syn: float64, mg_ext: float64,
    # State offsets
    off_s: int, off_m: int, off_h: int, off_n: int, off_r: int, off_w: int,
    off_w_im: int, off_ca: int, off_zsk: int, off_itca: int, off_im: int, off_nap: int, off_nar: int,
    off_vfilt_primary: int, off_vfilt_secondary: int,
) -> tuple[float64[:], float64[:]]:
    """Run native Hines solver with fixed Backward-Euler integration.
    
    Returns (t_out, y_out) arrays compatible with SciPy solve_ivp interface.
    """
    # Constants
    F_CONST = 96485.0  # C/mol
    R_GAS = 8.314     # J/(mol·K)
    
    # Extract state offsets
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
    
    # Time stepping
    n_steps = int(np.ceil(t_final / dt))
    n_out = max(1, int(np.ceil(t_final / dt_out)))
    every = max(1, int(np.ceil(dt_out / dt)))
    
    # Output arrays
    t_out = np.empty(n_out, dtype=np.float64)
    y_out = np.empty((n_out, y0.shape[0]), dtype=np.float64)
    
    # Copy initial conditions
    y = y0.copy()
    out_idx = 0
    
    # Pre-compute stimulus parameters
    s_map = {
        "const": 0, "pulse": 1, "alpha": 2, "ou_noise": 3,
        "AMPA": 4, "NMDA": 5, "GABAA": 6, "GABAB": 7,
        "Kainate": 8, "Nicotinic": 9, "zap": 10,
    }
    stim_mode_map = {"soma": 0, "ais": 1, "dendritic_filtered": 2}
    
    t = 0.0
    step = 0
    
    # ── Main time loop ──
    while step < n_steps:
        # ── 0. Record output ──
        if step % every == 0 and out_idx < n_out:
            t_out[out_idx] = t
            for s in range(y0.shape[0]):
                y_out[s, out_idx] = y[s]
            out_idx += 1
        
        # ── 1. Compute i_ca_influx at V_n (for Ca dynamics) ──
        for i in range(n_comp):
            vi = y[i]
            si = y[off_s + i] if en_sk else 0.0
            ui = y[off_u + i] if en_itca else 0.0
            pi = y[off_p + i] if en_im else 0.0
            qi = y[off_q + i] if en_nap else 0.0
            wi = y[off_w + i] if en_nar else 0.0
            xi = y[off_xi + i] if en_im else 0.0
            yi = y[off_yi + i] if en_im else 0.0
            ji = y[off_ji + i] if en_im else 0.0
            zi = y[off_zsk + i] if en_sk else 0.0
            ca_i_val = y[off_ca + i] if dyn_ca else ca_rest
            
            # Compute reversal potentials
            e_syn = _get_syn_reversal(si, ui, pi, qi, wi, xi, yi, ji, zi)
            
            # Apply NMDA Mg block if needed
            if is_nmda:
                g_syn = i_syn  # conductance-based
                g_syn *= nmda_mg_block(vi, mg_ext)
                i_syn_eff = -g_syn * (vi - e_syn)
            else:
                i_syn_eff = i_syn
            
            # Compute ionic currents
            i_ion, i_ca_influx = compute_ionic_currents_scalar(
                vi, si, ui, pi, qi, wi, xi, yi, ji, zi,
                en_ih, en_ica, en_ia, en_sk, en_itca, en_im, en_nap, en_nar, dyn_ca,
                np.zeros(n_comp), np.zeros(n_comp), np.zeros(n_comp), np.zeros(n_comp),
                np.zeros(n_comp), np.zeros(n_comp), np.zeros(n_comp), np.zeros(n_comp),
                np.zeros(n_comp), np.zeros(n_comp), np.zeros(n_comp),
                ena, ek, el, eih, ea,
                ca_i_val, ca_ext, ca_rest, t_kelvin,
            )
            
            # Total membrane current (excluding synaptic)
            i_ion_total = i_ion - i_syn_eff
            
            # ── 2. Update gating variables analytically ──
            update_gates_analytic(
                y, dt,
                m_inf, h_inf, n_inf,
                alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n,
                off_m, off_h, off_n, off_r, off_w, off_w_im,
                m_idx, h_idx, n_idx, r_idx, w_idx, w_im_idx,
            )
        
        # ── 3. Build Hines system ──
        for i in range(n_comp):
            vi = y[i]
            cm_over_dt = cm_v[i] / dt
            d[i] = cm_over_dt + g_total - l_diag[i]   # l_diag[i] < 0

            # Coupling off-diagonals (positive values from g_axial arrays)
            a[i] = g_axial_to_parent[i]
            b[i] = g_axial_parent_to_child[i]

            # Read dendritic filter states
            v_filt_1 = y[off_vfilt_primary] if use_dfilter_primary == 1 else 0.0
            v_filt_2 = y[off_vfilt_secondary] if use_dfilter_secondary == 1 else 0.0

            # Primary stimulus for compartment i
            i_stim_primary = distributed_stimulus_current_for_comp(
                i, n_comp, base_current, stim_comp, stim_mode,
                use_dfilter_primary, dfilter_attenuation, dfilter_tau_ms, v_filt_1
            )
            if is_conductance_based:
                g_syn = i_stim_primary
                if is_nmda: g_syn *= nmda_mg_block(vi, mg_ext)
                i_stim_eff = -g_syn * (vi - e_syn)
            else:
                i_stim_eff = i_stim_primary

            # Secondary stimulus for compartment i
            if dual_stim_enabled == 1:
                i_stim_sec = distributed_stimulus_current_for_comp(
                    i, n_comp, base_current_2, stim_comp_2, stim_mode_2,
                    use_dfilter_secondary, dfilter_attenuation_2, dfilter_tau_ms_2, v_filt_2
                )
                if is_cond_2:
                    g2 = i_stim_sec
                    if is_nmda_2:
                        g2 *= nmda_mg_block(vi, mg_ext)
                    i_stim_eff -= g2 * (vi - e_syn_2)
                else:
                    i_stim_eff += i_stim_sec

            # CRITICAL FIX: Add i_stim_eff (Do not subtract!)
            rhs[i] = cm_over_dt * vi + e_eff + i_stim_eff

        # ── 4. Solve Hines system → V_{n+1} ──
        hines_solve(d, a, b, parent_idx, hines_order, rhs, v_new)

        # Write V_{n+1} back into state vector
        for i in range(n_comp):
            y[i] = v_new[i]

        # ── 5. Update Dendritic Filter States (Backward Euler) ──
        if use_dfilter_primary == 1 and dfilter_tau_ms > 0.0:
            i_att = base_current * dfilter_attenuation
            factor = dt / dfilter_tau_ms
            y[off_vfilt_primary] = (y[off_vfilt_primary] + factor * i_att) / (1.0 + factor)

        if use_dfilter_secondary == 1 and dfilter_tau_ms_2 > 0.0:
            i_att_2 = base_current_2 * dfilter_attenuation_2
            factor_2 = dt / dfilter_tau_ms_2
            y[off_vfilt_secondary] = (y[off_vfilt_secondary] + factor_2 * i_att_2) / (1.0 + factor_2)

        t += dt
        step += 1
    
    return t_out, y_out
