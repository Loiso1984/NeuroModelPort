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
  a[i]   = g_axial_to_parent[i]                    (positive, child->parent)
  b[i]   = g_axial_parent_to_child[i]              (positive, parent←child)
  rhs[i] = Cm_i/dt*V_n + e_eff_i + I_stim_eff_i

Hines system (solved by hines.py::hines_solve):
  d[i]*V[i] - a[i]*V[parent] - sum_c b[c]*V[c]  = rhs[i]
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .physics_params import PhysicsParams, unpack_conductances, unpack_env_params, unpack_temperature_scaling
from .rhs import (
    get_stim_current, get_event_driven_conductance,
    _get_syn_reversal, nernst_ca_ion, nernst_na_ion, nernst_k_ion, nmda_mg_block,
    CA_I_MIN_M_M, CA_I_MAX_M_M,
    compute_na_k_pump_current,
    compute_metabolism_and_pump,
    compute_ionic_conductances_scalar,
)
from .dual_stimulation import distributed_stimulus_current_for_comp
from .hines import hines_solve, update_gates_analytic
from .kinetics import am_lut, bm_lut, ah_lut, bh_lut, an_lut, bn_lut

# _NA_PUMP_FACTOR and _CA_PUMP_FACTOR now live in rhs.py (used by compute_metabolism_and_pump)


@njit(cache=True)
def set_numba_random_seed(seed: int) -> None:
    """Seed Numba's internal NumPy RNG state for reproducible stochastic runs."""
    np.random.seed(seed)


# NOTE: _compute_ionic_currents_vectorized REMOVED v11.6
# Replaced by unified compute_ionic_conductances_scalar in rhs.py
# Native loop now uses scalar indexing directly for zero-allocation performance


@njit(cache=True)
def _extract_gate_scalars(y, i, off_m, off_h, off_n):
    """Extract HH gate scalars from state vector - zero-slice indexing."""
    return y[off_m + i], y[off_h + i], y[off_n + i]


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
    g_axial_to_parent,       # float64[n_comp]  — positive, child->parent
    g_axial_parent_to_child, # float64[n_comp]  - positive, parent->child
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

    # ── Calcium / SK / ATP environment pack ──
    (
        t_kelvin,
        ca_ext,
        ca_rest,
        tau_ca,
        mg_ext,
        tau_sk,
        im_speed_multiplier,
        g_katp_max,
        katp_kd_atp_mM,
        atp_max_mM,
        atp_synthesis_rate,
        na_i_rest_mM,
        na_ext_mM,
        k_i_mM,
        k_o_rest_mM,
        ion_drift_gain,
        k_o_clearance_tau_ms,
    ) = unpack_env_params(physics.env_params)
    b_ca = physics.b_ca
    nmda_mg_block_mM = physics.nmda_mg_block_mM

    # ── Shared state offsets ──
    offsets = physics.state_offsets
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
    off_atp = int(offsets.off_atp)
    off_na_i = int(offsets.off_na_i)
    off_k_o = int(offsets.off_k_o)
    off_ifilt_primary = int(offsets.off_ifilt_primary)
    off_ifilt_secondary = int(offsets.off_ifilt_secondary)

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
    i_ca_dummy_v = np.zeros(n_comp, dtype=np.float64)  # Required by update_gates_analytic
    # v11.6: Ionic conductance buffers for unified scalar helper
    g_total_arr = np.empty(n_comp, dtype=np.float64)  # Total membrane conductance per compartment
    e_eff_arr = np.empty(n_comp, dtype=np.float64)    # Conductance-weighted reversal potential
    # NOTE: v11.6 - removed gate array buffers, using scalar indexing directly
    # Minimal buffers for Nernst potentials (avoid recomputing in second loop)
    ena_arr_buf = np.empty(n_comp, dtype=np.float64)
    ek_arr_buf = np.empty(n_comp, dtype=np.float64)

    # ── Stimulus flags (computed once) ──
    is_cond   = (physics.stype >= 4)
    is_cond_2 = (physics.stype_2 >= 4)
    e_syn     = _get_syn_reversal(physics.stype, physics.e_rev_syn_primary, physics.e_rev_syn_secondary)   if is_cond   else 0.0
    e_syn_2   = _get_syn_reversal(physics.stype_2, physics.e_rev_syn_primary, physics.e_rev_syn_secondary) if is_cond_2 else 0.0
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
        # 1. Update gating variables analytically at V_n (exact exponential)
        # ─────────────────────────────────────────────────────────────
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
            im_speed_multiplier,
            ca_rest, tau_ca, tau_sk, b_ca,
            i_ca_dummy_v,
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
        # 4. Build Hines linear system at V_n, gates at n+1
        #    Row i: d[i]*V[i] - a[i]*V[parent] - Σ b[c]*V[c] = rhs[i]
        # ─────────────────────────────────────────────────────────────
        # v11.6: Zero-slice optimization - use scalar indexing directly
        # All ionic conductances computed per-compartment using unified helper
        # This eliminates array buffer allocations and guarantees physics consistency
        # with the SciPy BDF solver path (both use compute_ionic_conductances_scalar)

        # Compute g_total, e_eff, i_ca_influx per compartment using unified scalar helper
        for i in range(n_comp):
            # Extract all gate scalars directly from state vector (NO SLICES)
            vi = y[i]
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
            wi = y[off_w + i] if en_im else 0.0
            xi = y[off_x + i] if en_nap else 0.0
            yi = y[off_y + i] if en_nar else 0.0
            ji = y[off_j + i] if en_nar else 0.0
            zi = y[off_zsk + i] if en_sk else 0.0
            ca_i_val = y[off_ca + i] if dyn_ca else ca_rest
            
            # Compute Nernst potentials if dynamic metabolism
            if dyn_atp:
                na_i_val = y[off_na_i + i]
                k_o_val = y[off_k_o + i]
                ena_i = nernst_na_ion(na_i_val, na_ext_mM, t_kelvin)
                ek_i = nernst_k_ion(k_i_mM, k_o_val, t_kelvin)
            else:
                ena_i = ena
                ek_i = ek
            
            # Store for second loop (Hines RHS building)
            ena_arr_buf[i] = ena_i
            ek_arr_buf[i] = ek_i
            
            # Unified scalar conductance computation (SINGLE SOURCE OF TRUTH)
            g_total_arr[i], e_eff_arr[i], i_ca_influx_v[i] = compute_ionic_conductances_scalar(
                vi, mi, hi, ni,
                ri, si, ui, ai, bi, pi, qi, wi, xi, yi, ji, zi,
                en_ih, en_ica, en_ia, en_sk, en_itca, en_im, en_nap, en_nar, dyn_ca,
                gna_v[i], gk_v[i], gl_v[i], gih_v[i], gca_v[i], ga_v[i], gsk_v[i], gtca_v[i], gim_v[i], gnap_v[i], gnar_v[i],
                ena_i, ek_i, el, eih,
                ca_i_val, ca_ext, ca_rest, t_kelvin,
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
                physics.zap_f0_hz, physics.zap_f1_hz, physics.zap_rise_ms)

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
                    physics.zap_f0_hz_2, physics.zap_f1_hz_2, physics.zap_rise_ms_2)

        i_filt   = y[off_ifilt_primary]   if physics.use_dfilter_primary   == 1 else 0.0
        i_filt_2 = y[off_ifilt_secondary] if physics.use_dfilter_secondary == 1 else 0.0
        
        # Build Hines system using pre-computed g_total and e_eff
        for i in range(n_comp):
            vi = y[i]   # V_n
            ena_i = ena_arr_buf[i]
            ek_i = ek_arr_buf[i]
            
            # Hines diagonal: Cm/dt + g_ion - L_diag[i]   (L_diag[i] < 0)
            cm_over_dt = physics.cm_v[i] / dt
            d[i]     = cm_over_dt + g_total_arr[i] - l_diag[i]
            a_vec[i] = g_axial_to_parent[i]        # positive: child->parent coupling
            b_vec[i] = g_axial_parent_to_child[i]  # positive: parent←child coupling
            
            # ── Primary stimulus contribution at compartment i ──
            i_stim_p = distributed_stimulus_current_for_comp(
                i, n_comp, base_current,
                physics.stim_comp, physics.stim_mode,
                physics.use_dfilter_primary, physics.dfilter_attenuation,
                physics.dfilter_tau_ms, i_filt,
            )
            # RHS sign: rhs[i] = Cm/dt*V_n + e_eff + I_stim_eff
            # For current injection (positive = depolarising):
            #   I_stim_eff = +I_ext (already correct from distributed_stimulus_current_for_comp)
            # For conductance-based (explicit at V_n):
            #   I_stim_eff = -g_syn*(V_n - E_syn)
            if is_cond:
                g_syn = i_stim_p
                if is_nmda:
                    g_syn *= nmda_mg_block(vi, mg_ext, nmda_mg_block_mM)
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
                    physics.dfilter_tau_ms_2, i_filt_2,
                )
                if is_cond_2:
                    g2 = i_stim_s
                    if is_nmda_2:
                        g2 *= nmda_mg_block(vi, mg_ext, nmda_mg_block_mM)
                    i_stim_eff += -g2 * (vi - e_syn_2)
                    # Add g2 to diagonal for fully implicit treatment of fast synaptic events
                    d[i] += g2
                else:
                    i_stim_eff += i_stim_s

            katp_rhs = 0.0
            if dyn_atp:
                atp_val = y[off_atp + i]
                atp_ratio = atp_val / max(katp_kd_atp_mM, 1e-12)
                g_katp = g_katp_max / (1.0 + atp_ratio * atp_ratio)
                d[i] += g_katp
                katp_rhs = g_katp * ek_i

            # Na/K pump current (MM kinetics) - uses [Na+]i, [K+]o, [ATP]
            pump_atp_val = y[off_atp + i] if dyn_atp else atp_max_mM
            na_i_val = y[off_na_i + i] if dyn_atp else na_i_rest_mM
            k_o_val = y[off_k_o + i] if dyn_atp else k_o_rest_mM
            i_pump = compute_na_k_pump_current(na_i_val, k_o_val, pump_atp_val)

            rhs[i] = cm_over_dt * vi + e_eff_arr[i] + katp_rhs + i_stim_eff - i_pump

            # ── Additive membrane noise if enabled ──
            if physics.noise_sigma > 0.0:
                rhs[i] += (physics.noise_sigma * np.random.randn() * sqrt_dt) / dt

        # ─────────────────────────────────────────────────────────────
        # 5. Solve Hines tridiagonal system -> V_{n+1}
        # ─────────────────────────────────────────────────────────────
        hines_solve(d, a_vec, b_vec, parent_idx, hines_order, rhs, v_new)

        for i in range(n_comp):
            # Divergence guard should key off clearly non-physiological voltages,
            # not merely large step-to-step changes during fast spikes.
            if np.isnan(v_new[i]) or abs(v_new[i]) > 300.0:
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
            y[off_ifilt_primary] = (y[off_ifilt_primary] + factor * i_att) / (1.0 + factor)

        if physics.use_dfilter_secondary == 1 and physics.dfilter_tau_ms_2 > 0.0:
            factor_2 = dt / max(physics.dfilter_tau_ms_2, 1e-12)
            i_att_2  = base_current_2 * physics.dfilter_attenuation_2
            y[off_ifilt_secondary] = (y[off_ifilt_secondary] + factor_2 * i_att_2) / (1.0 + factor_2)

        # ─────────────────────────────────────────────────────────────
        # 6.5. Calcium dynamics — Semi-implicit (Backward Euler, unconditionally stable)
        #   d[Ca]/dt = B_Ca * I_Ca_influx - ([Ca] - [Ca]_rest) / tau_Ca
        #   Rearranged: influx = B_Ca*I_Ca + Ca_rest/tau,  decay = 1/tau
        #   [Ca]_{n+1} = ([Ca]_n + dt*influx) / (1 + dt*decay)
        # ─────────────────────────────────────────────────────────────
        if dyn_ca:
            for i in range(n_comp):
                ca_val = y[off_ca + i]
                tau_ca_safe = max(tau_ca, 1e-12)
                influx = b_ca[i] * i_ca_influx_v[i] + ca_rest / tau_ca_safe
                decay_rate = 1.0 / tau_ca_safe
                y[off_ca + i] = (ca_val + dt * influx) / (1.0 + dt * decay_rate)

        # ATP/Na_i/K_o dynamics — uses unified helper from rhs.py
        # ATP & Na_i: Forward Euler with soft bounds inside compute_metabolism_and_pump
        # K_o: Semi-implicit for clearance decay (unconditionally stable)
        if dyn_atp:
            for i in range(n_comp):
                atp_val = y[off_atp + i]
                na_i_val = y[off_na_i + i]
                k_o_val = y[off_k_o + i]
                vi = y[i]
                ena_i = nernst_na_ion(na_i_val, na_ext_mM, t_kelvin)
                ek_i = nernst_k_ion(k_i_mM, k_o_val, t_kelvin)

                mi = y[off_m + i]
                hi = y[off_h + i]
                ni = y[off_n + i]
                xi_v = y[off_x + i] if en_nap else 0.0
                yi_v = y[off_y + i] if en_nar else 0.0
                ji_v = y[off_j + i] if en_nar else 0.0
                ai_v = y[off_a + i] if en_ia else 0.0
                bi_v = y[off_b + i] if en_ia else 0.0
                zi_v = y[off_zsk + i] if en_sk else 0.0
                wi_v = y[off_w + i] if en_im else 0.0
                _i_pump, _i_katp, datp, dnai, dko = compute_metabolism_and_pump(
                    vi, mi, hi, ni, xi_v, yi_v, ji_v, ai_v, bi_v, zi_v, wi_v,
                    gna_v[i], gk_v[i], ga_v[i], gsk_v[i], gim_v[i], gnap_v[i], gnar_v[i],
                    ena_i, ek_i,
                    en_nap, en_nar, en_ia, en_sk, en_im, dyn_ca,
                    g_katp_max, katp_kd_atp_mM,
                    atp_val, atp_synthesis_rate,
                    na_i_val, k_o_val, k_o_rest_mM,
                    ion_drift_gain, k_o_clearance_tau_ms,
                    i_ca_influx_v[i],
                )

                # ATP: Forward Euler
                y[off_atp + i] = atp_val + datp * dt
                # Na_i: Forward Euler
                y[off_na_i + i] = na_i_val + dnai * dt
                # K_o: Forward Euler (затухание уже учтено внутри dko)
                y[off_k_o + i] = k_o_val + dko * dt

        t += dt

    # ── Final sample ──
    if out_idx < n_out:
        t_out[out_idx] = t
        for s in range(n_state):
            y_out[s, out_idx] = y[s]
        out_idx += 1

    return t_out[:out_idx], y_out[:, :out_idx], bool(diverged)

