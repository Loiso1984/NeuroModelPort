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


def make_lle_subspace_mask(
    n_comp: int,
    state_offsets,
    include_v: bool = True,
    include_gates: list | None = None,
    include_ca: bool = False,
    include_atp: bool = False,
    include_nai: bool = False,
    include_ko: bool = False,
    include_ifilt: bool = False,
) -> np.ndarray:
    """Create boolean mask for LLE custom subspace analysis.

    Parameters
    ----------
    n_comp : int
        Number of compartments
    state_offsets : StateOffsets or object with offset attributes
        Offset indices for state variables (off_m, off_h, off_n, off_ca, etc.)
    include_v : bool
        Include voltage compartments (default: True)
    include_gates : list[str] | None
        List of gate names to include: ["m", "h", "n", "r", "s", "u", "a", "b",
        "p", "q", "w", "x", "y", "j", "zsk"]. None = include all gates.
    include_ca : bool
        Include calcium concentration variables
    include_atp : bool
        Include ATP concentration variables
    include_nai : bool
        Include intracellular sodium variables
    include_ko : bool
        Include extracellular potassium variables
    include_ifilt : bool
        Include dendritic filter states

    Returns
    -------
    np.ndarray
        Boolean mask array of shape (n_state,) where True = include in LLE

    Examples
    --------
    >>> # Analyze only voltage and sodium gates for Na+ sensitivity
    >>> mask = make_lle_subspace_mask(
    ...     n_comp=2,
    ...     state_offsets=offsets,
    ...     include_v=True,
    ...     include_gates=["m", "h"],  # Na activation and inactivation
    ... )
    """
    n_state = state_offsets.n_state
    mask = np.zeros(n_state, dtype=np.bool_)

    if include_v:
        mask[:n_comp] = True  # Voltage compartments

    # Helper to set gate range
    def _set_gate_range(off_start, n_comp_val):
        mask[off_start:off_start + n_comp_val] = True

    # Determine which gates to include
    all_gates = ["m", "h", "n", "r", "s", "u", "a", "b", "p", "q", "w", "x", "y", "j", "zsk"]
    if include_gates is None:
        include_gates = all_gates

    # Map gate names to offsets
    gate_map = {
        "m": state_offsets.off_m,
        "h": state_offsets.off_h,
        "n": state_offsets.off_n,
        "r": state_offsets.off_r,
        "s": state_offsets.off_s,
        "u": state_offsets.off_u,
        "a": state_offsets.off_a,
        "b": state_offsets.off_b,
        "p": state_offsets.off_p,
        "q": state_offsets.off_q,
        "w": state_offsets.off_w,
        "x": state_offsets.off_x,
        "y": state_offsets.off_y,
        "j": state_offsets.off_j,
        "zsk": state_offsets.off_zsk,
    }

    for gate in include_gates:
        if gate in gate_map and gate_map[gate] != -1:
            _set_gate_range(gate_map[gate], n_comp)

    # Metabolic and other variables
    if include_ca and state_offsets.off_ca != -1:
        _set_gate_range(state_offsets.off_ca, n_comp)
    if include_atp and state_offsets.off_atp != -1:
        _set_gate_range(state_offsets.off_atp, n_comp)
    if include_nai and state_offsets.off_na_i != -1:
        _set_gate_range(state_offsets.off_na_i, n_comp)
    if include_ko and state_offsets.off_k_o != -1:
        _set_gate_range(state_offsets.off_k_o, n_comp)
    if include_ifilt:
        if state_offsets.off_ifilt_primary != -1:
            mask[state_offsets.off_ifilt_primary] = True
        if state_offsets.off_ifilt_secondary != -1:
            mask[state_offsets.off_ifilt_secondary] = True

    return mask


def make_lle_weights(mask: np.ndarray, n_comp: int, state_offsets) -> np.ndarray:
    """Create auto-normalized weights for LLE custom subspace.

    Automatically assigns weights based on typical value ranges:
    - Voltage (~10 mV range): weight = 0.1
    - Gates (0-1 range): weight = 1.0
    - Calcium (~1e-5 mM range): weight = 1e5
    - ATP (~1-5 mM range): weight = 0.2
    - Na_i (~5-15 mM range): weight = 0.1
    - K_o (~3-5 mM range): weight = 0.2

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask from make_lle_subspace_mask()
    n_comp : int
        Number of compartments
    state_offsets : StateOffsets
        Offset indices

    Returns
    -------
    np.ndarray
        Float64 weights array (same shape as mask)
    """
    weights = np.ones(mask.shape[0], dtype=np.float64)

    if not np.any(mask):
        return weights

    # Voltage compartments: normalize ~10 mV to ~1
    v_range = np.arange(n_comp)
    weights[v_range] = 0.1

    # Gates: already 0-1, keep weight 1.0
    # No change needed for gate ranges

    # Calcium: normalize ~1e-5 mM to ~1
    if state_offsets.off_ca != -1:
        ca_range = np.arange(state_offsets.off_ca, state_offsets.off_ca + n_comp)
        weights[ca_range] = 1e5

    # ATP: normalize ~5 mM to ~1
    if state_offsets.off_atp != -1:
        atp_range = np.arange(state_offsets.off_atp, state_offsets.off_atp + n_comp)
        weights[atp_range] = 0.2

    # Na_i: normalize ~10 mM to ~1
    if state_offsets.off_na_i != -1:
        nai_range = np.arange(state_offsets.off_na_i, state_offsets.off_na_i + n_comp)
        weights[nai_range] = 0.1

    # K_o: normalize ~5 mM to ~1
    if state_offsets.off_k_o != -1:
        ko_range = np.arange(state_offsets.off_k_o, state_offsets.off_k_o + n_comp)
        weights[ko_range] = 0.2

    # Zero out weights for excluded variables
    weights[~mask] = 0.0

    return weights


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
    # ── LLE (Benettin algorithm) ──
    calc_lle=False,          # bool — enable dual-trajectory Lyapunov computation
    lle_delta=1e-6,          # float64 — perturbation amplitude for Benettin
    lle_t_evolve=1.0,        # float64 — re-orthonormalization interval (ms)
    lle_subspace_mode=0,     # int — 0=v_only, 1=v_and_gates, 2=full_state, 3=custom (Numba-compatible)
    lle_custom_mask=None,    # bool[N_state] or None — which variables to include (custom mode)
    lle_weights=None,        # float64[N_state] or None — optional weighting factors
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
    eca = physics.eca  # Static calcium reversal potential (used when dyn_ca=False)

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
        pump_max_capacity,
        km_na,
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

    # ── Working state copies ──
    y = y0.copy()

    # ── LLE trajectory management (Benettin algorithm) ──
    n_traj = 2 if calc_lle else 1
    if calc_lle:
        y_pert = y0.copy()
        y_pert[0] += lle_delta  # perturb V_soma
    else:
        y_pert = np.empty(0, dtype=np.float64)  # unused dummy

    lle_accum = 0.0
    lle_count = 0
    lle_out = np.zeros(n_out, dtype=np.float64)
    lle_t_next_renorm = lle_t_evolve if calc_lle else t_sim + 1.0

    # ── Hines system buffers (pre-allocated, reused every step per trajectory) ──
    d     = np.empty(n_comp, dtype=np.float64)
    a_vec = np.empty(n_comp, dtype=np.float64)
    b_vec = np.empty(n_comp, dtype=np.float64)
    rhs   = np.empty(n_comp, dtype=np.float64)
    v_new = np.empty(n_comp, dtype=np.float64)
    # i_ca_influx: 2D when LLE enabled to prevent cross-talk between trajectories
    i_ca_influx_2d = np.zeros((n_traj, n_comp), dtype=np.float64)
    i_ca_dummy_v = np.zeros(n_comp, dtype=np.float64)  # Required by update_gates_analytic
    # v11.6: Ionic conductance buffers for unified scalar helper
    g_total_arr = np.empty(n_comp, dtype=np.float64)  # Total membrane conductance per compartment
    e_eff_arr = np.empty(n_comp, dtype=np.float64)    # Conductance-weighted reversal potential
    # Minimal buffers for Nernst potentials (avoid recomputing in second loop)
    ena_arr_buf = np.empty(n_comp, dtype=np.float64)
    ek_arr_buf = np.empty(n_comp, dtype=np.float64)
    # v11.7: Pre-allocated noise buffers for Langevin stochastic gating (same noise to both trajectories)
    noise_m_arr = np.empty(n_comp, dtype=np.float64)
    noise_h_arr = np.empty(n_comp, dtype=np.float64)
    noise_n_arr = np.empty(n_comp, dtype=np.float64)

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
        # ── Record output (main trajectory only) ──
        if step % every == 0 and out_idx < n_out:
            t_out[out_idx] = t
            for s in range(n_state):
                y_out[s, out_idx] = y[s]
            # Note: LLE is recorded only at final output section after all computations
            out_idx += 1

        # ─────────────────────────────────────────────────────────────
        # 0. Compute stimulus currents at time t (SAME for both trajectories —
        #    external input is independent of state)
        # ─────────────────────────────────────────────────────────────
        if physics.n_events > 0 and is_cond:
            base_current = get_event_driven_conductance(
                t, physics.stype, physics.iext,
                physics.event_times_arr, physics.n_events, physics.atau)
        else:
            base_current = get_stim_current(
                t, physics.stype, physics.iext,
                physics.t0, physics.td, physics.atau,
                physics.zap_f0_hz, physics.zap_f1_hz, physics.zap_rise_ms,
                physics.zap_win_t, physics.zap_win_g, physics.zap_win_size)

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
                    physics.zap_f0_hz_2, physics.zap_f1_hz_2, physics.zap_rise_ms_2,
                    physics.zap_win_t_2, physics.zap_win_g_2, physics.zap_win_size_2)

        # ─────────────────────────────────────────────────────────────
        # PRECOMPUTE Nernst potentials (Variant A: use start-of-step values)
        # Concentrations are identical for both trajectories at step start
        # ─────────────────────────────────────────────────────────────
        if dyn_atp:
            for i in range(n_comp):
                na_i_val = y[off_na_i + i]  # Main trajectory values at step start
                k_o_val = y[off_k_o + i]
                ena_arr_buf[i] = nernst_na_ion(na_i_val, na_ext_mM, t_kelvin)
                ek_arr_buf[i] = nernst_k_ion(k_i_mM, k_o_val, t_kelvin)
        else:
            for i in range(n_comp):
                ena_arr_buf[i] = ena
                ek_arr_buf[i] = ek

        # ─────────────────────────────────────────────────────────────
        # DUAL TRAJECTORY LOOP — traj 0 = main, traj 1 = perturbed (LLE)
        # ─────────────────────────────────────────────────────────────
        for traj_idx in range(n_traj):
            y_active = y if traj_idx == 0 else y_pert

            # ── 1. Update gating variables analytically at V_n ──
            update_gates_analytic(
                y_active, dt, n_comp,
                en_ih, en_ica, en_ia, en_sk, en_itca, en_im, en_nap, en_nar, dyn_ca,
                off_m, off_h, off_n,
                off_r, off_s, off_u, off_a, off_b,
                off_p, off_q, off_w, off_x, off_y, off_j, off_zsk, off_ca,
                phi_na, phi_k, phi_ih, phi_ca, _phi_ia,
                im_speed_multiplier,
                ca_rest, tau_ca, tau_sk, b_ca,
                i_ca_dummy_v,
            )

            # ── 2. Langevin gate noise (same noise applied to both trajectories) ──
            # v11.7: Generate and apply noise INSIDE trajectory loop for LLE correctness
            # Both trajectories must experience identical noise to measure chaos, not noise divergence
            # Noise variance depends on updated gate values (after update_gates_analytic)
            if physics.stoch_gating:
                # Generate noise for main trajectory (traj_idx == 0) and reuse for perturbed
                if traj_idx == 0:
                    for i in range(n_comp):
                        vi = y_active[i]
                        am_v, bm_v = am_lut(vi), bm_lut(vi)
                        m_val = y_active[off_m + i]
                        var_m = max(0.0, am_v * (1.0 - m_val) + bm_v * m_val) / N_Na[i]
                        noise_m_arr[i] = np.sqrt(var_m) * phi_na[i] * np.random.randn() * sqrt_dt

                        ah_v, bh_v = ah_lut(vi), bh_lut(vi)
                        h_val = y_active[off_h + i]
                        var_h = max(0.0, ah_v * (1.0 - h_val) + bh_v * h_val) / N_Na[i]
                        noise_h_arr[i] = np.sqrt(var_h) * phi_na[i] * np.random.randn() * sqrt_dt

                        an_v, bn_v = an_lut(vi), bn_lut(vi)
                        n_val = y_active[off_n + i]
                        var_n = max(0.0, an_v * (1.0 - n_val) + bn_v * n_val) / N_K[i]
                        noise_n_arr[i] = np.sqrt(var_n) * phi_k[i] * np.random.randn() * sqrt_dt

                        # Apply and clamp noise inline
                        y_active[off_m + i] = max(0.0, min(1.0, y_active[off_m + i] + noise_m_arr[i]))
                        y_active[off_h + i] = max(0.0, min(1.0, y_active[off_h + i] + noise_h_arr[i]))
                        y_active[off_n + i] = max(0.0, min(1.0, y_active[off_n + i] + noise_n_arr[i]))
                else:
                    # Apply SAME pre-generated noise to perturbed trajectory (identical clamping)
                    for i in range(n_comp):
                        y_active[off_m + i] = max(0.0, min(1.0, y_active[off_m + i] + noise_m_arr[i]))
                        y_active[off_h + i] = max(0.0, min(1.0, y_active[off_h + i] + noise_h_arr[i]))
                        y_active[off_n + i] = max(0.0, min(1.0, y_active[off_n + i] + noise_n_arr[i]))

            # ── 3. Compute ionic conductances per-compartment (zero-slice) ──
            for i in range(n_comp):
                vi = y_active[i]
                mi = y_active[off_m + i]
                hi = y_active[off_h + i]
                ni = y_active[off_n + i]
                ri = y_active[off_r + i] if en_ih else 0.0
                si = y_active[off_s + i] if en_ica else 0.0
                ui = y_active[off_u + i] if en_ica else 0.0
                ai = y_active[off_a + i] if en_ia else 0.0
                bi = y_active[off_b + i] if en_ia else 0.0
                pi = y_active[off_p + i] if en_itca else 0.0
                qi = y_active[off_q + i] if en_itca else 0.0
                wi = y_active[off_w + i] if en_im else 0.0
                xi = y_active[off_x + i] if en_nap else 0.0
                yi = y_active[off_y + i] if en_nar else 0.0
                ji = y_active[off_j + i] if en_nar else 0.0
                zi = y_active[off_zsk + i] if en_sk else 0.0
                ca_i_val = y_active[off_ca + i] if dyn_ca else ca_rest

                # Use precomputed Nernst potentials (optimized, same for both trajectories)
                ena_i = ena_arr_buf[i]
                ek_i = ek_arr_buf[i]

                g_total_arr[i], e_eff_arr[i], i_ca_influx_2d[traj_idx, i] = compute_ionic_conductances_scalar(
                    vi, mi, hi, ni,
                    ri, si, ui, ai, bi, pi, qi, wi, xi, yi, ji, zi,
                    en_ih, en_ica, en_ia, en_sk, en_itca, en_im, en_nap, en_nar, dyn_ca,
                    gna_v[i], gk_v[i], gl_v[i], gih_v[i], gca_v[i], ga_v[i], gsk_v[i], gtca_v[i], gim_v[i], gnap_v[i], gnar_v[i],
                    ena_i, ek_i, el, eih, eca,
                    ca_i_val, ca_ext, ca_rest, t_kelvin,
                )

            # ── 4. Build Hines linear system ──
            i_filt   = y_active[off_ifilt_primary]   if physics.use_dfilter_primary   == 1 else 0.0
            i_filt_2 = y_active[off_ifilt_secondary] if physics.use_dfilter_secondary == 1 else 0.0

            for i in range(n_comp):
                vi = y_active[i]
                ena_i = ena_arr_buf[i]
                ek_i = ek_arr_buf[i]

                cm_over_dt = physics.cm_v[i] / dt
                d[i]     = cm_over_dt + g_total_arr[i] - l_diag[i]
                a_vec[i] = g_axial_to_parent[i]
                b_vec[i] = g_axial_parent_to_child[i]

                i_stim_p = distributed_stimulus_current_for_comp(
                    i, n_comp, base_current,
                    physics.stim_comp, physics.stim_mode,
                    physics.use_dfilter_primary, physics.dfilter_attenuation,
                    physics.dfilter_tau_ms, i_filt,
                )
                if is_cond:
                    g_syn = i_stim_p
                    if is_nmda:
                        g_syn *= nmda_mg_block(vi, mg_ext, nmda_mg_block_mM)
                    i_stim_eff = -g_syn * (vi - e_syn)
                    d[i] += g_syn
                else:
                    i_stim_eff = i_stim_p

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
                        d[i] += g2
                    else:
                        i_stim_eff += i_stim_s

                katp_rhs = 0.0
                if dyn_atp:
                    atp_val = y_active[off_atp + i]
                    atp_ratio = atp_val / max(katp_kd_atp_mM, 1e-12)
                    g_katp = g_katp_max / (1.0 + atp_ratio * atp_ratio)
                    d[i] += g_katp
                    katp_rhs = g_katp * ek_i

                pump_atp_val = y_active[off_atp + i] if dyn_atp else atp_max_mM
                na_i_val = y_active[off_na_i + i] if dyn_atp else na_i_rest_mM
                k_o_val = y_active[off_k_o + i] if dyn_atp else k_o_rest_mM
                i_pump = compute_na_k_pump_current(na_i_val, k_o_val, pump_atp_val, pump_max_capacity, km_na)

                rhs[i] = cm_over_dt * vi + e_eff_arr[i] + katp_rhs + i_stim_eff - i_pump

                # Additive membrane noise (main trajectory only)
                if physics.noise_sigma > 0.0 and traj_idx == 0:
                    rhs[i] += (physics.noise_sigma * np.random.randn() * sqrt_dt) / dt

            # ── 5. Solve Hines tridiagonal system -> V_{n+1} ──
            hines_solve(d, a_vec, b_vec, parent_idx, hines_order, rhs, v_new)

            traj_diverged = False
            for i in range(n_comp):
                if np.isnan(v_new[i]) or abs(v_new[i]) > 300.0:
                    traj_diverged = True
                    break
                y_active[i] = v_new[i]

            if traj_diverged:
                if traj_idx == 0:
                    diverged = 1
                break  # break trajectory loop

            # ── 6. Dendritic filter states — Backward Euler ──
            if physics.use_dfilter_primary == 1 and physics.dfilter_tau_ms > 0.0:
                factor = dt / max(physics.dfilter_tau_ms, 1e-12)
                i_att  = base_current * physics.dfilter_attenuation
                y_active[off_ifilt_primary] = (y_active[off_ifilt_primary] + factor * i_att) / (1.0 + factor)

            if physics.use_dfilter_secondary == 1 and physics.dfilter_tau_ms_2 > 0.0:
                factor_2 = dt / max(physics.dfilter_tau_ms_2, 1e-12)
                i_att_2  = base_current_2 * physics.dfilter_attenuation_2
                y_active[off_ifilt_secondary] = (y_active[off_ifilt_secondary] + factor_2 * i_att_2) / (1.0 + factor_2)

            # ── 7. Calcium dynamics — Semi-implicit ──
            if dyn_ca:
                for i in range(n_comp):
                    ca_val = y_active[off_ca + i]
                    tau_ca_safe = max(tau_ca, 1e-12)
                    influx = b_ca[i] * i_ca_influx_2d[traj_idx, i] + ca_rest / tau_ca_safe
                    decay_rate = 1.0 / tau_ca_safe
                    y_active[off_ca + i] = (ca_val + dt * influx) / (1.0 + dt * decay_rate)

            # ── 8. ATP/Na_i/K_o dynamics ──
            if dyn_atp:
                # Reuse cached Nernst potentials from conductance loop (step 3)
                for i in range(n_comp):
                    atp_val = y_active[off_atp + i]
                    na_i_val = y_active[off_na_i + i]
                    k_o_val = y_active[off_k_o + i]
                    vi = y_active[i]
                    # Use cached Nernst potentials (computed in step 3) instead of recomputing
                    ena_i = ena_arr_buf[i]
                    ek_i = ek_arr_buf[i]

                    mi = y_active[off_m + i]
                    hi = y_active[off_h + i]
                    ni = y_active[off_n + i]
                    xi_v = y_active[off_x + i] if en_nap else 0.0
                    yi_v = y_active[off_y + i] if en_nar else 0.0
                    ji_v = y_active[off_j + i] if en_nar else 0.0
                    ai_v = y_active[off_a + i] if en_ia else 0.0
                    bi_v = y_active[off_b + i] if en_ia else 0.0
                    zi_v = y_active[off_zsk + i] if en_sk else 0.0
                    wi_v = y_active[off_w + i] if en_im else 0.0
                    _i_pump, _i_katp, datp, dnai, dko = compute_metabolism_and_pump(
                        vi, mi, hi, ni, xi_v, yi_v, ji_v, ai_v, bi_v, zi_v, wi_v,
                        gna_v[i], gk_v[i], ga_v[i], gsk_v[i], gim_v[i], gnap_v[i], gnar_v[i],
                        ena_i, ek_i,
                        en_nap, en_nar, en_ia, en_sk, en_im, dyn_ca,
                        g_katp_max, katp_kd_atp_mM,
                        atp_val, atp_synthesis_rate,
                        na_i_val, k_o_val, k_o_rest_mM,
                        ion_drift_gain, k_o_clearance_tau_ms,
                        i_ca_influx_2d[traj_idx, i],
                        pump_max_capacity,
                        km_na,
                    )
                    y_active[off_atp + i] = atp_val + datp * dt
                    y_active[off_na_i + i] = na_i_val + dnai * dt
                    y_active[off_k_o + i] = k_o_val + dko * dt

        # end trajectory loop

        if diverged == 1:
            break

        t += dt

        # ─────────────────────────────────────────────────────────────
        # 9. Benettin re-orthonormalization (periodic)
        # ─────────────────────────────────────────────────────────────
        if calc_lle and t >= lle_t_next_renorm:
            dist_sq = 0.0

            # Subspace-aware distance calculation
            # 0=v_only, 1=v_and_gates, 2=full_state, 3=custom
            if lle_subspace_mode == 3 and lle_custom_mask is not None:
                # Custom mode (3): use provided mask and optional weights
                for i in range(n_state):
                    if lle_custom_mask[i]:
                        w = lle_weights[i] if lle_weights is not None else 1.0
                        delta = (y_pert[i] - y[i]) * w
                        dist_sq += delta * delta
            elif lle_subspace_mode == 0:
                # v_only (0): Voltage only - first n_comp compartments (fast, standard neurophysiology)
                for i in range(n_comp):
                    delta = y_pert[i] - y[i]
                    dist_sq += delta * delta
            elif lle_subspace_mode == 1:
                # v_and_gates (1): Voltage + all gating variables (HH phase space)
                # Includes: V, m, h, n, r, s, u, a, b, p, q, w, x, y, j, zsk
                for i in range(n_comp):  # V compartments
                    delta = y_pert[i] - y[i]
                    dist_sq += delta * delta
                # Gate offsets: m, h, n are always present
                for off in (off_m, off_h, off_n):
                    for i in range(n_comp):
                        delta = y_pert[off + i] - y[off + i]
                        dist_sq += delta * delta
                # Optional gates based on channel flags
                if en_ih:
                    for i in range(n_comp):
                        delta = y_pert[off_r + i] - y[off_r + i]
                        dist_sq += delta * delta
                if en_ica:
                    for off in (off_s, off_u):
                        for i in range(n_comp):
                            delta = y_pert[off + i] - y[off + i]
                            dist_sq += delta * delta
                if en_ia:
                    for off in (off_a, off_b):
                        for i in range(n_comp):
                            delta = y_pert[off + i] - y[off + i]
                            dist_sq += delta * delta
                if en_itca:
                    for off in (off_p, off_q):
                        for i in range(n_comp):
                            delta = y_pert[off + i] - y[off + i]
                            dist_sq += delta * delta
                if en_im:
                    for i in range(n_comp):
                        delta = y_pert[off_w + i] - y[off_w + i]
                        dist_sq += delta * delta
                if en_nap:
                    for i in range(n_comp):
                        delta = y_pert[off_x + i] - y[off_x + i]
                        dist_sq += delta * delta
                if en_nar:
                    for off in (off_y, off_j):
                        for i in range(n_comp):
                            delta = y_pert[off + i] - y[off + i]
                            dist_sq += delta * delta
                if en_sk:
                    for i in range(n_comp):
                        delta = y_pert[off_zsk + i] - y[off_zsk + i]
                        dist_sq += delta * delta
            else:  # mode 2 (full_state) or default
                # Full state vector (all variables including Ca, ATP, etc.)
                for i in range(n_state):
                    delta = y_pert[i] - y[i]
                    dist_sq += delta * delta

            dist = np.sqrt(dist_sq)

            if dist > 1e-12:  # More reasonable threshold than 1e-30
                lle_accum += np.log(dist / lle_delta)
                lle_count += 1
                # Renormalize perturbation vector to lle_delta
                scale = lle_delta / dist
                for i in range(n_state):
                    y_pert[i] = y[i] + (y_pert[i] - y[i]) * scale

            lle_t_next_renorm = t + lle_t_evolve  # Use absolute time, not incremental

    # ── Final sample ──
    if out_idx < n_out:
        t_out[out_idx] = t
        for s in range(n_state):
            y_out[s, out_idx] = y[s]
        if calc_lle and t > 0.0 and lle_count > 0:
            lle_out[out_idx] = lle_accum / t
        out_idx += 1

    # Return empty LLE array if not computed to avoid confusion with zero values
    lle_result = lle_out[:out_idx] if calc_lle else np.zeros(0, dtype=np.float64)
    return t_out[:out_idx], y_out[:, :out_idx], bool(diverged), lle_result

