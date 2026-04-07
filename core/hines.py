"""Native Hines solver kernels — v11.0.

O(N) direct tree solver replacing SciPy BDF for the voltage equation.
The semi-implicit split-operator scheme:
  1. Update all gating variables analytically (exact exponential).
  2. Compute ionic conductances and update-gate currents at V_n.
  3. Build the Hines tridiagonal system for V_{n+1} (Backward Euler).
  4. Solve in O(N) via leaf→root elimination + root→leaf back-substitution.

References
----------
- Hines 1984, Int J Biomed Comput 15:69-76.
- Carnevale & Hines 2006, The NEURON Book. Cambridge University Press.
"""
from __future__ import annotations

import numpy as np
from numba import njit, float64, int32

from .kinetics import (
    am, bm, ah, bh, an, bn,
    ar_Ih, br_Ih,
    as_Ca, bs_Ca, au_Ca, bu_Ca,
    aa_IA, ba_IA, ab_IA, bb_IA,
    am_TCa, bm_TCa, ah_TCa, bh_TCa,
    aw_IM, bw_IM,
    ax_NaP, bx_NaP,
    ay_NaR, by_NaR, aj_NaR, bj_NaR,
    z_inf_SK,
)
from .rhs import nernst_ca_ion, CA_I_MIN_M_M, CA_I_MAX_M_M, CA_DAMPING_FACTOR


# ─────────────────────────────────────────────────────────────────────────────
# Gate analytic update: x_new = x_inf + (x_old - x_inf) * exp(-dt/tau_x)
# where x_inf = alpha/(alpha+beta), tau_x = 1/(alpha+beta).
# Exact solution of dx/dt = alpha*(1-x) - beta*x over interval dt.
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _gate_step(x_old, alpha, beta, dt):
    """Exact exponential update for a single HH gate."""
    ab = alpha + beta
    if ab < 1e-15:
        return x_old
    x_inf = alpha / ab
    return x_inf + (x_old - x_inf) * np.exp(-ab * dt)


@njit(cache=True)
def update_gates_analytic(
    y, dt, n_comp,
    # channel enable flags
    en_ih, en_ica, en_ia, en_sk, en_itca, en_im, en_nap, en_nar, dyn_ca,
    # state offsets (same layout as rhs.py)
    off_m, off_h, off_n,
    off_r, off_s, off_u, off_a, off_b,
    off_p, off_q, off_w, off_x, off_y, off_j, off_zsk, off_ca,
    # phi temperature vectors [n_comp]
    phi_na, phi_k, phi_ih, phi_ca, phi_k2,   # phi_k2 = phi_k (IA,IM use K family)
    # calcium / SK
    ca_rest, tau_ca, tau_sk, b_ca,
    # I_Ca influx array written by caller before this function (size n_comp)
    i_ca_influx_v,
):
    """Update all gating variables analytically in-place on y[].

    Voltage y[0:n_comp] must already be set to V_{n+1} before calling
    (or to V_n for the first step).  Gate update uses the voltage at the
    START of the time-step for the kinetic rates (operator-splitting).

    Returns nothing — modifies y[] in place.
    """
    for i in range(n_comp):
        vi = y[i]                  # voltage at start of step

        # ── Na gates (m, h) — phi_na ──
        phi = phi_na[i]
        eff_dt_na = dt * phi
        y[off_m + i] = _gate_step(y[off_m + i], am(vi), bm(vi), eff_dt_na)
        y[off_h + i] = _gate_step(y[off_h + i], ah(vi), bh(vi), eff_dt_na)

        # ── K gate (n) — phi_k ──
        phi = phi_k[i]
        eff_dt_k = dt * phi
        y[off_n + i] = _gate_step(y[off_n + i], an(vi), bn(vi), eff_dt_k)

        # ── Ih gate (r) — phi_ih ──
        if en_ih:
            phi = phi_ih[i]
            y[off_r + i] = _gate_step(y[off_r + i], ar_Ih(vi), br_Ih(vi), dt * phi)

        # ── ICa gates (s, u) — phi_ca ──
        if en_ica:
            phi = phi_ca[i]
            eff_dt_ca = dt * phi
            y[off_s + i] = _gate_step(y[off_s + i], as_Ca(vi), bs_Ca(vi), eff_dt_ca)
            y[off_u + i] = _gate_step(y[off_u + i], au_Ca(vi), bu_Ca(vi), eff_dt_ca)

        # ── IA gates (a, b) — phi_k ──
        if en_ia:
            y[off_a + i] = _gate_step(y[off_a + i], aa_IA(vi), ba_IA(vi), eff_dt_k)
            y[off_b + i] = _gate_step(y[off_b + i], ab_IA(vi), bb_IA(vi), eff_dt_k)

        # ── ITCa gates (p, q) — phi_ca ──
        if en_itca:
            if not en_ica:
                phi = phi_ca[i]
                eff_dt_ca = dt * phi
            y[off_p + i] = _gate_step(y[off_p + i], am_TCa(vi), bm_TCa(vi), eff_dt_ca)
            y[off_q + i] = _gate_step(y[off_q + i], ah_TCa(vi), bh_TCa(vi), eff_dt_ca)

        # ── IM gate (w) — phi_k ──
        if en_im:
            y[off_w + i] = _gate_step(y[off_w + i], aw_IM(vi), bw_IM(vi), eff_dt_k)

        # ── NaP gate (x) — phi_nap (use phi_na family) ──
        if en_nap:
            y[off_x + i] = _gate_step(y[off_x + i], ax_NaP(vi), bx_NaP(vi), eff_dt_na)

        # ── NaR gates (y, j) — phi_na family ──
        if en_nar:
            y[off_y + i] = _gate_step(y[off_y + i], ay_NaR(vi), by_NaR(vi), eff_dt_na)
            y[off_j + i] = _gate_step(y[off_j + i], aj_NaR(vi), bj_NaR(vi), eff_dt_na)

        # ── SK gate (z) — driven by [Ca²⁺], phi_k ──
        if en_sk:
            if dyn_ca:
                ca_sk = min(max(y[off_ca + i], CA_I_MIN_M_M), CA_I_MAX_M_M)
            else:
                ca_sk = ca_rest
            z_inf = z_inf_SK(ca_sk)
            # SK uses first-order relaxation with tau_sk / phi_k
            tau_eff = tau_sk / phi_k[i] if phi_k[i] > 1e-15 else tau_sk
            y[off_zsk + i] = z_inf + (y[off_zsk + i] - z_inf) * np.exp(-dt / tau_eff)

        # ── Calcium dynamics (Euler — stiff but bounded) ──
        if dyn_ca:
            ca_val = y[off_ca + i]
            dca = b_ca[i] * i_ca_influx_v[i] - (ca_val - ca_rest) / tau_ca
            ca_at_min = (ca_val <= CA_I_MIN_M_M and dca < 0.0)
            ca_at_max = (ca_val >= CA_I_MAX_M_M and dca > 0.0)
            if ca_at_min:
                dca = abs(dca) * CA_DAMPING_FACTOR
            elif ca_at_max:
                dca = -abs(dca) * CA_DAMPING_FACTOR
            y[off_ca + i] = ca_val + dt * dca


# ─────────────────────────────────────────────────────────────────────────────
# Hines direct solver — O(N) for a branched cable tree.
#
# System: (Cm/dt + g_total_i) * V_i - g_axial_to_parent[i] * V_parent
#         - sum_children g_axial_parent_to_child[c] * V_c = rhs_i
#
# Arrays d[], a[], b[], rhs[] are passed pre-allocated and filled by
# run_native_loop.  This function modifies them in place and writes
# the solution into V_out[].
#
# Asymmetric coupling note:
#   a[i] = g_axial_to_parent[i]       (off-diag in row i,   child→parent)
#   b[i] = g_axial_parent_to_child[i] (off-diag in row p,   parent←child)
#   Elimination: factor = b[i] / d[i]
#                d[parent] -= factor * a[i]
#                rhs[parent] -= factor * rhs[i]
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def hines_solve(d, a, b, parent_idx, hines_order, rhs_vec, V_out):
    """Solve the Hines linear system in-place.

    Parameters
    ----------
    d         : float64[N]  — diagonal (modified in place during elimination)
    a         : float64[N]  — child→parent off-diagonal coupling
    b         : float64[N]  — parent←child off-diagonal coupling
    parent_idx: int32[N]    — parent compartment index (-1 for root)
    hines_order: int32[N]   — leaves-to-root ordering
    rhs_vec   : float64[N]  — right-hand side (modified in place)
    V_out     : float64[N]  — solution written here
    """
    n = d.shape[0]

    # ── Forward elimination: leaves → root (skip last = root) ──
    #
    # System for compartment i (with parent p):
    #   d[i]*V_i  - a[i]*V_p  - sum_c(b[c]*V_c) = rhs[i]
    #
    # Eliminating child i from parent p's equation:
    #   factor = b[i] / d[i]
    #   d[p]    -= factor * a[i]    (Schur complement update)
    #   rhs[p]  += factor * rhs[i]  (+, not −: derived from substituting V_i)
    #
    # Child rows (d[i], rhs[i]) are NOT modified — only parent rows change.
    for k in range(n - 1):
        i = hines_order[k]
        p = parent_idx[i]
        if p < 0:
            continue                    # root should appear last, not here
        di = d[i]
        if abs(di) < 1e-30:
            continue
        factor = b[i] / di
        d[p]       -= factor * a[i]
        rhs_vec[p] += factor * rhs_vec[i]

    # ── Back-substitution: root → leaves ──
    #
    # After elimination the root equation is:  d'[root]*V[root] = rhs'[root]
    # For non-root i:  d[i]*V[i] - a[i]*V[parent] = rhs[i]  (d, rhs unchanged)
    #   => V[i] = (rhs[i] + a[i]*V[parent]) / d[i]
    root = hines_order[n - 1]
    V_out[root] = rhs_vec[root] / d[root]

    for k in range(n - 2, -1, -1):
        i = hines_order[k]
        p = parent_idx[i]
        if p < 0:
            V_out[i] = rhs_vec[i] / d[i]
        else:
            V_out[i] = (rhs_vec[i] + a[i] * V_out[p]) / d[i]
