"""Native Hines solver kernels — v11.2.

O(N) direct tree solver replacing SciPy BDF for the voltage equation.
The semi-implicit split-operator scheme:
  1. Compute Ca²⁺ influx at V_n (for Ca dynamics).
  2. Update all gating variables analytically (exact exponential).
  3. Compute ionic conductances at gate_n+1.
  4. Build the Hines tridiagonal system for V_{n+1} (Backward Euler).
  5. Solve in O(N) via leaf→root elimination + root→leaf back-substitution.

References
----------
- Hines 1984, Int J Biomed Comput 15:69-76.
- Carnevale & Hines 2006, The NEURON Book. Cambridge University Press.
"""
from __future__ import annotations

import numpy as np
from numba import njit, float64, int32

from .kinetics import (
    am_lut, bm_lut, ah_lut, bh_lut, an_lut, bn_lut,
    ar_Ih_lut, br_Ih_lut,
    as_Ca_lut, bs_Ca_lut, au_Ca_lut, bu_Ca_lut,
    aa_IA_lut, ba_IA_lut, ab_IA_lut, bb_IA_lut,
    am_TCa_lut, bm_TCa_lut, ah_TCa_lut, bh_TCa_lut,
    aw_IM_lut, bw_IM_lut,
    ax_NaP_lut, bx_NaP_lut,
    ay_NaR_lut, by_NaR_lut, aj_NaR_lut, bj_NaR_lut,
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


@njit
def update_gates_analytic(
    y, dt, n_comp,
    # channel enable flags
    en_ih, en_ica, en_ia, en_sk, en_itca, en_im, en_nap, en_nar, dyn_ca,
    # state offsets (same layout as rhs.py / channels.py)
    off_m, off_h, off_n,
    off_r, off_s, off_u, off_a, off_b,
    off_p, off_q, off_w, off_x, off_y, off_j, off_zsk, off_ca,
    # phi temperature vectors [n_comp]
    phi_na, phi_k, phi_ih, phi_ca, phi_k2,   # phi_k2 = phi_k (IA, IM use K family)
    im_speed_multiplier,
    # calcium / SK
    ca_rest, tau_ca, tau_sk, b_ca,
    # I_Ca influx array written by caller before this function (size n_comp)
    i_ca_influx_v,
):
    """Update all gating variables analytically in-place on y[].

    Voltage y[0:n_comp] is read as V_n (start of time-step).
    Gate update uses the voltage at the START of the step (operator-splitting).

    Returns nothing — modifies y[] in place.
    """
    for i in range(n_comp):
        vi = y[i]                  # voltage at start of step

        # ── Na gates (m, h) — phi_na ──
        phi = phi_na[i]
        eff_dt_na = dt * phi
        y[off_m + i] = _gate_step(y[off_m + i], am_lut(vi), bm_lut(vi), eff_dt_na)
        y[off_h + i] = _gate_step(y[off_h + i], ah_lut(vi), bh_lut(vi), eff_dt_na)

        # ── K gate (n) — phi_k ──
        phi = phi_k[i]
        eff_dt_k = dt * phi
        y[off_n + i] = _gate_step(y[off_n + i], an_lut(vi), bn_lut(vi), eff_dt_k)

        # ── Ih gate (r) — phi_ih ──
        if en_ih:
            phi = phi_ih[i]
            y[off_r + i] = _gate_step(y[off_r + i], ar_Ih_lut(vi), br_Ih_lut(vi), dt * phi)

        # ── ICa gates (s, u) — phi_ca ──
        if en_ica:
            phi = phi_ca[i]
            eff_dt_ca = dt * phi
            y[off_s + i] = _gate_step(y[off_s + i], as_Ca_lut(vi), bs_Ca_lut(vi), eff_dt_ca)
            y[off_u + i] = _gate_step(y[off_u + i], au_Ca_lut(vi), bu_Ca_lut(vi), eff_dt_ca)

        # ── IA gates (a, b) — phi_k2 ──
        if en_ia:
            y[off_a + i] = _gate_step(y[off_a + i], aa_IA_lut(vi), ba_IA_lut(vi), eff_dt_k)
            y[off_b + i] = _gate_step(y[off_b + i], ab_IA_lut(vi), bb_IA_lut(vi), eff_dt_k)

        # ── ITCa gates (p, q) — phi_ca ──
        if en_itca:
            if not en_ica:
                phi = phi_ca[i]
                eff_dt_ca = dt * phi
            y[off_p + i] = _gate_step(y[off_p + i], am_TCa_lut(vi), bm_TCa_lut(vi), eff_dt_ca)
            y[off_q + i] = _gate_step(y[off_q + i], ah_TCa_lut(vi), bh_TCa_lut(vi), eff_dt_ca)

        # ── IM gate (w) — phi_k2 ──
        if en_im:
            y[off_w + i] = _gate_step(
                y[off_w + i],
                aw_IM_lut(vi),
                bw_IM_lut(vi),
                eff_dt_k * im_speed_multiplier,
            )

        # ── NaP gate (x) — phi_na family ──
        if en_nap:
            y[off_x + i] = _gate_step(y[off_x + i], ax_NaP_lut(vi), bx_NaP_lut(vi), eff_dt_na)

        # ── NaR gates (y, j) — phi_na family ──
        if en_nar:
            y[off_y + i] = _gate_step(y[off_y + i], ay_NaR_lut(vi), by_NaR_lut(vi), eff_dt_na)
            y[off_j + i] = _gate_step(y[off_j + i], aj_NaR_lut(vi), bj_NaR_lut(vi), eff_dt_na)

        # ── SK gate ODE: dz/dt = phi_k * (z_inf(Ca) - z) / tau_SK
        # SK is a Ca-activated K channel — temperature scaling via phi_k.
        # Hirschberg et al. 1998, J Gen Physiol 111:565
        if en_sk:
            zi = y[off_zsk + i]
            ca_val = y[off_ca + i] if dyn_ca else ca_rest
            ca_sk = ca_val if ca_val > 0 else ca_rest
            z_inf = z_inf_SK(ca_sk)
            tau_eff = max(tau_sk, 1e-12) / max(phi_k[i], 1e-12) if phi_k[i] > 1e-15 else max(tau_sk, 1e-12)
            y[off_zsk + i] = z_inf + (zi - z_inf) * np.exp(-dt / tau_eff)


# ─────────────────────────────────────────────────────────────────────────────
# Hines direct solver — O(N) for a branched cable tree.
#
# System stored per compartment i (parent p, children c):
#   d[i]*V[i]  - a[i]*V[p]  - Σ_c b[c]*V[c]  = rhs[i]
#
# where:
#   d[i]   = Cm[i]/dt + g_ion[i] - L_diag[i]   (positive diagonal)
#   a[i]   = g_axial_to_parent[i]               (positive, child→parent off-diag)
#   b[i]   = g_axial_parent_to_child[i]         (positive, parent←child off-diag)
#
# Forward elimination (leaf → root, skipping root):
#   factor  = b[i] / d[i]
#   d[p]   -= factor * a[i]     (Schur-complement update of parent diagonal)
#   rhs[p] += factor * rhs[i]   (+, derived from substituting V[i])
#
# Back-substitution (root → leaf):
#   V[root] = rhs'[root] / d'[root]
#   V[i]    = (rhs[i] + a[i]*V[parent]) / d[i]   (+ because row has -a[i]*V[p])
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def hines_solve(d, a, b, parent_idx, hines_order, rhs_vec, V_out):
    """Solve the Hines linear system in-place.

    Parameters
    ----------
    d         : float64[N]  — diagonal (modified in place during elimination)
    a         : float64[N]  — child→parent off-diagonal (positive)
    b         : float64[N]  — parent←child off-diagonal (positive)
    parent_idx: int32[N]    — parent compartment index (-1 for root)
    hines_order: int32[N]   — leaves-to-root ordering (root is last element)
    rhs_vec   : float64[N]  — right-hand side (modified in place)
    V_out     : float64[N]  — solution written here
    """
    n = d.shape[0]

    # ── Forward elimination: leaves → root (skip last = root) ──
    for k in range(n - 1):
        i = hines_order[k]
        p = parent_idx[i]
        if p < 0:
            continue                    # root should appear last, not here
        di = d[i]
        if abs(di) < 1e-30:
            continue
        factor = b[i] / di             # b[i] = parent←child coupling
        d[p]       -= factor * a[i]    # a[i] = child→parent coupling
        rhs_vec[p] += factor * rhs_vec[i]   # + (not −): see derivation above

    # ── Back-substitution: root → leaves ──
    root = hines_order[n - 1]
    V_out[root] = rhs_vec[root] / d[root]

    for k in range(n - 2, -1, -1):
        i = hines_order[k]
        p = parent_idx[i]
        if p < 0:
            V_out[i] = rhs_vec[i] / d[i]
        else:
            V_out[i] = (rhs_vec[i] + a[i] * V_out[p]) / d[i]   # + (row has -a*V[p])
