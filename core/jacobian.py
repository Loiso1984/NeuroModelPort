from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from .kinetics import (
    aa_IA,
    ab_IA,
    ah,
    ah_TCa,
    am,
    am_TCa,
    an,
    ar_Ih,
    as_Ca,
    au_Ca,
    ba_IA,
    bb_IA,
    bh,
    bh_TCa,
    bm,
    bm_TCa,
    bn,
    br_Ih,
    bs_Ca,
    bu_Ca,
)
from .rhs import F_CONST, R_GAS


def _state_slices(
    n_comp: int,
    en_ih: bool,
    en_ica: bool,
    en_ia: bool,
    dyn_ca: bool,
    use_dfilter_primary: int,
    use_dfilter_secondary: int,
    en_itca: bool = False,
) -> tuple[dict[str, slice | int | None], int]:
    cursor = 0
    out: dict[str, slice | int | None] = {}
    out["v"] = slice(cursor, cursor + n_comp)
    cursor += n_comp
    out["m"] = slice(cursor, cursor + n_comp)
    cursor += n_comp
    out["h"] = slice(cursor, cursor + n_comp)
    cursor += n_comp
    out["n"] = slice(cursor, cursor + n_comp)
    cursor += n_comp

    out["r"] = None
    if en_ih:
        out["r"] = slice(cursor, cursor + n_comp)
        cursor += n_comp

    out["s"] = None
    out["u"] = None
    if en_ica:
        out["s"] = slice(cursor, cursor + n_comp)
        cursor += n_comp
        out["u"] = slice(cursor, cursor + n_comp)
        cursor += n_comp

    out["a"] = None
    out["b"] = None
    if en_ia:
        out["a"] = slice(cursor, cursor + n_comp)
        cursor += n_comp
        out["b"] = slice(cursor, cursor + n_comp)
        cursor += n_comp

    out["p"] = None  # T-type Ca activation
    out["q"] = None  # T-type Ca inactivation
    if en_itca:
        out["p"] = slice(cursor, cursor + n_comp)
        cursor += n_comp
        out["q"] = slice(cursor, cursor + n_comp)
        cursor += n_comp

    out["ca"] = None
    if dyn_ca:
        out["ca"] = slice(cursor, cursor + n_comp)
        cursor += n_comp

    out["dfilter_primary"] = None
    if use_dfilter_primary == 1:
        out["dfilter_primary"] = cursor
        cursor += 1

    out["dfilter_secondary"] = None
    if use_dfilter_secondary == 1:
        out["dfilter_secondary"] = cursor
        cursor += 1

    return out, cursor


def _rate_derivative(fn, v: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    return (fn(v + eps) - fn(v - eps)) / (2.0 * eps)


def build_jacobian_sparsity(
    n_comp: int,
    en_ih: bool,
    en_ica: bool,
    en_ia: bool,
    en_sk: bool,
    dyn_ca: bool,
    l_indices: np.ndarray,
    l_indptr: np.ndarray,
    use_dfilter_primary: int,
    use_dfilter_secondary: int,
    en_itca: bool = False,
) -> csr_matrix:
    idx, n_state = _state_slices(
        n_comp, en_ih, en_ica, en_ia, dyn_ca, use_dfilter_primary, use_dfilter_secondary,
        en_itca=en_itca,
    )
    sp = lil_matrix((n_state, n_state), dtype=float)

    v_slice = idx["v"]
    m_slice = idx["m"]
    h_slice = idx["h"]
    n_slice = idx["n"]
    r_slice = idx["r"]
    s_slice = idx["s"]
    u_slice = idx["u"]
    a_slice = idx["a"]
    b_slice = idx["b"]
    p_slice = idx["p"]
    q_slice = idx["q"]
    ca_slice = idx["ca"]
    dfilter_primary_idx = idx["dfilter_primary"]
    dfilter_secondary_idx = idx["dfilter_secondary"]

    for i in range(n_comp):
        v_row = v_slice.start + i
        # Axial dependencies from Laplacian row.
        for p in range(int(l_indptr[i]), int(l_indptr[i + 1])):
            v_col = int(l_indices[p])
            sp[v_row, v_slice.start + v_col] = 1.0

        # Local channel-state dependencies in voltage equation.
        sp[v_row, m_slice.start + i] = 1.0
        sp[v_row, h_slice.start + i] = 1.0
        sp[v_row, n_slice.start + i] = 1.0
        if r_slice is not None:
            sp[v_row, r_slice.start + i] = 1.0
        if s_slice is not None:
            sp[v_row, s_slice.start + i] = 1.0
            sp[v_row, u_slice.start + i] = 1.0
        if a_slice is not None:
            sp[v_row, a_slice.start + i] = 1.0
            sp[v_row, b_slice.start + i] = 1.0
        if p_slice is not None:
            sp[v_row, p_slice.start + i] = 1.0
            sp[v_row, q_slice.start + i] = 1.0
        if ca_slice is not None and (en_ica or en_sk or en_itca):
            sp[v_row, ca_slice.start + i] = 1.0

        # Optional soma dependency on dendritic-filter states.
        if i == 0:
            if dfilter_primary_idx is not None:
                sp[v_row, int(dfilter_primary_idx)] = 1.0
            if dfilter_secondary_idx is not None:
                sp[v_row, int(dfilter_secondary_idx)] = 1.0

        # HH core gates.
        m_row = m_slice.start + i
        h_row = h_slice.start + i
        n_row = n_slice.start + i
        sp[m_row, v_row] = 1.0
        sp[m_row, m_row] = 1.0
        sp[h_row, v_row] = 1.0
        sp[h_row, h_row] = 1.0
        sp[n_row, v_row] = 1.0
        sp[n_row, n_row] = 1.0

        if r_slice is not None:
            r_row = r_slice.start + i
            sp[r_row, v_row] = 1.0
            sp[r_row, r_row] = 1.0
        if s_slice is not None:
            s_row = s_slice.start + i
            u_row = u_slice.start + i
            sp[s_row, v_row] = 1.0
            sp[s_row, s_row] = 1.0
            sp[u_row, v_row] = 1.0
            sp[u_row, u_row] = 1.0
        if a_slice is not None:
            a_row = a_slice.start + i
            b_row = b_slice.start + i
            sp[a_row, v_row] = 1.0
            sp[a_row, a_row] = 1.0
            sp[b_row, v_row] = 1.0
            sp[b_row, b_row] = 1.0

        if p_slice is not None:
            p_row = p_slice.start + i
            q_row = q_slice.start + i
            sp[p_row, v_row] = 1.0
            sp[p_row, p_row] = 1.0
            sp[q_row, v_row] = 1.0
            sp[q_row, q_row] = 1.0

        if ca_slice is not None:
            ca_row = ca_slice.start + i
            sp[ca_row, ca_row] = 1.0
            if en_ica:
                sp[ca_row, v_row] = 1.0
                sp[ca_row, s_slice.start + i] = 1.0
                sp[ca_row, u_slice.start + i] = 1.0
            if en_itca:
                sp[ca_row, v_row] = 1.0
                sp[ca_row, p_slice.start + i] = 1.0
                sp[ca_row, q_slice.start + i] = 1.0

    if dfilter_primary_idx is not None:
        sp[int(dfilter_primary_idx), int(dfilter_primary_idx)] = 1.0
    if dfilter_secondary_idx is not None:
        sp[int(dfilter_secondary_idx), int(dfilter_secondary_idx)] = 1.0

    return sp.tocsr()


def _build_csr_pos_map(csr: csr_matrix) -> dict:
    """Build (row, col) -> data index lookup for fast CSR updates."""
    pos = {}
    for row in range(csr.shape[0]):
        for k in range(csr.indptr[row], csr.indptr[row + 1]):
            pos[(row, int(csr.indices[k]))] = int(k)
    return pos


def make_analytic_jacobian(sparsity_csr: csr_matrix):
    """
    Factory: returns a Jacobian callback that reuses a pre-allocated CSR matrix.
    Avoids lil_matrix creation and tocsr() conversion on every call.
    """
    J_csr = sparsity_csr.copy().astype(float)
    J_csr.data[:] = 0.0
    pos = _build_csr_pos_map(J_csr)

    def _set(row, col, val):
        k = pos.get((row, col))
        if k is not None:
            J_csr.data[k] = val

    def _add(row, col, val):
        k = pos.get((row, col))
        if k is not None:
            J_csr.data[k] += val

    def jac_fn(
        t, y, n_comp,
        en_ih, en_ica, en_ia, en_sk, dyn_ca, en_itca,
        gna_v, gk_v, gl_v, gih_v, gca_v, ga_v, gsk_v, gtca_v,
        ena, ek, el, eih, ea,
        cm_v, l_data, l_indices, l_indptr,
        phi_na, phi_k, phi_ih, phi_ca, phi_ia, phi_tca,
        t_kelvin, ca_ext, ca_rest, tau_ca, b_ca,
        stype, iext, t0, td, atau, stim_comp, stim_mode,
        use_dfilter_primary, dfilter_attenuation, dfilter_tau_ms,
        dual_stim_enabled,
        stype_2, iext_2, t0_2, td_2, atau_2, stim_comp_2, stim_mode_2,
        use_dfilter_secondary, dfilter_attenuation_2, dfilter_tau_ms_2,
    ):
        # Zero all entries
        J_csr.data[:] = 0.0

        idx, n_state = _state_slices(
            n_comp, en_ih, en_ica, en_ia, dyn_ca, use_dfilter_primary, use_dfilter_secondary,
            en_itca=en_itca,
        )

        v = y[idx["v"]]
        m = y[idx["m"]]
        h = y[idx["h"]]
        n = y[idx["n"]]
        r = y[idx["r"]] if idx["r"] is not None else np.zeros(n_comp)
        s = y[idx["s"]] if idx["s"] is not None else np.zeros(n_comp)
        u = y[idx["u"]] if idx["u"] is not None else np.zeros(n_comp)
        a = y[idx["a"]] if idx["a"] is not None else np.zeros(n_comp)
        b = y[idx["b"]] if idx["b"] is not None else np.zeros(n_comp)
        ca_i = y[idx["ca"]] if idx["ca"] is not None else np.full(n_comp, ca_rest)

        k_nernst = (R_GAS * t_kelvin / (2.0 * F_CONST)) * 1000.0

        am_v = am(v);  bm_v = bm(v)
        ah_v = ah(v);  bh_v = bh(v)
        an_v = an(v);  bn_v = bn(v)
        dam_v = _rate_derivative(am, v); dbm_v = _rate_derivative(bm, v)
        dah_v = _rate_derivative(ah, v); dbh_v = _rate_derivative(bh, v)
        dan_v = _rate_derivative(an, v); dbn_v = _rate_derivative(bn, v)

        if en_ih:
            ar_v = ar_Ih(v);  br_v = br_Ih(v)
            dar_v = _rate_derivative(ar_Ih, v); dbr_v = _rate_derivative(br_Ih, v)
        if en_ica:
            as_v = as_Ca(v);  bs_v = bs_Ca(v)
            au_v = au_Ca(v);  bu_v = bu_Ca(v)
            das_v = _rate_derivative(as_Ca, v); dbs_v = _rate_derivative(bs_Ca, v)
            dau_v = _rate_derivative(au_Ca, v); dbu_v = _rate_derivative(bu_Ca, v)
        if en_ia:
            aa_v = aa_IA(v);  ba_v = ba_IA(v)
            ab_v = ab_IA(v);  bb_v = bb_IA(v)
            daa_v = _rate_derivative(aa_IA, v); dba_v = _rate_derivative(ba_IA, v)
            dab_v = _rate_derivative(ab_IA, v); dbb_v = _rate_derivative(bb_IA, v)
        if en_itca:
            p_g = y[idx["p"]] if idx["p"] is not None else np.zeros(n_comp)
            q_g = y[idx["q"]] if idx["q"] is not None else np.zeros(n_comp)
            amt_v = am_TCa(v);  bmt_v = bm_TCa(v)
            aht_v = ah_TCa(v);  bht_v = bh_TCa(v)
            damt_v = _rate_derivative(am_TCa, v); dbmt_v = _rate_derivative(bm_TCa, v)
            daht_v = _rate_derivative(ah_TCa, v); dbht_v = _rate_derivative(bh_TCa, v)

        for i in range(n_comp):
            v_row = idx["v"].start + i
            cm = cm_v[i]

            # Axial coupling
            for p in range(int(l_indptr[i]), int(l_indptr[i + 1])):
                col = int(l_indices[p])
                _add(v_row, idx["v"].start + col, l_data[p] / cm)

            d_iion_dv = gl_v[i] + gna_v[i] * (m[i] ** 3) * h[i] + gk_v[i] * (n[i] ** 4)
            d_iion_dm = gna_v[i] * 3.0 * (m[i] ** 2) * h[i] * (v[i] - ena)
            d_iion_dh = gna_v[i] * (m[i] ** 3) * (v[i] - ena)
            d_iion_dn = gk_v[i] * 4.0 * (n[i] ** 3) * (v[i] - ek)
            d_iion_dca = 0.0

            if en_ih:
                d_iion_dv += gih_v[i] * r[i]
                _add(v_row, idx["r"].start + i, -(gih_v[i] * (v[i] - eih)) / cm)

            i_ca_val = 0.0
            if en_ica:
                if dyn_ca:
                    ca_safe = max(ca_i[i], 1e-9)
                    eca = k_nernst * np.log(ca_ext / ca_safe)
                    deca_dca = -k_nernst / ca_safe
                else:
                    eca = 120.0
                    deca_dca = 0.0
                dIcadv = gca_v[i] * (s[i] ** 2) * u[i]
                dIcads = gca_v[i] * 2.0 * s[i] * u[i] * (v[i] - eca)
                dIcadu = gca_v[i] * (s[i] ** 2) * (v[i] - eca)
                dIcadca = gca_v[i] * (s[i] ** 2) * u[i] * (-deca_dca)
                i_ca_val = gca_v[i] * (s[i] ** 2) * u[i] * (v[i] - eca)
                d_iion_dv += dIcadv
                _add(v_row, idx["s"].start + i, -dIcads / cm)
                _add(v_row, idx["u"].start + i, -dIcadu / cm)
                if dyn_ca:
                    d_iion_dca += dIcadca

            if en_ia:
                d_iion_dv += ga_v[i] * a[i] * b[i]
                _add(v_row, idx["a"].start + i, -(ga_v[i] * b[i] * (v[i] - ea)) / cm)
                _add(v_row, idx["b"].start + i, -(ga_v[i] * a[i] * (v[i] - ea)) / cm)

            i_tca_val = 0.0
            if en_itca and idx["p"] is not None:
                if dyn_ca:
                    ca_safe_t = max(ca_i[i], 1e-9)
                    eca_t = k_nernst * np.log(ca_ext / ca_safe_t)
                    deca_t_dca = -k_nernst / ca_safe_t
                else:
                    eca_t = 120.0
                    deca_t_dca = 0.0
                # If ICa already computed eca for this compartment, reuse it
                if en_ica and dyn_ca:
                    eca_t = k_nernst * np.log(ca_ext / max(ca_i[i], 1e-9))
                    deca_t_dca = -k_nernst / max(ca_i[i], 1e-9)
                dItdv = gtca_v[i] * (p_g[i] ** 2) * q_g[i]
                dItdp = gtca_v[i] * 2.0 * p_g[i] * q_g[i] * (v[i] - eca_t)
                dItdq = gtca_v[i] * (p_g[i] ** 2) * (v[i] - eca_t)
                dItdca = gtca_v[i] * (p_g[i] ** 2) * q_g[i] * (-deca_t_dca)
                i_tca_val = gtca_v[i] * (p_g[i] ** 2) * q_g[i] * (v[i] - eca_t)
                d_iion_dv += dItdv
                _add(v_row, idx["p"].start + i, -dItdp / cm)
                _add(v_row, idx["q"].start + i, -dItdq / cm)
                if dyn_ca:
                    d_iion_dca += dItdca

            if en_sk:
                ca_safe = max(ca_i[i], 1e-9)
                kd = 0.0004
                q = (kd / ca_safe) ** 4
                z = 1.0 / (1.0 + q)
                dz_dca = (4.0 * (kd ** 4) / (ca_safe ** 5)) / ((1.0 + q) ** 2)
                d_iion_dv += gsk_v[i] * z
                if dyn_ca:
                    d_iion_dca += gsk_v[i] * dz_dca * (v[i] - ek)

            _add(v_row, idx["v"].start + i, -d_iion_dv / cm)
            _add(v_row, idx["m"].start + i, -d_iion_dm / cm)
            _add(v_row, idx["h"].start + i, -d_iion_dh / cm)
            _add(v_row, idx["n"].start + i, -d_iion_dn / cm)
            if dyn_ca and idx["ca"] is not None:
                _add(v_row, idx["ca"].start + i, -d_iion_dca / cm)
            if i == 0:
                if idx["dfilter_primary"] is not None:
                    _add(v_row, int(idx["dfilter_primary"]), 1.0 / cm)
                if idx["dfilter_secondary"] is not None:
                    _add(v_row, int(idx["dfilter_secondary"]), 1.0 / cm)

            # HH gate rows
            v_col = idx["v"].start + i
            m_row = idx["m"].start + i
            h_row = idx["h"].start + i
            n_row = idx["n"].start + i
            _set(m_row, v_col, phi_na * (dam_v[i] * (1.0 - m[i]) - dbm_v[i] * m[i]))
            _set(m_row, m_row, -phi_na * (am_v[i] + bm_v[i]))
            _set(h_row, v_col, phi_na * (dah_v[i] * (1.0 - h[i]) - dbh_v[i] * h[i]))
            _set(h_row, h_row, -phi_na * (ah_v[i] + bh_v[i]))
            _set(n_row, v_col, phi_k * (dan_v[i] * (1.0 - n[i]) - dbn_v[i] * n[i]))
            _set(n_row, n_row, -phi_k * (an_v[i] + bn_v[i]))

            if en_ih and idx["r"] is not None:
                r_row = idx["r"].start + i
                _set(r_row, v_col, phi_ih * (dar_v[i] * (1.0 - r[i]) - dbr_v[i] * r[i]))
                _set(r_row, r_row, -phi_ih * (ar_v[i] + br_v[i]))

            if en_ica and idx["s"] is not None:
                s_row = idx["s"].start + i
                u_row = idx["u"].start + i
                _set(s_row, v_col, phi_ca * (das_v[i] * (1.0 - s[i]) - dbs_v[i] * s[i]))
                _set(s_row, s_row, -phi_ca * (as_v[i] + bs_v[i]))
                _set(u_row, v_col, phi_ca * (dau_v[i] * (1.0 - u[i]) - dbu_v[i] * u[i]))
                _set(u_row, u_row, -phi_ca * (au_v[i] + bu_v[i]))

            if en_ia and idx["a"] is not None:
                a_row = idx["a"].start + i
                b_row = idx["b"].start + i
                _set(a_row, v_col, phi_ia * (daa_v[i] * (1.0 - a[i]) - dba_v[i] * a[i]))
                _set(a_row, a_row, -phi_ia * (aa_v[i] + ba_v[i]))
                _set(b_row, v_col, phi_ia * (dab_v[i] * (1.0 - b[i]) - dbb_v[i] * b[i]))
                _set(b_row, b_row, -phi_ia * (ab_v[i] + bb_v[i]))

            if en_itca and idx["p"] is not None:
                p_row = idx["p"].start + i
                q_row = idx["q"].start + i
                _set(p_row, v_col, phi_tca * (damt_v[i] * (1.0 - p_g[i]) - dbmt_v[i] * p_g[i]))
                _set(p_row, p_row, -phi_tca * (amt_v[i] + bmt_v[i]))
                _set(q_row, v_col, phi_tca * (daht_v[i] * (1.0 - q_g[i]) - dbht_v[i] * q_g[i]))
                _set(q_row, q_row, -phi_tca * (aht_v[i] + bht_v[i]))

            if dyn_ca and idx["ca"] is not None:
                ca_row = idx["ca"].start + i
                d_dca = -1.0 / tau_ca
                if en_ica and i_ca_val < 0.0:
                    _add(ca_row, v_col, b_ca[i] * (-dIcadv))
                    _add(ca_row, idx["s"].start + i, b_ca[i] * (-dIcads))
                    _add(ca_row, idx["u"].start + i, b_ca[i] * (-dIcadu))
                    d_dca += b_ca[i] * (-dIcadca)
                if en_itca and idx["p"] is not None and i_tca_val < 0.0:
                    _add(ca_row, v_col, b_ca[i] * (-dItdv))
                    _add(ca_row, idx["p"].start + i, b_ca[i] * (-dItdp))
                    _add(ca_row, idx["q"].start + i, b_ca[i] * (-dItdq))
                    d_dca += b_ca[i] * (-dItdca)
                _set(ca_row, ca_row, d_dca)

        if idx["dfilter_primary"] is not None:
            dr = int(idx["dfilter_primary"])
            _set(dr, dr, -1.0 / dfilter_tau_ms if dfilter_tau_ms > 0.0 else 0.0)
        if idx["dfilter_secondary"] is not None:
            dr2 = int(idx["dfilter_secondary"])
            _set(dr2, dr2, -1.0 / dfilter_tau_ms_2 if dfilter_tau_ms_2 > 0.0 else 0.0)

        return J_csr

    return jac_fn


# Legacy interface for backward compatibility
def analytic_sparse_jacobian(*args, **kwargs):
    """Fallback: creates lil_matrix each call. Use make_analytic_jacobian instead."""
    # Extract enough args to build slices
    (t, y, n_comp, en_ih, en_ica, en_ia, en_sk, dyn_ca, en_itca) = args[:9]
    use_dfp = args[38] if len(args) > 38 else 0
    use_dfs = args[49] if len(args) > 49 else 0
    idx, n_state = _state_slices(
        n_comp, en_ih, en_ica, en_ia, dyn_ca, use_dfp, use_dfs, en_itca=en_itca,
    )
    sp = build_jacobian_sparsity(
        n_comp, en_ih, en_ica, en_ia, en_sk, dyn_ca,
        args[23], args[24],  # l_indices, l_indptr
        use_dfp, use_dfs,
        en_itca=en_itca,
    )
    fn = make_analytic_jacobian(sp)
    return fn(*args)
