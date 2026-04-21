from __future__ import annotations

import hashlib
import threading

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from .kinetics import (
    aa_IA,
    ab_IA,
    ah,
    ah_TCa,
    aj_NaR,
    am,
    am_TCa,
    an,
    ar_Ih,
    as_Ca,
    au_Ca,
    aw_IM,
    ax_NaP,
    ay_NaR,
    ba_IA,
    bb_IA,
    bh,
    bh_TCa,
    bj_NaR,
    bm,
    bm_TCa,
    bn,
    br_Ih,
    bs_Ca,
    bu_Ca,
    bw_IM,
    bx_NaP,
    by_NaR,
)
from .rhs import (
    CA_I_MAX_M_M,
    CA_I_MIN_M_M,
    F_CONST,
    K_O_MIN_M_M,
    NA_I_MIN_M_M,
    R_GAS,
)
from .physics_params import (
    PhysicsParams,
    build_state_offsets,
    state_slices_from_offsets,
    unpack_conductances,
    unpack_env_params,
    unpack_temperature_scaling,
)

_LEGACY_JACOBIAN_CACHE: dict[tuple, object] = {}
_LEGACY_JACOBIAN_CACHE_LOCK = threading.RLock()
_MAX_JAC_CACHE = 8


def _evict_oldest_if_needed() -> None:
    with _LEGACY_JACOBIAN_CACHE_LOCK:
        if len(_LEGACY_JACOBIAN_CACHE) >= _MAX_JAC_CACHE:
            oldest_key = next(iter(_LEGACY_JACOBIAN_CACHE))
            del _LEGACY_JACOBIAN_CACHE[oldest_key]


def _sparse_structure_signature(l_indices: np.ndarray, l_indptr: np.ndarray) -> tuple:
    """Hashable sparse-topology signature without tupleizing large arrays."""
    idx = np.ascontiguousarray(l_indices, dtype=np.int64)
    ptr = np.ascontiguousarray(l_indptr, dtype=np.int64)
    idx_hash = hashlib.blake2b(idx.view(np.uint8), digest_size=32).hexdigest()
    ptr_hash = hashlib.blake2b(ptr.view(np.uint8), digest_size=32).hexdigest()
    return (idx.shape, idx_hash, ptr.shape, ptr_hash)


def _state_slices(
    n_comp: int,
    en_ih: bool,
    en_ica: bool,
    en_ia: bool,
    dyn_ca: bool,
    dyn_atp: bool,
    use_dfilter_primary: int,
    use_dfilter_secondary: int,
    en_itca: bool = False,
    en_im: bool = False,
    en_nap: bool = False,
    en_nar: bool = False,
    en_sk: bool = False,
) -> tuple[dict[str, slice | int | None], int]:
    offsets = build_state_offsets(
        n_comp,
        en_ih=en_ih,
        en_ica=en_ica,
        en_ia=en_ia,
        en_sk=en_sk,
        dyn_ca=dyn_ca,
        en_itca=en_itca,
        en_im=en_im,
        en_nap=en_nap,
        en_nar=en_nar,
        dyn_atp=dyn_atp,
        use_dfilter_primary=use_dfilter_primary,
        use_dfilter_secondary=use_dfilter_secondary,
    )
    return state_slices_from_offsets(offsets, n_comp), int(offsets.n_state)


def _rate_derivative(fn, v: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    return (fn(v + eps) - fn(v - eps)) / (2.0 * eps)


def build_jacobian_sparsity(
    n_comp: int,
    en_ih: bool,
    en_ica: bool,
    en_ia: bool,
    en_sk: bool,
    dyn_ca: bool,
    dyn_atp: bool,
    l_indices: np.ndarray,
    l_indptr: np.ndarray,
    use_dfilter_primary: int,
    use_dfilter_secondary: int,
    en_itca: bool = False,
    en_im: bool = False,
    en_nap: bool = False,
    en_nar: bool = False,
    gNa_v: "np.ndarray | None" = None,
    _gna_sparsity_threshold: float = 2.0,
) -> csr_matrix:
    """Build the sparsity structure for the sparse-FD Jacobian.

    Parameters
    ----------
    gNa_v : array, optional
        Per-compartment Na conductance (mS/cm²).  When provided, m- and
        h-gate → voltage dependencies are **omitted** for compartments
        where gNa_v[i] < _gna_sparsity_threshold.  This prevents the
        near-zero columns produced by demyelinated trunk segments
        (gNa ≈ 0) from making the LU factor exactly singular in
        scipy BDF's SuperLU back-end.  The RHS is **not** affected;
        only the sparsity pattern used by the finite-difference
        Jacobian approximation is tightened.
    _gna_sparsity_threshold : float
        Conductance cut-off in mS/cm² (default 2.0).  Compartments
        with gNa < this value have their m/h columns excluded from the
        voltage-equation sparsity row.
    """
    idx, n_state = _state_slices(
        n_comp, en_ih, en_ica, en_ia, dyn_ca, dyn_atp, use_dfilter_primary, use_dfilter_secondary,
        en_itca=en_itca, en_im=en_im, en_nap=en_nap, en_nar=en_nar, en_sk=en_sk,
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
    w_slice = idx["w"]
    x_slice = idx["x"]
    y_nr_slice = idx["y_nr"]
    j_nr_slice = idx["j_nr"]
    z_sk_slice = idx["z_sk"]
    ca_slice = idx["ca"]
    atp_slice = idx["atp"]
    na_i_slice = idx["na_i"]
    k_o_slice = idx["k_o"]
    dfilter_primary_idx = idx["dfilter_primary"]
    dfilter_secondary_idx = idx["dfilter_secondary"]

    for i in range(n_comp):
        v_row = v_slice.start + i
        # Axial dependencies from Laplacian row.
        for p in range(int(l_indptr[i]), int(l_indptr[i + 1])):
            v_col = int(l_indices[p])
            sp[v_row, v_slice.start + v_col] = 1.0

        # Local channel-state dependencies in voltage equation.
        # Skip m/h→V entries for compartments with negligible Na conductance
        # (e.g. demyelinated trunk, gNa_trunk_mult≪1).  Near-zero columns in
        # the Na-gating block cause SuperLU to produce an exactly-singular LU
        # factor (numerical cancellation across identical passive rows).
        # K-channel n-gate is retained because gK is always full-density.
        _has_active_na = (
            gNa_v is None or float(gNa_v[i]) >= _gna_sparsity_threshold
        )
        if _has_active_na:
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
        if w_slice is not None:
            sp[v_row, w_slice.start + i] = 1.0
        if x_slice is not None:
            sp[v_row, x_slice.start + i] = 1.0
        if y_nr_slice is not None:
            sp[v_row, y_nr_slice.start + i] = 1.0
            sp[v_row, j_nr_slice.start + i] = 1.0
        if z_sk_slice is not None:
            sp[v_row, z_sk_slice.start + i] = 1.0
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

        if w_slice is not None:
            w_row = w_slice.start + i
            sp[w_row, v_row] = 1.0
            sp[w_row, w_row] = 1.0

        if x_slice is not None:
            x_row = x_slice.start + i
            sp[x_row, v_row] = 1.0
            sp[x_row, x_row] = 1.0

        if y_nr_slice is not None:
            y_row = y_nr_slice.start + i
            j_row = j_nr_slice.start + i
            sp[y_row, v_row] = 1.0
            sp[y_row, y_row] = 1.0
            sp[j_row, v_row] = 1.0
            sp[j_row, j_row] = 1.0

        if z_sk_slice is not None:
            zsk_row = z_sk_slice.start + i
            sp[zsk_row, zsk_row] = 1.0  # dz/dt depends on z
            if ca_slice is not None:
                sp[zsk_row, ca_slice.start + i] = 1.0  # dz/dt depends on Ca

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
        if atp_slice is not None:
            atp_row = atp_slice.start + i
            sp[atp_row, atp_row] = 1.0
            sp[atp_row, v_row] = 1.0
        if na_i_slice is not None:
            nai_row = na_i_slice.start + i
            sp[nai_row, nai_row] = 1.0
            sp[nai_row, v_row] = 1.0
            sp[nai_row, m_slice.start + i] = 1.0
            sp[nai_row, h_slice.start + i] = 1.0
        if k_o_slice is not None:
            ko_row = k_o_slice.start + i
            sp[ko_row, ko_row] = 1.0
            sp[ko_row, v_row] = 1.0
            sp[ko_row, n_slice.start + i] = 1.0

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
    zero_vec_cache: dict[int, np.ndarray] = {}
    ca_rest_vec_cache: dict[tuple[int, float], np.ndarray] = {}
    state_slices_cache: dict[tuple[int, bool, bool, bool, bool, bool, int, int, bool, bool, bool, bool, bool], tuple[dict[str, slice | int | None], int]] = {}

    def _zero_vec(n: int) -> np.ndarray:
        v = zero_vec_cache.get(n)
        if v is None:
            v = np.zeros(n, dtype=float)
            zero_vec_cache[n] = v
        return v

    def _ca_rest_vec(n: int, ca_rest_scalar: float) -> np.ndarray:
        key = (n, float(ca_rest_scalar))
        v = ca_rest_vec_cache.get(key)
        if v is None:
            v = np.full(n, float(ca_rest_scalar), dtype=float)
            ca_rest_vec_cache[key] = v
        return v

    def _const_vec(n: int, scalar: float) -> np.ndarray:
        return np.full(n, float(scalar), dtype=float)

    def _cached_state_slices(
        n_comp_local: int,
        en_ih_local: bool,
        en_ica_local: bool,
        en_ia_local: bool,
        dyn_ca_local: bool,
        dyn_atp_local: bool,
        use_dfilter_primary_local: int,
        use_dfilter_secondary_local: int,
        en_itca_local: bool,
        en_im_local: bool,
        en_nap_local: bool,
        en_nar_local: bool,
        en_sk_local: bool,
    ) -> tuple[dict[str, slice | int | None], int]:
        key = (
            int(n_comp_local),
            bool(en_ih_local),
            bool(en_ica_local),
            bool(en_ia_local),
            bool(dyn_ca_local),
            bool(dyn_atp_local),
            int(use_dfilter_primary_local),
            int(use_dfilter_secondary_local),
            bool(en_itca_local),
            bool(en_im_local),
            bool(en_nap_local),
            bool(en_nar_local),
            bool(en_sk_local),
        )
        out = state_slices_cache.get(key)
        if out is None:
            out = _state_slices(
                n_comp_local,
                en_ih_local,
                en_ica_local,
                en_ia_local,
                dyn_ca_local,
                dyn_atp_local,
                use_dfilter_primary_local,
                use_dfilter_secondary_local,
                en_itca=en_itca_local,
                en_im=en_im_local,
                en_nap=en_nap_local,
                en_nar=en_nar_local,
                en_sk=en_sk_local,
            )
            state_slices_cache[key] = out
        return out

    def _set(row, col, val):
        k = pos.get((row, col))
        if k is not None:
            J_csr.data[k] = val

    def _add(row, col, val):
        k = pos.get((row, col))
        if k is not None:
            J_csr.data[k] += val

    def jac_fn(
        t, y, physics_params
    ):
        """Jacobian function using structured PhysicsParams."""
        # Unpack physics parameters
        n_comp = physics_params.n_comp
        en_ih = physics_params.en_ih
        en_ica = physics_params.en_ica
        en_ia = physics_params.en_ia
        en_sk = physics_params.en_sk
        dyn_ca = physics_params.dyn_ca
        dyn_atp = physics_params.dyn_atp
        en_itca = physics_params.en_itca
        en_im = physics_params.en_im
        en_nap = physics_params.en_nap
        en_nar = physics_params.en_nar
        
        # Unpack conductance matrix
        (gna_v, gk_v, gl_v, gih_v, gca_v, ga_v, gsk_v, 
         gtca_v, gim_v, gnap_v, gnar_v) = unpack_conductances(physics_params.gbar_mat, n_comp)
        
        # Reversal potentials
        ena = physics_params.ena
        ek = physics_params.ek
        el = physics_params.el
        eih = physics_params.eih
        
        # Morphology and axial coupling
        cm_v = physics_params.cm_v
        l_data = physics_params.l_data
        l_indices = physics_params.l_indices
        l_indptr = physics_params.l_indptr
        
        # Unpack temperature scaling
        (phi_na, phi_k, phi_ih, phi_ca, phi_ia, phi_tca, 
         phi_im, phi_nap, phi_nar) = unpack_temperature_scaling(physics_params.phi_mat, n_comp)
        
        # Environment and calcium (packed env_params is the single source of truth)
        (
            t_kelvin,
            ca_ext,
            ca_rest,
            tau_ca,
            mg_ext,
            tau_sk,
            _im_speed_multiplier,
            _g_katp_max,
            _katp_kd_atp_mM,
            _atp_max_mM,
            _atp_synthesis_rate,
            na_i_rest_mM,
            na_ext_mM,
            k_i_mM,
            k_o_rest_mM,
            ion_drift_gain,
            k_o_clearance_tau_ms,
        ) = unpack_env_params(physics_params.env_params)
        b_ca = physics_params.b_ca
        
        # Stimulation parameters (needed for state slice calculation)
        stype = physics_params.stype
        iext = physics_params.iext
        t0 = physics_params.t0
        td = physics_params.td
        atau = physics_params.atau
        zap_f0_hz = physics_params.zap_f0_hz
        zap_f1_hz = physics_params.zap_f1_hz
        event_times_arr = physics_params.event_times_arr
        n_events = physics_params.n_events
        stim_comp = physics_params.stim_comp
        stim_mode = physics_params.stim_mode
        use_dfilter_primary = physics_params.use_dfilter_primary
        dfilter_attenuation = physics_params.dfilter_attenuation
        dfilter_tau_ms = physics_params.dfilter_tau_ms
        
        # Secondary stimulation
        dual_stim_enabled = physics_params.dual_stim_enabled
        stype_2 = physics_params.stype_2
        iext_2 = physics_params.iext_2
        t0_2 = physics_params.t0_2
        td_2 = physics_params.td_2
        atau_2 = physics_params.atau_2
        zap_f0_hz_2 = physics_params.zap_f0_hz_2
        zap_f1_hz_2 = physics_params.zap_f1_hz_2
        stim_comp_2 = physics_params.stim_comp_2
        stim_mode_2 = physics_params.stim_mode_2
        use_dfilter_secondary = physics_params.use_dfilter_secondary
        dfilter_attenuation_2 = physics_params.dfilter_attenuation_2
        dfilter_tau_ms_2 = physics_params.dfilter_tau_ms_2
        # Zero all entries
        J_csr.data[:] = 0.0

        idx = state_slices_from_offsets(physics_params.state_offsets, n_comp)
        n_state = int(physics_params.state_offsets.n_state)

        v = y[idx["v"]]
        m = y[idx["m"]]
        h = y[idx["h"]]
        n = y[idx["n"]]
        zero_v = _zero_vec(n_comp)
        r = y[idx["r"]] if idx["r"] is not None else zero_v
        s = y[idx["s"]] if idx["s"] is not None else zero_v
        u = y[idx["u"]] if idx["u"] is not None else zero_v
        a = y[idx["a"]] if idx["a"] is not None else zero_v
        b = y[idx["b"]] if idx["b"] is not None else zero_v
        w_g = y[idx["w"]] if idx["w"] is not None else zero_v
        x_g = y[idx["x"]] if idx["x"] is not None else zero_v
        y_nr = y[idx["y_nr"]] if idx["y_nr"] is not None else zero_v
        j_nr = y[idx["j_nr"]] if idx["j_nr"] is not None else zero_v
        ca_i = y[idx["ca"]] if idx["ca"] is not None else _ca_rest_vec(n_comp, ca_rest)
        atp_i = y[idx["atp"]] if idx["atp"] is not None else _zero_vec(n_comp)
        na_i = y[idx["na_i"]] if idx["na_i"] is not None else _const_vec(n_comp, na_i_rest_mM)
        k_o = y[idx["k_o"]] if idx["k_o"] is not None else _const_vec(n_comp, k_o_rest_mM)

        k_nernst = (R_GAS * t_kelvin / (2.0 * F_CONST)) * 1000.0
        mono_nernst = (R_GAS * t_kelvin / F_CONST) * 1000.0
        if dyn_atp and idx["na_i"] is not None:
            na_safe = np.maximum(na_i, max(NA_I_MIN_M_M, 1e-12))
            ena_v = mono_nernst * np.log(na_ext_mM / na_safe)
            dena_dna = -mono_nernst / na_safe
        else:
            ena_v = _const_vec(n_comp, ena)
            dena_dna = zero_v
        if dyn_atp and idx["k_o"] is not None:
            k_safe = np.maximum(k_o, max(K_O_MIN_M_M, 1e-12))
            ek_v = mono_nernst * np.log(k_i_mM / k_safe)
            dek_dko = -mono_nernst / k_safe
        else:
            ek_v = _const_vec(n_comp, ek)
            dek_dko = zero_v

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
            p_g = y[idx["p"]] if idx["p"] is not None else zero_v
            q_g = y[idx["q"]] if idx["q"] is not None else zero_v
            amt_v = am_TCa(v);  bmt_v = bm_TCa(v)
            aht_v = ah_TCa(v);  bht_v = bh_TCa(v)
            damt_v = _rate_derivative(am_TCa, v); dbmt_v = _rate_derivative(bm_TCa, v)
            daht_v = _rate_derivative(ah_TCa, v); dbht_v = _rate_derivative(bh_TCa, v)
        if en_im:
            awm_v = aw_IM(v);  bwm_v = bw_IM(v)
            dawm_v = _rate_derivative(aw_IM, v); dbwm_v = _rate_derivative(bw_IM, v)
        if en_nap:
            axp_v = ax_NaP(v);  bxp_v = bx_NaP(v)
            daxp_v = _rate_derivative(ax_NaP, v); dbxp_v = _rate_derivative(bx_NaP, v)
        if en_nar:
            ayr_v = ay_NaR(v);  byr_v = by_NaR(v)
            ajr_v = aj_NaR(v);  bjr_v = bj_NaR(v)
            dayr_v = _rate_derivative(ay_NaR, v); dbyr_v = _rate_derivative(by_NaR, v)
            dajr_v = _rate_derivative(aj_NaR, v); dbjr_v = _rate_derivative(bj_NaR, v)

        for i in range(n_comp):
            v_row = idx["v"].start + i
            cm = max(cm_v[i], 1e-12)  # Guard against zero capacitance
            ena_i = ena_v[i]
            ek_i = ek_v[i]
            dena_dna_i = dena_dna[i]
            dek_dko_i = dek_dko[i]

            # Axial coupling
            for p in range(int(l_indptr[i]), int(l_indptr[i + 1])):
                col = int(l_indices[p])
                _add(v_row, idx["v"].start + col, l_data[p] / cm)

            d_iion_dv = gl_v[i] + gna_v[i] * (m[i] ** 3) * h[i] + gk_v[i] * (n[i] ** 4)
            d_iion_dm = gna_v[i] * 3.0 * (m[i] ** 2) * h[i] * (v[i] - ena_i)
            d_iion_dh = gna_v[i] * (m[i] ** 3) * (v[i] - ena_i)
            d_iion_dn = gk_v[i] * 4.0 * (n[i] ** 3) * (v[i] - ek_i)
            d_iion_dca = 0.0
            d_iion_dna = 0.0
            d_iion_dko = 0.0
            d_iion_datp = 0.0
            d_iion_dz = 0.0

            if dyn_atp and idx["na_i"] is not None:
                d_iion_dna += gna_v[i] * (m[i] ** 3) * h[i] * (-dena_dna_i)
            if dyn_atp and idx["k_o"] is not None:
                d_iion_dko += gk_v[i] * (n[i] ** 4) * (-dek_dko_i)

            if en_ih:
                d_iion_dv += gih_v[i] * r[i]
                _add(v_row, idx["r"].start + i, -(gih_v[i] * (v[i] - eih)) / cm)

            i_ca_val = 0.0
            if en_ica:
                if dyn_ca:
                    ca_safe = min(max(ca_i[i], CA_I_MIN_M_M), CA_I_MAX_M_M)
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
                _add(v_row, idx["a"].start + i, -(ga_v[i] * b[i] * (v[i] - ek_i)) / cm)
                _add(v_row, idx["b"].start + i, -(ga_v[i] * a[i] * (v[i] - ek_i)) / cm)
                if dyn_atp and idx["k_o"] is not None:
                    d_iion_dko += ga_v[i] * a[i] * b[i] * (-dek_dko_i)

            i_tca_val = 0.0
            if en_itca and idx["p"] is not None:
                if dyn_ca:
                    ca_safe_t = min(max(ca_i[i], CA_I_MIN_M_M), CA_I_MAX_M_M)
                    eca_t = k_nernst * np.log(ca_ext / ca_safe_t)
                    deca_t_dca = -k_nernst / ca_safe_t
                else:
                    eca_t = 120.0
                    deca_t_dca = 0.0
                # If ICa already computed eca for this compartment, reuse it
                if en_ica and dyn_ca:
                    ca_safe_t = min(max(ca_i[i], CA_I_MIN_M_M), CA_I_MAX_M_M)
                    eca_t = k_nernst * np.log(ca_ext / ca_safe_t)
                    deca_t_dca = -k_nernst / ca_safe_t
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

            if en_sk and idx["z_sk"] is not None:
                z_sk_val = y[idx["z_sk"].start + i]
                d_iion_dv += gsk_v[i] * z_sk_val
                d_iion_dz = gsk_v[i] * (v[i] - ek_i)
                _add(v_row, idx["z_sk"].start + i, -d_iion_dz / cm)
                if dyn_atp and idx["k_o"] is not None:
                    d_iion_dko += gsk_v[i] * z_sk_val * (-dek_dko_i)

            if en_im and idx["w"] is not None:
                d_iion_dv += gim_v[i] * w_g[i]
                _add(v_row, idx["w"].start + i, -(gim_v[i] * (v[i] - ek_i)) / cm)
                if dyn_atp and idx["k_o"] is not None:
                    d_iion_dko += gim_v[i] * w_g[i] * (-dek_dko_i)

            if en_nap and idx["x"] is not None:
                d_iion_dv += gnap_v[i] * x_g[i]
                _add(v_row, idx["x"].start + i, -(gnap_v[i] * (v[i] - ena_i)) / cm)
                if dyn_atp and idx["na_i"] is not None:
                    d_iion_dna += gnap_v[i] * x_g[i] * (-dena_dna_i)

            if en_nar and idx["y_nr"] is not None:
                d_iion_dv += gnar_v[i] * y_nr[i] * j_nr[i]
                _add(v_row, idx["y_nr"].start + i, -(gnar_v[i] * j_nr[i] * (v[i] - ena_i)) / cm)
                _add(v_row, idx["j_nr"].start + i, -(gnar_v[i] * y_nr[i] * (v[i] - ena_i)) / cm)
                if dyn_atp and idx["na_i"] is not None:
                    d_iion_dna += gnar_v[i] * y_nr[i] * j_nr[i] * (-dena_dna_i)

            if dyn_atp and idx["atp"] is not None:
                kd_safe = max(_katp_kd_atp_mM, 1e-12)
                atp_safe = max(atp_i[i], 1e-12)
                atp_ratio = atp_safe / kd_safe
                g_katp = _g_katp_max / (1.0 + atp_ratio * atp_ratio)
                dg_katp_datp = (-2.0 * _g_katp_max * atp_safe / (kd_safe * kd_safe)) / ((1.0 + atp_ratio * atp_ratio) ** 2)
                d_iion_dv += g_katp
                d_iion_datp += dg_katp_datp * (v[i] - ek_i)
                if dyn_atp and idx["k_o"] is not None:
                    d_iion_dko += g_katp * (-dek_dko_i)

            _add(v_row, idx["v"].start + i, -d_iion_dv / cm)
            _add(v_row, idx["m"].start + i, -d_iion_dm / cm)
            _add(v_row, idx["h"].start + i, -d_iion_dh / cm)
            _add(v_row, idx["n"].start + i, -d_iion_dn / cm)
            if dyn_ca and idx["ca"] is not None:
                _add(v_row, idx["ca"].start + i, -d_iion_dca / cm)
            if dyn_atp and idx["na_i"] is not None:
                _add(v_row, idx["na_i"].start + i, -d_iion_dna / cm)
            if dyn_atp and idx["k_o"] is not None:
                _add(v_row, idx["k_o"].start + i, -d_iion_dko / cm)
            if dyn_atp and idx["atp"] is not None:
                _add(v_row, idx["atp"].start + i, -d_iion_datp / cm)
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
            _set(m_row, v_col, phi_na[i] * (dam_v[i] * (1.0 - m[i]) - dbm_v[i] * m[i]))
            _set(m_row, m_row, -phi_na[i] * (am_v[i] + bm_v[i]))
            _set(h_row, v_col, phi_na[i] * (dah_v[i] * (1.0 - h[i]) - dbh_v[i] * h[i]))
            _set(h_row, h_row, -phi_na[i] * (ah_v[i] + bh_v[i]))
            _set(n_row, v_col, phi_k[i] * (dan_v[i] * (1.0 - n[i]) - dbn_v[i] * n[i]))
            _set(n_row, n_row, -phi_k[i] * (an_v[i] + bn_v[i]))

            if en_ih and idx["r"] is not None:
                r_row = idx["r"].start + i
                _set(r_row, v_col, phi_ih[i] * (dar_v[i] * (1.0 - r[i]) - dbr_v[i] * r[i]))
                _set(r_row, r_row, -phi_ih[i] * (ar_v[i] + br_v[i]))

            if en_ica and idx["s"] is not None:
                s_row = idx["s"].start + i
                u_row = idx["u"].start + i
                _set(s_row, v_col, phi_ca[i] * (das_v[i] * (1.0 - s[i]) - dbs_v[i] * s[i]))
                _set(s_row, s_row, -phi_ca[i] * (as_v[i] + bs_v[i]))
                _set(u_row, v_col, phi_ca[i] * (dau_v[i] * (1.0 - u[i]) - dbu_v[i] * u[i]))
                _set(u_row, u_row, -phi_ca[i] * (au_v[i] + bu_v[i]))

            if en_ia and idx["a"] is not None:
                a_row = idx["a"].start + i
                b_row = idx["b"].start + i
                # A-current (IA) is a K channel — use phi_k (matches rhs.py)
                _set(a_row, v_col, phi_k[i] * (daa_v[i] * (1.0 - a[i]) - dba_v[i] * a[i]))
                _set(a_row, a_row, -phi_k[i] * (aa_v[i] + ba_v[i]))
                _set(b_row, v_col, phi_k[i] * (dab_v[i] * (1.0 - b[i]) - dbb_v[i] * b[i]))
                _set(b_row, b_row, -phi_k[i] * (ab_v[i] + bb_v[i]))

            if en_itca and idx["p"] is not None:
                p_row = idx["p"].start + i
                q_row = idx["q"].start + i
                # T-type Ca is a Ca channel — use phi_ca (matches rhs.py)
                _set(p_row, v_col, phi_ca[i] * (damt_v[i] * (1.0 - p_g[i]) - dbmt_v[i] * p_g[i]))
                _set(p_row, p_row, -phi_ca[i] * (amt_v[i] + bmt_v[i]))
                _set(q_row, v_col, phi_ca[i] * (daht_v[i] * (1.0 - q_g[i]) - dbht_v[i] * q_g[i]))
                _set(q_row, q_row, -phi_ca[i] * (aht_v[i] + bht_v[i]))

            if en_im and idx["w"] is not None:
                w_row = idx["w"].start + i
                # M-type K (IM) is a K channel — use phi_k (matches rhs.py)
                _set(w_row, v_col, phi_k[i] * (dawm_v[i] * (1.0 - w_g[i]) - dbwm_v[i] * w_g[i]))
                _set(w_row, w_row, -phi_k[i] * (awm_v[i] + bwm_v[i]))

            if en_nap and idx["x"] is not None:
                x_row = idx["x"].start + i
                _set(x_row, v_col, phi_nap[i] * (daxp_v[i] * (1.0 - x_g[i]) - dbxp_v[i] * x_g[i]))
                _set(x_row, x_row, -phi_nap[i] * (axp_v[i] + bxp_v[i]))

            if en_nar and idx["y_nr"] is not None:
                y_row = idx["y_nr"].start + i
                j_row = idx["j_nr"].start + i
                _set(y_row, v_col, phi_nar[i] * (dayr_v[i] * (1.0 - y_nr[i]) - dbyr_v[i] * y_nr[i]))
                _set(y_row, y_row, -phi_nar[i] * (ayr_v[i] + byr_v[i]))
                _set(j_row, v_col, phi_nar[i] * (dajr_v[i] * (1.0 - j_nr[i]) - dbjr_v[i] * j_nr[i]))
                _set(j_row, j_row, -phi_nar[i] * (ajr_v[i] + bjr_v[i]))

            # SK gate: dz/dt = (z_inf - z) / tau_eff, tau_eff = tau_sk / phi_k
            if en_sk and idx["z_sk"] is not None:
                zsk_row = idx["z_sk"].start + i
                tau_eff_sk = max(tau_sk, 1e-12) / max(phi_k[i], 1e-12)
                _set(zsk_row, zsk_row, -1.0 / tau_eff_sk)
                if dyn_ca and idx["ca"] is not None:
                    ca_safe = min(max(ca_i[i], CA_I_MIN_M_M), CA_I_MAX_M_M)
                    ca_sk = min(max(ca_safe * 10.0, CA_I_MIN_M_M), 1.0)
                    ca_safe = max(ca_sk, 1e-12)
                    kd = 0.0004
                    q = (kd / ca_safe) ** 4
                    dz_inf_dca = (4.0 * (kd ** 4) / (ca_safe ** 5)) / ((1.0 + q) ** 2)
                    _set(zsk_row, idx["ca"].start + i, 10.0 * dz_inf_dca / tau_eff_sk)

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

            if dyn_atp and idx["atp"] is not None:
                atp_row = idx["atp"].start + i
                # ATP relaxation time constant ~100 ms (v11.6: removed ATP_PUMP_FAILURE_THRESHOLD dependency)
                _set(atp_row, atp_row, -1.0 / 100.0)
            if dyn_atp and idx["na_i"] is not None:
                na_row = idx["na_i"].start + i
                _set(na_row, na_row, -max(ion_drift_gain, 1e-9))
            if dyn_atp and idx["k_o"] is not None:
                k_row = idx["k_o"].start + i
                _set(k_row, k_row, -(1.0 / max(k_o_clearance_tau_ms, 1e-12) + max(ion_drift_gain, 1e-9)))

        if idx["dfilter_primary"] is not None:
            dr = int(idx["dfilter_primary"])
            _set(dr, dr, -1.0 / dfilter_tau_ms if dfilter_tau_ms > 0.0 else 0.0)
        if idx["dfilter_secondary"] is not None:
            dr2 = int(idx["dfilter_secondary"])
            _set(dr2, dr2, -1.0 / dfilter_tau_ms_2 if dfilter_tau_ms_2 > 0.0 else 0.0)

        return J_csr

    return jac_fn


# Legacy interface — REMOVED.  rhs_contract.py was deleted in v11.0.
# Use make_analytic_jacobian(sparsity) which returns a closure over PhysicsParams.
def analytic_sparse_jacobian(*args, **kwargs):
    """Removed in v11.0: positional-arg contract (rhs_contract.py) has been deleted.

    Use make_analytic_jacobian(sparsity) instead, which accepts a PhysicsParams struct.
    """
    raise NotImplementedError(
        "analytic_sparse_jacobian was removed in v11.0.  "
        "Use make_analytic_jacobian(sparsity) which returns a closure over PhysicsParams."
    )


def _analytic_sparse_jacobian_impl(*args, **kwargs):
    """Internal implementation — kept for reference; not called by any active path."""
    rhs = args[2] if len(args) > 2 else {}
    n_comp = rhs["n_comp"]
    en_ih = rhs["en_ih"]
    en_ica = rhs["en_ica"]
    en_ia = rhs["en_ia"]
    en_sk = rhs["en_sk"]
    dyn_ca = rhs["dyn_ca"]
    dyn_atp = rhs.get("dyn_atp", False)
    en_itca = rhs["en_itca"]
    en_im = rhs["en_im"]
    en_nap = rhs["en_nap"]
    en_nar = rhs["en_nar"]
    stim1_vec = rhs["stim1_vec"]
    stim2_vec = rhs["stim2_vec"]
    use_dfp = int(stim1_vec[9])
    use_dfs = int(stim2_vec[10])
    cache_key = (
        n_comp, en_ih, en_ica, en_ia, en_sk, dyn_ca, dyn_atp, en_itca, en_im, en_nap, en_nar,
        use_dfp, use_dfs, _sparse_structure_signature(rhs["l_indices"], rhs["l_indptr"]),
    )
    with _LEGACY_JACOBIAN_CACHE_LOCK:
        fn = _LEGACY_JACOBIAN_CACHE.get(cache_key)
    if fn is None:
        sp = build_jacobian_sparsity(
            n_comp, en_ih, en_ica, en_ia, en_sk, dyn_ca, dyn_atp,
            rhs["l_indices"], rhs["l_indptr"],
            use_dfp, use_dfs,
            en_itca=en_itca, en_im=en_im, en_nap=en_nap, en_nar=en_nar,
        )
        built_fn = make_analytic_jacobian(sp)
        with _LEGACY_JACOBIAN_CACHE_LOCK:
            fn = _LEGACY_JACOBIAN_CACHE.get(cache_key)
            if fn is None:
                _evict_oldest_if_needed()
                _LEGACY_JACOBIAN_CACHE[cache_key] = built_fn
                fn = built_fn
    return fn(*args)


def clear_legacy_jacobian_cache() -> None:
    """Clear cached legacy Jacobian callables (test/debug helper)."""
    with _LEGACY_JACOBIAN_CACHE_LOCK:
        _LEGACY_JACOBIAN_CACHE.clear()
