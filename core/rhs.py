import numpy as np
from numba import njit, float64, int32
from .kinetics import *
from .dual_stimulation import (
    distributed_stimulus_current_for_comp,
)

# Константы | Constants
F_CONST = 96485.33
R_GAS = 8.314
TEMP_ZERO = 273.15

@njit(float64(float64, float64, float64), cache=True)
def nernst_ca_ion(ca_i, ca_ext, t_kelvin):
    """Динамический потенциал Нернста для Кальция (z=2). | Dynamic Nernst potential for Calcium (z=2)."""
    ca_i_safe = max(ca_i, 1e-9)
    return (R_GAS * t_kelvin / (2.0 * F_CONST)) * np.log(ca_ext / ca_i_safe) * 1000.0

@njit(float64(float64, float64, float64, float64), cache=True)
def _biexp_waveform(t, t0, tau_rise, tau_decay):
    """Normalised dual-exponential waveform, peak = 1.0 at t_peak."""
    if t < t0:
        return 0.0
    dt = t - t0
    t_peak = tau_rise * tau_decay / (tau_decay - tau_rise) * np.log(tau_decay / tau_rise)
    norm = np.exp(-t_peak / tau_decay) - np.exp(-t_peak / tau_rise)
    return (np.exp(-dt / tau_decay) - np.exp(-dt / tau_rise)) / norm

@njit(float64(float64, int32, float64, float64, float64, float64, float64, float64), cache=True)
def get_stim_current(t, stype, iext, t0, td, atau, zap_f0_hz, zap_f1_hz):
    """Stimulus waveform for all types.

    For synaptic types (stype >= 4): returns CONDUCTANCE waveform g(t)
    scaled by |iext| (= g_max in mS/cm²). The RHS multiplies by (V - E_syn).
    For const/pulse/alpha: returns current directly (µA/cm²).
    """
    if stype == 1:  # pulse
        return iext if t0 <= t <= t0 + td else 0.0
    elif stype == 2:  # alpha (EPSC) — current-based
        if t < t0:
            return 0.0
        dt = (t - t0) / atau
        return iext * dt * np.exp(1.0 - dt)
    elif stype == 4:  # AMPA (conductance-based)
        return abs(iext) * _biexp_waveform(t, t0, 0.5, 3.0)
    elif stype == 5:  # NMDA (conductance-based, Mg block applied in RHS)
        return abs(iext) * _biexp_waveform(t, t0, 5.0, 80.0)
    elif stype == 6:  # GABA-A (conductance-based)
        return abs(iext) * _biexp_waveform(t, t0, 1.0, 7.0)
    elif stype == 7:  # GABA-B (conductance-based)
        return abs(iext) * _biexp_waveform(t, t0, 50.0, 300.0)
    elif stype == 8:  # Kainate (conductance-based)
        return abs(iext) * _biexp_waveform(t, t0, 1.5, 12.0)
    elif stype == 9:  # Nicotinic ACh (conductance-based)
        return abs(iext) * _biexp_waveform(t, t0, 3.0, 25.0)
    elif stype == 10:  # ZAP/Chirp current (frequency sweep)
        if td <= 0.0 or t < t0 or t > (t0 + td):
            return 0.0
        dt = t - t0  # ms
        k_hz_per_ms = (zap_f1_hz - zap_f0_hz) / td
        phase = 2.0 * np.pi * ((zap_f0_hz * dt / 1000.0) + 0.5 * (k_hz_per_ms * dt * dt / 1000.0))
        return iext * np.sin(phase)
    # Default: const (stype == 0)
    return iext

@njit(float64(float64, float64), cache=True)
def nmda_mg_block(V, Mg_ext):
    """Voltage-dependent Mg²⁺ block of NMDA receptors.

    B(V) = 1 / (1 + [Mg²⁺]/3.57 * exp(-0.062 * V))
    Reference: Jahr & Stevens 1990, J Neurosci 10:1830
    """
    return 1.0 / (1.0 + (Mg_ext / 3.57) * np.exp(-0.062 * V))

@njit(float64(int32), cache=True)
def _get_syn_reversal(stype):
    """Return synaptic reversal potential for conductance-based types."""
    if stype == 6:   # GABA-A (Cl⁻, Bormann 1988)
        return -75.0
    elif stype == 7:  # GABA-B (K⁺ via GIRK, Lüscher 1997)
        return -95.0
    # Excitatory: AMPA(4), NMDA(5), Kainate(8), Nicotinic(9) — cation, ~0 mV
    return 0.0

@njit(cache=True)
def get_event_driven_conductance(t, stype, iext, event_times, n_events):
    """Sum biexponential conductance [mS/cm²] from all events in the synaptic queue.

    Stage 6.3 — event-driven synaptic stimulation (preparation for network connectivity).
    Returns 0.0 if n_events == 0 or stype is not a conductance-based type.
    """
    if n_events == 0:
        return 0.0
    if stype == 4:          # AMPA
        tau_r, tau_d = 0.5, 3.0
    elif stype == 5:        # NMDA
        tau_r, tau_d = 5.0, 80.0
    elif stype == 6:        # GABA-A
        tau_r, tau_d = 1.0, 7.0
    elif stype == 7:        # GABA-B
        tau_r, tau_d = 50.0, 300.0
    elif stype == 8:        # Kainate
        tau_r, tau_d = 1.5, 12.0
    elif stype == 9:        # Nicotinic
        tau_r, tau_d = 3.0, 25.0
    else:
        return 0.0
    g = 0.0
    for k in range(n_events):
        g += _biexp_waveform(t, event_times[k], tau_r, tau_d)
    return abs(iext) * g


@njit(cache=True)
def rhs_multicompartment(
    t, y, n_comp,
    # Флаги включения каналов
    en_ih, en_ica, en_ia, en_sk, dyn_ca, en_itca, en_im, en_nap, en_nar,
    # Векторы проводимостей (уже с учетом AIS-множителей)
    gna_v, gk_v, gl_v, gih_v, gca_v, ga_v, gsk_v, gtca_v, gim_v, gnap_v, gnar_v,
    # Потенциалы реверсии
    ena, ek, el, eih, ea,
    # Морфология и среда
    cm_v, l_data, l_indices, l_indptr,
    phi_na, phi_k, phi_ih, phi_ca, phi_ia, phi_tca, phi_im, phi_nap, phi_nar,
    t_kelvin, ca_ext, ca_rest, tau_ca, b_ca, mg_ext, tau_sk,
    # Стимуляция (primary)
    stype, iext, t0, td, atau, zap_f0_hz, zap_f1_hz, event_times_arr, n_events, stim_comp, stim_mode,
    use_dfilter_primary, dfilter_attenuation, dfilter_tau_ms,
    # Dual stimulation (secondary) - optional
    dual_stim_enabled,
    stype_2, iext_2, t0_2, td_2, atau_2, zap_f0_hz_2, zap_f1_hz_2, stim_comp_2, stim_mode_2,
    use_dfilter_secondary, dfilter_attenuation_2, dfilter_tau_ms_2,
    # Pre-allocated output buffer (avoids heap allocation per solver step)
    dydt
):
    """
    Высокопроизводительное ядро ОДУ v10.1 (C-style scalar loop).
    Все токи рассчитываются как скаляры, без промежуточных аллокаций.
    dydt is a pre-allocated output array passed from the solver; zeroed here
    via a Numba loop (no numpy allocation inside @njit).
    """

    # --- Compute variable offsets in state vector y ---
    off_v = 0
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
        off_s = cursor
        cursor += n_comp
        off_u = cursor
        cursor += n_comp
    off_a = cursor
    off_b = cursor
    if en_ia:
        off_a = cursor
        cursor += n_comp
        off_b = cursor
        cursor += n_comp
    off_p = cursor   # T-type Ca activation
    off_q = cursor   # T-type Ca inactivation
    if en_itca:
        off_p = cursor
        cursor += n_comp
        off_q = cursor
        cursor += n_comp
    off_w = cursor   # M-type K activation
    if en_im:
        off_w = cursor
        cursor += n_comp
    off_x = cursor   # Persistent Na activation
    if en_nap:
        off_x = cursor
        cursor += n_comp
    off_y = cursor   # Resurgent Na activation
    off_j = cursor   # Resurgent Na inactivation/block
    if en_nar:
        off_y = cursor
        cursor += n_comp
        off_j = cursor
        cursor += n_comp
    off_zsk = cursor   # SK gate (delayed kinetics)
    if en_sk:
        off_zsk = cursor
        cursor += n_comp
    off_ca = cursor
    if dyn_ca:
        cursor += n_comp

    # Dendritic filter state offsets
    off_vfilt_primary = cursor
    if use_dfilter_primary == 1:
        cursor += 1
    off_vfilt_secondary = cursor
    if use_dfilter_secondary == 1:
        cursor += 1

    # --- Stimulus: compute once, apply to target compartments ---
    # For stype >= 4 (synaptic): base_current is conductance g(t) [mS/cm²]
    # and will be multiplied by (V - E_syn) per-compartment in the main loop.
    # For stype < 4 (const/pulse/alpha): base_current is raw current [µA/cm²].
    is_conductance_based = (stype >= 4)
    # Stage 6.3: event-driven mode if queue is non-empty and stim is synaptic
    if n_events > 0 and is_conductance_based:
        base_current = get_event_driven_conductance(t, stype, iext, event_times_arr, n_events)
    else:
        base_current = get_stim_current(t, stype, iext, t0, td, atau, zap_f0_hz, zap_f1_hz)
    e_syn = _get_syn_reversal(stype) if is_conductance_based else 0.0
    is_nmda = (stype == 5)

    v_filtered_primary = 0.0
    if use_dfilter_primary == 1:
        v_filtered_primary = y[off_vfilt_primary]

    v_filtered_secondary = 0.0
    if use_dfilter_secondary == 1:
        v_filtered_secondary = y[off_vfilt_secondary]

    d_vfiltered_dt_primary = 0.0
    if stim_mode == 2 and use_dfilter_primary == 1 and dfilter_tau_ms > 0.0:
        attenuated_current = dfilter_attenuation * base_current
        d_vfiltered_dt_primary = (attenuated_current - v_filtered_primary) / dfilter_tau_ms

    is_cond_2 = False
    e_syn_2 = 0.0
    is_nmda_2 = False
    d_vfiltered_dt_secondary = 0.0
    base_current_2 = 0.0
    if dual_stim_enabled == 1:
        is_cond_2 = (stype_2 >= 4)
        e_syn_2 = _get_syn_reversal(stype_2) if is_cond_2 else 0.0
        is_nmda_2 = (stype_2 == 5)
        base_current_2 = get_stim_current(t, stype_2, iext_2, t0_2, td_2, atau_2, zap_f0_hz_2, zap_f1_hz_2)
        if stim_mode_2 == 2 and use_dfilter_secondary == 1 and dfilter_tau_ms_2 > 0.0:
            attenuated_current_2 = dfilter_attenuation_2 * base_current_2
            d_vfiltered_dt_secondary = (attenuated_current_2 - v_filtered_secondary) / dfilter_tau_ms_2

    # --- Zero the pre-allocated output buffer (Numba-native; no heap alloc) ---
    for _k in range(len(dydt)):
        dydt[_k] = 0.0

    # --- Main C-style loop: all currents as scalars ---
    for i in range(n_comp):
        vi = y[off_v + i]
        mi = y[off_m + i]
        hi = y[off_h + i]
        ni = y[off_n + i]

        # Ionic currents (scalar accumulation)
        i_ion = gl_v[i] * (vi - el)
        i_ion += gna_v[i] * (mi * mi * mi) * hi * (vi - ena)
        i_ion += gk_v[i] * (ni * ni * ni * ni) * (vi - ek)

        i_ca_influx = 0.0  # only inward Ca for calcium dynamics

        if en_ih:
            ri = y[off_r + i]
            i_ion += gih_v[i] * ri * (vi - eih)

        if en_ica:
            si = y[off_s + i]
            ui = y[off_u + i]
            # Nernst for Ca: per-compartment when dynamic, else fixed
            if dyn_ca:
                ca_i_val = y[off_ca + i]
                eca_i = nernst_ca_ion(ca_i_val, ca_ext, t_kelvin)
            else:
                ca_i_val = ca_rest
                eca_i = 120.0
            i_ca_current = gca_v[i] * (si * si) * ui * (vi - eca_i)
            i_ion += i_ca_current
            # Only inward (negative) Ca current contributes to Ca accumulation
            if i_ca_current < 0.0:
                i_ca_influx = -i_ca_current

        if en_ia:
            ai = y[off_a + i]
            bi = y[off_b + i]
            i_ion += ga_v[i] * ai * bi * (vi - ea)

        # T-type Ca (low-threshold, Destexhe 1998)
        if en_itca:
            pi = y[off_p + i]
            qi = y[off_q + i]
            if dyn_ca:
                if not en_ica:  # eca_i not yet computed
                    ca_i_val = y[off_ca + i]
                    eca_i = nernst_ca_ion(ca_i_val, ca_ext, t_kelvin)
            else:
                if not en_ica:
                    eca_i = 120.0
            i_tca = gtca_v[i] * (pi * pi) * qi * (vi - eca_i)
            i_ion += i_tca
            if i_tca < 0.0:
                i_ca_influx += -i_tca

        if en_sk:
            zi = y[off_zsk + i]  # SK gate state variable (ODE-based)
            i_ion += gsk_v[i] * zi * (vi - ek)

        # M-type K (slow non-inactivating, KCNQ2/3)
        if en_im:
            wi = y[off_w + i]
            i_ion += gim_v[i] * wi * (vi - ek)

        # Persistent Na (subthreshold, non-inactivating)
        if en_nap:
            xi = y[off_x + i]
            i_ion += gnap_v[i] * xi * (vi - ena)

        # Resurgent Na (repolarization-activated window current)
        if en_nar:
            yi = y[off_y + i]
            ji = y[off_j + i]
            i_ion += gnar_v[i] * yi * ji * (vi - ena)

        # Axial coupling (sparse Laplacian row i)
        i_ax = 0.0
        for j_idx in range(l_indptr[i], l_indptr[i + 1]):
            col = l_indices[j_idx]
            i_ax += l_data[j_idx] * y[off_v + col]

        # Stimulus current: evaluate distributed contribution per-compartment.
        i_stim_primary = distributed_stimulus_current_for_comp(
            i, n_comp, base_current, stim_comp, stim_mode,
            use_dfilter_primary, dfilter_attenuation, v_filtered_primary,
        )
        if is_conductance_based:
            g_syn = i_stim_primary  # distributed conductance [mS/cm²]
            if is_nmda:
                g_syn *= nmda_mg_block(vi, mg_ext)
            i_stim_eff = -(g_syn * (vi - e_syn))
        else:
            i_stim_eff = i_stim_primary

        # Dual stim contribution
        if dual_stim_enabled == 1:
            i_stim_secondary = distributed_stimulus_current_for_comp(
                i, n_comp, base_current_2, stim_comp_2, stim_mode_2,
                use_dfilter_secondary, dfilter_attenuation_2, v_filtered_secondary,
            )
            if is_cond_2:
                g2 = i_stim_secondary
                if is_nmda_2:
                    g2 *= nmda_mg_block(vi, mg_ext)
                i_stim_eff -= g2 * (vi - e_syn_2)
            else:
                i_stim_eff += i_stim_secondary

        # dV/dt
        dydt[off_v + i] = (i_stim_eff - i_ion + i_ax) / cm_v[i]

        # Gate derivatives (HH core) — per-compartment Q10 (Stage 6.2: thermal gradient)
        dydt[off_m + i] = phi_na[i] * (am(vi) * (1.0 - mi) - bm(vi) * mi)
        dydt[off_h + i] = phi_na[i] * (ah(vi) * (1.0 - hi) - bh(vi) * hi)
        dydt[off_n + i] = phi_k[i] * (an(vi) * (1.0 - ni) - bn(vi) * ni)

        # Optional gate derivatives
        if en_ih:
            ri = y[off_r + i]
            dydt[off_r + i] = phi_ih[i] * (ar_Ih(vi) * (1.0 - ri) - br_Ih(vi) * ri)

        if en_ica:
            si = y[off_s + i]
            ui = y[off_u + i]
            dydt[off_s + i] = phi_ca[i] * (as_Ca(vi) * (1.0 - si) - bs_Ca(vi) * si)
            dydt[off_u + i] = phi_ca[i] * (au_Ca(vi) * (1.0 - ui) - bu_Ca(vi) * ui)

        if en_ia:
            ai = y[off_a + i]
            bi = y[off_b + i]
            dydt[off_a + i] = phi_ia[i] * (aa_IA(vi) * (1.0 - ai) - ba_IA(vi) * ai)
            dydt[off_b + i] = phi_ia[i] * (ab_IA(vi) * (1.0 - bi) - bb_IA(vi) * bi)

        if en_itca:
            pi = y[off_p + i]
            qi = y[off_q + i]
            dydt[off_p + i] = phi_tca[i] * (am_TCa(vi) * (1.0 - pi) - bm_TCa(vi) * pi)
            dydt[off_q + i] = phi_tca[i] * (ah_TCa(vi) * (1.0 - qi) - bh_TCa(vi) * qi)

        if en_im:
            wi = y[off_w + i]
            dydt[off_w + i] = phi_im[i] * (aw_IM(vi) * (1.0 - wi) - bw_IM(vi) * wi)

        if en_nap:
            xi = y[off_x + i]
            dydt[off_x + i] = phi_nap[i] * (ax_NaP(vi) * (1.0 - xi) - bx_NaP(vi) * xi)

        if en_nar:
            yi = y[off_y + i]
            ji = y[off_j + i]
            dydt[off_y + i] = phi_nar[i] * (ay_NaR(vi) * (1.0 - yi) - by_NaR(vi) * yi)
            dydt[off_j + i] = phi_nar[i] * (aj_NaR(vi) * (1.0 - ji) - bj_NaR(vi) * ji)

        # SK gate ODE: dz/dt = (z_inf(Ca) - z) / tau_SK
        # Hirschberg et al. 1998, J Gen Physiol 111:565
        if en_sk:
            zi = y[off_zsk + i]
            if dyn_ca:
                ca_sk = y[off_ca + i]
            else:
                ca_sk = ca_rest
            z_inf = z_inf_SK(ca_sk)
            dydt[off_zsk + i] = (z_inf - zi) / tau_sk

        # Calcium dynamics
        if dyn_ca:
            ca_i_val = y[off_ca + i]
            dca = b_ca[i] * i_ca_influx - (ca_i_val - ca_rest) / tau_ca
            # Hard clamp: prevent negative calcium
            if ca_i_val < 1e-9 and dca < 0.0:
                dca = 0.0
            dydt[off_ca + i] = dca

    # Dendritic filter ODEs (outside main loop — not per-compartment)
    if use_dfilter_primary == 1:
        dydt[off_vfilt_primary] = d_vfiltered_dt_primary
    if use_dfilter_secondary == 1:
        dydt[off_vfilt_secondary] = d_vfiltered_dt_secondary

    return dydt
