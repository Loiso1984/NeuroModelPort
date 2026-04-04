import numpy as np
from numba import njit, float64, int32
from .kinetics import *
from .dual_stimulation import (
    apply_primary_stimulus_current,
    apply_secondary_stimulus_current,
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

@njit(float64(float64, int32, float64, float64, float64, float64), cache=True)
def get_stim_current(t, stype, iext, t0, td, atau):
    """Математика всех типов стимулов v10. | Mathematics of all stimulus types v10."""
    if stype == 1: # pulse
        return iext if t0 <= t <= t0 + td else 0.0
    elif stype == 2: # alpha (EPSC)
        if t < t0: return 0.0
        dt = (t - t0) / atau
        return iext * dt * np.exp(1.0 - dt)
    elif stype == 4:  # AMPA (fast excitatory)
        if t < t0:
            return 0.0
        dt = t - t0
        tau_rise = 0.5
        tau_decay = 3.0
        t_peak = tau_rise * tau_decay / (tau_decay - tau_rise) * np.log(tau_decay / tau_rise)
        norm = np.exp(-t_peak / tau_decay) - np.exp(-t_peak / tau_rise)
        return abs(iext) * (np.exp(-dt / tau_decay) - np.exp(-dt / tau_rise)) / norm
    elif stype == 5:  # NMDA (slow excitatory)
        if t < t0:
            return 0.0
        dt = t - t0
        tau_rise = 5.0
        tau_decay = 80.0
        t_peak = tau_rise * tau_decay / (tau_decay - tau_rise) * np.log(tau_decay / tau_rise)
        norm = np.exp(-t_peak / tau_decay) - np.exp(-t_peak / tau_rise)
        return abs(iext) * (np.exp(-dt / tau_decay) - np.exp(-dt / tau_rise)) / norm
    elif stype == 6:  # GABA-A (fast inhibitory)
        if t < t0:
            return 0.0
        dt = t - t0
        tau_rise = 1.0
        tau_decay = 7.0
        t_peak = tau_rise * tau_decay / (tau_decay - tau_rise) * np.log(tau_decay / tau_rise)
        norm = np.exp(-t_peak / tau_decay) - np.exp(-t_peak / tau_rise)
        return -abs(iext) * (np.exp(-dt / tau_decay) - np.exp(-dt / tau_rise)) / norm
    elif stype == 7:  # GABA-B (slow inhibitory)
        if t < t0:
            return 0.0
        dt = t - t0
        tau_rise = 50.0
        tau_decay = 300.0
        t_peak = tau_rise * tau_decay / (tau_decay - tau_rise) * np.log(tau_decay / tau_rise)
        norm = np.exp(-t_peak / tau_decay) - np.exp(-t_peak / tau_rise)
        return -abs(iext) * (np.exp(-dt / tau_decay) - np.exp(-dt / tau_rise)) / norm
    elif stype == 8:  # Kainate (intermediate excitation)
        if t < t0:
            return 0.0
        dt = t - t0
        tau_rise = 1.5
        tau_decay = 12.0
        t_peak = tau_rise * tau_decay / (tau_decay - tau_rise) * np.log(tau_decay / tau_rise)
        norm = np.exp(-t_peak / tau_decay) - np.exp(-t_peak / tau_rise)
        return abs(iext) * (np.exp(-dt / tau_decay) - np.exp(-dt / tau_rise)) / norm
    elif stype == 9:  # Nicotinic ACh (fast excitation, slightly slower than AMPA)
        if t < t0:
            return 0.0
        dt = t - t0
        tau_rise = 3.0
        tau_decay = 25.0
        t_peak = tau_rise * tau_decay / (tau_decay - tau_rise) * np.log(tau_decay / tau_rise)
        norm = np.exp(-t_peak / tau_decay) - np.exp(-t_peak / tau_rise)
        return abs(iext) * (np.exp(-dt / tau_decay) - np.exp(-dt / tau_rise)) / norm
    # По умолчанию const (0) | Default const (0)
    return iext

@njit(cache=True)
def rhs_multicompartment(
    t, y, n_comp,
    # Флаги включения каналов
    en_ih, en_ica, en_ia, en_sk, dyn_ca,
    # Векторы проводимостей (уже с учетом AIS-множителей)
    gna_v, gk_v, gl_v, gih_v, gca_v, ga_v, gsk_v,
    # Потенциалы реверсии
    ena, ek, el, eih, ea,
    # Морфология и среда
    cm_v, l_data, l_indices, l_indptr,
    phi_na, phi_k, phi_ih, phi_ca, phi_ia,
    t_kelvin, ca_ext, ca_rest, tau_ca, b_ca,
    # Стимуляция (primary)
    stype, iext, t0, td, atau, stim_comp, stim_mode,
    use_dfilter_primary, dfilter_attenuation, dfilter_tau_ms,
    # Dual stimulation (secondary) - optional
    dual_stim_enabled,
    stype_2, iext_2, t0_2, td_2, atau_2, stim_comp_2, stim_mode_2,
    use_dfilter_secondary, dfilter_attenuation_2, dfilter_tau_ms_2
):
    """
    Высокопроизводительное ядро ОДУ v10.1 (C-style scalar loop).
    Все токи рассчитываются как скаляры, без промежуточных аллокаций.
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
    # We use i_stim array because stimulus targets specific compartments
    i_stim = np.zeros(n_comp)
    base_current = get_stim_current(t, stype, iext, t0, td, atau)

    v_filtered_primary = 0.0
    if use_dfilter_primary == 1:
        v_filtered_primary = y[off_vfilt_primary]

    v_filtered_secondary = 0.0
    if use_dfilter_secondary == 1:
        v_filtered_secondary = y[off_vfilt_secondary]

    d_vfiltered_dt_primary = apply_primary_stimulus_current(
        i_stim, n_comp, base_current,
        stim_comp, stim_mode,
        use_dfilter_primary, dfilter_attenuation, dfilter_tau_ms,
        v_filtered_primary,
    )
    d_vfiltered_dt_secondary = 0.0
    if dual_stim_enabled == 1:
        base_current_2 = get_stim_current(t, stype_2, iext_2, t0_2, td_2, atau_2)
        d_vfiltered_dt_secondary = apply_secondary_stimulus_current(
            i_stim, n_comp, base_current_2,
            stim_comp_2, stim_mode_2,
            use_dfilter_secondary, dfilter_attenuation_2, dfilter_tau_ms_2,
            v_filtered_secondary,
        )

    # --- Output array ---
    dydt = np.zeros_like(y)

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

        if en_sk:
            if dyn_ca:
                ca_sk = y[off_ca + i]
            else:
                ca_sk = ca_rest
            z_act = z_inf_SK(ca_sk)
            i_ion += gsk_v[i] * z_act * (vi - ek)

        # Axial coupling (sparse Laplacian row i)
        i_ax = 0.0
        for j_idx in range(l_indptr[i], l_indptr[i + 1]):
            col = l_indices[j_idx]
            i_ax += l_data[j_idx] * y[off_v + col]

        # dV/dt
        dydt[off_v + i] = (i_stim[i] - i_ion + i_ax) / cm_v[i]

        # Gate derivatives (HH core) — channel-specific Q10
        dydt[off_m + i] = phi_na * (am(vi) * (1.0 - mi) - bm(vi) * mi)
        dydt[off_h + i] = phi_na * (ah(vi) * (1.0 - hi) - bh(vi) * hi)
        dydt[off_n + i] = phi_k * (an(vi) * (1.0 - ni) - bn(vi) * ni)

        # Optional gate derivatives
        if en_ih:
            ri = y[off_r + i]
            dydt[off_r + i] = phi_ih * (ar_Ih(vi) * (1.0 - ri) - br_Ih(vi) * ri)

        if en_ica:
            si = y[off_s + i]
            ui = y[off_u + i]
            dydt[off_s + i] = phi_ca * (as_Ca(vi) * (1.0 - si) - bs_Ca(vi) * si)
            dydt[off_u + i] = phi_ca * (au_Ca(vi) * (1.0 - ui) - bu_Ca(vi) * ui)

        if en_ia:
            ai = y[off_a + i]
            bi = y[off_b + i]
            dydt[off_a + i] = phi_ia * (aa_IA(vi) * (1.0 - ai) - ba_IA(vi) * ai)
            dydt[off_b + i] = phi_ia * (ab_IA(vi) * (1.0 - bi) - bb_IA(vi) * bi)

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
