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
        # Typical: rise 1–2 ms, decay 10–15 ms
        tau_rise = 1.5
        tau_decay = 12.0
        t_peak = tau_rise * tau_decay / (tau_decay - tau_rise) * np.log(tau_decay / tau_rise)
        norm = np.exp(-t_peak / tau_decay) - np.exp(-t_peak / tau_rise)
        return abs(iext) * (np.exp(-dt / tau_decay) - np.exp(-dt / tau_rise)) / norm
    elif stype == 9:  # Nicotinic ACh (fast excitation, slightly slower than AMPA)
        if t < t0:
            return 0.0
        dt = t - t0
        # Representative central nAChR: rise ~3–5 ms, decay 20–50 ms
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
    phi, t_kelvin, ca_ext, ca_rest, tau_ca, b_ca,
    # Стимуляция (primary)
    stype, iext, t0, td, atau, stim_comp, stim_mode,
    use_dfilter_primary, dfilter_attenuation, dfilter_tau_ms,
    # Dual stimulation (secondary) - optional
    dual_stim_enabled,
    stype_2, iext_2, t0_2, td_2, atau_2, stim_comp_2, stim_mode_2,
    use_dfilter_secondary, dfilter_attenuation_2, dfilter_tau_ms_2
):
    """
    Высокопроизводительное ядро ОДУ v10.0.
    Рассчитывает производные для всех гейтов и мембранного потенциала.
    """
    
    # --- 1. Распаковка вектора состояния y ---
    # Индексация жестко фиксирована: [V, m, h, n, [r], [s, u], [a, b], [Ca]]
    cursor = 0
    v = y[cursor : cursor + n_comp]; cursor += n_comp
    m = y[cursor : cursor + n_comp]; cursor += n_comp
    h = y[cursor : cursor + n_comp]; cursor += n_comp
    n = y[cursor : cursor + n_comp]; cursor += n_comp
    
    # Опциональные гейты (распаковываем только если включены)
    r = np.zeros(n_comp)
    s = np.zeros(n_comp)
    u = np.zeros(n_comp)
    a = np.zeros(n_comp)
    b = np.zeros(n_comp)
    if en_ih:
        r = y[cursor : cursor + n_comp]; cursor += n_comp
    if en_ica:
        s = y[cursor : cursor + n_comp]; cursor += n_comp
        u = y[cursor : cursor + n_comp]; cursor += n_comp
    if en_ia:
        a = y[cursor : cursor + n_comp]; cursor += n_comp
        b = y[cursor : cursor + n_comp]; cursor += n_comp
    
    # Кальций
    ca_i = np.full(n_comp, ca_rest)
    if dyn_ca:
        ca_i = y[cursor : cursor + n_comp]; cursor += n_comp

    # --- 2. Расчет ионных токов ---
    i_ion = gl_v * (v - el) # Ток утечки
    i_ion += gna_v * (m**3) * h * (v - ena) # Натрий
    i_ion += gk_v * (n**4) * (v - ek)  # Калий
    
    i_ca_total = np.zeros(n_comp)
    
    if en_ih:
        i_ion += gih_v * r * (v - eih)
        
    if en_ica:
        # Реверсия Ca: по-компартментный Нернст при динамическом Ca,
        # иначе фиксированный E_Ca.
        if dyn_ca:
            eca_eff = np.empty(n_comp)
            for i in range(n_comp):
                eca_eff[i] = nernst_ca_ion(ca_i[i], ca_ext, t_kelvin)
        else:
            eca_eff = np.full(n_comp, 120.0)
        i_ca_current = gca_v * (s**2) * u * (v - eca_eff)
        i_ion += i_ca_current
        # Для кальциевой динамики: кальций входит когда I_Ca < 0 (входящий ток)
        # Отрицательный ток = входящий кальций = положительный influx
        i_ca_total = -np.minimum(i_ca_current, 0)  # Только входящий ток
        
    if en_ia:
        i_ion += ga_v * a * b * (v - ea)
        
    if en_sk:
        # Адаптация: SK-канал активируется кальцием (z_inf_SK из kinetics)
        # Мы предполагаем мгновенную активацию относительно динамики V
        for i in range(n_comp):
            z_act = z_inf_SK(ca_i[i])
            i_ion[i] += gsk_v[i] * z_act * (v[i] - ek)

    # --- 3. Аксиальные связи (Лапласиан) ---
    i_ax = np.zeros(n_comp)
    for i in range(n_comp):
        for j_idx in range(l_indptr[i], l_indptr[i+1]):
            col = l_indices[j_idx]
            i_ax[i] += l_data[j_idx] * v[col]

    # Optional dendritic-filter dynamic states as extra ODE variables
    v_filtered_primary = 0.0
    v_filtered_secondary = 0.0
    if use_dfilter_primary == 1:
        v_filtered_primary = y[cursor]
        cursor += 1
    if use_dfilter_secondary == 1:
        v_filtered_secondary = y[cursor]
        cursor += 1

    # --- 4. External stimulation (primary + optional secondary) ---
    i_stim = np.zeros(n_comp)
    base_current = get_stim_current(t, stype, iext, t0, td, atau)
    d_vfiltered_dt_primary = apply_primary_stimulus_current(
        i_stim,
        n_comp,
        base_current,
        stim_comp,
        stim_mode,
        use_dfilter_primary,
        dfilter_attenuation,
        dfilter_tau_ms,
        v_filtered_primary,
    )
    d_vfiltered_dt_secondary = 0.0

    # --- 4b. Dual stimulation (secondary stimulus) ---
    # Apply secondary stimulus ONLY if dual_stim_enabled == 1
    if dual_stim_enabled == 1:
        base_current_2 = get_stim_current(t, stype_2, iext_2, t0_2, td_2, atau_2)
        d_vfiltered_dt_secondary = apply_secondary_stimulus_current(
            i_stim,
            n_comp,
            base_current_2,
            stim_comp_2,
            stim_mode_2,
            use_dfilter_secondary,
            dfilter_attenuation_2,
            dfilter_tau_ms_2,
            v_filtered_secondary,
        )

    # --- 5. Сборка производных dy/dt ---
    dydt = np.zeros_like(y)
    cursor = 0
    
    # dV/dt = (I_ext - I_ion + I_axial) / Cm
    dydt[cursor : cursor + n_comp] = (i_stim - i_ion + i_ax) / cm_v
    cursor += n_comp
    
    # Производные гейтов (HH)
    dydt[cursor : cursor + n_comp] = phi * (am(v) * (1.0 - m) - bm(v) * m)
    cursor += n_comp
    dydt[cursor : cursor + n_comp] = phi * (ah(v) * (1.0 - h) - bh(v) * h)
    cursor += n_comp
    dydt[cursor : cursor + n_comp] = phi * (an(v) * (1.0 - n) - bn(v) * n)
    cursor += n_comp
    
    # Производные опциональных гейтов
    if en_ih:
        dydt[cursor : cursor + n_comp] = phi * (ar_Ih(v) * (1.0 - r) - br_Ih(v) * r)
        cursor += n_comp
    if en_ica:
        dydt[cursor : cursor + n_comp] = phi * (as_Ca(v) * (1.0 - s) - bs_Ca(v) * s)
        cursor += n_comp
        dydt[cursor : cursor + n_comp] = phi * (au_Ca(v) * (1.0 - u) - bu_Ca(v) * u)
        cursor += n_comp
    if en_ia:
        dydt[cursor : cursor + n_comp] = phi * (aa_IA(v) * (1.0 - a) - ba_IA(v) * a)
        cursor += n_comp
        dydt[cursor : cursor + n_comp] = phi * (ab_IA(v) * (1.0 - b) - bb_IA(v) * b)
        cursor += n_comp
        
    # Динамика концентрации кальция
    if dyn_ca:
        # d[Ca]/dt = +B*I_Ca,influx - ([Ca]-Ca_rest)/tau_Ca
        # Упрощенная Numba-совместимая версия
        if i_ca_total.size == n_comp:
            dca = b_ca * i_ca_total - (ca_i - ca_rest) / tau_ca
        else:
            # Если i_ca_total скаляр, используем broadcasting
            dca = b_ca * i_ca_total - (ca_i - ca_rest) / tau_ca
        
        # Защита от отрицательной концентрации (Hard clamp)
        for i in range(n_comp):
            if ca_i[i] < 1e-9 and dca[i] < 0:
                dca[i] = 0.0
        dydt[cursor : cursor + n_comp] = dca
        cursor += n_comp

    if use_dfilter_primary == 1:
        dydt[cursor] = d_vfiltered_dt_primary
        cursor += 1
    if use_dfilter_secondary == 1:
        dydt[cursor] = d_vfiltered_dt_secondary

    return dydt
