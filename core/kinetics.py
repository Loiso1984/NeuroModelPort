import numpy as np
from numba import vectorize, float64

# =====================================================================
# Ядро Ходжкина-Хаксли (Na, K) - HH 1952 | Hodgkin-Huxley Core (Na, K) - HH 1952
# =====================================================================

@vectorize([float64(float64)], nopython=True, cache=True)
def am(V):
    """Активация Na (m). Определяет резкий фронт потенциала действия. | Na activation (m). Determines sharp upstroke of action potential."""
    dV = V + 40.0
    if abs(dV) < 1e-7:
        return 1.0  # Правило Лопиталя при V = -40 мВ | L'Hôpital's rule at V = -40 mV
    return 0.1 * dV / (1.0 - np.exp(-dV / 10.0))

@vectorize([float64(float64)], nopython=True, cache=True)
def bm(V):
    """Деактивация Na (m). | Na deactivation (m)."""
    return 4.0 * np.exp(-(V + 65.0) / 18.0)

@vectorize([float64(float64)], nopython=True, cache=True)
def ah(V):
    """Снятие инактивации Na (h). | Na inactivation removal (h)."""
    return 0.07 * np.exp(-(V + 65.0) / 20.0)

@vectorize([float64(float64)], nopython=True, cache=True)
def bh(V):
    """Инактивация Na (h). Медленнее 'm', завершает спайк. | Na inactivation (h). Slower than m, completes spike."""
    return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

@vectorize([float64(float64)], nopython=True, cache=True)
def an(V):
    """Активация K (n). Медленный ток реполяризации. | K activation (n). Slow repolarizing current."""
    dV = V + 55.0
    if abs(dV) < 1e-7:
        return 0.1  # Правило Лопиталя при V = -55 мВ | L'Hôpital's rule at V = -55 mV
    return 0.01 * dV / (1.0 - np.exp(-dV / 10.0))

@vectorize([float64(float64)], nopython=True, cache=True)
def bn(V):
    """Деактивация K (n). | K deactivation (n)."""
    return 0.125 * np.exp(-(V + 65.0) / 80.0)

# =====================================================================
# Опциональный канал Ih (HCN) - Ток гиперполяризации (Destexhe 1993) | Optional Ih channel (HCN) - Hyperpolarization current (Destexhe 1993)
# =====================================================================

@vectorize([float64(float64)], nopython=True, cache=True)
def ar_Ih(V):
    """Ih activation alpha (Destexhe 1993) - V1/2 ~ -78mV"""
    return 0.001 * np.exp(-(V + 78.0) / 18.0)

@vectorize([float64(float64)], nopython=True, cache=True)
def br_Ih(V):
    """Ih activation beta (Destexhe 1993)"""
    return 0.001 / (1.0 + np.exp(-(V + 78.0) / 18.0))

# =====================================================================
# Опциональный канал ICa (L-type Ca2+) - Высокопороговый (Huguenard 1992) | Optional ICa channel (L-type Ca2+) - High threshold (Huguenard 1992)
# =====================================================================

@vectorize([float64(float64)], nopython=True, cache=True)
def as_Ca(V):
    """ICa activation alpha (Huguenard 1992) - High Threshold L-type"""
    return 1.6 / (1.0 + np.exp(-0.072 * (V - 5.0)))

@vectorize([float64(float64)], nopython=True, cache=True)
def bs_Ca(V):
    """ICa activation beta (Huguenard 1992)"""
    # dV = V + 8.9 ensures activation starts around -20 to -10 mV
    dV = V + 8.9
    if abs(dV) < 1e-7:
        return 0.1
    return 0.02 * dV / (np.exp(dV / 5.0) - 1.0)

@vectorize([float64(float64)], nopython=True, cache=True)
def au_Ca(V):
    """ICa inactivation alpha (Huguenard 1992) - Very slow inactivation"""
    return 0.000457 * np.exp(-(V + 13.0) / 50.0)

@vectorize([float64(float64)], nopython=True, cache=True)
def bu_Ca(V):
    """ICa inactivation beta (Huguenard 1992)"""
    return 0.0065 / (np.exp(-(V + 15.0) / 28.0) + 1.0)

# =====================================================================
# Опциональный канал IA - Быстрый транзиентный K+ ток (Connor-Stevens 1971) | Optional IA channel - Fast transient K+ current (Connor-Stevens 1971)
# =====================================================================

@vectorize([float64(float64)], nopython=True, cache=True)
def aa_IA(V):
    """IA activation alpha - Connor-Stevens 1971"""
    dV = V - 20.0  # Shift for V_½ ~ -40 mV
    if abs(dV) < 1e-7:
        return 0.2
    return 0.02 * dV / (1.0 - np.exp(-dV / 10.0))

@vectorize([float64(float64)], nopython=True, cache=True)
def ba_IA(V):
    """IA activation beta - Connor-Stevens 1971 kinetics"""
    return 0.0175 * np.exp(-(V + 65.0) / 20.0)

@vectorize([float64(float64)], nopython=True, cache=True)
def ab_IA(V):
    """IA inactivation alpha - Connor-Stevens 1971"""
    return 0.0016 * np.exp(-(V + 50.0) / 18.0)  # V_½ ~ -60 mV

@vectorize([float64(float64)], nopython=True, cache=True)
def bb_IA(V):
    """IA inactivation beta - Connor-Stevens 1971"""
    return 0.05 / (1.0 + np.exp(-(V + 50.0) / 5.0))  # V_½ ~ -60 mV

# =====================================================================
# НОВЫЙ КАНАЛ: ISK - Ca2+-зависимый K+ ток (Спайковая адаптация) | NEW CHANNEL: ISK - Ca2+-dependent K+ current (Spike adaptation)
# =====================================================================

@vectorize([float64(float64)], nopython=True, cache=True)
def z_inf_SK(Ca_i):
    """
    Равновероятность открытия SK канала. Зависит от концентрации Кальция! | SK channel opening probability. Depends on Calcium concentration!
    Ca_i: внутриклеточный кальций в мМ (миллимоль) | intracellular calcium in mM (millimolar).
    Kd = 0.0004 мМ (400 нМ) - полуактивация. Степень Хилла n=4 | Kd = 0.0004 mM (400 nM) - half-activation. Hill coefficient n=4.
    """
    Kd = 0.0004
    # Защита от отрицательного кальция при жестких шагах интегратора | Protection against negative calcium during stiff integrator steps
    ca_safe = max(Ca_i, 1e-9)
    return 1.0 / (1.0 + (Kd / ca_safe)**4)


# =====================================================================
# I_T — Low-Threshold Transient Ca²⁺ Current (T-type, CaV3.x)
# Destexhe et al. 1998 (J Neurosci 18:3574); Huguenard & McCormick 1992.
# Gating: m²h (m = activation, h = inactivation).
# Kinetics use Boltzmann inf/tau converted to alpha/beta for HH formalism:
#   alpha = inf / tau,  beta = (1 - inf) / tau
# Reference temperature: 24°C. Q10_m = 5.0, Q10_h = 3.0 (Destexhe 1998).
# Screening charge shift = +2 mV for 2 mM external Ca²⁺.
# =====================================================================

_IT_SHIFT = 2.0  # Screening charge shift for 2 mM Ca_ext (Destexhe 1998)

@vectorize([float64(float64)], nopython=True, cache=True)
def am_TCa(V):
    """I_T activation alpha (Destexhe 1998). V½ ≈ -57 mV."""
    Vs = V + _IT_SHIFT
    m_inf = 1.0 / (1.0 + np.exp(-(Vs + 57.0) / 6.2))
    tau_m = 0.612 + 1.0 / (np.exp(-(Vs + 132.0) / 16.7) + np.exp((Vs + 16.8) / 18.2))
    return m_inf / tau_m

@vectorize([float64(float64)], nopython=True, cache=True)
def bm_TCa(V):
    """I_T activation beta (Destexhe 1998)."""
    Vs = V + _IT_SHIFT
    m_inf = 1.0 / (1.0 + np.exp(-(Vs + 57.0) / 6.2))
    tau_m = 0.612 + 1.0 / (np.exp(-(Vs + 132.0) / 16.7) + np.exp((Vs + 16.8) / 18.2))
    return (1.0 - m_inf) / tau_m

@vectorize([float64(float64)], nopython=True, cache=True)
def ah_TCa(V):
    """I_T inactivation alpha (Destexhe 1998). V½ ≈ -81 mV. Slow recovery."""
    Vs = V + _IT_SHIFT
    h_inf = 1.0 / (1.0 + np.exp((Vs + 81.0) / 4.0))
    if Vs < -80.0:
        tau_h = np.exp((Vs + 467.0) / 66.6)
    else:
        tau_h = 28.0 + np.exp(-(Vs + 22.0) / 10.5)
    return h_inf / tau_h

@vectorize([float64(float64)], nopython=True, cache=True)
def bh_TCa(V):
    """I_T inactivation beta (Destexhe 1998)."""
    Vs = V + _IT_SHIFT
    h_inf = 1.0 / (1.0 + np.exp((Vs + 81.0) / 4.0))
    if Vs < -80.0:
        tau_h = np.exp((Vs + 467.0) / 66.6)
    else:
        tau_h = 28.0 + np.exp(-(Vs + 22.0) / 10.5)
    return (1.0 - h_inf) / tau_h


# =====================================================================
# I_M — M-type Potassium Current (KCNQ2/3, Kv7)
# Yamada, Koch & Adams 1989 (Methods in Neuronal Modeling, Koch & Segev,
# MIT Press, pp 97-133).  Single activation gate 'w', non-inactivating.
# V½ = −35 mV, slope k = 10 mV.
# τ_w given by 1/(α+β) with cosh-based denominator; τ_w(V½) ≈ 151 ms.
# Reference temperature: ~23 °C.  Q10 ≈ 2.5 (Pan et al. 2006, J Physiol
# 576:215-228 for KCNQ2/3 channels).
# Reversal: E_K (same as delayed-rectifier K⁺).
# =====================================================================

@vectorize([float64(float64)], nopython=True, cache=True)
def aw_IM(V):
    """I_M activation alpha (Yamada, Koch & Adams 1989). V½ ≈ -35 mV."""
    x = (V + 35.0) / 20.0
    # cosh-based tau: tau_w = 1000 / (3.3 * 2 * cosh(x))
    # alpha = w_inf / tau_w
    w_inf = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    tau_w = 1000.0 / (3.3 * (np.exp(x) + np.exp(-x)))
    return w_inf / tau_w

@vectorize([float64(float64)], nopython=True, cache=True)
def bw_IM(V):
    """I_M activation beta (Yamada, Koch & Adams 1989)."""
    x = (V + 35.0) / 20.0
    w_inf = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    tau_w = 1000.0 / (3.3 * (np.exp(x) + np.exp(-x)))
    return (1.0 - w_inf) / tau_w
