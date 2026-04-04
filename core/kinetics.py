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
