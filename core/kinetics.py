import numpy as np
from numba import vectorize, float64, njit

# =====================================================================
# LUT (Look-Up Table) Infrastructure for Gate Kinetics
# =====================================================================

# Voltage range: -120 to 120 mV, 0.1 mV step
V_MIN = -200.0
V_MAX = 200.0
V_STEP = 0.1
N_V = int((V_MAX - V_MIN) / V_STEP) + 1  # 2401 points

# Pre-computed voltage array for LUT
_V_LUT = np.linspace(V_MIN, V_MAX, N_V, dtype=np.float64)

# LUT arrays for all gate functions (alpha and beta)
_am_LUT = np.zeros(N_V, dtype=np.float64)
_bm_LUT = np.zeros(N_V, dtype=np.float64)
_ah_LUT = np.zeros(N_V, dtype=np.float64)
_bh_LUT = np.zeros(N_V, dtype=np.float64)
_an_LUT = np.zeros(N_V, dtype=np.float64)
_bn_LUT = np.zeros(N_V, dtype=np.float64)

# Optional channel LUTs
_ar_Ih_LUT = np.zeros(N_V, dtype=np.float64)
_br_Ih_LUT = np.zeros(N_V, dtype=np.float64)
_as_Ca_LUT = np.zeros(N_V, dtype=np.float64)
_bs_Ca_LUT = np.zeros(N_V, dtype=np.float64)
_au_Ca_LUT = np.zeros(N_V, dtype=np.float64)
_bu_Ca_LUT = np.zeros(N_V, dtype=np.float64)
_aa_IA_LUT = np.zeros(N_V, dtype=np.float64)
_ba_IA_LUT = np.zeros(N_V, dtype=np.float64)
_ab_IA_LUT = np.zeros(N_V, dtype=np.float64)
_bb_IA_LUT = np.zeros(N_V, dtype=np.float64)
_am_TCa_LUT = np.zeros(N_V, dtype=np.float64)
_bm_TCa_LUT = np.zeros(N_V, dtype=np.float64)
_ah_TCa_LUT = np.zeros(N_V, dtype=np.float64)
_bh_TCa_LUT = np.zeros(N_V, dtype=np.float64)
_aw_IM_LUT = np.zeros(N_V, dtype=np.float64)
_bw_IM_LUT = np.zeros(N_V, dtype=np.float64)
_ax_NaP_LUT = np.zeros(N_V, dtype=np.float64)
_bx_NaP_LUT = np.zeros(N_V, dtype=np.float64)
_ay_NaR_LUT = np.zeros(N_V, dtype=np.float64)
_by_NaR_LUT = np.zeros(N_V, dtype=np.float64)
_aj_NaR_LUT = np.zeros(N_V, dtype=np.float64)
_bj_NaR_LUT = np.zeros(N_V, dtype=np.float64)

# Initialize all LUTs
def _init_luts():
    """Pre-compute all gate function LUTs at module load."""
    for i, v in enumerate(_V_LUT):
        _am_LUT[i] = am(v)
        _bm_LUT[i] = bm(v)
        _ah_LUT[i] = ah(v)
        _bh_LUT[i] = bh(v)
        _an_LUT[i] = an(v)
        _bn_LUT[i] = bn(v)
        _ar_Ih_LUT[i] = ar_Ih(v)
        _br_Ih_LUT[i] = br_Ih(v)
        _as_Ca_LUT[i] = as_Ca(v)
        _bs_Ca_LUT[i] = bs_Ca(v)
        _au_Ca_LUT[i] = au_Ca(v)
        _bu_Ca_LUT[i] = bu_Ca(v)
        _aa_IA_LUT[i] = aa_IA(v)
        _ba_IA_LUT[i] = ba_IA(v)
        _ab_IA_LUT[i] = ab_IA(v)
        _bb_IA_LUT[i] = bb_IA(v)
        _am_TCa_LUT[i] = am_TCa(v)
        _bm_TCa_LUT[i] = bm_TCa(v)
        _ah_TCa_LUT[i] = ah_TCa(v)
        _bh_TCa_LUT[i] = bh_TCa(v)
        _aw_IM_LUT[i] = aw_IM(v)
        _bw_IM_LUT[i] = bw_IM(v)
        _ax_NaP_LUT[i] = ax_NaP(v)
        _bx_NaP_LUT[i] = bx_NaP(v)
        _ay_NaR_LUT[i] = ay_NaR(v)
        _by_NaR_LUT[i] = by_NaR(v)
        _aj_NaR_LUT[i] = aj_NaR(v)
        _bj_NaR_LUT[i] = bj_NaR(v)

@njit(fastmath=True, cache=True)
def _lut_interp(V, lut):
    """Linear interpolation from pre-computed LUT.
    
    Parameters
    ----------
    V : float64
        Membrane potential (mV)
    lut : float64[:]
        Pre-computed LUT array
        
    Returns
    -------
    float64
        Interpolated value
    """
    # Guard against solver blow-ups before any float->int cast.
    if not np.isfinite(V):
        return lut[N_V // 2]

    # Clamp to LUT range
    if V <= V_MIN:
        return lut[0]
    if V >= V_MAX:
        return lut[-1]
    
    # Compute index and fractional position
    idx_float = (V - V_MIN) / V_STEP
    idx = int(idx_float)
    frac = idx_float - idx
    
    # Linear interpolation
    return lut[idx] + frac * (lut[idx + 1] - lut[idx])

# LUT-based gate functions (fast, no exponential calls)
@njit(fastmath=True, cache=True)
def am_lut(V):
    """Na activation alpha via LUT."""
    return _lut_interp(V, _am_LUT)

@njit(fastmath=True, cache=True)
def bm_lut(V):
    """Na deactivation beta via LUT."""
    return _lut_interp(V, _bm_LUT)

@njit(fastmath=True, cache=True)
def ah_lut(V):
    """Na inactivation alpha via LUT."""
    return _lut_interp(V, _ah_LUT)

@njit(fastmath=True, cache=True)
def bh_lut(V):
    """Na inactivation beta via LUT."""
    return _lut_interp(V, _bh_LUT)

@njit(fastmath=True, cache=True)
def an_lut(V):
    """K activation alpha via LUT."""
    return _lut_interp(V, _an_LUT)

@njit(fastmath=True, cache=True)
def bn_lut(V):
    """K deactivation beta via LUT."""
    return _lut_interp(V, _bn_LUT)

@njit(fastmath=True, cache=True)
def ar_Ih_lut(V):
    """Ih activation alpha via LUT."""
    return _lut_interp(V, _ar_Ih_LUT)

@njit(fastmath=True, cache=True)
def br_Ih_lut(V):
    """Ih activation beta via LUT."""
    return _lut_interp(V, _br_Ih_LUT)

@njit(fastmath=True, cache=True)
def as_Ca_lut(V):
    """ICa activation alpha via LUT."""
    return _lut_interp(V, _as_Ca_LUT)

@njit(fastmath=True, cache=True)
def bs_Ca_lut(V):
    """ICa activation beta via LUT."""
    return _lut_interp(V, _bs_Ca_LUT)

@njit(fastmath=True, cache=True)
def au_Ca_lut(V):
    """ICa inactivation alpha via LUT."""
    return _lut_interp(V, _au_Ca_LUT)

@njit(fastmath=True, cache=True)
def bu_Ca_lut(V):
    """ICa inactivation beta via LUT."""
    return _lut_interp(V, _bu_Ca_LUT)

@njit(fastmath=True, cache=True)
def aa_IA_lut(V):
    """IA activation alpha via LUT."""
    return _lut_interp(V, _aa_IA_LUT)

@njit(fastmath=True, cache=True)
def ba_IA_lut(V):
    """IA activation beta via LUT."""
    return _lut_interp(V, _ba_IA_LUT)

@njit(fastmath=True, cache=True)
def ab_IA_lut(V):
    """IA inactivation alpha via LUT."""
    return _lut_interp(V, _ab_IA_LUT)

@njit(fastmath=True, cache=True)
def bb_IA_lut(V):
    """IA inactivation beta via LUT."""
    return _lut_interp(V, _bb_IA_LUT)

@njit(fastmath=True, cache=True)
def am_TCa_lut(V):
    """I_T activation alpha via LUT."""
    return _lut_interp(V, _am_TCa_LUT)

@njit(fastmath=True, cache=True)
def bm_TCa_lut(V):
    """I_T activation beta via LUT."""
    return _lut_interp(V, _bm_TCa_LUT)

@njit(fastmath=True, cache=True)
def ah_TCa_lut(V):
    """I_T inactivation alpha via LUT."""
    return _lut_interp(V, _ah_TCa_LUT)

@njit(fastmath=True, cache=True)
def bh_TCa_lut(V):
    """I_T inactivation beta via LUT."""
    return _lut_interp(V, _bh_TCa_LUT)

@njit(fastmath=True, cache=True)
def aw_IM_lut(V):
    """I_M activation alpha via LUT."""
    return _lut_interp(V, _aw_IM_LUT)

@njit(fastmath=True, cache=True)
def bw_IM_lut(V):
    """I_M activation beta via LUT."""
    return _lut_interp(V, _bw_IM_LUT)

@njit(fastmath=True, cache=True)
def ax_NaP_lut(V):
    """I_NaP activation alpha via LUT."""
    return _lut_interp(V, _ax_NaP_LUT)

@njit(fastmath=True, cache=True)
def bx_NaP_lut(V):
    """I_NaP activation beta via LUT."""
    return _lut_interp(V, _bx_NaP_LUT)

@njit(fastmath=True, cache=True)
def ay_NaR_lut(V):
    """I_NaR activation alpha via LUT."""
    return _lut_interp(V, _ay_NaR_LUT)

@njit(fastmath=True, cache=True)
def by_NaR_lut(V):
    """I_NaR activation beta via LUT."""
    return _lut_interp(V, _by_NaR_LUT)

@njit(fastmath=True, cache=True)
def aj_NaR_lut(V):
    """I_NaR inactivation alpha via LUT."""
    return _lut_interp(V, _aj_NaR_LUT)

@njit(fastmath=True, cache=True)
def bj_NaR_lut(V):
    """I_NaR inactivation beta via LUT."""
    return _lut_interp(V, _bj_NaR_LUT)

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

# =====================================================================
# I_M — Muscarinic-sensitive K⁺ Current (KCNQ/Kv7, spike-frequency adaptation)
# Yamada, Koch & Adams 1989, Methods in Neuronal Modeling (MIT Press).
# Gating: w (single activation, no inactivation).
# I_M = g_M * w * (V - E_K)
# w_inf(V) = 1/(1+exp(-(V+35)/10)),  V½ = -35 mV, k = 10 mV
# tau_w(V) = taumax / (3.3*exp((V+35)/20) + exp(-(V+35)/20))
# Original kinetics defined at 36°C with taumax=1000 ms, Q10=2.3.
# Pre-scaled to T_ref=6.3°C: taumax_raw = 1000 * 2.3^((36-6.3)/10)
# so that our standard phi_channel(Q10_M) handles temperature correctly.
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
    w = 1.0 / (1.0 + np.exp((Vs + 80.0) / 2.0))
    tau_h_hyper = np.exp((Vs + 467.0) / 66.6)
    tau_h_depol = 28.0 + np.exp(-(Vs + 22.0) / 10.5)
    tau_h = w * tau_h_hyper + (1.0 - w) * tau_h_depol
    return h_inf / tau_h

@vectorize([float64(float64)], nopython=True, cache=True)
def bh_TCa(V):
    """I_T inactivation beta (Destexhe 1998)."""
    Vs = V + _IT_SHIFT
    h_inf = 1.0 / (1.0 + np.exp((Vs + 81.0) / 4.0))
    w = 1.0 / (1.0 + np.exp((Vs + 80.0) / 2.0))
    tau_h_hyper = np.exp((Vs + 467.0) / 66.6)
    tau_h_depol = 28.0 + np.exp(-(Vs + 22.0) / 10.5)
    tau_h = w * tau_h_hyper + (1.0 - w) * tau_h_depol
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


# =====================================================================
# I_NaP — Persistent Sodium Current (non-inactivating)
# Magistretti & Alonso 1999 (J Gen Physiol 114:491-509); Pospischil
# et al. 2008 (Biol Cybern 99:427-441).
# Single activation gate 'x', power 1.  No inactivation.
# V½ = −52 mV, k = 5 mV.  Very fast kinetics (τ ≈ 0.3 ms at V½).
# Reference temperature: ~23 °C.  Q10 ≈ 2.2 (same family as Na).
# Reversal: E_Na.
# Default g_NaP ≈ 0.1 mS/cm² (cortical pyramidal, Pospischil 2008).
# =====================================================================

@vectorize([float64(float64)], nopython=True, cache=True)
def ax_NaP(V):
    """I_NaP activation alpha (Magistretti & Alonso 1999). V½ ≈ -52 mV."""
    x_inf = 1.0 / (1.0 + np.exp(-(V + 52.0) / 5.0))
    c = (V + 52.0) / 15.0
    tau_x = 0.15 + 1.0 / (3.3 * (np.exp(c) + np.exp(-c)))
    return x_inf / tau_x

@vectorize([float64(float64)], nopython=True, cache=True)
def bx_NaP(V):
    """I_NaP activation beta (Magistretti & Alonso 1999)."""
    x_inf = 1.0 / (1.0 + np.exp(-(V + 52.0) / 5.0))
    c = (V + 52.0) / 15.0
    tau_x = 0.15 + 1.0 / (3.3 * (np.exp(c) + np.exp(-c)))
    return (1.0 - x_inf) / tau_x


# =====================================================================
# I_NaR — Resurgent Sodium Current (phenomenological HH adaptation)
# Based on Raman & Bean 2001 (Biophys J 80:729-737).
# Full resurgent Na requires a Markov blocking-particle model; this is
# a simplified two-gate HH approximation capturing the essential
# behavior: window current that peaks ~-40 mV during spike repolarization.
#
# Gate 'y' (activation):  V½ = −48 mV, k = 6 mV, τ ~ 3 ms at V½
# Gate 'j' (inactivation/block): V½ = −33 mV, k = 4 mV, τ ~ 18 ms
# Resurgent dynamics: during the spike falling phase, y remains high
# (fast activation) while j slowly recovers from block, producing a
# transient inward current burst.
#
# Reference temperature: ~23 °C.  Q10 ≈ 2.2.
# Reversal: E_Na.
# Default g_NaR ≈ 0.2 mS/cm² (Purkinje/FS interneurons).
# =====================================================================

@vectorize([float64(float64)], nopython=True, cache=True)
def ay_NaR(V):
    """I_NaR activation alpha (Raman & Bean 2001 adaptation). V½ ≈ -48 mV."""
    y_inf = 1.0 / (1.0 + np.exp(-(V + 48.0) / 6.0))
    c = (V + 48.0) / 15.0
    tau_y = 0.5 + 2.5 / (np.exp(c) + np.exp(-c))
    return y_inf / tau_y

@vectorize([float64(float64)], nopython=True, cache=True)
def by_NaR(V):
    """I_NaR activation beta."""
    y_inf = 1.0 / (1.0 + np.exp(-(V + 48.0) / 6.0))
    c = (V + 48.0) / 15.0
    tau_y = 0.5 + 2.5 / (np.exp(c) + np.exp(-c))
    return (1.0 - y_inf) / tau_y

@vectorize([float64(float64)], nopython=True, cache=True)
def aj_NaR(V):
    """I_NaR inactivation/block alpha (steep). V½ ≈ -33 mV."""
    j_inf = 1.0 / (1.0 + np.exp((V + 33.0) / 4.0))
    c = (V + 33.0) / 10.0
    tau_j = 3.0 + 15.0 / (np.exp(c) + np.exp(-c))
    return j_inf / tau_j

@vectorize([float64(float64)], nopython=True, cache=True)
def bj_NaR(V):
    """I_NaR inactivation/block beta."""
    j_inf = 1.0 / (1.0 + np.exp((V + 33.0) / 4.0))
    c = (V + 33.0) / 10.0
    tau_j = 3.0 + 15.0 / (np.exp(c) + np.exp(-c))
    return (1.0 - j_inf) / tau_j


# Initialize LUTs at module load (after all gate functions are defined)
_init_luts()


# =====================================================================
# Na+/K+ ATPase Pump Electrogenic Current
# Michaelis-Menten kinetics with Hill coefficients (Na:3, K:2)
# =====================================================================

@njit(float64(float64, float64, float64), fastmath=True, cache=True)
def pump_current(na_i_mM, k_o_mM, atp_mM):
    """
    Electrogenic Na+/K+ pump current density (µA/cm²).
    
    Michaelis-Menten kinetics: I_pump = I_max * [Na]_i³/(K_m,Na³ + [Na]_i³) 
                                        * [K]_o²/(K_m,K² + [K]_o²) * f(ATP)
    Simplified Hill form: (1 + K_m/[S])^(-n)
    
    Parameters
    ----------
    na_i_mM : float
        Intracellular sodium concentration [mM]
    k_o_mM : float
        Extracellular potassium concentration [mM]
    atp_mM : float
        ATP concentration [mM] (2.0 mM = healthy/fully active)
    
    Returns
    -------
    float
        Pump current density [µA/cm²], positive = outward (hyperpolarizing)
    """
    # Physiological parameters (standard literature values)
    I_MAX = 0.25               # Max pump current density [µA/cm²]
    KM_NA = 15.0             # Intracellular Na+ half-saturation [mM]
    KM_K = 2.0               # Extracellular K+ half-saturation [mM]
    ATP_HALF = 2.0           # ATP concentration for half-max activity [mM]
    
    # Guard against non-positive concentrations
    na_safe = na_i_mM if na_i_mM > 1e-9 else 1e-9
    k_safe = k_o_mM if k_o_mM > 1e-9 else 1e-9
    
    # ATP availability factor (linear scaling, saturates at 1.0)
    atp_factor = atp_mM / ATP_HALF if atp_mM < ATP_HALF else 1.0
    if atp_factor < 0.0:
        atp_factor = 0.0
    
    # Hill kinetics: (1 + Km/[S])^(-n)
    na_factor = 1.0 / (1.0 + KM_NA / na_safe)
    na_factor = na_factor * na_factor * na_factor  # Hill coeff = 3
    
    k_factor = 1.0 / (1.0 + KM_K / k_safe)
    k_factor = k_factor * k_factor                   # Hill coeff = 2
    
    return I_MAX * na_factor * k_factor * atp_factor
