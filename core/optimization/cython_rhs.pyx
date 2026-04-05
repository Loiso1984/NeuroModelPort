"""
Optimization module: Fast RHS computations using Cython
Compile with: python setup.py build_ext --inplace
"""

import numpy as np
cimport numpy as np
from libc.math cimport exp, log, pow, fabs

# Constants
cdef double F_CONST = 96485.33
cdef double R_GAS = 8.314
cdef double TEMP_ZERO = 273.15

# Type definitions
dtype = np.float64
ctypedef np.float64_t DTYPE_t

# Fast stimulus current calculation
cdef inline double get_stim_current_fast(
    double t, int stype, double iext, double t0, double td, double atau
) nogil:
    """Optimized stimulus current calculation - no Python overhead"""
    cdef double dt, tau_rise, tau_decay, t_peak, norm
    
    if stype == 1:  # pulse
        return iext if t0 <= t <= t0 + td else 0.0
    elif stype == 2:  # alpha (EPSC)
        if t < t0:
            return 0.0
        dt = (t - t0) / atau
        return iext * dt * exp(1.0 - dt)
    elif stype == 4:  # AMPA
        if t < t0:
            return 0.0
        dt = t - t0
        tau_rise = 0.5
        tau_decay = 3.0
        t_peak = tau_rise * tau_decay / (tau_decay - tau_rise) * log(tau_decay / tau_rise)
        norm = exp(-t_peak / tau_decay) - exp(-t_peak / tau_rise)
        return fabs(iext) * (exp(-dt / tau_decay) - exp(-dt / tau_rise)) / norm
    elif stype == 6:  # GABA-A
        if t < t0:
            return 0.0
        dt = t - t0
        tau_rise = 1.0
        tau_decay = 7.0
        t_peak = tau_rise * tau_decay / (tau_decay - tau_rise) * log(tau_decay / tau_rise)
        norm = exp(-t_peak / tau_decay) - exp(-t_peak / tau_rise)
        return -fabs(iext) * (exp(-dt / tau_decay) - exp(-dt / tau_rise)) / norm
    
    return 0.0


# Fast kinetic functions
cdef inline double alpha_m_Na(double V) nogil:
    """Na activation alpha - faster than Python version"""
    cdef double dV = V + 40.0
    if fabs(dV) < 1e-6:
        return 1.0
    return 0.1 * dV / (1.0 - exp(-dV / 10.0))


cdef inline double beta_m_Na(double V) nogil:
    """Na activation beta"""
    return 4.0 * exp(-(V + 65.0) / 18.0)


cdef inline double alpha_h_Na(double V) nogil:
    """Na inactivation alpha"""
    return 0.07 * exp(-(V + 65.0) / 20.0)


cdef inline double beta_h_Na(double V) nogil:
    """Na inactivation beta"""
    cdef double dV = V + 35.0
    if fabs(dV) < 1e-6:
        return 1.0
    return 1.0 / (1.0 + exp(-dV / 10.0))


cdef inline double alpha_n_K(double V) nogil:
    """K activation alpha"""
    cdef double dV = V + 55.0
    if fabs(dV) < 1e-6:
        return 0.1
    return 0.01 * dV / (1.0 - exp(-dV / 10.0))


cdef inline double beta_n_K(double V) nogil:
    """K activation beta"""
    return 0.125 * exp(-(V + 65.0) / 80.0)


# Main fast RHS function
def rhs_multicompartment_fast(
    double t,
    np.ndarray[DTYPE_t, ndim=1] y,
    int n_comp,
    int enable_Ih, int enable_ICa, int enable_IA, int enable_SK,
    int dynamic_Ca,
    np.ndarray[DTYPE_t, ndim=1] gNa_v,
    np.ndarray[DTYPE_t, ndim=1] gK_v,
    np.ndarray[DTYPE_t, ndim=1] gL_v,
    np.ndarray[DTYPE_t, ndim=1] gIh_v,
    np.ndarray[DTYPE_t, ndim=1] gCa_v,
    np.ndarray[DTYPE_t, ndim=1] gA_v,
    np.ndarray[DTYPE_t, ndim=1] gSK_v,
    double ENa, double EK, double EL, double E_Ih, double E_A,
    np.ndarray[DTYPE_t, ndim=1] Cm_v,
    # ... additional parameters would continue
):
    """
    Fast Cython version of rhs_multicompartment
    
    IMPORTANT:
    This backend is intentionally disabled because the implementation is
    incomplete (no sparse coupling, partial channel set, incomplete Ca dynamics).
    It must not be used for scientific simulations until full feature parity with
    core.rhs.rhs_multicompartment is implemented and validated.
    """
    raise NotImplementedError(
        "core/optimization/cython_rhs.pyx is a non-production skeleton. "
        "Use NeuronSolver (Numba+SciPy) backend."
    )

    cdef np.ndarray[DTYPE_t, ndim=1] dydt = np.zeros_like(y)
    cdef int i
    cdef double V, m, h, n_k
    cdef double am, bm, ah, bh, an, bn
    cdef double I_Na, I_K, I_Leak
    
    # Extract state variables
    cdef np.ndarray[DTYPE_t, ndim=1] v = y[0:n_comp]
    cdef np.ndarray[DTYPE_t, ndim=1] m_gate = y[n_comp:2*n_comp]
    cdef np.ndarray[DTYPE_t, ndim=1] h_gate = y[2*n_comp:3*n_comp]
    cdef np.ndarray[DTYPE_t, ndim=1] n_gate = y[3*n_comp:4*n_comp]
    
    # TODO: Full implementation of channel currents
    # TODO: Compartment coupling via sparse matrix
    # TODO: Calcium dynamics if enabled
    # TODO: All optional channels (Ih, ICa, IA, SK)
    
    with nogil:
        for i in range(n_comp):
            V = v[i]
            m = m_gate[i]
            h = h_gate[i]
            n_k = n_gate[i]
            
            # Na channel kinetics
            am = alpha_m_Na(V)
            bm = beta_m_Na(V)
            ah = alpha_h_Na(V)
            bh = beta_h_Na(V)
            
            # K channel kinetics  
            an = alpha_n_K(V)
            bn = beta_n_K(V)
            
            # Channel currents
            I_Na = gNa_v[i] * pow(m, 3) * h * (V - ENa)
            I_K = gK_v[i] * pow(n_k, 4) * (V - EK)
            I_Leak = gL_v[i] * (V - EL)
            
            # dV/dt = -(I_Na + I_K + I_Leak + I_stim) / Cm
            # TODO: Add stimulation and other channels
            # TODO: Add compartment coupling
            
            # Gate derivatives
            dydt[n_comp + i] = am * (1.0 - m) - bm * m
            dydt[2*n_comp + i] = ah * (1.0 - h) - bh * h
            dydt[3*n_comp + i] = an * (1.0 - n_k) - bn * n_k
    
    return dydt


# Benchmark function
def benchmark_rhs_call(n_comp=10, n_calls=1000):
    """Benchmark RHS evaluation speed"""
    raise NotImplementedError(
        "Benchmark disabled: cython_rhs backend is non-production skeleton."
    )

    import time
    
    # Setup dummy data
    y = np.zeros(4 * n_comp, dtype=np.float64)
    for i in range(n_comp):
        y[i] = -65.0  # V
        y[n_comp + i] = 0.05  # m
        y[2*n_comp + i] = 0.6  # h
        y[3*n_comp + i] = 0.3  # n
    
    gNa = np.ones(n_comp) * 100.0
    gK = np.ones(n_comp) * 10.0
    gL = np.ones(n_comp) * 0.3
    Cm = np.ones(n_comp) * 1.0
    
    # Warmup
    for _ in range(10):
        rhs_multicompartment_fast(
            0.0, y, n_comp, 0, 0, 0, 0, 0,
            gNa, gK, gL, gNa, gNa, gNa, gNa,
            50.0, -77.0, -65.0, -43.0, -80.0,
            Cm
        )
    
    # Benchmark
    t_start = time.time()
    for _ in range(n_calls):
        rhs_multicompartment_fast(
            0.0, y, n_comp, 0, 0, 0, 0, 0,
            gNa, gK, gL, gNa, gNa, gNa, gNa,
            50.0, -77.0, -65.0, -43.0, -80.0,
            Cm
        )
    t_elapsed = time.time() - t_start
    
    print(f"Cython RHS: {n_calls} calls in {t_elapsed:.3f}s")
    print(f"Speed: {n_calls / t_elapsed:.0f} calls/sec")
    
    return t_elapsed
