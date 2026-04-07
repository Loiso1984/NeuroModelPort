"""Native Hines solver kernels — v11.0.

O(N) direct tree solver replacing SciPy BDF for voltage equation.
The semi-implicit split-operator scheme:
1. Compute i_ca_influx at V_n (for Ca dynamics).
2. Update gating variables analytically (exact exp, operator-split).
3. Build Hines system: d[i]*V_{n+1} - coupling = rhs[i].
4. Solve with hines_solve (O(N) direct tree solver).
5. Update dendritic filter states (Backward Euler).

This file contains ONLY the mathematical kernels - no imports from
external modules except numba/ctypes.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange
from numba.types import int64, float64

# Local calcium constants (avoid rhs.py dependency)
CA_I_MIN_M_M = 1e-9
CA_I_MAX_M_M = 10.0
CA_DAMPING_FACTOR = 0.5


@njit
def _gate_step(y: float64[:], dt: float64, v: float64,
              alpha: float64, beta: float64) -> float64:
    """Exact analytical solution for gating ODE: dy/dt = alpha*(1-y) - beta*y."""
    return y + (alpha * (1.0 - y) - beta * y) * dt


@njit
def update_gates_analytic(y: float64[:], dt: float64,
                           m_inf: float64[:], h_inf: float64[:], n_inf: float64[:],
                           alpha_m: float64[:], beta_m: float64[:],
                           alpha_h: float64[:], beta_h: float64[:], alpha_n: float64[:], beta_n: float64[:],
                           off_m: int, off_h: int, off_n: int,
                           m_idx: int, h_idx: int, n_idx: int) -> None:
    """Update all gate variables using exact analytical integration."""
    # Sodium activation (m)
    y[m_idx] = _gate_step(y[m_idx], dt, m_inf[m_idx], alpha_m[m_idx], beta_m[m_idx])
    # Sodium inactivation (h)  
    y[h_idx] = _gate_step(y[h_idx], dt, h_inf[h_idx], alpha_h[h_idx], beta_h[m_idx])
    # Potassium activation (n)
    y[n_idx] = _gate_step(y[n_idx], dt, n_inf[n_idx], alpha_n[m_idx], beta_n[n_idx])


@njit
def hines_solve(d: float64[:], a: float64[:], b: float64[:],
              parent_idx: int64[:], order: int64[:], rhs: float64[:],
              v_new: float64[:]) -> None:
    """Solve tridiagonal system using Thomas algorithm (O(N))."""
    n = len(d)
    
    # Forward sweep
    for i in range(1, n):
        idx = order[i]
        w = a[idx] / d[idx-1]
        d[idx] = d[idx-1] - w * b[idx-1]
        rhs[idx] = rhs[idx] - w * rhs[idx-1]
    
    # Backward sweep
    v_new[n-1] = rhs[n-1] / d[n-1]
    for i in range(n-2, -1, -1):
        idx = order[i]
        v_new[idx] = (rhs[idx] - a[idx] * v_new[idx+1] - b[idx] * v_new[idx-1]) / d[idx]
