"""Native Hines solver kernels — v11.1.

TRUE O(N) tree solver for branched morphologies replacing SciPy BDF.
The semi-implicit split-operator scheme:
1. Compute i_ca_influx at V_n (for Ca dynamics).
2. Update gating variables using Exponential Euler (stable, dt-independent).
3. Build Hines system: d[i]*V_{n+1} - coupling = rhs[i].
4. Solve with hines_solve (TRUE tree solver, handles branches).
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
def _gate_step(y_old: float64, dt: float64, alpha: float64, beta: float64) -> float64:
    """Exponential Euler: dy/dt = alpha*(1-y) - beta*y.
    
    Analytical solution for stability (dt-independent).
    """
    sum_ab = alpha + beta
    if sum_ab < 1e-12:
        return y_old
    y_inf = alpha / sum_ab
    tau = 1.0 / sum_ab
    return y_inf + (y_old - y_inf) * np.exp(-dt / tau)



@njit
def hines_solve(d: float64[:], a: float64[:], b: float64[:],
              parent_idx: int64[:], order: int64[:], rhs: float64[:],
              v_new: float64[:]) -> None:
    """TRUE Linear-time (O(N)) tree solver for branched morphologies.
    
    Correctly handles tree structures (not just tridiagonal chains).
    Forward elimination from leaves to root, backward substitution root to leaves.
    """
    n = len(d)
    # Temporary working buffers to avoid mutating original d/rhs
    d_work = d.copy()
    rhs_work = rhs.copy()

    # 1. Forward elimination (Leaves to Root)
    # 'order' contains indices from distal to proximal
    for i in range(n - 1):  # Root is last, don't eliminate it
        idx = order[i]
        p = parent_idx[idx]
        if p < 0:
            continue
        
        factor = a[idx] / d_work[idx]
        d_work[p] -= factor * b[idx]
        rhs_work[p] -= factor * rhs_work[idx]

    # 2. Backward substitution (Root to Leaves)
    # Root is at the end of 'order'
    root_idx = order[n-1]
    v_new[root_idx] = rhs_work[root_idx] / d_work[root_idx]
    
    for i in range(n - 2, -1, -1):
        idx = order[i]
        p = parent_idx[idx]
        v_new[idx] = (rhs_work[idx] - b[idx] * v_new[p]) / d_work[idx]
