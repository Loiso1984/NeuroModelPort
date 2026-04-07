"""Physics parameter container for RHS unification.

This module provides a Numba-compatible structured container for all
static physics parameters, eliminating the "argument explosion" problem
and reducing silent-failure risk during signature updates.
"""

from numba import njit
from numba.types import float64, int32, boolean
import numpy as np
from typing import NamedTuple, Optional


class PhysicsParams(NamedTuple):
    """Structured container for all static physics parameters.
    
    This replaces 50+ positional arguments with a single, well-structured
    object that Numba can efficiently unpack without heap allocations.
    """
    # Model size and feature flags
    n_comp: int32
    en_ih: boolean
    en_ica: boolean
    en_ia: boolean
    en_sk: boolean
    dyn_ca: boolean
    en_itca: boolean
    en_im: boolean
    en_nap: boolean
    en_nar: boolean
    
    # Conductance matrix [11 x n_comp]
    # Rows: [gna, gk, gl, gih, gca, ga, gsk, gtca, gim, gnap, gnar]
    gbar_mat: np.ndarray
    
    # Reversal potentials
    ena: float64
    ek: float64
    el: float64
    eih: float64
    ea: float64
    
    # Morphology and axial coupling
    cm_v: np.ndarray
    l_data: np.ndarray
    l_indices: np.ndarray
    l_indptr: np.ndarray
    
    # Temperature scaling matrix [9 x n_comp]
    # Rows: [phi_na, phi_k, phi_ih, phi_ca, phi_ia, phi_tca, phi_im, phi_nap, phi_nar]
    phi_mat: np.ndarray
    
    # Environment and calcium dynamics
    t_kelvin: float64
    ca_ext: float64
    ca_rest: float64
    tau_ca: float64
    b_ca: np.ndarray
    mg_ext: float64
    tau_sk: float64
    
    # Primary stimulation parameters
    stype: int32
    iext: float64
    t0: float64
    td: float64
    atau: float64
    zap_f0_hz: float64
    zap_f1_hz: float64
    event_times_arr: np.ndarray
    n_events: int32
    stim_comp: int32
    stim_mode: int32
    use_dfilter_primary: int32
    dfilter_attenuation: float64
    dfilter_tau_ms: float64
    
    # Secondary stimulation (dual)
    dual_stim_enabled: int32
    stype_2: int32
    iext_2: float64
    t0_2: float64
    td_2: float64
    atau_2: float64
    zap_f0_hz_2: float64
    zap_f1_hz_2: float64
    stim_comp_2: int32
    stim_mode_2: int32
    use_dfilter_secondary: int32
    dfilter_attenuation_2: float64
    dfilter_tau_ms_2: float64
    
    # Stochastic parameters
    stoch_gating: boolean  # Enable Langevin gate noise
    noise_sigma: float64    # Additive membrane current noise
    rng_state: Optional[np.ndarray]  # RNG state for reproducibility


def create_physics_params(**kwargs) -> PhysicsParams:
    """Factory function to create PhysicsParams from keyword arguments.
    
    This function provides a clean interface for creating PhysicsParams
    from existing solver configuration, maintaining backward compatibility
    while enabling the new structured interface.
    
    Args:
        **kwargs: All physics parameters including stochastic settings
        
    Returns:
            PhysicsParams: Structured parameter container
    """
    # Set default stochastic parameters only if not provided
    stochastic_defaults = {
        'stoch_gating': False,
        'noise_sigma': 0.0,
        'rng_state': None
    }
    
    # Only set defaults for missing parameters
    for key, default_val in stochastic_defaults.items():
        if key not in kwargs:
            kwargs[key] = default_val
    
    return PhysicsParams(**kwargs)


@njit
def unpack_conductances(gbar_mat, n_comp):
    """Unpack conductance matrix into individual vectors.
    
    This function extracts individual conductance vectors from the
    packed matrix, avoiding repeated indexing in the main loop.
    """
    gna_v = gbar_mat[0]
    gk_v = gbar_mat[1]
    gl_v = gbar_mat[2]
    gih_v = gbar_mat[3]
    gca_v = gbar_mat[4]
    ga_v = gbar_mat[5]
    gsk_v = gbar_mat[6]
    gtca_v = gbar_mat[7]
    gim_v = gbar_mat[8]
    gnap_v = gbar_mat[9]
    gnar_v = gbar_mat[10]
    
    return (gna_v, gk_v, gl_v, gih_v, gca_v, ga_v, gsk_v, 
            gtca_v, gim_v, gnap_v, gnar_v)


@njit
def unpack_temperature_scaling(phi_mat, n_comp):
    """Unpack temperature scaling matrix into individual vectors."""
    phi_na = phi_mat[0]
    phi_k = phi_mat[1]
    phi_ih = phi_mat[2]
    phi_ca = phi_mat[3]
    phi_ia = phi_mat[4]
    phi_tca = phi_mat[5]
    phi_im = phi_mat[6]
    phi_nap = phi_mat[7]
    phi_nar = phi_mat[8]
    
    return (phi_na, phi_k, phi_ih, phi_ca, phi_ia, phi_tca, 
            phi_im, phi_nap, phi_nar)
