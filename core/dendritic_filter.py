"""
Dendritic Filter Module

Implements cable equation-based filtering for synaptic inputs.
When stimulus is applied at dendrites instead of soma,
it must be attenuated and temporally filtered before reaching soma.

Physics:
1. Amplitude attenuation: A = exp(-distance / space_constant)
2. Temporal low-pass filter: dI_fil/dt = (I_in - I_fil) / tau_dendritic
"""

import math
import numpy as np
from numba import njit, float64


class DendriticFilterState:
    """Handle to dendritic filter state variables across integration steps."""
    
    def __init__(self, distance_um: float, space_constant_um: float, tau_dendritic_ms: float):
        """
        Initialize dendritic filter parameters.
        
        Parameters
        ----------
        distance_um : float
            Distance from soma to stimulation site (µm)
            Typical: 50-300 µm for synaptic inputs on different dendrite segments
        
        space_constant_um : float
            Cable space constant λ in micrometers
            λ = sqrt(a / (4ρ*g_m)) where:
              a = fiber radius [cm]
              ρ = axial resistivity [Ω·cm]
              g_m = membrane conductance per length [S/cm]
            Typical for soma dendrites: 100-200 µm
        
        tau_dendritic_ms : float
            Low-pass temporal filter time constant [ms]
            Related to dendritic-soma impedance transfer
            Typical: 5-20 ms
        """
        self.distance_um = distance_um
        self.space_constant_um = space_constant_um
        self.tau_dendritic_ms = tau_dendritic_ms
        
        # Pre-compute attenuation factor
        space_const_safe = max(space_constant_um, 1e-12)
        self.attenuation = np.exp(-distance_um / space_const_safe)
        
        # State: filtered (low-pass) version of input
        self.I_filtered = 0.0  # Will be updated each step
    
    def get_attenuation(self) -> float:
        """Return amplitude attenuation factor [dimensionless]."""
        return self.attenuation
    
    def get_soma_current(self, I_dend: float) -> float:
        """
        Compute soma current from dendritic input.
        
        In practice, this uses the pre-filtered value from previous step:
        I_soma = attenuation * I_filtered
        
        where I_filtered evolves as:
        dI_fil/dt = (I_dend - I_fil) / tau_dendritic
        """
        return self.attenuation * self.I_filtered
    
    def step(self, I_dend_input: float, dt: float):
        """
        Update filtered state based on input current.
        
        Parameters
        ----------
        I_dend_input : float
            Input current at dendrite (µA/cm²)
        dt : float
            Integration time step (ms)
        """
        if self.tau_dendritic_ms > 0:
            f = dt / self.tau_dendritic_ms
            self.I_filtered = (self.I_filtered + f * I_dend_input) / (1.0 + f)
        else:
            self.I_filtered = I_dend_input


@njit(cache=True)
def apply_dendritic_filter(
    I_dend: float,
    I_filtered_prev: float,
    tau_dendritic_ms: float,
    attenuation: float,
    dt: float
) -> tuple:
    """
    Apply dendritic cable filtering to current.
    
    Numba-compiled for speed (called from RHS kernel).
    
    Returns
    -------
    (I_soma, I_filtered_new) : tuple
        I_soma: current arriving at soma (attenuated, filtered)
        I_filtered_new: updated low-pass filter state
    """
    
    if tau_dendritic_ms > 0.0:
        f = dt / max(tau_dendritic_ms, 1e-12)
        I_filtered_new = (I_filtered_prev + f * I_dend) / (1.0 + f)
    else:
        I_filtered_new = I_dend
    
    # Amplitude attenuation (cable decay with distance)
    I_soma = attenuation * I_filtered_new
    
    return I_soma, I_filtered_new


def validate_dendritic_filter(distance_um: float, space_constant_um: float) -> bool:
    """
    Sanity check on dendritic filter parameters.
    
    Warns if parameters are outside typical ranges.
    """
    
    if distance_um < 10 or distance_um > 500:
        print(f"⚠️  Warning: distance_um = {distance_um} µm (typical: 50-300 µm)")
    
    if space_constant_um < 50 or space_constant_um > 300:
        print(f"⚠️  Warning: space_constant_um = {space_constant_um} µm (typical: 100-200 µm)")
    
    attenuation = np.exp(-distance_um / max(space_constant_um, 1e-12))
    if attenuation < 0.05:
        print(f"⚠️  Warning: attenuation = {attenuation:.4f} (very small, signal nearly blocked)")
    if attenuation > 0.99:
        print(f"⚠️  Warning: attenuation = {attenuation:.4f} (very large, minimal filtering)")
    
    return True


@njit(float64(float64, float64, float64, float64), fastmath=True, cache=True)
def get_ac_attenuation(distance_um, lambda_dc_um, tau_m_ms, freq_hz):
    """
    AC signal attenuation for dendritic propagation (Numba-jitted).
    
    High-frequency signals attenuate faster than DC space constant predicts.
    AC space constant: λ_AC = λ_DC / sqrt(1 + j·ω·τ_m)
    Attenuation magnitude: |A| = exp(-x · Re(1/λ_AC))
    
    Algebraic simplification: Re(1/λ_AC) = sqrt((sqrt(1+(ωτ)²) + 1) / 2) / λ_DC
    """
    # Guard against invalid space constant
    if lambda_dc_um <= 0.0:
        return 0.0
    
    # Angular frequency × time constant [dimensionless]
    # tau_m in ms, freq in Hz -> ωτ = 2π·f·τ·0.001
    omega_tau = 6.283185307179586 * freq_hz * tau_m_ms * 0.001
    
    # Magnitude of (1 + jωτ): r = sqrt(1 + (ωτ)²)
    r = math.sqrt(1.0 + omega_tau * omega_tau)
    
    # Re(1/λ_AC) = sqrt((r + 1)/2) / λ_DC
    re_inv_lambda = math.sqrt((r + 1.0) * 0.5) / lambda_dc_um
    
    # Attenuation magnitude: exp(-x · Re(1/λ_AC))
    return math.exp(-distance_um * re_inv_lambda)
