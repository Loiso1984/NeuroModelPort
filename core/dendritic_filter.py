"""
Dendritic Filter Module

Implements cable equation-based filtering for synaptic inputs.
When stimulus is applied at dendrites instead of soma,
it must be attenuated, delayed, and temporally filtered before reaching soma.

Physics:
1. Amplitude attenuation: A = exp(-distance / space_constant)
2. Temporal low-pass filter: dI_fil/dt = (I_in - I_fil) / tau_dendritic
3. Signal propagation delay: Δt = distance / conduction_velocity (NEW)
"""

import math
import numpy as np
from numba import njit, float64


class DendriticFilterState:
    """Handle to dendritic filter state variables across integration steps."""
    
    # Typical passive propagation velocity in dendrites (µm/ms = m/s)
    # Myelinated axons: ~1-100 m/s; passive dendrites: ~0.1-0.5 m/s = 100-500 µm/ms
    DEFAULT_CONDUCTION_VELOCITY_UM_MS: float = 250.0  # ~0.25 m/s
    
    def __init__(
        self,
        distance_um: float,
        space_constant_um: float,
        tau_dendritic_ms: float,
        conduction_velocity_um_ms: float | None = None
    ):
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
            
        conduction_velocity_um_ms : float, optional
            Passive signal conduction velocity [µm/ms]
            Determines propagation delay: delay_ms = distance / velocity
            Typical for passive dendrites: 100-500 µm/ms (0.1-0.5 m/s)
            Default: 250 µm/ms
        """
        self.distance_um = distance_um
        self.space_constant_um = space_constant_um
        self.tau_dendritic_ms = tau_dendritic_ms
        
        # Conduction velocity and propagation delay
        self.conduction_velocity_um_ms = conduction_velocity_um_ms or self.DEFAULT_CONDUCTION_VELOCITY_UM_MS
        self.propagation_delay_ms = distance_um / max(self.conduction_velocity_um_ms, 1e-12)
        
        # Pre-compute attenuation factor
        space_const_safe = max(space_constant_um, 1e-12)
        self.attenuation = np.exp(-distance_um / space_const_safe)
        
        # State: filtered (low-pass) version of input
        self.I_filtered = 0.0  # Will be updated each step
        
        # Delay line for propagation delay (circular buffer)
        self._delay_buffer: np.ndarray | None = None
        self._delay_idx: int = 0
        if self.propagation_delay_ms > 0:
            # Will be initialized on first step with proper dt
            self._delay_buffer = None  # Lazy init
            self._delay_steps: int = 0
    
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
    
    def step(self, I_dend_input: float, dt: float) -> float:
        """
        Update filtered state based on input current and return delayed soma current.
        
        Now includes proper signal propagation delay via delay line buffer.
        
        Parameters
        ----------
        I_dend_input : float
            Input current at dendrite (µA/cm²)
        dt : float
            Integration time step (ms)
            
        Returns
        -------
        float
            Current arriving at soma after all filtering (attenuated, low-pass filtered, delayed)
        """
        # 1. Low-pass temporal filtering at dendrite site
        if self.tau_dendritic_ms > 0:
            f = dt / self.tau_dendritic_ms
            I_lp = (self.I_filtered + f * I_dend_input) / (1.0 + f)
        else:
            I_lp = I_dend_input
        self.I_filtered = I_lp
        
        # 2. Propagation delay via circular buffer
        if self.propagation_delay_ms <= dt:
            # Delay less than one timestep - apply immediately
            I_delayed = I_lp
        else:
            # Initialize delay buffer on first call with proper dt
            if self._delay_buffer is None:
                self._delay_steps = max(1, int(round(self.propagation_delay_ms / dt)))
                self._delay_buffer = np.zeros(self._delay_steps)
                self._delay_idx = 0
            
            # Store current filtered value
            self._delay_buffer[self._delay_idx] = I_lp
            
            # Read delayed value (oldest in buffer)
            read_idx = (self._delay_idx + 1) % self._delay_steps
            I_delayed = self._delay_buffer[read_idx]
            
            # Advance write pointer
            self._delay_idx = read_idx
        
        # 3. Amplitude attenuation (spatial decay)
        I_soma = self.attenuation * I_delayed
        
        return I_soma
    
    def get_propagation_delay_ms(self) -> float:
        """Return signal propagation delay from dendrite to soma [ms]."""
        return self.propagation_delay_ms
    
    def reset(self):
        """Reset filter state (for new simulation run)."""
        self.I_filtered = 0.0
        self._delay_idx = 0
        if self._delay_buffer is not None:
            self._delay_buffer.fill(0.0)


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


def validate_dendritic_filter(
    distance_um: float,
    space_constant_um: float,
    tau_dendritic_ms: float = 10.0,
    conduction_velocity_um_ms: float | None = None
) -> dict:
    """
    Sanity check on dendritic filter parameters.
    
    Warns if parameters are outside typical ranges.
    
    Returns
    -------
    dict
        Dictionary with computed properties (attenuation, delay, etc.)
    """
    warnings = []
    
    if distance_um < 10 or distance_um > 500:
        warnings.append(f"distance_um = {distance_um} µm (typical: 50-300 µm)")
    
    if space_constant_um < 50 or space_constant_um > 300:
        warnings.append(f"space_constant_um = {space_constant_um} µm (typical: 100-200 µm)")
    
    if tau_dendritic_ms < 1 or tau_dendritic_ms > 50:
        warnings.append(f"tau_dendritic_ms = {tau_dendritic_ms} ms (typical: 5-20 ms)")
    
    v_cond = conduction_velocity_um_ms or DendriticFilterState.DEFAULT_CONDUCTION_VELOCITY_UM_MS
    if v_cond < 50 or v_cond > 1000:
        warnings.append(f"conduction_velocity = {v_cond} µm/ms (typical: 100-500 µm/ms)")
    
    attenuation = np.exp(-distance_um / max(space_constant_um, 1e-12))
    if attenuation < 0.05:
        warnings.append(f"attenuation = {attenuation:.4f} (very small, signal nearly blocked)")
    if attenuation > 0.99:
        warnings.append(f"attenuation = {attenuation:.4f} (very large, minimal filtering)")
    
    delay_ms = distance_um / max(v_cond, 1e-12)
    
    # Print all warnings
    for w in warnings:
        print(f"⚠️  Warning: {w}")
    
    return {
        'attenuation': attenuation,
        'propagation_delay_ms': delay_ms,
        'effective_tau_ms': tau_dendritic_ms,
        'warnings': warnings
    }


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


@njit(cache=True)
def apply_dendritic_filter_with_delay(
    I_dend: float,
    I_filtered_prev: float,
    tau_dendritic_ms: float,
    attenuation: float,
    dt: float,
    delay_buffer: np.ndarray,
    delay_idx: int
) -> tuple:
    """
    Apply full dendritic filtering with propagation delay (Numba-jitted).
    
    This is an extended version that includes delay line for proper
    temporal separation of dendritic inputs.
    
    Parameters
    ----------
    I_dend : float
        Input current at dendrite
    I_filtered_prev : float
        Previous low-pass filtered value
    tau_dendritic_ms : float
        Low-pass filter time constant
    attenuation : float
        Amplitude attenuation factor
    dt : float
        Time step
    delay_buffer : np.ndarray
        Circular buffer for delay line (modified in-place)
    delay_idx : int
        Current write index in buffer
        
    Returns
    -------
    tuple
        (I_soma, I_filtered_new, new_delay_idx)
    """
    # Low-pass filter
    if tau_dendritic_ms > 0.0:
        f = dt / max(tau_dendritic_ms, 1e-12)
        I_filtered_new = (I_filtered_prev + f * I_dend) / (1.0 + f)
    else:
        I_filtered_new = I_dend
    
    # Store in delay buffer and read delayed value
    n_delay = len(delay_buffer)
    if n_delay > 0:
        delay_buffer[delay_idx] = I_filtered_new
        read_idx = (delay_idx + 1) % n_delay
        I_delayed = delay_buffer[read_idx]
        new_idx = read_idx
    else:
        I_delayed = I_filtered_new
        new_idx = 0
    
    # Apply attenuation
    I_soma = attenuation * I_delayed
    
    return I_soma, I_filtered_new, new_idx
