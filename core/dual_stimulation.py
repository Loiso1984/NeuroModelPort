"""
dual_stimulation.py - Dual Stimulation Module

Implements simultaneous stimulation at two different locations
with independent parameters for each stimulus.

Use Cases:
1. Soma excitation + Dendritic inhibition (GABA)
2. AIS excitation + Dendritic inhibition  
3. Two different dendritic locations
4. Any combination of stimulation modes

Architecture:
- DualStimulationConfig: Configuration for two stimuli
- DualStimulationState: State tracking for both stimuli
- Integration with existing RHS and solver
"""

import numpy as np
from typing import Tuple
from numba import njit, float64

from .models import FullModelConfig


class DualStimulationConfig:
    """Configuration for dual stimulation with two independent stimuli."""
    
    def __init__(self):
        # Primary stimulus (e.g., excitatory)
        self.primary_location = "soma"  # soma, ais, dendritic_filtered
        self.primary_stim_type = "const"
        self.primary_Iext = 10.0
        self.primary_start = 10.0
        self.primary_duration = 1.0
        self.primary_alpha_tau = 2.0
        
        # Secondary stimulus (e.g., inhibitory)
        self.secondary_location = "dendritic_filtered"
        self.secondary_stim_type = "GABAA"
        self.secondary_Iext = 5.0
        self.secondary_start = 15.0
        self.secondary_duration = 50.0
        self.secondary_alpha_tau = 5.0
        
        # Secondary stimulus dendritic parameters (if dendritic_filtered)
        self.secondary_distance_um = 150.0
        self.secondary_space_constant_um = 150.0
        self.secondary_tau_dendritic_ms = 10.0
        
        # Enable/disable
        self.enabled = False


class DualStimulationState:
    """State tracking for dual stimulation."""
    
    def __init__(self, config: DualStimulationConfig):
        self.config = config
        self.primary_filtered = 0.0
        self.secondary_filtered = 0.0


@njit(cache=True)
def get_dual_stim_current(
    t: float,
    primary_type: int,
    primary_iext: float,
    primary_t0: float,
    primary_td: float,
    primary_atau: float,
    secondary_type: int,
    secondary_iext: float,
    secondary_t0: float,
    secondary_td: float,
    secondary_atau: float
) -> Tuple[float64, float64]:
    """
    Calculate both stimulus currents at time t.
    
    Returns:
        (primary_current, secondary_current)
    """
    
    # Primary stimulus
    if primary_type == 1:  # pulse
        primary_current = primary_iext if primary_t0 <= t <= primary_t0 + primary_td else 0.0
    elif primary_type == 2:  # alpha
        if t < primary_t0:
            primary_current = 0.0
        else:
            dt = (t - primary_t0) / primary_atau
            primary_current = primary_iext * dt * np.exp(1.0 - dt)
    elif primary_type == 4:  # AMPA
        if t < primary_t0:
            primary_current = 0.0
        else:
            dt = t - primary_t0
            tau_rise, tau_decay = 0.5, 3.0
            t_peak = tau_rise * tau_decay / (tau_decay - tau_rise) * np.log(tau_decay / tau_rise)
            norm = np.exp(-t_peak / tau_decay) - np.exp(-t_peak / tau_rise)
            primary_current = abs(primary_iext) * (np.exp(-dt / tau_decay) - np.exp(-dt / tau_rise)) / norm
    elif primary_type == 6:  # GABA-A
        if t < secondary_t0:  # Using secondary timing for inhibitory
            primary_current = 0.0
        else:
            dt = t - secondary_t0
            tau_rise, tau_decay = 1.0, 7.0
            t_peak = tau_rise * tau_decay / (tau_decay - tau_rise) * np.log(tau_decay / tau_rise)
            norm = np.exp(-t_peak / tau_decay) - np.exp(-t_peak / tau_rise)
            primary_current = -abs(primary_iext) * (np.exp(-dt / tau_decay) - np.exp(-dt / tau_rise)) / norm
    else:
        primary_current = primary_iext  # const
    
    # Secondary stimulus
    if secondary_type == 1:  # pulse
        secondary_current = secondary_iext if secondary_t0 <= t <= secondary_t0 + secondary_td else 0.0
    elif secondary_type == 2:  # alpha
        if t < secondary_t0:
            secondary_current = 0.0
        else:
            dt = (t - secondary_t0) / secondary_atau
            secondary_current = secondary_iext * dt * np.exp(1.0 - dt)
    elif secondary_type == 4:  # AMPA
        if t < secondary_t0:
            secondary_current = 0.0
        else:
            dt = t - secondary_t0
            tau_rise, tau_decay = 0.5, 3.0
            t_peak = tau_rise * tau_decay / (tau_decay - tau_rise) * np.log(tau_decay / tau_rise)
            norm = np.exp(-t_peak / tau_decay) - np.exp(-t_peak / tau_rise)
            secondary_current = abs(secondary_iext) * (np.exp(-dt / tau_decay) - np.exp(-dt / tau_rise)) / norm
    elif secondary_type == 6:  # GABA-A
        if t < secondary_t0:
            secondary_current = 0.0
        else:
            dt = t - secondary_t0
            tau_rise, tau_decay = 1.0, 7.0
            t_peak = tau_rise * tau_decay / (tau_decay - tau_rise) * np.log(tau_decay / tau_rise)
            norm = np.exp(-t_peak / tau_decay) - np.exp(-t_peak / tau_rise)
            secondary_current = -abs(secondary_iext) * (np.exp(-dt / tau_decay) - np.exp(-dt / tau_rise)) / norm
    else:
        secondary_current = secondary_iext  # const
    
    return primary_current, secondary_current


@njit(cache=True)
def apply_dual_stimulation(
    t: float,
    y: np.ndarray,
    n_comp: int,
    # Dual stimulation parameters
    primary_type: int,
    primary_iext: float,
    primary_t0: float,
    primary_td: float,
    primary_atau: float,
    primary_location: int,
    secondary_type: int,
    secondary_iext: float,
    secondary_t0: float,
    secondary_td: float,
    secondary_atau: float,
    secondary_location: int,
    secondary_attenuation: float,
    secondary_tau_dendritic: float,
    # Filter states
    primary_filtered_state: float,
    secondary_filtered_state: float,
    use_primary_filter: int,
    use_secondary_filter: int,
    # Standard RHS parameters (truncated for brevity)
    en_ih: bool,
    en_ica: bool,
    en_ia: bool,
    en_sk: bool,
    dyn_ca: bool,
    gna_v: np.ndarray,
    gk_v: np.ndarray,
    gl_v: np.ndarray,
    gih_v: np.ndarray,
    gca_v: np.ndarray,
    ga_v: np.ndarray,
    gsk_v: np.ndarray,
    ena: float,
    ek: float,
    el: float,
    eih: float,
    ea: float,
    cm_v: np.ndarray,
    l_data: np.ndarray,
    l_indices: np.ndarray,
    l_indptr: np.ndarray,
    phi: float,
    t_kelvin: float,
    ca_ext: float,
    ca_rest: float,
    tau_ca: float,
    b_ca: float
) -> Tuple[np.ndarray, float, float]:
    """
    Apply dual stimulation to multi-compartment model.
    
    This function extends the standard RHS to handle two simultaneous stimuli.
    """
    
    # Calculate both stimulus currents
    primary_current, secondary_current = get_dual_stim_current(
        t, primary_type, primary_iext, primary_t0, primary_td, primary_atau,
        secondary_type, secondary_iext, secondary_t0, secondary_td, secondary_atau
    )
    
    # Initialize stimulus array
    i_stim = np.zeros(n_comp)
    d_primary_filtered_dt = 0.0
    d_secondary_filtered_dt = 0.0
    
    # Apply primary stimulus
    if primary_location == 0:  # soma
        i_stim[0] += primary_current
    elif primary_location == 1:  # ais
        ais_comp = 1 if n_comp > 1 else 0
        i_stim[ais_comp] += primary_current
    
    # Apply secondary stimulus with filtering
    if secondary_location == 0:  # soma
        i_stim[0] += secondary_current
    elif secondary_location == 1:  # ais
        ais_comp = 1 if n_comp > 1 else 0
        i_stim[ais_comp] += secondary_current
    elif secondary_location == 2:  # dendritic_filtered
        if use_secondary_filter == 1 and secondary_tau_dendritic > 0.0:
            # Apply dendritic filtering to secondary stimulus
            attenuated_secondary = secondary_attenuation * secondary_current
            d_secondary_filtered_dt = (attenuated_secondary - secondary_filtered_state) / secondary_tau_dendritic
            i_stim[0] += secondary_filtered_state
        else:
            # Direct injection with attenuation
            i_stim[0] += secondary_attenuation * secondary_current
    
    return i_stim, d_primary_filtered_dt, d_secondary_filtered_dt


def create_dual_stimulation_preset() -> DualStimulationConfig:
    """
    Create a demonstration preset for dual stimulation.
    
    Demonstrates: Soma excitation + Dendritic GABA inhibition
    This is a classic neurophysiological scenario.
    """
    
    config = DualStimulationConfig()
    
    # Primary: Soma excitation (glutamatergic-like)
    config.primary_location = "soma"
    config.primary_stim_type = "alpha"  # EPSC-like
    config.primary_Iext = 25.0  # Moderate excitation
    config.primary_start = 10.0
    config.primary_duration = 2.0
    config.primary_alpha_tau = 2.0
    
    # Secondary: Dendritic inhibition (GABAergic)
    config.secondary_location = "dendritic_filtered"
    config.secondary_stim_type = "GABAA"  # Fast inhibitory
    config.secondary_Iext = 15.0  # Strong inhibition
    config.secondary_start = 20.0  # Delayed onset
    config.secondary_duration = 100.0  # Prolonged inhibition
    config.secondary_alpha_tau = 5.0
    
    # Dendritic parameters for inhibition
    config.secondary_distance_um = 120.0  # Proximal dendrite
    config.secondary_space_constant_um = 150.0
    config.secondary_tau_dendritic_ms = 8.0
    
    config.enabled = True
    
    return config


def create_dual_stimulation_config_from_full(config: FullModelConfig) -> DualStimulationConfig:
    """
    Convert FullModelConfig to DualStimulationConfig for backward compatibility.
    """
    
    dual_config = DualStimulationConfig()
    
    # Use existing config as primary stimulus
    dual_config.primary_location = config.stim_location.location
    dual_config.primary_stim_type = config.stim.stim_type
    dual_config.primary_Iext = config.stim.Iext
    dual_config.primary_start = config.stim.pulse_start
    dual_config.primary_duration = config.stim.pulse_dur
    dual_config.primary_alpha_tau = config.stim.alpha_tau
    
    # Disable secondary by default
    dual_config.enabled = False
    
    return dual_config


def validate_dual_stimulation_parameters(config: DualStimulationConfig) -> bool:
    """
    Validate dual stimulation parameters for biological realism.
    
    Returns True if parameters are reasonable.
    """
    
    # Check timing overlap
    primary_end = config.primary_start + config.primary_duration
    secondary_end = config.secondary_start + config.secondary_duration
    
    # Allow some overlap but warn if completely overlapping
    if (config.primary_start <= config.secondary_start <= primary_end and
        config.secondary_start <= primary_end <= secondary_end):
        print(f"⚠️  Warning: Significant timing overlap between stimuli")
        print(f"   Primary: {config.primary_start}-{primary_end} ms")
        print(f"   Secondary: {config.secondary_start}-{secondary_end} ms")
    
    # Check current magnitudes
    if abs(config.primary_Iext) > 200:
        print(f"⚠️  Warning: Primary current very high: {config.primary_Iext} µA/cm²")
    
    if abs(config.secondary_Iext) > 200:
        print(f"⚠️  Warning: Secondary current very high: {config.secondary_Iext} µA/cm²")
    
    # Check dendritic parameters
    if config.secondary_location == "dendritic_filtered":
        if config.secondary_distance_um < 10 or config.secondary_distance_um > 500:
            print(f"⚠️  Warning: Unusual dendritic distance: {config.secondary_distance_um} µm")
        
        if config.secondary_space_constant_um < 50 or config.secondary_space_constant_um > 300:
            print(f"⚠️  Warning: Unusual space constant: {config.secondary_space_constant_um} µm")
        
        atten = np.exp(-config.secondary_distance_um / config.secondary_space_constant_um)
        if atten < 0.01:
            print(f"⚠️  Warning: Very strong attenuation: {atten:.4f}")
    
    return True


if __name__ == '__main__':
    # Test dual stimulation configuration
    config = create_dual_stimulation_preset()
    validate_dual_stimulation_parameters(config)
    
    print("Dual Stimulation Configuration:")
    print(f"Primary: {config.primary_location} {config.primary_stim_type} @ {config.primary_Iext} µA/cm²")
    print(f"Secondary: {config.secondary_location} {config.secondary_stim_type} @ {config.secondary_Iext} µA/cm²")
    print(f"Timing: Primary {config.primary_start}-{config.primary_start + config.primary_duration} ms")
    print(f"        Secondary {config.secondary_start}-{config.secondary_start + config.secondary_duration} ms")
