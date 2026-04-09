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
- DualStimulationConfig: Configuration for two stimuli (Pydantic model for serialization)
- DualStimulationState: State tracking for both stimuli
- Integration with existing RHS and solver
"""

import numpy as np
from typing import Tuple, TYPE_CHECKING, List
from numba import njit, float64
from pydantic import BaseModel, Field, ConfigDict

if TYPE_CHECKING:
    from .models import FullModelConfig


class DualStimulationConfig(BaseModel):
    """Configuration for dual stimulation with two independent stimuli (Pydantic model for JSON serialization)."""
    
    # Primary stimulus (e.g., excitatory)
    primary_location: str = "soma"  # soma, ais, dendritic_filtered
    primary_stim_type: str = "const"
    primary_Iext: float = 10.0
    primary_start: float = 10.0
    primary_duration: float = 1.0
    primary_alpha_tau: float = 2.0
    primary_event_times: List[float] = Field(default_factory=list)  # Event queue for primary (e.g., [10, 20, 30] ms)
    
    # Secondary stimulus (e.g., inhibitory)
    secondary_location: str = "dendritic_filtered"
    secondary_stim_type: str = "GABAA"
    secondary_Iext: float = 5.0
    secondary_start: float = 15.0
    secondary_duration: float = 50.0
    secondary_alpha_tau: float = 5.0
    secondary_event_times: List[float] = Field(default_factory=list)  # Event queue for secondary (e.g., [30, 40] ms)
    
    # Secondary train generator (ephemeral, does not mutate manual event_times)
    secondary_train_type: str = Field(default='none', description="Auto-generate spike train (none/regular/poisson)")
    secondary_train_freq_hz: float = Field(default=40.0, ge=0.1, le=500.0)
    secondary_train_duration_ms: float = Field(default=200.0, ge=1.0)
    
    # Secondary stimulus dendritic parameters (if dendritic_filtered)
    secondary_distance_um: float = 150.0
    secondary_space_constant_um: float = 150.0
    secondary_tau_dendritic_ms: float = 10.0
    
    # Enable/disable
    enabled: bool = False
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class DualStimulationState:
    """State tracking for dual stimulation."""
    
    def __init__(self, config: DualStimulationConfig):
        self.config = config
        self.primary_filtered = 0.0
        self.secondary_filtered = 0.0




@njit(cache=True)
def apply_primary_stimulus_current(
    i_stim: np.ndarray,
    n_comp: int,
    base_current: float,
    stim_comp: int,
    stim_mode: int,
    use_dfilter_primary: int,
    dfilter_attenuation: float,
    dfilter_tau_ms: float,
    v_filtered_primary: float,
) -> float:
    """
    Apply primary stimulus current to compartment current vector in-place.

    stim_mode:
    0 = soma, 1 = AIS, 2 = dendritic_filtered
    """
    d_vfiltered_dt_primary = 0.0
    if stim_mode == 0:
        if 0 <= stim_comp < n_comp:
            i_stim[stim_comp] = base_current
    elif stim_mode == 1:
        ais_comp = 1 if n_comp > 1 else 0
        i_stim[ais_comp] = base_current
    elif stim_mode == 2:
        if use_dfilter_primary == 1 and dfilter_tau_ms > 0.0:
            attenuated_current = dfilter_attenuation * base_current
            d_vfiltered_dt_primary = (attenuated_current - v_filtered_primary) / dfilter_tau_ms
            i_stim[0] = v_filtered_primary
        else:
            i_stim[0] = dfilter_attenuation * base_current
    else:
        if 0 <= stim_comp < n_comp:
            i_stim[stim_comp] = base_current
    return d_vfiltered_dt_primary


@njit(cache=True)
def apply_secondary_stimulus_current(
    i_stim: np.ndarray,
    n_comp: int,
    base_current_2: float,
    stim_comp_2: int,
    stim_mode_2: int,
    use_dfilter_secondary: int,
    dfilter_attenuation_2: float,
    dfilter_tau_ms_2: float,
    v_filtered_secondary: float,
) -> float:
    """
    Apply secondary stimulus current to compartment current vector in-place.

    stim_mode_2:
    0 = soma, 1 = AIS, 2 = dendritic_filtered
    """
    d_vfiltered_dt_secondary = 0.0
    if stim_mode_2 == 0:
        if 0 <= stim_comp_2 < n_comp:
            i_stim[stim_comp_2] += base_current_2
    elif stim_mode_2 == 1:
        ais_comp = 1 if n_comp > 1 else 0
        i_stim[ais_comp] += base_current_2
    elif stim_mode_2 == 2:
        if use_dfilter_secondary == 1 and dfilter_tau_ms_2 > 0.0:
            attenuated_current_2 = dfilter_attenuation_2 * base_current_2
            d_vfiltered_dt_secondary = (
                attenuated_current_2 - v_filtered_secondary
            ) / dfilter_tau_ms_2
            i_stim[0] += v_filtered_secondary
        else:
            i_stim[0] += dfilter_attenuation_2 * base_current_2
    else:
        if 0 <= stim_comp_2 < n_comp:
            i_stim[stim_comp_2] += base_current_2
    return d_vfiltered_dt_secondary


@njit(cache=True)
def distributed_stimulus_current_for_comp(
    comp_idx: int,
    n_comp: int,
    base_current: float,
    stim_comp: int,
    stim_mode: int,
    use_dfilter: int,
    dfilter_attenuation: float,
    dfilter_tau_ms: float,
    v_filtered: float,
) -> float:
    """Return stimulus contribution for a single compartment without temp vectors.

    stim_mode:
      0 = soma/custom compartment (uses stim_comp),
      1 = AIS,
      2 = dendritic_filtered (projects to soma via optional filter state).
    """
    if stim_mode == 0:
        return base_current if (0 <= stim_comp < n_comp and comp_idx == stim_comp) else 0.0
    if stim_mode == 1:
        ais_comp = 1 if n_comp > 1 else 0
        return base_current if comp_idx == ais_comp else 0.0
    if stim_mode == 2:
        if comp_idx != 0:
            return 0.0
        # tau <= 0 means filter is effectively disabled (pure attenuation path).
        if use_dfilter == 1 and dfilter_tau_ms > 0.0:
            return v_filtered
        return dfilter_attenuation * base_current
    return base_current if (0 <= stim_comp < n_comp and comp_idx == stim_comp) else 0.0




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


def create_dual_stimulation_config_from_full(config: "FullModelConfig") -> DualStimulationConfig:
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
