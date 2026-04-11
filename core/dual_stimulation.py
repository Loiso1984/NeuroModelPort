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
    """Configuration for dual stimulation with secondary stimulus only.
    
    Primary stimulus is always configured in cfg.stim and cfg.stim_location.
    This config only controls the optional secondary stimulus.
    """
    
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
        """
        Initialize DualStimulationState with a configuration and reset runtime filter state.
        
        Parameters:
            config (DualStimulationConfig): Configuration for the secondary stimulus; stored on the instance.
        
        """
        self.config = config
        self.secondary_filtered = 0.0


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
    i_filtered: float,
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
            return i_filtered
        return dfilter_attenuation * base_current
    return base_current if (0 <= stim_comp < n_comp and comp_idx == stim_comp) else 0.0




def create_dual_stimulation_preset() -> DualStimulationConfig:
    """
    Create a preset that configures a dendritic GABAA secondary stimulus while leaving the primary stimulus to be configured elsewhere.
    
    The preset sets the secondary stimulus to "dendritic_filtered" with GABAA kinetics, an external current of 15.0, start at 20.0 ms, duration 100.0 ms, alpha tau 5.0 ms, dendritic distance 120.0 µm, space constant 150.0 µm, dendritic tau 8.0 ms, and enables the secondary stimulus.
    
    Returns:
        DualStimulationConfig: Configuration object populated with the described secondary-stimulus parameters and enabled=True.
    """
    
    config = DualStimulationConfig()
    
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
    Create a DualStimulationConfig compatible with older code that expected a conversion from FullModelConfig.
    
    This constructs a new DualStimulationConfig with secondary stimulation fields only and sets its `enabled` flag to False; it does not copy any primary-stimulus settings from the provided `FullModelConfig`.
    
    Parameters:
        config (FullModelConfig): The full model configuration being converted (not read or inspected).
    
    Returns:
        DualStimulationConfig: A newly created dual-stimulation configuration with secondary fields initialized and `enabled` set to False.
    """
    
    dual_config = DualStimulationConfig()
    
    # Disable secondary by default
    dual_config.enabled = False
    
    return dual_config


def validate_dual_stimulation_parameters(config: DualStimulationConfig) -> bool:
    """
    Check secondary stimulation parameters for plausibility and print warnings for outliers.
    
    Performs lightweight, non-throwing checks on secondary stimulus values and prints warning messages when values appear biologically implausible (e.g., very large current, unusual dendritic distance or space constant, or very strong dendritic attenuation).
    
    Parameters:
        config (DualStimulationConfig): Configuration containing secondary stimulus fields to validate.
    
    Returns:
        bool: Always returns `True`.
    """
    
    # Check secondary current magnitude
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
    print(f"Secondary: {config.secondary_location} {config.secondary_stim_type} @ {config.secondary_Iext} µA/cm²")
    print(f"Timing: Secondary {config.secondary_start}-{config.secondary_start + config.secondary_duration} ms")
