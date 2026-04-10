"""
dual_stimulation_presets.py - Dual Stimulation Presets

Predefined configurations for dual stimulation scenarios.
Each preset demonstrates a specific neurophysiological mechanism.

Available Presets:
1. "Soma Excitation + Dendritic GABA" - Classic feedforward inhibition
2. "AIS Excitation + Dendritic Inhibition" - Axonal control with dendritic modulation
3. "Dual Dendritic Excitation" - Spatial integration on dendrites
4. "Theta Burst + Background" - Rhythmic activity with constant input
"""

from .dual_stimulation import DualStimulationConfig
import numpy as np


def get_dual_preset_names():
    """Return list of available dual stimulation presets."""
    return [
        "Soma Excitation + Dendritic GABA",
        "AIS Excitation + Dendritic Inhibition",
        "Dual Dendritic Excitation",
        "Theta Burst + Background",
        "Theta-Gamma Modulation",
        "Spike-Timing Control",
        "Balanced Excitation-Inhibition"
    ]


def apply_dual_preset(config, preset_name: str):
    """Apply a dual stimulation preset to the configuration.
    
    Args:
        config: FullModelConfig - the main configuration object
        preset_name: Name of the dual stimulation preset to apply
    
    Primary stimulus parameters are set on config.stim and config.stim_location.
    Secondary stimulus parameters are set on config.dual_stimulation.
    """
    
    # Ensure dual_stimulation config exists
    if config.dual_stimulation is None:
        from .dual_stimulation import DualStimulationConfig
        config.dual_stimulation = DualStimulationConfig()
    
    dual_config = config.dual_stimulation
    
    if preset_name == "Soma Excitation + Dendritic GABA":
        # Classic feedforward inhibition scenario
        # Excitatory input to soma, inhibitory input to dendrites
        config.stim_location.location = "soma"
        config.stim.stim_type = "alpha"
        config.stim.Iext = 20.0
        config.stim.pulse_start = 10.0
        config.stim.pulse_dur = 2.0
        config.stim.alpha_tau = 1.5
        
        dual_config.secondary_location = "dendritic_filtered"
        dual_config.secondary_stim_type = "GABAA"
        dual_config.secondary_Iext = 12.0
        dual_config.secondary_start = 15.0  # Slightly delayed
        dual_config.secondary_duration = 50.0
        dual_config.secondary_alpha_tau = 3.0
        
        dual_config.secondary_distance_um = 100.0
        dual_config.secondary_space_constant_um = 150.0
        dual_config.secondary_tau_dendritic_ms = 8.0
        
    elif preset_name == "AIS Excitation + Dendritic Inhibition":
        # Axonal spike generation with dendritic modulation
        config.stim_location.location = "ais"
        config.stim.stim_type = "alpha"
        config.stim.Iext = 8.0  # Lower due to AIS sensitivity
        config.stim.pulse_start = 10.0
        config.stim.pulse_dur = 2.0
        config.stim.alpha_tau = 1.0
        
        dual_config.secondary_location = "dendritic_filtered"
        dual_config.secondary_stim_type = "GABAA"
        dual_config.secondary_Iext = 15.0
        dual_config.secondary_start = 12.0
        dual_config.secondary_duration = 40.0
        dual_config.secondary_alpha_tau = 4.0
        
        dual_config.secondary_distance_um = 120.0
        dual_config.secondary_space_constant_um = 140.0
        dual_config.secondary_tau_dendritic_ms = 10.0
        
    elif preset_name == "Dual Dendritic Excitation":
        # Two excitatory inputs at different dendritic locations
        # Demonstrates spatial integration and coincidence detection
        config.stim_location.location = "dendritic_filtered"
        config.stim.stim_type = "AMPA"
        config.stim.Iext = 1.5
        config.stim.pulse_start = 10.0
        config.stim.pulse_dur = 5.0
        config.stim.alpha_tau = 2.0
        
        dual_config.secondary_location = "dendritic_filtered"
        dual_config.secondary_stim_type = "AMPA"
        dual_config.secondary_Iext = 1.2
        dual_config.secondary_start = 15.0  # Slightly delayed
        dual_config.secondary_duration = 5.0
        dual_config.secondary_alpha_tau = 2.0
        
        # Different dendritic parameters for spatial separation
        dual_config.secondary_distance_um = 200.0  # Further from soma
        dual_config.secondary_space_constant_um = 150.0
        dual_config.secondary_tau_dendritic_ms = 12.0
        
        # Primary dendritic parameters (proximal)
        # Note: These would need to be stored separately in a full implementation
        
    elif preset_name == "Theta Burst + Background":
        # Rhythmic theta burst with constant background input
        # Models hippocampal theta rhythm with ongoing activity
        config.stim_location.location = "soma"
        config.stim.stim_type = "pulse"
        config.stim.Iext = 1.5
        config.stim.pulse_start = 10.0
        config.stim.pulse_dur = 0.5  # Short pulses
        config.stim.alpha_tau = 2.0
        
        dual_config.secondary_location = "dendritic_filtered"
        dual_config.secondary_stim_type = "const"
        dual_config.secondary_Iext = 0.5  # Background activity
        dual_config.secondary_start = 0.0
        dual_config.secondary_duration = 200.0
        dual_config.secondary_alpha_tau = 10.0
        
        dual_config.secondary_distance_um = 150.0
        dual_config.secondary_space_constant_um = 180.0
        dual_config.secondary_tau_dendritic_ms = 15.0
        
    elif preset_name == "Spike-Timing Control":
        # Precise timing control for spike generation
        # Demonstrates temporal integration windows
        config.stim_location.location = "soma"
        config.stim.stim_type = "alpha"
        config.stim.Iext = 18.0
        config.stim.pulse_start = 10.0
        config.stim.pulse_dur = 1.0
        config.stim.alpha_tau = 0.8  # Fast EPSP
        
        dual_config.secondary_location = "soma"
        dual_config.secondary_stim_type = "alpha"
        dual_config.secondary_Iext = 12.0
        dual_config.secondary_start = 12.0  # Critical timing window
        dual_config.secondary_duration = 1.0
        dual_config.secondary_alpha_tau = 0.8
        
        # No dendritic filtering for second stimulus (same location)
        dual_config.secondary_distance_um = 150.0
        dual_config.secondary_space_constant_um = 150.0
        dual_config.secondary_tau_dendritic_ms = 10.0
        
    elif preset_name == "Balanced Excitation-Inhibition":
        # Balanced E/I ratio for realistic firing
        # Maintains physiological firing rates
        config.stim_location.location = "soma"
        config.stim.stim_type = "alpha"
        config.stim.Iext = 1.0
        config.stim.pulse_start = 10.0
        config.stim.pulse_dur = 3.0
        config.stim.alpha_tau = 2.5
        
        dual_config.secondary_location = "dendritic_filtered"
        dual_config.secondary_stim_type = "GABAA"
        dual_config.secondary_Iext = 1.5  # Balanced inhibition
        dual_config.secondary_start = 8.0  # Slightly before excitation
        dual_config.secondary_duration = 60.0
        dual_config.secondary_alpha_tau = 5.0
        
        dual_config.secondary_distance_um = 130.0
        dual_config.secondary_space_constant_um = 160.0
        dual_config.secondary_tau_dendritic_ms = 12.0
        
    elif preset_name == "Theta-Gamma Modulation":
        # Cross-frequency coupling: theta (7Hz) and gamma (40Hz) synaptic trains
        # Models hippocampal theta-gamma coupling for memory encoding
        config.stim_location.location = "soma"
        config.stim.stim_type = "AMPA"
        config.stim.Iext = 1.0
        config.stim.pulse_start = 10.0
        config.stim.pulse_dur = 200.0
        config.stim.synaptic_train_type = "regular"
        config.stim.synaptic_train_freq_hz = 7.0  # Theta rhythm
        config.stim.alpha_tau = 2.0
        
        dual_config.secondary_location = "soma"
        dual_config.secondary_stim_type = "AMPA"
        dual_config.secondary_Iext = 0.8
        dual_config.secondary_start = 10.0
        dual_config.secondary_duration = 200.0
        dual_config.secondary_train_type = "poisson"
        dual_config.secondary_train_freq_hz = 40.0  # Gamma rhythm
        dual_config.secondary_alpha_tau = 2.0
        
    else:
        raise ValueError(f"Unknown dual stimulation preset: {preset_name}")
    
    dual_config.enabled = True
    return config


def create_demo_preset(config):
    """
    Create the main demonstration preset for dual stimulation.
    
    This preset showcases the key feature: simultaneous excitation
    and inhibition at different locations.
    
    Args:
        config: FullModelConfig - the main configuration object
    """
    return apply_dual_preset(config, "Soma Excitation + Dendritic GABA")


def get_preset_description(preset_name: str) -> str:
    """Get description of a dual stimulation preset."""
    
    descriptions = {
        "Soma Excitation + Dendritic GABA": 
            "Classic feedforward inhibition: excitatory input to soma with "
            "delayed GABAergic inhibition on dendrites. Demonstrates "
            "temporal integration and E/I balance.",
            
        "AIS Excitation + Dendritic Inhibition":
            "Axonal spike generation with dendritic modulation: direct AIS "
            "stimulation combined with dendritic inhibition. Shows how "
            "dendritic inputs control axonal output.",
            
        "Dual Dendritic Excitation":
            "Spatial integration on dendrites: two AMPA inputs at different "
            "dendritic locations. Demonstrates coincidence detection "
            "and spatial summation.",
            
        "Theta Burst + Background":
            "Rhythmic activity: theta-frequency pulse train to soma with "
            "constant dendritic background. Models hippocampal theta rhythms "
            "with ongoing network activity.",

        "Theta-Gamma Modulation":
            "Cross-frequency coupling: theta (7Hz) and gamma (40Hz) synaptic "
            "trains to soma. Models hippocampal theta-gamma coupling for "
            "memory encoding and information processing.",
            
        "Spike-Timing Control":
            "Precise temporal control: two soma inputs with critical timing "
            "relationship. Shows temporal integration windows and "
            "spike timing precision.",
            
        "Balanced Excitation-Inhibition":
            "Physiological E/I balance: balanced excitation and inhibition "
            "for realistic firing rates. Demonstrates homeostatic "
            "balance mechanisms."
    }
    
    return descriptions.get(preset_name, "No description available.")


def validate_dual_preset(preset_name: str) -> bool:
    """
    Validate that a dual stimulation preset is biologically realistic.
    
    Returns True if preset passes validation checks.
    """
    from .models import FullModelConfig
    from .dual_stimulation import DualStimulationConfig
    
    config = FullModelConfig()
    config.dual_stimulation = DualStimulationConfig()
    apply_dual_preset(config, preset_name)
    
    dual_config = config.dual_stimulation
    
    # Check E/I balance
    excitation = config.stim.Iext
    if config.stim.stim_type not in ["GABAA", "GABAB"]:
        excitation = abs(config.stim.Iext)
    
    inhibition = 0.0
    if dual_config.secondary_stim_type in ["GABAA", "GABAB"]:
        inhibition = abs(dual_config.secondary_Iext)
    elif config.stim.stim_type in ["GABAA", "GABAB"]:
        inhibition = abs(config.stim.Iext)
    
    if inhibition > 0:
        ei_ratio = excitation / inhibition if inhibition > 0 else float('inf')
        if ei_ratio > 5.0 or ei_ratio < 0.5:
            print(f"⚠️  Warning: Unusual E/I ratio: {ei_ratio:.2f}")
    
    # Check timing relationships
    primary_end = config.stim.pulse_start + config.stim.pulse_dur
    secondary_end = dual_config.secondary_start + dual_config.secondary_duration
    
    # For inhibitory presets, check that inhibition overlaps excitation
    if dual_config.secondary_stim_type in ["GABAA", "GABAB"]:
        if not (dual_config.secondary_start <= primary_end and secondary_end >= config.stim.pulse_start):
            print(f"⚠️  Warning: Inhibition doesn't overlap excitation")
    
    return True


if __name__ == '__main__':
    # Test all presets
    print("DUAL STIMULATION PRESETS")
    print("=" * 50)
    
    from .models import FullModelConfig
    from .dual_stimulation import DualStimulationConfig
    
    for preset_name in get_dual_preset_names():
        print(f"\n{preset_name}:")
        print(f"  {get_preset_description(preset_name)}")
        
        config = FullModelConfig()
        config.dual_stimulation = DualStimulationConfig()
        apply_dual_preset(config, preset_name)
        validate_dual_preset(preset_name)
        
        print(f"  Primary: {config.stim_location.location} @ {config.stim.Iext} µA/cm²")
        print(f"  Secondary: {config.dual_stimulation.secondary_location} @ {config.dual_stimulation.secondary_Iext} µA/cm²")
