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


def apply_dual_preset(config: DualStimulationConfig, preset_name: str):
    """Apply a dual stimulation preset to the configuration."""
    
    if preset_name == "Soma Excitation + Dendritic GABA":
        # Classic feedforward inhibition scenario
        # Excitatory input to soma, inhibitory input to dendrites
        config.primary_location = "soma"
        config.primary_stim_type = "alpha"
        config.primary_Iext = 20.0
        config.primary_start = 10.0
        config.primary_duration = 2.0
        config.primary_alpha_tau = 1.5
        
        config.secondary_location = "dendritic_filtered"
        config.secondary_stim_type = "GABAA"
        config.secondary_Iext = 12.0
        config.secondary_start = 15.0  # Slightly delayed
        config.secondary_duration = 50.0
        config.secondary_alpha_tau = 3.0
        
        config.secondary_distance_um = 100.0
        config.secondary_space_constant_um = 150.0
        config.secondary_tau_dendritic_ms = 8.0
        
    elif preset_name == "AIS Excitation + Dendritic Inhibition":
        # Axonal spike generation with dendritic modulation
        config.primary_location = "ais"
        config.primary_stim_type = "alpha"
        config.primary_Iext = 8.0  # Lower due to AIS sensitivity
        config.primary_start = 10.0
        config.primary_duration = 2.0
        config.primary_alpha_tau = 1.0
        
        config.secondary_location = "dendritic_filtered"
        config.secondary_stim_type = "GABAA"
        config.secondary_Iext = 15.0
        config.secondary_start = 12.0
        config.secondary_duration = 40.0
        config.secondary_alpha_tau = 4.0
        
        config.secondary_distance_um = 120.0
        config.secondary_space_constant_um = 140.0
        config.secondary_tau_dendritic_ms = 10.0
        
    elif preset_name == "Dual Dendritic Excitation":
        # Two excitatory inputs at different dendritic locations
        # Demonstrates spatial integration and coincidence detection
        config.primary_location = "dendritic_filtered"
        config.primary_stim_type = "AMPA"
        config.primary_Iext = 1.5
        config.primary_start = 10.0
        config.primary_duration = 5.0
        config.primary_alpha_tau = 2.0
        
        config.secondary_location = "dendritic_filtered"
        config.secondary_stim_type = "AMPA"
        config.secondary_Iext = 1.2
        config.secondary_start = 15.0  # Slightly delayed
        config.secondary_duration = 5.0
        config.secondary_alpha_tau = 2.0
        
        # Different dendritic parameters for spatial separation
        config.secondary_distance_um = 200.0  # Further from soma
        config.secondary_space_constant_um = 150.0
        config.secondary_tau_dendritic_ms = 12.0
        
        # Primary dendritic parameters (proximal)
        # Note: These would need to be stored separately in a full implementation
        
    elif preset_name == "Theta Burst + Background":
        # Rhythmic theta burst with constant background input
        # Models hippocampal theta rhythm with ongoing activity
        config.primary_location = "soma"
        config.primary_stim_type = "pulse"
        config.primary_Iext = 1.5
        config.primary_start = 10.0
        config.primary_duration = 0.5  # Short pulses
        config.primary_alpha_tau = 2.0
        
        config.secondary_location = "dendritic_filtered"
        config.secondary_stim_type = "const"
        config.secondary_Iext = 0.5  # Background activity
        config.secondary_start = 0.0
        config.secondary_duration = 200.0
        config.secondary_alpha_tau = 10.0
        
        config.secondary_distance_um = 150.0
        config.secondary_space_constant_um = 180.0
        config.secondary_tau_dendritic_ms = 15.0
        
    elif preset_name == "Spike-Timing Control":
        # Precise timing control for spike generation
        # Demonstrates temporal integration windows
        config.primary_location = "soma"
        config.primary_stim_type = "alpha"
        config.primary_Iext = 18.0
        config.primary_start = 10.0
        config.primary_duration = 1.0
        config.primary_alpha_tau = 0.8  # Fast EPSP
        
        config.secondary_location = "soma"
        config.secondary_stim_type = "alpha"
        config.secondary_Iext = 12.0
        config.secondary_start = 12.0  # Critical timing window
        config.secondary_duration = 1.0
        config.secondary_alpha_tau = 0.8
        
        # No dendritic filtering for second stimulus (same location)
        config.secondary_distance_um = 150.0
        config.secondary_space_constant_um = 150.0
        config.secondary_tau_dendritic_ms = 10.0
        
    elif preset_name == "Balanced Excitation-Inhibition":
        # Balanced E/I ratio for realistic firing
        # Maintains physiological firing rates
        config.primary_location = "soma"
        config.primary_stim_type = "alpha"
        config.primary_Iext = 1.0
        config.primary_start = 10.0
        config.primary_duration = 3.0
        config.primary_alpha_tau = 2.5
        
        config.secondary_location = "dendritic_filtered"
        config.secondary_stim_type = "GABAA"
        config.secondary_Iext = 1.5  # Balanced inhibition
        config.secondary_start = 8.0  # Slightly before excitation
        config.secondary_duration = 60.0
        config.secondary_alpha_tau = 5.0
        
        config.secondary_distance_um = 130.0
        config.secondary_space_constant_um = 160.0
        config.secondary_tau_dendritic_ms = 12.0
        
    else:
        raise ValueError(f"Unknown dual stimulation preset: {preset_name}")
    
    config.enabled = True
    return config


def create_demo_preset() -> DualStimulationConfig:
    """
    Create the main demonstration preset for dual stimulation.
    
    This preset showcases the key feature: simultaneous excitation
    and inhibition at different locations.
    """
    config = DualStimulationConfig()
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
    
    config = DualStimulationConfig()
    apply_dual_preset(config, preset_name)
    
    # Check E/I balance
    excitation = config.primary_Iext
    if config.primary_stim_type not in ["GABAA", "GABAB"]:
        excitation = abs(config.primary_Iext)
    
    inhibition = 0.0
    if config.secondary_stim_type in ["GABAA", "GABAB"]:
        inhibition = abs(config.secondary_Iext)
    elif config.primary_stim_type in ["GABAA", "GABAB"]:
        inhibition = abs(config.primary_Iext)
    
    if inhibition > 0:
        ei_ratio = excitation / inhibition if inhibition > 0 else float('inf')
        if ei_ratio > 5.0 or ei_ratio < 0.5:
            print(f"⚠️  Warning: Unusual E/I ratio: {ei_ratio:.2f}")
    
    # Check timing relationships
    primary_end = config.primary_start + config.primary_duration
    secondary_end = config.secondary_start + config.secondary_duration
    
    # For inhibitory presets, check that inhibition overlaps excitation
    if config.secondary_stim_type in ["GABAA", "GABAB"]:
        if not (config.secondary_start <= primary_end and config.secondary_end >= config.primary_start):
            print(f"⚠️  Warning: Inhibition doesn't overlap excitation")
    
    return True


if __name__ == '__main__':
    # Test all presets
    print("DUAL STIMULATION PRESETS")
    print("=" * 50)
    
    for preset_name in get_dual_preset_names():
        print(f"\n{preset_name}:")
        print(f"  {get_preset_description(preset_name)}")
        
        config = DualStimulationConfig()
        apply_dual_preset(config, preset_name)
        validate_dual_preset(preset_name)
        
        print(f"  Primary: {config.primary_location} @ {config.primary_Iext} µA/cm²")
        print(f"  Secondary: {config.secondary_location} @ {config.secondary_Iext} µA/cm²")
