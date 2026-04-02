"""
Unit Conversion Utilities for Display and User Interface v10.0

This module provides functions to convert between internal representation units 
and user-facing display units.

INTERNAL REPRESENTATION:
  - Conductances: mS (millisiemens) - absolute, pre-multiplied by area
  - Capacitances: µF (microfarad) - absolute, pre-multiplied by area
  - Currents (Iext, noise): pA (picoamperes) - absolute injected current
  - Time: ms (milliseconds)
  - Voltage: mV (millivolts)

DISPLAY UNITS (User-Friendly):
  - Conductance density: mS/cm² (for reference, showing g_max before area scaling)
  - Capacitance density: µF/cm² (for reference, showing Cm before area scaling)
  - Currents: nanoamperes (nA) - easier to read than picoamperes
  - Time: milliseconds (ms)
  - Voltage: millivolts (mV)
"""

import numpy as np
from typing import Union

def pa_to_na(i_pa: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert picoamperes (pA) to nanoamperes (nA).
    
    Parameters
    ----------
    i_pa : float or array
        Current in picoamperes
        
    Returns
    -------
    float or array
        Current in nanoamperes (1 nA = 1000 pA)
    """
    return i_pa / 1000.0

def na_to_pa(i_na: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert nanoamperes (nA) to picoamperes (pA).
    
    Parameters
    ----------
    i_na : float or array
        Current in nanoamperes
        
    Returns
    -------
    float or array
        Current in picoamperes
    """
    return i_na * 1000.0

def pa_to_ua(i_pa: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert picoamperes (pA) to microamperes (µA).
    
    Parameters
    ----------
    i_pa : float or array
        Current in picoamperes
        
    Returns
    -------
    float or array
        Current in microamperes (1 µA = 10^6 pA)
    """
    return i_pa / 1e6

def ua_to_pa(i_ua: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert microamperes (µA) to picoamperes (pA).
    
    Parameters
    ----------
    i_ua : float or array
        Current in microamperes
        
    Returns
    -------
    float or array
        Current in picoamperes
    """
    return i_ua * 1e6

def format_current_for_display(i_pA: float, prefer_unit: str = 'auto') -> str:
    """
    Format a current value in pA for user-readable display.
    
    Parameters
    ----------
    i_pA : float
        Current in picoamperes (internal representation)
    prefer_unit : str
        'auto'  - Choose unit based on magnitude (default)
        'nA'    - Always display in nanoamperes
        'µA'    - Always display in microamperes
        'pA'    - Always display in picoamperes
        
    Returns
    -------
    str
        Formatted string with value and unit, e.g. "5.2 nA" or "12.5 µA"
        
    Examples
    --------
    >>> format_current_for_display(1500)      # 1.5 nanoamperes
    '1.5 nA'
    
    >>> format_current_for_display(5000000)   # 5 microamperes
    '5.0 µA'
    
    >>> format_current_for_display(50)        # 50 picoamperes
    '50 pA'
    """
    
    abs_i = abs(i_pA)
    sign = '-' if i_pA < 0 else ''
    
    if prefer_unit == 'pA':
        return f"{sign}{abs_i:.0f} pA"
    elif prefer_unit == 'nA':
        return f"{sign}{abs_i/1000:.2f} nA"
    elif prefer_unit == 'µA':
        return f"{sign}{abs_i/1e6:.2f} µA"
    else:  # 'auto'
        if abs_i < 100:  # < 0.1 nA: show in pA
            return f"{sign}{abs_i:.0f} pA"
        elif abs_i < 1e6:  # 0.1 nA to 1 µA: show in nA
            return f"{sign}{abs_i/1000:.2f} nA"
        else:  # ≥ 1 µA: show in µA
            return f"{sign}{abs_i/1e6:.2f} µA"

def format_conductance_density(g_mS_cm2: float) -> str:
    """
    Format conductance density for display.
    
    Parameters
    ----------
    g_mS_cm2 : float
        Conductance density in mS/cm²
        
    Returns
    -------
    str
        Formatted string, e.g. "120.0 mS/cm²"
    """
    return f"{g_mS_cm2:.1f} mS/cm²"

def format_capacitance_density(cm_uF_cm2: float) -> str:
    """
    Format capacitance density for display.
    
    Parameters
    ----------
    cm_uF_cm2 : float
        Capacitance density in µF/cm²
        
    Returns
    -------
    str
        Formatted string, e.g. "1.0 µF/cm²"
    """
    return f"{cm_uF_cm2:.2f} µF/cm²"

def describe_conductance(g_density_mS_cm2: float, soma_area_cm2: float) -> str:
    """
    Describe a conductance with both density and absolute values.
    
    This is useful for showing users the "physical" values they're working with.
    
    Parameters
    ----------
    g_density_mS_cm2 : float
        Conductance density in mS/cm²
    soma_area_cm2 : float
        Soma surface area in cm²
        
    Returns
    -------
    str
        Formatted description, e.g. "120.0 mS/cm² (× 7.85e-5 cm² → 9.4 mS absolute)"
    """
    g_absolute_mS = g_density_mS_cm2 * soma_area_cm2
    # Only show detailed breakdown if area is significantly different from 1 cm²
    if soma_area_cm2 < 0.1 or soma_area_cm2 > 10.0:
        return f"{g_density_mS_cm2:.1f} mS/cm² (→ {g_absolute_mS:.3e} mS absolute)"
    else:
        return format_conductance_density(g_density_mS_cm2)

def describe_stimulus_current(i_pA: float, soma_diameter_um: float = 29) -> str:
    """
    Describe a stimulus current with conversion factor shown.
    
    Helps users understand the relationship between internal currents and real values.
    
    Parameters
    ----------
    i_pA : float
        Stimulus current in picoamperes
    soma_diameter_um : float
        Soma diameter in micrometers (for informative context)
        
    Returns
    -------
    str
        Description like "1.5 nA (soma: 29 µm)"
    """
    return f"{format_current_for_display(i_pA, prefer_unit='nA')} (soma: {soma_diameter_um:.0f} µm)"

# ═══════════════════════════════════════════════════════════════════════════════
# REFERENCE TABLES FOR UNIT CONVERSIONS
# ═══════════════════════════════════════════════════════════════════════════════

COMMON_SOMA_AREAS = {
    "Squid giant axon": 7.85e-1,      # 500 µm diameter → 0.785 cm²
    "L5 pyramidal": 2.64e-5,           # 29 µm diameter
    "FS interneuron": 7.07e-5,         # 15 µm diameter
    "Motoneuron": 1.13e-3,             # 60 µm diameter
    "Purkinje": 1.96e-4,               # 25 µm diameter
    "Thalamic relay": 1.96e-4,         # 25 µm diameter
    "C-fiber": 3.14e-5,                # 10 µm diameter
    "CA1 pyramidal": 1.26e-4,          # 20 µm diameter
}

# ═══════════════════════════════════════════════════════════════════════════════
# DUAL REPRESENTATION (Density + Absolute) FOR UI DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

def describe_conductance_dual(g_density_mS_cm2: float, soma_area_cm2: float) -> str:
    """
    Describe conductance in BOTH density and absolute units for user display.

    Helps users understand both the biophysical property (density) and
    the absolute value used in simulation (area-scaled).

    Parameters
    ----------
    g_density_mS_cm2 : float
        Conductance density in mS/cm²
    soma_area_cm2 : float
        Soma surface area in cm²

    Returns
    -------
    str
        Format: "56 mS/cm² [7.0e-4 mS absolute]" for display
    """
    g_absolute_mS = g_density_mS_cm2 * soma_area_cm2
    return f"{g_density_mS_cm2:.1f} mS/cm² [{g_absolute_mS:.2e} mS absolute]"

def density_to_absolute_current(i_density_uA_cm2: float, soma_area_cm2: float) -> float:
    """
    Convert current density to absolute current in nanoamperes.

    Parameters
    ----------
    i_density_uA_cm2 : float
        Current density in µA/cm²
    soma_area_cm2 : float
        Soma surface area in cm²

    Returns
    -------
    float
        Absolute current in nanoamperes (nA)

    Examples
    --------
    >>> soma_area = np.pi * (20e-4)**2  # 20 µm soma
    >>> density_to_absolute_current(50, soma_area)
    6.283  # ~6.3 nA
    """
    return float(i_density_uA_cm2 * soma_area_cm2 * 1000.0)

def describe_current_dual(i_density_uA_cm2: float, soma_area_cm2: float) -> str:
    """
    Describe stimulus current in BOTH density and absolute units.

    Parameters
    ----------
    i_density_uA_cm2 : float
        Current density in µA/cm²
    soma_area_cm2 : float
        Soma surface area in cm²

    Returns
    -------
    str
        Format: "50 µA/cm² [1.3 nA absolute]" for display
    """
    i_absolute_nA = density_to_absolute_current(i_density_uA_cm2, soma_area_cm2)
    return f"{i_density_uA_cm2:.2f} µA/cm² [{i_absolute_nA:.3f} nA absolute]"

def describe_configuration_summary(config) -> str:
    """
    Generate a full configuration summary with dual units for analytics/output.

    Parameters
    ----------
    config : FullModelConfig
        The configuration object

    Returns
    -------
    str
        Multi-line summary with both density and absolute units
    """
    soma_area = np.pi * (config.morphology.d_soma ** 2)

    lines = [
        "═════════════════════════════════════════════════════════════",
        "NEURON CONFIGURATION SUMMARY (Dual Units)",
        "═════════════════════════════════════════════════════════════",
        f"Soma diameter: {config.morphology.d_soma * 1e4:.1f} µm",
        f"Soma area: {soma_area:.2e} cm²",
        "",
        "ION CHANNELS (Density / Absolute):",
        f"  gNa: {describe_conductance_dual(config.channels.gNa_max, soma_area)}",
        f"  gK:  {describe_conductance_dual(config.channels.gK_max, soma_area)}",
        f"  gL:  {describe_conductance_dual(config.channels.gL, soma_area)}",
        f"  Cm:  {config.channels.Cm:.2f} µF/cm² [{config.channels.Cm * soma_area:.2e} µF absolute]",
        "",
        "STIMULATION:",
        f"  Type: {config.stim.stim_type}",
        f"  Iext: {describe_current_dual(config.stim.Iext, soma_area)}",
        f"  Temperature: {config.env.T_celsius}°C",
        "═════════════════════════════════════════════════════════════",
    ]
    return "\n".join(lines)

if __name__ == "__main__":
    print("Unit Conversion Examples:")
    print()
    print(f"1500 pA  → {format_current_for_display(1500)}")
    print(f"50 pA    → {format_current_for_display(50)}")
    print(f"5000000 pA → {format_current_for_display(5000000)}")
    print()
    print(f"gNa_max = 120.0 mS/cm² on L5 soma (29 µm):")
    l5_area = np.pi * (29e-4)**2
    print(f"  → {120.0 * l5_area:.3e} mS absolute")
