"""
Static Parameter Audit for NeuroModelPort Presets

Performs biophysical validation of preset parameters without running simulations.
Much faster than full simulation validation - useful for quick auditing.

Usage:
    python tests/branches/preset_parameter_audit.py
"""

import sys
sys.path.insert(0, 'c:\\NeuroModelPort')

from core.models import FullModelConfig
from core.presets import apply_preset, get_preset_names


# Literature reference data for parameter validation
LITERATURE_PARAMS = {
    "A": {
        "name": "Squid Giant Axon (HH 1952)",
        "citation": "Hodgkin & Huxley 1952",
        "expected": {
            "gNa_max": (100.0, 140.0),  # Classic HH: 120 mS/cm²
            "gK_max": (30.0, 40.0),     # Classic HH: 36 mS/cm²
            "gL": (0.2, 0.4),           # Classic HH: 0.3 mS/cm²
            "ENa": (45.0, 55.0),        # ~+50 mV
            "EK": (-80.0, -75.0),       # ~-77 mV
            "EL": (-60.0, -50.0),       # ~-54 mV
            "Cm": (0.9, 1.1),           # 1.0 µF/cm²
            "T": (5.0, 8.0),            # 6.3°C
        }
    },
    "B": {
        "name": "Pyramidal L5 (Mainen 1996)",
        "citation": "Mainen et al. 1996, Nature 382:363-366",
        "expected": {
            "gNa_max": (30.0, 80.0),    # Somatic density varies
            "gK_max": (4.0, 10.0),      # Delayed rectifier
            "gL": (0.015, 0.04),        # Low leak for high Rin
            "ENa": (45.0, 55.0),
            "EK": (-95.0, -85.0),       # High K driving force
            "EL": (-75.0, -65.0),
            "Cm": (0.7, 1.0),
            "T": (20.0, 24.0),          # Slice temp
            "d_soma_um": (15.0, 25.0),  # Mainen: ~20µm
        }
    },
    "C": {
        "name": "FS Interneuron (Wang-Buzsaki)",
        "citation": "Wang & Buzsaki 1996, J Neurosci 16:6402",
        "expected": {
            "gNa_max": (80.0, 150.0),   # High for fast spiking
            "gK_max": (30.0, 60.0),     # Strong K for repolarization
            "gL": (0.08, 0.15),
            "ENa": (50.0, 60.0),
            "EK": (-95.0, -85.0),
            "EL": (-70.0, -60.0),
            "Cm": (0.9, 1.1),
            "T": (20.0, 24.0),
            "d_soma_um": (10.0, 20.0),
        }
    },
    "D": {
        "name": "Alpha-Motoneuron (Powers 2001)",
        "citation": "Powers et al. 2001, J Neurophysiol 85:1109",
        "expected": {
            "gNa_max": (80.0, 120.0),
            "gK_max": (20.0, 40.0),
            "gL": (0.08, 0.15),
            "ENa": (45.0, 55.0),
            "EK": (-80.0, -75.0),
            "EL": (-65.0, -55.0),
            "Cm": (1.0, 2.0),            # Higher for large cell
            "T": (35.0, 39.0),
            "d_soma_um": (50.0, 80.0),  # Large soma 60-70µm
        }
    },
    "E": {
        "name": "Cerebellar Purkinje (De Schutter)",
        "citation": "De Schutter & Bower 1994",
        "expected": {
            "gNa_max": (50.0, 100.0),
            "gK_max": (15.0, 30.0),
            "gL": (0.03, 0.08),
            "ENa": (40.0, 50.0),
            "EK": (-90.0, -80.0),
            "EL": (-75.0, -60.0),
            "Cm": (0.9, 1.1),
            "T": (35.0, 39.0),
            "d_soma_um": (20.0, 30.0),
        }
    },
    "K": {
        "name": "Thalamic Relay (McCormick)",
        "citation": "McCormick & Huguenard 1992",
        "expected": {
            "gNa_max": (70.0, 110.0),
            "gK_max": (10.0, 18.0),
            "gL": (0.03, 0.08),
            "gIh_max": (0.01, 0.05),      # Ih present
            "gTCa_max": (1.5, 2.5),       # T-type Ca
            "ENa": (45.0, 55.0),
            "EK": (-95.0, -85.0),
            "EL": (-75.0, -65.0),
            "Cm": (0.9, 1.1),
            "T": (35.0, 39.0),
            "d_soma_um": (20.0, 30.0),
        }
    },
    "L": {
        "name": "Hippocampal CA1 Pyramidal (Magee)",
        "citation": "Magee 1998, J Neurosci 18:7613",
        "expected": {
            "gNa_max": (40.0, 70.0),
            "gK_max": (6.0, 12.0),
            "gL": (0.03, 0.08),
            "gIh_max": (0.01, 0.03),      # Ih for theta resonance
            "gA_max": (0.2, 0.5),         # A-type K
            "ENa": (45.0, 55.0),
            "EK": (-90.0, -80.0),
            "EL": (-70.0, -60.0),
            "Cm": (0.9, 1.1),
            "T": (35.0, 39.0),
            "d_soma_um": (15.0, 25.0),
        }
    },
    "Q": {
        "name": "Striatal SPN (Nisenbaum)",
        "citation": "Nisenbaum & Wilson 1995",
        "expected": {
            "gNa_max": (60.0, 100.0),
            "gK_max": (6.0, 12.0),
            "gL": (0.03, 0.06),
            "gA_max": (0.5, 1.0),          # Strong I_A for delay
            "gIh_max": (0.005, 0.02),
            "ENa": (45.0, 55.0),
            "EK": (-95.0, -85.0),
            "EL": (-85.0, -75.0),          # Very hyperpolarized
            "Cm": (0.9, 1.1),
            "T": (35.0, 39.0),
            "d_soma_um": (12.0, 20.0),
        }
    },
}


def extract_preset_params(cfg: FullModelConfig) -> dict:
    """Extract key biophysical parameters from config."""
    return {
        "gNa_max": cfg.channels.gNa_max,
        "gK_max": cfg.channels.gK_max,
        "gL": cfg.channels.gL,
        "ENa": cfg.channels.ENa,
        "EK": cfg.channels.EK,
        "EL": cfg.channels.EL,
        "Cm": cfg.channels.Cm,
        "T": cfg.env.T_celsius,
        "d_soma_um": cfg.morphology.d_soma * 1e4,  # Convert cm to µm
        "gIh_max": getattr(cfg.channels, 'gIh_max', 0.0),
        "gA_max": getattr(cfg.channels, 'gA_max', 0.0),
        "gTCa_max": getattr(cfg.channels, 'gTCa_max', 0.0),
        "gCa_max": getattr(cfg.channels, 'gCa_max', 0.0),
        "gSK_max": getattr(cfg.channels, 'gSK_max', 0.0),
        "gIM_max": getattr(cfg.channels, 'gIM_max', 0.0),
        "gNaR_max": getattr(cfg.channels, 'gNaR_max', 0.0),
    }


def check_param(name: str, value: float, expected_range: tuple, unit: str = "") -> tuple:
    """
    Check if parameter is within expected range.
    Returns (status, message).
    """
    low, high = expected_range
    unit_str = f" {unit}" if unit else ""
    
    if value < low:
        return ("⚠", f"{name} = {value:.3g}{unit_str} (LOW, expected {low}-{high})")
    elif value > high:
        return ("⚠", f"{name} = {value:.3g}{unit_str} (HIGH, expected {low}-{high})")
    else:
        return ("✓", f"{name} = {value:.3g}{unit_str}")


def analyze_preset(preset_name: str) -> dict:
    """Analyze a single preset against literature."""
    code = preset_name.split(":")[0].strip()
    
    try:
        cfg = FullModelConfig()
        apply_preset(cfg, preset_name)
        params = extract_preset_params(cfg)
        
        result = {
            "code": code,
            "name": preset_name,
            "status": "OK",
            "checks": [],
            "warnings": [],
            "info": []
        }
        
        # Check against literature if available
        if code in LITERATURE_PARAMS:
            ref = LITERATURE_PARAMS[code]
            result["citation"] = ref["citation"]
            
            for param_name, expected_range in ref["expected"].items():
                if param_name in params:
                    status, msg = check_param(param_name, params[param_name], expected_range)
                    if status == "✓":
                        result["checks"].append(msg)
                    else:
                        result["warnings"].append(msg)
        
        # Additional biophysical sanity checks
        
        # 1. Conductance ratios
        if params["gK_max"] > 0:
            na_k_ratio = params["gNa_max"] / params["gK_max"]
            if na_k_ratio < 2.0:
                result["warnings"].append(f"gNa/gK = {na_k_ratio:.1f} (LOW - may not spike)")
            elif na_k_ratio > 20.0:
                result["warnings"].append(f"gNa/gK = {na_k_ratio:.1f} (HIGH - check if realistic)")
            else:
                result["checks"].append(f"gNa/gK = {na_k_ratio:.1f} (OK)")
        
        if params["gL"] > 0:
            na_l_ratio = params["gNa_max"] / params["gL"]
            if na_l_ratio < 50:
                result["warnings"].append(f"gNa/gL = {na_l_ratio:.0f} (LOW - excessive leak)")
            else:
                result["info"].append(f"gNa/gL = {na_l_ratio:.0f}")
        
        # 2. Reversal potential sanity
        if params["ENa"] <= params["EK"]:
            result["warnings"].append(f"ENa ({params['ENa']}) ≤ EK ({params['EK']}) - IMPOSIBLE!")
        
        if not (-150 < params["EK"] < 0):
            result["warnings"].append(f"EK = {params['EK']} mV (suspicious)")
        
        if not (0 < params["ENa"] < 150):
            result["warnings"].append(f"ENa = {params['ENa']} mV (suspicious)")
        
        # 3. Temperature check
        if params["T"] < 0 or params["T"] > 50:
            result["warnings"].append(f"T = {params['T']}°C (physiologically implausible)")
        
        # 4. Soma size check
        if not (5 < params["d_soma_um"] < 100):
            result["warnings"].append(f"d_soma = {params['d_soma_um']:.1f} µm (outside typical range 5-100)")
        
        # Count warnings
        if len(result["warnings"]) > 0:
            result["status"] = "WARN"
        
        return result
        
    except Exception as e:
        return {
            "code": code,
            "name": preset_name,
            "status": "ERROR",
            "error": str(e)
        }


def print_analysis(result: dict):
    """Print formatted analysis results."""
    code = result["code"]
    status = result["status"]
    
    # Status emoji
    emoji = {"OK": "✓", "WARN": "⚠", "ERROR": "✗"}.get(status, "?")
    
    print(f"\n[{code}] {result['name']}")
    if "citation" in result:
        print(f"    Ref: {result['citation']}")
    print(f"    Status: {emoji} {status}")
    
    if "error" in result:
        print(f"    ERROR: {result['error']}")
        return
    
    # Print checks (passed)
    for check in result.get("checks", []):
        print(f"    ✓ {check}")
    
    # Print info
    for info in result.get("info", []):
        print(f"    ℹ {info}")
    
    # Print warnings
    for warning in result.get("warnings", []):
        print(f"    ⚠ {warning}")


def main():
    print("=" * 80)
    print("BIOPHYSICAL PARAMETER AUDIT")
    print("=" * 80)
    print("Validating preset parameters against published literature values")
    print("-" * 80)
    
    preset_names = get_preset_names()
    
    results = []
    for preset_name in preset_names:
        result = analyze_preset(preset_name)
        results.append(result)
        print_analysis(result)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    ok_count = sum(1 for r in results if r["status"] == "OK")
    warn_count = sum(1 for r in results if r["status"] == "WARN")
    error_count = sum(1 for r in results if r["status"] == "ERROR")
    
    print(f"Total presets analyzed: {len(results)}")
    print(f"  ✓ OK:     {ok_count}")
    print(f"  ⚠ WARN:   {warn_count}")
    print(f"  ✗ ERROR:  {error_count}")
    
    # Critical issues
    critical = []
    for r in results:
        if "warnings" in r:
            for w in r["warnings"]:
                if "IMPOSIBLE" in w or "implausible" in w:
                    critical.append(f"[{r['code']}] {w}")
    
    if critical:
        print("\n" + "=" * 80)
        print("CRITICAL PHYSICAL IMPOSSIBILITIES")
        print("=" * 80)
        for c in critical:
            print(f"  ✗ {c}")
    
    # Literature coverage
    analyzed = sum(1 for r in results if "citation" in r)
    print(f"\nLiterature validation coverage: {analyzed}/{len(results)} presets")
    
    return results


if __name__ == "__main__":
    results = main()
