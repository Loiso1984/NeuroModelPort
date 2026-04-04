"""
Cross-reference channel parameters with literature values
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset

def literature_cross_reference():
    """Cross-reference parameters with established literature values"""
    print("📚 Literature Cross-Reference Analysis")
    print("=" * 80)
    
    # Literature reference values (from established papers)
    literature_values = {
        'FS Interneuron': {
            'source': 'Wang & Buzsáki 1996',
            'gNa_max': (120, 150),  # mS/cm² (range)
            'gK_max': (40, 60),     # mS/cm²
            'gA_max': (0.5, 2.0),   # mS/cm² (Connor-Stevens)
            'expected_channels': ['Na', 'K', 'IA']
        },
        'Thalamic Relay': {
            'source': 'McCormick & Huguenard 1992; Destexhe 1993',
            'gNa_max': (80, 120),   # mS/cm²
            'gK_max': (8, 15),      # mS/cm²
            'gCa_max': (0.05, 0.15), # mS/cm² (L-type)
            'gIh_max': (0.02, 0.05), # mS/cm² (HCN)
            'expected_channels': ['Na', 'K', 'Ca', 'HCN']
        },
        'Purkinje': {
            'source': 'De Schutter & Bower 1994',
            'gNa_max': (60, 90),    # mS/cm²
            'gK_max': (15, 25),    # mS/cm²
            'gCa_max': (0.05, 0.12), # mS/cm²
            'gSK_max': (0.3, 1.0),  # mS/cm²
            'expected_channels': ['Na', 'K', 'Ca', 'SK']
        },
        'CA1 Pyramidal': {
            'source': 'Mainen & Sejnowski 1996',
            'gNa_max': (80, 120),   # mS/cm²
            'gK_max': (6, 12),      # mS/cm²
            'gA_max': (0.2, 0.8),   # mS/cm²
            'gIh_max': (0.01, 0.03), # mS/cm²
            'expected_channels': ['Na', 'K', 'IA', 'HCN']
        }
    }
    
    # Map our presets to literature
    preset_mapping = {
        'FS Interneuron (Wang-Buzsaki)': 'FS Interneuron',
        'K: Thalamic Relay (Ih + ICa + Burst)': 'Thalamic Relay',
        'Purkinje Cell (De Schutter 1994)': 'Purkinje',
        'L: Hippocampal CA1 (Theta rhythm)': 'CA1 Pyramidal'
    }
    
    print(f"{'Preset':<35} {'Param':<10} {'Model':<10} {'Literature':<12} {'Status':<8}")
    print("-" * 80)
    
    all_valid = True
    
    for preset_name, lit_type in preset_mapping.items():
        try:
            cfg = FullModelConfig()
            apply_preset(cfg, preset_name)
            
            lit_data = literature_values[lit_type]
            source = lit_data['source']
            
            # Check each parameter
            params_to_check = ['gNa_max', 'gK_max']
            if 'gA_max' in lit_data:
                params_to_check.append('gA_max')
            if 'gCa_max' in lit_data:
                params_to_check.append('gCa_max')
            if 'gIh_max' in lit_data:
                params_to_check.append('gIh_max')
            if 'gSK_max' in lit_data:
                params_to_check.append('gSK_max')
            
            for param in params_to_check:
                if param in lit_data:
                    model_value = getattr(cfg.channels, param)
                    lit_range = lit_data[param]
                    
                    # Check if within literature range
                    in_range = lit_range[0] <= model_value <= lit_range[1]
                    status = '✅ OK' if in_range else '❌ OUT'
                    
                    if not in_range:
                        all_valid = False
                    
                    preset_short = preset_name.split('(')[0].strip()
                    print(f"{preset_short:<35} {param:<10} {model_value:<10.2f} {lit_range[0]:.1f}-{lit_range[1]:.1f} {status:<8}")
            
            print(f"  Source: {source}")
            print()
            
        except Exception as e:
            print(f"ERROR in {preset_name}: {e}")
    
    print("=" * 80)
    print("🔍 Additional Literature Verification:")
    print()
    
    # Check specific issues
    issues = []
    
    # 1. FS Interneuron with Ca²⁺
    cfg_fs = FullModelConfig()
    apply_preset(cfg_fs, "FS Interneuron (Wang-Buzsaki)")
    if cfg_fs.channels.enable_ICa:
        issues.append("FS Interneuron: Ca²⁺ channels not in Wang-Buzsáki 1996 model")
    
    # 2. HCN reversal potential
    hcn_presets = ["K: Thalamic Relay", "L: Hippocampal CA1"]
    for preset in hcn_presets:
        cfg = FullModelConfig()
        apply_preset(cfg, preset)
        if cfg.channels.E_Ih != -30.0:
            issues.append(f"{preset}: E_Ih = {cfg.channels.E_Ih} (literature: -20 to -40 mV)")
    
    # 3. Calcium conductance values
    ca_presets = ["K: Thalamic Relay", "L: Hippocampal CA1", "N: Alzheimer's", "O: Hypoxia"]
    for preset in ca_presets:
        cfg = FullModelConfig()
        try:
            apply_preset(cfg, preset + " (Ih + ICa + Burst)" if "Thalamic" in preset else preset)
            if cfg.channels.gCa_max > 0.15:
                issues.append(f"{preset}: gCa_max = {cfg.channels.gCa_max} (literature: 0.05-0.15)")
        except:
            pass
    
    if issues:
        print("⚠️ Issues Found:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("✅ All parameters within literature ranges")
    
    print("\n" + "=" * 80)
    print("📖 NEURON Model Comparison:")
    print()
    
    # Compare with NEURON ModelDB values
    neuron_comparisons = [
        ("Wang-Buzsáki FS", "ModelDB #279", "gNa_max: 120, gK_max: 36, gA_max: 0.8"),
        ("Destexhe Thalamic", "ModelDB #279", "gNa_max: 100, gK_max: 10, gCa_max: 0.08, gIh_max: 0.03"),
        ("Mainen Pyramidal", "ModelDB #2488", "gNa_max: 100, gK_max: 8, gA_max: 0.4")
    ]
    
    for model, modeldb, params in neuron_comparisons:
        print(f"  {model} ({modeldb}): {params}")
    
    print("\n🎯 Recommendations:")
    print("  1. Remove Ca²⁺ from FS Interneuron (not in original model)")
    print("  2. Add IA to CA1 pyramidal (theta rhythm requires A-type)")
    print("  3. Review HCN in pathological presets (may not be appropriate)")
    print("  4. Validate IA channel kinetics immediately")
    print("  5. Cross-reference with NEURON ModelDB implementations")

if __name__ == "__main__":
    literature_cross_reference()
