"""
Analyze preset channel combinations and validation status
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset

def analyze_preset_channels():
    """Analyze all presets and their channel combinations"""
    print("🔍 Preset Channel Combination Analysis")
    print("=" * 80)
    
    # All preset names
    presets = [
        "FS Interneuron (Wang-Buzsaki)",
        "alpha-Motoneuron (Powers 2001)",
        "Purkinje Cell (De Schutter 1994)",
        "K: Thalamic Relay (Ih + ICa + Burst)",
        "L: Hippocampal CA1 (Theta rhythm)",
        "M: Epilepsy (v10 SCN1A mutation)",
        "N: Alzheimer's (v10 Calcium Toxicity)",
        "O: Hypoxia (v10 ATP-pump failure)"
    ]
    
    print(f"{'Preset':<35} {'Na':<3} {'K':<3} {'Ca':<3} {'HCN':<3} {'IA':<3} {'SK':<3} {'Total':<5}")
    print("-" * 80)
    
    for preset_name in presets:
        try:
            cfg = FullModelConfig()
            apply_preset(cfg, preset_name)
            
            # Count active channels
            channels = {
                'Na': cfg.channels.gNa_max > 0,
                'K': cfg.channels.gK_max > 0,
                'Ca': cfg.channels.enable_ICa and cfg.channels.gCa_max > 0,
                'HCN': cfg.channels.enable_Ih and cfg.channels.gIh_max > 0,
                'IA': cfg.channels.enable_IA and cfg.channels.gA_max > 0,
                'SK': cfg.channels.enable_SK and cfg.channels.gSK_max > 0
            }
            
            # Count total active
            total_active = sum(channels.values())
            
            # Display
            preset_short = preset_name.split('(')[0].strip()
            channel_str = f"{preset_short:<35}"
            for ch, active in channels.items():
                channel_str += f" {'✅' if active else '❌':<3}"
            channel_str += f" {total_active:<5}"
            
            print(channel_str)
            
            # Show conductance values for active channels
            if total_active > 3:  # Complex presets
                print(f"  Details: Na={cfg.channels.gNa_max:.1f}, K={cfg.channels.gK_max:.1f}", end="")
                if channels['Ca']:
                    print(f", Ca={cfg.channels.gCa_max:.2f}", end="")
                if channels['HCN']:
                    print(f", HCN={cfg.channels.gIh_max:.2f}", end="")
                if channels['IA']:
                    print(f", IA={cfg.channels.gA_max:.2f}", end="")
                if channels['SK']:
                    print(f", SK={cfg.channels.gSK_max:.1f}", end="")
                print()
            
        except Exception as e:
            print(f"{preset_name:<35} ERROR: {e}")
    
    print("\n" + "=" * 80)
    print("📊 Channel Validation Status:")
    print()
    
    # Check validation status
    validation_status = {
        'Na': '✅ Basic (Hodgkin-Huxley)',
        'K': '✅ Basic (Hodgkin-Huxley)', 
        'Ca': '✅ VALIDATED (5/5 tests)',
        'HCN': '⚠️ Partial (input resistance issues)',
        'IA': '❌ NOT VALIDATED',
        'SK': '❌ NOT VALIDATED'
    }
    
    for channel, status in validation_status.items():
        print(f"  {channel:<3}: {status}")
    
    print("\n" + "=" * 80)
    print("🔬 Physiological Plausibility Analysis:")
    print()
    
    # Analyze specific presets
    analyses = [
        ("FS Interneuron", "Expected: Na + K + IA (fast-spiking)", "Actual: Na + K + IA + Ca + SK"),
        ("Thalamic Relay", "Expected: Na + K + HCN + Ca (bursting)", "Actual: Na + K + HCN + Ca"),
        ("Purkinje", "Expected: Na + K + Ca + SK (complex spikes)", "Actual: Na + K + Ca + SK + IA"),
        ("CA1", "Expected: Na + K + HCN + IA (theta)", "Actual: Na + K + HCN + Ca"),
        ("Epilepsy", "Expected: Na + K + Ca (hyperexcitable)", "Actual: Na + K + Ca + HCN"),
        ("Alzheimer's", "Expected: Na + K + Ca + SK (adaptation)", "Actual: Na + K + Ca + HCN + SK"),
        ("Hypoxia", "Expected: Na + K + Ca (pathological)", "Actual: Na + K + Ca + HCN")
    ]
    
    for preset, expected, actual in analyses:
        print(f"  {preset}:")
        print(f"    Expected: {expected}")
        print(f"    Actual:   {actual}")
        print(f"    Status:   {'✅' if expected == actual else '⚠️ Mismatch'}")
        print()
    
    print("🎯 Key Issues Identified:")
    print("  1. FS Interneuron: Has Ca²⁺ + SK (may not be necessary)")
    print("  2. CA1: Ca²⁺ instead of IA (theta rhythm typically uses IA)")
    print("  3. Epilepsy: HCN present (may counteract hyperexcitability)")
    print("  4. Alzheimer's: HCN present (not typical for AD pathology)")
    print("  5. Hypoxia: HCN present (may be affected by ATP depletion)")

if __name__ == "__main__":
    analyze_preset_channels()
