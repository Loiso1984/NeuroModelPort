"""
Debug B_Ca values in presets
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset

def debug_b_ca():
    """Debug B_Ca values in calcium presets"""
    print("🐛 Debug: B_Ca Values in Presets")
    print("=" * 50)
    
    presets = [
        "K: Thalamic Relay (Ih + ICa + Burst)",
        "L: Hippocampal CA1 (Theta rhythm)", 
        "M: Epilepsy (v10 SCN1A mutation)",
        "N: Alzheimer's (v10 Calcium Toxicity)",
        "O: Hypoxia (v10 ATP-pump failure)"
    ]
    
    for preset_name in presets:
        cfg = FullModelConfig()
        apply_preset(cfg, preset_name)
        
        print(f"{preset_name}:")
        print(f"  B_Ca: {cfg.calcium.B_Ca}")
        print(f"  gCa_max: {cfg.channels.gCa_max}")
        print(f"  enable_ICa: {cfg.channels.enable_ICa}")
        print()

if __name__ == "__main__":
    debug_b_ca()
