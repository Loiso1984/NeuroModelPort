"""
Quick check of B_Ca values
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset

def check_b_ca():
    """Check B_Ca values"""
    print("🔍 B_Ca Value Check")
    print("=" * 40)
    
    # Default config
    cfg_default = FullModelConfig()
    print(f"Default B_Ca: {cfg_default.calcium.B_Ca}")
    
    # Preset
    cfg = FullModelConfig()
    apply_preset(cfg, "K: Thalamic Relay (Ih + ICa + Burst)")
    print(f"Preset B_Ca: {cfg.calcium.B_Ca}")
    
    # Check if they're the same object
    print(f"Same object? {cfg is cfg_default}")

if __name__ == "__main__":
    check_b_ca()
