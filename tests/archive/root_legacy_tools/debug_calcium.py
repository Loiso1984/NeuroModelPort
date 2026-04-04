"""
Quick debug test for calcium channel activation
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset
from core.morphology import MorphologyBuilder
from core.solver import NeuronSolver

def debug_calcium():
    """Debug calcium channel activation"""
    print("🐛 Debug: Calcium Channel Activation")
    print("=" * 50)
    
    # Создаем простой тест
    cfg = FullModelConfig()
    apply_preset(cfg, "K: Thalamic Relay (Ih + ICa + Burst)")
    
    print(f"ICa enabled: {cfg.channels.enable_ICa}")
    print(f"gCa_max: {cfg.channels.gCa_max}")
    print(f"dynamic_Ca: {cfg.calcium.dynamic_Ca}")
    print(f"B_Ca: {cfg.calcium.B_Ca}")
    
    # Запускаем симуляцию с сильным стимулом
    cfg.stim.Iext = 50.0  # Очень сильный стимул
    cfg.stim.stim_type = 'const'
    
    solver = NeuronSolver(cfg)
    result = solver.run_single()
    
    print(f"V_max: {result.v_soma.max():.1f} mV")
    print(f"V_min: {result.v_soma.min():.1f} mV")
    
    if hasattr(result, 'ca_i') and result.ca_i is not None:
        soma_ca = result.ca_i[0, :]
        print(f"Ca_i range: {soma_ca.min():.2e} to {soma_ca.max():.2e} M")
        print(f"Ca_i change: {soma_ca.max() - soma_ca.min():.2e} M")
        
        # Проверяем есть ли изменение
        if soma_ca.max() > soma_ca.min() * 1.01:  # >1% изменение
            print("✅ Calcium dynamics detected!")
        else:
            print("❌ No significant calcium dynamics")
    else:
        print("❌ No calcium data in results")

if __name__ == "__main__":
    debug_calcium()
