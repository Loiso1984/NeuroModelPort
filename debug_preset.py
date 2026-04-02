"""debug_preset.py - Check what presets are actually setting"""

from core.models import FullModelConfig
from core.presets import apply_preset

cfg = FullModelConfig()
print("BEFORE apply_preset:")
print(f"  stim_location: {cfg.stim_location.location}")
print(f"  stim_type: {cfg.stim.stim_type}")
print(f"  Iext: {cfg.stim.Iext}")

apply_preset(cfg, 'B: Pyramidal L5 (Mainen 1996)')

print("\nAFTER apply_preset (B: Pyramidal L5):")
print(f"  stim_location: {cfg.stim_location.location}")
print(f"  stim_type: {cfg.stim.stim_type}")
print(f"  Iext: {cfg.stim.Iext}")
print(f"  pulse_start: {cfg.stim.pulse_start}")
print(f"  t_sim: {cfg.stim.t_sim}")

# Also check what the dendritic filter settings are
print(f"\n  dendritic_filter.enabled: {cfg.dendritic_filter.enabled}")
print(f"  dendritic_filter.distance_um: {cfg.dendritic_filter.distance_um}")
print(f"  dendritic_filter.space_constant_um: {cfg.dendritic_filter.space_constant_um}")
print(f"  dendritic_filter.tau_dendritic_ms: {cfg.dendritic_filter.tau_dendritic_ms}")
