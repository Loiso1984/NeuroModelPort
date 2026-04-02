import math

import numpy as np

from core.models import FullModelConfig
from core.presets import apply_preset, get_preset_names


MAMMALIAN_KEY_PRESETS = [
    "B: Pyramidal L5 (Mainen 1996)",
    "C: FS Interneuron (Wang-Buzsaki)",
    "D: alpha-Motoneuron (Powers 2001)",
    "E: Cerebellar Purkinje (De Schutter)",
    "K: Thalamic Relay (Ih + ICa + Burst)",
    "L: Hippocampal CA1 (Theta rhythm)",
    "J: C-Fiber (Pain / Unmyelinated)",
    "I: In Vitro Slice (Mammalian 23°C)",
]


def _compute_attenuation(distance_um: float, space_constant_um: float) -> float:
    return math.exp(-distance_um / space_constant_um)


def _compute_cutoff_hz(tau_ms: float) -> float:
    # f_c = 1 / (2 * pi * tau)
    return 1.0 / (2.0 * math.pi * (tau_ms / 1000.0))


def test_dendritic_filter_physical_ranges():
    """
    Ensure dendritic filter parameters for mammalian presets are within
    physiologically reasonable bounds (distance, lambda, tau).
    """
    cfg = FullModelConfig()

    for preset_name in MAMMALIAN_KEY_PRESETS:
        apply_preset(cfg, preset_name)

        # Squid and non-dendritic cases are handled elsewhere; here we expect
        # dendritic filtering to be enabled for mammalian neurons.
        assert cfg.dendritic_filter.enabled, f"{preset_name}: dendritic_filter should be enabled"

        d = cfg.dendritic_filter.distance_um
        lam = cfg.dendritic_filter.space_constant_um
        tau = cfg.dendritic_filter.tau_dendritic_ms

        # Distance soma→synapse: 50–300 µm typical range
        assert 10.0 < d < 500.0, f"{preset_name}: distance_um={d} outside [10, 500] µm"

        # Space constant λ: 50–300 µm typical range
        assert 50.0 <= lam <= 300.0, f"{preset_name}: space_constant_um={lam} outside [50, 300] µm"

        # Ratio distance/λ should be O(1) to avoid extreme attenuation
        ratio = d / lam
        assert 0.2 <= ratio <= 3.0, f"{preset_name}: distance/λ={ratio:.2f} unrealistic"

        # Tau_dendritic: 2–50 ms
        assert 2.0 <= tau <= 50.0, f"{preset_name}: tau_dendritic_ms={tau} outside [2, 50] ms"


def test_dendritic_filter_transfer_characteristics():
    """
    Check that attenuation and cutoff frequency implied by presets stay
    within literature-inspired physiological bands.
    """
    cfg = FullModelConfig()

    for preset_name in MAMMALIAN_KEY_PRESETS:
        apply_preset(cfg, preset_name)

        if not cfg.dendritic_filter.enabled:
            continue

        d = cfg.dendritic_filter.distance_um
        lam = cfg.dendritic_filter.space_constant_um
        tau = cfg.dendritic_filter.tau_dendritic_ms

        attenuation = _compute_attenuation(d, lam)
        f_c = _compute_cutoff_hz(tau)

        # For dendrites 50–300 µm from soma, exp(-d/λ) should not be extreme
        # (avoid full clamp or full block). 10–90% passing band is acceptable.
        assert 0.1 < attenuation < 0.9, (
            f"{preset_name}: attenuation={attenuation:.3f} outside (0.1, 0.9)"
        )

        # Corner frequency for dendritic low-pass should land in ~5–100 Hz band
        # (typical for L5 and related neurons).
        assert 2.0 < f_c < 100.0, f"{preset_name}: cutoff={f_c:.1f} Hz outside (2, 100) Hz"

