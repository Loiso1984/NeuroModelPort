"""
Quick multi-preset physiology validation for v10.1.

This script is intentionally lightweight and non-destructive:
- runs a selected list of presets
- reports peak voltage, spike count, and firing rate
- compares soma vs dendritic_filtered routing for alpha stimulus

Use this during iterative calibration to avoid overfitting one neuron type.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from core.analysis import detect_spikes
from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver


PRESETS_TO_CHECK = [
    "B: Pyramidal L5 (Mainen 1996)",
    "C: FS Interneuron (Wang-Buzsaki)",
    "E: Cerebellar Purkinje (De Schutter)",
    "K: Thalamic Relay (Ih + ITCa + Burst)",
]


@dataclass
class Metrics:
    v_peak: float
    spikes: int
    firing_hz: float
    v_rest: float


def _extract_metrics(t: np.ndarray, v: np.ndarray) -> Metrics:
    p_idx, p_t, _ = detect_spikes(v, t, threshold=-20.0)
    sim_s = max(float(t[-1]) / 1000.0, 1e-9)
    return Metrics(
        v_peak=float(np.max(v)),
        spikes=int(len(p_idx)),
        firing_hz=float(len(p_idx) / sim_s),
        v_rest=float(v[0]),
    )


def run_validation() -> int:
    print("=" * 78)
    print("v10.1 alpha-current validation across key presets")
    print("=" * 78)
    print("Columns: mode | Vpeak(mV) | spikes | firing(Hz) | Vrest(mV)")
    print()

    for preset in PRESETS_TO_CHECK:
        cfg = FullModelConfig()
        apply_preset(cfg, preset)
        cfg.stim.stim_type = "alpha"

        print(f"[{preset}]")
        for mode in ("soma", "dendritic_filtered"):
            cfg.stim_location.location = mode
            result = NeuronSolver(cfg).run_single()
            m = _extract_metrics(result.t, result.v_soma)
            print(
                f"  {mode:19s} | {m.v_peak:8.2f} | {m.spikes:6d} | "
                f"{m.firing_hz:9.2f} | {m.v_rest:8.2f}"
            )

        # Show attenuation implied by current preset parameters
        if cfg.dendritic_filter.enabled:
            att = math.exp(
                -cfg.dendritic_filter.distance_um / max(cfg.dendritic_filter.space_constant_um, 1e-9)
            )
            print(
                f"  dendritic params: d={cfg.dendritic_filter.distance_um:.1f} µm, "
                f"lambda={cfg.dendritic_filter.space_constant_um:.1f} µm, "
                f"tau={cfg.dendritic_filter.tau_dendritic_ms:.1f} ms, attenuation={att:.3f}"
            )
        print()

    print("=" * 78)
    print("Done. Use these metrics as calibration baseline before tightening assertions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_validation())

