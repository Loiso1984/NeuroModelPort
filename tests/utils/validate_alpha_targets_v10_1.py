"""
Physiology-oriented target check for selected presets.

This script uses broad biological ranges to flag large mismatches without
hard overfitting to one dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.analysis import detect_spikes
from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver


@dataclass
class Target:
    spikes_min: int
    spikes_max: int
    vmax_min: float
    vmax_max: float


TARGETS = {
    # Broad target bands for alpha-driven responses in current reduced model.
    "B: Pyramidal L5 (Mainen 1996)": Target(spikes_min=3, spikes_max=8, vmax_min=20.0, vmax_max=55.0),
    "C: FS Interneuron (Wang-Buzsaki)": Target(spikes_min=1, spikes_max=8, vmax_min=0.0, vmax_max=55.0),
    "E: Cerebellar Purkinje (De Schutter)": Target(spikes_min=1, spikes_max=8, vmax_min=10.0, vmax_max=55.0),
    "K: Thalamic Relay (Ih + ICa + Burst)": Target(spikes_min=3, spikes_max=10, vmax_min=10.0, vmax_max=50.0),
}


def main() -> int:
    print("v10.1 alpha target validation (dendritic_filtered defaults)")
    print("-" * 72)
    failed = 0
    for preset, tg in TARGETS.items():
        cfg = FullModelConfig()
        apply_preset(cfg, preset)
        cfg.stim.stim_type = "alpha"
        cfg.stim_location.location = "dendritic_filtered"
        res = NeuronSolver(cfg).run_single()

        spikes = len(detect_spikes(res.v_soma, res.t, threshold=-20.0)[0])
        vmax = float(res.v_soma.max())

        ok_spikes = tg.spikes_min <= spikes <= tg.spikes_max
        ok_vmax = tg.vmax_min <= vmax <= tg.vmax_max
        ok = ok_spikes and ok_vmax
        status = "OK" if ok else "FAIL"
        print(
            f"{status:4s} | {preset:44s} | spikes={spikes:2d} "
            f"(target {tg.spikes_min}-{tg.spikes_max}), "
            f"vmax={vmax:6.2f} mV (target {tg.vmax_min:.0f}-{tg.vmax_max:.0f})"
        )
        if not ok:
            failed += 1
    print("-" * 72)
    if failed:
        print(f"FAILED presets: {failed}")
        return 1
    print("All selected presets within broad physiological target bands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

