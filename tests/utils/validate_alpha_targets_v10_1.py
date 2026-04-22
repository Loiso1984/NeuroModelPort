"""
Physiology-oriented target check for selected presets.

This script uses broad biological ranges to flag large mismatches without
hard overfitting to one dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.analysis import detect_spikes
from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver


@dataclass
class Target:
    expected_state: str
    spikes_min: int
    spikes_max: int
    vmax_min: float
    vmax_max: float


TARGETS = {
    # Alpha pulse on dendritic-filtered input is treated as a sparse barrage proxy.
    # In this reduced model, B/E/K are expected to remain largely subthreshold
    # under this specific probe, while C retains strong fast-spiking recruitment.
    "B: Pyramidal L5 (Mainen 1996)": Target(
        expected_state="subthreshold", spikes_min=0, spikes_max=1, vmax_min=-80.0, vmax_max=5.0
    ),
    "C: FS Interneuron (Wang-Buzsaki)": Target(
        expected_state="spiking", spikes_min=8, spikes_max=30, vmax_min=0.0, vmax_max=60.0
    ),
    "E: Cerebellar Purkinje (De Schutter)": Target(
        expected_state="subthreshold", spikes_min=0, spikes_max=1, vmax_min=-80.0, vmax_max=10.0
    ),
    "K: Thalamic Relay (Ih + ITCa + Burst)": Target(
        expected_state="subthreshold", spikes_min=0, spikes_max=2, vmax_min=-85.0, vmax_max=15.0
    ),
}


def main() -> int:
    print("v10.1 alpha target validation (dendritic_filtered alpha probe)")
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
            f"({tg.expected_state}, target {tg.spikes_min}-{tg.spikes_max}), "
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

