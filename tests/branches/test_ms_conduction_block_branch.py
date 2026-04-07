"""Branch validation for preset F demyelination conduction attenuation.

Acceptance goal:
- Preset F should exhibit a strong demyelination signature in default conditions:
  either (a) clear propagation attenuation with ratio <= 0.30, or
  (b) conduction block (no suprathreshold soma spike).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver


def _run(preset: str) -> dict:
    cfg = FullModelConfig()
    apply_preset(cfg, preset)
    cfg.stim.t_sim = 320.0
    cfg.stim.dt_eval = 0.15
    cfg.stim.jacobian_mode = "sparse_fd"

    res = NeuronSolver(cfg).run_single()
    if cfg.morphology.N_trunk > 0:
        j = min(1 + cfg.morphology.N_ais + cfg.morphology.N_trunk - 1, res.n_comp - 1)
    elif cfg.morphology.N_ais > 0:
        j = min(cfg.morphology.N_ais, res.n_comp - 1)
    else:
        j = min(1, res.n_comp - 1)

    soma_peak = float(np.max(res.v_soma))
    junction_peak = float(np.max(res.v_all[j, :]))
    ratio = float(junction_peak / max(soma_peak, 1e-9))

    return {
        "stable": bool(np.all(np.isfinite(res.v_soma))),
        "soma_peak": soma_peak,
        "junction_peak": junction_peak,
        "ratio": ratio,
    }


def test_ms_preset_has_strong_conduction_attenuation_vs_control():
    d = _run("D: alpha-Motoneuron (Powers 2001)")
    f = _run("F: Multiple Sclerosis (Demyelination)")

    assert d["stable"] and f["stable"], "simulation produced non-finite traces"
    assert d["soma_peak"] > 0.0, f"control soma peak too low: {d['soma_peak']:.2f}"
    # Demyelination acceptance (block-or-attenuation mode):
    # - If soma does not spike above 0 mV, treat as conduction block signature.
    # - Otherwise require strong attenuation ratio.
    if f["soma_peak"] <= 0.0:
        assert f["junction_peak"] <= 0.0, (
            "F preset entered block mode at soma but not at junction: "
            f"soma_peak={f['soma_peak']:.2f}, junction_peak={f['junction_peak']:.2f}"
        )
    else:
        assert f["ratio"] <= 0.30, (
            "F preset should show strong propagation attenuation "
            f"(ratio={f['ratio']:.3f}, control={d['ratio']:.3f})"
        )
