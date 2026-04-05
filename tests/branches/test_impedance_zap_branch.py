from __future__ import annotations

import pytest
pytest.importorskip("pydantic")

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.analysis import compute_membrane_impedance, reconstruct_stimulus_trace
from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver


def _run_impedance_case(preset_name: str) -> dict:
    cfg = FullModelConfig()
    apply_preset(cfg, preset_name)
    cfg.stim.stim_type = "zap"
    cfg.stim.pulse_start = 50.0
    cfg.stim.pulse_dur = 900.0
    cfg.stim.t_sim = 1100.0
    cfg.stim.dt_eval = 0.25
    cfg.stim.zap_f0_hz = 0.5
    cfg.stim.zap_f1_hz = 40.0
    cfg.stim.Iext = max(2.0, float(cfg.stim.Iext) * 0.5)

    res = NeuronSolver(cfg).run_single()
    i_stim = reconstruct_stimulus_trace(res)
    imp = compute_membrane_impedance(res.t, res.v_soma, i_stim, fmin_hz=0.5, fmax_hz=80.0)
    return {
        "valid": bool(imp.get("valid", False)),
        "f_res": float(imp.get("f_res_hz", np.nan)),
        "z_res": float(imp.get("z_res_kohm_cm2", np.nan)),
    }


def test_impedance_zap_trn_and_cholinergic_are_computable():
    rows = {
        "P": _run_impedance_case("P: Thalamic Reticular Nucleus (TRN Spindles)"),
        "R": _run_impedance_case("R: Cholinergic Neuromodulation (ACh)"),
        "K": _run_impedance_case("K: Thalamic Relay (Ih + IT + Burst)"),
    }

    for key, row in rows.items():
        assert row["valid"], f"{key}: impedance estimate invalid"
        assert np.isfinite(row["f_res"]), f"{key}: f_res is not finite"
        assert 0.1 <= row["f_res"] <= 80.0, f"{key}: f_res out of range ({row['f_res']})"
        assert row["z_res"] > 0.0, f"{key}: z_res must be positive"


def _run_as_script() -> int:
    try:
        test_impedance_zap_trn_and_cholinergic_are_computable()
        print("[PASS] test_impedance_zap_trn_and_cholinergic_are_computable")
        return 0
    except Exception as exc:
        print(f"[FAIL] test_impedance_zap_trn_and_cholinergic_are_computable: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(_run_as_script())
