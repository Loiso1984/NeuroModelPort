from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np

import tests.utils.run_impedance_zap_report as rpt


def _inject_fake_pydantic(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "pydantic", types.ModuleType("pydantic"))


def _inject_fake_core_runtime(monkeypatch, *, f_res_hz: float, z_res: float = 3.0, valid: bool = True) -> None:
    fake_analysis = types.ModuleType("core.analysis")
    fake_models = types.ModuleType("core.models")
    fake_presets = types.ModuleType("core.presets")
    fake_solver = types.ModuleType("core.solver")

    def compute_membrane_impedance(t, v_soma, i_stim, fmin_hz, fmax_hz):
        return {
            "valid": valid,
            "f_res_hz": float(f_res_hz),
            "z_res_kohm_cm2": float(z_res),
        }

    def reconstruct_stimulus_trace(res):
        return np.ones_like(res.t)

    class _Stim:
        stim_type = "step"
        pulse_start = 0.0
        pulse_dur = 0.0
        t_sim = 0.0
        dt_eval = 0.25
        zap_f0_hz = 0.5
        zap_f1_hz = 40.0
        Iext = 4.0

    class FullModelConfig:
        def __init__(self):
            self.stim = _Stim()

    def apply_preset(cfg, preset_name):
        return None

    class _Res:
        t = np.linspace(0.0, 1000.0, 16)
        v_soma = np.linspace(-70.0, -40.0, 16)

    class NeuronSolver:
        def __init__(self, cfg):
            self.cfg = cfg

        def run_single(self):
            return _Res()

    fake_analysis.compute_membrane_impedance = compute_membrane_impedance
    fake_analysis.reconstruct_stimulus_trace = reconstruct_stimulus_trace
    fake_models.FullModelConfig = FullModelConfig
    fake_presets.apply_preset = apply_preset
    fake_solver.NeuronSolver = NeuronSolver

    monkeypatch.setitem(sys.modules, "core.analysis", fake_analysis)
    monkeypatch.setitem(sys.modules, "core.models", fake_models)
    monkeypatch.setitem(sys.modules, "core.presets", fake_presets)
    monkeypatch.setitem(sys.modules, "core.solver", fake_solver)


def test_run_case_uses_intersection_range_for_guards(monkeypatch):
    _inject_fake_core_runtime(monkeypatch, f_res_hz=25.0, z_res=2.0, valid=True)

    row = rpt.run_case(
        {"id": "X", "preset": "X", "i_scale": 0.5, "f_res_range": (0.5, 40.0)},
        fmin_hz=5.0,
        fmax_hz=20.0,
    )

    assert row["f_res_expected_range_hz"] == [5.0, 20.0]
    assert row["guard_ok"] is False
    assert row["guard_reasons"] == ["f_res_out_of_expected_range"]


def test_run_case_marks_empty_expected_range(monkeypatch):
    _inject_fake_core_runtime(monkeypatch, f_res_hz=25.0, z_res=2.0, valid=True)

    row = rpt.run_case(
        {"id": "X", "preset": "X", "i_scale": 0.5, "f_res_range": (0.5, 40.0)},
        fmin_hz=50.0,
        fmax_hz=80.0,
    )

    assert row["f_res_expected_range_hz"] == [50.0, 40.0]
    assert row["guard_ok"] is False
    assert row["guard_reasons"] == ["empty_expected_range"]


def test_impedance_report_rejects_invalid_frequency_bounds(monkeypatch, tmp_path):
    _inject_fake_pydantic(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_impedance_zap_report.py", "--fmin", "5.0", "--fmax", "5.0"])

    rc = rpt.main()

    assert rc == 2
    assert not Path("_test_results/impedance_zap_report.json").exists()


def test_impedance_report_strict_mode_fails_on_guard(monkeypatch, tmp_path):
    _inject_fake_pydantic(monkeypatch)
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        rpt,
        "CASES",
        [
            {"id": "ok", "preset": "ok", "f_res_range": (0.5, 40.0)},
            {"id": "bad", "preset": "bad", "f_res_range": (0.5, 40.0)},
        ],
    )

    def fake_run_case(case: dict, *, fmin_hz: float, fmax_hz: float) -> dict:
        assert fmin_hz == 0.5
        assert fmax_hz == 80.0
        return {
            "id": case["id"],
            "preset": case["preset"],
            "valid": True,
            "guard_ok": case["id"] != "bad",
            "guard_reasons": [] if case["id"] != "bad" else ["f_res_out_of_expected_range"],
            "f_res_hz": 10.0,
            "z_res_kohm_cm2": 5.0,
            "f_res_expected_range_hz": [0.5, 40.0],
            "v_peak_mV": -40.0,
            "v_min_mV": -70.0,
        }

    monkeypatch.setattr(rpt, "run_case", fake_run_case)
    monkeypatch.setattr(sys, "argv", ["run_impedance_zap_report.py", "--strict"])

    rc = rpt.main()
    artifact = json.loads(Path("_test_results/impedance_zap_report.json").read_text(encoding="utf-8"))

    assert rc == 1
    assert artifact["ok"] == 1
    assert artifact["total"] == 2
    assert artifact["strict_mode"] is True
    assert artifact["all_guard_ok"] is False
    assert artifact["failed_case_ids"] == ["bad"]


def test_impedance_report_strict_mode_passes_when_all_guards_ok(monkeypatch, tmp_path):
    _inject_fake_pydantic(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rpt, "CASES", [{"id": "ok", "preset": "ok", "f_res_range": (0.5, 40.0)}])

    def fake_run_case(case: dict, *, fmin_hz: float, fmax_hz: float) -> dict:
        assert fmin_hz == 1.0
        assert fmax_hz == 50.0
        return {
            "id": case["id"],
            "preset": case["preset"],
            "valid": True,
            "guard_ok": True,
            "guard_reasons": [],
            "f_res_hz": 8.0,
            "z_res_kohm_cm2": 3.5,
            "f_res_expected_range_hz": [1.0, 40.0],
            "v_peak_mV": -42.0,
            "v_min_mV": -72.0,
        }

    monkeypatch.setattr(rpt, "run_case", fake_run_case)
    monkeypatch.setattr(sys, "argv", ["run_impedance_zap_report.py", "--strict", "--fmin", "1.0", "--fmax", "50.0"])

    rc = rpt.main()
    artifact = json.loads(Path("_test_results/impedance_zap_report.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert artifact["ok"] == 1
    assert artifact["total"] == 1
    assert artifact["analysis_band_hz"] == [1.0, 50.0]
    assert artifact["strict_mode"] is True
    assert artifact["all_guard_ok"] is True
    assert artifact["failed_case_ids"] == []
