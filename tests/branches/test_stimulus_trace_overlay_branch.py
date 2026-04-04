"""
Branch checks for oscilloscope stimulus-input overlay.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.solver import NeuronSolver
from gui.plots import OscilloscopeWidget


def test_oscilloscope_includes_stim_input_curve():
    app = QApplication.instance() or QApplication([])

    cfg = FullModelConfig()
    cfg.morphology.single_comp = True
    cfg.stim.stim_type = "alpha"
    cfg.stim.Iext = 12.0
    cfg.stim.pulse_start = 5.0
    cfg.stim.alpha_tau = 2.0
    cfg.stim.t_sim = 40.0
    cfg.stim.dt_eval = 0.2
    res = NeuronSolver(cfg).run_single()

    w = OscilloscopeWidget()
    try:
        w.update_plots(res)
        assert "Stim_input" in w._curves_i
        assert "Stim_filtered" not in w._curves_i
    finally:
        w.close()


def test_stim_input_checkbox_controls_visibility():
    app = QApplication.instance() or QApplication([])

    cfg = FullModelConfig()
    cfg.morphology.single_comp = True
    cfg.stim.stim_type = "pulse"
    cfg.stim.Iext = 8.0
    cfg.stim.pulse_start = 5.0
    cfg.stim.pulse_dur = 3.0
    cfg.stim.t_sim = 25.0
    cfg.stim.dt_eval = 0.2
    res = NeuronSolver(cfg).run_single()

    w = OscilloscopeWidget()
    try:
        w.update_plots(res)
        c = w._curves_i["Stim_input"]
        w._cb_i["Stim_input"].setChecked(False)
        app.processEvents()
        assert not c.isVisible()
        w._cb_i["Stim_input"].setChecked(True)
        app.processEvents()
        assert c.isVisible()
    finally:
        w.close()


def _run_as_script() -> int:
    tests = [
        test_oscilloscope_includes_stim_input_curve,
        test_stim_input_checkbox_controls_visibility,
    ]
    passed = 0
    for fn in tests:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] {fn.__name__}: {exc}")
    print(f"\nSummary: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(_run_as_script())

