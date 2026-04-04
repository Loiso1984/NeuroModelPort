"""
Branch checks for GUI stimulation-type synchronization and dual-preview behavior.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gui.main_window import MainWindow


def _is_shown(win: MainWindow, field_name: str) -> bool:
    w = win.form_stim.widgets_map[field_name]
    l = win.form_stim.labels_map[field_name]
    return (not w.isHidden()) and (not l.isHidden())


def test_stim_type_visibility_and_synaptic_defaults():
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    try:
        w.config.stim.stim_type = "const"
        w._sync_stim_type_controls()
        assert _is_shown(w, "alpha_tau") is False
        assert _is_shown(w, "pulse_start") is False
        assert _is_shown(w, "pulse_dur") is False

        w._on_stim_field_changed("stim_type", "NMDA")
        assert str(w.config.stim.stim_type) == "NMDA"
        assert abs(float(w.config.stim.alpha_tau) - 70.0) < 1e-9
        assert abs(float(w.config.stim.Iext) - 0.8) < 1e-9
        assert _is_shown(w, "alpha_tau") is True
        assert _is_shown(w, "pulse_start") is True
        assert _is_shown(w, "pulse_dur") is False
    finally:
        w.close()


def test_dual_mode_mirrors_primary_stim_in_main_panel():
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    try:
        w.dual_stim_widget.config.enabled = True
        w.dual_stim_widget.config.primary_stim_type = "alpha"
        w.dual_stim_widget.config.primary_Iext = 42.0
        w.dual_stim_widget.config.primary_start = 12.5
        w.dual_stim_widget.config.primary_alpha_tau = 1.7
        w._on_dual_stim_config_changed()

        stim_type_widget = w.form_stim.widgets_map["stim_type"]
        alpha_widget = w.form_stim.widgets_map["alpha_tau"]
        assert stim_type_widget.currentText() == "alpha"
        assert abs(float(alpha_widget.value()) - 1.7) < 1e-9
        assert _is_shown(w, "alpha_tau") is True
    finally:
        w.close()


def _run_as_script() -> int:
    tests = [
        test_stim_type_visibility_and_synaptic_defaults,
        test_dual_mode_mirrors_primary_stim_in_main_panel,
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

