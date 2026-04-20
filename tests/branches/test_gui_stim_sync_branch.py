"""Branch checks for GUI stimulation-type synchronization and dual-stim state."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gui.main_window import MainWindow


def _app():
    return QApplication.instance() or QApplication([])


def _is_shown(win: MainWindow, field_name: str) -> bool:
    w = win.form_stim.widgets_map[field_name]
    l = win.form_stim.labels_map[field_name]
    return (not w.isHidden()) and (not l.isHidden())


def _window_in_temp_cwd() -> MainWindow:
    # Caller owns cwd restoration through the context that invokes this helper.
    return MainWindow()


def test_stim_type_visibility_and_synaptic_defaults():
    app = _app()
    with tempfile.TemporaryDirectory() as tmp:
        old_cwd = os.getcwd()
        os.chdir(tmp)
        try:
            w = _window_in_temp_cwd()
            try:
                cfg = w.config_manager.config
                cfg.stim.stim_type = "const"
                w._sync_stim_type_controls()
                assert _is_shown(w, "alpha_tau") is False
                assert _is_shown(w, "pulse_start") is False
                assert _is_shown(w, "pulse_dur") is False

                w._on_stim_field_changed("stim_type", "NMDA")
                assert str(cfg.stim.stim_type) == "NMDA"
                assert abs(float(cfg.stim.alpha_tau) - 1.0) < 1e-9
                assert abs(float(cfg.stim.Iext) - 6.0) < 1e-9
                assert _is_shown(w, "alpha_tau") is True
                assert _is_shown(w, "pulse_start") is True
                assert _is_shown(w, "pulse_dur") is False
            finally:
                w.close()
        finally:
            os.chdir(old_cwd)


def test_dual_mode_keeps_primary_in_canonical_stim_and_syncs_secondary_config():
    app = _app()
    with tempfile.TemporaryDirectory() as tmp:
        old_cwd = os.getcwd()
        os.chdir(tmp)
        try:
            w = _window_in_temp_cwd()
            try:
                cfg = w.config_manager.config
                cfg.stim.stim_type = "alpha"
                cfg.stim.Iext = 42.0
                cfg.stim.pulse_start = 12.5
                cfg.stim.alpha_tau = 1.7

                w.dual_stim_widget.config.enabled = True
                w.dual_stim_widget.config.secondary_stim_type = "GABAA"
                w.dual_stim_widget.config.secondary_Iext = 15.0
                w._on_dual_stim_config_changed()

                assert cfg.dual_stimulation is not None
                assert cfg.dual_stimulation.enabled is True
                assert cfg.dual_stimulation.secondary_stim_type == "GABAA"
                assert cfg.stim.stim_type == "alpha"
                assert abs(float(cfg.stim.alpha_tau) - 1.7) < 1e-9

                stim_type_widget = w.form_stim.widgets_map["stim_type"]
                alpha_widget = w.form_stim.widgets_map["alpha_tau"]
                assert stim_type_widget.currentText() == "alpha"
                assert abs(float(alpha_widget.value()) - 1.7) < 1e-9
                assert _is_shown(w, "alpha_tau") is True
            finally:
                w.close()
        finally:
            os.chdir(old_cwd)


def _run_as_script() -> int:
    tests = [
        test_stim_type_visibility_and_synaptic_defaults,
        test_dual_mode_keeps_primary_in_canonical_stim_and_syncs_secondary_config,
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
