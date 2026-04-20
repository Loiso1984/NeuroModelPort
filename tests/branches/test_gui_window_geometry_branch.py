"""Branch checks for dock-based MainWindow laptop geometry and session restore."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QToolBar

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.models import FullModelConfig
from gui.main_window import MainWindow


def _app():
    return QApplication.instance() or QApplication([])


def test_clean_start_uses_named_default_preset_and_visible_primary_docks():
    app = _app()
    with tempfile.TemporaryDirectory() as tmp:
        old_cwd = os.getcwd()
        os.chdir(tmp)
        try:
            w = MainWindow()
            w.show()
            app.processEvents()
            try:
                assert w.config_manager.current_preset_name.startswith("A:")
                assert w.combo_presets.currentText().startswith("A:")
                assert w._sidebar_preset_combo.currentText().startswith("A:")
                assert w._dock_params.isVisible()
                assert w._dock_analytics.isVisible()
                assert w.centralWidget() is w.oscilloscope
                assert isinstance(w.main_toolbar, QToolBar)
                assert isinstance(w.settings_toolbar, QToolBar)
            finally:
                w.close()
        finally:
            os.chdir(old_cwd)


def test_restored_config_without_preset_identity_is_marked_custom():
    app = _app()
    with tempfile.TemporaryDirectory() as tmp:
        old_cwd = os.getcwd()
        os.chdir(tmp)
        try:
            FullModelConfig().save_to_file(".last_session.json")
            w = MainWindow()
            w.show()
            app.processEvents()
            try:
                assert w.config_manager.current_preset_name == "Custom Config"
                assert w.combo_presets.currentText() == "Custom Config"
                assert w._sidebar_preset_combo.currentText() == "Custom Config"
            finally:
                w.close()
        finally:
            os.chdir(old_cwd)


def test_layout_guard_recovers_hidden_primary_docks_for_laptop():
    app = _app()
    with tempfile.TemporaryDirectory() as tmp:
        old_cwd = os.getcwd()
        os.chdir(tmp)
        try:
            w = MainWindow()
            w.show()
            app.processEvents()
            try:
                w._dock_params.setVisible(False)
                w._dock_analytics.setVisible(False)
                w.resize(1100, 700)
                w._ensure_usable_layout()
                app.processEvents()
                assert w._dock_params.isVisible()
                assert w._dock_analytics.isVisible()
                assert w.btn_run.isVisible()
                assert w.combo_presets.isVisible()
                assert w._sidebar_frame.maximumWidth() <= 430
            finally:
                w.close()
        finally:
            os.chdir(old_cwd)


def test_desktop_and_debug_layouts_allow_wide_parameter_panel():
    app = _app()
    with tempfile.TemporaryDirectory() as tmp:
        old_cwd = os.getcwd()
        os.chdir(tmp)
        try:
            w = MainWindow()
            w.show()
            app.processEvents()
            try:
                w._apply_layout_preset("Desktop", resize_window=False)
                assert w._sidebar_frame.maximumWidth() >= 700
                w._apply_layout_preset("Debug", resize_window=False)
                assert w._sidebar_frame.maximumWidth() >= 850
            finally:
                w.close()
        finally:
            os.chdir(old_cwd)


def test_live_slider_stops_pending_analytics_timer():
    app = _app()
    with tempfile.TemporaryDirectory() as tmp:
        old_cwd = os.getcwd()
        os.chdir(tmp)
        try:
            w = MainWindow()
            w.show()
            app.processEvents()
            try:
                # Guard: ensure live sliders are initialized
                assert len(w._live_sliders) > 0, "Live sliders not initialized"
                w._analytics_debounce_timer.start(5000)
                assert w._analytics_debounce_timer.isActive()
                w._on_live_slider_moved(0, w._live_sliders[0].value())
                assert not w._analytics_debounce_timer.isActive()
                assert w._live_timer.isActive()
            finally:
                w.close()
        finally:
            os.chdir(old_cwd)


def _run_as_script() -> int:
    tests = [
        test_clean_start_uses_named_default_preset_and_visible_primary_docks,
        test_restored_config_without_preset_identity_is_marked_custom,
        test_layout_guard_recovers_hidden_primary_docks_for_laptop,
        test_desktop_and_debug_layouts_allow_wide_parameter_panel,
        test_live_slider_stops_pending_analytics_timer,
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
