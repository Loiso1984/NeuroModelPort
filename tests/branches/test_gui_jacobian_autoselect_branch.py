"""
Branch checks for GUI Jacobian auto-selection on heavy presets.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gui.main_window import MainWindow


def test_heavy_preset_autoselects_sparse_jacobian():
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    try:
        win.load_preset("F: Multiple Sclerosis (Demyelination)")
        app.processEvents()
        assert win.config.stim.jacobian_mode == "sparse_fd"
    finally:
        win.close()


def test_nonheavy_preset_keeps_dense_default():
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    try:
        win.load_preset("A: Squid Giant Axon (HH 1952)")
        app.processEvents()
        assert win.config.stim.jacobian_mode == "dense_fd"
    finally:
        win.close()


def test_preset_load_resets_dual_stim_to_disabled():
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    try:
        win.dual_stim_widget.check_enabled.setChecked(True)
        app.processEvents()
        assert win.dual_stim_widget.config.enabled
        win.load_preset("B: Pyramidal L5 (Mainen 1996)")
        app.processEvents()
        assert not win.dual_stim_widget.config.enabled
        assert not win.dual_stim_widget.check_enabled.isChecked()
    finally:
        win.close()


def _run_as_script() -> int:
    tests = [
        test_heavy_preset_autoselects_sparse_jacobian,
        test_nonheavy_preset_keeps_dense_default,
        test_preset_load_resets_dual_stim_to_disabled,
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
