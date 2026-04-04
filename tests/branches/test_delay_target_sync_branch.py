"""
Branch checks for delay-target sync between Oscilloscope and Topology.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gui.main_window import MainWindow


def _mk_window() -> MainWindow:
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.load_preset("F: Multiple Sclerosis (Demyelination)")
    app.processEvents()
    return win


def test_delay_target_sync_to_topology_info():
    app = QApplication.instance() or QApplication([])
    win = _mk_window()
    try:
        win.oscilloscope._combo_delay_target.setCurrentText("AIS")
        app.processEvents()
        info_ais = win.topology._info.text()
        assert "Delay target: AIS (idx 1)" in info_ais

        win.oscilloscope._combo_delay_target.setCurrentText("Trunk Junction")
        app.processEvents()
        info_junction = win.topology._info.text()
        assert "Delay target: fork (idx " in info_junction

        win.oscilloscope._combo_delay_target.setCurrentText("Custom Compartment")
        win.oscilloscope._spin_delay_comp.setValue(7)
        app.processEvents()
        info_custom = win.topology._info.text()
        assert "Delay target: comp[7] (idx 7)" in info_custom
    finally:
        win.close()


def test_delay_controls_range_syncs_on_preset_load():
    win = _mk_window()
    try:
        spin = win.oscilloscope._spin_delay_comp
        assert spin.minimum() == 1
        assert spin.maximum() > 1
    finally:
        win.close()


def test_topology_preview_updates_on_stim_location_edit():
    app = QApplication.instance() or QApplication([])
    win = _mk_window()
    try:
        win.form_stim_loc.widgets_map["location"].setCurrentText("soma")
        app.processEvents()
        info = win.topology._info.text()
        assert "Stim: soma/" in info
    finally:
        win.close()


def _run_as_script() -> int:
    tests = [
        test_delay_target_sync_to_topology_info,
        test_delay_controls_range_syncs_on_preset_load,
        test_topology_preview_updates_on_stim_location_edit,
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
