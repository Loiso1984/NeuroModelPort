"""
Branch checks for topology fullscreen expansion.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset
from gui.topology import TopologyWidget


def test_topology_fullscreen_window_opens():
    app = QApplication.instance() or QApplication([])
    cfg = FullModelConfig()
    apply_preset(cfg, "D: alpha-Motoneuron (Powers 2001)")
    w = TopologyWidget()
    try:
        w.draw_neuron(cfg, delay_target_name="Terminal", delay_custom_index=1)
        assert len(w._fullscreen_windows) == 0
        w.open_fullscreen()
        assert len(w._fullscreen_windows) == 1
        fs = w._fullscreen_windows[0]
        assert "Full Screen" in fs.windowTitle()
        assert fs.centralWidget() is not None
    finally:
        for fs in list(getattr(w, "_fullscreen_windows", [])):
            fs.close()
        w.close()


def _run_as_script() -> int:
    tests = [test_topology_fullscreen_window_opens]
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

