"""
Branch checks for analytics fullscreen expansion.
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
from core.solver import NeuronSolver
from gui.analytics import AnalyticsWidget


def test_analytics_fullscreen_opens_and_keeps_tab_index():
    app = QApplication.instance() or QApplication([])
    cfg = FullModelConfig()
    apply_preset(cfg, "D: alpha-Motoneuron (Powers 2001)")
    cfg.stim.t_sim = 120.0
    cfg.stim.dt_eval = 0.2
    cfg.stim.jacobian_mode = "sparse_fd"
    res = NeuronSolver(cfg).run_single()

    w = AnalyticsWidget()
    try:
        w.update_analytics(res)
        target_idx = min(4, w.count() - 1)
        w.setCurrentIndex(target_idx)
        w.open_fullscreen()
        assert len(w._fullscreen_windows) == 1
        fs_win = w._fullscreen_windows[0]
        fs_widget = fs_win.centralWidget()
        assert fs_widget is not None
        assert int(fs_widget.currentIndex()) == target_idx
        assert "Full Screen" in fs_win.windowTitle()
    finally:
        for win in list(getattr(w, "_fullscreen_windows", [])):
            win.close()
        w.close()


def _run_as_script() -> int:
    tests = [test_analytics_fullscreen_opens_and_keeps_tab_index]
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

