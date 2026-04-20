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
from core.morphology import MorphologyBuilder
from core.presets import apply_preset
from core.analysis import full_analysis
from core.solver import NeuronSolver
from gui.analytics import AnalyticsWidget
from gui.main_window import MainWindow


def test_analytics_fullscreen_opens_and_keeps_tab_index():
    app = QApplication.instance() or QApplication([])
    cfg = FullModelConfig()
    apply_preset(cfg, "D: alpha-Motoneuron (Powers 2001)")
    cfg.stim.t_sim = 120.0
    cfg.stim.dt_eval = 0.2
    cfg.stim.jacobian_mode = "native_hines"
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


def test_main_window_single_result_populates_analytics_immediately():
    app = QApplication.instance() or QApplication([])
    cfg = FullModelConfig()
    apply_preset(cfg, "A: Squid Giant Axon (HH 1952)")
    cfg.stim.t_sim = 40.0
    cfg.stim.dt_eval = 0.2
    cfg.stim.jacobian_mode = "native_hines"
    res = NeuronSolver(cfg).run_single()
    morph = MorphologyBuilder.build(cfg)
    stats = full_analysis(res, compute_lyapunov=False)

    win = MainWindow()
    try:
        win._on_simulation_done({"single": res, "stats": stats, "morph": morph})
        app.processEvents()
        assert win.analytics._last_result is res
        assert win.analytics._last_stats is stats
        assert win.analytics.passport_view.toPlainText().strip()
        assert "Analysis ready" in win.analytics._passport_status_label.text()
    finally:
        win.close()


def test_analytics_workspace_keeps_dock_visible_and_raises_passport():
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    try:
        win.show()
        app.processEvents()
        win.analytics.setCurrentIndex(min(2, win.analytics.count() - 1))
        win._dock_analytics.setVisible(False)
        win._focus_analytics_workspace()
        app.processEvents()
        assert win._dock_analytics.isVisible()
        assert win.analytics.currentIndex() == 0
    finally:
        win.close()


def _run_as_script() -> int:
    tests = [
        test_analytics_fullscreen_opens_and_keeps_tab_index,
        test_main_window_single_result_populates_analytics_immediately,
        test_analytics_workspace_keeps_dock_visible_and_raises_passport,
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

