"""
Branch checks for spike mechanism analytics tab.
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


def test_spike_mechanism_tab_populates_axes():
    app = QApplication.instance() or QApplication([])
    cfg = FullModelConfig()
    apply_preset(cfg, "D: alpha-Motoneuron (Powers 2001)")
    cfg.stim.t_sim = 180.0
    cfg.stim.dt_eval = 0.2
    cfg.stim.jacobian_mode = "native_hines"
    res = NeuronSolver(cfg).run_single()

    w = AnalyticsWidget()
    try:
        w.update_analytics(res)
        w._ensure_built("_build_tab_spike_mech")
        w._update_spike_mechanism(res, w._last_stats)
        assert hasattr(w, "fig_spike_mech")
        assert len(w.fig_spike_mech.axes) >= 5
        # Top axis should include soma voltage trace.
        assert len(w.fig_spike_mech.axes[0].lines) >= 1
    finally:
        w.close()


def _run_as_script() -> int:
    tests = [test_spike_mechanism_tab_populates_axes]
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

