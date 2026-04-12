"""
Branch checks for preset-mode visibility and scroll affordances in the modern GUI.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QScrollArea

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gui.analytics import AnalyticsWidget
from gui.main_window import MainWindow
from gui.plots import OscilloscopeWidget


def test_mode_controls_follow_preset_code_visibility():
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    try:
        cases = [
            ("R: Cholinergic Neuromodulation (ACh)", "ach_mode"),
            ("S: Pathology: Dravet Syndrome (SCN1A LOF)", "dravet_mode"),
            ("G: Local Anesthesia (gNa Block)", "anesthesia_mode"),
            ("N: Alzheimer's (v10 Calcium Toxicity)", "alzheimer_mode"),
            ("K: Thalamic Relay (Ih + ITCa + Burst)", "k_mode"),
        ]
        for preset_name, active_field in cases:
            win.load_preset(preset_name)
            app.processEvents()
            for field_name, widget in win.form_preset_modes.widgets_map.items():
                expected = field_name == active_field
                assert (not widget.isHidden()) is expected, (preset_name, field_name, expected)
    finally:
        win.close()


def test_sweep_param_control_is_visible_and_editable():
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    try:
        app.processEvents()
        assert hasattr(win, "_sweep_param_combo")
        assert win._sweep_param_combo.isEditable()
        assert win._sweep_param_combo.currentText() == "stim.Iext"
    finally:
        win.close()


def test_oscilloscope_plot_stack_is_scrollable():
    _ = QApplication.instance() or QApplication([])
    osc = OscilloscopeWidget()
    try:
        assert osc._plot_scroll.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded
        assert osc._plot_scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded
        assert osc._plot_container.minimumHeight() >= 700
    finally:
        osc.close()


def test_analytics_wide_tabs_allow_bidirectional_scroll():
    _ = QApplication.instance() or QApplication([])
    analytics = AnalyticsWidget()
    try:
        chaos_tab = analytics._build_tab_chaos()
        scrolls = chaos_tab.findChildren(QScrollArea)
        assert scrolls, "Chaos tab should expose a scroll area"
        assert any(s.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded for s in scrolls)
        assert any(s.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded for s in scrolls)
    finally:
        analytics.close()
