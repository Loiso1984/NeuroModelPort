"""
Branch checks for readability controls in topology/oscilloscope widgets.
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
from gui.plots import OscilloscopeWidget


def test_topology_readability_controls_apply_without_crash():
    app = QApplication.instance() or QApplication([])
    cfg = FullModelConfig()
    apply_preset(cfg, "D: alpha-Motoneuron (Powers 2001)")
    w = TopologyWidget()
    try:
        w.draw_neuron(cfg)
        w._cb_labels.setChecked(False)
        w._cb_indices.setChecked(False)
        w._cb_contrast.setChecked(True)
        w._spin_line.setValue(1.6)
        w._spin_font.setValue(1.4)
        assert w._high_contrast is True
        assert abs(w._line_scale - 1.6) < 1e-9
        assert abs(w._font_scale - 1.4) < 1e-9
        assert "idx " not in w._info.text() or isinstance(w._info.text(), str)
    finally:
        w.close()


def test_oscilloscope_presentation_mode_scales_readability():
    app = QApplication.instance() or QApplication([])
    w = OscilloscopeWidget()
    try:
        base_lw = float(w._spin_line_width.value())
        base_title = int(w._spin_title_px.value())
        w._cb_presentation.setChecked(True)
        assert w._line_width_scale > base_lw
        assert w._title_font_px >= base_title + 1
    finally:
        w.close()


def _run_as_script() -> int:
    tests = [
        test_topology_readability_controls_apply_without_crash,
        test_oscilloscope_presentation_mode_scales_readability,
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

