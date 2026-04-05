"""Branch checks for MainWindow minimum-size constraints (geometry safety)."""

from __future__ import annotations

import pytest
pytest.importorskip("pydantic")

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gui.main_window import MainWindow


def test_main_window_has_bounded_minimum_size():
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    try:
        min_size = w.minimumSize()
        assert min_size.width() <= 1000, f"minimum width too large: {min_size.width()}"
        assert min_size.height() <= 800, f"minimum height too large: {min_size.height()}"
        assert w.tabs.sizePolicy().horizontalPolicy() == w.tabs.sizePolicy().Policy.Expanding
    finally:
        w.close()


def _run_as_script() -> int:
    tests = [test_main_window_has_bounded_minimum_size]
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
