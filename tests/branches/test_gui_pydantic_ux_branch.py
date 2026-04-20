"""Branch checks for the Pydantic generated-form UX layer."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import SimulationParams
from gui.widgets.form_generator import PydanticFormWidget


def _app():
    return QApplication.instance() or QApplication([])


def test_pydantic_form_priority_filter_keeps_model_binding():
    app = _app()
    stim = SimulationParams()
    form = PydanticFormWidget(
        stim,
        "Stimulus",
        field_priorities={"Iext": "critical", "stim_type": "critical", "pulse_dur": "advanced"},
    )
    try:
        form.show()
        app.processEvents()
        form.set_priority_filter("critical")
        assert not form.widgets_map["Iext"].isHidden()
        assert not form.widgets_map["stim_type"].isHidden()
        assert form.widgets_map["pulse_dur"].isHidden()

        form.widgets_map["Iext"].setValue(12.5)
        assert abs(float(stim.Iext) - 12.5) < 1e-9
    finally:
        form.close()


def test_pydantic_form_search_filter_does_not_mutate_hidden_fields():
    app = _app()
    stim = SimulationParams()
    before = stim.pulse_dur
    form = PydanticFormWidget(stim, "Stimulus")
    try:
        form.show()
        app.processEvents()
        form.set_search_filter("alpha")
        assert not form.widgets_map["alpha_tau"].isHidden()
        assert form.widgets_map["pulse_dur"].isHidden()
        assert stim.pulse_dur == before
    finally:
        form.close()


def _run_as_script() -> int:
    tests = [
        test_pydantic_form_priority_filter_keeps_model_binding,
        test_pydantic_form_search_filter_does_not_mutate_hidden_fields,
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
