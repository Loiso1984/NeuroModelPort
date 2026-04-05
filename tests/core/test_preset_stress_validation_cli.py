from __future__ import annotations

import importlib


def test_main_returns_dependency_diagnostic_when_import_error(monkeypatch, capsys):
    mod = importlib.import_module("tests.utils.run_preset_stress_validation")

    monkeypatch.setattr(mod, "_IMPORT_ERROR", ModuleNotFoundError(name="pydantic"))

    rc = mod.main()
    captured = capsys.readouterr()

    assert rc == 2
    assert "preset-stress" in captured.out
    assert "pydantic" in captured.out
