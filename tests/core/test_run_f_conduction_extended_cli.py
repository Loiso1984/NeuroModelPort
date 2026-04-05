from __future__ import annotations

import importlib

from tests.utils.runtime_import_guard import missing_dependency_name


def test_main_returns_diagnostic_when_runtime_dependency_missing(monkeypatch, capsys):
    mod = importlib.import_module("tests.utils.run_f_conduction_extended")

    monkeypatch.setattr(mod, "_IMPORT_ERROR", ModuleNotFoundError(name='pydantic'))
    monkeypatch.setattr("sys.argv", ["run_f_conduction_extended.py"])

    rc = mod.main()
    captured = capsys.readouterr()

    assert rc == 2
    assert "dependency diagnostic" in captured.err
    assert "pydantic" in captured.err


def test_missing_dependency_name_falls_back_to_exception_text():
    exc = ModuleNotFoundError("No module named 'pydantic'")
    assert missing_dependency_name(exc) == "pydantic"
