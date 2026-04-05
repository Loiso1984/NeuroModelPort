from __future__ import annotations

import importlib

from tests.utils.runtime_import_guard import missing_dependency_name


def test_main_returns_diagnostic_when_runtime_dependency_missing(monkeypatch, capsys, tmp_path):
    mod = importlib.import_module("tests.utils.run_f_conduction_extended")
    out_file = tmp_path / "f_diag.json"

    monkeypatch.setattr(mod, "_IMPORT_ERROR", ModuleNotFoundError(name='pydantic'))
    monkeypatch.setattr(
        "sys.argv",
        ["run_f_conduction_extended.py", "--output", str(out_file)],
    )

    rc = mod.main()
    captured = capsys.readouterr()

    assert rc == 2
    assert "dependency diagnostic" in captured.err
    assert "pydantic" in captured.err
    assert out_file.exists()
    payload = out_file.read_text(encoding="utf-8")
    assert "dependency_error" in payload
    assert "\"gate\"" in payload


def test_missing_dependency_name_falls_back_to_exception_text():
    exc = ModuleNotFoundError("No module named 'pydantic'")
    assert missing_dependency_name(exc) == "pydantic"


def test_exit_code_for_anomalies_policy():
    mod = importlib.import_module("tests.utils.run_f_conduction_extended")

    assert mod._exit_code_for_anomalies(anomaly_count=0, fail_on_anomaly=True) == 0
    assert mod._exit_code_for_anomalies(anomaly_count=3, fail_on_anomaly=False) == 0
    assert mod._exit_code_for_anomalies(anomaly_count=1, fail_on_anomaly=True) == 1
