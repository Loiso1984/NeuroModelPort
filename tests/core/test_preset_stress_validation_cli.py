from __future__ import annotations

import importlib


def test_main_returns_dependency_diagnostic_when_import_error(monkeypatch, capsys, tmp_path):
    mod = importlib.import_module("tests.utils.run_preset_stress_validation")
    out_json = tmp_path / "preset_diag.json"
    out_md = tmp_path / "preset_diag.md"

    monkeypatch.setattr(mod, "_IMPORT_ERROR", ModuleNotFoundError(name="pydantic"))
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_preset_stress_validation.py",
            "--out",
            str(out_json),
            "--report-md",
            str(out_md),
        ],
    )

    rc = mod.main()
    captured = capsys.readouterr()

    assert rc == 2
    assert "preset-stress" in captured.out
    assert "pydantic" in captured.out
    assert out_json.exists()
    assert out_md.exists()
    payload = out_json.read_text(encoding="utf-8")
    assert "dependency_error" in payload
    assert "\"gate\"" in payload


def test_exit_code_policy_for_summary_counts():
    mod = importlib.import_module("tests.utils.run_preset_stress_validation")

    assert mod._exit_code_for_summary(0, 0, fail_on_fail=True, fail_on_warn=False) == 0
    assert mod._exit_code_for_summary(2, 0, fail_on_fail=False, fail_on_warn=False) == 0
    assert mod._exit_code_for_summary(1, 0, fail_on_fail=True, fail_on_warn=False) == 1
    assert mod._exit_code_for_summary(0, 3, fail_on_fail=True, fail_on_warn=True) == 1


def test_overall_status_computation():
    mod = importlib.import_module("tests.utils.run_preset_stress_validation")

    assert mod._overall_status(fail_count=0, warn_count=0) == "PASS"
    assert mod._overall_status(fail_count=0, warn_count=2) == "WARN"
    assert mod._overall_status(fail_count=1, warn_count=10) == "FAIL"
