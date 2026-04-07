from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import tests.utils.run_active_branch_suite as suite


def test_run_single_utility_marks_missing_dependency_as_warning(monkeypatch):
    class _P:
        returncode = 2
        stdout = "[WARN] missing dependency for report execution: No module named 'pydantic'\n"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: _P())

    row = suite._run_single_utility("u", ["fake.py"])

    assert row["status"] == "warn"
    assert row["returncode"] == 2


def test_run_single_utility_marks_nonwarning_nonzero_as_fail(monkeypatch):
    class _P:
        returncode = 2
        stdout = "some other error\n"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: _P())

    row = suite._run_single_utility("u", ["fake.py"])

    assert row["status"] == "fail"
    assert row["returncode"] == 2


def test_main_writes_utility_checks_section(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(suite, "ACTIVE_TESTS", [])
    monkeypatch.setattr(suite, "ACTIVE_UTILS", [{"name": "u", "cmd": ["fake.py"]}])
    monkeypatch.setattr(suite, "_run_single_utility", lambda name, cmd: {
        "name": name,
        "cmd": [sys.executable, *cmd],
        "returncode": 0,
        "status": "pass",
        "elapsed_sec": 0.01,
        "stdout_tail": [],
        "stderr_tail": [],
    })
    monkeypatch.setattr(sys, "argv", ["run_active_branch_suite.py", "--workers", "1"])

    rc = suite.main()
    report = json.loads(Path("_test_results/active_branch_suite.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert report["all_ok"] is True
    assert report["utility_checks"][0]["name"] == "u"
    assert report["utility_checks"][0]["status"] == "pass"
    assert report["fail_on_warn"] is False
    assert report["utility_warn_count"] == 0
    assert report["utility_fail_count"] == 0


def test_main_fail_on_warn_enforced(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(suite, "ACTIVE_TESTS", [])
    monkeypatch.setattr(suite, "ACTIVE_UTILS", [{"name": "u", "cmd": ["fake.py"]}])
    monkeypatch.setattr(suite, "_run_single_utility", lambda name, cmd: {
        "name": name,
        "cmd": [sys.executable, *cmd],
        "returncode": 2,
        "status": "warn",
        "elapsed_sec": 0.01,
        "stdout_tail": ["[WARN] missing dependency for report execution: No module named 'pydantic'"],
        "stderr_tail": [],
    })
    monkeypatch.setattr(sys, "argv", ["run_active_branch_suite.py", "--workers", "1", "--fail-on-warn"])

    rc = suite.main()
    report = json.loads(Path("_test_results/active_branch_suite.json").read_text(encoding="utf-8"))

    assert rc == 1
    assert report["all_ok"] is False
    assert report["fail_on_warn"] is True
    assert report["utility_warn_count"] == 1
    assert report["utility_fail_count"] == 0
