from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import tests.utils.run_priority_physiology_cluster as cluster


def test_cluster_marks_pydantic_missing_as_warning(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    class _P:
        returncode = 2
        stdout = "[WARN] missing dependency for report execution: No module named 'pydantic'\n"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: _P())
    monkeypatch.setattr(sys, "argv", ["run_priority_physiology_cluster.py"])

    rc = cluster.main()
    artifact = json.loads(Path("_test_results/priority_physiology_cluster.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert artifact["warn_count"] == len(cluster.TASKS)
    assert artifact["fail_count"] == 0
    assert artifact["critical_ok"] is False
    assert artifact["all_ok"] is True
    assert len(artifact["next_actions"]) >= 1


def test_cluster_fail_on_warn(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    class _P:
        returncode = 2
        stdout = "No module named 'pydantic'\n"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: _P())
    monkeypatch.setattr(sys, "argv", ["run_priority_physiology_cluster.py", "--fail-on-warn"])

    rc = cluster.main()
    artifact = json.loads(Path("_test_results/priority_physiology_cluster.json").read_text(encoding="utf-8"))

    assert rc == 1
    assert artifact["warn_count"] == len(cluster.TASKS)
    assert artifact["all_ok"] is False


def test_cluster_strict_critical(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    class _P:
        returncode = 0
        stdout = "ok\n"
        stderr = ""

    def fake_run(cmd, **kwargs):
        if "run_f_conduction_extended.py" in " ".join(cmd):
            p = _P()
            p.returncode = 2
            p.stdout = "No module named 'pydantic'\n"
            return p
        return _P()

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["run_priority_physiology_cluster.py", "--strict-critical"])

    rc = cluster.main()
    artifact = json.loads(Path("_test_results/priority_physiology_cluster.json").read_text(encoding="utf-8"))

    assert rc == 1
    assert artifact["strict_critical"] is True
    assert artifact["critical_ok"] is False
