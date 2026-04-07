from __future__ import annotations

from tests.utils.run_p0_p1_gate import _classify_exit_code


def test_classify_exit_code_policy():
    assert _classify_exit_code(0) == "PASS"
    assert _classify_exit_code(1) == "FAIL"
    assert _classify_exit_code(2) == "WARN_DEPENDENCY"
    assert _classify_exit_code(124) == "TIMEOUT"
    assert _classify_exit_code(5) == "ERROR"
