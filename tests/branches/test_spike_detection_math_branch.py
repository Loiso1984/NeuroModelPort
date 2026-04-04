"""
Branch tests for spike-detection math correctness.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.analysis import detect_spikes


def _gaussian_spike(t: np.ndarray, t0: float, amp: float, sigma: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((t - t0) / sigma) ** 2)


def _build_trace(dt_ms: float) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(0.0, 120.0 + dt_ms, dt_ms)
    v = np.full_like(t, -65.0)
    v += _gaussian_spike(t, 30.0, 95.0, 0.7)   # true AP-like spike
    v += _gaussian_spike(t, 70.0, 26.0, 3.0)   # shallow suprathreshold bump
    return t, v


def test_prominence_parameter_changes_detection_result():
    t, v = _build_trace(0.1)

    _, st_lo, _ = detect_spikes(v, t, threshold=-45.0, prominence=2.0, baseline_threshold=-55.0)
    _, st_hi, _ = detect_spikes(v, t, threshold=-45.0, prominence=30.0, baseline_threshold=-55.0)

    assert len(st_lo) >= len(st_hi), "Higher prominence must not increase detected spike count"
    assert len(st_lo) >= 2, "Low prominence should keep both strong and shallow peaks"
    assert len(st_hi) == 1, "High prominence should keep only the strong spike"


def test_detection_count_is_approximately_dt_invariant():
    t1, v1 = _build_trace(0.05)
    t2, v2 = _build_trace(0.25)

    _, st1, _ = detect_spikes(v1, t1, threshold=-45.0, prominence=8.0, baseline_threshold=-55.0)
    _, st2, _ = detect_spikes(v2, t2, threshold=-45.0, prominence=8.0, baseline_threshold=-55.0)

    assert abs(len(st1) - len(st2)) <= 1, (
        f"Spike count should be stable across dt; got fine={len(st1)} coarse={len(st2)}"
    )


def test_threshold_transition_not_sample_count():
    t = np.linspace(0.0, 100.0, 2001)
    v = np.full_like(t, -65.0)
    v[(t >= 25.0) & (t <= 40.0)] = 5.0  # long plateau above threshold

    _, st, _ = detect_spikes(v, t, threshold=-20.0, prominence=5.0, baseline_threshold=-50.0)
    assert len(st) <= 1, "A long suprathreshold plateau should not become many spikes"


def _run_as_script() -> int:
    tests = [
        test_prominence_parameter_changes_detection_result,
        test_detection_count_is_approximately_dt_invariant,
        test_threshold_transition_not_sample_count,
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
