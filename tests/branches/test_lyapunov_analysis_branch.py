"""
Branch checks for FTLE/LLE analysis utilities.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.analysis import classify_lyapunov, estimate_ftle_lle


def _damped_oscillator_series(n: int = 4000, dt_ms: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(n, dtype=float) * dt_ms
    x = np.exp(-t / 120.0) * np.sin(2.0 * np.pi * t / 20.0)
    return x, t


def _logistic_series(n: int = 5000, discard: int = 500, r: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros(n + discard, dtype=float)
    x[0] = 0.217
    for i in range(1, n + discard):
        x[i] = r * x[i - 1] * (1.0 - x[i - 1])
    xs = x[discard:]
    t = np.arange(len(xs), dtype=float)  # arbitrary step
    return xs, t


def test_lle_ftle_sign_stable_vs_chaotic_reference():
    x_stable, t_stable = _damped_oscillator_series()
    s = estimate_ftle_lle(
        x_stable,
        t_stable,
        embedding_dim=3,
        lag_steps=2,
        min_separation_ms=5.0,
        fit_start_ms=5.0,
        fit_end_ms=30.0,
    )
    assert np.isfinite(s["lle_per_ms"]), "stable case should produce finite FTLE/LLE"
    assert s["lle_per_ms"] < 0.0, f"stable damped oscillator expected negative FTLE/LLE, got {s['lle_per_ms']}"

    x_chaos, t_chaos = _logistic_series()
    c = estimate_ftle_lle(
        x_chaos,
        t_chaos,
        embedding_dim=3,
        lag_steps=1,
        min_separation_ms=10.0,
        fit_start_ms=2.0,
        fit_end_ms=20.0,
    )
    assert np.isfinite(c["lle_per_ms"]), "chaotic case should produce finite FTLE/LLE"
    assert c["lle_per_ms"] > 0.0, f"chaotic logistic reference expected positive FTLE/LLE, got {c['lle_per_ms']}"


def test_lyapunov_classification_labels():
    assert classify_lyapunov(-0.01) == "stable"
    assert classify_lyapunov(0.01) in {"unstable_or_chaotic", "candidate chaotic"}
    assert classify_lyapunov(0.0) in {"limit_cycle_like", "limit-cycle-like"}


def _run_as_script() -> int:
    tests = [test_lle_ftle_sign_stable_vs_chaotic_reference, test_lyapunov_classification_labels]
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
