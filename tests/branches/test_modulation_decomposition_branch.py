"""
Branch checks for non-FFT spike modulation decomposition.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.analysis import estimate_spike_modulation, full_analysis
from core.models import FullModelConfig


def _build_theta_locked_trace(
    *,
    freq_hz: float = 8.0,
    duration_ms: float = 4000.0,
    dt_ms: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(0.0, duration_ms, dt_ms)
    phase = 2.0 * np.pi * freq_hz * (t / 1000.0)
    v = -62.0 + 3.0 * np.sin(phase)

    period_ms = 1000.0 / freq_hz
    spike_times = []
    k = 0
    while True:
        ts = (k + 0.25) * period_ms
        if ts >= duration_ms - 50.0:
            break
        spike_times.append(ts)
        v += 95.0 * np.exp(-0.5 * ((t - ts) / 0.8) ** 2)
        k += 1

    return t, v, np.asarray(spike_times, dtype=float)


def test_modulation_estimator_detects_phase_locking():
    t, v, spike_times = _build_theta_locked_trace()
    out = estimate_spike_modulation(
        spike_times,
        t,
        v,
        low_hz=4.0,
        high_hz=12.0,
        phase_bins=18,
        surrogate_count=50,
    )
    assert out["valid"], "phase-locking case should be analyzable"
    assert out["plv"] > 0.70, f"expected strong phase locking, got PLV={out['plv']:.3f}"
    assert out["surrogate_p_value"] < 0.05, f"expected significant coupling, got p={out['surrogate_p_value']}"


def test_modulation_estimator_rejects_unlocked_spike_pattern():
    t, v, _ = _build_theta_locked_trace()
    # Deterministic quasi-random schedule (incommensurate with theta cycle).
    spike_times = ((np.arange(30, dtype=float) * 97.0) % 3400.0) + 200.0
    out = estimate_spike_modulation(
        spike_times,
        t,
        v,
        low_hz=4.0,
        high_hz=12.0,
        phase_bins=18,
        surrogate_count=50,
    )
    assert out["valid"], "unlocked case should still be analyzable"
    assert out["plv"] < 0.40, f"expected weaker locking, got PLV={out['plv']:.3f}"
    assert out["surrogate_p_value"] > 0.05, f"expected non-significant coupling, got p={out['surrogate_p_value']}"


def test_full_analysis_exposes_modulation_metrics():
    t, v, _ = _build_theta_locked_trace()
    cfg = FullModelConfig()
    cfg.analysis.enable_modulation_decomposition = True
    cfg.analysis.modulation_source = "voltage"
    cfg.analysis.modulation_low_hz = 4.0
    cfg.analysis.modulation_high_hz = 12.0
    cfg.analysis.modulation_phase_bins = 18
    cfg.analysis.modulation_surrogates = 40

    class DummyResult:
        pass

    res = DummyResult()
    res.t = t
    res.v_soma = v
    res.v_all = np.vstack([v])
    res.n_comp = 1
    res.config = cfg
    res.currents = {
        "Na": np.zeros_like(v),
        "K": np.zeros_like(v),
        "Leak": np.zeros_like(v),
    }
    res.atp_estimate = 0.0
    res.v_dendritic_filtered = None

    stats = full_analysis(res)
    assert "modulation_plv" in stats
    assert stats["modulation_valid"]
    assert np.isfinite(stats["modulation_plv"])
    assert stats["modulation_plv"] > 0.5


def _run_as_script() -> int:
    tests = [
        test_modulation_estimator_detects_phase_locking,
        test_modulation_estimator_rejects_unlocked_spike_pattern,
        test_full_analysis_exposes_modulation_metrics,
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
