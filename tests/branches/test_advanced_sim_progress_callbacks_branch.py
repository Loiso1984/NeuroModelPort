"""
Branch checks that advanced_sim runners accept progress callbacks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import core.advanced_sim as adv
from core.models import FullModelConfig


class _DummyResult:
    def __init__(self, t: np.ndarray, v: np.ndarray):
        self.t = t
        self.v_soma = v
        self.v_all = np.vstack([v])
        self.n_comp = 1


class _DummySolver:
    def __init__(self, cfg):
        self.cfg = cfg

    def run_single(self):
        t = np.linspace(0.0, 30.0, 301)
        v = np.full_like(t, -65.0)
        # For larger Iext emulate a suprathreshold AP-like hump.
        if float(self.cfg.stim.Iext) > 5.0:
            v = v + 95.0 * np.exp(-0.5 * ((t - 10.0) / 0.6) ** 2)
        return _DummyResult(t, v)


def test_run_sd_curve_accepts_progress_callback():
    cfg = FullModelConfig()
    calls = []
    orig = adv.NeuronSolver
    adv.NeuronSolver = _DummySolver
    try:
        out = adv.run_sd_curve(
            cfg,
            durations=np.array([0.5, 1.0], dtype=float),
            progress_cb=lambda i, n, v: calls.append((i, n, float(v))),
        )
        assert "durations" in out and len(out["durations"]) == 2
        assert len(calls) >= 2
    finally:
        adv.NeuronSolver = orig


def test_run_excitability_map_accepts_progress_callback():
    cfg = FullModelConfig()
    calls = []
    orig = adv.NeuronSolver
    adv.NeuronSolver = _DummySolver
    try:
        out = adv.run_excitability_map(
            cfg,
            I_range=np.array([1.0, 8.0], dtype=float),
            dur_range=np.array([1.0], dtype=float),
            progress_cb=lambda i, n, v: calls.append((i, n, float(v))),
        )
        assert out["spike_matrix"].shape == (2, 1)
        assert len(calls) == 2
    finally:
        adv.NeuronSolver = orig


def _run_as_script() -> int:
    tests = [
        test_run_sd_curve_accepts_progress_callback,
        test_run_excitability_map_accepts_progress_callback,
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

