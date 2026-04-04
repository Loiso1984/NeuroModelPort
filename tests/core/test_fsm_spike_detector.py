"""
tests/core/test_fsm_spike_detector.py — FSM spike detector validation (Stage 3.1)

Tests the Numba-jitted Finite State Machine spike detector against
synthetic waveforms with known spike counts and compares results
with the legacy peak_repolarization algorithm on a real HH simulation.
"""
import numpy as np
import pytest

from core.analysis import detect_spikes


# ─────────────────────────────────────────────────────────────────────
#  Helpers: synthetic waveforms with known spike structure
# ─────────────────────────────────────────────────────────────────────

def _make_spike_train(n_spikes: int = 5, dt_ms: float = 0.025,
                      duration_ms: float = 100.0,
                      v_rest: float = -65.0,
                      v_peak: float = 30.0,
                      v_ahp: float = -75.0) -> tuple:
    """Generate a synthetic voltage trace with *n_spikes* HH-like spikes."""
    t = np.arange(0, duration_ms, dt_ms)
    V = np.full_like(t, v_rest)
    spike_interval_ms = duration_ms / (n_spikes + 1)

    for k in range(n_spikes):
        center = spike_interval_ms * (k + 1)
        # Rising phase (0.3 ms)
        mask_rise = (t >= center - 0.3) & (t < center)
        V[mask_rise] = v_rest + (v_peak - v_rest) * ((t[mask_rise] - (center - 0.3)) / 0.3)
        # Peak
        mask_peak = (t >= center) & (t < center + 0.05)
        V[mask_peak] = v_peak
        # Falling phase (0.5 ms)
        mask_fall = (t >= center + 0.05) & (t < center + 0.55)
        V[mask_fall] = v_peak + (v_ahp - v_peak) * ((t[mask_fall] - (center + 0.05)) / 0.5)
        # AHP recovery (1.5 ms)
        mask_ahp = (t >= center + 0.55) & (t < center + 2.05)
        V[mask_ahp] = v_ahp + (v_rest - v_ahp) * ((t[mask_ahp] - (center + 0.55)) / 1.5)

    return V, t


def _make_plateau(dt_ms: float = 0.025, duration_ms: float = 50.0,
                  v_rest: float = -65.0, v_plateau: float = -10.0) -> tuple:
    """Generate a sustained depolarization plateau — should NOT be counted as a spike.

    The voltage rises to v_plateau and stays there until the end of the trace,
    never returning below baseline_threshold.
    """
    t = np.arange(0, duration_ms, dt_ms)
    V = np.full_like(t, v_rest)
    # Rise to plateau at 10 ms, stay elevated until end — no repolarization
    mask_rise = (t >= 9.5) & (t < 10.0)
    V[mask_rise] = v_rest + (v_plateau - v_rest) * ((t[mask_rise] - 9.5) / 0.5)
    mask_plat = t >= 10.0
    V[mask_plat] = v_plateau
    return V, t


# ─────────────────────────────────────────────────────────────────────
#  Tests
# ─────────────────────────────────────────────────────────────────────

class TestFSMSyntheticWaveforms:
    """Validate FSM against synthetic traces with known spike counts."""

    @pytest.mark.parametrize("n_spikes", [0, 1, 3, 10])
    def test_correct_spike_count(self, n_spikes):
        if n_spikes == 0:
            V = np.full(2000, -65.0)
            t = np.linspace(0, 50, 2000)
        else:
            V, t = _make_spike_train(n_spikes=n_spikes, duration_ms=max(50, n_spikes * 15))
        idx, times, amps = detect_spikes(V, t, threshold=-20.0,
                                         baseline_threshold=-50.0,
                                         algorithm="fsm")
        assert len(idx) == n_spikes, f"Expected {n_spikes} spikes, got {len(idx)}"

    def test_plateau_rejected(self):
        """Sustained depolarization without repolarization must not be counted."""
        V, t = _make_plateau()
        idx, times, amps = detect_spikes(V, t, threshold=-20.0,
                                         baseline_threshold=-50.0,
                                         algorithm="fsm")
        assert len(idx) == 0, f"Plateau should produce 0 spikes, got {len(idx)}"

    def test_peak_indices_at_maxima(self):
        """FSM should report peak indices at the actual voltage maxima."""
        V, t = _make_spike_train(n_spikes=3, duration_ms=60)
        idx, _, amps = detect_spikes(V, t, threshold=-20.0,
                                     baseline_threshold=-50.0,
                                     algorithm="fsm")
        for i, pk in enumerate(idx):
            # Peak should be a local maximum (within ±1 sample)
            lo = max(0, pk - 1)
            hi = min(len(V) - 1, pk + 1)
            assert V[pk] >= V[lo] and V[pk] >= V[hi], \
                f"Spike {i}: index {pk} is not a local max"

    def test_refractory_enforcement(self):
        """Two spikes closer than refractory_ms should not both be detected."""
        # Build two spikes only 0.5 ms apart (refractory = 1.0 ms)
        dt = 0.01
        t = np.arange(0, 10, dt)
        V = np.full_like(t, -65.0)

        # Spike 1 at t=3 ms
        for offset, val in [(2.7, -40.0), (2.8, -10.0), (2.9, 20.0),
                            (3.0, 30.0), (3.1, 10.0), (3.2, -30.0),
                            (3.3, -60.0), (3.4, -70.0)]:
            V[int(offset / dt)] = val

        # Spike 2 at t=3.5 ms (only 0.5 ms later)
        for offset, val in [(3.5, -40.0), (3.55, 0.0), (3.6, 25.0),
                            (3.65, 5.0), (3.7, -40.0), (3.8, -65.0)]:
            idx_pt = min(int(offset / dt), len(V) - 1)
            V[idx_pt] = val

        idx, _, _ = detect_spikes(V, t, threshold=-20.0,
                                  baseline_threshold=-50.0,
                                  refractory_ms=1.0,
                                  algorithm="fsm")
        assert len(idx) == 1, f"Should detect 1 spike (refractory), got {len(idx)}"

    def test_empty_arrays(self):
        """Empty input should return empty output."""
        idx, times, amps = detect_spikes(np.array([]), np.array([]),
                                         algorithm="fsm")
        assert len(idx) == 0


class TestFSMvsLegacy:
    """Compare FSM results against legacy peak_repolarization on a real simulation."""

    @pytest.fixture(scope="class")
    def squid_result(self):
        """Run a standard Squid HH simulation."""
        from core.models import FullModelConfig
        from core.presets import apply_preset
        from core.solver import NeuronSolver
        cfg = FullModelConfig()
        apply_preset(cfg, "A: Squid Giant Axon (HH 1952)")
        cfg.stim.Iext = 10.0
        cfg.stim.t_sim = 50.0
        solver = NeuronSolver(cfg)
        return solver.run_single()

    def test_spike_counts_agree(self, squid_result):
        """FSM and peak_repolarization should find the same spike count on clean HH spikes."""
        V = squid_result.v_soma
        t = squid_result.t

        _, st_legacy, _ = detect_spikes(V, t, threshold=-20.0,
                                        baseline_threshold=-50.0,
                                        algorithm="peak_repolarization")
        _, st_fsm, _ = detect_spikes(V, t, threshold=-20.0,
                                     baseline_threshold=-50.0,
                                     algorithm="fsm")
        # Allow ±1 spike tolerance for edge effects near trace boundaries
        assert abs(len(st_fsm) - len(st_legacy)) <= 1, \
            f"FSM={len(st_fsm)}, legacy={len(st_legacy)} — spike counts diverged"

    def test_spike_times_close(self, squid_result):
        """Detected spike times should be within 1 ms of each other."""
        V = squid_result.v_soma
        t = squid_result.t

        _, st_legacy, _ = detect_spikes(V, t, threshold=-20.0,
                                        baseline_threshold=-50.0,
                                        algorithm="peak_repolarization")
        _, st_fsm, _ = detect_spikes(V, t, threshold=-20.0,
                                     baseline_threshold=-50.0,
                                     algorithm="fsm")
        n = min(len(st_legacy), len(st_fsm))
        if n > 0:
            diffs = np.abs(st_fsm[:n] - st_legacy[:n])
            assert np.all(diffs < 1.0), \
                f"Spike time differences exceed 1 ms: max={np.max(diffs):.3f} ms"
