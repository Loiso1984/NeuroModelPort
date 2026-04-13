"""Shared test utilities — spike detection helpers used across branches and utils.

Consolidates the copy-pasted _spike_times / _first_cross implementations
into a single canonical source of truth.
"""
from __future__ import annotations

import numpy as np


def _spike_times(v: np.ndarray, t: np.ndarray, threshold: float = -20.0) -> np.ndarray:
    """Extract spike times from voltage trace via upward threshold crossing.

    Returns times at which v crosses *threshold* from below, with a minimum
    inter-spike interval of 1 ms to avoid double-counting.
    """
    idx = np.where((v[:-1] < threshold) & (v[1:] >= threshold))[0] + 1
    if len(idx) == 0:
        return np.array([], dtype=float)
    st = t[idx]
    keep = [0]
    for i in range(1, len(st)):
        if st[i] - st[keep[-1]] >= 1.0:
            keep.append(i)
    return st[keep]


# Alias for files that imported the function under an alternate name
_spike_times_crossing = _spike_times
_spike_times_by_crossing = _spike_times


def _first_cross(v: np.ndarray, t: np.ndarray, threshold: float = 0.0) -> float:
    """Return the time of the first upward threshold crossing, or NaN."""
    idx = np.where((v[:-1] < threshold) & (v[1:] >= threshold))[0]
    return float(t[idx[0] + 1]) if len(idx) else float("nan")
