from __future__ import annotations

import numpy as np

from core.analysis import extract_spatial_traces


def test_extract_spatial_traces_reuses_soma_when_trace_is_1d():
    trace = np.array([1.0, 2.0, 3.0], dtype=float)
    soma, ais, terminal = extract_spatial_traces(trace, n_comp=3)
    assert np.array_equal(soma, trace)
    assert np.array_equal(ais, trace)
    assert np.array_equal(terminal, trace)


def test_extract_spatial_traces_maps_ais_and_terminal_for_multicomp():
    trace = np.array(
        [
            [10.0, 11.0, 12.0],  # soma
            [20.0, 21.0, 22.0],  # ais
            [30.0, 31.0, 32.0],  # terminal
        ],
        dtype=float,
    )
    soma, ais, terminal = extract_spatial_traces(trace, n_comp=3)
    assert np.array_equal(soma, trace[0])
    assert np.array_equal(ais, trace[1])
    assert np.array_equal(terminal, trace[2])

