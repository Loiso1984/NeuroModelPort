import pytest
pytest.importorskip("pydantic")

import numpy as np

from core.rhs import get_stim_current


def test_zap_stimulus_is_zero_outside_pulse_window():
    stype_zap = 10
    iext = 5.0
    t0 = 100.0
    td = 400.0
    tau = 2.0

    assert get_stim_current(50.0, stype_zap, iext, t0, td, tau, 1.0, 20.0) == 0.0
    assert get_stim_current(510.0, stype_zap, iext, t0, td, tau, 1.0, 20.0) == 0.0


def test_zap_stimulus_frequency_increases_over_time():
    stype_zap = 10
    iext = 3.0
    t0 = 0.0
    td = 1000.0
    tau = 2.0
    f0, f1 = 2.0, 20.0

    t = np.arange(0.0, td, 1.0)
    x = np.array([get_stim_current(tt, stype_zap, iext, t0, td, tau, f0, f1) for tt in t])

    early = x[:300]
    late = x[700:1000]
    # Number of sign changes is a rough frequency proxy on equal window lengths.
    zc_early = np.sum(np.signbit(early[:-1]) != np.signbit(early[1:]))
    zc_late = np.sum(np.signbit(late[:-1]) != np.signbit(late[1:]))

    assert zc_late > zc_early
