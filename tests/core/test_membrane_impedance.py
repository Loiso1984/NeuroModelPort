import numpy as np

from core.analysis import compute_membrane_impedance


def test_compute_membrane_impedance_detects_resonance_peak_on_sinusoid():
    dt_ms = 0.1
    t = np.arange(0.0, 2000.0, dt_ms)
    f0 = 8.0  # Hz

    i = np.sin(2 * np.pi * f0 * (t / 1000.0))
    # Linear RC-like gain/phase mock: V lags current by 30 deg and has fixed gain.
    gain = 5.0  # kOhm*cm^2 equivalent in current units
    phase = np.deg2rad(-30.0)
    v = gain * np.sin(2 * np.pi * f0 * (t / 1000.0) + phase)

    out = compute_membrane_impedance(t, v, i, fmin_hz=1.0, fmax_hz=40.0)
    assert out["valid"]
    assert abs(out["f_res_hz"] - f0) < 1.0
    assert out["z_res_kohm_cm2"] > 0.1


def test_compute_membrane_impedance_handles_insufficient_input():
    t = np.array([0.0, 1.0, 2.0])
    v = np.array([0.0, 0.1, 0.2])
    i = np.array([0.0, 0.0, 0.0])
    out = compute_membrane_impedance(t, v, i)
    assert not out["valid"]
