from __future__ import annotations

import numpy as np

from core.unit_converter import density_to_absolute_current


def test_density_to_absolute_current_matches_reference_formula():
    soma_d_um = 29.0
    soma_area_cm2 = np.pi * (soma_d_um * 1e-4) ** 2
    i_density = 10.0  # µA/cm²
    expected_na = i_density * soma_area_cm2 * 1000.0

    got = density_to_absolute_current(i_density, soma_area_cm2)

    assert np.isclose(got, expected_na, rtol=0.0, atol=1e-12)


def test_density_to_absolute_current_preserves_sign():
    soma_area_cm2 = np.pi * (20e-4) ** 2
    assert density_to_absolute_current(-5.0, soma_area_cm2) < 0.0
