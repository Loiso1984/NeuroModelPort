from __future__ import annotations

import pytest
from core.jacobian import analytic_sparse_jacobian


def test_analytic_sparse_jacobian_raises_not_implemented():
    """analytic_sparse_jacobian was removed in v11.0 — must raise NotImplementedError."""
    with pytest.raises(NotImplementedError):
        analytic_sparse_jacobian(0.0, [])
