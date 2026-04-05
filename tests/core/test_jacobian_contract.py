from __future__ import annotations

import numpy as np
from core.jacobian import analytic_sparse_jacobian
from core.rhs_contract import RHS_ARG_COUNT


def test_analytic_sparse_jacobian_rejects_incomplete_rhs_args():
    t = 0.0
    y = np.zeros(4, dtype=float)

    try:
        analytic_sparse_jacobian(t, y, *([0] * (RHS_ARG_COUNT - 1)))
        assert False, "Expected ValueError for incomplete RHS positional args"
    except ValueError as exc:
        assert "expected" in str(exc).lower()
