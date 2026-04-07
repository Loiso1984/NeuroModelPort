"""analytic_sparse_jacobian legacy cache tests — removed in v11.0.

The positional-RHS-args interface (rhs_contract.py) was deleted.
analytic_sparse_jacobian now raises NotImplementedError on any call.
Tests for the PhysicsParams-based Jacobian live in test_jacobian_contract.py.
"""
from __future__ import annotations

import pytest
from core.jacobian import analytic_sparse_jacobian


def test_analytic_sparse_jacobian_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        analytic_sparse_jacobian(0.0, [], 0, 0)
