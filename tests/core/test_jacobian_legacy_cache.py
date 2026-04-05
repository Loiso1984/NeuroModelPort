from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pydantic")

import core.jacobian as jac
from core.rhs_contract import RHS_ARG_ORDER


def _make_rhs_args_for_legacy_call(*, l_indices=None, l_indptr=None):
    n = 2
    vals = {name: 0 for name in RHS_ARG_ORDER}
    vals.update({
        "n_comp": n,
        "en_ih": False,
        "en_ica": False,
        "en_ia": False,
        "en_sk": False,
        "dyn_ca": False,
        "en_itca": False,
        "en_im": False,
        "en_nap": False,
        "en_nar": False,
        "l_indices": np.array([0, 1] if l_indices is None else l_indices, dtype=np.int64),
        "l_indptr": np.array([0, 1, 2] if l_indptr is None else l_indptr, dtype=np.int64),
    })
    return tuple(vals[name] for name in RHS_ARG_ORDER)


def test_analytic_sparse_jacobian_caches_legacy_callable(monkeypatch):
    jac.clear_legacy_jacobian_cache()
    calls = {"build": 0, "make": 0, "fn": 0}

    def fake_build(*args, **kwargs):
        calls["build"] += 1
        return "SP"

    def fake_make(sp):
        calls["make"] += 1

        def _fn(*args, **kwargs):
            calls["fn"] += 1
            return "J"

        return _fn

    monkeypatch.setattr(jac, "build_jacobian_sparsity", fake_build)
    monkeypatch.setattr(jac, "make_analytic_jacobian", fake_make)

    t = 0.0
    y = np.zeros(2, dtype=float)
    rhs_args = _make_rhs_args_for_legacy_call()

    out1 = jac.analytic_sparse_jacobian(t, y, *rhs_args)
    out2 = jac.analytic_sparse_jacobian(t, y, *rhs_args)

    assert out1 == "J"
    assert out2 == "J"
    assert calls["build"] == 1
    assert calls["make"] == 1
    assert calls["fn"] == 2


def test_analytic_sparse_jacobian_cache_key_tracks_sparse_topology(monkeypatch):
    jac.clear_legacy_jacobian_cache()
    calls = {"build": 0}

    def fake_build(*args, **kwargs):
        calls["build"] += 1
        return f"SP-{calls['build']}"

    def fake_make(sp):
        def _fn(*args, **kwargs):
            return sp
        return _fn

    monkeypatch.setattr(jac, "build_jacobian_sparsity", fake_build)
    monkeypatch.setattr(jac, "make_analytic_jacobian", fake_make)

    t = 0.0
    y = np.zeros(2, dtype=float)
    rhs_a = _make_rhs_args_for_legacy_call(l_indices=[0, 1], l_indptr=[0, 1, 2])
    rhs_b = _make_rhs_args_for_legacy_call(l_indices=[0, 0], l_indptr=[0, 1, 2])

    out_a1 = jac.analytic_sparse_jacobian(t, y, *rhs_a)
    out_a2 = jac.analytic_sparse_jacobian(t, y, *rhs_a)
    out_b = jac.analytic_sparse_jacobian(t, y, *rhs_b)

    assert out_a1 == "SP-1"
    assert out_a2 == "SP-1"
    assert out_b == "SP-2"
    assert calls["build"] == 2
