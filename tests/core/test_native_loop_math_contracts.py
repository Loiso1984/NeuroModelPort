from __future__ import annotations

import inspect


def test_conductance_synapse_rhs_uses_matrix_consistent_form():
    from core import native_loop

    source = inspect.getsource(native_loop.run_native_loop)
    assert "rhs_stim_add = 0.0" in source
    assert "rhs_stim_add = g_syn * e_syn" in source
    assert "rhs_stim_add += g2 * e_syn_2" in source
    assert "rhs[i] = cm_over_dt * vi + e_eff_arr[i] + katp_rhs + rhs_stim_add - i_pump" in source
    assert "rhs[i] = cm_over_dt * vi + e_eff_arr[i] + katp_rhs + i_stim_eff - i_pump" not in source
