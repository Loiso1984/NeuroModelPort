from __future__ import annotations

import numpy as np


def test_ghk_hines_linearization_matches_actual_current_at_expansion_point():
    from core.rhs import F_CONST, R_GAS, compute_ionic_conductances_scalar, ghk_current

    vi = -65.0
    ca_i = 50e-6
    ca_ext = 2.0
    t_kelvin = 310.15
    gca = 1.0
    si = 0.45
    ui = 0.80
    factor = (R_GAS * t_kelvin) / (4.0 * F_CONST * F_CONST)
    expected_i = gca * factor * (si ** 2) * ui * ghk_current(vi, ca_i, ca_ext, 2.0, t_kelvin)

    g_tot, e_eff, influx = compute_ionic_conductances_scalar(
        vi, 0.0, 0.0, 0.0,
        0.0, si, ui, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        False, True, False, False, False, False, False, False, True,
        0.0, 0.0, 0.0, 0.0, gca, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        50.0, -77.0, -54.387, -30.0, 120.0,
        ca_i, ca_ext, 50e-6, t_kelvin,
    )

    reconstructed_i = g_tot * vi - e_eff
    assert np.isclose(reconstructed_i, expected_i, rtol=1e-6, atol=1e-9)
    assert influx == max(0.0, -expected_i)


def test_fix_crit_a_native_loop_comment_contract():
    import inspect
    from core import native_loop

    source = inspect.getsource(native_loop.run_native_loop)
    assert "Moved to end of step AFTER physics" in source
    assert "i_stim_soma_count" in source
    assert "n_steps // every + 3" in source


def test_scipy_solver_attaches_precomputed_i_stim_total():
    from core.models import FullModelConfig
    from core.solver import NeuronSolver

    cfg = FullModelConfig()
    cfg.morphology.single_comp = True
    cfg.stim.jacobian_mode = "dense_fd"
    cfg.stim.stim_type = "pulse"
    cfg.stim.Iext = 5.0
    cfg.stim.pulse_start = 1.0
    cfg.stim.pulse_dur = 1.0
    cfg.stim.t_sim = 3.0
    cfg.stim.dt_eval = 0.2

    result = NeuronSolver(cfg).run_single()

    assert result.i_stim_total is not None
    assert result.i_stim_total.shape == result.t.shape
    assert result.i_stim_total[0] == 0.0
    assert result.i_stim_total.max() > 0.0


def test_post_process_marks_ghk_currents_as_ssot():
    from core.models import FullModelConfig
    from core.solver import NeuronSolver

    cfg = FullModelConfig()
    cfg.morphology.single_comp = True
    cfg.channels.enable_ICa = True
    cfg.channels.gCa_max = 0.2
    cfg.calcium.dynamic_Ca = True
    cfg.stim.t_sim = 2.0
    cfg.stim.dt_eval = 0.2
    cfg.stim.jacobian_mode = "dense_fd"

    result = NeuronSolver(cfg).run_single()

    assert result.use_ghk is True
    assert "ICa" in result.currents
    assert "ICa" in result.currents_ghk
    assert np.allclose(result.currents["ICa"], result.currents_ghk["ICa"], rtol=1e-12, atol=1e-12)
