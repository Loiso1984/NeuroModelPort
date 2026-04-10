from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.channels import ChannelRegistry
from core.jacobian import build_jacobian_sparsity, make_analytic_jacobian
from core.models import FullModelConfig
from core.morphology import MorphologyBuilder
from core.physics_params import create_physics_params
from core.presets import apply_preset
from core.rhs import rhs_multicompartment
from core.solver import NeuronSolver


def _spike_times(v: np.ndarray, t: np.ndarray, threshold: float = -20.0) -> np.ndarray:
    idx = np.where((v[:-1] < threshold) & (v[1:] >= threshold))[0] + 1
    if len(idx) == 0:
        return np.array([], dtype=float)
    st = t[idx]
    keep = [0]
    for i in range(1, len(st)):
        if st[i] - st[keep[-1]] >= 1.0:
            keep.append(i)
    return st[keep]


def _solver_args_from_cfg(cfg: FullModelConfig):
    """Build (y0, args, dydt_buf) using the canonical pack_rhs_args contract.

    Mirrors solver.py NeuronSolver.run_single arg construction so that direct
    calls to rhs_multicompartment in tests use the same positional order as
    the live solver path.  Returns a pre-allocated dydt buffer sized to y0.
    """
    morph = MorphologyBuilder.build(cfg)
    n_comp = morph["N_comp"]
    y0 = ChannelRegistry().compute_initial_states(-65.0, cfg)

    s_map = {
        "const": 0, "pulse": 1, "alpha": 2, "ou_noise": 3,
        "AMPA": 4, "NMDA": 5, "GABAA": 6, "GABAB": 7,
        "Kainate": 8, "Nicotinic": 9, "zap": 10,
    }
    stim_mode_map = {"soma": 0, "ais": 1, "dendritic_filtered": 2}

    t_kelvin = cfg.env.T_celsius + 273.15
    stype = s_map.get(cfg.stim.stim_type, 0)
    stim_mode = stim_mode_map.get(cfg.stim_location.location, 0)
    use_dfilter = int(
        stim_mode == 2
        and cfg.dendritic_filter.enabled
        and cfg.dendritic_filter.tau_dendritic_ms > 0.0
    )
    if use_dfilter == 1:
        y0 = np.concatenate([y0, np.array([0.0])])

    attenuation = 1.0
    if stim_mode == 2 and cfg.dendritic_filter.space_constant_um > 0:
        attenuation = np.exp(
            -cfg.dendritic_filter.distance_um / cfg.dendritic_filter.space_constant_um
        )

    # Per-compartment B_Ca (surface/volume ratio scaling — mirrors solver._build_b_ca_vector)
    b_ca_v = NeuronSolver._build_b_ca_vector(cfg, morph)

    rhs_values = {
        "n_comp": n_comp,
        "en_ih": cfg.channels.enable_Ih,
        "en_ica": cfg.channels.enable_ICa,
        "en_ia": cfg.channels.enable_IA,
        "en_sk": cfg.channels.enable_SK,
        "dyn_ca": cfg.calcium.dynamic_Ca,
        "en_itca": cfg.channels.enable_ITCa,
        "en_im": cfg.channels.enable_IM,
        "en_nap": cfg.channels.enable_NaP,
        "en_nar": cfg.channels.enable_NaR,
        "gbar_mat": np.vstack([
            morph["gNa_v"], morph["gK_v"], morph["gL_v"], morph["gIh_v"], morph["gCa_v"],
            morph["gA_v"], morph["gSK_v"], morph["gTCa_v"], morph["gIM_v"], morph["gNaP_v"], morph["gNaR_v"],
        ]),
        "ena": cfg.channels.ENa, "ek": cfg.channels.EK, "el": cfg.channels.EL,
        "eih": cfg.channels.E_Ih, "ea": cfg.channels.E_A,
        "cm_v": morph["Cm_v"], "l_data": morph["L_data"],
        "l_indices": morph["L_indices"], "l_indptr": morph["L_indptr"],
        "phi_mat": np.vstack([
            cfg.env.build_phi_vector(cfg.env.Q10_Na, n_comp),
            cfg.env.build_phi_vector(cfg.env.Q10_K, n_comp),
            cfg.env.build_phi_vector(cfg.env.Q10_Ih, n_comp),
            cfg.env.build_phi_vector(cfg.env.Q10_Ca, n_comp),
            cfg.env.build_phi_vector(cfg.env.Q10_IA, n_comp),
            cfg.env.build_phi_vector(cfg.env.Q10_TCa, n_comp),
            cfg.env.build_phi_vector(cfg.env.Q10_IM, n_comp),
            cfg.env.build_phi_vector(cfg.env.Q10_NaP, n_comp),
            cfg.env.build_phi_vector(cfg.env.Q10_NaR, n_comp),
        ]),
        "env_vec": np.array([
            t_kelvin, cfg.calcium.Ca_ext, cfg.calcium.Ca_rest, cfg.calcium.tau_Ca, cfg.env.Mg_ext, cfg.channels.tau_SK,
        ], dtype=np.float64),
        "b_ca": b_ca_v,
        "stim1_vec": np.array([
            stype, cfg.stim.Iext, cfg.stim.pulse_start, cfg.stim.pulse_dur, cfg.stim.alpha_tau,
            cfg.stim.zap_f0_hz, cfg.stim.zap_f1_hz, cfg.stim.stim_comp, stim_mode,
            use_dfilter, attenuation, cfg.dendritic_filter.tau_dendritic_ms,
        ], dtype=np.float64),
        "event_times_arr": np.array(cfg.stim.event_times or [], dtype=np.float64),
        "n_events": int(len(cfg.stim.event_times or [])),
        "stim2_vec": np.array([
            0, 0, 0.0, 0.0, 0.0, 1.0, cfg.stim.zap_f0_hz, cfg.stim.zap_f1_hz,
            0, 0, 0, 1.0, 0.0,
        ], dtype=np.float64),
    }
    # Create PhysicsParams from rhs_values dictionary
    physics_params = create_physics_params(**rhs_values)
    dydt_buf = np.empty(len(y0), dtype=np.float64)
    return y0, physics_params, dydt_buf


@pytest.mark.skip(reason="Test rewrite pending: needs sparsity construction from config")
def test_analytic_sparse_jacobian_matches_fd_on_single_comp():
    cfg = FullModelConfig()
    cfg.morphology.single_comp = True
    cfg.stim_location.location = "soma"
    cfg.dendritic_filter.enabled = False
    cfg.channels.enable_Ih = True
    cfg.channels.enable_ICa = True
    cfg.channels.enable_IA = True
    cfg.channels.enable_SK = True
    cfg.calcium.dynamic_Ca = True
    cfg.stim.stim_type = "const"
    cfg.stim.Iext = 8.0

    y0, physics_params, dydt_buf = _solver_args_from_cfg(cfg)
    morph = MorphologyBuilder.build(cfg)
    
    # Build sparsity pattern
    sparsity = build_jacobian_sparsity(
        n_comp=physics_params.n_comp,
        en_ih=physics_params.en_ih,
        en_ica=physics_params.en_ica,
        en_ia=physics_params.en_ia,
        en_sk=physics_params.en_sk,
        dyn_ca=physics_params.dyn_ca,
        l_indices=physics_params.l_indices,
        l_indptr=physics_params.l_indptr,
        use_dfilter_primary=physics_params.use_dfilter_primary,
        use_dfilter_secondary=physics_params.use_dfilter_secondary,
        en_itca=physics_params.en_itca,
        en_im=physics_params.en_im,
        en_nap=physics_params.en_nap,
        en_nar=physics_params.en_nar,
    )
    
    # Create Jacobian function
    jac_fn = make_analytic_jacobian(sparsity)

    # JIT warmup for RHS before finite-difference loop.
    rhs_multicompartment(0.0, y0, physics_params, dydt_buf)
    j_an = jac_fn(0.0, y0, physics_params).toarray()

    eps = 1e-6
    n_state = len(y0)
    j_fd = np.zeros((n_state, n_state), dtype=float)
    for col in range(n_state):
        yp = y0.copy()
        ym = y0.copy()
        yp[col] += eps
        ym[col] -= eps
        fp = rhs_multicompartment(0.0, yp, physics_params, dydt_buf).copy()
        fm = rhs_multicompartment(0.0, ym, physics_params, dydt_buf).copy()
        j_fd[:, col] = (fp - fm) / (2.0 * eps)

    rel_err = np.linalg.norm(j_an - j_fd) / max(np.linalg.norm(j_fd), 1e-12)
    assert rel_err < 0.15, f"Analytic sparse Jacobian mismatch too large: rel_err={rel_err:.4f}"


def test_jacobian_modes_preserve_main_spiking_behavior():
    preset = "K: Thalamic Relay (Ih + ICa + Burst)"
    rows = {}
    for mode in ["dense_fd", "sparse_fd", "analytic_sparse"]:
        cfg = FullModelConfig()
        apply_preset(cfg, preset)
        cfg.stim.t_sim = 140.0
        cfg.stim.dt_eval = 0.3
        cfg.stim.jacobian_mode = mode
        res = NeuronSolver(cfg).run_single()
        st = _spike_times(res.v_soma, res.t)
        rows[mode] = {
            "n_spikes": int(len(st)),
            "v_peak": float(np.max(res.v_soma)),
            "v_tail": float(np.mean(res.v_soma[-60:])),
        }
        assert np.all(np.isfinite(res.v_soma)), f"{mode}: non-finite voltage"

    dense = rows["dense_fd"]
    sparse = rows["sparse_fd"]
    analytic = rows["analytic_sparse"]

    assert abs(sparse["n_spikes"] - dense["n_spikes"]) <= 1
    assert abs(analytic["n_spikes"] - dense["n_spikes"]) <= 2
    assert abs(sparse["v_peak"] - dense["v_peak"]) <= 6.0
    assert abs(analytic["v_peak"] - dense["v_peak"]) <= 8.0


def _run_as_script() -> int:
    tests = [
        test_analytic_sparse_jacobian_matches_fd_on_single_comp,
        test_jacobian_modes_preserve_main_spiking_behavior,
    ]
    passed = 0
    for fn in tests:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] {fn.__name__}: {exc}")
    print(f"\nSummary: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(_run_as_script())
