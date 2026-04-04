from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.channels import ChannelRegistry
from core.jacobian import analytic_sparse_jacobian
from core.models import FullModelConfig
from core.morphology import MorphologyBuilder
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
    morph = MorphologyBuilder.build(cfg)
    n_comp = morph["N_comp"]
    y0 = ChannelRegistry().compute_initial_states(-65.0, cfg)

    s_map = {
        "const": 0,
        "pulse": 1,
        "alpha": 2,
        "ou_noise": 3,
        "AMPA": 4,
        "NMDA": 5,
        "GABAA": 6,
        "GABAB": 7,
        "Kainate": 8,
        "Nicotinic": 9,
    }
    stype = s_map.get(cfg.stim.stim_type, 0)
    t_kelvin = cfg.env.T_celsius + 273.15

    stim_mode_map = {"soma": 0, "ais": 1, "dendritic_filtered": 2}
    stim_mode = stim_mode_map.get(cfg.stim_location.location, 0)
    use_dfilter = int(stim_mode == 2 and cfg.dendritic_filter.enabled)
    if use_dfilter == 1:
        y0 = np.concatenate([y0, np.array([0.0])])

    attenuation = 1.0
    if use_dfilter == 1 and cfg.dendritic_filter.space_constant_um > 0:
        attenuation = np.exp(
            -cfg.dendritic_filter.distance_um / cfg.dendritic_filter.space_constant_um
        )

    dual_stim_enabled = 0
    stype_2, iext_2, t0_2, td_2, atau_2, stim_comp_2, stim_mode_2 = (
        0,
        0.0,
        0.0,
        0.0,
        0.0,
        0,
        0,
    )
    dfilter_attenuation_2, dfilter_tau_ms_2 = 1.0, 0.0
    if cfg.dual_stimulation is not None and getattr(cfg.dual_stimulation, "enabled", False):
        dual_stim_enabled = 1
        stype_2 = s_map.get(cfg.dual_stimulation.secondary_stim_type, 0)
        iext_2 = cfg.dual_stimulation.secondary_Iext
        t0_2 = cfg.dual_stimulation.secondary_start
        td_2 = cfg.dual_stimulation.secondary_duration
        atau_2 = cfg.dual_stimulation.secondary_alpha_tau
        stim_mode_2 = stim_mode_map.get(cfg.dual_stimulation.secondary_location, 0)
        if stim_mode_2 == 2 and cfg.dual_stimulation.secondary_space_constant_um > 0:
            dfilter_attenuation_2 = np.exp(
                -cfg.dual_stimulation.secondary_distance_um
                / cfg.dual_stimulation.secondary_space_constant_um
            )
            dfilter_tau_ms_2 = cfg.dual_stimulation.secondary_tau_dendritic_ms

    args = (
        n_comp,
        cfg.channels.enable_Ih,
        cfg.channels.enable_ICa,
        cfg.channels.enable_IA,
        cfg.channels.enable_SK,
        cfg.calcium.dynamic_Ca,
        morph["gNa_v"],
        morph["gK_v"],
        morph["gL_v"],
        morph["gIh_v"],
        morph["gCa_v"],
        morph["gA_v"],
        morph["gSK_v"],
        cfg.channels.ENa,
        cfg.channels.EK,
        cfg.channels.EL,
        cfg.channels.E_Ih,
        cfg.channels.E_A,
        morph["Cm_v"],
        morph["L_data"],
        morph["L_indices"],
        morph["L_indptr"],
        cfg.env.phi,
        t_kelvin,
        cfg.calcium.Ca_ext,
        cfg.calcium.Ca_rest,
        cfg.calcium.tau_Ca,
        cfg.calcium.B_Ca,
        stype,
        cfg.stim.Iext,
        cfg.stim.pulse_start,
        cfg.stim.pulse_dur,
        cfg.stim.alpha_tau,
        cfg.stim.stim_comp,
        stim_mode,
        use_dfilter,
        attenuation,
        cfg.dendritic_filter.tau_dendritic_ms if use_dfilter == 1 else 0.0,
        dual_stim_enabled,
        stype_2,
        iext_2,
        t0_2,
        td_2,
        atau_2,
        stim_comp_2,
        stim_mode_2,
        dfilter_attenuation_2,
        dfilter_tau_ms_2,
    )
    return y0, args


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

    y0, args = _solver_args_from_cfg(cfg)

    # JIT warmup for RHS before finite-difference loop.
    rhs_multicompartment(0.0, y0, *args)
    j_an = analytic_sparse_jacobian(0.0, y0, *args).toarray()

    eps = 1e-6
    n_state = len(y0)
    j_fd = np.zeros((n_state, n_state), dtype=float)
    for col in range(n_state):
        yp = y0.copy()
        ym = y0.copy()
        yp[col] += eps
        ym[col] -= eps
        fp = rhs_multicompartment(0.0, yp, *args)
        fm = rhs_multicompartment(0.0, ym, *args)
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
