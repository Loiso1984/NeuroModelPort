"""
Deterministic robustness sweep for pathology mode behavior (N/O).

Goal:
- validate progressive/terminal mode separation under parameter perturbations,
- keep checks physiological and numerically stable,
- avoid random sampling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset
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


def _run_case(
    preset: str,
    mode: str,
    perturb: tuple[float, float, float, float],
    *,
    t_sim: float = 260.0,
    dt_eval: float = 0.25,
) -> dict:
    cfg = FullModelConfig()
    if preset.startswith("N:"):
        cfg.preset_modes.alzheimer_mode = mode
    else:
        cfg.preset_modes.hypoxia_mode = mode
    apply_preset(cfg, preset)

    i_mult, gl_mult, tau_mult, gca_mult = perturb
    cfg.stim.Iext *= i_mult
    cfg.channels.gL *= gl_mult
    if cfg.calcium.dynamic_Ca:
        cfg.calcium.tau_Ca *= tau_mult
        cfg.channels.gCa_max *= gca_mult

    cfg.stim.t_sim = t_sim
    cfg.stim.dt_eval = dt_eval
    cfg.stim.jacobian_mode = "native_hines"

    res = NeuronSolver(cfg).run_single()
    st = _spike_times(res.v_soma, res.t)
    mid = 0.5 * t_sim
    first = int(np.sum(st < mid))
    second = int(np.sum(st >= mid))
    return {
        "n_spikes": int(len(st)),
        "first_half": first,
        "second_half": second,
        "v_peak": float(np.max(res.v_soma)),
        "v_tail": float(np.mean(res.v_soma[-80:])),
        "stable": bool(np.all(np.isfinite(res.v_soma))),
    }


def _evaluate_mode_pair(preset: str) -> tuple[list[dict], list[dict]]:
    # Deterministic perturbation set around defaults.
    perturbations = [
        (0.85, 0.85, 0.85, 0.85),
        (0.85, 1.00, 1.00, 1.00),
        (1.00, 0.85, 1.00, 1.00),
        (1.00, 1.00, 0.85, 1.00),
        (1.00, 1.00, 1.00, 0.85),
        (1.15, 1.00, 1.00, 1.00),
        (1.00, 1.15, 1.00, 1.00),
        (1.15, 1.15, 1.15, 1.15),
    ]
    progressive = [_run_case(preset, "progressive", p) for p in perturbations]
    terminal = [_run_case(preset, "terminal", p) for p in perturbations]
    return progressive, terminal


def test_pathology_modes_robustness_n_o():
    for preset in [
        "N: Alzheimer's (v10 Calcium Toxicity)",
        "O: Hypoxia (v10 ATP-pump failure)",
    ]:
        prog_rows, term_rows = _evaluate_mode_pair(preset)

        # Numerical and broad physiological guards.
        for row in prog_rows + term_rows:
            assert row["stable"], f"{preset}: non-finite trace in mode sweep"
            assert -120.0 < row["v_peak"] < 70.0, f"{preset}: v_peak out of guard range ({row['v_peak']:.2f})"
            assert -120.0 < row["v_tail"] < 5.0, f"{preset}: v_tail out of guard range ({row['v_tail']:.2f})"

        prog_first = float(np.mean([r["first_half"] for r in prog_rows]))
        prog_second = float(np.mean([r["second_half"] for r in prog_rows]))
        term_n = float(np.mean([r["n_spikes"] for r in term_rows]))

        # Progressive should show early activity with attenuation.
        if preset.startswith("N:"):
            assert prog_first >= 2.0, f"{preset}: progressive early activity too weak ({prog_first:.2f})"
        else:
            assert prog_first >= 1.8, f"{preset}: progressive early activity too weak ({prog_first:.2f})"
        assert prog_second <= prog_first, (
            f"{preset}: progressive should attenuate later activity ({prog_first:.2f} vs {prog_second:.2f})"
        )

        # Terminal should be clearly less excitable on average.
        assert term_n <= 1.5, f"{preset}: terminal mode too excitable ({term_n:.2f} spikes avg)"

        pairwise_ok = sum(
            1 for p, t in zip(prog_rows, term_rows)
            if t["n_spikes"] <= p["n_spikes"]
        )
        assert pairwise_ok >= 6, (
            f"{preset}: terminal should be <= progressive in most perturbations "
            f"(ok={pairwise_ok}/8)"
        )


def _run_as_script() -> int:
    tests = [test_pathology_modes_robustness_n_o]
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
