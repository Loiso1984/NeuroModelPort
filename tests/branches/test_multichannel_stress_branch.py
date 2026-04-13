"""
Stress validation of multi-channel interactions.

Scenarios:
1) Ih + ICa
2) IA + SK
3) All optional channels enabled
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.solver import NeuronSolver
from tests.shared_utils import _spike_times


def _run_case(cfg: FullModelConfig) -> dict:
    res = NeuronSolver(cfg).run_single()
    st = _spike_times(res.v_soma, res.t)
    out = {
        "n_spikes": int(len(st)),
        "v_peak": float(np.max(res.v_soma)),
        "v_min": float(np.min(res.v_soma)),
        "v_tail": float(np.mean(res.v_soma[-80:])),
        "finite": bool(np.all(np.isfinite(res.v_soma))),
        "ca_peak_nM": float(np.max(res.ca_i[0, :]) * 1e6) if res.ca_i is not None else None,
        "ca_min_nM": float(np.min(res.ca_i[0, :]) * 1e6) if res.ca_i is not None else None,
    }
    return out


def _assert_physio_guards(row: dict, label: str) -> None:
    assert row["finite"], f"{label}: non-finite voltage trace"
    assert -130.0 < row["v_min"] < 90.0, f"{label}: v_min out of guard range ({row['v_min']:.2f} mV)"
    assert -130.0 < row["v_peak"] < 90.0, f"{label}: v_peak out of guard range ({row['v_peak']:.2f} mV)"
    if row["ca_peak_nM"] is not None:
        assert row["ca_min_nM"] >= 0.0, f"{label}: negative Ca_i"
        assert row["ca_peak_nM"] <= 5000.0, f"{label}: Ca_i overload ({row['ca_peak_nM']:.1f} nM)"


def test_stress_ih_ica_interaction():
    spike_counts = []
    for g_ih in [0.01, 0.02, 0.03, 0.04]:
        for g_ca in [0.04, 0.08, 0.12]:
            for iext in [8.0, 14.0, 22.0]:
                cfg = FullModelConfig()
                cfg.morphology.single_comp = True
                cfg.stim_location.location = "soma"
                cfg.dendritic_filter.enabled = False
                cfg.stim.jacobian_mode = "native_hines"
                cfg.stim.stim_type = "const"
                cfg.stim.Iext = iext
                cfg.stim.t_sim = 180.0
                cfg.stim.dt_eval = 0.3

                cfg.channels.enable_Ih = True
                cfg.channels.gIh_max = g_ih
                cfg.channels.enable_ICa = True
                cfg.channels.gCa_max = g_ca
                cfg.channels.enable_IA = False
                cfg.channels.enable_SK = False
                cfg.calcium.dynamic_Ca = True
                cfg.calcium.B_Ca = 1e-5
                cfg.calcium.tau_Ca = 250.0

                row = _run_case(cfg)
                _assert_physio_guards(row, f"Ih+ICa(gIh={g_ih},gCa={g_ca},I={iext})")
                spike_counts.append(row["n_spikes"])

    assert max(spike_counts) >= 1, "Ih+ICa sweep should contain at least one spiking regime"


def test_stress_ia_sk_interaction():
    spike_counts = []
    for g_a in [0.2, 0.4, 0.8]:
        for g_sk in [0.5, 1.0, 2.0]:
            for iext in [8.0, 14.0, 22.0]:
                cfg = FullModelConfig()
                cfg.morphology.single_comp = True
                cfg.stim_location.location = "soma"
                cfg.dendritic_filter.enabled = False
                cfg.stim.jacobian_mode = "native_hines"
                cfg.stim.stim_type = "const"
                cfg.stim.Iext = iext
                cfg.stim.t_sim = 200.0
                cfg.stim.dt_eval = 0.3

                cfg.channels.enable_Ih = False
                cfg.channels.enable_ICa = True
                cfg.channels.gCa_max = 0.08
                cfg.channels.enable_IA = True
                cfg.channels.gA_max = g_a
                cfg.channels.enable_SK = True
                cfg.channels.gSK_max = g_sk
                cfg.calcium.dynamic_Ca = True
                cfg.calcium.B_Ca = 1e-5
                cfg.calcium.tau_Ca = 250.0

                row = _run_case(cfg)
                _assert_physio_guards(row, f"IA+SK(gA={g_a},gSK={g_sk},I={iext})")
                spike_counts.append(row["n_spikes"])

    assert max(spike_counts) >= 1, "IA+SK sweep should contain at least one spiking regime"


def test_stress_all_channels_enabled():
    spike_counts = []
    for iext in [8.0, 14.0, 20.0]:
        for g_scale in [0.75, 1.0, 1.25]:
            cfg = FullModelConfig()
            cfg.morphology.single_comp = True
            cfg.stim_location.location = "soma"
            cfg.dendritic_filter.enabled = False
            cfg.stim.jacobian_mode = "native_hines"
            cfg.stim.stim_type = "const"
            cfg.stim.Iext = iext
            cfg.stim.t_sim = 180.0
            cfg.stim.dt_eval = 0.3

            cfg.channels.enable_Ih = True
            cfg.channels.gIh_max = 0.02 * g_scale
            cfg.channels.enable_ICa = True
            cfg.channels.gCa_max = 0.08 * g_scale
            cfg.channels.enable_IA = True
            cfg.channels.gA_max = 0.4 * g_scale
            cfg.channels.enable_SK = True
            cfg.channels.gSK_max = 1.0 * g_scale
            cfg.calcium.dynamic_Ca = True
            cfg.calcium.B_Ca = 1e-5
            cfg.calcium.tau_Ca = 250.0

            row = _run_case(cfg)
            _assert_physio_guards(row, f"ALL(g={g_scale},I={iext})")
            spike_counts.append(row["n_spikes"])

    assert max(spike_counts) >= 1, "All-channel sweep should include spiking cases"


def _run_as_script() -> int:
    tests = [
        test_stress_ih_ica_interaction,
        test_stress_ia_sk_interaction,
        test_stress_all_channels_enabled,
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
