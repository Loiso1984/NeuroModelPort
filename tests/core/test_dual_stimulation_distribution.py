from __future__ import annotations

import pytest

pytest.importorskip("pydantic")

from core.dual_stimulation import distributed_stimulus_current_for_comp


def test_distributed_stimulus_current_for_comp_soma_mode():
    vals = [
        distributed_stimulus_current_for_comp(
            comp_idx=i,
            n_comp=4,
            base_current=3.0,
            stim_comp=2,
            stim_mode=0,
            use_dfilter=0,
            dfilter_attenuation=1.0,
            dfilter_tau_ms=10.0,
            v_filtered=0.0,
        )
        for i in range(4)
    ]
    assert vals == [0.0, 0.0, 3.0, 0.0]


def test_distributed_stimulus_current_for_comp_ais_mode():
    vals = [
        distributed_stimulus_current_for_comp(
            comp_idx=i,
            n_comp=4,
            base_current=2.5,
            stim_comp=0,
            stim_mode=1,
            use_dfilter=0,
            dfilter_attenuation=1.0,
            dfilter_tau_ms=10.0,
            v_filtered=0.0,
        )
        for i in range(4)
    ]
    assert vals == [0.0, 2.5, 0.0, 0.0]


def test_distributed_stimulus_current_for_comp_dfilter_mode():
    vals_no_filter = [
        distributed_stimulus_current_for_comp(
            comp_idx=i,
            n_comp=3,
            base_current=4.0,
            stim_comp=0,
            stim_mode=2,
            use_dfilter=0,
            dfilter_attenuation=0.25,
            dfilter_tau_ms=10.0,
            v_filtered=0.0,
        )
        for i in range(3)
    ]
    assert vals_no_filter == [1.0, 0.0, 0.0]

    vals_filter = [
        distributed_stimulus_current_for_comp(
            comp_idx=i,
            n_comp=3,
            base_current=4.0,
            stim_comp=0,
            stim_mode=2,
            use_dfilter=1,
            dfilter_attenuation=0.25,
            dfilter_tau_ms=10.0,
            v_filtered=0.33,
        )
        for i in range(3)
    ]
    assert vals_filter == [0.33, 0.0, 0.0]


def test_distributed_stimulus_current_for_comp_tau_nonpositive_disables_filter():
    vals_tau_zero = [
        distributed_stimulus_current_for_comp(
            comp_idx=i,
            n_comp=3,
            base_current=4.0,
            stim_comp=0,
            stim_mode=2,
            use_dfilter=1,
            dfilter_attenuation=0.25,
            dfilter_tau_ms=0.0,
            v_filtered=0.33,
        )
        for i in range(3)
    ]
    assert vals_tau_zero == [1.0, 0.0, 0.0]
