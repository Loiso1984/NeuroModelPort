"""RHS positional-contract helpers.

This module centralizes the positional argument order for
`rhs_multicompartment(t, y, *args)` and Jacobian callbacks.

Goal: reduce silent argument-order drift risk by making packing explicit.
"""

from __future__ import annotations

from typing import Any, Mapping


RHS_ARG_ORDER: tuple[str, ...] = (
    # Model size / feature flags
    "n_comp",
    "en_ih", "en_ica", "en_ia", "en_sk", "dyn_ca", "en_itca", "en_im", "en_nap", "en_nar",
    # Conductance vectors
    "gna_v", "gk_v", "gl_v", "gih_v", "gca_v", "ga_v", "gsk_v", "gtca_v", "gim_v", "gnap_v", "gnar_v",
    # Reversal potentials
    "ena", "ek", "el", "eih", "ea",
    # Morphology + axial coupling
    "cm_v", "l_data", "l_indices", "l_indptr",
    # Per-channel temperature scaling
    "phi_na", "phi_k", "phi_ih", "phi_ca", "phi_ia", "phi_tca", "phi_im", "phi_nap", "phi_nar",
    # Environment/calcium
    "t_kelvin", "ca_ext", "ca_rest", "tau_ca", "b_ca", "mg_ext", "tau_sk",
    # Primary stimulation
    "stype", "iext", "t0", "td", "atau", "zap_f0_hz", "zap_f1_hz", "event_times_arr", "n_events", "stim_comp", "stim_mode",
    "use_dfilter_primary", "dfilter_attenuation", "dfilter_tau_ms",
    # Secondary stimulation (dual)
    "dual_stim_enabled",
    "stype_2", "iext_2", "t0_2", "td_2", "atau_2", "zap_f0_hz_2", "zap_f1_hz_2", "stim_comp_2", "stim_mode_2",
    "use_dfilter_secondary", "dfilter_attenuation_2", "dfilter_tau_ms_2",
)

RHS_ARG_INDEX: dict[str, int] = {name: i for i, name in enumerate(RHS_ARG_ORDER)}


def pack_rhs_args(values: Mapping[str, Any]) -> tuple[Any, ...]:
    """Pack named RHS/Jacobian arguments into canonical positional tuple.

    Raises
    ------
    KeyError
        If required fields are missing or unknown keys are supplied.
    """
    missing = [k for k in RHS_ARG_ORDER if k not in values]
    extra = [k for k in values.keys() if k not in RHS_ARG_INDEX]
    if missing or extra:
        chunks: list[str] = []
        if missing:
            chunks.append(f"missing={missing}")
        if extra:
            chunks.append(f"extra={extra}")
        raise KeyError("Invalid RHS args mapping: " + ", ".join(chunks))
    return tuple(values[k] for k in RHS_ARG_ORDER)
