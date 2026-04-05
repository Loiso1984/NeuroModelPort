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
RHS_ARG_COUNT: int = len(RHS_ARG_ORDER)


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


def unpack_rhs_args(args: tuple[Any, ...]) -> dict[str, Any]:
    """Unpack canonical RHS positional tuple into named mapping.

    Raises
    ------
    ValueError
        If positional argument length differs from the canonical contract.
    """
    if len(args) != RHS_ARG_COUNT:
        raise ValueError(f"Invalid RHS args length: got {len(args)}, expected {RHS_ARG_COUNT}")
    return {name: args[i] for i, name in enumerate(RHS_ARG_ORDER)}


def validate_rhs_args_values(values: Mapping[str, Any]) -> None:
    """Fail-fast structural checks for high-risk RHS positional inputs.

    This guards against silent scientific corruption caused by shape/order drift
    before values reach Numba kernels.
    """
    n_comp = int(values["n_comp"])
    if n_comp <= 0:
        raise ValueError(f"Invalid n_comp={n_comp}: must be positive")

    vector_keys = (
        "gna_v", "gk_v", "gl_v", "gih_v", "gca_v", "ga_v", "gsk_v", "gtca_v", "gim_v", "gnap_v", "gnar_v",
        "cm_v",
        "phi_na", "phi_k", "phi_ih", "phi_ca", "phi_ia", "phi_tca", "phi_im", "phi_nap", "phi_nar",
        "b_ca",
    )
    for key in vector_keys:
        arr = values[key]
        if len(arr) != n_comp:
            raise ValueError(f"Invalid {key} length={len(arr)} for n_comp={n_comp}")

    l_data = values["l_data"]
    l_indices = values["l_indices"]
    l_indptr = values["l_indptr"]
    if len(l_data) != len(l_indices):
        raise ValueError(
            f"Invalid Laplacian sparse vectors: len(l_data)={len(l_data)} != len(l_indices)={len(l_indices)}"
        )
    if len(l_indptr) != n_comp + 1:
        raise ValueError(f"Invalid l_indptr length={len(l_indptr)} for n_comp={n_comp} (expected n_comp+1)")
    if len(l_indptr) > 0:
        if int(l_indptr[0]) != 0:
            raise ValueError("Invalid l_indptr: first element must be 0")
        if int(l_indptr[-1]) != len(l_indices):
            raise ValueError(
                f"Invalid l_indptr tail={int(l_indptr[-1])}: expected len(l_indices)={len(l_indices)}"
            )
    prev = 0
    for k in range(len(l_indptr)):
        cur = int(l_indptr[k])
        if cur < prev:
            raise ValueError("Invalid l_indptr: must be non-decreasing")
        prev = cur
    for idx in l_indices:
        if int(idx) < 0 or int(idx) >= n_comp:
            raise ValueError(f"Invalid l_indices entry={int(idx)} for n_comp={n_comp}")

    n_events = int(values["n_events"])
    event_times_arr = values["event_times_arr"]
    if n_events < 0:
        raise ValueError(f"Invalid n_events={n_events}: must be non-negative")
    if n_events > len(event_times_arr):
        raise ValueError(
            f"Invalid n_events={n_events}: exceeds len(event_times_arr)={len(event_times_arr)}"
        )

    stim_mode = int(values["stim_mode"])
    stim_mode_2 = int(values["stim_mode_2"])
    if stim_mode not in (0, 1, 2):
        raise ValueError(f"Invalid stim_mode={stim_mode}: expected 0|1|2")
    if stim_mode_2 not in (0, 1, 2):
        raise ValueError(f"Invalid stim_mode_2={stim_mode_2}: expected 0|1|2")

    stim_comp = int(values["stim_comp"])
    stim_comp_2 = int(values["stim_comp_2"])
    if not (0 <= stim_comp < n_comp):
        raise ValueError(f"Invalid stim_comp={stim_comp} for n_comp={n_comp}")
    if not (0 <= stim_comp_2 < n_comp):
        raise ValueError(f"Invalid stim_comp_2={stim_comp_2} for n_comp={n_comp}")

    if float(values["dfilter_tau_ms"]) < 0.0:
        raise ValueError("Invalid dfilter_tau_ms: must be >= 0")
    if float(values["dfilter_tau_ms_2"]) < 0.0:
        raise ValueError("Invalid dfilter_tau_ms_2: must be >= 0")
