"""RHS positional-contract helpers.

This module centralizes the positional argument order for
`rhs_multicompartment(t, y, *args)` and Jacobian callbacks.

Goal: reduce silent argument-order drift risk by making packing explicit.
"""

from __future__ import annotations

from typing import Any, Mapping
import math


RHS_ARG_ORDER: tuple[str, ...] = (
    # Model size / feature flags
    "n_comp",
    "en_ih", "en_ica", "en_ia", "en_sk", "dyn_ca", "en_itca", "en_im", "en_nap", "en_nar",
    # Packed vectors
    "gbar_mat",
    # Reversal potentials
    "ena", "ek", "el", "eih", "ea",
    # Morphology + axial coupling
    "cm_v", "l_data", "l_indices", "l_indptr",
    # Packed per-channel temperature scaling
    "phi_mat",
    # Environment/calcium
    "t_kelvin", "ca_ext", "ca_rest", "tau_ca", "b_ca", "mg_ext", "tau_sk",
    # Primary stimulation
    "stype", "iext", "t0", "td", "atau", "zap_f0_hz", "zap_f1_hz", "event_times_arr", "n_events", "stim_comp", "stim_mode",
    "use_dfilter_primary", "dfilter_attenuation", "dfilter_tau_ms",
    # Secondary stimulation (dual)
    "dual_stim_enabled",
    "stype_2", "iext_2", "t0_2", "td_2", "atau_2", "zap_f0_hz_2", "zap_f1_hz_2", "stim_comp_2", "stim_mode_2",
    "use_dfilter_secondary", "dfilter_attenuation_2", "dfilter_tau_ms_2",
    # Stochastic parameters
    "stoch_gating", "noise_sigma", "rng_state",
)

RHS_ARG_INDEX: dict[str, int] = {name: i for i, name in enumerate(RHS_ARG_ORDER)}
RHS_ARG_COUNT: int = len(RHS_ARG_ORDER)


def _as_float(values: Mapping[str, Any], key: str) -> float:
    return float(values[key])


def _require_finite(values: Mapping[str, Any], key: str) -> float:
    v = _as_float(values, key)
    if not math.isfinite(v):
        raise ValueError(f"Invalid {key}: must be finite")
    return v


def _require_positive(values: Mapping[str, Any], key: str) -> float:
    v = _require_finite(values, key)
    if v <= 0.0:
        raise ValueError(f"Invalid {key}: must be finite and > 0")
    return v


def _require_nonnegative(values: Mapping[str, Any], key: str) -> float:
    v = _require_finite(values, key)
    if v < 0.0:
        raise ValueError(f"Invalid {key}: must be finite and >= 0")
    return v


def _require_binary_flag(values: Mapping[str, Any], key: str) -> int:
    v = int(values[key])
    if v not in (0, 1):
        raise ValueError(f"Invalid {key}: expected 0|1")
    return v


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

    vector_keys = ("cm_v", "b_ca")
    for key in vector_keys:
        arr = values[key]
        if len(arr) != n_comp:
            raise ValueError(f"Invalid {key} length={len(arr)} for n_comp={n_comp}")
        for val in arr:
            vf = float(val)
            if not math.isfinite(vf):
                raise ValueError(f"Invalid {key}: entries must be finite")
            if key == "cm_v" and vf <= 0.0:
                raise ValueError("Invalid cm_v: entries must be > 0")
            if key == "b_ca" and vf < 0.0:
                raise ValueError("Invalid b_ca: entries must be >= 0")

    gbar_mat = values["gbar_mat"]
    if len(gbar_mat) != 11:
        raise ValueError(f"Invalid gbar_mat rows={len(gbar_mat)} (expected 11)")
    for row_idx in range(11):
        if len(gbar_mat[row_idx]) != n_comp:
            raise ValueError(
                f"Invalid gbar_mat row {row_idx} length={len(gbar_mat[row_idx])} for n_comp={n_comp}"
            )
        for val in gbar_mat[row_idx]:
            vf = float(val)
            if (not math.isfinite(vf)) or vf < 0.0:
                raise ValueError("Invalid gbar_mat: entries must be finite and >= 0")

    phi_mat = values["phi_mat"]
    if len(phi_mat) != 9:
        raise ValueError(f"Invalid phi_mat rows={len(phi_mat)} (expected 9)")
    for row_idx in range(9):
        if len(phi_mat[row_idx]) != n_comp:
            raise ValueError(
                f"Invalid phi_mat row {row_idx} length={len(phi_mat[row_idx])} for n_comp={n_comp}"
            )
        for val in phi_mat[row_idx]:
            vf = float(val)
            if (not math.isfinite(vf)) or vf <= 0.0:
                raise ValueError("Invalid phi_mat: entries must be finite and > 0")

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
    prev_evt = -float("inf")
    for k in range(n_events):
        evt = float(event_times_arr[k])
        if not math.isfinite(evt):
            raise ValueError("Invalid event_times_arr: entries must be finite")
        if evt < prev_evt:
            raise ValueError("Invalid event_times_arr: first n_events entries must be non-decreasing")
        prev_evt = evt

    env_vec = values["env_vec"]
    if len(env_vec) != 6:
        raise ValueError(f"Invalid env_vec length={len(env_vec)} (expected 6)")
    t_kelvin, ca_ext, ca_rest, tau_ca, mg_ext, tau_sk = [float(v) for v in env_vec]
    if not math.isfinite(t_kelvin) or t_kelvin <= 0.0:
        raise ValueError("Invalid t_kelvin: must be finite and > 0")
    if not math.isfinite(tau_ca) or tau_ca <= 0.0:
        raise ValueError("Invalid tau_ca: must be finite and > 0")
    if not math.isfinite(tau_sk) or tau_sk <= 0.0:
        raise ValueError("Invalid tau_sk: must be finite and > 0")
    for key, val in (("ca_ext", ca_ext), ("ca_rest", ca_rest), ("mg_ext", mg_ext)):
        if (not math.isfinite(val)) or val < 0.0:
            raise ValueError(f"Invalid {key}: must be finite and >= 0")

    stim1_vec = values["stim1_vec"]
    if len(stim1_vec) != 12:
        raise ValueError(f"Invalid stim1_vec length={len(stim1_vec)} (expected 12)")
    (
        stype, iext, t0, td, atau, zap_f0_hz, zap_f1_hz,
        stim_comp, stim_mode, use_dfilter_primary, dfilter_attenuation, dfilter_tau_ms,
    ) = [float(v) for v in stim1_vec]
    stim_mode_i = int(stim_mode)
    if stim_mode_i not in (0, 1, 2):
        raise ValueError(f"Invalid stim_mode={stim_mode_i}: expected 0|1|2")

    valid_stypes = (0, 1, 2, 4, 5, 6, 7, 8, 9, 10)
    stype_i = int(stype)
    if stype_i not in valid_stypes:
        raise ValueError(f"Invalid stype={stype_i}: unsupported stimulation type")

    stim_comp_i = int(stim_comp)
    if not (0 <= stim_comp_i < n_comp):
        raise ValueError(f"Invalid stim_comp={stim_comp_i} for n_comp={n_comp}")

    if int(use_dfilter_primary) not in (0, 1):
        raise ValueError("Invalid use_dfilter_primary: expected 0|1")

    if not math.isfinite(dfilter_attenuation):
        raise ValueError("Invalid dfilter_attenuation: must be finite")
    if (not math.isfinite(dfilter_tau_ms)) or dfilter_tau_ms < 0.0:
        raise ValueError("Invalid dfilter_tau_ms: must be finite and >= 0")

    if (not math.isfinite(atau)) or atau <= 0.0:
        raise ValueError("Invalid atau: must be finite and > 0")
    for key, val in (("iext", iext), ("t0", t0), ("td", td), ("zap_f0_hz", zap_f0_hz), ("zap_f1_hz", zap_f1_hz)):
        if not math.isfinite(val):
            raise ValueError(f"Invalid {key}: must be finite")
    for key, val in (("td", td), ("zap_f0_hz", zap_f0_hz), ("zap_f1_hz", zap_f1_hz)):
        if val < 0.0:
            raise ValueError(f"Invalid {key}: must be >= 0")

    stim2_vec = values["stim2_vec"]
    if len(stim2_vec) != 13:
        raise ValueError(f"Invalid stim2_vec length={len(stim2_vec)} (expected 13)")
    (
        dual_stim_enabled,
        stype_2, iext_2, t0_2, td_2, atau_2, zap_f0_hz_2, zap_f1_hz_2,
        stim_comp_2, stim_mode_2, use_dfilter_secondary, dfilter_attenuation_2, dfilter_tau_ms_2,
    ) = [float(v) for v in stim2_vec]

    dual_stim_enabled_i = int(dual_stim_enabled)
    if dual_stim_enabled_i not in (0, 1):
        raise ValueError("Invalid dual_stim_enabled: expected 0|1")

    if int(use_dfilter_secondary) not in (0, 1):
        raise ValueError("Invalid use_dfilter_secondary: expected 0|1")

    if not math.isfinite(dfilter_attenuation_2):
        raise ValueError("Invalid dfilter_attenuation_2: must be finite")

    if dual_stim_enabled_i == 1:
        stim_mode_2_i = int(stim_mode_2)
        if stim_mode_2_i not in (0, 1, 2):
            raise ValueError(f"Invalid stim_mode_2={stim_mode_2_i}: expected 0|1|2")

        stype_2_i = int(stype_2)
        if stype_2_i not in valid_stypes:
            raise ValueError(f"Invalid stype_2={stype_2_i}: unsupported stimulation type")

        stim_comp_2_i = int(stim_comp_2)
        if not (0 <= stim_comp_2_i < n_comp):
            raise ValueError(f"Invalid stim_comp_2={stim_comp_2_i} for n_comp={n_comp}")

        if (not math.isfinite(dfilter_tau_ms_2)) or dfilter_tau_ms_2 < 0.0:
            raise ValueError("Invalid dfilter_tau_ms_2: must be finite and >= 0")

        if (not math.isfinite(atau_2)) or atau_2 <= 0.0:
            raise ValueError("Invalid atau_2: must be finite and > 0")

        for key, val in (("iext_2", iext_2), ("t0_2", t0_2), ("td_2", td_2), ("zap_f0_hz_2", zap_f0_hz_2), ("zap_f1_hz_2", zap_f1_hz_2)):
            if not math.isfinite(val):
                raise ValueError(f"Invalid {key}: must be finite")

        for key, val in (("td_2", td_2), ("zap_f0_hz_2", zap_f0_hz_2), ("zap_f1_hz_2", zap_f1_hz_2)):
            if val < 0.0:
                raise ValueError(f"Invalid {key}: must be >= 0")
