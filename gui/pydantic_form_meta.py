"""Small UX metadata helpers for generated Pydantic forms."""

from __future__ import annotations


CRITICAL_FIELDS = {
    "stim_type",
    "Iext",
    "pulse_start",
    "pulse_dur",
    "alpha_tau",
    "jacobian_mode",
    "t_sim",
    "dt_eval",
}

BASIC_FIELDS = CRITICAL_FIELDS | {
    "gNa_max",
    "gK_max",
    "gL",
    "gNa_ais_mult",
    "gK_ais_mult",
    "T_celsius",
    "single_comp",
    "Ra",
    "d_soma",
}


def default_priority_for_field(field_name: str) -> str:
    if field_name in CRITICAL_FIELDS:
        return "critical"
    if field_name in BASIC_FIELDS:
        return "basic"
    return "research"
