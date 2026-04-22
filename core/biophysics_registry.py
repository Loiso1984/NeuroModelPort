"""Central biophysics registry for physiology-critical preset families.

This module is the canonical source for channel-density "truth" used by preset
construction. It keeps literature values and reduced-model operational values in
one place to prevent silent drift between docs and executable presets.

Truth priority policy:
1) Literature values (from docs/reference/*.md)
2) Operational constraints of the reduced HH-like model
3) Legacy hard-coded numbers in presets
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConductanceProfile:
    """Primary membrane conductances in mS/cm^2 plus capacitance in uF/cm^2."""

    gNa_max: float
    gK_max: float
    gL: float
    Cm: float = 1.0


@dataclass(frozen=True)
class FrequencyTarget:
    """Target firing window for default mode validation."""

    low_hz: float
    high_hz: float
    label: str


@dataclass(frozen=True)
class PresetBiophysics:
    """Biophysics contract for one preset family."""

    code: str
    literature_source: str
    literature: ConductanceProfile
    operational: ConductanceProfile
    frequency_target: FrequencyTarget
    burst_target: FrequencyTarget | None = None
    notes: str = ""


BIOPHYSICS_REGISTRY: dict[str, PresetBiophysics] = {
    "B": PresetBiophysics(
        code="B",
        literature_source="Mainen & Sejnowski 1996",
        literature=ConductanceProfile(gNa_max=130.0, gK_max=40.0, gL=0.08, Cm=1.0),
        # L5 conflict resolved: gNa_max is fixed to literature 130.0.
        operational=ConductanceProfile(gNa_max=130.0, gK_max=40.0, gL=0.08, Cm=1.0),
        frequency_target=FrequencyTarget(low_hz=10.0, high_hz=20.0, label="regular-spiking"),
        notes=(
            "In vivo-like near-threshold operation is represented via dendritic/event-driven "
            "default drive instead of high direct soma clamp."
        ),
    ),
    "C": PresetBiophysics(
        code="C",
        literature_source="Wang & Buzsaki 1996",
        literature=ConductanceProfile(gNa_max=120.0, gK_max=40.0, gL=0.15, Cm=1.0),
        # Reduced model keeps slightly lower leak to preserve robust branch-test excitability.
        operational=ConductanceProfile(gNa_max=120.0, gK_max=40.0, gL=0.10, Cm=1.0),
        frequency_target=FrequencyTarget(low_hz=80.0, high_hz=200.0, label="fast-spiking"),
    ),
    "E": PresetBiophysics(
        code="E",
        literature_source="De Schutter & Bower 1994",
        literature=ConductanceProfile(gNa_max=50.0, gK_max=22.0, gL=0.08, Cm=1.0),
        operational=ConductanceProfile(gNa_max=50.0, gK_max=22.0, gL=0.08, Cm=1.0),
        frequency_target=FrequencyTarget(low_hz=30.0, high_hz=50.0, label="simple-spike tonic"),
        notes=(
            "Default Purkinje mode is calibrated with distributed dendritic event drive to avoid "
            "excitotoxic 200+ Hz drift while preserving simple-spike physiology."
        ),
    ),
    "K": PresetBiophysics(
        code="K",
        literature_source="McCormick & Huguenard 1992",
        literature=ConductanceProfile(gNa_max=80.0, gK_max=25.0, gL=0.10, Cm=1.0),
        operational=ConductanceProfile(gNa_max=80.0, gK_max=25.0, gL=0.10, Cm=1.0),
        frequency_target=FrequencyTarget(low_hz=10.0, high_hz=30.0, label="relay mode"),
        burst_target=FrequencyTarget(low_hz=130.0, high_hz=170.0, label="activated burst mode"),
        notes=(
            "Relay mode is modeled as event-driven thalamocortical throughput; burst mode uses "
            "stronger depolarizing context."
        ),
    ),
}


def _normalize_code(name_or_code: str) -> str:
    if not isinstance(name_or_code, str):
        return ""
    head, _, _ = name_or_code.partition(":")
    return head.strip().upper()


def get_registry_entry(name_or_code: str) -> PresetBiophysics | None:
    """Return registry entry by preset name ("B: ...") or code ("B")."""

    return BIOPHYSICS_REGISTRY.get(_normalize_code(name_or_code))


def get_operational_conductance(name_or_code: str) -> ConductanceProfile | None:
    entry = get_registry_entry(name_or_code)
    return entry.operational if entry is not None else None


def get_literature_conductance(name_or_code: str) -> ConductanceProfile | None:
    entry = get_registry_entry(name_or_code)
    return entry.literature if entry is not None else None


def get_frequency_targets(name_or_code: str) -> dict[str, Any] | None:
    entry = get_registry_entry(name_or_code)
    if entry is None:
        return None
    return {
        "default": {
            "low_hz": entry.frequency_target.low_hz,
            "high_hz": entry.frequency_target.high_hz,
            "label": entry.frequency_target.label,
        },
        "burst": (
            None
            if entry.burst_target is None
            else {
                "low_hz": entry.burst_target.low_hz,
                "high_hz": entry.burst_target.high_hz,
                "label": entry.burst_target.label,
            }
        ),
        "notes": entry.notes,
        "source": entry.literature_source,
    }
