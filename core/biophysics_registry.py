"""Central biophysics registry for physiology-critical preset families.

This module is the canonical source for executable biophysical truth used by:
- preset construction (`core/presets.py`)
- passport/reference diagnostics (`gui/analytics.py`)
- validation utilities and stress protocols (`tests/utils/*`)

Truth priority policy:
1) Literature values (docs/reference/*.md, cited per profile)
2) Reduced-model operational constraints
3) Legacy hard-coded values in presets
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
class SpikeShapeTarget:
    """Reference envelope for spike-shape diagnostics."""

    amplitude_mV: tuple[float, float]
    halfwidth_ms: tuple[float, float]
    threshold_mV: tuple[float, float]
    ahp_mV: tuple[float, float]
    dvdt_up_mV_per_ms: tuple[float, float]
    dvdt_down_mV_per_ms: tuple[float, float]
    latency_ms: tuple[float, float]
    cv_isi: tuple[float, float]
    burst_ratio: tuple[float, float]


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


@dataclass(frozen=True)
class ReferenceProfile:
    """Core6 reference profile used by passport/diagnostics."""

    code: str
    variant: str
    context: str
    source: str
    frequency_target: FrequencyTarget
    spike_shape: SpikeShapeTarget
    radar_baseline: tuple[float, float, float, float, float, float]
    notes: str = ""


_CORE6_CODES: tuple[str, ...] = ("B", "C", "E", "K", "L", "Q")


BIOPHYSICS_REGISTRY: dict[str, PresetBiophysics] = {
    "B": PresetBiophysics(
        code="B",
        literature_source="Mainen & Sejnowski 1996",
        literature=ConductanceProfile(gNa_max=130.0, gK_max=40.0, gL=0.08, Cm=1.0),
        operational=ConductanceProfile(gNa_max=130.0, gK_max=40.0, gL=0.08, Cm=1.0),
        frequency_target=FrequencyTarget(low_hz=10.0, high_hz=20.0, label="regular-spiking"),
        notes=(
            "In vivo-like near-threshold operation is represented via dendritic/event-driven "
            "drive instead of strong direct soma clamp."
        ),
    ),
    "C": PresetBiophysics(
        code="C",
        literature_source="Wang & Buzsaki 1996",
        literature=ConductanceProfile(gNa_max=120.0, gK_max=40.0, gL=0.15, Cm=1.0),
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
            "Default Purkinje mode uses distributed dendritic event drive to avoid "
            "excitotoxic 200+ Hz runaway."
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
            "Relay mode is modeled as low-throughput event-driven throughput; "
            "activated mode reflects strong depolarizing context."
        ),
    ),
    "L": PresetBiophysics(
        code="L",
        literature_source="Magee 1998",
        literature=ConductanceProfile(gNa_max=56.0, gK_max=8.0, gL=0.05, Cm=1.0),
        operational=ConductanceProfile(gNa_max=40.0, gK_max=12.0, gL=0.07, Cm=1.0),
        frequency_target=FrequencyTarget(low_hz=4.0, high_hz=12.0, label="theta-paced adapting"),
        notes=(
            "Operational CA1 uses higher repolarization reserve plus stronger leak under dendritic/event-driven "
            "drive so the reduced model remains in low-throughput theta-paced regime."
        ),
    ),
    "Q": PresetBiophysics(
        code="Q",
        literature_source="Nisenbaum & Wilson 1995",
        literature=ConductanceProfile(gNa_max=80.0, gK_max=8.0, gL=0.04, Cm=1.0),
        operational=ConductanceProfile(gNa_max=70.0, gK_max=8.0, gL=0.04, Cm=1.0),
        frequency_target=FrequencyTarget(low_hz=1.0, high_hz=15.0, label="late-recruiting SPN"),
        notes=(
            "SPN operational preset keeps delayed recruitment but reduces tonic up-state proxy current to avoid "
            "non-physiological high-throughput firing."
        ),
    ),
}


CORE6_REFERENCE_PROFILES: dict[tuple[str, str, str], ReferenceProfile] = {
    # B: L5 pyramidal
    ("B", "normal", "in_vivo_like"): ReferenceProfile(
        code="B",
        variant="normal",
        context="in_vivo_like",
        source="Mainen & Sejnowski 1996",
        frequency_target=FrequencyTarget(10.0, 20.0, "L5 RS"),
        spike_shape=SpikeShapeTarget(
            amplitude_mV=(75.0, 110.0),
            halfwidth_ms=(0.7, 1.5),
            threshold_mV=(-55.0, -35.0),
            ahp_mV=(-90.0, -62.0),
            dvdt_up_mV_per_ms=(20.0, 120.0),
            dvdt_down_mV_per_ms=(-120.0, -15.0),
            latency_ms=(5.0, 90.0),
            cv_isi=(0.02, 0.35),
            burst_ratio=(0.0, 0.3),
        ),
        radar_baseline=(0.12, 0.72, 0.38, 0.45, 0.32, 0.70),
    ),
    ("B", "normal", "in_vitro"): ReferenceProfile(
        code="B",
        variant="normal",
        context="in_vitro",
        source="Mainen & Sejnowski 1996 (slice context)",
        frequency_target=FrequencyTarget(5.0, 15.0, "L5 RS slice"),
        spike_shape=SpikeShapeTarget(
            amplitude_mV=(70.0, 110.0),
            halfwidth_ms=(0.9, 2.0),
            threshold_mV=(-55.0, -35.0),
            ahp_mV=(-92.0, -62.0),
            dvdt_up_mV_per_ms=(10.0, 90.0),
            dvdt_down_mV_per_ms=(-100.0, -10.0),
            latency_ms=(8.0, 140.0),
            cv_isi=(0.03, 0.45),
            burst_ratio=(0.0, 0.35),
        ),
        radar_baseline=(0.10, 0.68, 0.45, 0.48, 0.28, 0.68),
    ),
    ("B", "high_ach", "in_vivo_like"): ReferenceProfile(
        code="B",
        variant="high_ach",
        context="in_vivo_like",
        source="ACh modulation overlay",
        frequency_target=FrequencyTarget(12.0, 35.0, "L5 high-ACh"),
        spike_shape=SpikeShapeTarget(
            amplitude_mV=(70.0, 110.0),
            halfwidth_ms=(0.6, 1.4),
            threshold_mV=(-55.0, -32.0),
            ahp_mV=(-88.0, -58.0),
            dvdt_up_mV_per_ms=(20.0, 130.0),
            dvdt_down_mV_per_ms=(-130.0, -15.0),
            latency_ms=(3.0, 80.0),
            cv_isi=(0.02, 0.40),
            burst_ratio=(0.0, 0.35),
        ),
        radar_baseline=(0.18, 0.58, 0.35, 0.40, 0.40, 0.72),
    ),
    # C: FS
    ("C", "default", "in_vivo_like"): ReferenceProfile(
        code="C",
        variant="default",
        context="in_vivo_like",
        source="Wang & Buzsaki 1996",
        frequency_target=FrequencyTarget(80.0, 200.0, "FS"),
        spike_shape=SpikeShapeTarget(
            amplitude_mV=(60.0, 95.0),
            halfwidth_ms=(0.2, 0.7),
            threshold_mV=(-55.0, -30.0),
            ahp_mV=(-95.0, -60.0),
            dvdt_up_mV_per_ms=(40.0, 220.0),
            dvdt_down_mV_per_ms=(-260.0, -40.0),
            latency_ms=(1.0, 30.0),
            cv_isi=(0.01, 0.25),
            burst_ratio=(0.0, 0.2),
        ),
        radar_baseline=(0.72, 0.52, 0.18, 0.22, 0.64, 0.66),
    ),
    # E: Purkinje
    ("E", "tonic", "in_vivo_like"): ReferenceProfile(
        code="E",
        variant="tonic",
        context="in_vivo_like",
        source="De Schutter & Bower 1994",
        frequency_target=FrequencyTarget(30.0, 50.0, "Purkinje simple spike"),
        spike_shape=SpikeShapeTarget(
            amplitude_mV=(55.0, 90.0),
            halfwidth_ms=(0.3, 0.9),
            threshold_mV=(-58.0, -35.0),
            ahp_mV=(-90.0, -60.0),
            dvdt_up_mV_per_ms=(25.0, 160.0),
            dvdt_down_mV_per_ms=(-180.0, -20.0),
            latency_ms=(2.0, 80.0),
            cv_isi=(0.02, 0.40),
            burst_ratio=(0.0, 0.35),
        ),
        radar_baseline=(0.30, 0.62, 0.22, 0.30, 0.58, 0.64),
        notes="Recalibrated from prior excitotoxic 200+ Hz drift.",
    ),
    ("E", "climbing_fiber", "in_vivo_like"): ReferenceProfile(
        code="E",
        variant="climbing_fiber",
        context="in_vivo_like",
        source="Purkinje climbing-fiber surrogate",
        frequency_target=FrequencyTarget(1.0, 10.0, "complex-spike burst packets"),
        spike_shape=SpikeShapeTarget(
            amplitude_mV=(50.0, 95.0),
            halfwidth_ms=(0.5, 2.2),
            threshold_mV=(-60.0, -35.0),
            ahp_mV=(-92.0, -60.0),
            dvdt_up_mV_per_ms=(20.0, 170.0),
            dvdt_down_mV_per_ms=(-200.0, -18.0),
            latency_ms=(1.0, 40.0),
            cv_isi=(0.20, 0.90),
            burst_ratio=(0.25, 1.0),
        ),
        radar_baseline=(0.12, 0.40, 0.44, 0.32, 0.72, 0.70),
    ),
    # K: thalamic
    ("K", "baseline", "in_vivo_like"): ReferenceProfile(
        code="K",
        variant="baseline",
        context="in_vivo_like",
        source="McCormick & Huguenard 1992",
        frequency_target=FrequencyTarget(10.0, 30.0, "thalamic relay"),
        spike_shape=SpikeShapeTarget(
            amplitude_mV=(60.0, 95.0),
            halfwidth_ms=(0.6, 1.8),
            threshold_mV=(-60.0, -35.0),
            ahp_mV=(-95.0, -62.0),
            dvdt_up_mV_per_ms=(20.0, 120.0),
            dvdt_down_mV_per_ms=(-150.0, -18.0),
            latency_ms=(3.0, 120.0),
            cv_isi=(0.05, 0.60),
            burst_ratio=(0.0, 0.50),
        ),
        radar_baseline=(0.15, 0.52, 0.30, 0.38, 0.35, 0.62),
        notes="Baseline relay remains low-throughput; theta-like rhythm is treated as network-driven.",
    ),
    ("K", "activated", "in_vivo_like"): ReferenceProfile(
        code="K",
        variant="activated",
        context="in_vivo_like",
        source="McCormick & Huguenard 1992",
        frequency_target=FrequencyTarget(130.0, 170.0, "burst relay"),
        spike_shape=SpikeShapeTarget(
            amplitude_mV=(55.0, 95.0),
            halfwidth_ms=(0.5, 1.5),
            threshold_mV=(-60.0, -30.0),
            ahp_mV=(-92.0, -58.0),
            dvdt_up_mV_per_ms=(20.0, 170.0),
            dvdt_down_mV_per_ms=(-180.0, -20.0),
            latency_ms=(1.0, 80.0),
            cv_isi=(0.10, 0.80),
            burst_ratio=(0.15, 1.0),
        ),
        radar_baseline=(0.88, 0.42, 0.26, 0.35, 0.72, 0.68),
    ),
    ("K", "delta_oscillator", "in_vivo_like"): ReferenceProfile(
        code="K",
        variant="delta_oscillator",
        context="in_vivo_like",
        source="Thalamic delta surrogate",
        frequency_target=FrequencyTarget(1.0, 8.0, "delta-like sparse"),
        spike_shape=SpikeShapeTarget(
            amplitude_mV=(50.0, 95.0),
            halfwidth_ms=(0.6, 2.5),
            threshold_mV=(-62.0, -32.0),
            ahp_mV=(-98.0, -62.0),
            dvdt_up_mV_per_ms=(10.0, 130.0),
            dvdt_down_mV_per_ms=(-170.0, -12.0),
            latency_ms=(20.0, 280.0),
            cv_isi=(0.30, 1.00),
            burst_ratio=(0.2, 1.0),
        ),
        radar_baseline=(0.08, 0.45, 0.42, 0.44, 0.30, 0.60),
    ),
    # L: CA1
    ("L", "default", "in_vivo_like"): ReferenceProfile(
        code="L",
        variant="default",
        context="in_vivo_like",
        source="Magee 1998",
        frequency_target=FrequencyTarget(4.0, 12.0, "theta-paced CA1"),
        spike_shape=SpikeShapeTarget(
            amplitude_mV=(70.0, 105.0),
            halfwidth_ms=(0.3, 2.2),
            threshold_mV=(-60.0, -34.0),
            ahp_mV=(-95.0, -62.0),
            dvdt_up_mV_per_ms=(18.0, 120.0),
            dvdt_down_mV_per_ms=(-140.0, -15.0),
            latency_ms=(5.0, 220.0),
            cv_isi=(0.0, 0.80),
            burst_ratio=(0.0, 0.45),
        ),
        radar_baseline=(0.09, 0.65, 0.52, 0.62, 0.36, 0.66),
    ),
    # Q: SPN
    ("Q", "default", "in_vivo_like"): ReferenceProfile(
        code="Q",
        variant="default",
        context="in_vivo_like",
        source="Nisenbaum & Wilson 1995",
        frequency_target=FrequencyTarget(1.0, 15.0, "SPN delayed"),
        spike_shape=SpikeShapeTarget(
            amplitude_mV=(60.0, 130.0),
            halfwidth_ms=(0.4, 2.5),
            threshold_mV=(-60.0, -35.0),
            ahp_mV=(-100.0, -68.0),
            dvdt_up_mV_per_ms=(10.0, 120.0),
            dvdt_down_mV_per_ms=(-120.0, -10.0),
            latency_ms=(20.0, 260.0),
            cv_isi=(0.01, 0.80),
            burst_ratio=(0.0, 0.50),
        ),
        radar_baseline=(0.07, 0.70, 0.56, 0.72, 0.28, 0.64),
    ),
}


def _normalize_code(name_or_code: str) -> str:
    if not isinstance(name_or_code, str):
        return ""
    head, _, _ = name_or_code.partition(":")
    return head.strip().upper()


def _normalize_context(context: str | None) -> str:
    raw = str(context or "in_vivo_like").strip().lower()
    aliases = {
        "in_vivo": "in_vivo_like",
        "in-vivo": "in_vivo_like",
        "invivo": "in_vivo_like",
        "in_vitro": "in_vitro",
        "in-vitro": "in_vitro",
        "invitro": "in_vitro",
    }
    return aliases.get(raw, raw)


def _profile_key(code: str, variant: str, context: str) -> tuple[str, str, str]:
    return (
        _normalize_code(code),
        str(variant or "default").strip().lower() or "default",
        _normalize_context(context),
    )


def get_core6_codes() -> tuple[str, ...]:
    return _CORE6_CODES


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


def get_reference_profile(
    name_or_code: str,
    *,
    variant: str = "default",
    context: str = "in_vivo_like",
) -> ReferenceProfile | None:
    """Return Core6 reference profile by code+variant+context.

    Fallback order:
    1) exact (code, variant, context)
    2) (code, variant, in_vivo_like)
    3) (code, default, context)
    4) (code, default, in_vivo_like)
    """
    code = _normalize_code(name_or_code)
    var = str(variant or "default").strip().lower() or "default"
    ctx = _normalize_context(context)
    candidates = [
        (code, var, ctx),
        (code, var, "in_vivo_like"),
        (code, "default", ctx),
        (code, "default", "in_vivo_like"),
    ]
    for key in candidates:
        profile = CORE6_REFERENCE_PROFILES.get(key)
        if profile is not None:
            return profile
    return None


def get_radar_baseline(
    name_or_code: str,
    *,
    variant: str = "default",
    context: str = "in_vivo_like",
) -> list[float] | None:
    profile = get_reference_profile(name_or_code, variant=variant, context=context)
    if profile is None:
        return None
    return list(profile.radar_baseline)


def _extract_code_from_notes(cfg: Any) -> str:
    notes = str(getattr(cfg, "notes", "") or "").strip()
    if "preset_code=" not in notes:
        return ""
    marker = notes.split("preset_code=", 1)[1]
    return _normalize_code(marker[:1])


def _infer_code_from_conductance(cfg: Any) -> str:
    try:
        gna = float(cfg.channels.gNa_max)
        gk = float(cfg.channels.gK_max)
        gl = float(cfg.channels.gL)
    except Exception:
        return ""

    best_code = ""
    best_dist = float("inf")
    for code in _CORE6_CODES:
        entry = BIOPHYSICS_REGISTRY.get(code)
        if entry is None:
            continue
        op = entry.operational
        dist = (
            abs(gna - op.gNa_max) / max(1.0, abs(op.gNa_max))
            + abs(gk - op.gK_max) / max(1.0, abs(op.gK_max))
            + abs(gl - op.gL) / max(1e-3, abs(op.gL))
        )
        if dist < best_dist:
            best_dist = dist
            best_code = code
    return best_code if best_dist <= 1.5 else ""


def infer_reference_selector(cfg: Any) -> tuple[str, str, str]:
    """Infer (code, variant, context) from config for diagnostics UI."""
    code = _extract_code_from_notes(cfg) or _infer_code_from_conductance(cfg)

    variant = "default"
    modes = getattr(cfg, "preset_modes", None)
    if code == "B":
        mode = str(getattr(modes, "l5_mode", "normal")).strip().lower()
        variant = "high_ach" if mode == "high_ach" else "normal"
    elif code == "E":
        mode = str(getattr(modes, "purkinje_mode", "tonic")).strip().lower()
        variant = "climbing_fiber" if mode == "climbing_fiber" else "tonic"
    elif code == "K":
        mode = str(getattr(modes, "k_mode", "baseline")).strip().lower()
        if mode in {"baseline", "activated", "delta_oscillator"}:
            variant = mode
        else:
            variant = "baseline"

    t_c = float(getattr(getattr(cfg, "env", None), "T_celsius", 37.0))
    context = "in_vitro" if t_c <= 25.0 else "in_vivo_like"
    return code, variant, context
