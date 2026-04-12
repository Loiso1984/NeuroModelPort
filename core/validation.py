"""Centralized simulation parameter validation, warnings, and runtime estimates."""

from __future__ import annotations

from typing import Dict, List

from .errors import SimulationParameterError
from .models import FullModelConfig, PresetModeParams


def estimate_simulation_runtime(cfg: FullModelConfig) -> Dict[str, float]:
    """
    Return a rough runtime estimate for UI/logging guidance.

    The estimate is intentionally conservative and only used for warnings.
    """
    n_steps = int(cfg.stim.t_sim / max(cfg.stim.dt_eval, 1e-12))
    n_comp = (
        1
        if cfg.morphology.single_comp
        else int(
            cfg.morphology.N_ais
            + cfg.morphology.N_trunk
            + cfg.morphology.N_b1
            + cfg.morphology.N_b2
            + 1
        )
    )
    n_channels = int(
        3  # Na, K, Leak
        + int(bool(cfg.channels.enable_Ih))
        + 2 * int(bool(cfg.channels.enable_ICa))
        + 2 * int(bool(cfg.channels.enable_IA))
        + int(bool(cfg.channels.enable_SK))
        + 2 * int(bool(cfg.channels.enable_ITCa))
        + int(bool(cfg.channels.enable_IM))
        + int(bool(cfg.channels.enable_NaP))
        + 2 * int(bool(cfg.channels.enable_NaR))
    )

    if n_comp == 1 and n_channels <= 3:
        est_steps_per_sec = 1500.0
    elif n_comp == 1:
        est_steps_per_sec = 700.0
    elif n_channels <= 3:
        est_steps_per_sec = 220.0
    else:
        est_steps_per_sec = 110.0

    est_time_sec = float(n_steps / max(est_steps_per_sec, 1.0))
    return {
        "n_steps": float(n_steps),
        "n_comp": float(n_comp),
        "n_channels": float(n_channels),
        "estimated_seconds": est_time_sec,
    }


def validate_simulation_config(cfg: FullModelConfig) -> List[str]:
    """
    Validate key solver parameters before integration.

    Raises
    ------
    SimulationParameterError
        When a critical inconsistency would make the run invalid.

    Returns
    -------
    list[str]
        Non-fatal warning messages for edge-case configurations.
    """
    warnings: list[str] = []

    if cfg.stim.dt_eval > cfg.stim.t_sim:
        raise SimulationParameterError(
            f"dt_eval ({cfg.stim.dt_eval}) must be <= t_sim ({cfg.stim.t_sim})."
        )

    if cfg.stim.stim_type in {"alpha", "AMPA", "NMDA", "GABAA", "GABAB", "Kainate", "Nicotinic"}:
        if cfg.stim.alpha_tau <= 0.0:
            raise SimulationParameterError(
                f"alpha_tau must be > 0 for stim_type={cfg.stim.stim_type}, got {cfg.stim.alpha_tau}."
            )

    if cfg.calcium.dynamic_Ca:
        if cfg.calcium.Ca_ext <= 0.0:
            raise SimulationParameterError(f"Ca_ext must be > 0, got {cfg.calcium.Ca_ext}.")
        if cfg.calcium.Ca_rest <= 0.0:
            raise SimulationParameterError(f"Ca_rest must be > 0, got {cfg.calcium.Ca_rest}.")
        if cfg.calcium.tau_Ca <= 0.0:
            raise SimulationParameterError(f"tau_Ca must be > 0, got {cfg.calcium.tau_Ca}.")
        if cfg.calcium.B_Ca <= 0.0:
            raise SimulationParameterError(f"B_Ca must be > 0, got {cfg.calcium.B_Ca}.")
        if cfg.calcium.Ca_ext <= cfg.calcium.Ca_rest:
            warnings.append(
                f"Ca_ext ({cfg.calcium.Ca_ext}) <= Ca_rest ({cfg.calcium.Ca_rest}); "
                "E_Ca may become non-physiological."
            )

    if cfg.channels.enable_Ih and cfg.channels.gIh_max <= 0.0:
        raise SimulationParameterError("Ih is enabled but gIh_max <= 0.")
    if cfg.channels.enable_ICa and cfg.channels.gCa_max <= 0.0:
        raise SimulationParameterError("ICa is enabled but gCa_max <= 0.")
    if cfg.channels.enable_ITCa and cfg.channels.gTCa_max <= 0.0:
        raise SimulationParameterError("ITCa is enabled but gTCa_max <= 0.")
    if cfg.channels.enable_IA and cfg.channels.gA_max <= 0.0:
        raise SimulationParameterError("IA is enabled but gA_max <= 0.")
    if cfg.channels.enable_SK and cfg.channels.gSK_max <= 0.0:
        raise SimulationParameterError("SK is enabled but gSK_max <= 0.")
    if cfg.channels.enable_IM and cfg.channels.gIM_max <= 0.0:
        raise SimulationParameterError("IM is enabled but gIM_max <= 0.")
    if cfg.channels.enable_NaP and cfg.channels.gNaP_max <= 0.0:
        raise SimulationParameterError("NaP is enabled but gNaP_max <= 0.")
    if cfg.channels.enable_NaR and cfg.channels.gNaR_max <= 0.0:
        raise SimulationParameterError("NaR is enabled but gNaR_max <= 0.")

    dual = getattr(cfg, "dual_stimulation", None)
    if dual is not None and getattr(dual, "enabled", False):
        warnings.append(
            "Dual stimulation is enabled: Secondary stimulus configured in Dual Stim tab."
        )
        if getattr(dual, "secondary_duration", 0.0) < 0.0:
            raise SimulationParameterError("dual_stimulation.secondary_duration must be >= 0.")
        sec_tau = float(getattr(dual, "secondary_alpha_tau", 1.0))
        if getattr(dual, "secondary_stim_type", "const") in {"GABAA", "GABAB", "AMPA", "NMDA"} and sec_tau <= 0.0:
            raise SimulationParameterError(
                f"secondary_alpha_tau must be > 0 for {dual.secondary_stim_type}, got {sec_tau}."
            )
        if getattr(dual, "secondary_location", "soma") == "dendritic_filtered":
            tau2 = float(getattr(dual, "secondary_tau_dendritic_ms", 0.0))
            if tau2 < 0.0:
                raise SimulationParameterError("secondary_tau_dendritic_ms must be >= 0.")

    # Non-fatal warnings for potentially expensive/risky runs.
    n_samples = cfg.stim.t_sim / max(cfg.stim.dt_eval, 1e-12)
    if n_samples > 15000:
        warnings.append(
            f"High output sample count ({int(n_samples):,}); consider increasing dt_eval for faster sweeps."
        )
    if abs(cfg.stim.Iext) > 250.0:
        warnings.append(f"High |Iext| ({cfg.stim.Iext}); response may be non-physiological.")

    rt = estimate_simulation_runtime(cfg)
    if rt["estimated_seconds"] > 30.0:
        warnings.append(
            "Heavy simulation estimate: "
            f"{int(rt['n_steps']):,} steps, {int(rt['n_comp'])} compartments, "
            f"{int(rt['n_channels'])} channel-gates, ~{rt['estimated_seconds']:.1f}s expected."
        )

    return warnings


def build_preset_mode_warnings(cfg: FullModelConfig, preset_name: str) -> List[str]:
    """
    Return user-facing mode notes/warnings for currently selected preset mode.

    These are non-fatal and intended for GUI preflight visibility.
    """
    warnings: list[str] = []
    if not preset_name:
        return warnings

    p = preset_name.lower()
    pm = cfg.preset_modes
    pm_default = PresetModeParams()

    if "thalamic" in p:
        if pm.k_mode == "activated":
            warnings.append(
                "K mode=activated: high-throughput relay state enabled; expect higher spike rates than baseline."
            )
        else:
            warnings.append(
                "K mode=baseline: lower-throughput relay state enabled (theta-like global envelope)."
            )

    if "alzheimer" in p and pm.alzheimer_mode == "terminal":
        warnings.append(
            "N mode=terminal: late-stage pathology profile selected; near-silent/strongly reduced excitability is expected."
        )

    if "hypoxia" in p and pm.hypoxia_mode == "terminal":
        warnings.append(
            "O mode=terminal: severe failure profile selected; depolarization-block-like behavior is expected."
        )

    if "multiple sclerosis" in p:
        warnings.append(
            "F preset is currently single-stage; progressive/terminal pathology mode flags do not apply."
        )

    if ("thalamic" not in p) and ("alzheimer" not in p) and ("hypoxia" not in p):
        if (
            pm.k_mode != pm_default.k_mode
            or pm.alzheimer_mode != pm_default.alzheimer_mode
            or pm.hypoxia_mode != pm_default.hypoxia_mode
        ):
            warnings.append(
                "K/N/O mode flags are ignored for this preset."
            )

    return warnings
