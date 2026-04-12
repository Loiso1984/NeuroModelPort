"""
core/advanced_sim.py - Advanced Simulation Modes v10.0

Ports Scilab SWEEP, ANA_SD_CURVE, ANA_EXCMAP modes.
"""
import copy
from dataclasses import dataclass, field
import numpy as np

from core.analysis import detect_spikes
from core.models import FullModelConfig
from core.solver import NeuronSolver


@dataclass
class LightweightResult:
    """Minimal sweep payload to avoid retaining full SimulationResult objects."""

    t: np.ndarray = field(default_factory=lambda: np.array([]))
    v_soma: np.ndarray = field(default_factory=lambda: np.array([]))
    config: FullModelConfig | None = None


def run_euler_maruyama(config: FullModelConfig):
    """Compatibility shim for legacy stochastic tests.

    The project now routes stochastic dynamics through the primary solver path,
    but stochastic gating and additive current noise only live on the native
    fixed-step path. Preserve legacy Euler-Maruyama semantics by forcing the
    native solver whenever stochastic terms are requested.
    """
    return NeuronSolver(config).run_native(config)


# -----------------------------------------------------------------------------
#  PARAMETER SWEEP
# -----------------------------------------------------------------------------

_SWEEP_SECTION_FALLBACKS = (
    "stim",
    "channels",
    "env",
    "morphology",
    "calcium",
    "analysis",
    "metabolism",
    "preset_modes",
    "stim_location",
    "dendritic_filter",
)


def _resolve_sweep_target(cfg: FullModelConfig, param_path: str):
    """Resolve a sweep parameter path, supporting both dotted paths and legacy short names."""
    parts = [part for part in str(param_path).split(".") if part]
    if not parts:
        raise ValueError("Sweep parameter path cannot be empty")

    if len(parts) == 1:
        attr = parts[0]
        if hasattr(cfg, attr):
            return cfg, attr
        for section_name in _SWEEP_SECTION_FALLBACKS:
            section = getattr(cfg, section_name, None)
            if section is not None and hasattr(section, attr):
                return section, attr
        raise AttributeError(f'"FullModelConfig" object has no field "{attr}"')

    obj = cfg
    for part in parts[:-1]:
        if not hasattr(obj, part):
            raise AttributeError(f'"{type(obj).__name__}" object has no field "{part}"')
        obj = getattr(obj, part)
    if not hasattr(obj, parts[-1]):
        raise AttributeError(f'"{type(obj).__name__}" object has no field "{parts[-1]}"')
    return obj, parts[-1]


def _set_sweep_param(cfg: FullModelConfig, param_path: str, val: float):
    """Apply a single sweep parameter value dynamically."""
    obj, attr = _resolve_sweep_target(cfg, param_path)
    setattr(obj, attr, float(val))


def run_sweep(
    config: FullModelConfig,
    param_name: str,
    param_values: np.ndarray,
    progress_cb=None,
) -> list:
    """
    Parametric sweep over a single parameter path.

    Parameters
    ----------
    config       : base FullModelConfig
    param_name   : parameter path to vary (e.g. "channels.gNa_max")
    param_values : array of values to test
    progress_cb  : optional callable(i, n, val) for progress reporting

    Returns
    -------
    list of (param_value, lightweight_result | None)
    """
    results = []
    n = len(param_values)

    for i, val in enumerate(param_values):
        if progress_cb:
            progress_cb(i, n, val)

        try:
            cfg = copy.deepcopy(config)
            _set_sweep_param(cfg, param_name, float(val))
            res = NeuronSolver(cfg).run_single()
            # Store only what analytics needs to avoid RAM explosion in long sweeps.
            lite = LightweightResult()
            lite.t = res.t
            lite.v_soma = res.v_soma
            lite.config = res.config
            results.append((float(val), lite))
        except Exception as e:
            print(f"[SWEEP] {param_name}={val:.4g} -> {e}")
            results.append((float(val), None))

    return results


# -----------------------------------------------------------------------------
#  STRENGTH-DURATION CURVE
# -----------------------------------------------------------------------------

_DEFAULT_DURATIONS = np.array([
    0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0
])


def run_sd_curve(
    config: FullModelConfig,
    durations: np.ndarray = None,
    progress_cb=None,
) -> dict:
    """
    Strength-Duration curve via binary search (Scilab ANA_SD_CURVE).

    For each pulse duration, finds the minimum current threshold I_thr
    using 16 iterations of bisection -> precision ~= 0.003 uA/cm^2.

    Returns
    -------
    dict with keys:
      durations, I_threshold, rheobase, chronaxie,
      weiss_fit, Q_threshold
    """
    if durations is None:
        durations = _DEFAULT_DURATIONS

    I_threshold = np.zeros(len(durations))
    n = len(durations)

    for di, dur in enumerate(durations):
        if progress_cb:
            progress_cb(di, n, dur)

        I_lo, I_hi = 0.1, 1000.0

        for _ in range(18):  # 18 bisection steps -> < 0.001 uA/cm^2
            I_try = (I_lo + I_hi) / 2.0
            cfg = copy.deepcopy(config)
            # Force deterministic physics for binary search.
            cfg.stim.stoch_gating = False
            cfg.stim.noise_sigma = 0.0
            # Prevent metabolic drift from shifting threshold during search.
            cfg.metabolism.enable_dynamic_atp = False
            cfg.stim.stim_type = 'pulse'
            cfg.stim.Iext = I_try
            cfg.stim.pulse_start = 2.0
            cfg.stim.pulse_dur = dur
            cfg.stim.t_sim = max(30.0, dur * 3 + 15.0)
            cfg.stim.dt_eval = 0.05

            try:
                res = NeuronSolver(cfg).run_single()
                pks, _, _ = detect_spikes(res.v_soma, res.t)
                if len(pks) > 0:
                    I_hi = I_try
                else:
                    I_lo = I_try
            except Exception:
                I_lo = I_try

        I_threshold[di] = I_hi

    rheobase = float(np.min(I_threshold))
    chronaxie = np.nan
    idx = np.where(I_threshold <= 2.0 * rheobase * 1.05)[0]
    if len(idx) > 0:
        chronaxie = float(durations[idx[0]])

    # Weiss' Law fit: I_thr = I_rh * (1 + t_ch / dur)
    if not np.isnan(chronaxie):
        weiss_fit = rheobase * (1.0 + chronaxie / durations)
    else:
        weiss_fit = None

    return {
        'durations': durations,
        'I_threshold': I_threshold,
        'rheobase': rheobase,
        'chronaxie': chronaxie,
        'weiss_fit': weiss_fit,
        'Q_threshold': I_threshold * durations,
    }


# -----------------------------------------------------------------------------
#  2-D EXCITABILITY MAP
# -----------------------------------------------------------------------------

def run_excitability_map(
    config: FullModelConfig,
    I_range: np.ndarray = None,
    dur_range: np.ndarray = None,
    progress_cb=None,
) -> dict:
    """
    2-D excitability map: spike count & mean frequency as function of
    (I_ext, pulse_duration). Scilab ANA_EXCMAP.

    Returns
    -------
    dict with I_range, dur_range, spike_matrix, freq_matrix
    """
    ana = config.analysis

    if I_range is None:
        I_range = np.linspace(ana.excmap_I_min, ana.excmap_I_max, ana.excmap_NI)
    if dur_range is None:
        dur_range = np.linspace(ana.excmap_D_min, ana.excmap_D_max, ana.excmap_ND)

    NI, ND = len(I_range), len(dur_range)
    spike_matrix = np.zeros((NI, ND), dtype=np.int32)
    freq_matrix = np.zeros((NI, ND))

    t_sim = float(max(50.0, dur_range[-1] * 3 + 30.0))
    total = NI * ND

    for ii, I_val in enumerate(I_range):
        for di, dur_val in enumerate(dur_range):
            if progress_cb:
                progress_cb(ii * ND + di, total, I_val)

            cfg = copy.deepcopy(config)
            # Force deterministic physics for clean 2-D threshold maps.
            cfg.stim.stoch_gating = False
            cfg.stim.noise_sigma = 0.0
            cfg.metabolism.enable_dynamic_atp = False
            cfg.stim.stim_type = 'pulse'
            cfg.stim.Iext = float(I_val)
            cfg.stim.pulse_start = 2.0
            cfg.stim.pulse_dur = float(dur_val)
            cfg.stim.t_sim = t_sim
            cfg.stim.dt_eval = 0.05

            try:
                res = NeuronSolver(cfg).run_single()
                pks, sp_t, _ = detect_spikes(res.v_soma, res.t)
                n_sp = len(pks)
                spike_matrix[ii, di] = n_sp
                if n_sp > 1:
                    isi_mean = float(np.mean(np.diff(sp_t)))
                    freq_matrix[ii, di] = 1000.0 / isi_mean if isi_mean > 0 else 0.0
            except Exception:
                pass

    return {
        'I_range': I_range,
        'dur_range': dur_range,
        'spike_matrix': spike_matrix,
        'freq_matrix': freq_matrix,
    }
