"""Compact batch validation path for sweep/search utilities.

This module is intentionally parallel to the full `NeuronSolver.run_single` path:
- GUI and deep analytics keep using full traces + post-processing.
- Sweep/search validators can use compact metrics and status codes.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
import copy
import time
from typing import Any, Iterable

import numpy as np

from core.models import FullModelConfig
from core.solver import NeuronSolver


STATUS_OK = "OK"
STATUS_SILENT = "SILENT"
STATUS_D_BLOCK = "D_BLOCK"
STATUS_UNSTABLE = "UNSTABLE"

_STATUS_ORDER = (STATUS_OK, STATUS_SILENT, STATUS_D_BLOCK, STATUS_UNSTABLE)


@dataclass(frozen=True)
class BatchValidationResult:
    rows: list[dict[str, Any]]
    elapsed_sec: float
    n_cases: int
    status_counts: dict[str, int]


def _spike_times_threshold(
    v_soma: np.ndarray,
    t_ms: np.ndarray,
    *,
    threshold_mV: float = -20.0,
    refractory_ms: float = 1.0,
) -> np.ndarray:
    if v_soma.size < 2 or t_ms.size < 2:
        return np.zeros(0, dtype=float)
    st = []
    last_t = -1e12
    for i in range(1, len(v_soma)):
        v0 = float(v_soma[i - 1])
        v1 = float(v_soma[i])
        if v0 < threshold_mV <= v1:
            t0 = float(t_ms[i - 1])
            t1 = float(t_ms[i])
            if abs(v1 - v0) > 1e-12:
                frac = (threshold_mV - v0) / (v1 - v0)
                ts = t0 + frac * (t1 - t0)
            else:
                ts = t1
            if ts - last_t >= refractory_ms:
                st.append(ts)
                last_t = ts
    return np.asarray(st, dtype=float)


def _has_depol_block(
    v_soma: np.ndarray,
    t_ms: np.ndarray,
    *,
    start_ms: float = 100.0,
    hold_ms: float = 20.0,
    depol_threshold_mV: float = -30.0,
) -> bool:
    if v_soma.size < 2 or t_ms.size < 2:
        return False
    mask = t_ms >= start_ms
    if not np.any(mask):
        return False
    idx = np.flatnonzero(mask)
    if idx.size < 2:
        return False
    dt = float(np.median(np.diff(t_ms[idx])))
    dt = max(dt, 1e-6)
    need_count = max(1, int(hold_ms / dt))
    run = 0
    for i in idx:
        if float(v_soma[i]) > depol_threshold_mV:
            run += 1
            if run >= need_count:
                return True
        else:
            run = 0
    return False


def _is_silent(
    v_soma: np.ndarray,
    t_ms: np.ndarray,
    n_spikes: int,
    *,
    check_after_ms: float = 150.0,
    silence_ceiling_mV: float = -40.0,
) -> bool:
    if n_spikes > 0:
        return False
    if v_soma.size == 0 or t_ms.size == 0:
        return True
    if float(t_ms[-1]) < check_after_ms:
        return False
    mask = t_ms >= check_after_ms
    if not np.any(mask):
        return False
    v_slice = v_soma[mask]
    return bool(np.max(v_slice) < silence_ceiling_mV)


def _extract_atp_summary(res) -> tuple[float | None, float | None]:
    """Return (atp_end_mM, atp_min_mM) from dynamic-ATP state if available."""
    try:
        off_atp = int(getattr(res.state_offsets, "off_atp", -1))
    except Exception:
        off_atp = -1
    if off_atp < 0:
        return None, None
    if getattr(res, "y", None) is None or np.size(res.y) == 0:
        return None, None
    try:
        atp_trace = np.asarray(res.y[off_atp, :], dtype=float)
        if atp_trace.size == 0:
            return None, None
        return float(atp_trace[-1]), float(np.nanmin(atp_trace))
    except Exception:
        return None, None


def _extract_metrics(res, cfg: FullModelConfig) -> dict[str, Any]:
    v_soma = np.asarray(res.v_soma, dtype=float)
    t_arr = np.asarray(res.t, dtype=float)
    spike_t = _spike_times_threshold(v_soma, t_arr)

    duration_ms = float(t_arr[-1] - t_arr[0]) if t_arr.size > 1 else 0.0
    n_spikes = int(len(spike_t))
    freq_global = float(1000.0 * n_spikes / duration_ms) if duration_ms > 0 else 0.0

    v_peak = float(np.nanmax(v_soma)) if v_soma.size else float("nan")
    v_min = float(np.nanmin(v_soma)) if v_soma.size else float("nan")
    tail_n = min(80, int(v_soma.size))
    v_tail = float(np.nanmean(v_soma[-tail_n:])) if tail_n > 0 else float("nan")

    stable_finite = bool(np.all(np.isfinite(v_soma)))
    diverged = bool(getattr(res, "diverged", False))
    unstable = bool(
        diverged
        or (not stable_finite)
        or (np.nanmax(np.abs(v_soma)) > 300.0 if v_soma.size else True)
    )

    ca_max_nM = None
    ca_ok = True
    if getattr(res, "ca_i", None) is not None:
        ca = np.asarray(res.ca_i[0, :], dtype=float)
        if ca.size > 0:
            ca_max_nM = float(np.nanmax(ca) * 1e6)
            ca_ok = bool(np.all(ca >= 0.0) and ca_max_nM <= 10000.0)

    atp_end_mM, atp_min_mM = _extract_atp_summary(res)

    voltage_ok = bool(
        -140.0 < v_min < 80.0
        and -140.0 < v_peak < 80.0
        and -120.0 < v_tail < 20.0
    )

    if unstable:
        status = STATUS_UNSTABLE
    elif _has_depol_block(v_soma, t_arr):
        status = STATUS_D_BLOCK
    elif _is_silent(v_soma, t_arr, n_spikes):
        status = STATUS_SILENT
    else:
        status = STATUS_OK

    return {
        "status_code": status,
        "n_spikes": n_spikes,
        "freq_global_hz": freq_global,
        "v_peak_mV": v_peak,
        "v_min_mV": v_min,
        "v_tail_mV": v_tail,
        "ca_max_nM": ca_max_nM,
        "ca_ok": ca_ok,
        "stable_finite": stable_finite,
        "voltage_ok": voltage_ok,
        "guard_ok": bool(status == STATUS_OK and voltage_ok and ca_ok),
        "diverged": diverged,
        "duration_ms": duration_ms,
        "stim_Iext": float(cfg.stim.Iext),
        "atp_end_mM": atp_end_mM,
        "atp_min_mM": atp_min_mM,
    }


def _simulate_cfg(
    cfg: FullModelConfig,
    *,
    compact_native: bool = True,
    compact_dt_eval_ms: float | None = None,
):
    run_cfg = cfg
    if (
        compact_dt_eval_ms is not None
        and compact_dt_eval_ms > 0.0
        and float(cfg.stim.dt_eval) < float(compact_dt_eval_ms)
    ):
        run_cfg = copy.deepcopy(cfg)
        run_cfg.stim.dt_eval = float(compact_dt_eval_ms)

    solver = NeuronSolver(run_cfg)
    if compact_native and run_cfg.stim.jacobian_mode == "native_hines":
        return solver.run_native(run_cfg, post_process=False)
    return solver.run_single(run_cfg)


def _evaluate_case(
    cfg: FullModelConfig,
    *,
    compact_native: bool,
    quick_prune_ms: float | None,
    compact_dt_eval_ms: float | None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    pruned = False

    if quick_prune_ms is not None and quick_prune_ms > 0.0 and cfg.stim.t_sim > quick_prune_ms:
        quick_cfg = copy.deepcopy(cfg)
        quick_cfg.stim.t_sim = float(quick_prune_ms)
        quick_res = _simulate_cfg(
            quick_cfg,
            compact_native=compact_native,
            compact_dt_eval_ms=compact_dt_eval_ms,
        )
        quick_metrics = _extract_metrics(quick_res, quick_cfg)
        if quick_metrics["status_code"] in (STATUS_SILENT, STATUS_D_BLOCK, STATUS_UNSTABLE):
            row = dict(quick_metrics)
            row["pruned"] = True
            row["elapsed_sec"] = float(time.perf_counter() - t0)
            row["sim_t_ms"] = float(quick_cfg.stim.t_sim)
            return row
        pruned = True

    res = _simulate_cfg(
        cfg,
        compact_native=compact_native,
        compact_dt_eval_ms=compact_dt_eval_ms,
    )
    metrics = _extract_metrics(res, cfg)
    row = dict(metrics)
    row["pruned"] = pruned
    row["elapsed_sec"] = float(time.perf_counter() - t0)
    row["sim_t_ms"] = float(cfg.stim.t_sim)
    row["dt_eval_ms"] = float(res.t[1] - res.t[0]) if len(res.t) > 1 else float(cfg.stim.dt_eval)
    return row


def _evaluate_case_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    cfg = FullModelConfig.model_validate(payload["cfg"])
    return _evaluate_case(
        cfg,
        compact_native=bool(payload.get("compact_native", True)),
        quick_prune_ms=payload.get("quick_prune_ms", 180.0),
        compact_dt_eval_ms=payload.get("compact_dt_eval_ms", None),
    )


def run_validation_batch(
    configs: Iterable[FullModelConfig],
    *,
    workers: int = 1,
    compact_native: bool = True,
    quick_prune_ms: float | None = 180.0,
    compact_dt_eval_ms: float | None = None,
    parallel_backend: str = "thread",
) -> BatchValidationResult:
    cfg_list = list(configs)
    t0 = time.perf_counter()

    def _run_one(cfg: FullModelConfig) -> dict[str, Any]:
        return _evaluate_case(
            cfg,
            compact_native=compact_native,
            quick_prune_ms=quick_prune_ms,
            compact_dt_eval_ms=compact_dt_eval_ms,
        )

    backend = str(parallel_backend or "thread").strip().lower()
    if workers <= 1 or len(cfg_list) < 2 or backend == "serial":
        rows = [_run_one(cfg) for cfg in cfg_list]
    elif backend == "process":
        payloads = [
            {
                "cfg": cfg.model_dump(),
                "compact_native": bool(compact_native),
                "quick_prune_ms": quick_prune_ms,
                "compact_dt_eval_ms": compact_dt_eval_ms,
            }
            for cfg in cfg_list
        ]
        with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
            rows = list(pool.map(_evaluate_case_from_payload, payloads))
    else:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            rows = list(pool.map(_run_one, cfg_list))

    counts = {k: 0 for k in _STATUS_ORDER}
    for row in rows:
        code = str(row.get("status_code", STATUS_UNSTABLE))
        counts[code] = counts.get(code, 0) + 1

    return BatchValidationResult(
        rows=rows,
        elapsed_sec=float(time.perf_counter() - t0),
        n_cases=len(cfg_list),
        status_counts=counts,
    )
