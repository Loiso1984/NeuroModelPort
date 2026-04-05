from __future__ import annotations

"""
Wide-range preset stress validator (Phase 5 scaffold).

This utility is intentionally lightweight and report-oriented:
- sweeps presets across broad but bounded conditions;
- records core physiological sanity metrics;
- emits PASS/WARN/FAIL style summaries.
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.utils.runtime_import_guard import dependency_diagnostic

_IMPORT_ERROR: Exception | None = None
try:  # pragma: no cover - environment-dependent import path
    from core.analysis import detect_spikes
    from core.models import FullModelConfig
    from core.presets import apply_preset, get_preset_names
    from core.solver import NeuronSolver
except Exception as exc:  # pragma: no cover
    _IMPORT_ERROR = exc


@dataclass
class StressCase:
    preset: str
    iext_scale: float
    t_sim_ms: float
    temp_c: float
    noise_sigma: float
    v_min: float
    v_max: float
    spike_count: int
    firing_hz: float
    has_nan_inf: bool
    status: str
    note: str


def _exit_code_for_summary(
    fail_count: int,
    warn_count: int,
    *,
    fail_on_fail: bool,
    fail_on_warn: bool,
) -> int:
    if fail_on_fail and fail_count > 0:
        return 1
    if fail_on_warn and warn_count > 0:
        return 1
    return 0


def _overall_status(fail_count: int, warn_count: int) -> str:
    if fail_count > 0:
        return "FAIL"
    if warn_count > 0:
        return "WARN"
    return "PASS"


def _load_reference_ranges(path: Path) -> dict:
    if not path.exists():
        return {
            "default": {"firing_hz": [0.0, 250.0], "v_min": [-130.0, -40.0], "v_max": [0.0, 120.0]},
            "keyword_rules": [],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_ranges_for_preset(name: str, reference: dict) -> dict[str, tuple[float, float]]:
    n = name.lower()
    for rule in reference.get("keyword_rules", []):
        if str(rule.get("match", "")).lower() in n:
            return {
                "firing_hz": tuple(rule["firing_hz"]),
                "v_min": tuple(rule["v_min"]),
                "v_max": tuple(rule["v_max"]),
            }
    default = reference.get("default", {})
    return {
        "firing_hz": tuple(default.get("firing_hz", [0.0, 250.0])),
        "v_min": tuple(default.get("v_min", [-130.0, -40.0])),
        "v_max": tuple(default.get("v_max", [0.0, 120.0])),
    }


def _preset_param_sanity(cfg) -> list[str]:
    ch = cfg.channels
    mc = cfg.morphology
    issues: list[str] = []
    if not (ch.gNa_max > 0 and ch.gK_max > 0 and ch.gL > 0):
        issues.append("non-positive core conductance in preset")
    if not (ch.Cm > 0):
        issues.append("non-positive membrane capacitance")
    if not (0.0 < mc.Ra < 5000.0):
        issues.append("Ra outside broad plausible range")
    if not (-120.0 <= ch.EL <= -30.0):
        issues.append("EL outside broad physiological range")
    if not (0.0 <= ch.ENa <= 100.0):
        issues.append("ENa outside broad physiological range")
    if not (-130.0 <= ch.EK <= -40.0):
        issues.append("EK outside broad physiological range")
    return issues


def _run_case(
    preset: str,
    iext_scale: float,
    t_sim_ms: float,
    temp_c: float,
    noise_sigma: float,
    reference: dict,
) -> StressCase:
    cfg = FullModelConfig()
    apply_preset(cfg, preset)
    cfg.stim.Iext *= float(iext_scale)
    cfg.stim.t_sim = float(t_sim_ms)
    cfg.env.T_celsius = float(temp_c)
    cfg.stim.noise_sigma = float(noise_sigma)
    param_issues = _preset_param_sanity(cfg)

    notes: list[str] = []
    if param_issues:
        notes.extend(param_issues)
    try:
        res = NeuronSolver(cfg).run_single()
        v = np.asarray(res.v_soma, dtype=float)
        finite = np.isfinite(v)
        has_nan_inf = not bool(np.all(finite))
        if has_nan_inf:
            notes.append("non-finite voltage trace")
            return StressCase(
                preset, iext_scale, t_sim_ms, temp_c, noise_sigma,
                float(np.nanmin(v)), float(np.nanmax(v)), 0, float("nan"),
                True, "FAIL", "; ".join(notes)
            )
        pks, sp_t, _ = detect_spikes(v, np.asarray(res.t, dtype=float))
        n_sp = int(len(pks))
        firing_hz = 0.0
        if len(res.t) > 1 and n_sp > 1:
            dur_s = float(res.t[-1] - res.t[0]) / 1000.0
            if dur_s > 0:
                firing_hz = n_sp / dur_s

        v_min = float(np.min(v))
        v_max = float(np.max(v))

        status = "PASS"
        if notes:
            status = "WARN"

        if v_min < -140.0 or v_max > 120.0:
            status = "WARN"
            notes.append("voltage bounds outside global broad physiological envelope")
        if firing_hz > 600.0:
            status = "WARN"
            notes.append("extreme firing rate")

        expected = _expected_ranges_for_preset(preset, reference)
        f_lo, f_hi = expected["firing_hz"]
        vmin_lo, vmin_hi = expected["v_min"]
        vmax_lo, vmax_hi = expected["v_max"]
        if not (f_lo <= firing_hz <= f_hi):
            status = "WARN"
            notes.append(f"firing_hz outside expected range [{f_lo}, {f_hi}]")
        if not (vmin_lo <= v_min <= vmin_hi):
            status = "WARN"
            notes.append(f"v_min outside expected range [{vmin_lo}, {vmin_hi}]")
        if not (vmax_lo <= v_max <= vmax_hi):
            status = "WARN"
            notes.append(f"v_max outside expected range [{vmax_lo}, {vmax_hi}]")

        return StressCase(
            preset, iext_scale, t_sim_ms, temp_c, noise_sigma,
            v_min, v_max, n_sp, float(firing_hz), False, status, "; ".join(notes)
        )
    except Exception as exc:  # pragma: no cover - stress utility path
        notes.append(f"exception: {type(exc).__name__}")
        return StressCase(
            preset, iext_scale, t_sim_ms, temp_c, noise_sigma,
            float("nan"), float("nan"), 0, float("nan"),
            True, "FAIL", "; ".join(notes)
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Run wide-range stress validation for all presets.")
    ap.add_argument("--out", default="tests/artifacts/preset_stress_validation.json")
    ap.add_argument("--report-md", default="tests/artifacts/preset_stress_validation.md")
    ap.add_argument("--reference", default="tests/utils/preset_reference_ranges.json")
    ap.add_argument("--limit-presets", type=int, default=0, help="Optional cap for quick local runs.")
    ap.add_argument(
        "--fail-on-fail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Return non-zero when FAIL rows exist (default: enabled).",
    )
    ap.add_argument(
        "--fail-on-warn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Return non-zero when WARN rows exist (default: disabled).",
    )
    args = ap.parse_args()

    if _IMPORT_ERROR is not None:
        msg = dependency_diagnostic("preset-stress", _IMPORT_ERROR)
        print(msg)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "status": "dependency_error",
                    "tool": "preset-stress",
                    "message": msg,
                    "reference_path": args.reference,
                    "gate": {
                        "fail_on_fail": bool(args.fail_on_fail),
                        "fail_on_warn": bool(args.fail_on_warn),
                        "exit_code": 2,
                    },
                    "rows": [],
                    "by_preset": {},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        report_path = Path(args.report_md)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            "# Preset stress validation\n\n"
            "Status: dependency_error\n\n"
            f"{msg}\n",
            encoding="utf-8",
        )
        return 2

    reference = _load_reference_ranges(Path(args.reference))

    presets = list(get_preset_names())
    if args.limit_presets > 0:
        presets = presets[: args.limit_presets]

    iext_scales = (0.5, 1.0, 1.5)
    t_sims = (300.0, 800.0, 1500.0)
    temps = (22.0, 30.0, 37.0)
    noises = (0.0, 0.5)

    rows: list[StressCase] = []
    for p in presets:
        for s in iext_scales:
            for t_sim in t_sims:
                for tc in temps:
                    for ns in noises:
                        rows.append(_run_case(p, s, t_sim, tc, ns, reference))

    summary = {
        "status": _overall_status(
            fail_count=sum(r.status == "FAIL" for r in rows),
            warn_count=sum(r.status == "WARN" for r in rows),
        ),
        "total": len(rows),
        "pass": sum(r.status == "PASS" for r in rows),
        "warn": sum(r.status == "WARN" for r in rows),
        "fail": sum(r.status == "FAIL" for r in rows),
        "reference_path": args.reference,
        "gate": {
            "fail_on_fail": bool(args.fail_on_fail),
            "fail_on_warn": bool(args.fail_on_warn),
        },
        "rows": [asdict(r) for r in rows],
    }
    by_preset: dict[str, dict[str, int]] = {}
    for r in rows:
        cur = by_preset.setdefault(r.preset, {"PASS": 0, "WARN": 0, "FAIL": 0})
        cur[r.status] += 1
    summary["by_preset"] = by_preset

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    report_md = Path(args.report_md)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Preset Stress Validation Report",
        "",
        f"- Total cases: **{summary['total']}**",
        f"- Overall status: **{summary['status']}**",
        f"- PASS: **{summary['pass']}**",
        f"- WARN: **{summary['warn']}**",
        f"- FAIL: **{summary['fail']}**",
        "",
        "## By Preset",
        "",
        "| Preset | PASS | WARN | FAIL |",
        "|---|---:|---:|---:|",
    ]
    for preset, counts in sorted(summary["by_preset"].items()):
        lines.append(f"| {preset} | {counts['PASS']} | {counts['WARN']} | {counts['FAIL']} |")
    lines.extend(["", "## WARN/FAIL samples", ""])
    bad_rows = [r for r in summary["rows"] if r["status"] != "PASS"][:50]
    if not bad_rows:
        lines.append("- None")
    else:
        for r in bad_rows:
            lines.append(
                f"- `{r['preset']}` | status={r['status']} | "
                f"Iext×{r['iext_scale']}, T={r['temp_c']}°C, t_sim={r['t_sim_ms']} ms | {r['note']}"
            )
    report_md.write_text("\n".join(lines), encoding="utf-8")

    exit_code = _exit_code_for_summary(
        fail_count=int(summary["fail"]),
        warn_count=int(summary["warn"]),
        fail_on_fail=bool(args.fail_on_fail),
        fail_on_warn=bool(args.fail_on_warn),
    )
    summary["gate"]["exit_code"] = int(exit_code)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[preset-stress] wrote {out} | PASS={summary['pass']} WARN={summary['warn']} FAIL={summary['fail']}")
    print(f"[preset-stress] wrote {report_md}")
    if exit_code != 0:
        print("[preset-stress] Result status: FAIL gate triggered.")
    else:
        print("[preset-stress] Result status: PASS/WARN gate not triggered.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
