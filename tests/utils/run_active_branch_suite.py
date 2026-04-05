"""
Run active branch validation suite and produce a compact JSON report.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import subprocess
import sys
import time
from pathlib import Path


ACTIVE_TESTS = [
    "tests/branches/test_unified_preset_protocol_branch.py",
    "tests/branches/test_channel_physiology_branch.py",
    "tests/branches/test_dual_stim_branch.py",
    "tests/branches/test_spike_detection_math_branch.py",
    "tests/branches/test_jacobian_modes_branch.py",
    "tests/branches/test_multichannel_stress_branch.py",
    "tests/branches/test_cde_profiles_branch.py",
    "tests/branches/test_advanced_sim_progress_callbacks_branch.py",
    "tests/branches/test_lyapunov_analysis_branch.py",
    "tests/branches/test_modulation_decomposition_branch.py",
    "tests/branches/test_stimulus_trace_overlay_branch.py",
    "tests/branches/test_spike_mechanism_analytics_branch.py",
    "tests/branches/test_delay_target_sync_branch.py",
    "tests/branches/test_gui_jacobian_autoselect_branch.py",
    "tests/branches/test_gui_stim_sync_branch.py",
    "tests/branches/test_gui_fullscreen_plot_branch.py",
    "tests/branches/test_gui_fullscreen_analytics_branch.py",
    "tests/branches/test_gui_fullscreen_topology_branch.py",
    "tests/branches/test_gui_readability_controls_branch.py",
    "tests/branches/test_gui_window_geometry_branch.py",
    "tests/branches/test_passport_ml_classification_branch.py",
    "tests/branches/test_pathology_mode_sweep_branch.py",
    "tests/branches/test_solver_validation_branch.py",
    "tests/branches/test_impedance_zap_branch.py",
]

ACTIVE_UTILS = [
    {
        "name": "impedance_zap_strict_gate",
        "cmd": [
            "tests/utils/run_impedance_zap_report.py",
            "--strict",
            "--print-failures",
        ],
    },
]


def _run_single_test(test_file: str) -> dict:
    ts = time.perf_counter()
    p = subprocess.run([sys.executable, test_file], capture_output=True, text=True)
    elapsed = time.perf_counter() - ts
    return {
        "test_file": test_file,
        "returncode": p.returncode,
        "elapsed_sec": elapsed,
        "stdout_tail": (p.stdout or "").strip().splitlines()[-10:],
        "stderr_tail": (p.stderr or "").strip().splitlines()[-10:],
    }


def _run_single_utility(name: str, script_and_args: list[str]) -> dict:
    ts = time.perf_counter()
    p = subprocess.run([sys.executable, *script_and_args], capture_output=True, text=True)
    elapsed = time.perf_counter() - ts

    stdout_tail = (p.stdout or "").strip().splitlines()[-10:]
    stderr_tail = (p.stderr or "").strip().splitlines()[-10:]
    tail = "\n".join([*stdout_tail, *stderr_tail])
    is_env_warn = p.returncode == 2 and "[WARN] missing dependency" in tail
    status = "warn" if is_env_warn else ("pass" if p.returncode == 0 else "fail")
    return {
        "name": name,
        "cmd": [sys.executable, *script_and_args],
        "returncode": p.returncode,
        "status": status,
        "elapsed_sec": elapsed,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run active branch validation suite")
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel workers for independent test scripts",
    )
    parser.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="Treat utility warning status as suite failure",
    )
    args = parser.parse_args()

    out_dir = Path("_test_results")
    out_dir.mkdir(exist_ok=True)

    rows = []
    util_rows = []
    all_ok = True
    t0 = time.perf_counter()
    workers = max(1, int(args.workers))
    if workers == 1:
        for test_file in ACTIVE_TESTS:
            row = _run_single_test(test_file)
            rows.append(row)
            print(f"{test_file}: rc={row['returncode']} time={row['elapsed_sec']:.1f}s")
            if row["returncode"] != 0:
                all_ok = False
    else:
        by_file = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut_map = {ex.submit(_run_single_test, tf): tf for tf in ACTIVE_TESTS}
            for fut in as_completed(fut_map):
                row = fut.result()
                by_file[row["test_file"]] = row
        for test_file in ACTIVE_TESTS:
            row = by_file[test_file]
            rows.append(row)
            print(f"{test_file}: rc={row['returncode']} time={row['elapsed_sec']:.1f}s")
            if row["returncode"] != 0:
                all_ok = False

    for util in ACTIVE_UTILS:
        row = _run_single_utility(util["name"], util["cmd"])
        util_rows.append(row)
        print(f"{util['name']}: status={row['status']} rc={row['returncode']} time={row['elapsed_sec']:.1f}s")
        if row["status"] == "fail":
            all_ok = False
        if args.fail_on_warn and row["status"] == "warn":
            all_ok = False

    utility_warn_count = int(sum(1 for r in util_rows if r.get("status") == "warn"))
    utility_fail_count = int(sum(1 for r in util_rows if r.get("status") == "fail"))

    report = {
        "suite": "active_branch_validation",
        "workers": workers,
        "fail_on_warn": bool(args.fail_on_warn),
        "all_ok": all_ok,
        "elapsed_total_sec": time.perf_counter() - t0,
        "tests": rows,
        "utility_checks": util_rows,
        "utility_warn_count": utility_warn_count,
        "utility_fail_count": utility_fail_count,
    }
    out_file = out_dir / "active_branch_suite.json"
    out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out_file}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
