"""
Run active branch validation suite and produce a compact JSON report.
"""

from __future__ import annotations

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
    "tests/branches/test_lyapunov_analysis_branch.py",
    "tests/branches/test_modulation_decomposition_branch.py",
]


def main() -> int:
    out_dir = Path("_test_results")
    out_dir.mkdir(exist_ok=True)

    rows = []
    all_ok = True
    t0 = time.perf_counter()
    for test_file in ACTIVE_TESTS:
        ts = time.perf_counter()
        p = subprocess.run([sys.executable, test_file], capture_output=True, text=True)
        elapsed = time.perf_counter() - ts
        row = {
            "test_file": test_file,
            "returncode": p.returncode,
            "elapsed_sec": elapsed,
            "stdout_tail": (p.stdout or "").strip().splitlines()[-10:],
            "stderr_tail": (p.stderr or "").strip().splitlines()[-10:],
        }
        rows.append(row)
        print(f"{test_file}: rc={p.returncode} time={elapsed:.1f}s")
        if p.returncode != 0:
            all_ok = False

    report = {
        "suite": "active_branch_validation",
        "all_ok": all_ok,
        "elapsed_total_sec": time.perf_counter() - t0,
        "tests": rows,
    }
    out_file = out_dir / "active_branch_suite.json"
    out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out_file}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
