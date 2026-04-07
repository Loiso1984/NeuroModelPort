"""Run a compact high-priority physiology validation cluster.

The cluster focuses on the currently highest-priority physiology lanes and
produces a single JSON artifact summarizing pass/warn/fail outcomes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ARTIFACT = Path("_test_results/priority_physiology_cluster.json")

TASKS = [
    {
        "id": "unified_protocol",
        "cmd": ["tests/utils/run_unified_preset_protocol.py"],
        "priority": "high",
    },
    {
        "id": "pathology_focus",
        "cmd": ["tests/utils/run_pathology_focus_report.py"],
        "priority": "high",
    },
    {
        "id": "f_conduction_extended",
        "cmd": ["tests/utils/run_f_conduction_extended.py", "--temps", "37", "--i-scales", "1.0", "--ra-scales", "1.0"],
        "priority": "critical",
    },
]


def _run_task(task: dict) -> dict:
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, *task["cmd"]], capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    stdout_tail = (p.stdout or "").strip().splitlines()[-12:]
    stderr_tail = (p.stderr or "").strip().splitlines()[-12:]
    tail_blob = "\n".join([*stdout_tail, *stderr_tail])
    is_dep_warn = ("No module named 'pydantic'" in tail_blob) or ("[WARN] missing dependency" in tail_blob)
    status = "warn" if is_dep_warn else ("pass" if p.returncode == 0 else "fail")
    return {
        "id": task["id"],
        "priority": task["priority"],
        "cmd": [sys.executable, *task["cmd"]],
        "returncode": p.returncode,
        "status": status,
        "elapsed_sec": elapsed,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run prioritized physiology utility cluster")
    parser.add_argument("--fail-on-warn", action="store_true", help="Treat warning outcomes as failure")
    parser.add_argument("--strict-critical", action="store_true", help="Require critical-priority tasks to pass")
    parser.add_argument("--output", type=str, default=str(ARTIFACT), help="Output JSON artifact path")
    args = parser.parse_args()

    rows = [_run_task(task) for task in TASKS]
    fail_count = int(sum(1 for r in rows if r["status"] == "fail"))
    warn_count = int(sum(1 for r in rows if r["status"] == "warn"))
    pass_count = int(sum(1 for r in rows if r["status"] == "pass"))
    critical_rows = [r for r in rows if r["priority"] == "critical"]
    critical_ok = all(r["status"] == "pass" for r in critical_rows)
    all_ok = (fail_count == 0) and (warn_count == 0 if args.fail_on_warn else True)
    if args.strict_critical and not critical_ok:
        all_ok = False

    next_actions: list[str] = []
    if all(r["status"] == "warn" for r in rows):
        next_actions.append("Unblock environment dependencies (pydantic) to run physiology utilities.")
    if not critical_ok:
        next_actions.append("Prioritize F-conduction lane until critical task reaches pass status.")
    if all(r["status"] == "pass" for r in rows):
        next_actions.append("Proceed with O/N mode tuning and promote validated changes branch-first.")
    if not next_actions:
        next_actions.append("Investigate failed tasks first, then rerun cluster with --strict-critical.")

    artifact = {
        "rows": rows,
        "pass_count": pass_count,
        "warn_count": warn_count,
        "fail_count": fail_count,
        "critical_ok": critical_ok,
        "strict_critical": bool(args.strict_critical),
        "next_actions": next_actions,
        "all_ok": all_ok,
        "fail_on_warn": bool(args.fail_on_warn),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print(f"Saved: {out}")
    print(f"Pass/Warn/Fail: {pass_count}/{warn_count}/{fail_count}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
