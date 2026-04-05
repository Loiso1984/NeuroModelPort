from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class StepResult:
    name: str
    command: list[str]
    exit_code: int
    status: str
    stdout_tail: str
    stderr_tail: str


def _classify_exit_code(code: int) -> str:
    if code == 0:
        return "PASS"
    if code == 1:
        return "FAIL"
    if code == 2:
        return "WARN_DEPENDENCY"
    return "ERROR"


def _run_step(name: str, cmd: list[str]) -> StepResult:
    repo_root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))
    out_tail = "\n".join(proc.stdout.splitlines()[-30:])
    err_tail = "\n".join(proc.stderr.splitlines()[-30:])
    return StepResult(
        name=name,
        command=cmd,
        exit_code=int(proc.returncode),
        status=_classify_exit_code(int(proc.returncode)),
        stdout_tail=out_tail,
        stderr_tail=err_tail,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Run consolidated P0/P1 gate checks.")
    ap.add_argument("--out-dir", default="tests/artifacts/p0_p1_gate")
    ap.add_argument("--target-ratio", type=float, default=0.3)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = (repo_root / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = [
        (
            "core_contract_suite",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                str(repo_root / "tests/core/test_rhs_contract.py"),
                str(repo_root / "tests/core/test_jacobian_contract.py"),
                str(repo_root / "tests/core/test_dual_stimulation_distribution.py"),
                str(repo_root / "tests/core/test_delay_target_utils.py"),
            ],
        ),
        (
            "f_conduction_gate",
            [
                sys.executable,
                str(repo_root / "tests/utils/run_f_conduction_extended.py"),
                "--target-ratio",
                str(args.target_ratio),
                "--output",
                str(out_dir / "f_conduction_gate.json"),
            ],
        ),
        (
            "preset_stress_gate",
            [
                sys.executable,
                str(repo_root / "tests/utils/run_preset_stress_validation.py"),
                "--out",
                str(out_dir / "preset_stress_gate.json"),
                "--report-md",
                str(out_dir / "preset_stress_gate.md"),
                "--dt-eval",
                "0.2",
            ],
        ),
    ]

    results = [_run_step(name, cmd) for name, cmd in steps]

    blocking = [r for r in results if r.status in {"FAIL", "ERROR"}]
    summary = {
        "status": "FAIL" if blocking else "PASS",
        "blocking_steps": [r.name for r in blocking],
        "results": [asdict(r) for r in results],
    }

    summary_path = out_dir / "p0_p1_gate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# P0/P1 Gate Summary",
        "",
        f"- Overall status: **{summary['status']}**",
        f"- Blocking steps: **{', '.join(summary['blocking_steps']) if summary['blocking_steps'] else 'none'}**",
        "",
        "## Steps",
        "",
        "| Step | Status | Exit code |",
        "|---|---|---:|",
    ]
    for r in results:
        md_lines.append(f"| {r.name} | {r.status} | {r.exit_code} |")
    (out_dir / "p0_p1_gate_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[p0-p1-gate] wrote {summary_path}")
    print(f"[p0-p1-gate] overall={summary['status']}")
    return 1 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
