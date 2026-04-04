# NeuroModelPort Tests: Current Workflow

This document reflects the *current* local validation flow used in active development.

## Canonical Validation Entry Points

1. Active branch gate (main daily check):
   - `python tests/utils/run_active_branch_suite.py --workers 2`
2. Unified physiology protocol report:
   - `python tests/utils/run_unified_preset_protocol.py`
   - artifact: `_test_results/unified_preset_protocol.json`
3. Targeted heavy hypoxia search (deterministic):
   - `python tests/utils/hypoxia_deterministic_search.py ...`
   - default is sequential for stability/performance; threaded mode requires `--allow-parallel`.
4. All-preset deterministic stress matrix:
   - `python tests/utils/run_preset_stress_matrix.py --i-scales 0.8,1.0 --temps 23,37`
   - default uses fast single-comp proxy screening; use `--multicomp` only for expensive full cable stress runs.
5. Pathology-focused validation report (N/O/F):
   - `python tests/utils/run_pathology_focus_report.py`
   - artifact: `_test_results/pathology_focus_report.json`
   - includes mode-order checks (`progressive` vs `terminal`) and F conduction signature vs D control.
6. Pathology worst-case follow-up (strict recheck):
   - `python tests/utils/run_pathology_worstcase_followup.py --top-k 6`
   - reads hard-matrix candidates and reruns near-boundary cases with stricter settings,
   - artifact: `_test_results/pathology_worstcase_followup.json`.
7. Extended F-vs-D conduction grid:
   - `python tests/utils/run_f_conduction_extended.py ...`
   - validates demyelination conduction signature over configurable `T / Ra / gL / drive` grid,
   - artifact: `_test_results/pathology_f_conduction_extended.json`.
8. Extended C/D/E operating report:
   - `python tests/utils/run_cde_extended_report.py ...`
   - deterministic `I_scale x temperature` sweep for C/D/E with physiology flags,
   - artifact: `_test_results/cde_extended_report.json`.

## Directory Roles

- `tests/branches/`
  - Active validation gate for preset logic, channel physiology, Jacobian modes,
    dual stimulation, spike math, pathology mode sweeps, stress checks.
  - Legacy branch scripts are isolated under `tests/branches/legacy/`.
- `tests/utils/`
  - Protocol runners, benchmarks, deterministic calibration/search utilities.
- `tests/core/`, `tests/presets/`, `tests/stress/`, `tests/validation/`
  - Historical and support tests/tools. Useful, but not the primary active gate.
- `tests/archive/`
  - Archived artifacts/debug scripts.

## Important Notes

1. `pytest` may be unavailable in some local environments.
   - Active branch tests are executable directly via `python tests/branches/<script>.py`.
2. For changes in presets or core calculations:
   - update/add branch tests first,
   - run active suite,
   - only then promote logic/preset updates.
3. Spike detection must remain transition-based and physiologically validated.
4. Pathology checks must preserve mode ordering:
   - `K activated >= baseline`,
   - `N/O terminal <= progressive`.

## Runtime Expectations (current machine, approximate)

- `run_active_branch_suite.py --workers 2`: ~6-7 minutes.
- `run_unified_preset_protocol.py`: ~3-4 minutes.
- `hypoxia_deterministic_search.py` (small coarse run): ~1.5 minutes.

## Hygiene Rules

1. Keep `tests/**/__pycache__` out of committed test artifacts.
2. Avoid duplicate utilities; keep compatibility wrappers only when needed.
3. Prefer deterministic sweeps over random search for reproducibility.

## Output Artifacts

- `_test_results/active_branch_suite.json`
- `_test_results/unified_preset_protocol.json`
- `_test_results/hypoxia_deterministic_search.json`
- `_test_results/hypoxia_deterministic_search.jsonl`
- `_test_results/preset_stress_matrix.json`
- `_test_results/pathology_focus_report.json`
- `_test_results/pathology_hard_matrix.json`
- `_test_results/pathology_worstcase_followup.json`
- `_test_results/pathology_f_conduction_extended.json`
- `_test_results/cde_extended_report.json`

Last updated: 2026-04-04
