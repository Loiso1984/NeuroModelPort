# Utils Registry

This registry separates active deterministic validators from legacy/compatibility scripts.

## Active Deterministic Utilities

- `run_active_branch_suite.py`
- `run_unified_preset_protocol.py`
- `run_preset_stress_matrix.py`
- `run_pathology_focus_report.py`
- `run_pathology_worstcase_followup.py`
- `run_f_conduction_extended.py`
- `run_cde_extended_report.py`
- `run_hcn_ia_extended_report.py`
- `run_calcium_nernst_extended_report.py`
- `run_dual_stim_extended_report.py`
- `hypoxia_deterministic_search.py`
- `benchmark_jacobian_modes.py`
- `run_impedance_zap_report.py`
  - supports strict gate mode (`--strict`) and configurable analysis band (`--fmin/--fmax`).
  - emits `guard_reasons` (per row) and `failed_case_ids` (artifact-level) for triage.

## Legacy / Compatibility Utilities

These scripts are kept for historical comparison or compatibility, but should not be used as primary gate validators.

- `calibrate_presets.py`
  Reason: uses legacy sample-count style frequency estimation and non-branch protocol.
- `validate_alpha_presets_v10_1.py`
  Reason: legacy validation generation script.
- `validate_alpha_targets_v10_1.py`
  Reason: legacy target checker.
- `parallel_mode_sweep.py`
  Reason: exploratory helper; not part of active deterministic gate.
- `validate_stimulus_location_routes.py`
  Reason: compatibility wrapper; canonical implementation is `validate_stimulus_location_routes_fixed.py`.
- `validate_stimulus_location_routes_fixed.py`
  Reason: legacy-style targeted checker, not part of active branch gate.

## Rule

For preset/core-calculation promotion decisions, use the active deterministic utilities and branch suite artifacts in `_test_results`.
