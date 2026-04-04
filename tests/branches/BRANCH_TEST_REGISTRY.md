# Branch Test Registry

This registry separates **active validation tests** from **legacy exploratory scripts**.

## Active Validation Suite

- `tests/branches/test_unified_preset_protocol_branch.py`
- `tests/branches/test_channel_physiology_branch.py`
- `tests/branches/test_dual_stim_branch.py`
- `tests/branches/test_spike_detection_math_branch.py`
- `tests/branches/test_jacobian_modes_branch.py`
- `tests/branches/test_multichannel_stress_branch.py`
- `tests/branches/test_cde_profiles_branch.py`
- `tests/branches/test_lyapunov_analysis_branch.py`
- `tests/branches/test_modulation_decomposition_branch.py`

These are the current source of truth for branch-gated physiology validation.

## Legacy Exploratory Scripts (Not Gate-Criteria)

- `tests/branches/test_hcn_isolated_branch.py`
- `tests/branches/test_ia_isolated_branch.py`
- `tests/branches/test_preset_calibration_branch.py`
- `tests/branches/test_fine_preset_calibration_branch.py`
- `tests/branches/test_spike_detection_branch.py`

These scripts are kept for exploratory diagnostics and historical context.
They contain older assumptions/targets and should not be used as pass/fail gate
for current validation without explicit recalibration.
