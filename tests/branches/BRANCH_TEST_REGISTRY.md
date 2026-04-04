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
- `tests/branches/test_advanced_sim_progress_callbacks_branch.py`
- `tests/branches/test_lyapunov_analysis_branch.py`
- `tests/branches/test_modulation_decomposition_branch.py`
- `tests/branches/test_stimulus_trace_overlay_branch.py`
- `tests/branches/test_spike_mechanism_analytics_branch.py`
- `tests/branches/test_delay_target_sync_branch.py`
- `tests/branches/test_gui_jacobian_autoselect_branch.py`
- `tests/branches/test_gui_stim_sync_branch.py`
- `tests/branches/test_gui_fullscreen_plot_branch.py`
- `tests/branches/test_gui_fullscreen_analytics_branch.py`
- `tests/branches/test_passport_ml_classification_branch.py`
- `tests/branches/test_pathology_mode_sweep_branch.py`
- `tests/branches/test_solver_validation_branch.py`

These are the current source of truth for branch-gated physiology validation.

## Legacy Exploratory Scripts (Not Gate-Criteria)

- `tests/branches/legacy/test_hcn_isolated_branch.py`
- `tests/branches/legacy/test_ia_isolated_branch.py`
- `tests/branches/legacy/test_preset_calibration_branch.py`
- `tests/branches/legacy/test_fine_preset_calibration_branch.py`
- `tests/branches/legacy/test_spike_detection_branch.py`

These scripts are kept for exploratory diagnostics and historical context.
They contain older assumptions/targets and should not be used as pass/fail gate
for current validation without explicit recalibration.
