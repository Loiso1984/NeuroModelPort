# UI Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize the current PySide6 dock-based UI architecture for laptop-first scientific work while preserving lazy analytics tabs and Pydantic-driven forms.

**Architecture:** Keep the v13 dock cockpit, lazy tab construction, async analytics, and Pydantic model binding. Add explicit session/layout state validation, a laptop-first default shell, a thin UX metadata layer over generated forms, finite guards for analytics widgets, and updated GUI branch tests that match the dock architecture.

**Tech Stack:** Python, PySide6, pyqtgraph, matplotlib, Pydantic, pytest/offscreen Qt branch tests.

---

### Task 1: Session Restore And Preset Identity

**Files:**
- Modify: `gui/config_manager.py`
- Modify: `gui/main_window.py`
- Test: `tests/branches/test_gui_window_geometry_branch.py`

- [ ] Add tests proving restored JSON configs show a usable custom/known preset state and corrupted dock restore falls back to a visible laptop layout.
- [ ] Implement `ConfigManager.mark_custom_config()` and optional preset-name restoration.
- [ ] Add `MainWindow._restore_session_or_default()` and `MainWindow._restore_or_reset_dock_layout()` so startup never leaves preset selectors at a misleading placeholder with hidden primary docks.
- [ ] Verify with `python tests/branches/test_gui_window_geometry_branch.py`.

### Task 2: Laptop-First Shell And Layout Presets

**Files:**
- Modify: `gui/main_window.py`
- Create: `gui/ui_layout.py`
- Test: `tests/branches/test_gui_window_geometry_branch.py`

- [ ] Add a small layout helper module with named presets: `Laptop`, `Desktop`, `Presentation`, `Debug`.
- [ ] Make laptop dimensions the default startup contract and keep primary actions visible at 1100x700.
- [ ] Move secondary actions behind a compact action menu or hide them under laptop preset.
- [ ] Verify with offscreen geometry tests and a screenshot smoke.

### Task 3: Pydantic UX Layer

**Files:**
- Modify: `gui/widgets/form_generator.py`
- Modify: `gui/main_window.py`
- Test: `tests/branches/test_gui_pydantic_ux_branch.py`

- [ ] Add optional field metadata controls: priority, compact labels, search filtering, and priority filtering.
- [ ] Keep generated forms bound to the live Pydantic instances.
- [ ] Add a sidebar search and priority selector without replacing the generated forms.
- [ ] Verify hidden/filtered fields do not mutate model state.

### Task 4: Stimulation And Dual-Stim Contract

**Files:**
- Modify: `tests/branches/test_gui_stim_sync_branch.py`
- Modify: `gui/main_window.py`
- Modify: `gui/config_manager.py`

- [ ] Update tests from legacy `w.config` and removed dual-primary fields to current `config_manager.config` and `core.dual_stimulation.DualStimulationConfig`.
- [ ] Ensure dual-stim enabled state mirrors active stimulation in read-only preview and preset load resets/syncs correctly.
- [ ] Verify with `python tests/branches/test_gui_stim_sync_branch.py`.

### Task 5: Analytics Finite Guards And Lifecycle

**Files:**
- Modify: `gui/analytics.py`
- Test: `tests/branches/test_gui_fullscreen_analytics_branch.py`

- [ ] Add finite-value guards for progress bars and numeric widgets.
- [ ] Keep lazy tab construction intact.
- [ ] Ensure fullscreen analytics clone handles NaN stats and closes without retained windows.
- [ ] Verify with `python tests/branches/test_gui_fullscreen_analytics_branch.py`.

### Task 6: Modularization Helpers

**Files:**
- Create: `gui/ui_layout.py`
- Create: `gui/pydantic_form_meta.py`
- Modify: `gui/main_window.py`
- Modify: `gui/widgets/form_generator.py`

- [ ] Extract constants and helper functions only; do not perform a risky wholesale split of `MainWindow` in the same pass.
- [ ] Keep behavior stable while creating a clear path for later file decomposition.
- [ ] Verify focused GUI tests still pass.

### Task 7: Final Verification

**Files:**
- All touched files.

- [ ] Run `python -m py_compile gui/main_window.py gui/analytics.py gui/plots.py gui/topology.py gui/config_manager.py gui/widgets/form_generator.py gui/ui_layout.py gui/pydantic_form_meta.py`.
- [ ] Run focused GUI branch tests:
  - `python tests/branches/test_gui_window_geometry_branch.py`
  - `python tests/branches/test_gui_stim_sync_branch.py`
  - `python tests/branches/test_gui_fullscreen_analytics_branch.py`
  - `python tests/branches/test_gui_readability_controls_branch.py`
  - `python tests/branches/test_gui_fullscreen_plot_branch.py`
  - `python tests/branches/test_advanced_sim_progress_callbacks_branch.py`
- [ ] Report exact passes/failures and any residual risks.
