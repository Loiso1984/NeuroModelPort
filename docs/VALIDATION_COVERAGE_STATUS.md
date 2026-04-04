# Validation Coverage Status (Master 0-16)

Current status snapshot after latest branch-contour runs.

## Closed in validation contour

- `1` Calcium/Nernst calibration checks (core acceptance layer): active branch tests + extended deterministic report artifacts.
- `2` HCN validation (rest stability, V1/2, temperature behavior) and IA validation (suppression trends) in branch contour.
- `2.1` Dual-stim computational validation (branch tests + extended deterministic dual-stim report).
- `4` Dual-stimulation refactor to dedicated module (`core/dual_stimulation.py`) completed earlier and covered by branch tests.
- `6` Multichannel stress test branch coverage (`Ih+ICa`, `IA+SK`, all channels) plus broad preset stress matrix artifacts.
- Pathology mode validation core loop (`N/O/F`) currently green on focus + worst-case follow-up + extended F-vs-D conduction checks.

## Partially closed / continuing

- `0` Test grouping and cleanup: active contour grouped/cleaned; root ad-hoc + legacy `test_*`/`validate_*` helpers archived under `tests/archive/root_legacy_tools`; only final optional pruning remains.
- `3` Configurable spike detection in GUI: integrated and branch-tested; iterative GUI UX improvements still possible.
- GUI clarity hardening in progress:
  - fixed SD/Excitability worker callback crash path,
  - callback contract regression test added (`test_advanced_sim_progress_callbacks_branch.py`),
  - dual-stim now resets to disabled default on preset load and explicitly overrides primary stimulation fields only when enabled,
  - preset mode selectors are context-scoped to relevant K/N/O presets,
  - setup/dual-stim/analysis tab order and setup grouping were reworked for clearer workflow,
  - post-run tab focusing now uses explicit widget targets (no stale index jumps after tab reordering).
- `5` Error handling/logging/warnings: core layer implemented and tested; can be expanded with richer user-facing diagnostics.
- K/N/O pathology UX/documentation guidance: mode switches + guide + preflight/status-mode warnings integrated; final bilingual polish still needed.

## Not closed yet (planned follow-up)

- Deferred: targeted `C/L` baseline stimulation refinement pass (kept out of current high-priority lane).
- `7` Plot readability/interactive improvements (beyond current export integration).
  - progress update: theme + linewidth + spike/delay overlays + title-font/grid controls integrated in oscilloscope.
  - delay visualization now supports target selection (terminal / AIS / junction / custom compartment),
  - delay target controls are now preset-synced before first run (custom index range available pre-simulation).
- `8` Neuron passport with ML classification.
  - progress update: passport now includes Lyapunov/modulation blocks, channel-engagement/delay summaries,
    and a first hybrid rule+ML classifier baseline (prototype ML + confidence + source tag).
- `9` Topology/axon propagation visualization upgrades.
  - progress update: topology now includes compartment-index labeling and compact index map in info bar for better axonal targeting,
  - topology delay focus is now linked to oscilloscope delay-target selection (shared cross-tab context),
  - fork/index semantics aligned with morphology indexing across GUI + analytics + core conduction extraction and active validation utility reports; branch regression test added (`test_delay_target_sync_branch.py`).
- `10` Real-time sliders with immediate recompute.
  - progress update: lightweight real-time preview is now active for setup edits (stimulation/location/filter/channel/morphology) by instant topology redraw without solver rerun.
- `11` Extended export workflows polishing.
- `12` Full doc rewrite/reorganization (user+developer).
- `13` Further compute optimization beyond current Jacobian speedup path.
- `14` Full cleanup + bilingual polish final pass.
- `15` Additional future improvements (ongoing bucket).
- `16` Lyapunov practical integration and UX polishing (core feature + Passport summary integration added; final full UX polishing still pending).

## Current gating principle

Preset/core-calculation changes continue to follow:
1. branch contour first,
2. deterministic utility artifacts,
3. only then promotion to primary program defaults.

After full physiology validation is explicitly closed, analytics/plots/GUI/documentation tasks may proceed directly in the main contour; this exception does not apply to preset/channel/core-physiology logic.
