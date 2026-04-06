# Validation Coverage Status (Master 0-16)

Current status snapshot after latest branch-contour runs.

## v10.2 Critique Intake (newly confirmed priority)

Based on current code audit versus v10.2 critique, these items are explicitly treated as open and priority-ordered:

- `P0` RHS hot-path allocation removal in `core/rhs.py` + solver/jacobian contract synchronization.
- `P0` GUI analytics update-cycle memory hardening (remove repeated clear/cla in hot paths, shift to persistent artists).
- `P0` Preset F (MS) demyelination severity recalibration until conduction ratio reflects pathological attenuation.
- `P1` RHS argument explosion refactor (reduce positional-argument fragility).
- `P1` Calcium upper-bound safety clamp for long pathological runs.
- Addendum (2026-04-05) confirmed by post-implementation review:
  - keep `P0` on GUI update-cycle memory discipline (no hot-path artist/axes recreation),
  - keep `P0` on Preset F demyelination severity until ratio target is reproducibly met,
  - defer `P1` items (RHS scalar packing, GUI SSoT absolute-current cleanup, lazy heavy analytics tabs) until P0 gates are closed.

Execution source of truth for this queue is now `CurrentTasks10.2.md` + `AIDER_PLAN.md` (v10.2 section).

## Closed in validation contour

- `1` Calcium/Nernst calibration checks (core acceptance layer): active branch tests + extended deterministic report artifacts.
- `2` HCN validation (rest stability, V1/2, temperature behavior) and IA validation (suppression trends) in branch contour.
- `2.1` Dual-stim computational validation (branch tests + extended deterministic dual-stim report).
- `4` Dual-stimulation refactor to dedicated module (`core/dual_stimulation.py`) completed earlier and covered by branch tests.
- `6` Multichannel stress test branch coverage (`Ih+ICa`, `IA+SK`, all channels) plus broad preset stress matrix artifacts.
- Pathology mode validation core loop (`N/O/F`) is mostly green on focus + worst-case follow-up + extended F-vs-D conduction checks.
- Important caveat: F preset still shows near-complete soma→terminal propagation in current checkpoints (ratio ~1.0), so demyelination severity tuning remains open.
- `3` C/D/E tuning lane progress: deterministic quick recalibration grid (`18` cases) produced `0` anomalies in `_test_results/cde_extended_report_quick.json`.

## Partially closed / continuing

- `0` Test grouping and cleanup: active contour grouped/cleaned; root ad-hoc + legacy `test_*`/`validate_*` helpers archived under `tests/archive/root_legacy_tools`; only final optional pruning remains.
- `3` Configurable spike detection in GUI: integrated and branch-tested; iterative GUI UX improvements still possible.
- GUI clarity hardening in progress:
  - fixed SD/Excitability worker callback crash path,
  - callback contract regression test added (`test_advanced_sim_progress_callbacks_branch.py`),
  - dual-stim now resets to disabled default on preset load and explicitly overrides primary stimulation fields only when enabled,
  - preset mode selectors are context-scoped to relevant K/N/O presets,
  - GUI now surfaces Jacobian mode guidance and auto-selects `sparse_fd` for heavy multi-comp presets on load (with manual override preserved),
  - setup/dual-stim/analysis tab order and setup grouping were reworked for clearer workflow,
  - post-run tab focusing now uses explicit widget targets (no stale index jumps after tab reordering).
- `5` Error handling/logging/warnings: core layer implemented and tested; can be expanded with richer user-facing diagnostics.
- Validation orchestration hardening:
  - active suite now includes strict impedance utility gate as a utility-check phase,
  - environment-limited missing-dependency path is explicitly classified as warning (not false hard-fail),
  - utility check status mapping is unit-tested.
- K/N/O pathology UX/documentation guidance: mode switches + guide + preflight/status-mode warnings integrated; final bilingual polish still needed.

## Not closed yet (planned follow-up)

- Deferred: targeted `C/L` baseline stimulation refinement pass (kept out of current high-priority lane).
- `7` Plot readability/interactive improvements (beyond current export integration).
  - progress update: theme + linewidth + spike/delay overlays + title-font/grid controls integrated in oscilloscope.
  - delay visualization now supports target selection (terminal / AIS / junction / custom compartment),
  - delay target controls are now preset-synced before first run (custom index range available pre-simulation).
  - explicit stimulus waveform overlay `Stim(input)` added to currents panel and branch-tested.
- `8` Neuron passport with ML classification.
  - progress update: passport now includes Lyapunov/modulation blocks, channel-engagement/delay summaries,
    and a first hybrid rule+ML classifier baseline (prototype ML + confidence + source tag).
  - progress update: new Spike Mechanism analytics tab provides per-spike ion/Ca dynamics and attenuation-driver hypotheses.
  - progress update: Spike Mechanism now also includes time-resolved normalized channel/Ca trends to explain attenuation causality.
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
