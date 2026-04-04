# Current Validation Tasks

This file preserves the active high-priority validation backlog and current findings.
Canonical verbatim source: `docs/MASTER_BACKLOG_CONTRACT.md`.

## Active Priority Order

1. Full validation-first workflow before major program feature work.
2. Hypoxia (`O`) validation and tuning.
3. Alzheimer (`N`) validation and tuning.
4. HCN validation.
5. IA validation.
6. Combined channel stress validation (`Ih + ICa`, `IA + SK`, all enabled).
7. Demyelination (`F`) validation:
   - soma should still spike,
   - axonal propagation should be slowed / impaired versus control,
   - later visualization should make this visible.
8. Unified preset protocol for all presets.
9. Dual stimulation validation.
10. C / D / E further tuning after pathology screening work.
11. Sweep stress validation.
12. Cleanup / unification of old scripts and tests.
13. Only after branch validation passes: propagate validated changes into main logic/files.

## Latest Completed Checkpoint (2026-04-04)

1. Jacobian acceleration integrated in solver:
   - `dense_fd | sparse_fd | analytic_sparse`.
2. Benchmarked on hypoxia case:
   - `dense_fd`: ~237s,
   - `sparse_fd`: ~2.18s,
   - `analytic_sparse`: ~2.71s.
3. Hypoxia search contour updated to configurable Jacobian modes and deterministic checkpointed flow.
4. Hypoxia progressive mode tuned for visible early-spike phase (2 spikes) before attenuation.
5. Thalamic baseline mode tuned from silent to lower-throughput spiking regime.
6. New multi-channel stress branch suite added and passing:
   - `Ih+ICa`, `IA+SK`, all channels enabled.
7. Full local regression battery currently green:
   - unified preset protocol, channel physiology, dual stim, spike detection math,
     Jacobian-mode consistency, multi-channel stress.
8. Added preset-level calcium thermodynamics validation:
   - E_Ca range and temperature trend checks on dynamic-Ca presets (E/K/M/N/O).
9. Legacy branch scripts mirrored under `tests/branches/legacy` and separated in registry;
   active gate remains only the explicit active suite.
10. Dual-stimulation branch hardening completed:
   - dual primary settings now override `cfg.stim` when dual mode is enabled,
   - secondary dendritic filtered stimulation now has proper stateful tau dynamics,
   - branch tests cover both behaviors.
11. Heavy preset defaults updated to faster solver mode:
   - `F/K/N/O` now default to `stim.jacobian_mode = sparse_fd`,
   - guarded by branch test `test_heavy_presets_default_to_sparse_jacobian_mode`.
12. C/D/E validation expanded with deterministic drive-sweep branch checks:
   - stability and voltage guards across scaled drives,
   - excitability envelope checks for C, D, and E profiles.
13. Legacy-style parallel hypoxia sweep utility updated for safer defaults:
   - default executor switched to thread pool (avoids repeated process-level JIT cold-start cost),
   - explicit `--jacobian-mode` added (defaults to `sparse_fd`),
   - one-shot warmup added before batch execution.
14. Added deterministic pathology robustness sweep in active gate:
   - new branch test `test_pathology_mode_sweep_branch.py`,
   - validates `N/O` progressive vs terminal mode separation under parameter perturbations,
   - now part of `run_active_branch_suite.py` and currently passing.
15. Legacy test-script unification cleanup completed:
   - removed duplicate obsolete branch scripts from `tests/branches` root,
   - kept canonical legacy copies only under `tests/branches/legacy`,
   - active suite remains explicit and unaffected.
16. GUI user guidance updated for pathology modes:
   - integrated `K/N/O` mode interpretation in built-in Guide tab,
   - includes explicit progressive vs terminal behavior hints for users.
17. Dual-stimulation refactor progressed:
   - primary/secondary stimulus application logic extracted from `rhs.py` to `core/dual_stimulation.py`,
   - `rhs.py` now delegates compartment injection/filter-state math to dedicated dual-stim helpers.
18. Solver validation/error layer added:
   - `core/validation.py` + `SimulationParameterError` introduced,
   - pre-run guard checks for invalid parameter combinations,
   - branch coverage added in `test_solver_validation_branch.py` (active gate).
19. Configurable spike detection integrated:
   - new analysis parameters for algorithm/threshold/prominence/baseline/refractory/repolarization window,
   - `full_analysis` now uses user-configured detector settings,
   - GUI analytics plots now use the same configured detector (phase/sweep),
   - branch checks extended in `test_spike_detection_math_branch.py`.
20. Active branch-suite runtime optimized:
   - `run_active_branch_suite.py` now supports parallel execution (`--workers`, default `2`),
   - preserves deterministic result ordering in reports while reducing wall-clock validation time.
21. GUI plot export flow completed:
   - `OscilloscopeWidget.export_plot()` integrated into MainWindow with a dedicated `Export Plot` button,
   - supports PNG/JPG/BMP, SVG, and PDF output,
   - enabled after successful standard/stochastic runs.
22. Unified full preset protocol rerun completed:
   - report refreshed at `_test_results/unified_preset_protocol.json`,
   - mode checks confirm expected ordering (`K activated >= baseline`, `N/O terminal <= progressive`),
   - demyelination conduction delay remains increased vs control in current tuned branch.
23. Test-directory hygiene pass completed:
   - removed transient `tests/**/__pycache__` artifacts,
   - unified duplicate stimulus-route utility by keeping the fixed implementation as canonical
     and converting legacy script path to a compatibility wrapper.
24. Runtime-warning system strengthened and validated:
   - added centralized runtime estimator in `core/validation.py`,
   - heavy-run estimate now contributes explicit warning messages alongside edge-parameter warnings,
   - solver switched to shared estimator (no duplicated local formulas),
   - GUI preflight validation now surfaces warnings/errors before launching run/sweep/S-D/excitability jobs.
25. Post-change regression status:
   - branch tests updated for warning behavior (`test_solver_validation_branch.py` now 6/6),
   - full active branch suite rerun successfully after these changes (`_test_results/active_branch_suite.json`).
26. K-mode physiology refinement completed in branch contour:
   - `K baseline` switched to low-throughput alpha-driven regime (theta-like global envelope),
   - `K activated` preserved as high-throughput relay regime,
   - default `preset_modes.k_mode` set to `baseline` in model defaults.
27. Hypoxia deterministic search executor behavior hardened:
   - added `--workers` with deterministic mapping and ETA reporting,
   - benchmarked locally: threaded mode regressed runtime for stiff multi-comp jobs,
   - default behavior now forces sequential execution unless `--allow-parallel` is explicitly provided.
28. Test tree cleanup continued:
   - moved legacy top-level `ARCHIVE_test_*` files into `tests/archive/`,
   - root `tests/` now contains only active docs/markers and no stray legacy test scripts.
29. Added deterministic all-preset stress-matrix runner:
   - new utility `tests/utils/run_preset_stress_matrix.py`,
   - sweeps `(preset x Iext-scale x temperature)` with physiology guards and JSON artifact output,
   - default mode uses fast single-comp proxy to avoid hour-long multi-comp brute-force runs;
     full cable stress remains opt-in via `--multicomp`.
30. Ran stress-matrix smoke pass:
   - command: `python tests/utils/run_preset_stress_matrix.py --i-scales 0.8,1.0 --temps 23,37 --t-sim 100 --dt-eval 0.4`,
   - artifact: `_test_results/preset_stress_matrix.json`,
   - result: `guard_ok 60/60`.
31. Expanded broad stress matrix pass completed:
   - command: `python tests/utils/run_preset_stress_matrix.py --i-scales 0.6,0.8,1.0,1.2,1.4 --temps 23,30,37 --t-sim 120 --dt-eval 0.35`,
   - artifact updated: `_test_results/preset_stress_matrix.json`,
   - result: `guard_ok 225/225`.
32. Added pathology-focused report utility:
   - new script `tests/utils/run_pathology_focus_report.py`,
   - validates `N/O` progressive-vs-terminal ordering on deterministic grid and `F` conduction signature against `D`,
   - artifact: `_test_results/pathology_focus_report.json`,
   - latest result: mode ordering `18/18`, anomalies `0`.
33. Added regression guard for thalamic defaults:
   - `test_k_default_mode_is_baseline_and_low_throughput` in unified branch protocol,
   - active suite rerun green after guard addition.
34. Ran hard pathology matrix (`N/O/F`) with stronger deterministic perturbations:
   - artifact: `_test_results/pathology_hard_matrix.json`,
   - result: anomalies `0` under current hard-matrix criteria.
35. Dual-stim validation extended for thalamic activated mode:
   - added branch test for inhibitory secondary drive modulation in `K activated`,
   - updated expectation to physiology-consistent modulation (including possible rebound increase with Ih),
   - active suite remains green.
36. Added worst-case pathology follow-up utility:
   - new script `tests/utils/run_pathology_worstcase_followup.py`,
   - selects near-boundary cases from `_test_results/pathology_hard_matrix.json` and reruns them with stricter simulation settings,
   - artifact: `_test_results/pathology_worstcase_followup.json`,
   - latest run: anomalies `0`.
37. Added reusable extended demyelination conduction utility:
   - new script `tests/utils/run_f_conduction_extended.py`,
   - checks F-vs-D conduction signature on configurable `T / Ra / gL / drive` grid,
   - baseline smoke run completed (`4/4 ok`, anomalies `0`),
   - heavy one-off full grid already executed and saved as `_test_results/pathology_f_conduction_extended.json` (`81 cases`, anomalies `0`).
38. Added reusable extended C/D/E report utility:
   - new script `tests/utils/run_cde_extended_report.py`,
   - sweeps `I_scale x temperature` for `C/D/E` with deterministic physiology flags,
   - latest sweep artifact: `_test_results/cde_extended_report.json`,
   - current run status: anomalies `0` on tested grid.
39. C/D/E contour hardened to catch Purkinje silent-island regression:
   - added branch guard `test_e_moderate_low_drive_not_silent_at_37c` in `test_cde_profiles_branch.py`,
   - tuned `E` preset default drive (`Iext: 30 -> 32`) to preserve excitability at `0.8x / 37C`,
   - reran `run_cde_extended_report.py` and full active branch suite after tuning; both green,
   - refreshed `_test_results/unified_preset_protocol.json` after preset update per promotion rule.
40. Added deterministic HCN/IA extended sweep artifact:
   - new utility `tests/utils/run_hcn_ia_extended_report.py`,
   - combines isolated-channel probe checks (HCN pulse sag/Rin, IA excitability suppression) with preset-level sanity rows,
   - artifact: `_test_results/hcn_ia_extended_report.json`,
   - latest run status: anomalies `0`.
41. Re-ran broad all-preset stress matrix after E tuning:
   - command: `python tests/utils/run_preset_stress_matrix.py --i-scales 0.6,0.8,1.0,1.2,1.4 --temps 23,30,37 --t-sim 120 --dt-eval 0.35`,
   - refreshed artifact: `_test_results/preset_stress_matrix.json`,
   - latest result: guard pass `225/225` (100%).
42. Added strict calcium/Nernst extended report for K/L/M/N/O focus:
   - new utility `tests/utils/run_calcium_nernst_extended_report.py`,
   - includes `N/O` progressive/terminal mode variants,
   - validates Ca-range, E_Ca range, inward ICa proxy, temperature trend, and exports `B_Ca/gCa_max` audit fields,
   - artifact: `_test_results/calcium_nernst_extended_report.json`,
   - latest run status: anomalies `0/28`.
43. Added deterministic dual-stimulation extended report:
   - new utility `tests/utils/run_dual_stim_extended_report.py`,
   - exports branch-equivalent scenarios (dual-off baseline match, inhibitory secondary, soma+AIS, primary override, K-activated modulation),
   - artifact: `_test_results/dual_stim_extended_report.json`,
   - latest run status: anomalies `0/7`.
44. Utilities hygiene / unification pass:
   - removed runtime cache artifacts under active test folders (`tests/utils/__pycache__`, `tests/branches/__pycache__`),
   - added active-vs-legacy utility registry: `tests/utils/UTILS_REGISTRY.md`,
   - documented canonical usage path in `tests/README_TESTS.md`.
45. Added explicit master backlog coverage snapshot:
   - `docs/VALIDATION_COVERAGE_STATUS.md` now tracks closed / partial / pending items for 0–16 list,
   - used as checklist anchor before promoting validation-contour changes into primary defaults.
46. GUI preset-mode integration hardening:
   - preflight now includes preset-mode notes/warnings (`K activated`, `N terminal`, `O terminal`) via `build_preset_mode_warnings`,
   - status bar now shows active mode suffix when loading/changing K/N/O presets,
   - branch coverage extended in `test_solver_validation_branch.py`,
   - active branch suite rerun green after integration.
47. Root-level legacy cleanup pass (safe archival):
   - moved ad-hoc `check_*`, `debug_*`, `quick_*`, `simple_*`, `verify_*` python helpers from repository root
     into `tests/archive/root_legacy_tools/`,
   - added archive index `tests/archive/root_legacy_tools/README.md`,
   - reran active branch suite after move; green.
48. Root legacy `test_*` / `validate_*` cleanup pass:
   - moved non-active root `test_*` and `validate_*` scripts into `tests/archive/root_legacy_tools/`,
   - active validation entry points remain `tests/branches/*` + `tests/utils/*`,
   - reran active branch suite after move; green.
49. Master backlog contract encoding integrity fixed:
   - repaired `docs/MASTER_BACKLOG_CONTRACT.md` from mojibake to clean UTF-8,
   - restored canonical Russian verbatim source block and preserved added items (`17`, Jacobian GUI requirement, Lyapunov note).
50. Post-directive regression check:
   - reran `python tests/utils/run_active_branch_suite.py --workers 2`,
   - all active branch scripts passed (`11/11`), artifact refreshed:
     `_test_results/active_branch_suite.json`.
51. Analytics productization progress for backlog `16` / `17`:
   - GUI Passport now surfaces Lyapunov output (`class`, `LLE`, valid-pair count) when enabled,
   - GUI Passport now surfaces non-FFT modulation decomposition summary (source, band, PLV, preferred phase, depth/index, surrogate stats),
   - both analyses remain default-OFF and display explicit disabled/insufficient-data notes.
52. Manual GUI QA issues captured (user report):
   - fixed `run_sd_curve` / `run_excmap` background worker crash:
     `unexpected keyword argument 'progress_cb'`,
   - dual-stim default reset policy tightened: preset loading now resets Dual Stim to disabled state,
   - added explicit dual-stim precedence behavior in UI/validation (Dual tab overrides primary stimulation fields when enabled),
53. Analytics detail expansion for attenuation causality:
   - `Spike Mechanism` tab upgraded with a time-resolved normalized activity panel for enabled ionic currents + `Ca_i`,
   - per-spike diagnostics retained and now complemented by full-trace trend context for attenuation interpretation.
54. Demyelination (`F`) sanity check rerun (quick deterministic pass):
   - current branch still shows mostly preserved terminal propagation (`term/soma ~0.975`) with delayed conduction,
   - interpreted as mild conduction impairment, not strong terminal block; stronger severity calibration remains open.
55. Legacy Scilab (`C:\NeruroModelV9`) transfer candidates reaffirmed:
   - explicit per-compartment spike report (soma/AIS/fork/terminal),
   - dedicated Ca/E_Ca dynamics plots and current-balance overlays,
   - topology readability annotations that can be reused in Python GUI evolution.
56. GUI stimulation sync hardening (user-reported confusion lane):
   - stimulation-parameter visibility now follows active stimulation source (main or Dual-primary),
   - `alpha_tau` is now shown for synaptic stimulus types (`AMPA/NMDA/GABAA/GABAB/Kainate/Nicotinic`) to expose kinetics clearly,
   - when Dual Stim is enabled, main Stimulation panel mirrors active Dual-primary parameters in disabled preview mode.
57. Added dedicated GUI branch guard:
   - new `tests/branches/test_gui_stim_sync_branch.py`,
   - verifies `const/pulse/alpha/synaptic` visibility contract and Dual-primary preview sync,
   - relevant branch checks rerun green (`gui_stim_sync`, `dual_stim`, progress-callback tests).
   - S-D and Excitability Map analyses now run with dual-stim disabled by design to avoid metric contamination,
   - preset-mode controls are now context-sensitive (`K/N/O` only for relevant presets).
53. GUI structure/clarity pass (workflow-oriented):
   - tab order relabeled by workflow steps:
     `1) Setup -> 2) Dual Stim -> 3) Oscilloscope -> 4) Analytics -> 5) Topology -> 6) Axon Biophysics -> 7) Guide`,
   - Parameters page regrouped into:
     `Run Setup (Edit First)`, `Model Biophysics`, `Advanced Analysis Tools`,
   - added persistent priority hint panel:
     Dual Stim precedence + active mode interpretation in setup area,
   - non-relevant `K/N/O` mode controls are hidden for non-target presets (not just disabled),
   - Guide updated with explicit notes:
     `F` is single-stage and Dual Stim overrides primary stimulation fields when enabled.
54. Oscilloscope readability controls added:
   - view panel now includes plot theme selector, line-width scaling, spike-marker toggle, and soma→terminal delay overlay toggle,
   - top voltage title now includes soma→terminal delay when available.

55. GUI-only validation directive captured:
   - for pure GUI edits, default to lightweight validation (`py_compile` + focused smoke),
   - full preset branch suites are not mandatory after GUI-only changes.
56. Workflow tab-focus regression fixed:
   - replaced stale hardcoded `setCurrentIndex(...)` transitions with explicit `setCurrentWidget(...)`,
   - after run/stochastic, UI now reliably opens Oscilloscope tab,
   - after sweep/S-D/excitability map, UI now reliably opens Analytics tab.
57. Built-in Guide alignment with new workflow:
   - updated Quick Start text to reference `1) Setup` and `2) Dual Stim`,
   - added explicit note that Dual Stim ON overrides primary stimulation fields,
   - clarified post-run navigation as `3) Oscilloscope` + `4) Analytics`.
58. Dual-stim signal hookup noise fixed:
   - replaced eager `disconnect/connect` cycle with one-time guarded signal connection,
   - removed startup Qt warning risk during lightweight GUI smoke initialization.
59. Oscilloscope Monte-Carlo view consistency pass:
   - plot theme and line-width scaling now apply to MC cloud/mean/band and threshold line too,
   - verified via lightweight offscreen widget smoke run.
60. Stimulation-form context sync fix:
   - stimulation fields are now context-sensitive by `stim_type` (no alpha-only clutter for `const/pulse`),
   - selecting receptor modes (`AMPA/NMDA/GABAA/GABAB/Kainate/Nicotinic`) now auto-applies corresponding synaptic default parameters in GUI config,
   - added explicit setup hint clarifying that `const` in selected physiological presets is tonic-drive proxy (not a single synaptic event).
61. Deferred physiology tuning backlog:
   - postponed `C/L` baseline stimulation refinement (FS Interneuron / Hippocampal CA1) for a dedicated pass,
   - keep current validated settings for now and return after higher-priority master-plan items.
62. Oscilloscope readability follow-up:
   - added View controls for title font size and grid alpha,
   - titles now render with configurable font size in single-run and MC modes,
   - grid contrast now user-adjustable without rerunning physiology sweeps.
63. Axonal delay target selector in Oscilloscope:
   - delay overlay is no longer terminal-only; user can select `Terminal`, `AIS`, `Trunk Junction`, or `Custom Compartment`,
   - single-compartment runs auto-disable delay controls to avoid misleading metrics,
   - delay title tag now reports selected target (e.g., `delay soma->junction ... ms`).
64. Delay-target UX safety hardening:
   - delay-target options are auto-disabled when unavailable for current morphology (e.g., no AIS or no junction),
   - selector falls back to valid target automatically, preventing empty/non-physiological delay readouts.
65. Guide documentation sync:
   - built-in Guide now documents new oscilloscope delay-target options
     (terminal / AIS / junction / custom compartment).
66. Passport informativeness upgrade:
   - added `CHANNEL ENGAGEMENT` section in analytics passport with dominant channel by |Q|,
   - added per-channel peak inward/outward current and |Q| summaries (top channels),
   - added soma->junction and soma->terminal delay fields directly in passport text.
67. Hybrid passport classification baseline:
   - added lightweight prototype-ML classifier in `core.analysis` (`FS/RS/IB/LTS`) with confidence,
   - added hybrid decision layer (rule + ML) with source tag (`rule_only / rule+ml / ml_only / rule_priority`),
   - exported ML/hybrid fields through `full_analysis` and surfaced them in Passport text.
68. Topology readability enhancement (index map):
   - morphology view now annotates compartment index ranges on AIS/trunk/branches and fork index,
   - info bar includes compact index map (`soma/fork/terminal`) for easier targeting of custom compartment analyses.
69. Targeted regression checks for latest analytics/topology pass:
   - `test_passport_ml_classification_branch.py` passed (`3/3`),
   - `test_modulation_decomposition_branch.py` passed (`3/3`),
   - `test_spike_detection_math_branch.py` passed (`5/5`),
   - lightweight GUI offscreen smoke checks passed for analytics/topology/plots.

## Preset Mode Requirements

### K
- keep GUI-switchable modes,
- baseline / activated relay behavior,
- do not assume tonic high frequency is always physiological.

### N / O
- two pathology stages:
  - progressive: initial spikes then attenuation,
  - terminal: near-silent / depolarization-block-like endpoint,
- GUI-switchable,
- final documentation must explain terminal-stage interpretation for users.

## GUI Clarification Backlog (Detailed)

1. Dual-stim UX simplification:
   - make override priority explicit everywhere (status, warnings, guide text),
   - keep dual-stim disabled by default after every preset load,
   - eliminate ambiguous duplicated primary-stim editing paths where possible.
2. Preset-mode UX scoping:
   - only relevant mode controls should be active for selected preset,
   - show explicit “ignored for current preset” hints for non-relevant mode flags.
3. Tab information architecture:
   - reduce cognitive load in Parameters by regrouping forms and adding section-level guidance,
   - review tab naming/order to match user workflow (setup → run → inspect).
4. Pathology semantics clarity:
   - explain clearly that `F` (demyelination) is currently single-stage (no progressive/terminal switch),
   - ensure this is visible both in preflight warnings and user-facing documentation.

## Current Test-Contour Findings

1. Spike detection math was flawed:
   - `prominence` was effectively ignored,
   - repolarization window depended on sample count instead of milliseconds.
   Status: fixed in `core/analysis.py` and covered by branch tests.

2. Demyelination must be judged by propagation metrics, not only soma spikes.
   Status: added to branch protocol and memory.

3. Hypoxia search remained too slow even after initial improvements.
   Main reasons:
   - cold-start numba/JIT cost,
   - expensive multi-compartment BDF solves,
   - dense LU factorization dominates runtime,
   - coarse stage still did too many full solves before latest refactor.

4. Profiling result for heavy hypoxia case:
   - dominant cost is `scipy.integrate.solve_ivp(..., method='BDF')`,
   - especially repeated `scipy.linalg.lu_factor`,
   - Python orchestration is not the primary bottleneck.

5. Frequency metric caveat in preset reports:
   - pure ISI-based `freq_hz` can overstate rate for delayed short spike epochs,
   - unified protocol now also exports `freq_global_hz` and `freq_active_window_hz`
     for more physiological interpretation.

## Optimization Conclusion

Translating only test orchestration into C will not deliver the expected 25-100x speedup.
The real hotspots are:
- stiff ODE integration,
- Jacobian / LU factorization work,
- repeated cold-process startup.

Highest-leverage optimization directions:
1. keep batch searches inside one long-lived Python process,
2. warm up numba once,
3. use single-compartment proxy for coarse screening,
4. run full multi-compartment validation only for finalists,
5. investigate sparse/analytic Jacobian or compiled solver backend (future core optimization step).

## Post-Physiology Execution Directive

After explicit closure of full physiology validation, non-physiology tasks
(analytics, plots, GUI, documentation) may be implemented directly in the main contour.
Branch-first transfer flow remains mandatory for preset/channel/core-physiology logic changes.

## New Mandatory UI Follow-up

If Jacobian mode is validated as both stable and significantly faster, expose it in GUI as a user-selectable solver option and evaluate it as a possible default for heavy presets.

## Added Analysis Backlog

1. LLE/FTLE stability analysis (default OFF):
   - add `analysis.enable_lyapunov` flag,
   - compute FTLE/LLE as a post-run analysis option,
   - classify regimes (`stable`, `limit_cycle_like`, `unstable_or_chaotic`),
   - validate on synthetic reference scenarios.

## LLE/FTLE Progress (2026-04-04)

1. Core integration done:
   - added Lyapunov analysis flags in `AnalysisParams`,
   - added `estimate_ftle_lle()` and `classify_lyapunov()` in analysis module,
   - added optional Lyapunov outputs to `full_analysis`.
2. Branch validation added and passing:
   - `test_lyapunov_analysis_branch.py` checks expected sign on stable vs chaotic synthetic references.

## Modulation Decomposition Progress (2026-04-04)

1. Core integration done (default OFF):
   - added phase-based non-FFT decomposition in `core/analysis.py`,
   - outputs include PLV, preferred phase, phase-rate profile, modulation depth/index,
   - deterministic surrogate significance (`p` and `z`).
2. Added `full_analysis` export fields:
   - `modulation_*` metrics available when `analysis.enable_modulation_decomposition=True`.
3. Branch validation added:
   - `tests/branches/test_modulation_decomposition_branch.py` checks
     locked vs unlocked synthetic spike patterns and `full_analysis` integration.

## UI/Topology Sync Progress (2026-04-04, continued)

70. Delay-target context is now shared between Oscilloscope and Topology:
   - selecting delay target in Oscilloscope immediately updates Topology delay-focus highlight and info bar,
   - redraw happens without solver rerun (visual sync only).
71. Delay-target controls now sync from preset morphology even before first run:
   - custom-compartment index range (`1..N-1`) is available right after preset load,
   - avoids pre-run `1..1` lock for custom delay target.
72. Topology index semantics aligned with morphology indexing:
   - fork index now tracks the morphology fork compartment (last trunk comp),
   - index map and delay target labels use consistent compartment numbering.
73. Added branch regression test for this integration:
   - `tests/branches/test_delay_target_sync_branch.py` (`3/3` pass),
   - active branch suite registry and runner updated to include this test.
74. Lightweight real-time GUI preview hardening:
   - topology preview now updates immediately on setup edits (stimulation, location, dendritic filter, channels, morphology),
   - no solver rerun required for these UI-only preview updates.
75. Delay/junction semantics unified with morphology indexing:
   - `Trunk Junction` now maps to fork compartment (last trunk segment) consistently across Oscilloscope, Topology, Analytics and core conduction metric extraction.
76. Validation utility consistency pass:
   - updated active physiology report scripts to use the same fork/junction indexing (`run_unified_preset_protocol.py`, `run_f_conduction_extended.py`, `run_pathology_focus_report.py`, `run_pathology_worstcase_followup.py`),
   - avoids cross-report mismatch for junction peaks/delay interpretations.
77. Added regression guard for progress-callback path in advanced analyses:
   - new `tests/branches/test_advanced_sim_progress_callbacks_branch.py` (`2/2` pass),
   - protects `run_sd_curve` and `run_excitability_map` callback contract (`progress_cb`) from future regressions.
78. Stimulation UI visibility sanity check completed:
   - verified `const/pulse/alpha/AMPA` correctly show/hide only relevant controls in setup form.
79. Added explicit stimulus waveform visualization in Oscilloscope:
   - currents panel now shows `Stim(input)` trace reconstructed from active stimulation settings (including dual-stim contribution),
   - `Stim(filt)` remains separately visible when dendritic filter state exists.
80. Added branch regression coverage for stimulus overlay:
   - `tests/branches/test_stimulus_trace_overlay_branch.py` (`2/2` pass),
   - active suite registry/runner updated accordingly.
81. Jacobian usability update in GUI setup flow:
   - setup hint now always shows current `jacobian_mode` and recommends sparse modes for heavy multi-comp presets,
   - heavy presets (`K/N/O/F`) now auto-switch from `dense_fd` to `sparse_fd` on preset load (user can still change manually).
82. Added branch regression test for Jacobian GUI auto-selection:
   - `tests/branches/test_gui_jacobian_autoselect_branch.py` (`2/2` pass),
   - active suite registry/runner updated.
83. C/D/E deterministic quick recalibration pass executed:
   - `run_cde_extended_report.py` on grid `i_scale={0.8,1.0,1.2}`, `T={30,37}` (`18` cases),
   - output: `_test_results/cde_extended_report_quick.json`,
   - result: `0/18` anomalies under current guards (baseline windows for C/D/E satisfied on sampled grid).
84. Added expanded spike-causality analytics view:
   - new `Spike Mechanism` tab in Analytics with:
     - spike timeline and indexed peaks,
     - per-spike `V_peak` + `Ca_i@spike`,
     - per-spike sampled ionic currents (`Na/K/ICa/IA/SK/Ih/Leak` where present),
     - heuristic attenuation explanation (Na reduction / K-SK increase / Ca accumulation).
   - branch test added: `tests/branches/test_spike_mechanism_analytics_branch.py` (`1/1` pass).
85. Multiple sclerosis (F) status checkpoint (critical):
   - targeted verification currently shows soma→terminal spike count ratio `1.0` (thresholds `-20/-10/0 mV`),
   - this means present F preset still conducts spikes to terminal branch too well for intended “propagation failure” phenotype,
   - recalibration task for F remains open and high-priority.
86. Legacy Scilab v9 audit (folder `C:\\NeruroModelV9`) produced useful transfer candidates:
   - per-compartment spike report across soma/AIS/fork/branch terminals,
   - explicit Ca/E_Ca visualization panel,
   - current-balance diagnostic and energy overlays,
   - Remak law annotation + richer topology labels,
   - these are mapped to follow-up Python analytics/topology enhancements.
