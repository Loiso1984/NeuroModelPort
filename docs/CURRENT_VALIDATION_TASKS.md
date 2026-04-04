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
