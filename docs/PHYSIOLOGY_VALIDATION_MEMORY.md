# Physiology Validation Memory

This file captures mandatory project rules for all future simulation work.
Canonical verbatim backlog/rules source: `docs/MASTER_BACKLOG_CONTRACT.md`.

## Mandatory Rules

1. Validate ion-channel parameters against physiological literature before accepting changes.
2. For any change affecting presets or computational logic:
   - implement/adjust tests in `tests/branches` first,
   - run branch tests,
   - only then update core files.
3. Validate spike detection logic each time tests are updated:
   - verify threshold crossing is counted as transitions (not raw sample count),
   - verify repolarization/baseline return checks,
   - compare output against expected physiology (spike count, voltage, channel params, stimulus params).
4. Preset defaults must keep dual stimulation disabled unless explicitly required.
5. Presets should only enable channels that are biologically needed for that cell type.
6. HCN, IA, ICa, SK validation must include:
   - isolated channel behavior,
   - interaction behavior (e.g., Ih+ICa, IA+SK, all enabled),
   - stress sweeps over broad parameter ranges,
   - physiological/non-physiological boundary checks.
7. Calcium dynamics checks must always include:
   - sign and magnitude of Ca influx,
   - Nernst E_Ca range at target temperatures,
   - non-negative [Ca2+]i during simulation.
8. Demyelination preset validation (F) must always include axonal propagation checks:
   - soma still produces spikes,
   - conduction delay along axon is increased versus control motoneuron,
   - proximal axonal spike transfer effectiveness is reduced.
9. Long sweeps for branch validation should run in parallel with checkpoint/resume support
   (do not run monolithic single-process brute-force jobs for hours).
10. If Jacobian-based acceleration proves stable and physiologically correct, add Jacobian mode as a GUI-selectable solver option and evaluate making it default for heavy presets.
11. Lyapunov (LLE/FTLE) analysis stays default-OFF and is interpreted only as a stability descriptor together with physiological metrics, not as standalone realism proof.
12. Modulation decomposition (item 17) stays default-OFF and must report phase-locking metrics with surrogate significance before using any conclusion about theta-driven contribution to high-frequency spiking.
13. Computationally heavy presets (`F/K/N/O`) should default to `sparse_fd` Jacobian mode unless a validated reason requires otherwise.
14. Pathology mode validation for `N/O` must include deterministic perturbation sweeps:
   - progressive mode should preserve early spiking with later attenuation,
   - terminal mode should remain less excitable than progressive in most matched perturbations,
   - checks are mandatory in branch tests before promoting preset changes.
15. Spike detection used in analytics/passport must remain user-configurable and explicitly test-covered:
   - support algorithm selection (`peak_repolarization` vs `threshold_crossing`),
   - keep threshold/baseline/refractory/repolarization controls synchronized between config and analysis output,
   - verify stricter detector settings do not paradoxically increase detected spike count.
16. Pre-run solver validation must reject invalid parameter combinations before integration starts:
   - use custom `SimulationParameterError`,
   - keep guard checks branch-tested to prevent silent non-physiological runs.
17. After any preset tuning pass, rerun the unified protocol and persist the artifact:
   - `python tests/utils/run_unified_preset_protocol.py`,
   - refresh `_test_results/unified_preset_protocol.json`,
   - verify mode-order expectations (`K activated >= baseline`, `N/O terminal <= progressive`) and F conduction delay vs D.
18. User-facing warning policy:
   - keep centralized runtime/edge-parameter warnings in `core.validation`,
   - preserve GUI preflight validation for run/sweep/S-D/excmap so users see warnings before heavy jobs,
   - keep branch tests covering warning emission semantics.
19. Thalamic `K` mode policy:
   - keep two explicit modes (`baseline`, `activated`) in GUI and presets,
   - default `K` mode is `baseline` (low-throughput/theta-like global envelope),
   - `activated` must remain clearly higher-throughput than baseline in branch tests.
20. Hypoxia search runtime policy:
   - deterministic search remains default sequential on local machine,
   - threaded mode is opt-in only (`--allow-parallel`) due observed regressions on stiff multi-comp solves,
   - maintain ETA/progress reporting for long validation runs.
21. Broad preset stress validation policy:
   - use deterministic stress matrix runner for routine wide sweeps,
   - default to single-comp proxy for fast physiology guard screening,
   - run full multi-comp matrix only as an explicit, narrower follow-up stage.
22. Pathology focus policy:
   - run dedicated N/O/F focused report regularly (`run_pathology_focus_report.py`),
   - require `N/O` terminal mode to remain less excitable than progressive across deterministic perturbation grid,
   - require demyelination (`F`) conduction signature against control (`D`):
     delayed propagation and reduced transfer ratio or clear absolute attenuation.
23. Thalamic dual-stim interpretation:
   - for `K activated`, inhibitory secondary stimulation may trigger post-inhibitory rebound via Ih,
   - dual-stim tests should assert strong modulation of throughput/pattern, not only monotonic spike-count reduction.
24. Pathology escalation flow:
   - after hard deterministic sweep, run worst-case strict follow-up (`run_pathology_worstcase_followup.py`),
   - treat any follow-up anomaly as a blocker before promoting pathology preset changes.
25. Demyelination robustness flow:
   - periodically run extended F-vs-D conduction grid (`run_f_conduction_extended.py`),
   - keep acceptance on combined signature: increased delay + reduced ratio or strong absolute attenuation.
26. C/D/E operating envelope flow:
   - periodically run deterministic C/D/E extended report (`run_cde_extended_report.py`),
   - track temperature/drive sensitivity and preserve baseline frequency envelopes before preset promotion.
