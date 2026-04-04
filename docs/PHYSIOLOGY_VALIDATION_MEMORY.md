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
