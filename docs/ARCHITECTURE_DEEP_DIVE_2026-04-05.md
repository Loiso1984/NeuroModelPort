# NeuroModelPort — Architecture Deep Dive (2026-04-05)

## Scope and objective

This document captures a focused architecture audit before the final polishing phase. It aligns:

- `AIDER_PLAN.md` stage status,
- `ATTENTION.md` engineering invariants,
- `docs/MASTER_BACKLOG_CONTRACT.md` process rules,
- current code layout in `core/`, `gui/`, and `tests/`.

It is intended as an execution map for high-risk remaining items (especially RHS signature fragility, Stage 7 impedance, and GUI architecture cleanup).

---

## 1) High-level architecture map

### 1.1 Core simulation pipeline (`core/*`)

1. **Configuration / domain model** — `core/models.py`
   - Pydantic containers (`FullModelConfig`) for morphology, channels, calcium, environment, stimulation, analysis, preset modes.
   - Includes channel-specific Q10 fields and event-driven synaptic queue (`event_times`).
   - Contains current technical debt marker: dual representation `stim.Iext` + mutable `stim.Iext_absolute_nA`.

2. **Preset layer** — `core/presets.py`
   - Central preset catalogue A..S and mode modifiers (`k_mode`, `alzheimer_mode`, `hypoxia_mode`).
   - Applies complete reset-to-default before preset load (good anti-stale guard).
   - Performs density↔absolute conversion helper wiring for GUI display paths.

3. **Morphology and conductance vectorization** — `core/morphology.py`
   - Builds compartment count, areas, diameters, conductance vectors, axial Laplacian (CSR data/indices/indptr).
   - Applies AIS multipliers channel-wise.

4. **State layout / channel registry** — `core/channels.py`
   - Defines channel gate registry + initializes steady-state gates for enabled channels.
   - Provides ordering contract for state vector assembly used by solver/RHS.

5. **ODE RHS kernel** — `core/rhs.py`
   - `rhs_multicompartment` is Numba-jitted and currently receives a very large positional signature.
   - Includes primary+secondary stimulation routing, event-driven synaptic mode, per-channel phi vectors, optional calcium dynamics.

6. **Integration + Jacobian orchestration** — `core/solver.py`
   - Uses `solve_ivp(BDF)` with selectable jacobian mode (`dense_fd`, `sparse_fd`, `analytic_sparse`).
   - Builds all runtime vectors and forwards gigantic positional `args` tuple to RHS.
   - Returns `SimulationResult` with postprocessed currents/ATP and morphology context.

7. **Analysis layer** — `core/analysis.py`
   - Spike detection variants, Lyapunov FTLE/LLE, modulation decomposition, and post-run metrics.
   - Currently still coupled to `analysis.enable_lyapunov` parameter toggle (UI smell noted in `ATTENTION.md`).

8. **Advanced simulation wrappers** — `core/advanced_sim.py`
   - Sweep, S-D curve, excitability map, Euler-Maruyama stochastic loop.
   - Stochastic path currently still uses legacy global phi comment (`TODO: per-channel Q10`).

### 1.2 GUI layer (`gui/*`)

1. **Orchestration shell** — `gui/main_window.py`
   - MainWindow is currently a large multi-responsibility object (layout, run control, worker orchestration, preset logic, form sync, export, warnings).
   - Contains the current `Iext_absolute_nA` synchronization logic in both directions.

2. **Analytics surface** — `gui/analytics.py`
   - Multi-tab scientific UI: passport, traces, gates, currents, spike mechanism, phase, kymograph, balance, energy, bifurcation, sweep, S-D, excitability, spectrogram.
   - Passport currently reports Lyapunov as parameter-driven state, not explicit analysis action.

3. **Topology surface** — `gui/topology.py`
   - PyQtGraph-based morphology rendering with pan/zoom and readability controls.
   - Delay target highlighting already integrated.
   - Stage 4.4 “clickable compartment + neuron passport popup” remains incomplete.

4. **Dynamic form system** — `gui/widgets/form_generator.py`
   - Pydantic-driven auto-forms; useful for rapid extension but can obscure semantic grouping (important for Analysis vs Simulation separation).

### 1.3 Test/validation contour (`tests/*`)

- Active validation gate is explicitly documented in:
  - `tests/README_TESTS.md`
  - `tests/branches/BRANCH_TEST_REGISTRY.md`
- Branch tests under `tests/branches/` are the primary regression gate.
- Utilities under `tests/utils/` produce deterministic JSON artifacts in `_test_results/`.
- Legacy material is archived under `tests/archive/` and `tests/branches/legacy/`.

---

## 2) Current status vs your protocol (quick reconciliation)

Based on `AIDER_PLAN.md`, `ATTENTION.md`, and `docs/CURRENT_VALIDATION_TASKS.md`:

### 2.1 Already strong / partially complete

- Stages 1, 3, 5, 6 have substantial implemented surface.
- Jacobian modes are integrated and tested.
- Event-driven synaptic queue exists in model + RHS path.
- New channels/presets are integrated into core execution path.
- Active branch-suite process is documented and currently mature.

### 2.2 Confirmed open/high-risk items

1. **ATTENTION §6 Argument Explosion** (critical):
   - `rhs_multicompartment` still has a large positional signature.
   - Solver/Jacobian/RHS alignment risk remains the most dangerous silent-failure vector.

2. **Stage 7 Impedance Z(f)**:
   - Spectrogram exists, but dedicated ZAP→impedance pipeline does not appear productized.

3. **Stage 4.3 MainWindow decomposition**:
   - MainWindow still acts as a God Object.

4. **Lyapunov UI architecture bug**:
   - parameter toggle still represented inside AnalysisParams/form flow.

5. **Iext single-source-of-truth bug**:
   - mutable `Iext_absolute_nA` still stored as field and synchronized from GUI logic.

6. **Interactive topology stage completion**:
   - pan/zoom exists; clickable compartment intelligence and passport popup still pending.

7. **Full preset validation rerun needed**:
   - especially around TRN baseline behavior and Cholinergic scenario.

---

## 3) Risk register for immediate polishing phase

### 3.1 R1 — RHS positional API fragility (highest priority)

**Symptom:** very long positional argument path from `solver.py`/`jacobian.py` to `rhs.py`.

**Impact:** one argument shift can preserve runtime but break physics silently.

**Migration target (recommended):**
- Introduce a typed runtime parameter object (`PhysicsParams`) with explicit named fields.
- Keep an adapter layer for transition period (old tuple -> new structure).
- Add parity tests: same config, old-vs-new RHS checksum on `dydt` and key observables.

### 3.2 R2 — Dual current representation (`Iext` + `Iext_absolute_nA`)

**Symptom:** both fields are mutable in UI workflows.

**Impact:** drift between GUI-visible and solver-effective values.

**Migration target:**
- Make `Iext_absolute_nA` computed-only (`@property` / serializer helper).
- Keep conversion helpers in one place and enforce one-direction updates.

### 3.3 R3 — Lyapunov action placement

**Symptom:** analysis activation modeled as generic checkbox parameter.

**Impact:** architectural confusion (analysis action mixed with simulation config).

**Migration target:**
- remove `enable_lyapunov` from generic parameter form;
- explicit “Compute LLE” action in analytics UI with dedicated result payload.

### 3.4 R4 — MainWindow coupling

**Symptom:** logic concentration in `main_window.py`.

**Impact:** high change friction and high regression surface for GUI + run-control changes.

**Migration target:**
- `SimulationController` (execution, threads/processes, cancellation, progress)
- `ConfigManager` (preset apply/reset, dual-stim precedence, form sync)
- `MainWindow` (layout + signal wiring only)

### 3.5 R5 — Preset-level physiological uncertainty (TRN baseline)

**Symptom:** report that TRN baseline looks non-spiking.

**Impact:** could be either physiologic operating point or regression artifact.

**Validation path:**
- deterministic targeted matrix for TRN baseline/activated across I-scale and temperature,
- compare spike detector variants (`peak_repolarization`, `fsm`) to rule out detection artifact,
- inspect current decomposition (`ITCa`, `Ih`, `IA`, `SK`) in Passport/Spike Mechanism views.

### 3.6 R6 — Cholinergic fullscreen/setGeometry warning

**Observed text:** Qt geometry clamp warning (minimum size larger than available display height).

**Likely class:** UI layout/minimum-size aggregation issue (not core physics).

**Validation path:**
- run GUI smoke with reduced scale / constrained display,
- inspect large minimum-size contributors in tabs/forms,
- ensure fullscreen transitions do not hard-require > display height.

---

## 4) Recommended execution order (aligned with your priorities)

### Phase A — Safety hardening first

1. **RHS contract refactor prep (A1)**
   - design `PhysicsParams` schema and adapter,
   - freeze current argument order with explicit regression test.

2. **RHS migration (A2)**
   - migrate `rhs.py`, `solver.py`, `jacobian.py`, `advanced_sim.py` in lockstep,
   - run jacobian consistency + branch suite.

### Phase B — Scientific completeness

3. **Stage 7 impedance (B1)**
   - ZAP stimulus profile,
   - `compute_membrane_impedance()` in analysis,
   - analytics tab and export.

4. **TRN/cholinergic targeted physiology sweep (B2)**
   - focused deterministic validation + artifact report.

### Phase C — Architecture cleanup in GUI

5. **Lyapunov action decoupling (C1)**
6. **Iext absolute current SSoT fix (C2)**
7. **MainWindow decomposition baseline (C3)**
8. **Topology interaction completion (C4)**
9. **Passport channel completeness + bilingual audit (C5)**

---

## 5) Test strategy to use first (as requested)

Start with existing grouped gates before adding new tests:

1. Active branch gate:
   - `python tests/utils/run_active_branch_suite.py --workers 2`
2. Unified protocol:
   - `python tests/utils/run_unified_preset_protocol.py`
3. Targeted reports:
   - `python tests/utils/run_calcium_nernst_extended_report.py`
   - `python tests/utils/run_hcn_ia_extended_report.py`
   - `python tests/utils/run_dual_stim_extended_report.py`

For TRN/cholinergic bug-hunt add one focused utility report (new file under `tests/utils/`) instead of manual ad-hoc loops.

---

## 6) Practical definition of “done” for the remaining critical items

- **RHS contract done:** no call path uses giant positional tuple directly; tests include explicit contract/parity checks.
- **Impedance done:** Z(f) curve visible in analytics and reproducible from CLI/test utility.
- **Lyapunov UX done:** no checkbox-driven hidden heavy analysis in generic form; explicit user action path.
- **Iext done:** one canonical mutable field only.
- **TRN/cholinergic done:** deterministic report explains behavior as either validated physiology or corrected bug.

---

## 7) Notes for next implementation step

If we execute immediately from this audit, the first coding task should be **RHS contract hardening** (A1/A2), because all later physiology/analytics validity depends on not silently shuffling physics parameters.
