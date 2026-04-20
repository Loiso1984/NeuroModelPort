# MASTER EXECUTION CANON v10.2

> Unified operational document compiled from `ATTENTION.md`, `AIDER_PLAN.md`, `CurrentTasks10.2.md`, and `docs/MASTER_BACKLOG_CONTRACT.md`.
>  
> Purpose: one non-duplicated execution guide with strict priority order, architecture invariants, and verification protocol.  
> Source documents remain unchanged and authoritative for historical detail.

---

## 1) Priority and authority model

When instructions conflict, resolve in this order:

1. **System / developer / user instructions**
2. **This document (execution canon)**
3. **CurrentTasks10.2.md** (active queue and status log)
4. **ATTENTION.md** (architecture invariants)
5. **AIDER_PLAN.md** (implementation roadmap and technical constraints)
6. **docs/MASTER_BACKLOG_CONTRACT.md** (physiology/test process contract)

Execution axis is always:

**Functionality → Scientific correctness → Performance → UX/docs.**

---

## 2) Non-negotiable engineering invariants

### 2.1 Simulation/analysis/visualization separation
- Simulation core (RHS/solver) must not embed analysis controls.
- Analysis actions (e.g., Lyapunov) must be explicit and isolated.
- Visualization must not mutate simulation truth state.

### 2.2 SSoT and explicit contracts
- `Iext` density remains canonical source; absolute current is display-only.
- No fragile implicit positional drift across RHS/Jacobian/Solver interfaces.
- No silent fallback that can produce scientifically incorrect outputs.

### 2.3 Numba and hot-path rules
- No per-step heap allocation in `@njit` kernels.
- No hidden temporary structures in tight loops unless pre-allocated.

### 2.4 GUI update-loop rules
- No `fig.clear()/ax.cla()/axis recreate` in hot update loops.
- Persistent artists only (`set_data`, visibility toggles, relim/autoscale).

### 2.5 Determinism and anti-zombie policy
- Same inputs must give same outputs (within solver tolerance).
- Dead/duplicate execution paths are to be removed or explicitly disabled.

---

## 3) Unified backlog (de-duplicated)

## P0 — Blockers (must close first)

1. **P0-1: RHS/Jacobian hot-path closure**
   - Confirm no allocation regressions and no contract drift regressions.
   - Validate branch stability against current contour.

2. **P0-2: Preset F demyelination acceptance**
   - Achieve and verify sustained attenuation target `ratio_f <= 0.30`.
   - Must pass both branch test and extended conduction utility artifacts.

3. **P0-3: GUI analytics memory discipline completion**
   - Eliminate remaining update-loop recreate/remove hotspots.
   - Verify sweep/repeated-runs responsiveness is stable.

## P1 — High-risk architecture/scientific debt (after P0)

4. **P1-1: Long-run calcium safety verification**
   - Confirm no NaN/Inf on prolonged pathological runs.
   - Include artifacted long-run verification report.

5. **P1-2: RHS scalar arg compression**
   - Pack stimulus/environment scalars in structured container to reduce positional fragility.

6. **P1-3: Dual implementation drift**
   - Reduce physics duplication risk in `run_euler_maruyama` vs RHS shared logic.

7. **P1-4: Lazy init heavy analytics tabs**
   - Startup footprint reduction without correctness regressions.

## P2 — Important but non-blocking

8. MainWindow decomposition.
9. Extended topology interactivity and neuron passport evolution.
10. Docs/l10n/user-guide final polish.

---

## 4) Mandatory execution order (phase-gated)

### Phase A (P0): Core + GUI hot paths
- RHS/Jacobian allocation and contract integrity checks.
- Analytics update-loop persistence completion.

### Phase B (P0): Preset F physiology closure
- Tune pathology parameters only through measurable acceptance runs.

### Phase C (Verify gate)
- Run consolidated P0/P1 gate + branch + utility artifacts.
- Do not move forward unless gate criteria are satisfied or documented blockers exist.

### Phase D (P1): Architecture/scientific debt
- Scalar packing + drift reduction + long-run Ca verification.

### Phase E (P2): UX/docs cleanup
- Only after P0/P1 closure.

---

## 5) Test protocol and artifact contract

Minimum required checks per critical cycle:

1. **Unit/contract suite (fast)**
2. **Branch physiology check**
3. **Utility artifact run(s)**
4. **Consolidated gate run**

Required artifacts for host exchange:
- `tests/artifacts/p0_p1_gate_user/p0_p1_gate_summary.json`
- `tests/artifacts/p0_p1_gate_user/p0_p1_gate_summary.md`
- `tests/artifacts/f_conduction_user.json`
- `tests/artifacts/preset_stress_user.json`
- `tests/artifacts/preset_stress_user.md`
- Latest run log from `tests/artifacts/codex_requested_runs/`

If dependencies are missing:
- Produce diagnostic JSON/MD with explicit `dependency_error`.
- Do not report scientific pass/fail as closed.

If timeout occurs:
- Classify as `TIMEOUT`, include stderr/stdout tails and duration.
- Treat as blocking for acceptance until resolved or explicitly waived.

---

## 6) Physiology validation doctrine (promoted from backlog contract)

1. Preset validation is not “test-pass only”; it must match plausible physiology:
   - spike count/rate regime,
   - membrane range,
   - channel parameter sanity,
   - pathology signature consistency.
2. Channel/preset logic changes must be validated in branch/testing contour before considering mainline closure.
3. Stress sweeps are mandatory for “wide-range” confidence (not just nominal point checks).

---

## 7) Commit discipline and reporting

For each logical step:
- one focused change-set,
- immediate relevant tests,
- update `CurrentTasks10.2.md` with factual status,
- commit with clear scope.

Never:
- claim completion without artifact/test evidence,
- silently change physics and postpone validation,
- mix unrelated phases in one uncontrolled batch.

---

## 8) Definition of done (global)

A task is closed only when all are true:

1. Code change implemented and scoped.
2. Relevant tests pass (or blockers explicitly evidenced).
3. Required artifacts/logs exist.
4. `CurrentTasks10.2.md` status is updated to match reality.
5. No new silent scientific inconsistency introduced.

