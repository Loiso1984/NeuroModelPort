# 🧠 AI Engineering Checklist v3 (Full System Integrity Protocol)

> ⚠️ Это НЕ TODO список  
> Это **система инженерных инвариантов, эвристик и целевых решений**,  
> которую AI-агент ОБЯЗАН применять при работе с кодовой базой.

Agent MUST:
- validate system integrity (not continuosly but periodically)
- fix low-cost issues immediately
- escalate or backlog complex refactors
- NEVER introduce silent inconsistencies

---

# 🧭 0. SYSTEM INVARIANTS (SOURCE OF TRUTH)

These define what is considered **correct architecture**.

---

## 0.1 Separation of Concerns

System MUST be logically split into:

- Simulation → RHS, solver, integration
- Analysis → Lyapunov, bifurcation, SD curve
- Visualization → GUI, plotting

### ❌ Violations

- [ ] Analysis embedded inside simulation execution
- [ ] Analysis controlled via simulation toggles
- [ ] Visualization mutates simulation state

---

## 0.2 GUI / UX STRUCTURE RULES

GUI MUST reflect system architecture.

### Required grouping:

- Simulation controls
- Analysis tools
- Visualization settings

---

### 🔴 CRITICAL RULE: Lyapunov Placement

- [ ] Lyapunov exponent is ANALYSIS
- [ ] MUST NOT be:
  - checkbox
  - inline toggle
  - placed near Iext / dt / solver params

- [ ] MUST be:
  - explicit analysis action
  - separated from simulation controls

➡️ If violated → **UI architecture bug**

---

### UI Smells

- [ ] unrelated controls grouped together
- [ ] simulation + analysis mixed
- [ ] computed values editable as primary inputs

---

## 0.3 Single Source of Truth

Each parameter MUST have ONE canonical representation.

### ❌ Violations

- [ ] Python RHS ≠ Cython RHS
- [ ] Iext ≠ Iext_absolute_nA
- [ ] cached derived values not updated

---

## 0.4 Explicit Contracts Only

System MUST NOT rely on:

- positional argument order
- hasattr / duck typing
- implicit parameter copying

---

## 0.5 Deterministic Simulation

Same inputs MUST produce same outputs.

### ❌ Violations

- [ ] different results with/without Cython
- [ ] hidden state
- [ ] argument misalignment

---

## 0.6 Zombie Code Definition

Code is "zombie" if:

- [ ] not used in execution path
- [ ] duplicates active implementation
- [ ] legacy/debug-only
- [ ] unreachable via configs

---

# 🔍 1. DETECTION HEURISTICS (HOW AGENT THINKS)

---

## 1.1 Dual Implementation Drift

IF:
- same logic exists in multiple places

THEN:
- [ ] compare implementations
- [ ] detect divergence
- [ ] enforce sync or deprecate

---

## 1.2 Argument Explosion

IF:
- args > 20 → fragile
- args > 30 → high risk
- args ~40 → 🔴 CRITICAL

THEN:
- [ ] validate all call sites
- [ ] detect positional misalignment

---

## 1.3 Duck Typing Detection

IF:
- hasattr(...)
- Optional[Any]

THEN:
- [ ] mark weak contract
- [ ] check runtime failure risk

---

## 1.4 Hidden State Detection

IF:
- value derived from another

THEN:
- [ ] check synchronization
- [ ] detect stale cache

---

## 1.5 UI Smell Detection

IF:
- control grouping inconsistent

THEN:
- [ ] classify domain (simulation / analysis / visualization)
- [ ] detect mixing

---

## 1.6 Zombie Code Detection

IF:
- code unused / duplicated / unreachable

THEN:
- [ ] mark as zombie
- [ ] suggest removal/refactor

---

# 🏗 2. TARGET ARCHITECTURE RULES (PREFERRED SOLUTIONS)

---

## 2.1 Defaults + Override (CRITICAL)

System MUST remain flexible.

- [ ] DO NOT hardcode scientific parameters as immutable constants
- [ ] Use:
  - recommended defaults
  - safe ranges (literature-based)
  - override capability

---

### Behavior Rules

- valid but risky → ⚠️ Warning
- physically impossible → ❌ ValidationError
- NEVER silent failure

---

## 2.2 Units System (Density vs Absolute)

### Canonical Design

- [ ] `Iext` (density) → SOURCE OF TRUTH
- [ ] `Iext_nA` → computed property (@property)

---

### Forbidden Pattern

- [ ] storing both as mutable independent fields

---

### Required Behavior

- [ ] computed on access
- [ ] always consistent
- [ ] GUI uses computed values

---

## 2.3 RHS Parameter Architecture (CRITICAL)

### Current Anti-pattern

- positional arguments (~40)
- fragile ordering
- silent bugs

---

### Target Design

- [ ] Python layer:
  - flexible config (Pydantic)

- [ ] Solver layer:
  - compiled parameter containers:
    - jitclass (preferred)
    - or structured arrays

---

### Goals

- [ ] stable RHS signature
- [ ] no positional misalignment
- [ ] fast memory access

---

## 2.4 Config Validation Layer

### MUST exist

- [ ] model-level validation before simulation

---

### Must detect:

- impossible physical parameters
- inconsistent ranges
- invalid morphology / dimensions
- broken solver assumptions

---

### Behavior

- invalid → ❌ fail fast
- risky → ⚠️ warning

---

# 🛠 3. CONCRETE SYSTEM CHECKS

---

## 3.1 RHS Consistency (Python vs Cython)

- [ ] compare:
  - `core/rhs.py`
  - `cython_rhs.pyx`

- [ ] ensure sync after changes

### 🔴 Risk

- different physics paths

---

## 3.2 Solver ↔ RHS Contract

- [ ] validate argument order consistency
- [ ] check all call sites

- [ ] detect silent bug:
  - correct types, wrong semantics

### 🔴 Risk

- simulation runs with corrupted physics

---

## 3.3 Config Schema Integrity

- [ ] detect:
  - hasattr
  - missing fields

- [ ] enforce fail-fast

---

## 3.4 Simulation Time Validity

- [ ] validate:
  - t_sim
  - dt_eval

- [ ] ensure steady-state for analysis

### ⚠️ Use Defaults + Override

---

## 3.5 Type Safety

- [ ] eliminate Any
- [ ] enforce explicit models

---

## 3.6 Units Consistency

- [ ] detect dual representations
- [ ] enforce computed property model

---

## 3.7 RHS Input Validation

- [ ] n_comp == len(y)
- [ ] morphology integrity

- [ ] add guards before Numba

---

## 3.8 Stimulus Coupling

- [ ] inspect implicit copying
- [ ] detect hidden dependency

---

## 3.9 Zombie Code

- [ ] detect unused logic
- [ ] detect duplicated implementations
- [ ] mark for removal

---

# 🎛 4. GUI-SPECIFIC RULES

---

## 4.1 GUI MUST NOT:

- [ ] mix simulation + analysis controls
- [ ] expose derived values as editable state
- [ ] hide validation/warnings

---

## 4.2 GUI MUST:

- [ ] reflect system architecture
- [ ] separate domains visually
- [ ] show warnings for unsafe parameters
- [ ] display computed values correctly

---

## 4.3 Critical Violations

- [ ] Lyapunov as checkbox → ❌
- [ ] mixed control groups → ❌
- [ ] inconsistent units display → ❌

---

# 🧩 5. ACTION POLICY

---

## 5.1 Immediate Fix

Apply if:

- [ ] local
- [ ] safe
- [ ] no API break

---

## 5.2 Backlog

Add if:

- [ ] requires refactor
- [ ] multi-module impact

---

## 5.3 Escalate

- 🔴 RHS contract issues
- 🔴 physics inconsistency
- 🔴 hidden state corruption

---

# 🚨 6. HIGHEST RISK AREA

## 🔴 RHS Argument Explosion

Symptoms:

- ~40 positional args
- hard to maintain
- silent bugs possible

---

Impact:

- undetectable physics corruption
- broken scientific validity

---

# 💡 CORE PRINCIPLE

> If the system can silently produce incorrect scientific results —  
> it is already broken, even if it "runs".