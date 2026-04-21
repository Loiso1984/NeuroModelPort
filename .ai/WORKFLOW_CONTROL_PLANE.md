# NeuroModelPort Workflow Control Plane

This file is the first stop for AI agents working in this repository. It consolidates routing,
prompt shape, context budget, validation gates, and safety rules.

## Operating Priority

1. Preserve biophysical validity.
2. Keep physics, GUI, analytics, docs, and tests synchronized.
3. Prefer deterministic validation artifacts over visual plausibility.
4. Keep context small, current, and sourced.
5. Optimize speed and cost only after correctness is protected.

## Agent Skills Source Of Truth

- Canonical skills directory: `.ai/skills/`
- `.ai/skills/` is the ONLY source of truth for AI instructions.
- Do not treat editor-created shadow folders (`.codex/`, `.cursor/`, `.kiro/`, `.pi/`, `.trae/`, `.trae-cn/`) as canonical.
- New skills must be added only under `.ai/skills/` and referenced from there in prompts/handoffs.
- If shadow folders need to be repopulated, run `python scripts/sync_skills.py` to mirror `.ai/skills/` into each shadow folder.

## Source Of Truth

Read these in order before planning a non-trivial change:

1. `.maestro.md` - model/provider context, project risks, priorities.
2. `docs/PHYSIOLOGY_VALIDATION_MEMORY.md` - mandatory simulation and validation rules.
3. `docs/VALIDATION_COVERAGE_STATUS.md` - current validation contour and open priority items.
4. `tests/README_TESTS.md` - canonical validation entry points.
5. `tests/branches/BRANCH_TEST_REGISTRY.md` and `tests/utils/UTILS_REGISTRY.md` - active vs legacy gates.

If a referenced task source such as `CurrentTasks10.2.md` or `AIDER_PLAN.md` is missing, do not
invent its contents. Treat `docs/VALIDATION_COVERAGE_STATUS.md` as the fallback queue and note the
missing file in the handoff.

## Task Routing

| Task type | Default route | Required companion checks |
|---|---|---|
| Core physics, presets, channels, solver, RHS, Jacobian | Single senior agent first; add review agent only for isolated audit | Literature validation, branch tests, deterministic artifact |
| Cross-layer parameter or feature | Single implementation agent with explicit sync checklist | Model -> Physics -> Simulation -> GUI -> Save/Load -> Docs -> Tests |
| GUI-only polish with no physiology changes | Single GUI agent | Compile/syntax and focused GUI smoke checks |
| Performance optimization | Numba/performance reviewer after local evidence is gathered | Benchmark before/after and no physics drift |
| Documentation-only update | Single agent | Link to canonical source and avoid stale duplicate claims |
| Large audit | Parallel auditors only when scopes do not overlap | Consolidated findings before fixes |

Use parallel agents only when write scopes are disjoint and the handoff contract is explicit. Do not
delegate just to satisfy a tool preference when a single agent can complete the task safely.

## Prompt Template

Use this shape for task prompts and handoffs:

```markdown
## Role
[One clear role for this task.]

## Context
- Source files:
- Canonical docs:
- Known risks:

## Instructions
1. Read before editing.
2. Preserve physics validity and cross-layer sync.
3. Update tests/docs when behavior changes.
4. Do not use legacy validators as promotion gates unless explicitly requested.

## Validation
- Required commands:
- Required artifacts:
- Acceptable skips:

## Output
- Files changed:
- Validation run:
- Residual risks:
```

## Context Budget

For ordinary tasks, keep context under these rough limits:

| Bucket | Budget | Notes |
|---|---:|---|
| Project rules | 1-3 files | Prefer this control plane plus `.maestro.md` and one domain rule file |
| Code under change | 3-8 files | Read direct callers/callees before editing |
| Tests and validators | 1-5 files | Prefer active registries and touched-area tests |
| Historical docs | 0-2 files | Use archive files only for background, not current truth |

Put high-risk constraints at both the beginning and end of long prompts: literature-sourced
parameters, active validation gates, and cross-layer synchronization.

## Validation Gates

| Change class | Minimum gate |
|---|---|
| Preset/channel/core calculation | Add or update `tests/branches` coverage, run active branch gate, persist deterministic artifact when applicable |
| Solver/RHS/Jacobian/performance | Run focused solver/Jacobian tests and a before/after benchmark; check allocation-sensitive hot paths |
| GUI analytics/rendering | Run syntax/compile plus focused GUI callback or rendering tests for touched paths |
| Documentation-only | Check links/paths and ensure docs point to current canonical files |
| Workflow-only | Verify referenced files exist and no task gate points only to missing docs |

Never promote a physiology change because output "looks realistic." Promotion requires literature
basis or documented physiological acceptance criteria plus the relevant active test gate.

## Safety Guards

- Reject or stop on missing required target files; do not silently continue with guessed scope.
- Treat destructive file or git operations as confirmation-required.
- Put timeouts on long validation and simulation commands; record if a run was skipped for time.
- Log or summarize validation artifacts with enough detail for the next agent to reproduce.
- Keep legacy scripts clearly labeled as legacy when mentioned.
- For ambiguous parameter units, stop and confirm from models, docs, or literature before editing.

## Handoff Contract

Every agent handoff or final report should include:

- Scope completed.
- Files changed or inspected.
- Validation commands run and result.
- Artifacts written or reviewed.
- Open risks, especially physics drift, cross-layer desync, missing docs, or skipped validation.
