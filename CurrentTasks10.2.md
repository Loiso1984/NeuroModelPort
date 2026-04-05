# CurrentTasks10.2 — Critique Reconciliation & Execution Priority

## 🔁 Execution Reset (2026-04-05)

Чтобы вернуться к "действительно важным задачам", фиксируем жесткий reset порядка работ:

1. **Только P0/P1 до отдельного подтверждения.** Новые docs/UX-polish не делаем, пока не закрыты блокеры производительности/физиологии.
2. **Каждый рабочий цикл = 1 критическая задача + проверка.** Никаких смешанных "больших" PR.
3. **Definition of Done для каждого шага:**
   - есть измеримый критерий (perf/ratio/NaN safety),
   - есть профильный тест/утилита,
   - есть короткий факт-лог в этом файле.

### Immediate next queue (strict)
- [ ] **P0-1:** финально закрыть RHS/Jacobian hot-path аллокации и подтвердить безрегрессионность на branch-suite.
- [ ] **P0-2:** добить Preset F conduction attenuation до целевого `ratio_f < 0.3` на extended sweep.
- [ ] **P1-1:** довести Ca safety bounds на длинных патологических прогонах (без NaN/inf).

---

## Цель
Сопоставить новую критику с `ATTENTION.md`, `AIDER_PLAN.md`, `docs/MASTER_BACKLOG_CONTRACT.md` и зафиксировать **единый приоритетный план** без расползания контекста.

## Что считаем источниками истины
1. Архитектурные инварианты и запреты: `ATTENTION.md`.
2. Исполняемый пошаговый техплан: `AIDER_PLAN.md` (секция `AGENT ACTION PLAN v10.2`).
3. Контракт и процесс для физиологии/пресетов: `docs/MASTER_BACKLOG_CONTRACT.md`.
4. Фактический статус покрытия/долгов: `docs/VALIDATION_COVERAGE_STATUS.md`.

---

## Сводка критики → контрактные пункты

### P0 (блокирующие, must-fix до новых фич)
1. **RHS allocation в Numba**
   - Симптом: аллокация в RHS-цикле (`np.zeros_like(y)`), деградация производительности.
   - Контракт: ATTENTION 1.2/2.3 + AIDER Stage 2.1.
   - Файлы: `core/rhs.py`, `core/solver.py`, `core/jacobian.py`, `core/rhs_contract.py`.

2. **Matplotlib leak / re-create anti-pattern**
   - Симптом: `.clear()` / `.cla()` в update-циклах аналитики.
   - Контракт: AIDER Stage 4.1.
   - Файлы: `gui/analytics.py`.

3. **Preset F (MS) не демонстрирует демиелинизацию**
   - Симптом: почти полная проводимость soma→terminal (ratio ~1.0).
   - Контракт: backlog п.1/6 + VALIDATION caveat.
   - Файлы: `core/presets.py`, `tests/branches/test_pathology_mode_sweep_branch.py`, `tests/utils/run_f_conduction_extended.py`.

### P1 (высокий долг, следующий после P0)
4. **RHS argument explosion (~49 args)**
   - Симптом: высокая хрупкость сигнатур и каскадная стоимость изменений.
   - Контракт: ATTENTION 1.2 и 2.3.
   - Файлы: `core/rhs.py`, `core/solver.py`, `core/jacobian.py`, `core/rhs_contract.py`.

5. **Ca safety upper bound / NaN protection**
   - Симптом: риск разлета `Ca_i` на длинных патологических прогонах.
   - Контракт: AIDER Stage 2.3/3.4 + backlog п.1.
   - Файлы: `core/rhs.py`, профильные тесты пресетов N/O.

### P2 (важно, но не блокирует P0/P1)
6. **MainWindow monolith и state-sync anti-pattern** (`blockSignals`, ручной sync).
7. **Dual representation Iext cleanup (SSoT на computed field)**.
8. **Lazy tabs в analytics + downsampling для длинных трасс**.
9. **Topology interactivity (clickable compartments + passport focus)**.

---

## Фазный план исполнения (строго последовательно)

### Phase 1 — Numba & Memory (P0)
**Контекст только:** `core/rhs.py`, `core/solver.py`, `gui/analytics.py` (+ минимум нужных зависимостей).

**Задачи:**
1. Убрать внутренние аллокации из RHS, перейти на preallocated output.
2. Убрать `.clear()`/`.cla()` из hot-update paths в analytics, перейти на persistent `Line2D.set_data`.

**Критерии готовности:**
- Производственные тесты ветки проходят.
- Нет regressions в GUI update cycle.
- Отдельный sanity-check на повторные прогоны без роста времени перерисовки.

### Phase 2 — Biophysics (P0/P1)
**Контекст только:** `core/presets.py`, `tests/branches/test_pathology_mode_sweep_branch.py`.

**Задачи:**
1. Довести Preset F до затухания/блока проводимости (цель: ratio_f < 0.3).
2. Усилить safety bounds для Ca в `core/rhs.py` с тестовой верификацией на длинном прогоне.

**Критерии готовности:**
- `run_f_conduction_extended` подтверждает провал проводимости/сильное затухание.
- Нет NaN/inf в патологических сценариях при удлиненном `t_sim`.

### Phase 3 — Architecture (P1)
**Контекст только:** `core/rhs.py`, `core/solver.py`, `core/jacobian.py`.

**Задачи:**
1. Сократить число позиционных аргументов RHS через структурированную упаковку.
2. Синхронизировать контракты (`rhs_contract`) и тестовые проверки.

**Критерии готовности:**
- Совместимость с jacobian modes.
- Полный прогон активной валидации без расхождений физиологии.

### Phase 4 — GUI hardening (P2)
**Контекст:** `gui/main_window.py`, `gui/analytics.py`, `gui/plots.py`, `core/analysis.py`.

**Задачи:**
1. Lazy init тяжелых analytics tabs.
2. Decimation/peak-preserving plotting path.
3. Удаление грязного sync для `Iext_absolute_nA` из форм.

### Phase 5 — Preset Stress Validation (P1/P2)
**Контекст:** `core/presets.py`, `tests/branches/*`, `tests/utils/*`, `docs/VALIDATION_COVERAGE_STATUS.md`.

**Задачи:**
1. Стресс-валидация всех пресетов на широких диапазонах (`Iext`, `t_sim`, `T`, шум, morphology multipliers) с автоматическим сбором метрик.
2. Проверка нейрофизиологической правдоподобности пресетов:
   - частотные диапазоны, мембранные уровни, baseline firing regime;
   - согласованность параметров пресета (`g*`, `E_rev`, `Ra`, morphology) со справочными источниками;
   - sanity-check на «здравый смысл» (например, отсутствие небиофизичных спайк-режимов при базовых настройках).
3. Для каждого пресета — отчет «expected vs observed» с явным статусом: `PASS / WARN / FAIL`.

**Критерии готовности:**
- Покрыты все presets из `core/presets.py` с единым стресс-харнессом.
- Валидация не ограничивается «тест прошел», а содержит интерпретируемые физиологические метрики.
- `docs/VALIDATION_COVERAGE_STATUS.md` обновлен по фактическим результатам.

### Phase 6 — Final Polish & Documentation (P2)
**Контекст:** `docs/*`, `README.md`, `gui/locales.py`, пользовательские подписи GUI.

**Задачи:**
1. Дополнение документации после функциональных правок (что изменилось, как валидировать, ограничения).
2. Проверка двуязычности (RU/EN):
   - ключевые документы (README + основные docs);
   - основные элементы интерфейса и сообщения.
3. Подготовка краткого руководства пользователя (quick-start + типовые сценарии симуляции/аналитики).

**Критерии готовности:**
- Документация соответствует текущему состоянию кода и тестового покрытия.
- RU/EN строки не расходятся по смыслу в ключевых user flows.
- Есть компактный user guide для запуска, базовой настройки и чтения результатов.

---

## Тестовая матрица по фазам
- Phase 1: профильные branch-тесты + smoke run GUI analytics update path.
- Phase 2: `tests/branches/test_pathology_mode_sweep_branch.py` + `tests/utils/run_f_conduction_extended.py`.
- Phase 3: `python tests/utils/run_active_branch_suite.py`.
- Phase 4: GUI smoke + регрессионные веточные тесты по delay/analytics.
- Phase 5: стресс-харнесс пресетов + branch-suite + сверка с reference-диапазонами.
- Phase 6: doc/l10n checklist + smoke-run ключевых GUI сценариев RU/EN.

---

## Правила выполнения (операционные)
1. Не смешивать фазы в одном контексте.
2. После каждой фазы: тесты → commit.
3. Любые изменения в пресетах/физиологии: branch-валидация перед переносом дефолтов.
4. Новую документацию не плодить; обновлять только `CurrentTasks10.2.md`, `AIDER_PLAN.md`, `docs/VALIDATION_COVERAGE_STATUS.md` по факту.

---

## Progress Log
- ✅ Phase 1 (частично): hot-path updates для `core/solver.py` + `gui/analytics.py` начаты в отдельных коммитах.
- ✅ Phase 2 (in progress): внесена калибровка Preset F и добавлен строгий ratio-target gate в `tests/utils/run_f_conduction_extended.py`; добавлен branch-тест на attenuation для F.
- ✅ Phase 3 (in progress): начат рефактор RHS contract — отдельные `g*_v`/`phi_*` упакованы в `gbar_mat`/`phi_mat` для уменьшения позиционной хрупкости.

## Audit Delta (added after implementation-vs-doc review)
- 🟠 GUI memory hot-path migration is near-complete: Matplotlib gates/currents avoid `fig.clear()`, and Traces tab switched from `PlotItem.clear()` to persistent `PlotDataItem.setData()`; minor PyQtGraph reset debt remains in non-critical panes.
- ✅ Topology pane no longer uses full `PlotItem.clear()` per redraw; it now removes only tracked dynamic items and keeps shared plot infrastructure persistent.
- ✅ Oscilloscope plots now clear via tracked-item removal (no `PlotItem.clear()`/legend re-create loop), improving adherence to GUI persistent-artist recommendations.
- ✅ Dendritic filter edge case (`tau<=0`) is now guarded in shared distribution path: filter auto-disables to pure attenuation, preventing stale filtered-state sink behavior.
- ✅ Cython skeleton explicitly marked non-production: `core/optimization/cython_rhs.pyx` now raises `NotImplementedError` to prevent accidental scientific use before feature-parity validation.
- ✅ Lyapunov architecture debt closed in domain model: `analysis.enable_lyapunov` removed; FTLE/LLE now runs only via explicit action flag (`compute_lyapunov`) instead of hidden config coupling.
- ✅ Delay-target duplication addressed by shared GUI resolver utility (`gui/delay_target.py`) used in plots + topology.
- ✅ Stochastic Euler–Maruyama path migrated off legacy global `phi`: per-channel/per-compartment `phi_*` vectors now drive Na/K/Ih/Ca/IA gate kinetics and Langevin noise scaling.
- ✅ MainWindow absolute-current preview now uses shared conversion utility (`density_to_absolute_current`) instead of inline duplicated formula.
- ✅ RHS contract guardrails tightened: binary feature flags (`dual_stim_enabled`, `use_dfilter_primary`, `use_dfilter_secondary`) are now explicitly validated as `0|1` before kernel entry.
- ✅ RHS contract now validates packed matrix value domains: `gbar_mat` must be finite/non-negative, `phi_mat` must be finite/strictly-positive to prevent silent kernel corruption from bad inputs.
- ✅ RHS contract vector-domain checks extended to `cm_v` and `b_ca` (`cm_v>0`, `b_ca>=0`, finite), catching malformed solver vectors before Numba execution.
- ✅ Event-queue contract checks added: first `n_events` timestamps must be finite and non-decreasing, preventing malformed synaptic schedules from entering RHS kernels.
- ✅ RHS scalar-domain guards added for physical parameters (`t_kelvin`, `tau_ca`, `tau_sk`, `atau`, `atau_2`, `ca_ext`, `ca_rest`, `mg_ext`) to fail fast on non-physical/non-finite values.
- ✅ RHS contract now validates supported stimulation type IDs (`stype`, `stype_2`) against explicit allowed set to prevent silent fallback behavior from invalid mode codes.
- ✅ RHS stimulation scalar checks added: finite `iext/t0/td/zap_*` fields and non-negative duration/frequency bounds (`td*`, `zap_f*`) for both primary and secondary stimulus inputs.
- ✅ GUI decimation phase progressed: stride point-capping active in Analytics Traces + Oscilloscope, now with viewport-aware point budgets (adaptive to current plot width).
- 🟠 Phase 5 bootstrap extended: `tests/utils/run_preset_stress_validation.py` now includes broad sweeps + expected-range checks (firing/Vm envelopes) and preset parameter sanity audit to form `expected vs observed` PASS/WARN/FAIL reports.
- ✅ Stress-harness UX improved: missing runtime dependencies now produce explicit diagnostic message/exit code instead of raw traceback, simplifying CI/environment triage.
- ✅ Preset stress harness now emits both JSON and Markdown reports (`by_preset` summary + WARN/FAIL samples) for direct inclusion into validation docs/review notes.
- ✅ Stress-harness reference ranges externalized to `tests/utils/preset_reference_ranges.json` (keyword rules + default envelopes), enabling iterative literature alignment without code edits.
- ✅ RHS contract secondary-stim validation is now conditional on `dual_stim_enabled==1`; single-stim launches no longer fail on irrelevant secondary fields (fix for `Invalid atau_2` on regular runs).
- ✅ Jacobian Nernst path aligned with RHS calcium safety envelope: ICa/ITCa now clamp `Ca_i` to `[CA_I_MIN_M_M, CA_I_MAX_M_M]`, reducing Jacobian/RHS mismatch risk on long pathological simulations.

## Low-Priority GUI Backlog (compiled from AIDER + new recommendations)
> Ниже — только уникальные, реализуемые пункты (без дублей уже закрытых задач).

1. **Workflow navigation shell (Setup → Dashboard → Analytics)**  
   _Scope:_ `gui/main_window.py` (layout orchestration), reuse existing widgets.  
   _Deliverable:_ трехзонный поток без размножения табов, с backward-compatible маршрутизацией старых панелей.

2. **Interactive Topology v2 (click-select compartment + cross-highlight in plots)**  
   _Scope:_ `gui/topology.py`, `gui/plots.py`.  
   _Deliverable:_ клик по сегменту выставляет target compartment и подсвечивает соответствующие кривые тока/напряжения.

3. **Topology voltage heatmap overlay (real-time Vm colormap)**  
   _Scope:_ `gui/topology.py` + lightweight update bus from latest `SimulationResult`.  
   _Deliverable:_ color-mapped сегменты (blue→red) по текущему `V_m`.

4. **Passport cards widget (Excitability / Channel Balance / Spike Shape)**  
   _Scope:_ `gui/analytics.py` (new sub-widget), optional extraction helpers in `core/analysis.py`.  
   _Deliverable:_ карточки вместо текстового лога, в т.ч. pie-chart по вкладу токов.

5. **Additional analytics plots (Phase Plot, ISI histogram, V–m phase plane, gNa/gK traces)**  
   _Scope:_ `gui/analytics.py`.  
   _Deliverable:_ отдельные lightweight panels с guard-логикой на короткие/шумные записи.

6. **MainWindow decomposition (SimulationEngine / ConfigManager / VisualizerOrchestrator)**  
   _Scope:_ `gui/main_window.py` + новые controller modules.  
   _Deliverable:_ постепенный вынос orchestration-логики из God-object без поломки сигналов.

7. **Adaptive downsampling for long traces (stride now, LTTB optional next)**  
   _Scope:_ `gui/plots.py`, `gui/analytics.py`.  
   _Deliverable:_ cap на отображаемые точки по ширине viewport; перерасчет при zoom/pan.
