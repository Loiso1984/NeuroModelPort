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

## 🧩 Addendum: Code Review Reconciliation (2026-04-05)

> Этот блок не заменяет основной план, а уточняет **приоритеты исполнения** и критерии закрытия/незакрытия по свежему ревью.

### Статус по ревью: что уже закрыто и что остаётся открытым

#### ✅ Подтверждённо закрыто (оставляем как done)
1. **Ca safety bounds (P1-1 technical implementation):** клемпы `Ca_i` и защитный Nernst-path реализованы в core-слое; остаётся только долгий прогон-верификация на патологических сценариях.
2. **RHS hot-path аллокации (P0-1 implementation core):** переход на preallocated `dydt` и removal heap-allocation в RHS выполнены; остаётся проверить устойчивость/регрессию на branch contour.
3. **Decimation base-path:** базовые адаптивные downsampling-механизмы добавлены и используются в plotting/analytics path.
4. **Delay target SSoT utility:** унификация через `gui/delay_target.py` выполнена и сохранена как обязательный путь для delay-focused UI.

#### 🔴 Открытые критические блокеры (P0, must-fix до новых фич)
1. **GUI analytics memory discipline до конца не доведена.**
   - Запрещены hot-path пересоздания/удаления artist/axes (`clear/cla/remove/add_subplot`-паттерны) в update-циклах.
   - Требование: фиксированная заранее созданная сетка осей + show/hide + `set_data()` обновления.
   - DoD: повторные прогоны/sweep без деградации responsiveness и без роста memory-footprint в цикле обновлений.
2. **Preset F (MS) physiology ещё не соответствует цели.**
   - Требование: добиться устойчивого патологического затухания/блока проведения (`ratio_f <= 0.30`) в branch + extended sweep.
   - Примечание: tuning делается итеративно, с обязательной фиксацией артефактов.

#### 🟠 Открытые высокорисковые долги (P1, после закрытия P0)
1. **RHS scalar-args compression (продолжение борьбы с arg-explosion).**
   - Следующий шаг: упаковать scalar stimulus/environment параметры в структурированный контейнер (`np.ndarray` фиксированной структуры либо иной numba-safe packed format).
   - Цель: уменьшение позиционной хрупкости сигнатуры и упрощение solver/jacobian/rhs-contract синхронизации.
2. **GUI SSoT для `Iext_absolute_nA` не полностью соблюдён.**
   - Требование: абсолютный ток только read-only presentation (hint/label), без обратной записи в editable form controls.
3. **Lazy initialization тяжёлых analytics tabs.**
   - Требование: не создавать все тяжёлые MPL canvas на старте; инициализация по первому заходу на вкладку.

### Исполняемый алгоритм (обязательная последовательность)
1. **Step A / P0:** закрыть GUI analytics memory leak discipline (`gui/analytics.py`) без изменения физики модели.
2. **Step B / P0:** закрыть физиологию Preset F (`core/presets.py`) до прохождения ratio-gate.
3. **Step C / Verify:** прогнать branch+utility проверки и зафиксировать JSON/MD артефакты.
4. **Step D / P1:** только после A+B перейти к SSoT `Iext_absolute_nA`, scalar-args packing и lazy tabs.

### Gate-критерии перед переходом к P1
- Branch test по MS attenuation стабильно зелёный.
- Extended F-conduction report подтверждает целевой режим (не «почти полная проводимость»).
- GUI update-path работает без пересоздания осей/artist в hot-loop.
- Обновлены статусы в `CurrentTasks10.2.md` + `docs/VALIDATION_COVERAGE_STATUS.md` (факт, без wishful-thinking).

## 🧩 Addendum: Drift & Contract Risks Intake (2026-04-05, late)

> Дополнение к плану (не замена): новые пункты из свежего ревью, с сохранением приоритетной оси P0→P1.

### 🔴 P0 (critical)
1. **Analytics hot-path must avoid dynamic axis destruction/recreation.**
   - Внесено в план как жёсткий запрет на `ax.remove()`/recreate subplot в update-loop.
   - Текущий статус: **in progress** (фиксированная сетка осей уже внедрена; требуется финальный аудит оставшихся remove-paths в analytics).
2. **Preset F attenuation gate remains blocking until confirmed by real run artifacts.**
   - Цель сохраняется: `ratio_f <= 0.30` на branch + extended utility.
   - Текущий статус: **open / blocking**.

### 🟠 P1 (high risk)
1. **Dual implementation drift in `run_euler_maruyama` (`core/advanced_sim.py`).**
   - Риск: стохастический контур дублирует физику отдельно от RHS.
   - План: вынести единый расчёт производных/токов в shared numba-safe helper и переиспользовать в RHS + EM path.
   - Статус: **open**.
2. **RHS signature fragility (`rhs.py` vs `jacobian.py`) despite contract checks.**
   - Риск: позиционный сдвиг аргументов может дать silent scientific corruption.
   - План: compress scalar stimulus/environment args в packed container (`stim/env params` block).
   - Статус: **open**.
3. **SSoT for `Iext_absolute_nA` presentation in GUI.**
   - Требование: read-only display вне editable form fields.
   - Статус: **✅ closed in GUI hint path** (абсолютный ток теперь отображается только текстом в `lbl_params_hint`; обратная запись в form field удаляется).
4. **Dendritic filter tau-edge verification (`tau<=0`).**
   - Текущий статус: **✅ logic guarded** (filter auto-disables; no divide-by-zero path in active derivative branch).
   - Остаток: поддерживать regression coverage при будущих refactor.

---

## Цель
Сопоставить новую критику с `ATTENTION.md`, `AIDER_PLAN.md`, `docs/MASTER_BACKLOG_CONTRACT.md` и зафиксировать **единый приоритетный план** без расползания контекста.

## Что считаем источниками истины
1. Архитектурные инварианты и запреты: `ATTENTION.md`.
2. Исполняемый пошаговый техплан: `AIDER_PLAN.md` (секция `AGENT ACTION PLAN v10.2`).
3. Контракт и процесс для физиологии/пресетов: `docs/MASTER_BACKLOG_CONTRACT.md`.
4. Фактический статус покрытия/долгов: `docs/VALIDATION_COVERAGE_STATUS.md`.
5. Сводный исполняемый канон без дублей: `MASTER_EXECUTION_CANON_v10.2.md`.

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
- ✅ Analytics hot-path cleanup progressed further: sweep traces/colorbar, bifurcation peaks, impedance diagnostics, and spectrogram mesh now use persistent artists/state updates (no per-update artist/axis removals in these paths).
- ✅ Solver fallback hardening expanded: BDF primary step now falls back to LSODA on any solver-side exception class (not just `RuntimeError`), reducing platform-specific crash risk in stiff/singular episodes.
- 🟠 Preset F demyelination retuned toward stronger conduction block signature: increased `Ra`, leak (`gL`), and effective membrane capacitance (`Cm`) while reducing `gNa_max`; pending verification artifacts from branch/extended runs.
- ✅ Sweep analytics trace lifecycle corrected for sparse/partial sweep results: persistent trace slots now hide by actual plotted-series count, preventing stale curves from prior runs.
- ✅ P0/P1 gate runner hardened against hangs: per-step timeout policy added with explicit `TIMEOUT` classification and blocking semantics in consolidated summary.
- ✅ Windows execution helper synced with timeout-capable gate runner (`RUN_TESTS_FOR_Codex.bat` now passes `--step-timeout-sec`), reducing “stuck test” risk on host runs.
- ✅ Extended F-conduction acceptance policy aligned with demo objective: utility now supports conduction-block acceptance (`block_or_delay` / `block_only`) so F can pass on strong attenuation even when terminal delay is undefined due to propagation block.
- ✅ Oscilloscope hot-loop persistence improved: key voltage/gate/current traces now reuse persistent PlotDataItem objects (setData/setVisible) instead of full per-run recreation.

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
