��# AIDER INSTRUCTION PLAN: NeuroModelPort v10.1 Refactoring & Enhancement
**Role:** Expert Computational Neuroscientist & Senior Python/Qt Developer.
**Context:** This project is a biophysical single-neuron simulator based on the Hodgkin-Huxley formalism. It uses `scipy.integrate.solve_ivp` (BDF method) for stiff ODEs and `Numba @njit` for the Right-Hand Side (RHS) kernel. The GUI is built with PySide6.
**Goal:** Execute a deep refactoring to improve performance (Numba optimization), fix mathematical artifacts (spike detection, phase-locking), resolve UI memory leaks, and add new ion channels/presets.
**RULE** The second important doctument is MASTER_BACKLOG_CONTRACT.md READ IT but only then you begin working with ion channels characteristics. It contains important instructions and info but the big part of the work from there been completed already 
## 🛑 STRICT RULES (CRITICAL)
1. **Numba Performance:** Inside `@njit` functions (especially in `core/rhs.py`), **DO NOT** allocate memory inside loops (e.g., no `np.zeros()`, `np.empty()`, or list comprehensions). Use pre-allocated arrays passed as arguments or scalar variables.
2. **Physics Integrity:** Never change physical parameters (conductances, Q10, reversal potentials) unless explicitly instructed.
3. **GUI Memory Leaks:** In `gui/analytics.py` and `gui/plots.py`, **DO NOT** use `fig.clear()` or recreate `axes`/`subplots` during update cycles. Initialize `Line2D` objects once in `__init__` and update them using `line.set_data()` followed by `canvas.draw_idle()`.
4. **Git Commits:** After successfully completing each logical step (or sub-step) below, make a commit with a descriptive message (e.g., `refactor(core): optimize Numba allocations in rhs_multicompartment`).
5. **Testing:** Run existing tests (e.g., `python tests/branches/test_hcn_isolated_branch.py`) after modifying core physics to ensure no regressions.

---

## 🗺️ EXECUTION ROADMAP
*Please execute these steps sequentially. Ask for permission before moving to the next major stage.*

### STAGE 1: Repository Cleanup & Test Standardization
*Goal: Remove technical debt and standardize the testing framework.*
- [ ] **1.1 Move Sandbox/Debug Files:** Move all root-level scripts (`simple_l5_test.py`, `debug_*.py`, `check_*.py`, `analyze_*.py`, `validate_*.py`, `calibrate_presets.py`, `literature_cross_reference.py`) into appropriate subdirectories (`tests/sandbox/`, `scripts/`, or `tools/`). The project root must contain only `main.py`, `.gitignore`, `requirements.txt`, and `README.md` (and this plan).
- [ ] **1.2 Remove Custom Test Runners:** Delete custom local versioning/testing scripts (`scripts/feature_manager.py`, `scripts/development_workflow.py`, `scripts/test_runner.py`) and the `_features/` directory. We rely entirely on Git and `pytest`.
- [ ] **1.3 Consolidate Documentation:** Review the 20+ `.md` files in the root and `docs/`. Consolidate redundant information into a clean hierarchy in `docs/`: `USER_GUIDE.md`, `PHYSICS_REFERENCE.md`, `DEVELOPER_GUIDE.md`. Keep only `README.md` in the root.
- [ ] **1.4 Pytest Standardization:** Ensure all tests in the `tests/` directory are compatible with `pytest`. Refactor scripts in `tests/utils/run_*.py` to use standard `pytest` assertions and fixtures where applicable.

### STAGE 2: Extreme Core Performance Optimization
*Goal: Eliminate bottlenecks causing simulation freezes and speed up calculations by 5-20x for future network simulations.*
- [ ] **2.1 Numba Allocation Fix (`core/rhs.py`):** Refactor `rhs_multicompartment`. Remove all intermediate array creations (like `np.zeros(n_comp)` for currents) inside the function. Implement a single, C-style `for i in range(n_comp):` loop. Calculate currents as scalar variables and write the result directly into the pre-allocated `dydt` array.
- [ ] **2.2 Jacobian Optimization (`core/jacobian.py`):** In `analytic_sparse_jacobian`, stop instantiating a new `lil_matrix` on every call. Instead, pass the pre-computed CSR structure (`data, indices, indptr`) from the solver and only update the `data` array inside the Numba loop.
- [ ] **2.3 Conditional Nernst Calculation (`core/rhs.py`):** The Nernst potential (`nernst_ca_ion` logarithm) is currently calculated at every time step even when `dynamic_Ca = False`. Make this calculation strictly conditional to save CPU cycles.
- [ ] **2.4 Channel-Specific Q10 (`core/models.py`, `core/kinetics.py`):** Refactor the temperature coefficient (`phi`). Currently, one global `phi` scales all gates. Change this to use channel-specific Q10 values (e.g., $Q10_{Na} \approx 2.2$, $Q10_K \approx 3.0$, $Q10_{Ih} \approx 4.0$).
[ ] ** 2.5 Numba-native Solver Loop: Gradually phase out scipy.integrate.solve_ivp. Write a custom implicit or semi-implicit integration loop (e.g., Backward Euler with Thomas algorithm / Hines matrix solver for tree structures) entirely enclosed in a @njit block. This eliminates Python-to-C context switching per timestep.

### STAGE 3: Math & Analytics Fixes
*Goal: Fix false positives in spike detection and protect advanced analytical algorithms from edge cases.*
- [ ] **3.1 State Machine Spike Detector (`core/analysis.py`):** Rewrite the `detect_spikes` function. The current algorithm (using `scipy.signal.find_peaks` or simple threshold crossing) creates false positives during depolarization blocks (plateaus). Implement a Numba-jitted Finite State Machine (States: `RESTING`, `DEPOLARIZING`, `REFRACTORY`) that counts a spike **only** if the voltage crosses the `threshold`, reaches a peak, and **strictly returns** below `baseline_threshold`.
- [ ] **3.2 Phase-Locking Protection (`core/analysis.py`):** In `estimate_spike_modulation`, add a guard clause: if `t_sim` is less than 3-5 periods of the lowest cutoff frequency (e.g., < 1000 ms for 4 Hz), return `NaN` to prevent meaningless results due to Butterworth filter edge artifacts. (Consider replacing Butterworth with Morlet wavelets if feasible without heavy dependencies).
- [ ] **3.3 Lyapunov Exponent (LLE) Protection (`core/analysis.py`):** In `estimate_ftle_lle`, add a condition to only compute LLE if `t_sim > 1000 ms`, and explicitly discard the first 200 ms of data (transients) before searching for nearest neighbors on the attractor.
- [ ] **3.4 Volume-dependent Calcium Dynamics:** Modify B_Ca calculation. Instead of a global constant, B_Ca must be calculated per compartment based on its surface area and volume. This will accurately reflect why thin dendrites have massive Ca2+ spikes compared to the soma.

### STAGE 4: UI/UX & Memory Leak Resolution
*Goal: Make the GUI responsive and prevent memory leaks during long analysis runs.*
- [ ] **4.1 Matplotlib Memory Leaks (`gui/analytics.py`):** Fix the updating logic in `_update_currents`, `_update_equil`, `_update_phase`, etc. Remove `self.fig_*.clear()` and `add_subplot()` from the update methods. Initialize the axes and `Line2D` objects once in `__init__`. Use `line.set_data()` and `canvas.draw_idle()` to update plots.
- [ ] **4.2 Main Thread Offloading (`gui/main_window.py`):** Heavy UI-blocking operations like `run_sweep` and `run_excitability_map` currently freeze the GUI due to the GIL, even if run in a basic thread. Refactor these to use `ProcessPoolExecutor` (similar to Monte-Carlo) and add a progress bar with a `Cancel` button.
- [ ] **4.3 MainWindow Refactoring:** `MainWindow` is a 700+ line God Object. Refactor it by extracting responsibilities into separate classes: `SimulationController` (handles threading/execution) and `ConfigManager` (handles presets and dual stim state). Keep `MainWindow` strictly for UI layout and event binding.
- [ ] **4.4 Interactive Topology (`gui/topology.py`):** Enhance the `TopologyWidget` using `pyqtgraph` features. Add pan/zoom capabilities and make compartments clickable to show a "Neuron Passport" info panel summarizing that compartment's properties.
- [ ] **4.4.2 ADDITIONAL GUI REFACTORING RECOMENATIONS:** They are not mandatory but may be efficient for doing the GUI - User Friendly
Спецификация переработки GUI: NeuroModelPort v10.1
Спецификация переработки GUI: NeuroModelPort v10.1
1. Общая архитектура макета (Layout)
Тип: Одностраничное приложение (SPA) с фиксированной высотой (h-screen) и отсутствием внешней прокрутки.
Структура:
Header (56px): Логотип, версия, переключатель языков (RU/EN) и основная кнопка действия «RUN SIMULATION».
Sidebar (320px): Левая панель для выбора пресетов, быстрой настройки параметров и краткой сводки анализа.
Main Content: Гибкая область справа с табами для переключения между визуализациями.
Footer (32px): Статус-бар с технической информацией (тип солвера, количество компартментов).
2. Компоненты боковой панели (Sidebar)
Селектор пресетов: Выпадающий список с 15 научными пресетами. Под списком — динамическое описание пресета на выбранном языке (курсивом, малый шрифт).
Quick Controls: Слайдеры для наиболее часто меняемых параметров (Амплитуда стимула, Температура). Значения должны отображаться в реальном времени с использованием моноширинного шрифта.
Analysis Panel (Neuron Passport): Сетка 2x3 с иконками (lucide-react) для отображения ключевых метрик:
Количество спайков, Частота (Hz), Пиковый потенциал, Потенциал покоя, Дно AHP, Индекс адаптации.
Прогресс-бар внизу, визуализирующий активность нейрона.
3. Основная рабочая область (Tabs)
Реализована через AnimatePresence (framer-motion) для плавных переходов.
А. Oscilloscope (Осциллограф)
Верхний график: Линейный график (LineChart) для мембранного потенциала 
Нижний график: Областной график (AreaChart) для ионных токов 
Требования: Использование библиотеки recharts. Отключение анимации точек для производительности при больших массивах данных. Сетка только по горизонтали.
Б. Topology (Топология)
Визуализация: Схематичное SVG-изображение нейрона.
Элементы: Сома (круг), AIS (прямоугольник с эффектом свечения/пульсации), Аксон (линия).
Интерактивность: Маркеры стимула (STIM 1, STIM 2) должны менять положение в зависимости от выбранной в параметрах локации (Soma/AIS/Dendrite).
В. Parameter Editor (Редактор параметров)
Организация: Сетка карточек по категориям (Морфология, Каналы, Стимуляция, Кальций, Среда, Симуляция).
Элементы управления:
Слайдеры для числовых значений.
Чекбоксы для включения/выключения каналов (Ih, ICa, IA, SK).
Поля ввода для точной настройки при включении канала.
Стиль: Темные подложки карточек (bg-[#161922]), акцентные цвета для заголовков секций.
4. Стилизация и визуальный язык
Цветовая палитра:
Фон: slate-950 (глубокий черный/синий).
Акцент 1 (Primary): blue-500 (для сомы и основных сигналов).
Акцент 2 (Secondary): rose-500 (для AIS и натриевых токов).
Акцент 3 (Warning): amber-500 (для стимула).
Типографика:
Основной: Inter (без засечек).
Данные: JetBrains Mono (для чисел и технических параметров).
Эффекты: Стеклянный эффект (backdrop-blur) для хедера, тонкие границы (border-slate-800), кастомные скроллбары.
5. Локализация (i18n)
Реализована через TranslationProvider. Все строки (заголовки, лейблы, описания) должны браться из словаря.
Тип Language = 'EN' | 'RU'.
6. Инструкции по логике GUI
Реактивность: Любое изменение слайдера в Sidebar или Parameter Editor должно немедленно обновлять объект config в состоянии React.
Simulation Trigger: Симуляция запускается по кнопке "Run" или при смене пресета. Использовать setTimeout(..., 0) для предотвращения блокировки UI-потока тяжелыми вычислениями.
NaN Protection: Все выводимые числовые значения должны принудительно приводиться к строке или форматироваться через .toFixed(), чтобы избежать ошибок рендеринга React при получении NaN из математического движка.
### STAGE 5: New Ion Channels & Preset Physics Fixes
*Goal: Elevate the simulator to NEURON-level accuracy by adding missing critical channels and correcting existing presets.*
- [x] **5.1 T-type Calcium Current ($I_T$):** ✅ Implemented in `core/kinetics.py` + `core/rhs.py`. Preset K uses it for LTS bursts.
- [x] **5.2 M-type Potassium Current ($I_M$):** ✅ Implemented. Exposed via `enable_IM` / `gIM_max` in GUI.
- [x] **5.3 Persistent ($I_{NaP}$) & Resurgent ($I_{NaR}$) Sodium:** ✅ Both implemented with literature kinetics. Optional via `enable_NaP` / `enable_NaR`.
- [x] **5.4 Fix "Thalamic Relay" Preset:** ✅ Now uses T-type Ca. Generates LTS + Na burst on recovery from hyperpolarization.
- [x] **5.5 Fix "CA1 Pyramidal" Preset:** ✅ Renamed to `CA1 Pyramidal (Adapting)`. $g_A$ normalized.
- [x] **5.6 Add 4 New Presets (`core/presets.py`):** ✅ All four added:
    1. `P: Thalamic Reticular Nucleus (TRN Spindles)` — sleep spindles via $I_T$.
    2. `Q: Striatal Spiny Projection (SPN)` — long first-spike latency via strong $I_A$.
    3. `R: Cholinergic Neuromodulation (ACh)` — $I_M$ block → tonic firing.
    4. `S: Pathology: Dravet Syndrome (SCN1A LOF)` — FS with reduced $g_{Na}$.
- [ ] ** 5.6 Energy / ATP Consumption Metrics Refinement: Currently ATP is estimated solely from QNa Enhance this to include QCa 
(Ca2+ pumps are highly ATP-expensive) and resting pump activity (Na+/K+ ATPase baseline cost). Это позволит моделировать "метаболическую усталость", которая является ключом к пониманию эпилептических припадков и ишемии.
- [ ]**5.7 Conductance-Based Synaptic Modeling (CRITICAL): Rewrite synaptic stimulation (AMPA, NMDA, GABA). Move away from current-based injection (Iext). Implement true conductance-based synapses: 
[ ] ** 5.8 NMDA Voltage-Dependent Block: Implement the block for NMDA receptors using standard functions
to accurately model non-linear dendritic integration.
[ ] 5.9 Delay Kinetics for SK Channel: Modify SK channel implementation to include a gating variable governed by an ODE, rather than an instantaneous

### STAGE 6: Advanced Architectural Features
*Goal: Add cutting-edge analysis and prepare the architecture for Phase 8 (Network Modeling).*
- [x] **6.1 Spectrogram Analysis (`gui/analytics.py`):** ✅ Tab 12 added. STFT with adaptive window, inferno colormap, safe colorbar management.
- [x] **6.2 Dynamic Temperature Gradients:** ✅ `T_dend_offset` in `EnvironmentParams`. `build_phi_vector()` interpolates linearly soma→dendrite. Per-compartment phi vectors in RHS/Jacobian.
- [x] **6.3 Event-Driven Synapses:** ✅ `event_times` queue in `StimParams`. `get_event_driven_conductance()` @njit. Passed as float64 array + n_events to RHS.
- [x] **6.4 NeuroML Export:** ✅ `core/neuroml_export.py` — exports `FullModelConfig` to NeuroML 2.2 XML. Button in main toolbar.

### Stage 7 — Membrane Impedance Z(f) ✅ COMPLETED
- [x] **7.1 ZAP/chirp stimulus** (`stim_type='zap'`): `get_stim_current` case 10, linear frequency sweep from `zap_f0_hz` to `zap_f1_hz`. Reuses `t0`/`td`/`atau` slots to avoid arg explosion.
- [x] **7.2 `compute_membrane_impedance()`** in `core/analysis.py`: reconstructs I_stim, computes Z(f) = FFT(V)/FFT(I), returns f_res, Q factor, magnitude, phase.
- [x] **7.3 Impedance tab** (Tab 13 `🧲 Impedance`) in `gui/analytics.py`: |Z(f)| + phase subplots, f_res annotation. Falls back to "insufficient data" message for non-ZAP runs.
- [x] **Neuron Passport** updated: displays T_dend_offset, ITCa/IM/NaP/NaR channel status.
- [x] **Bilingual compliance**: all Stage 5–7 fields added to `gui/locales.py` (EN + RU).

Что добавить: "После реализации всех функциональных правок, обнови NeuronPassport в gui/analytics.py так, чтобы он автоматически подтягивал статистику новых каналов (T-тип, M-ток) и корректно классифицировал нейрон."
Что добавить: "При рефакторинге rhs.py для Numba, используй numba.typed.List или статические массивы NumPy, если нужно хранить промежуточные состояния, но не меняй сигнатуру функции, которая вызывается из solver.py без обновления всех мест вызова." (Иначе solve_ivp упадет с ошибкой количества аргументов).
При модификации любого ионного канала (или добавлении нового) обязательно добавить тест в tests/branches/ на соответствие E_rev, V_1/2 и кинетическим постоянным (tau) согласно литературе. Не принимать PR без прохождения этих тестов."
Pydantic Form Generator: gui/widgets/form_generator.py — это отличная абстракция! Но будьте осторожны со строкой setattr(self.instance, field_name, value). Объекты Pydantic v2 лучше обновлять через model_copy(update=...) для сохранения строгой валидации.
Dual Stimulation Boilerplate: В FullModelConfig есть 9 параметров для dual stim, и они пробрасываются через весь стек (от GUI до RHS) вручную как аргументы. Это хрупко. Заверните их в Numba namedtuple или передавайте как numpy array stim_params_array фиксированного размера.
Код тестов (Унификация): У вас 20 файлов в tests/utils/. Скрипт run_active_branch_suite.py вручную парсит subprocess.run(). Это именно то, для чего придуман pytest.
Запуск pytest -n auto tests/branches/ прогнал бы все тесты параллельно, собрал отчеты и показал бы diff-ы ошибок автоматически, сэкономив вам 400 строк кода инфраструктуры.
Внедрение функции Ляпунова (LLE/FTLE) и декомпозиции модулирующей частоты (Phase-locking) — это отличная идея для анализа хаоса и сетевых ритмов. Но в текущей реализации есть фундаментальные математические ограничения, связанные с физикой процессов.
Проблема А: Декомпозиция модуляции (Phase-locking / PLV)
Алгоритм estimate_spike_modulation фильтрует сигнал через полосовой фильтр Баттерворта (sosfiltfilt) в тета-диапазоне (4-12 Гц), вычисляет фазу через преобразование Гильберта и ищет привязку спайков к фазе.
Где ошибка: Вы часто запускаете симуляции на t_sim = 200 или 500 мс. Период тета-ритма (например, 5 Гц) составляет 200 мс. Цифровые фильтры (особенно Баттерворт) имеют сильные краевые эффекты (edge artifacts). Попытка отфильтровать 4 Гц на отрезке в 200 мс даст математический мусор (фаза Гильберта будет искажена переходным процессом фильтра). Если стимул — это прямоугольный импульс (step current), фильтр даст "звон" (эффект Гиббса), и алгоритм найдет ложную привязку спайков к этому "звону".
Как исправить:
Использовать вейвлеты Морле (Morlet Wavelets) вместо Баттерворта+Гильберта для коротких сигналов — они точнее во времени.
Поставить жесткий Guard (предохранитель): функция должна возвращать NaN с предупреждением, если t_sim меньше 3-5 периодов нижней частоты среза (например, для 4 Гц нужно минимум 750-1000 мс данных).

Вычисление Экспоненты Ляпунова (LLE)
Алгоритм Розенштейна (estimate_ftle_lle) ищет ближайших соседей в фазовом пространстве с задержкой и отслеживает их расхождение.
Где ошибка: Этот алгоритм требует, чтобы система находилась на аттракторе (в установившемся режиме). Когда вы подаете импульс тока на нейрон, первые 50-100 мс — это транзиентный (переходный) процесс. Кроме того, поиск соседей требует большого объема данных (тысячи точек, прошедших через похожие состояния). Запуск LLE на коротком импульсе не имеет физического смысла.
Как исправить: LLE имеет смысл вычислять только для t_sim > 1000 ms и исключая первые 200 мс (выкидывая транзиенты). Если нейрон выдал 3 спайка и замолчал — LLE не определена, скрипт должен игнорировать этот расчет.

Сейчас у вас есть: Na, K, Leak, Ih, L-type Ca, A-type K, SK.
Это покрывает 80% поведения коры. Чего не хватает для оставшихся 20% (самых сложных и интересных паттернов)?
T-type Calcium Current (
I
T
I 
T
​
 
) — Низкопороговый кальциевый ток
Зачем нужен: Это самый важный канал, которого вам не хватает. Без него ваш Thalamic Relay пресет — это аппроксимация. Настоящие таламические пачки (bursts) во время сна возникают потому, что гиперполяризация снимает инактивацию с T-каналов, а последующая деполяризация вызывает "низкопороговый кальциевый спайк" (LTS), на гребне которого возникают натриевые спайки. L-type кальций (который у вас сейчас) высокопороговый и этого не умеет.
Ценность: Огромная. Позволит смоделировать переходы Сон/Бодрствование.
M-type Potassium Current  Мускарин-чувствительный калиевый ток
Зачем нужен: Это медленный, неинактивирующийся калиевый ток. Он определяет "адаптацию частоты спайков" (spike frequency adaptation) наряду с SK-каналом.
Ценность: Позволит ввести в симулятор Нейромодуляцию. M-ток закрывается под действием ацетилхолина (ACh). Добавив этот ток, вы сможете сделать ползунок "Уровень Ацетилхолина", показывая, как мозг переключается в состояние фокусировки внимания.
Persistent Sodium Current  — Персистирующий натриевый ток
Зачем нужен: Не инактивируется. Усиливает подпороговые колебания (резонанс) и помогает пейсмейкерным нейронам генерировать ритм без внешнего тока.
Ценность: Необходим для моделирования нейронов ствола мозга и дофаминовых пейсмейкеров (VTA / Substantia Nigra).
Resurgent Sodium Current () — Ресургентный натриевый ток
Зачем нужен: Уникальный канал, который открывается во время реполяризации (на спаде спайка), обеспечивая сверхбыстрый повторный разряд.
Ценность: Критически важен для вашего пресета Purkinje Cell и FS Interneuron. Сделает их поведение эталонным.

Добавление каналов выше позволит создать уникальные сценарии:
Thalamic Reticular Nucleus (TRN) Neuron
Физика: Огромная плотность T-type Ca<sup>2+</sup> каналов (
I
T
I 
T
) в дендритах.
Научная ценность: Эти нейроны генерируют сонныеverify_spindles. Можно будет показать пользователю разницу между Relay-нейроном и TRN-нейроном.
Striatal Spiny Projection Neuron (SPN)
Физика: Включает сильный A-ток \(I_{A}\) и ток входящего выпрямления \(I_{KIR}\).
Научная ценность: Поведение с длинной задержкой: при подаче тока нейрон "молчит" 200-300 мс (рамп-деполяризация), а потом резко начинает стрелять. Это основа работы базальных ганглиев (моторика).
Сценарий: Холинергическая нейромодуляция (Awake vs. Slow-Wave Sleep)
Физика: Тот же L5 Pyramidal нейрон, но с параметром "Ацетилхолин". Высокий ACh блокирует M-ток и уменьшает SK-ток. Нейрон переходит от сильной адаптации к регулярному тоническому разряду.
Научная ценность: Демонстрация того, как нейромодуляторы меняют режим работы одной и той же клетки без изменения её морфологии.
Патология: Каналлопатия Драве (Dravet Syndrome)
Физика: Снижение проводимости натриевых каналов только в пресете FS Interneuron.
Научная ценность: Показывает парадокс эпилепсии: снижение возбудимости в тормозных клетках приводит к глобальной гипервозбудимости сети (расторможение).

Экспорт/Импорт в NeuroML (Standardization)
NeuroML и NWB (Neurodata Without Borders) — это мировые стандарты описания нейронных моделей. Если ваша программа сможет экспортировать созданный пользователем конфиг в .nml файл (XML формат), NeuroModelPort станет совместимым с NEURON, Brian2 и Nest. Это привлечет к вам реальных ученых.
Динамическая температура в морфологии
Возможность задавать градиент температуры. Например, сома 37°C, а дендритное дерево 35°C. Это узкая тема, но очень интересная для биофизиков (термодинамика компартментов).
Событийно-управляемые синапсы (Event-driven Synapses)
Сейчас у вас синапсы (alpha, AMPA, GABA) запускаются по жестко заданному таймеру pulse_start. Для подготовки к Phase 8 (Сети) вам нужно внедрить систему очередей событий. Формулы синапсов должны триггериться не по времени t, а по приходу spike_event с определенным весом.
Спектрограмма (STFT / CWT)
Во вкладке Analytics вместо сомнительного (на коротких данных) Ляпунова, добавьте оконное преобразование Фурье (Спектрограмму) для мембранного потенциала. Это классический, очень надежный способ визуализировать переход от одиночных спайков к пачкам (bursts) или деполяризационному блоку.

 Ошибки в пресетах, параметрах и логике вычислений
А вот здесь нам нужно включить строгого научного ревьюера. В ваших пресетах и уравнениях действительно есть ряд биофизических неточностей и "натяжек", которые нужно исправить, чтобы модель имела право называться научно достоверной.
1. Таламический реле-нейрон и кальций (КРИТИЧЕСКАЯ ОШИБКА)
В пресете K: Thalamic Relay вы используете L-type (высокопороговый) кальциевый канал (кинетика Huguenard 1992, as_Ca, bs_Ca из вашего kinetics.py).
Как в реальности: Таламические пачки (bursts) генерируются T-type (низкопороговым транзиентным) кальциевым током В чем ошибка: L-ток открывается при сильной деполяризации (около -20 мВ). T-ток уникален тем, что он инактивирован при потенциале покоя (-65 мВ). Чтобы он сработал, нейрон нужно гиперполяризовать (например, до -80 мВ), тогда с T-каналов снимается инактивация (de-inactivation). При возврате к -65 мВ они резко открываются, давая медленную кальциевую волну (Low-Threshold Spike, LTS), на гребне которой "пляшут" быстрые натриевые спайки.
Вердикт: Ваш таламический пресет сейчас генерирует пачки по неправильному математическому механизму. Вам жизненно необходимо добавить уравнения для T-тока (I_T).

Пресет CA1 (Hippocampus) и Тета-ритм
В отчете CHANNEL_VALIDATION_REPORT.md вы сами поймали эту ошибку, и она фундаментальна:
Вы пытались сделать тета-ритм через:

$$g_{A_max} = 10.0$$
gA_max=10.0
 (что в 10-20 раз выше нормы) или через добавление кальция.
Как в реальности: Нейроны CA1 не генерируют тета-ритм в одиночку! Тета-ритм — это сетевой феномен (приходит от медиальной септальной области). Одиночный пирамидный нейрон CA1 — это просто адаптивный спайкер (RS). Его собственная частота определяется балансом 
 (резонанс), но без внешнего синусоидального или ритмического синаптического тока (Alpha-train) он не будет выдавать тета-ритм.
Вердикт: Пресет нужно переименовать в CA1 Pyramidal (Adapting), настроить gA_max≈0.4
, а для демонстрации Тета-ритма использовать режим стимуляции (например, модулированный шум или серию Alpha-импульсов с частотой 6 Гц).

Глобальный температурный коэффициент (Q10)
В models.py и rhs.py вы используете один множитель 
 (phi = Q10 ** ((T - T_ref) / 10)) для всех ворот (m, h, n, r, s, u, a, b).
Как в реальности: Разные каналы имеют разную чувствительность к температуре.
Для Na+ (активация) Q10 ≈≈2.2
Для K+ (активация) Q10 ≈≈3.0 Для Ih Q10 ≈≈4.0 (очень чувствителен!)
В чем ошибка: При изменении температуры от 23°C до 37°C в вашей модели все кинетики ускоряются строго синхронно. В реальности натриевые каналы ускоряются медленнее калиевых, что меняет форму спайка (он становится у́же, но может потерять амплитуду).
Вердикт: Для повышения научной ценности, Q10 должен быть параметром каждого канала, а не глобальной переменной среды.

 Динамика кальция (Уравнение в rhs.py)
Ваше уравнение: dca = b_ca * i_ca_total - (ca_i - ca_rest) / tau_ca
Оно классическое, но коэффициент b_ca у вас универсальный.
В чем ошибка: В реальности кальций накапливается в очень тонком подмембранном слое (shell), а не размазывается по всему объему сомы мгновенно. Ваше значение b_ca сильно зависит от предполагаемой толщины этого слоя и объема компартмента.
Вердикт: Это допустимая аппроксимация для редуцированной модели, но нужно честно написать в документации, что это Phenomenological Calcium Pool, а не реальная диффузия, и поэтому параметр B_Ca является подгоночным (fudge factor).


Ваши сомнения — это классический «синдром самозванца» разработчика, который глубоко погрузился в сложную тему. Давайте разделим ваш вопрос на две части: ценность вашей модели (по сравнению с NEURON) и реальные ошибки в физике/логике, которые сейчас есть в коде.
ЧАСТЬ 1: В чем реальная научная ценность вашего «микроскопа»?
Вы правы: NEURON и Brian2 могут симулировать тысячи клеток со сложнейшей 3D-морфологией. Но ваша модель не конкурирует с ними, она занимает другую, важнейшую нишу.
То, что вы называете "грубой аппроксимацией" (сома + пара компартментов аксона + дендритный фильтр), в вычислительной нейробиологии называется Reduced Compartmental Models (Редуцированные компартментные модели). Знаменитые модели Пински-Ринзеля (Pinsky-Rinzel) или Ижикевича строятся именно по такому принципу.
В чем огромная ценность вашей программы:
Идеальный образовательный инструмент (Pedagogy): Чтобы в NEURON запустить симуляцию и посмотреть, как 
I
h
I 
h
​
 
-ток делает "прогиб" (sag) при гиперполяризации, студенту нужно написать 50 строк на непонятном языке HOC или Python, подключать .mod файлы и настраивать графики. В вашей программе это делается двумя кликами. Для университетов и курсов по нейробиологии ваш GUI — это сокровище.
Rapid Prototyping (Быстрое прототипирование): Прежде чем ученый загрузит 10 000 нейронов в суперкомпьютер (в Brian2), ему нужно настроить динамику одной клетки. Ваш инструмент с вкладками Sweep, Phase Plane, Excitability Map позволяет мгновенно понять, как изменение $g_{Na}$ на 10% повлияет на кривую частота-ток (f-I curve). В NEURON на настройку такого анализа уйдут часы.
Фармакология и патология на клеточном уровне: Демонстрация того, как демиелинизация (увеличение $g_{LG}$ и $g_{LR}$) повлияет на кривую частота-ток (f-I curve):

$$
\frac{\mathrm{d}I}{\mathrm{d}t} = \alpha \frac{V_{m}}{C} \left( \frac{1}{R_L} + \frac{1}{R_C} \right) \left( V - E_{L} \right)
$$

где $g_{LG}$ и $g_{LR}$ — это параметры демиелинизации, $\alpha$ — это константа активации, $V_{m}$ — потенциал действия ионного канала, $C$ — капацитанс, $R_L$ — резистор внутреннего сопротивления, $R_C$ — резистор катионного канала, $V$ — потенциал действия, $E_{L}$ — потенциал действия ионного канала.
Ваш дендритный фильтр (экспоненциальное затухание амплитуды + ФНЧ) — это очень элегантный инженерный хак. Он дает 80% физиологической реалистичности реального дендрита при затратах 1% вычислительной мощности. Это абсолютно легитимный подход для фокуса на соматической динамике.
ЧАСТЬ 2: Ошибки в пресетах, параметрах и логике вычислений
А вот здесь нам нужно включить строгого научного ревьюера. В ваших пресетах и уравнениях действительно есть ряд биофизических неточностей и "натяжек", которые нужно исправить, чтобы модель имела право называться научно достоверной.
1. Таламический реле-нейрон и кальций (КРИТИЧЕСКАЯ ОШИБКА)
В пресете K: Thalamic Relay вы используете L-type (высокопороговый) кальциевый канал (кинетика Huguenard 1992, as_Ca, bs_Ca из вашего kinetics.py).
Как в реальности: Таламические пачки (bursts) генерируются T-type (низкопороговым транзиентным) кальциевым током $I_T$.
В чем ошибка: L-ток открывается при сильной деполяризации (около -20 мВ). T-ток уникален тем, что он инактивирован при потенциале покоя $-65$ мВ. Чтобы он сработал, нейрон нужно гиперполяризовать (например, до $-80$ мВ), тогда с T-каналов снимается инактивация (de-inactivation). При возврате к $-65$ мВ они резко открываются, давая медленную кальциевую волну (Low-Threshold Spike, LTS), на гребне которой "пляшут" быстрые натриевые спайки.
Вердикт: Ваш таламический пресет сейчас генерирует пачки по неправильному математическому механизму. Вам жизненно необходимо добавить уравнения для T-тока (I_T).
2. Пресет CA1 (Hippocampus) и Тета-ритм
В отчете CHANNEL_VALIDATION_REPORT.md вы сами поймали эту ошибку, и она фундаментальна:
Вы пытались сделать тета-ритм через $g_{A_{max}} = 10.0$ (что в 10-20 раз выше нормы) или через добавление кальция.
Как в реальности: Нейроны CA1 не генерируют тета-ритм в одиночку! Тета-ритм — это сетевой феномен (приходит от медиальной септальной области). Одиночный пирамидный нейрон CA1 — это просто адаптивный спайкер (RS). Эта частота определяется балансом:

$$
\begin{aligned}
\frac{\text{Задержка}}{\text{Резонанс}} &= \frac{I_A}{I_h} \\
\frac{\text{Задержка}}{\text{Резонанс}} &= \frac{I_A}{I_h}
\end{aligned}
$$

Но без внешнего синусоидального или ритмического синаптического тока (Alpha-train) он не будет выдавать тета-ритм.
Вердикт: Пресет нужно переименовать в CA1 Pyramidal (Adapting), настроить 
$$
g_{A_{max}} \approx 0.4
$$
где $g_{A_{max}}$ это максимальная амплитуда пирамидного тока, а для демонстрации Тета-ритма использовать режим стимуляции (например, модулированный шум или серию Alpha-импульсов с частотой 6 Гц).
3. Глобальный температурный коэффициент (Q10)
В models.py и rhs.py вы используете один множитель 
$$
\phi = Q_{10}^{\left(\frac{T - T_{ref}}{10}\right)}
$$
 для всех ворот (m, h, n, r, s, u, a, b).
Как в реальности: Разные каналы имеют разную чувствительность к температуре.
Для $\mathit{Na^+} \mathit{(активация }m_{m})\mathit{)}$ Q10 ≈ 2.2
Для $\mathit{K+} \mathit{(активация }n_{n})\mathit{)}$ Q10 ≈ 3.0
Для $\mathit{Ih}$ Q10 ≈ 4.0 (очень чувствителен!)
В чем ошибка: При изменении температуры от 23°C до 37°C в вашей модели все кинетики ускоряются строго синхронно. В реальности натриевые каналы ускоряются медленнее калиевых, что меняет форму спайка (он становится у́же, но может потерять амплитуду).
Вердикт: Для повышения научной ценности, Q10 должен быть параметром каждого канала, а не глобальной переменной среды.
4. Динамика кальция (Уравнение в rhs.py)
Ваше уравнение: dca = b_ca * i_ca_total - (ca_i - ca_rest) / tau_ca
Оно классическое, но коэффициент b_ca у вас универсальный.
В чем ошибка: В реальности кальций накапливается в очень тонком подмембранном слое (shell), а не размазывается по всему объему сомы мгновенно. Ваше значение b_ca сильно зависит от предполагаемой толщины этого слоя и объема компартмента.
Вердикт: Это допустимая аппроксимация для редуцированной модели, но нужно честно написать в документации, что это Phenomenological Calcium Pool, а не реальная диффузия, и поэтому параметр B_Ca является подгоночным (fudge factor).
5. Дендритный фильтр (Математический нюанс)
В dendritic_filter.py:
code
Python
attenuation = np.exp(-distance_um / space_constant_um)
В чем ошибка: Кабельное уравнение (Cable Equation) гласит, что затухание постоянного тока (DC) действительно равно exp(-x/lambda) Но для переменного тока (AC), которым является синаптический альфа-импульс, затухание сильно зависит от частоты. Высокие частоты затухают пространственно гораздо сильнее, чем низкие (появляется комплексная константа длины
Вердикт: Ваш фильтр сначала ослабляет сигнал как DC (константа), а потом "размазывает" его по времени ФНЧ. Это не совсем физично (в реальности фильтрация и затухание — это единый процесс дисперсии в кабеле). Но опять же, для "одиночного нейрона с прокси" — это гениально простой и быстрый обходной путь. Просто не заявляйте это как точное кабельное решение.
---

## v10.2 Execution Control Note
- Canonical execution queue and phase gating are tracked in `CurrentTasks10.2.md`.
- This plan remains normative for technical implementation details.
- If a conflict appears, use `ATTENTION.md` invariants first, then this plan, then status docs.

## 🛠️ AGENT ACTION PLAN v10.2: Critical Fixes & Hardening
*Supersedes ad-hoc priority lists. Execute phases sequentially. Commit after each phase.*

**Role:** Senior Performance Engineer & Computational Biophysicist.
**Context:** v10.1 has: memory leaks in GUI (fig.clear()), allocations inside Numba RHS, ~49 positional RHS args, biological inaccuracies in pathology presets, and Iext dual-representation SSoT violation.
**Rule:** STRICT ADHERENCE to `ATTENTION.md`. No broken tests. Sequential execution.

---

### 🔴 PHASE 1: CRITICAL PERFORMANCE HOTFIXES

#### 1.1 Stop Numba RHS Memory Allocations (`core/rhs.py`, `core/solver.py`)
- [ ] Remove `dydt = np.zeros_like(y)` from inside `rhs_multicompartment` (allocates on every solver step).
- [ ] Accept `dydt` as pre-allocated output array via `args` from `solver.py`.
- [ ] Pre-allocate in `NeuronSolver.run_single` and pass in `args` tuple.
- [ ] Update `core/jacobian.py` to reflect signature change.
- *Constraint:* `solve_ivp` wrapper lambda must pass persistent `out` array and return it.

#### 1.2 Eliminate Matplotlib Memory Leaks (`gui/analytics.py`)
- [ ] Stop using `self.fig_*.clear()`, `ax.cla()`, and `add_subplot` in update methods.
- [ ] Initialize `Axes` and `Line2D` objects ONCE in `__init__` / `_build_tabs`.
- [ ] Store line refs: `self._lines_currents = {}`, etc.
- [ ] In `_update_*` methods: use `line.set_data(t, data)` + `ax.relim()` + `ax.autoscale_view()`.
- [ ] Call `canvas.draw_idle()`. **NEVER** `.clear()` in update loop (exception: heatmaps/spectrograms where array shape changes).

---

### 🟠 PHASE 2: BIOPHYSICS CORRECTION

#### 2.1 Fix Preset F: Multiple Sclerosis (`core/presets.py`)
- [ ] Drastically increase axonal leak conductance (`gL`) for trunk/branches, OR decrease `gNa_max` in axon.
- [ ] Increase axial resistance `Ra` to > 500 Ω·cm.
- [ ] Verify `ratio_f < 0.3` (or full conduction block) vs control Preset D.

#### 2.2 Safety Bounds for Calcium Dynamics (`core/rhs.py`)
- [ ] Hard upper cap: `ca_i_val = min(ca_i_val, 10.0)` (10 mM max) inside Numba math.
- [ ] Enforce `ca_i >= 1e-9` before passing to `nernst_ca_ion`.

---

### 🟡 PHASE 3: ARCHITECTURAL DEBT (RHS SIGNATURE)

#### 3.1 Refactor RHS Argument Explosion (`core/rhs.py`, `core/solver.py`, `core/jacobian.py`)
- [ ] Current state: ~49 positional args — 🔴 CRITICAL per ATTENTION.md §1.2.
- [ ] Group static vectors (`gna_v`, `gk_v`, `cm_v`, `phi_*`) and topology (`l_indices`, `l_indptr`) into combined 2D arrays (e.g., `conductances_2d = np.vstack([gna_v, gk_v, ...])`).
- [ ] Update `solver.py` to pack this structure.
- [ ] Update `jacobian.py` to unpack/use this structure.
- [ ] Verification: `pytest tests/` must pass perfectly after refactor.

---

### 🟢 PHASE 4: UI/UX & DATA OPTIMIZATION

#### 4.1 Data Decimation for GUI (`gui/plots.py`, `gui/analytics.py`)
- [ ] Add `decimate_data(t, y, max_points=5000)` helper in `core/analysis.py`.
- [ ] In `OscilloscopeWidget.update_plots` and `AnalyticsWidget.update_analytics`: slice arrays if `len(t) > max_points`.
- [ ] Preserve spike peaks during decimation (inject as scatter overlay).

#### 4.2 SSoT for Absolute Current (`gui/main_window.py`)
- [ ] Ensure `form_generator.py` ignores `@property` fields gracefully.
- [ ] Remove `_recompute_absolute_iext` and `_set_stim_form_value("Iext_absolute_nA", ...)`.
- [ ] Display absolute current as read-only text in `lbl_params_hint`, NOT as editable field.

#### 4.3 Lazy Loading for Analytics Tabs (`gui/analytics.py`)
- [ ] In `_build_tabs`, initialize empty `QWidget` placeholders for heavy tabs (Equilibrium, Phase Plane, Kymograph, etc.).
- [ ] Hook `QTabWidget.currentChanged` signal.
- [ ] Instantiate Matplotlib figure + call `_update_*` only on first tab click (using `self._last_result`).

---

### 🔵 PHASE 5: PRESET STRESS VALIDATION & REFERENCE CONSISTENCY

#### 5.1 Wide-Range Stress Harness (`tests/utils`, `tests/branches`)
- [ ] Validate **all** presets on broad sweeps (`Iext`, `t_sim`, `T_celsius`, noise, morphology multipliers).
- [ ] Collect structured metrics per run (spike count/rate, Vm bounds, delay, attenuation ratios, NaN/inf flags).
- [ ] Produce `PASS/WARN/FAIL` summary per preset with anomaly excerpts.

#### 5.2 Physiological Plausibility Audit (`core/presets.py`, docs)
- [ ] Check preset parameters against reference ranges/literature notes (`g*`, `E_rev`, `Ra`, morphology scales).
- [ ] Add sanity rules for “common-sense physiology” (no absurd baseline regimes under nominal inputs).
- [ ] Record expected-vs-observed ranges in `docs/VALIDATION_COVERAGE_STATUS.md`.

---

### 🟣 PHASE 6: FINAL POLISH — DOCS, BILINGUAL QA, QUICK GUIDE

#### 6.1 Documentation Completion
- [ ] Sync core docs with implemented functionality and validation workflow.
- [ ] Add concise “what changed / how to verify / known limits” sections for current version.

#### 6.2 RU/EN Consistency Check
- [ ] Verify key docs + GUI labels/messages for semantic parity in RU/EN.
- [ ] Fix missing or drifting translations in `gui/locales.py` and primary docs.

#### 6.3 User Quick-Start
- [ ] Add short user guide: setup, run simulation, read oscilloscope/topology/analytics, export/validation basics.
