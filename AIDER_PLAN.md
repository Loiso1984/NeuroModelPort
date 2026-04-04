# AIDER INSTRUCTION PLAN: NeuroModelPort v10.1 Refactoring & Enhancement
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

### STAGE 3: Math & Analytics Fixes
*Goal: Fix false positives in spike detection and protect advanced analytical algorithms from edge cases.*
- [ ] **3.1 State Machine Spike Detector (`core/analysis.py`):** Rewrite the `detect_spikes` function. The current algorithm (using `scipy.signal.find_peaks` or simple threshold crossing) creates false positives during depolarization blocks (plateaus). Implement a Numba-jitted Finite State Machine (States: `RESTING`, `DEPOLARIZING`, `REFRACTORY`) that counts a spike **only** if the voltage crosses the `threshold`, reaches a peak, and **strictly returns** below `baseline_threshold`.
- [ ] **3.2 Phase-Locking Protection (`core/analysis.py`):** In `estimate_spike_modulation`, add a guard clause: if `t_sim` is less than 3-5 periods of the lowest cutoff frequency (e.g., < 1000 ms for 4 Hz), return `NaN` to prevent meaningless results due to Butterworth filter edge artifacts. (Consider replacing Butterworth with Morlet wavelets if feasible without heavy dependencies).
- [ ] **3.3 Lyapunov Exponent (LLE) Protection (`core/analysis.py`):** In `estimate_ftle_lle`, add a condition to only compute LLE if `t_sim > 1000 ms`, and explicitly discard the first 200 ms of data (transients) before searching for nearest neighbors on the attractor.

### STAGE 4: UI/UX & Memory Leak Resolution
*Goal: Make the GUI responsive and prevent memory leaks during long analysis runs.*
- [ ] **4.1 Matplotlib Memory Leaks (`gui/analytics.py`):** Fix the updating logic in `_update_currents`, `_update_equil`, `_update_phase`, etc. Remove `self.fig_*.clear()` and `add_subplot()` from the update methods. Initialize the axes and `Line2D` objects once in `__init__`. Use `line.set_data()` and `canvas.draw_idle()` to update plots.
- [ ] **4.2 Main Thread Offloading (`gui/main_window.py`):** Heavy UI-blocking operations like `run_sweep` and `run_excitability_map` currently freeze the GUI due to the GIL, even if run in a basic thread. Refactor these to use `ProcessPoolExecutor` (similar to Monte-Carlo) and add a progress bar with a `Cancel` button.
- [ ] **4.3 MainWindow Refactoring:** `MainWindow` is a 700+ line God Object. Refactor it by extracting responsibilities into separate classes: `SimulationController` (handles threading/execution) and `ConfigManager` (handles presets and dual stim state). Keep `MainWindow` strictly for UI layout and event binding.
- [ ] **4.4 Interactive Topology (`gui/topology.py`):** Enhance the `TopologyWidget` using `pyqtgraph` features. Add pan/zoom capabilities and make compartments clickable to show a "Neuron Passport" info panel summarizing that compartment's properties.
//ADDITIONAL GUI REFACTORING RECOMENATIONS. They are not mandatory but may be efficient for doing the GUI - User Friendly
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
- [ ] **5.1 T-type Calcium Current ($I_T$):** Implement the low-threshold transient calcium current (kinetics from *Destexhe 1998* or *Huguenard 1992*) in `core/kinetics.py` and integrate it into `core/rhs.py`.
- [ ] **5.2 M-type Potassium Current ($I_M$):** Implement the slow, non-inactivating, muscarinic-sensitive potassium current responsible for spike-frequency adaptation.
- [ ] **5.3 Persistent ($I_{NaP}$) & Resurgent ($I_{NaR}$) Sodium:** Add these channels based on established literature kinetics. Make them optional components in the RHS.
- [ ] **5.4 Fix "Thalamic Relay" Preset (`core/presets.py`):** Replace the high-threshold L-type calcium channel with the newly implemented **T-type ($I_T$)**. Ensure the neuron generates Low-Threshold Spikes (LTS) and bursts *only* when recovering from hyperpolarization.
- [ ] **5.5 Fix "CA1 Pyramidal" Preset (`core/presets.py`):** Remove intrinsic Theta rhythm generation (reduce $g_A$ from the unphysiological 10.0 to ~0.4). Rename the preset to `CA1 Pyramidal (Adapting)`. Note in comments that theta rhythm should be driven by external synaptic input, not intrinsic channels.
- [ ] **5.6 Add 4 New Presets (`core/presets.py`):**
    1. `Thalamic Reticular Nucleus (TRN)`: High density of $I_T$ in dendrites (sleep spindles).
    2. `Striatal Spiny Projection Neuron (SPN)`: Demonstrates long latency to first spike (requires strong $I_A$ and inward rectifiers).
    3. `Cholinergic Neuromodulation (Awake vs Sleep)`: Based on L5 Pyramidal, but blocking $I_M$ to show the shift from adaptation to tonic firing.
    4. `Pathology: Dravet Syndrome`: FS Interneuron with reduced $g_{Na}$ demonstrating paradoxical network disinhibition (at the single-cell level).

### STAGE 6: Advanced Architectural Features
*Goal: Add cutting-edge analysis and prepare the architecture for Phase 8 (Network Modeling).*
- [ ] **6.1 Spectrogram Analysis (`gui/analytics.py`):** Add a new tab/plot showing the Short-Time Fourier Transform (STFT) or Continuous Wavelet Transform (CWT) of the membrane potential to visualize the transition from single spikes to bursts.
- [ ] **6.2 Dynamic Temperature Gradients:** Modify the morphology/environment setup to allow different temperatures for the soma vs. dendrites.
- [ ] **6.3 Event-Driven Synapses:** Refactor synaptic stimulation (`alpha`, `AMPA`, `GABA`). Instead of triggering at a fixed `pulse_start` time, implement an `event queue`. Synaptic conductances should update based on incoming `spike_event` timestamps (preparation for network connectivity).
- [ ] **6.4 NeuroML Export (Bonus):** Implement a utility to export the `FullModelConfig` into the standardized NeuroML (XML) format for interoperability with NEURON and Brian2.