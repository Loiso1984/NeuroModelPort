"""
gui/locales.py — Internationalization (i18n) v10.0

Provides English and Russian translations for all UI strings
and scientific parameter descriptions (used as tooltips).
"""


class Translator:
    """Manages UI and tooltip translations with extended bilingual support."""

    TEXTS = {
        # ─────────────────────────────────────────────────────────────
        'EN': {
            # Window / tabs
            'app_title':    'Hodgkin-Huxley Neuron Simulator v10.0 — Research Platform',
            'tab_params':   'Parameters',
            'tab_plots':    'Oscilloscope',
            'tab_analytics':'Analytics',
            'tab_topology': 'Topology',
            'tab_help':     'Guide',

            # Buttons
            'btn_run':      '▶ RUN SIMULATION',
            'btn_stoch':    '🎲 STOCHASTIC',
            'btn_sweep':    '↔ SWEEP',
            'btn_sd':       '⏱ S-D Curve',
            'btn_excmap':   '🗺 Excit. Map',
            'btn_export':   '💾 Export CSV',

            # Status / labels
            'status_ready':     'Ready.',
            'status_computing': 'Computing… (Numba JIT active)',
            'preset_label':     'Preset:',
            'lbl_language':     'Language:',

            # ── Morphology descriptions ──────────────────────────────
            'single_comp':   'Single-compartment: neuron is treated as a point (0-D), no axon.',
            'd_soma':        'Soma diameter (cm). Larger soma = more capacitance = harder to excite.',
            'N_ais':         'AIS segments. The axon initial segment is the spike trigger zone.',
            'd_ais':         'AIS diameter (cm).',
            'gNa_ais_mult':  'gNa multiplier in AIS (typically 40–100×). Sets excitability threshold.',
            'gK_ais_mult':   'gK multiplier in AIS.',
            'gIh_ais_mult':  'gIh multiplier in AIS.',
            'gCa_ais_mult':  'gCa multiplier in AIS.',
            'gA_ais_mult':   'gA multiplier in AIS.',
            'N_trunk':       'Number of axon trunk segments.',
            'd_trunk':       'Axon trunk diameter (cm). Wider = faster conduction, less resistance.',
            'N_b1':          'Branch 1 segment count.',
            'd_b1':          'Branch 1 diameter (cm).',
            'N_b2':          'Branch 2 segment count.',
            'd_b2':          'Branch 2 diameter (cm).',
            'Ra':            'Axial resistance (Ω·cm). High Ra = signal attenuates along axon.',
            'dx':            'Compartment length (cm).',

            # ── Channel descriptions ─────────────────────────────────
            'Cm':        'Membrane capacitance (µF/cm²). Electrical inertia — higher Cm slows response.',
            'gNa_max':   'Max Na⁺ conductance. Drives the fast depolarising upstroke of the AP.',
            'gK_max':    'Max K⁺ conductance. Repolarises the membrane after the spike.',
            'gL':        'Leak conductance. Sets resting resistance; higher = shorter τ_m.',
            'ENa':       'Na⁺ reversal potential (mV). Spike peaks approach this value.',
            'EK':        'K⁺ reversal potential (mV). After-hyperpolarisation floor.',
            'EL':        'Leak reversal potential (mV). Approximates the resting membrane potential.',
            'enable_Ih': 'Enable Ih (HCN pacemaker). Causes rhythmic spontaneous firing.',
            'gIh_max':   'Max Ih conductance (mS/cm²).',
            'E_Ih':      'Ih reversal potential (mV).',
            'enable_ICa':'Enable L-type Ca²⁺ current (Huguenard 1992). Enables plateau potentials.',
            'gCa_max':   'Max ICa conductance (mS/cm²).',
            'E_Ca':      'Ca²⁺ reversal potential (mV, overridden by Nernst if dynamic Ca is on).',
            'enable_IA': 'Enable A-current (Connor-Stevens). Transient K⁺; delays first spike.',
            'gA_max':    'Max IA conductance (mS/cm²).',
            'E_A':       'IA reversal potential (mV).',
            'enable_SK': 'Enable SK channel (Ca-activated K⁺). Spike-frequency adaptation.',
            'gSK_max':   'Max SK conductance. Strength of calcium-driven adaptation.',

            # ── Calcium descriptions ─────────────────────────────────
            'dynamic_Ca':'Enable dynamic [Ca²⁺]ᵢ tracking and Nernst correction for E_Ca.',
            'Ca_ext':    'Extracellular [Ca²⁺] (mM). Used in Nernst equation.',
            'Ca_rest':   'Resting [Ca²⁺]ᵢ (mM). Baseline before stimulation.',
            'tau_Ca':    'Ca²⁺ pump time constant (ms). Long τ → prolonged SK activation.',
            'B_Ca':      'Current-to-concentration factor (mM / (µA/cm² · ms)).',

            # ── Environment descriptions ─────────────────────────────
            'T_celsius': 'Experiment temperature (°C). Scales all kinetics via Q10.',
            'T_ref':     'Reference temperature for channel kinetics (°C).',
            'Q10':       'Q10 coefficient. Rate doubles per 10°C if Q10 = 2.',

            # ── Simulation descriptions ──────────────────────────────
            't_sim':       'Total simulation time (ms).',
            'dt_eval':     'Output sample interval (ms). Does not affect solver accuracy.',
            'stim_type':   'Stimulus waveform: const / pulse / alpha (synapse) / OU noise.',
            'Iext':        'Stimulus amplitude (µA/cm²).',
            'pulse_start': 'Pulse onset time (ms).',
            'pulse_dur':   'Pulse duration (ms).',
            'alpha_tau':   'Alpha-synapse time constant (ms). Peak at t = τ after onset.',
            'stim_comp':   'Compartment index to inject current into (0 = soma).',
            'stoch_gating':'Langevin gate noise via Euler-Maruyama (Fox & Lu 1994). Use 🎲 button.',
            'noise_sigma': 'Additive white noise amplitude σ (µA/cm²). Added to dV/dt.',

            # ── Analysis descriptions ────────────────────────────────
            'run_mc':       'Run Monte-Carlo: parallel trials with ±5% gNa/gK variability.',
            'mc_trials':    'Number of MC trials.',
            'run_bifurcation': 'Compute bifurcation diagram after standard run.',
            'bif_param':    'Parameter for bifurcation sweep.',
            'bif_min':      'Bifurcation range start.',
            'bif_max':      'Bifurcation range end.',
            'bif_steps':    'Number of bifurcation steps.',
            'run_sweep':    'Enable parametric sweep (use ↔ SWEEP button).',
            'sweep_param':  'Parameter to vary in sweep. Options: Iext, gNa_max, gK_max, T_celsius, …',
            'sweep_min':    'Sweep start value.',
            'sweep_max':    'Sweep end value.',
            'sweep_steps':  'Number of sweep points.',
            'run_sd_curve': 'Compute Strength-Duration curve (use ⏱ S-D button).',
            'run_excmap':   'Compute 2-D excitability map (use 🗺 button).',
            'excmap_I_min': 'Excitability map: minimum current (µA/cm²).',
            'excmap_I_max': 'Excitability map: maximum current (µA/cm²).',
            'excmap_NI':    'Excitability map: number of current steps.',
            'excmap_D_min': 'Excitability map: minimum pulse duration (ms).',
            'excmap_D_max': 'Excitability map: maximum pulse duration (ms).',
            'excmap_ND':    'Excitability map: number of duration steps.',

            # ── Core Function Translations ───────────────────────────
            'func_gax': 'Axial conductance of uniform cylinder (mS).',
            'func_gax_pair': 'Conductance at junction of two compartments of different diameters (soma->axon).',
            'func_add_link': 'Adds connection between compartments i and j with conductance g.',
            'class_morphology_builder': 'Class for assembling neuron geometry, membrane areas, and Laplacian.',
            'func_nernst_ca': 'Dynamic Nernst potential for Calcium (z=2).',
            'func_get_stim_current': 'Mathematics of all stimulus types v10.',
            'func_am': 'Na activation (m). Determines the sharp upstroke of action potential.',
            'func_bm': 'Na deactivation (m).',
            'func_ah': 'Na inactivation removal (h).',
            'func_bh': 'Na inactivation (h). Slower than m, completes the spike.',
            'func_an': 'K activation (n). Slow repolarizing current.',
            'func_bn': 'K deactivation (n).',
            'class_channel_registry': 'OOP registry of channels for GUI interaction and state vector assembly.',
        },

        # ─────────────────────────────────────────────────────────────
        'RU': {
            # Окно / вкладки
            'app_title':    'Симулятор Нейрона Ходжкина-Хаксли v10.0 — Исследовательская Платформа',
            'tab_params':   'Параметры',
            'tab_plots':    'Осциллограф',
            'tab_analytics':'Аналитика',
            'tab_topology': 'Топология',
            'tab_help':     'Руководство',

            # Кнопки
            'btn_run':    '▶ ЗАПУСТИТЬ СИМУЛЯЦИЮ',
            'btn_stoch':  '🎲 СТОХАСТИКА',
            'btn_sweep':  '↔ ПЕРЕБОР',
            'btn_sd':     '⏱ Кривая C-Д',
            'btn_excmap': '🗺 Карта возб.',
            'btn_export': '💾 Экспорт CSV',

            # Статус / подписи
            'status_ready':     'Готов.',
            'status_computing': 'Вычисление… (Numba JIT активен)',
            'preset_label':     'Пресет:',
            'lbl_language':     'Язык:',

            # ── Морфология ──────────────────────────────────────────
            'single_comp':   'Одиночная сома: нейрон — точка, без аксона.',
            'd_soma':        'Диаметр сомы (см). Больше → больше ёмкость → труднее возбудить.',
            'N_ais':         'Сегменты AIS. AIS — «курок» нейрона, зона инициации спайка.',
            'd_ais':         'Диаметр AIS (см).',
            'gNa_ais_mult':  'Множитель gNa в AIS (обычно 40–100×). Задаёт порог возбудимости.',
            'gK_ais_mult':   'Множитель gK в AIS.',
            'gIh_ais_mult':  'Множитель gIh в AIS.',
            'gCa_ais_mult':  'Множитель gCa в AIS.',
            'gA_ais_mult':   'Множитель gA в AIS.',
            'N_trunk':       'Сегментов ствола аксона.',
            'd_trunk':       'Диаметр ствола (см). Толще → быстрее проведение.',
            'N_b1':          'Сегментов ветви 1.',
            'd_b1':          'Диаметр ветви 1 (см).',
            'N_b2':          'Сегментов ветви 2.',
            'd_b2':          'Диаметр ветви 2 (см).',
            'Ra':            'Аксиальное сопротивление (Ом·см). Высокое Ra гасит сигнал.',
            'dx':            'Длина одного сегмента (см).',

            # ── Каналы ───────────────────────────────────────────────
            'Cm':        'Ёмкость мембраны (мкФ/см²). Инерция: большая Cm замедляет реакцию.',
            'gNa_max':   'Макс. проводимость Na⁺. Отвечает за быстрый фронт спайка.',
            'gK_max':    'Макс. проводимость K⁺. Реполяризует мембрану после спайка.',
            'gL':        'Проводимость утечки. Задаёт входное сопротивление и τ_m.',
            'ENa':       'Потенциал реверсии Na⁺ (мВ). Пик спайка стремится к этому значению.',
            'EK':        'Потенциал реверсии K⁺ (мВ). «Дно» реполяризации.',
            'EL':        'Потенциал реверсии утечки (мВ). Потенциал покоя.',
            'enable_Ih': 'Включить Ih (пейсмейкерный ток). Вызывает спонтанный ритм.',
            'gIh_max':   'Макс. проводимость Ih (мСм/см²).',
            'E_Ih':      'Потенциал реверсии Ih (мВ).',
            'enable_ICa':'Включить L-тип Ca²⁺ (Huguenard 1992). Позволяет плато-потенциалы.',
            'gCa_max':   'Макс. проводимость ICa (мСм/см²).',
            'E_Ca':      'Потенциал реверсии Ca²⁺ (мВ). При включённом Ca перезаписывается Нернстом.',
            'enable_IA': 'Включить A-ток (Connor-Stevens). Задерживает первый спайк.',
            'gA_max':    'Макс. проводимость IA (мСм/см²).',
            'E_A':       'Потенциал реверсии IA (мВ).',
            'enable_SK': 'Включить SK-канал (Ca²⁺-зависимый K⁺). Спайковая адаптация.',
            'gSK_max':   'Макс. проводимость SK. Сила кальций-зависимого торможения.',

            # ── Кальций ──────────────────────────────────────────────
            'dynamic_Ca':'Динамический [Ca²⁺]ᵢ с пересчётом Нернста для E_Ca.',
            'Ca_ext':    'Внеклеточный [Ca²⁺] (мМ). Используется в уравнении Нернста.',
            'Ca_rest':   'Покоящийся [Ca²⁺]ᵢ (мМ). Базовый уровень до стимула.',
            'tau_Ca':    'Время откачки Ca²⁺ (мс). Длинный τ → долгий SK-ток → адаптация.',
            'B_Ca':      'Конверсия тока в концентрацию (мМ / (мкА/см² · мс)).',

            # ── Среда ────────────────────────────────────────────────
            'T_celsius': 'Температура эксперимента (°C). Масштабирует кинетику через Q10.',
            'T_ref':     'Референсная температура каналов (°C).',
            'Q10':       'Коэффициент Q10. При Q10=3 скорость растёт в 3 раза за 10°C.',

            # ── Симуляция ────────────────────────────────────────────
            't_sim':       'Длительность симуляции (мс).',
            'dt_eval':     'Шаг вывода (мс). Не влияет на точность интегратора.',
            'stim_type':   'Тип стимула: const / pulse / alpha (ВПСП) / шум OU.',
            'Iext':        'Амплитуда стимула (мкА/см²).',
            'pulse_start': 'Начало импульса (мс).',
            'pulse_dur':   'Длительность импульса (мс).',
            'alpha_tau':   'Постоянная времени alpha-синапса (мс). Пик при t = τ.',
            'stim_comp':   'Компартмент стимуляции (0 = сома).',
            'stoch_gating':'Шум Ланжевена в гейтах (Эйлер-Маруяма). Кнопка 🎲.',
            'noise_sigma': 'Амплитуда белого шума на мембрану σ (мкА/см²).',

            # ── Анализ ───────────────────────────────────────────────
            'run_mc':       'Монте-Карло: параллельные запуски с ±5% разбросом gNa/gK.',
            'mc_trials':    'Число МК-попыток.',
            'run_bifurcation': 'Бифуркационный анализ после основного запуска.',
            'bif_param':    'Параметр для бифуркации.',
            'bif_min':      'Начало диапазона бифуркации.',
            'bif_max':      'Конец диапазона бифуркации.',
            'bif_steps':    'Количество точек бифуркации.',
            'run_sweep':    'Включить параметрический перебор (кнопка ↔ ПЕРЕБОР).',
            'sweep_param':  'Перебираемый параметр: Iext, gNa_max, gK_max, T_celsius…',
            'sweep_min':    'Начальное значение перебора.',
            'sweep_max':    'Конечное значение перебора.',
            'sweep_steps':  'Число точек перебора.',
            'run_sd_curve': 'Кривая сила-длительность (кнопка ⏱ C-Д).',
            'run_excmap':   'Карта возбудимости 2D (кнопка 🗺).',
            'excmap_I_min': 'Мин. ток для карты (мкА/см²).',
            'excmap_I_max': 'Макс. ток для карты (мкА/см²).',
            'excmap_NI':    'Шагов по току.',
            'excmap_D_min': 'Мин. длительность импульса (мс).',
            'excmap_D_max': 'Макс. длительность импульса (мс).',
            'excmap_ND':    'Шагов по длительности.',

            # ── Перевод основных функций ───────────────────────────
            'func_gax': 'Аксиальная проводимость однородного цилиндра (мСм).',
            'func_gax_pair': 'Проводимость на стыке двух компартментов разного диаметра (сома->аксон).',
            'func_add_link': 'Добавляет связь между компартментами i и j с проводимостью g.',
            'class_morphology_builder': 'Класс для сборки геометрии нейрона, площадей мембраны и Лапласиана.',
            'func_nernst_ca': 'Динамический потенциал Нернста для Кальция (z=2).',
            'func_get_stim_current': 'Математика всех типов стимулов v10.',
            'func_am': 'Активация Na (m). Определяет резкий фронт потенциала действия.',
            'func_bm': 'Деактивация Na (m).',
            'func_ah': 'Снятие инактивации Na (h).',
            'func_bh': 'Инактивация Na (h). Медленнее m, завершает спайк.',
            'func_an': 'Активация K (n). Медленный ток реполяризации.',
            'func_bn': 'Деактивация K (n).',
            'class_channel_registry': 'ООП-реестр каналов для взаимодействия с GUI и сборки вектора состояний.',
        },
    }

    def __init__(self, lang: str = 'EN'):
        self.lang = lang if lang in self.TEXTS else 'EN'

    def set_language(self, lang: str):
        if lang in self.TEXTS:
            self.lang = lang

    def tr(self, key: str) -> str:
        """Return UI string translation."""
        return self.TEXTS.get(self.lang, {}).get(key, key)

    def desc(self, key: str) -> str:
        """Return parameter description (for tooltips)."""
        return self.TEXTS.get(self.lang, {}).get(key, '')

    def func_desc(self, key: str) -> str:
        """Return function description for documentation tooltips."""
        return self.TEXTS.get(self.lang, {}).get(key, '')

    def get_preset_translation(self, preset_name: str) -> str:
        """Get translated preset name if available, otherwise return original."""
        preset_translations = {
            'EN': {
                "A: Squid Giant Axon (HH 1952)": "A: Squid Giant Axon (HH 1952)",
                "B: Pyramidal L5 (Mainen 1996)": "B: Pyramidal L5 (Mainen 1996)",
                "C: FS Interneuron (Wang-Buzsaki)": "C: FS Interneuron (Wang-Buzsaki)",
                "D: alpha-Motoneuron (Powers 2001)": "D: alpha-Motoneuron (Powers 2001)",
                "E: Cerebellar Purkinje (De Schutter)": "E: Cerebellar Purkinje (De Schutter)",
                "F: Multiple Sclerosis (Demyelination)": "F: Multiple Sclerosis (Demyelination)",
                "G: Local Anesthesia (gNa Block)": "G: Local Anesthesia (gNa Block)",
                "H: Severe Hyperkalemia (High EK)": "H: Severe Hyperkalemia (High EK)",
                "I: In Vitro Slice (Mammalian 23°C)": "I: In Vitro Slice (Mammalian 23°C)",
                "J: C-Fiber (Pain / Unmyelinated)": "J: C-Fiber (Pain / Unmyelinated)",
                "K: Thalamic Relay (Ih + IT + Burst)": "K: Thalamic Relay (Ih + IT + Burst)",
                "L: Hippocampal CA1 (Theta rhythm)": "L: Hippocampal CA1 (Theta rhythm)",
                "M: Epilepsy (v10 SCN1A mutation)": "M: Epilepsy (v10 SCN1A mutation)",
                "N: Alzheimer's (v10 Calcium Toxicity)": "N: Alzheimer's (v10 Calcium Toxicity)",
                "O: Hypoxia (v10 ATP-pump failure)": "O: Hypoxia (v10 ATP-pump failure)",
            },
            'RU': {
                "A: Squid Giant Axon (HH 1952)": "A: Гигантский аксон кальмара (ХХ 1952)",
                "B: Pyramidal L5 (Mainen 1996)": "B: Пирамидный L5 (Mainen 1996)",
                "C: FS Interneuron (Wang-Buzsaki)": "C: Быстрый интернейрон (Wang-Buzsaki)",
                "D: alpha-Motoneuron (Powers 2001)": "D: Альфа-мотонейрон (Powers 2001)",
                "E: Cerebellar Purkinje (De Schutter)": "E: Клетка Пуркинье мозжечка (De Schutter)",
                "F: Multiple Sclerosis (Demyelination)": "F: Рассеянный склероз (демиелинизация)",
                "G: Local Anesthesia (gNa Block)": "G: Местная анестезия (блокада gNa)",
                "H: Severe Hyperkalemia (High EK)": "H: Тяжелая гиперкалиемия (высокий EK)",
                "I: In Vitro Slice (Mammalian 23°C)": "I: Срез in vitro (млекопитающие 23°C)",
                "J: C-Fiber (Pain / Unmyelinated)": "J: C-волокно (боль / безмиелиновое)",
                "K: Thalamic Relay (Ih + IT + Burst)": "K: Таламическое реле (Ih + IT + пачки)",
                "L: Hippocampal CA1 (Theta rhythm)": "L: Гиппокамп CA1 (тета-ритм)",
                "M: Epilepsy (v10 SCN1A mutation)": "M: Эпилепсия (мутация SCN1A v10)",
                "N: Alzheimer's (v10 Calcium Toxicity)": "N: Болезнь Альцгеймера (токсичность кальция v10)",
                "O: Hypoxia (v10 ATP-pump failure)": "O: Гипоксия (отказ АТФ-насоса v10)",
            }
        }
        return preset_translations.get(self.lang, {}).get(preset_name, preset_name)


# Global translator instance — default English
T = Translator('EN')
