"""
gui/locales.py - Internationalization (i18n) v11.3

Provides English and Russian translations for all UI strings
and scientific parameter descriptions (used as tooltips).
"""

from gui.text_sanitize import repair_text


class Translator:
    """Manages UI and tooltip translations with extended bilingual support."""

    TEXTS = {
        # ─────────────────────────────────────────────────────────────
        'EN': {
            # Window / tabs
            'app_title':    'Hodgkin-Huxley Neuron Simulator v11.3 - Research Platform',
            'tab_params':   'Parameters',
            'tab_plots':    'Oscilloscope',
            'tab_analytics':'Analytics',
            'tab_topology': 'Topology',
            'tab_help':     'Guide',

            # Buttons
            'btn_run':      'RUN SIMULATION',
            'btn_stoch':    'STOCHASTIC',
            'btn_sweep':    'SWEEP',
            'btn_sd':       'S-D Curve',
            'btn_excmap':   'Excit. Map',
            'btn_export':   'Export CSV',
            'btn_load':     'Load',

            # Status / labels
            'status_ready':     'Ready.',
            'status_computing': 'Computing... (Numba JIT active)',
            'preset_label':     'Preset:',
            'lbl_language':     'Language:',
            'dual_presets':     'Dual Stimulation Presets',
            'secondary_group':  'Secondary Stimulus (Inhibitory)',
            'lbl_location':     'Location:',
            'lbl_type':         'Type:',
            'lbl_current':      'Current (uA/cm2):',
            'lbl_start':        'Start (ms):',
            'lbl_duration':     'Duration (ms):',
            'lbl_enable_secondary': 'Enable secondary stimulus',

            # ── Morphology descriptions ──────────────────────────────
            'single_comp':   'Single-compartment: neuron is treated as a point (0-D), no axon.',
            'd_soma':        'Soma diameter (cm). Larger soma = more capacitance = harder to excite.',
            'N_ais':         'AIS segments. The axon initial segment is the spike trigger zone.',
            'd_ais':         'AIS diameter (cm).',
            'l_ais':         'AIS segment length (cm).',
            'gNa_ais_mult':  'gNa multiplier in AIS (typically 40–100x). Sets excitability threshold.',
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
            'Cm':        'Membrane capacitance (µF/cmÂ˛). Electrical inertia — higher Cm slows response.',
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
            'tau_Ca':    'CaÂ˛âş pump time constant (ms). Long τ -> prolonged SK activation.',
            'B_Ca':      'Current-to-concentration factor (mM / (µA/cmÂ˛ · ms)).',

            # ── Metabolism descriptions ───────────────────────────────
            'enable_dynamic_atp': 'Enable dynamic ATP pool metabolism. ATP depletion opens K_ATP channels, hyperpolarizing the cell.',
            'g_katp_max': 'Maximum ATP-sensitive K⁺ conductance (mS/cm²). Opens when ATP < 0.5 mM.',
            'katp_kd_atp_mM': 'ATP concentration for half-activation of K_ATP (mM).',
            'atp_max_mM': 'Baseline intracellular ATP concentration (mM). Healthy neurons: 2-5 mM.',
            'atp_synthesis_rate': 'ATP synthesis rate via oxidative phosphorylation (mM/s). Healthy ~0.5; hypoxia 0.005–0.05.',

            # ── Environment descriptions ─────────────────────────────
            'T_celsius':     'Experiment temperature (°C). Scales all kinetics via Q10.',
            'T_ref':         'Reference temperature for channel kinetics (°C).',
            'Q10':           'Q10 coefficient. Rate doubles per 10°C if Q10 = 2.',
            'T_dend_offset': 'Dendrite–soma temperature difference (°C). Positive = dendrites warmer. Creates linear axial gradient.',

            # ── Simulation descriptions ──────────────────────────────
            't_sim':       'Total simulation time (ms).',
            'dt_eval':     'Output sample interval (ms). Does not affect solver accuracy.',
            'stim_type':   'Stimulus waveform: const / pulse / alpha / OU noise / synaptic receptors / zap chirp.',
            'Iext':        'Stimulus amplitude (µA/cmÂ˛).',
            'pulse_start': 'Pulse onset time (ms).',
            'pulse_dur':   'Pulse duration (ms).',
            'alpha_tau':   'Alpha-synapse time constant (ms). Peak at t = τ after onset.',
            'zap_f0_hz':   'ZAP start frequency (Hz) for chirp stimulus.',
            'zap_f1_hz':   'ZAP end frequency (Hz) for chirp stimulus.',
            'stim_comp':   'Compartment index to inject current into (0 = soma).',
            'event_times': 'Synaptic event queue (ms). Comma-separated timestamps; e.g. "10,25,40". Overrides pulse_start for AMPA/NMDA/GABA types.',
            'synaptic_train_type': 'Spike Train Generator: none / regular (fixed ISI) / poisson (random).',
            'synaptic_train_freq_hz': 'Frequency of the generated spike train (Hz).',
            'synaptic_train_duration_ms': 'Duration of the spike train (ms).',
            'stoch_gating':'Langevin gate noise via Native Hines solver. Use the STOCHASTIC button.',
            'noise_sigma': 'Additive white noise amplitude Ď (µA/cmÂ˛). Added to dV/dt.',

            # ── Analysis descriptions ────────────────────────────────
            'run_mc':       'Run Monte-Carlo: parallel trials with ±5% gNa/gK variability.',
            'mc_trials':    'Number of MC trials.',
            'run_bifurcation': 'Compute bifurcation diagram after standard run.',
            'bif_param':    'Parameter for bifurcation sweep.',
            'bif_min':      'Bifurcation range start.',
            'bif_max':      'Bifurcation range end.',
            'bif_steps':    'Number of bifurcation steps.',
            'run_sweep':    'Enable parametric sweep (use ↔ SWEEP button).',
            'sweep_param':  'Parameter to vary in sweep. Options: Iext, gNa_max, gK_max, T_celsius, ...',
            'sweep_min':    'Sweep start value.',
            'sweep_max':    'Sweep end value.',
            'sweep_steps':  'Number of sweep points.',
            'run_sd_curve': 'Compute Strength-Duration curve (use ⏱ S-D button).',
            'run_excmap':   'Compute 2-D excitability map (use 🗺 button).',
            'excmap_I_min': 'Excitability map: minimum current (µA/cmÂ˛).',
            'excmap_I_max': 'Excitability map: maximum current (µA/cmÂ˛).',
            'excmap_NI':    'Excitability map: number of current steps.',
            'excmap_D_min': 'Excitability map: minimum pulse duration (ms).',
            'excmap_D_max': 'Excitability map: maximum pulse duration (ms).',
            'excmap_ND':    'Excitability map: number of duration steps.',

            # ── Preset Mode Labels ─────────────────────────────────
            'k_mode_sleep': 'Sleep (Bursts)',
            'k_mode_awake': 'Awake (Relay)',
            'l5_mode_normal': 'Standard',
            'l5_mode_ach': 'Attention (ACh ON)',

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
            'app_title':    'Симулятор нейрона Ходжкина-Хаксли v11.3 - Исследовательская платформа',
            'tab_params':   'Параметры',
            'tab_plots':    'Осциллоскоп',
            'tab_analytics':'Аналитика',
            'tab_topology': 'Топология',
            'tab_help':     'Руководство',

            # Кнопки
            'btn_run':    'ЗАПУСТИТЬ СИМУЛЯЦИЮ',
            'btn_stoch':  'СТОХАСТИКА',
            'btn_sweep':  'ПЕРЕБОР',
            'btn_sd':     'Кривая C-D',
            'btn_excmap': 'Карта возбуд.',
            'btn_export': 'Экспорт CSV',
            'btn_load':   'Загрузить',

            # Статус / подписи
            'status_ready':     'Готово.',
            'status_computing': 'Вычисление... (Numba JIT активен)',
            'preset_label':     'Пресет:',
            'lbl_language':     'Язык:',
            'dual_presets':     'Пресеты двойной стимуляции',
            'secondary_group':  'Вторичный стимул (ингибиторный)',
            'lbl_location':     'Локация:',
            'lbl_type':         'Тип:',
            'lbl_current':      'Ток (uA/cm2):',
            'lbl_start':        'Старт (мс):',
            'lbl_duration':     'Длительность (мс):',
            'lbl_enable_secondary': 'Включить вторичный стимул',

            # ── Морфология ──────────────────────────────────────────
            'single_comp':   'ĐžĐ´Đ¸Đ˝ĐľŃ‡Đ˝Đ°ŃŹ ŃĐľĐĽĐ°: Đ˝ĐµĐąŃ€ĐľĐ˝ — Ń‚ĐľŃ‡ĐşĐ°, Đ±ĐµĐ· Đ°ĐşŃĐľĐ˝Đ°.',
            'd_soma':        'Диаметр сомы (см). Больше -> больше ёмкость -> труднее возбудить.',
            'N_ais':         'ĐˇĐµĐłĐĽĐµĐ˝Ń‚Ń‹ AIS. AIS — Â«ĐşŃŃ€ĐľĐşÂ» Đ˝ĐµĐąŃ€ĐľĐ˝Đ°, Đ·ĐľĐ˝Đ° Đ¸Đ˝Đ¸Ń†Đ¸Đ°Ń†Đ¸Đ¸ ŃĐżĐ°ĐąĐşĐ°.',
            'd_ais':         'Диаметр AIS (см).',
            'l_ais':         'Длина сегмента AIS (см).',
            'gNa_ais_mult':  'ĐśĐ˝ĐľĐ¶Đ¸Ń‚ĐµĐ»ŃŚ gNa Đ˛ AIS (ĐľĐ±Ń‹Ń‡Đ˝Đľ 40–100x). Đ—Đ°Đ´Đ°Ń‘Ń‚ ĐżĐľŃ€ĐľĐł Đ˛ĐľĐ·Đ±ŃĐ´Đ¸ĐĽĐľŃŃ‚Đ¸.',
            'gK_ais_mult':   'Множитель gK в AIS.',
            'gIh_ais_mult':  'Множитель gIh в AIS.',
            'gCa_ais_mult':  'Множитель gCa в AIS.',
            'gA_ais_mult':   'Множитель gA в AIS.',
            'N_trunk':       'Сегментов ствола аксона.',
            'd_trunk':       'Диаметр ствола (см). Толще -> быстрее проведение.',
            'N_b1':          'Сегментов ветви 1.',
            'd_b1':          'Диаметр ветви 1 (см).',
            'N_b2':          'Сегментов ветви 2.',
            'd_b2':          'Диаметр ветви 2 (см).',
            'Ra':            'ĐĐşŃĐ¸Đ°Đ»ŃŚĐ˝ĐľĐµ ŃĐľĐżŃ€ĐľŃ‚Đ¸Đ˛Đ»ĐµĐ˝Đ¸Đµ (ĐžĐĽ·ŃĐĽ). Đ’Ń‹ŃĐľĐşĐľĐµ Ra ĐłĐ°ŃĐ¸Ń‚ ŃĐ¸ĐłĐ˝Đ°Đ».',
            'dx':            'Длина одного сегмента (см).',

            # ── Каналы ───────────────────────────────────────────────
            'Cm':        'Ёмкость мембраны (мкФ/см²). Инерция: большая Cm замедляет реакцию.',
            'gNa_max':   'Макс. проводимость Na⁺. Отвечает за быстрый фронт спайка.',
            'gK_max':    'Макс. проводимость K⁺. Реполяризует мембрану после спайка.',
            'gL':        'ĐźŃ€ĐľĐ˛ĐľĐ´Đ¸ĐĽĐľŃŃ‚ŃŚ ŃŃ‚ĐµŃ‡ĐşĐ¸. Đ—Đ°Đ´Đ°Ń‘Ń‚ Đ˛Ń…ĐľĐ´Đ˝ĐľĐµ ŃĐľĐżŃ€ĐľŃ‚Đ¸Đ˛Đ»ĐµĐ˝Đ¸Đµ Đ¸ τ_m.',
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
            'tau_Ca':    'Đ’Ń€ĐµĐĽŃŹ ĐľŃ‚ĐşĐ°Ń‡ĐşĐ¸ CaÂ˛âş (ĐĽŃ). Đ”Đ»Đ¸Đ˝Đ˝Ń‹Đą τ -> Đ´ĐľĐ»ĐłĐ¸Đą SK-Ń‚ĐľĐş -> Đ°Đ´Đ°ĐżŃ‚Đ°Ń†Đ¸ŃŹ.',
            'B_Ca':      'ĐšĐľĐ˝Đ˛ĐµŃ€ŃĐ¸ŃŹ Ń‚ĐľĐşĐ° Đ˛ ĐşĐľĐ˝Ń†ĐµĐ˝Ń‚Ń€Đ°Ń†Đ¸ŃŽ (ĐĽĐś / (ĐĽĐşĐ/ŃĐĽÂ˛ · ĐĽŃ)).',

            # ── Метаболизм ───────────────────────────────────────────
            'enable_dynamic_atp': 'Включить динамический пул АТФ. Истощение АТФ открывает K_ATP каналы, гиперполяризуя клетку.',
            'g_katp_max': 'Макс. проводимость ATP-чувствительных K⁺ каналов (мСм/см²). Открываются при АТФ < 0.5 мМ.',
            'katp_kd_atp_mM': 'Концентрация АТФ для полуактивации K_ATP (мМ).',
            'atp_max_mM': 'Базовая внутриклеточная концентрация АТФ (мМ). Здоровые нейроны: 2-5 мМ.',
            'atp_synthesis_rate': 'Скорость синтеза АТФ через окислительное фосфорилирование (мМ/с). Норма ~0.5; гипоксия 0.005–0.05.',

            # ── Среда ────────────────────────────────────────────────
            'T_celsius':     'Đ˘ĐµĐĽĐżĐµŃ€Đ°Ń‚ŃŃ€Đ° ŃŤĐşŃĐżĐµŃ€Đ¸ĐĽĐµĐ˝Ń‚Đ° (°C). ĐśĐ°ŃŃŃ‚Đ°Đ±Đ¸Ń€ŃĐµŃ‚ ĐşĐ¸Đ˝ĐµŃ‚Đ¸ĐşŃ Ń‡ĐµŃ€ĐµĐ· Q10.',
            'T_ref':         'Đ ĐµŃ„ĐµŃ€ĐµĐ˝ŃĐ˝Đ°ŃŹ Ń‚ĐµĐĽĐżĐµŃ€Đ°Ń‚ŃŃ€Đ° ĐşĐ°Đ˝Đ°Đ»ĐľĐ˛ (°C).',
            'Q10':           'ĐšĐľŃŤŃ„Ń„Đ¸Ń†Đ¸ĐµĐ˝Ń‚ Q10. ĐźŃ€Đ¸ Q10=3 ŃĐşĐľŃ€ĐľŃŃ‚ŃŚ Ń€Đ°ŃŃ‚Ń‘Ń‚ Đ˛ 3 Ń€Đ°Đ·Đ° Đ·Đ° 10°C.',
            'T_dend_offset': 'Đ Đ°Đ·Đ˝Đ¸Ń†Đ° Ń‚ĐµĐĽĐżĐµŃ€Đ°Ń‚ŃŃ€ Đ´ĐµĐ˝Đ´Ń€Đ¸Ń‚Ń‹–ŃĐľĐĽĐ° (°C). ĐźĐ»ŃŽŃ = Đ´ĐµĐ˝Đ´Ń€Đ¸Ń‚Ń‹ Ń‚ĐµĐżĐ»ĐµĐµ. Đ›Đ¸Đ˝ĐµĐąĐ˝Ń‹Đą Đ°ĐşŃĐ¸Đ°Đ»ŃŚĐ˝Ń‹Đą ĐłŃ€Đ°Đ´Đ¸ĐµĐ˝Ń‚.',

            # ── Симуляция ────────────────────────────────────────────
            't_sim':       'Длительность симуляции (мс).',
            'dt_eval':     'Шаг вывода (мс). Не влияет на точность интегратора.',
            'stim_type':   'Тип стимула: const / pulse / alpha / шум OU / синаптические рецепторы / ZAP-чирп.',
            'Iext':        'Амплитуда стимула (мкА/см²).',
            'pulse_start': 'Начало импульса (мс).',
            'pulse_dur':   'Длительность импульса (мс).',
            'alpha_tau':   'ĐźĐľŃŃ‚ĐľŃŹĐ˝Đ˝Đ°ŃŹ Đ˛Ń€ĐµĐĽĐµĐ˝Đ¸ alpha-ŃĐ¸Đ˝Đ°ĐżŃĐ° (ĐĽŃ). ĐźĐ¸Đş ĐżŃ€Đ¸ t = τ.',
            'zap_f0_hz':   'Начальная частота ZAP-чирпа (Гц).',
            'zap_f1_hz':   'Конечная частота ZAP-чирпа (Гц).',
            'stim_comp':   'Компартмент стимуляции (0 = сома).',
            'event_times': 'Очередь синаптических событий (мс). Временные метки через запятую: "10,25,40". Замещает pulse_start для AMPA/NMDA/GABA.',
            'synaptic_train_type': 'Генератор поездок импульсов: none (отсутствует) / regular (фиксированный ИПВ) / poisson (случайный).',
            'synaptic_train_freq_hz': 'Частота сгенерированного поезда импульсов (Гц).',
            'synaptic_train_duration_ms': 'Длительность поезда импульсов (мс).',
            'stoch_gating':'Шум Ланжевена в гейтах (Нативный Hines-солвер). Используйте кнопку СТОХАСТИКА.',
            'noise_sigma': 'Амплитуда белого шума на мембрану σ (мкА/см²).',

            # ── Анализ ───────────────────────────────────────────────
            'run_mc':       'ĐśĐľĐ˝Ń‚Đµ-ĐšĐ°Ń€Đ»Đľ: ĐżĐ°Ń€Đ°Đ»Đ»ĐµĐ»ŃŚĐ˝Ń‹Đµ Đ·Đ°ĐżŃŃĐşĐ¸ Ń ±5% Ń€Đ°Đ·Đ±Ń€ĐľŃĐľĐĽ gNa/gK.',
            'mc_trials':    'Число МК-попыток.',
            'run_bifurcation': 'Бифуркационный анализ после основного запуска.',
            'bif_param':    'Параметр для бифуркации.',
            'bif_min':      'Начало диапазона бифуркации.',
            'bif_max':      'Конец диапазона бифуркации.',
            'bif_steps':    'Количество точек бифуркации.',
            'run_sweep':    'Включить параметрический перебор (кнопка ↔).',
            'sweep_param':  'Перебираемый параметр: Iext, gNa_max, gK_max, T_celsius, ...',
            'sweep_min':    'Начальное значение перебора.',
            'sweep_max':    'Конечное значение перебора.',
            'sweep_steps':  'Количество точек перебора.',
            'run_sd_curve': 'Кривая Сила-Длительность (кнопка ⏱).',
            'run_excmap':   'Карта возбудимости 2-D (кнопка 🗺).',
            'excmap_I_min': 'Карта: мин. ток (мкА/см²).',
            'excmap_I_max': 'Карта: макс. ток (мкА/см²).',
            'excmap_NI':    'Карта: число шагов по току.',
            'excmap_D_min': 'Карта: мин. длительность (мс).',
            'excmap_D_max': 'Карта: макс. длительность (мс).',
            'excmap_ND':    'Карта: число шагов по длительности.',

            # ── Метки режимов пресетов ───────────────────────────────
            'k_mode_sleep': 'Сон (Всплески)',
            'k_mode_awake': 'Бодрствование (Реле)',
            'l5_mode_normal': 'Стандарт',
            'l5_mode_ach': 'Внимание (АCh ВКЛ)',

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
        return repair_text(self.TEXTS.get(self.lang, {}).get(key, key))

    def desc(self, key: str) -> str:
        """Return parameter description (for tooltips)."""
        return repair_text(self.TEXTS.get(self.lang, {}).get(key, ''))

    def func_desc(self, key: str) -> str:
        """Return function description for documentation tooltips."""
        return repair_text(self.TEXTS.get(self.lang, {}).get(key, ''))

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
                "K: Thalamic Relay (Ih + ITCa + Burst)": "K: Thalamic Relay (Ih + ITCa + Burst)",
                "L: Hippocampal CA1 Pyramidal (Adapting)": "L: Hippocampal CA1 Pyramidal (Adapting)",
                "M: Epilepsy (v10 SCN1A mutation)": "M: Epilepsy (v10 SCN1A mutation)",
                "N: Alzheimer's (v10 Calcium Toxicity)": "N: Alzheimer's (v10 Calcium Toxicity)",
                "O: Hypoxia (v10 ATP-pump failure)": "O: Hypoxia (v10 ATP-pump failure)",
                "P: Thalamic Reticular Nucleus (TRN Spindles)": "P: Thalamic Reticular Nucleus (TRN Spindles)",
                "Q: Striatal Spiny Projection (SPN)": "Q: Striatal Spiny Projection (SPN)",
                "R: Cholinergic Neuromodulation (ACh)": "R: Cholinergic Neuromodulation (ACh)",
                "S: Pathology: Dravet Syndrome (SCN1A LOF)": "S: Pathology: Dravet Syndrome (SCN1A LOF)",
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
                "I: In Vitro Slice (Mammalian 23°C)": "I: ĐˇŃ€ĐµĐ· in vitro (ĐĽĐ»ĐµĐşĐľĐżĐ¸Ń‚Đ°ŃŽŃ‰Đ¸Đµ 23°C)",
                "J: C-Fiber (Pain / Unmyelinated)": "J: C-волокно (боль / безмиелиновое)",
                "K: Thalamic Relay (Ih + ITCa + Burst)": "K: Таламическое реле (Ih + ITCa + пачки)",
                "L: Hippocampal CA1 Pyramidal (Adapting)": "L: Гиппокамп CA1 пирамидный (адаптивный)",
                "M: Epilepsy (v10 SCN1A mutation)": "M: Эпилепсия (мутация SCN1A v10)",
                "N: Alzheimer's (v10 Calcium Toxicity)": "N: Болезнь Альцгеймера (токсичность кальция v10)",
                "O: Hypoxia (v10 ATP-pump failure)": "O: Гипоксия (отказ АТФ-насоса v10)",
                "P: Thalamic Reticular Nucleus (TRN Spindles)": "P: Ретикулярное ядро таламуса (TRN веретёна)",
                "Q: Striatal Spiny Projection (SPN)": "Q: Шиповатый проекционный нейрон (SPN)",
                "R: Cholinergic Neuromodulation (ACh)": "R: Холинергическая нейромодуляция (ACh)",
                "S: Pathology: Dravet Syndrome (SCN1A LOF)": "S: Патология: Синдром Драве (SCN1A LOF)",
            }
        }
        return repair_text(preset_translations.get(self.lang, {}).get(preset_name, preset_name))


# Global translator instance — default English
T = Translator('EN')
