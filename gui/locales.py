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
            'tau_Ca':    'Ca²⁺ pump time constant (ms). Long τ -> prolonged SK activation.',
            'B_Ca':      'Current-to-concentration factor (mM / (µA/cm² · ms)).',

            # ── Metabolism descriptions ───────────────────────────────
            'enable_dynamic_atp': 'Enable dynamic ATP pool metabolism. ATP depletion opens K_ATP channels, hyperpolarizing the cell.',
            'g_katp_max': 'Maximum ATP-sensitive K⁺ conductance (mS/cm²). Opens when ATP < 0.5 mM.',
            'katp_kd_atp_mM': 'ATP concentration for half-activation of K_ATP (mM).',
            'atp_max_mM': 'Baseline intracellular ATP concentration (mM). Healthy neurons: 2-5 mM.',
            'atp_synthesis_rate': 'ATP synthesis rate via oxidative phosphorylation (mM/s). Healthy ~0.5; hypoxia 0.005–0.05.',
            'pump_max_capacity': 'Max Na+/K+ pump current density (µA/cm²). Literature: 0.1-0.5 µA/cm².',
            'km_na': 'Na+ half-saturation constant for pump (mM). Literature: 10-20 mM.',
            'enable_atp_limiting': 'Enable ATP-dependent pump limitation (pump slows when ATP < 2 mM).',
            'filter_mode': 'Attenuation mode: Classic (DC) for steady-state, Physiological (AC) for synaptic inputs.',
            'input_frequency': 'Input signal frequency for AC attenuation (Hz). Typical synaptic: 100-200 Hz.',

            # ── Environment descriptions ─────────────────────────────
            'T_celsius':     'Experiment temperature (°C). Scales all kinetics via Q10.',
            'T_ref':         'Reference temperature for channel kinetics (°C).',
            'Q10':           'Q10 coefficient. Rate doubles per 10°C if Q10 = 2.',
            'T_dend_offset': 'Dendrite–soma temperature difference (°C). Positive = dendrites warmer. Creates linear axial gradient.',

            # ── Simulation descriptions ──────────────────────────────
            't_sim':       'Total simulation time (ms).',
            'dt_eval':     'Output sample interval (ms). Does not affect solver accuracy.',
            'stim_type':   'Stimulus waveform: const / pulse / alpha / OU noise / synaptic receptors / zap chirp.',
            'Iext':        'Stimulus amplitude (µA/cm²).',
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
            'sweep_param':  'Parameter to vary in sweep. Options: Iext, gNa_max, gK_max, T_celsius, ...',
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

            # ── Preset Mode Labels ─────────────────────────────────
            'k_mode_sleep': 'Sleep (Bursts)',
            'k_mode_awake': 'Awake (Relay)',
            'l5_mode_normal': 'Standard',
            'l5_mode_ach': 'Attention (ACh ON)',
            'preset_e_recalibrated': (
                'Purkinje preset now uses dendritic event-drive calibration to keep simple-spike '
                'rates in a physiological 30-50 Hz window instead of excitotoxic 200+ Hz runaway.'
            ),
            'preset_k_recalibrated': (
                'Thalamic relay preset now separates low-throughput relay mode from activated burst mode; '
                'theta-like behavior is treated as event-driven network context, not intrinsic resonance.'
            ),
            'preset_l_recalibrated': (
                'CA1 preset now uses stronger leak/repolarization reserve and distal filtering so theta-paced '
                'firing remains in physiological low-throughput range under aggregated synaptic drive.'
            ),
            'preset_q_recalibrated': (
                'SPN preset now reduces tonic up-state proxy drive and keeps delayed recruitment, preventing '
                'non-physiological high-rate firing while preserving dendritic-filtered integration.'
            ),

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

            # ── Favorites widget ───────────────────────────────────────
            'favorites_label': 'Favorites:',
            'favorites_manage': 'Manage favorites',
            'favorites_add_current': 'Add current preset',
            'favorites_edit': 'Edit favorites...',
            'favorites_reset': 'Reset to defaults',
            'favorites_add': 'Add...',
            'favorites_remove': 'Remove',
            'favorites_add_title': 'Add Favorite',
            'favorites_add_prompt': 'Enter preset name:',
            'favorites_edit_title': 'Edit Favorites',
            'favorites_edit_hint': 'Drag to reorder. Double-click to edit.',
            'dialog_ok': 'OK',
            'dialog_cancel': 'Cancel',

            # ── Quick Stats widget ───────────────────────────────────
            'stat_spikes': 'Spikes:',
            'stat_rate': 'Rate:',
            'stat_vmin': 'Vmin:',
            'stat_vmax': 'Vmax:',
            'stat_atp': 'ATP:',
            'stat_ready': 'Ready',
            'stat_active': 'Active',
            'stat_silent': 'Silent',
            'btn_quick_export': 'Quick Export',
            'tt_quick_export': 'Export current results to CSV',
            'tt_stat_spikes': 'Total spike count',
            'tt_stat_rate': 'Initial/steady firing rate',
            'tt_stat_vmin': 'Minimum membrane potential',
            'tt_stat_vmax': 'Maximum membrane potential',
            'tt_stat_atp': 'Minimum ATP concentration',

            # ── Analytics tabs ───────────────────────────────────────
            'tab_chaos_lle': 'Chaos & LLE',
            'tab_phase_plane': 'Phase Plane',
            'tab_spike_mech': 'Spike Mechanism',
            'tab_kymograph': 'Kymograph',
            'tab_spectrogram': 'Spectrogram',
            'tab_impedance': 'Impedance',
            'tab_energy_balance': 'Energy Balance',
            'tab_currents': 'Currents',
            'tab_bifurcation': 'Bifurcation',
            'tab_poincare': 'Poincaré (ISI)',
            'tab_isi_dist': 'ISI Distribution',
            'tab_modulation': 'Phase-Locking',
            'tab_passport': 'Passport',

            # ── Chaos & LLE specific ─────────────────────────────────
            'lbl_embed': 'Embed',
            'lbl_lag': 'Lag',
            'lbl_fit_start': 'Fit start',
            'lbl_fit_end': 'Fit end',
            'lbl_min_sep': 'Min sep',
            'lbl_pair': 'Pair',
            'pair_representative': 'Representative',
            'pair_first': 'First pair',
            'pair_index': 'Pair index',
            'tip_chaos_click': 'Click the divergence curve to inspect a specific separation horizon.',
            'summary_class': 'class',
            'summary_lle': 'LLE',
            'summary_pairs': 'pairs',
            'summary_selected': 'selected k',
            'summary_mode': 'mode',
            'stoch_warning': 'STOCHASTIC INPUT: LLE mainly reflects noise entropy.',

            # ── Phase Plane specific ─────────────────────────────────
            'pp_gate_m': 'm (Na activation)',
            'pp_gate_h': 'h (Na inactivation)',
            'pp_gate_n': 'n (K activation)',
            'pp_gate_r': 'r (Ih)',
            'pp_gate_s': 's (SK)',

            # ── Keyboard shortcuts ───────────────────────────────────
            'shortcuts_title': 'Keyboard Shortcuts',
            'menu_help_shortcuts': 'Keyboard Shortcuts...',
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
            'single_comp':   'Одиночная сома: нейрон — точка, без аксона.',
            'd_soma':        'Диаметр сомы (см). Больше -> больше ёмкость -> труднее возбудить.',
            'N_ais':         'Сегменты AIS. AIS — «курок» нейрона, зона инициации спайка.',
            'd_ais':         'Диаметр AIS (см).',
            'l_ais':         'Длина сегмента AIS (см).',
            'gNa_ais_mult':  'Множитель gNa в AIS (обычно 40–100x). Задаёт порог возбудимости.',
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
            'tau_Ca':    'Время откачки Ca²⁺ (мс). Длинный τ -> долгий SK-ток -> адаптация.',
            'B_Ca':      'Конверсия тока в концентрацию (мМ / (мкА/см² · мс)).',

            # ── Метаболизм ───────────────────────────────────────────
            'enable_dynamic_atp': 'Включить динамический пул АТФ. Истощение АТФ открывает K_ATP каналы, гиперполяризуя клетку.',
            'g_katp_max': 'Макс. проводимость ATP-чувствительных K⁺ каналов (мСм/см²). Открываются при АТФ < 0.5 мМ.',
            'katp_kd_atp_mM': 'Концентрация АТФ для полуактивации K_ATP (мМ).',
            'atp_max_mM': 'Базовая внутриклеточная концентрация АТФ (мМ). Здоровые нейроны: 2-5 мМ.',
            'atp_synthesis_rate': 'Скорость синтеза АТФ через окислительное фосфорилирование (мМ/с). Норма ~0.5; гипоксия 0.005–0.05.',
            'pump_max_capacity': 'Макс. плотность тока насоса Na+/K+ (мкА/см²). Литература: 0.1-0.5 мкА/см².',
            'km_na': 'Полунасыщение Na+ для насоса (мМ). Литература: 10-20 мМ.',
            'enable_atp_limiting': 'Включить АТФ-зависимое ограничение насоса (замедляется при АТФ < 2 мМ).',
            'filter_mode': 'Режим затухания: Классика (DC) для стационарного, Физиологический (AC) для синаптических входов.',
            'input_frequency': 'Частота входного сигнала для AC затухания (Гц). Типичная синаптическая: 100-200 Гц.',

            # ── Среда ────────────────────────────────────────────────
            'T_celsius':     'Температура эксперимента (°C). Масштабирует кинетику через Q10.',
            'T_ref':         'Референсная температура каналов (°C).',
            'Q10':           'Коэффициент Q10. При Q10=3 скорость растёт в 3 раза за 10°C.',
            'T_dend_offset': 'Разница температур дендриты–сома (°C). Плюс = дендриты теплее. Линейный аксиальный градиент.',

            # ── Симуляция ────────────────────────────────────────────
            't_sim':       'Длительность симуляции (мс).',
            'dt_eval':     'Шаг вывода (мс). Не влияет на точность интегратора.',
            'stim_type':   'Тип стимула: const / pulse / alpha / шум OU / синаптические рецепторы / ZAP-чирп.',
            'Iext':        'Амплитуда стимула (мкА/см²).',
            'pulse_start': 'Начало импульса (мс).',
            'pulse_dur':   'Длительность импульса (мс).',
            'alpha_tau':   'Постоянная времени alpha-синапса (мс). Пик при t = τ.',
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
            'run_mc':       'Монте-Карло: параллельные запуски с ±5% разбросом gNa/gK.',
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
            'preset_e_recalibrated': (
                '?????? ???????? ?????????????? ????? ?????????-?????????? ?????: '
                '??????? ?????? ???????????? ? ??????????????? ???? 30-50 ?? '
                '?????? ??????????????????? ??????? >200 ??.'
            ),
            'preset_k_recalibrated': (
                '???????????? ?????? ????????? relay-????? (?????? ?????????? ???????????) '
                '? activated burst-?????; ???? ?????????? ??? ??????? ?????????? ????????, '
                '? ?? ??? ?????????? ????????? ?????? ???????.'
            ),
            'preset_l_recalibrated': (
                'CA1 preset recalibrated: stronger leak/repolarization plus distal filtering keeps theta-paced '
                'activity in low-throughput physiological range.'
            ),
            'preset_q_recalibrated': (
                'SPN preset recalibrated: weaker tonic up-state proxy drive preserves delayed recruitment and '
                'prevents non-physiological high-rate spiking.'
            ),

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

            # ── Favorites widget ───────────────────────────────────────
            'favorites_label': 'Избранное:',
            'favorites_manage': 'Управление избранным',
            'favorites_add_current': 'Добавить текущий пресет',
            'favorites_edit': 'Редактировать избранное...',
            'favorites_reset': 'Сбросить на стандартные',
            'favorites_add': 'Добавить...',
            'favorites_remove': 'Удалить',
            'favorites_add_title': 'Добавить в избранное',
            'favorites_add_prompt': 'Введите имя пресета:',
            'favorites_edit_title': 'Редактирование избранного',
            'favorites_edit_hint': 'Перетаскивайте для изменения порядка. Двойной клик для редактирования.',
            'dialog_ok': 'OK',
            'dialog_cancel': 'Отмена',

            # ── Quick Stats widget ───────────────────────────────────
            'stat_spikes': 'Спайки:',
            'stat_rate': 'Частота:',
            'stat_vmin': 'Vmin:',
            'stat_vmax': 'Vmax:',
            'stat_atp': 'АТФ:',
            'stat_ready': 'Готов',
            'stat_active': 'Активен',
            'stat_silent': 'Молчит',
            'btn_quick_export': 'Быстрый экспорт',
            'tt_quick_export': 'Экспортировать результаты в CSV',
            'tt_stat_spikes': 'Общее число спайков',
            'tt_stat_rate': 'Начальная/установившаяся частота',
            'tt_stat_vmin': 'Минимальный мембранный потенциал',
            'tt_stat_vmax': 'Максимальный мембранный потенциал',
            'tt_stat_atp': 'Минимальная концентрация АТФ',

            # ── Аналитические вкладки ────────────────────────────────
            'tab_chaos_lle': 'Хаос и ЛЯ',
            'tab_phase_plane': 'Фазовая плоскость',
            'tab_spike_mech': 'Механика спайка',
            'tab_kymograph': 'Кимограф',
            'tab_spectrogram': 'Спектрограмма',
            'tab_impedance': 'Импеданс',
            'tab_energy_balance': 'Энергетический баланс',
            'tab_currents': 'Токи',
            'tab_bifurcation': 'Бифуркация',
            'tab_poincare': 'Пуанкаре (ИПВ)',
            'tab_isi_dist': 'Распределение ИПВ',
            'tab_modulation': 'Фазовая синхронизация',
            'tab_passport': 'Паспорт',

            # ── Chaos & LLE специфичные ───────────────────────────────
            'lbl_embed': 'Размерность',
            'lbl_lag': 'Лаг',
            'lbl_fit_start': 'Начало фита',
            'lbl_fit_end': 'Конец фита',
            'lbl_min_sep': 'Мин. разделение',
            'lbl_pair': 'Пара',
            'pair_representative': 'Репрезентативная',
            'pair_first': 'Первая пара',
            'pair_index': 'Индекс пары',
            'tip_chaos_click': 'Кликните на кривую расхождения для просмотра конкретного горизонта.',
            'summary_class': 'класс',
            'summary_lle': 'ЛЯ',
            'summary_pairs': 'пары',
            'summary_selected': 'выбрано k',
            'summary_mode': 'режим',
            'stoch_warning': 'СТОХАСТИЧЕСКИЙ ВХОД: ЛЯ в основном отражает энтропию шума.',

            # ── Фазовая плоскость ────────────────────────────────────
            'pp_gate_m': 'm (активация Na)',
            'pp_gate_h': 'h (инаактивация Na)',
            'pp_gate_n': 'n (активация K)',
            'pp_gate_r': 'r (Ih)',
            'pp_gate_s': 's (SK)',

            # ── Keyboard shortcuts ───────────────────────────────────
            'shortcuts_title': 'Горячие клавиши',
            'menu_help_shortcuts': 'Горячие клавиши...',
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
                "I: In Vitro Slice (Mammalian 23°C)": "I: Срез in vitro (млекопитающие 23°C)",
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
