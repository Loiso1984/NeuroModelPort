"""
gui/locales.py - Internationalization (i18n) v11.3

Provides English and Russian translations for all UI strings
and scientific parameter descriptions (used as tooltips).
"""

from gui.text_sanitize import repair_text


class Translator:
    """Manages UI and tooltip translations with extended bilingual support."""

    TEXTS = {
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

            # â”€â”€ Morphology descriptions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            'single_comp':   'Single-compartment: neuron is treated as a point (0-D), no axon.',
            'd_soma':        'Soma diameter (cm). Larger soma = more capacitance = harder to excite.',
            'N_ais':         'AIS segments. The axon initial segment is the spike trigger zone.',
            'd_ais':         'AIS diameter (cm).',
            'l_ais':         'AIS segment length (cm).',
            'gNa_ais_mult':  'gNa multiplier in AIS (typically 40â€“100Ă—). Sets excitability threshold.',
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
            'Ra':            'Axial resistance (Î©Â·cm). High Ra = signal attenuates along axon.',
            'dx':            'Compartment length (cm).',

            # â”€â”€ Channel descriptions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            'Cm':        'Membrane capacitance (ÂµF/cmÂ˛). Electrical inertia â€” higher Cm slows response.',
            'gNa_max':   'Max Naâş conductance. Drives the fast depolarising upstroke of the AP.',
            'gK_max':    'Max Kâş conductance. Repolarises the membrane after the spike.',
            'gL':        'Leak conductance. Sets resting resistance; higher = shorter Ď„_m.',
            'ENa':       'Naâş reversal potential (mV). Spike peaks approach this value.',
            'EK':        'Kâş reversal potential (mV). After-hyperpolarisation floor.',
            'EL':        'Leak reversal potential (mV). Approximates the resting membrane potential.',
            'enable_Ih': 'Enable Ih (HCN pacemaker). Causes rhythmic spontaneous firing.',
            'gIh_max':   'Max Ih conductance (mS/cmÂ˛).',
            'E_Ih':      'Ih reversal potential (mV).',
            'enable_ICa':'Enable L-type CaÂ˛âş current (Huguenard 1992). Enables plateau potentials.',
            'gCa_max':   'Max ICa conductance (mS/cmÂ˛).',
            'E_Ca':      'CaÂ˛âş reversal potential (mV, overridden by Nernst if dynamic Ca is on).',
            'enable_IA': 'Enable A-current (Connor-Stevens). Transient Kâş; delays first spike.',
            'gA_max':    'Max IA conductance (mS/cmÂ˛).',
            'E_A':       'IA reversal potential (mV).',
            'enable_SK': 'Enable SK channel (Ca-activated Kâş). Spike-frequency adaptation.',
            'gSK_max':   'Max SK conductance. Strength of calcium-driven adaptation.',

            # â”€â”€ Calcium descriptions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            'dynamic_Ca':'Enable dynamic [CaÂ˛âş]áµ˘ tracking and Nernst correction for E_Ca.',
            'Ca_ext':    'Extracellular [CaÂ˛âş] (mM). Used in Nernst equation.',
            'Ca_rest':   'Resting [CaÂ˛âş]áµ˘ (mM). Baseline before stimulation.',
            'tau_Ca':    'CaÂ˛âş pump time constant (ms). Long Ď„ â†’ prolonged SK activation.',
            'B_Ca':      'Current-to-concentration factor (mM / (ÂµA/cmÂ˛ Â· ms)).',

            # â”€â”€ Metabolism descriptions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            'enable_dynamic_atp': 'Enable dynamic ATP pool metabolism. ATP depletion opens K_ATP channels, hyperpolarizing the cell.',
            'g_katp_max': 'Maximum ATP-sensitive Kâş conductance (mS/cmÂ˛). Opens when ATP < 0.5 mM.',
            'katp_kd_atp_mM': 'ATP concentration for half-activation of K_ATP (mM).',
            'atp_max_mM': 'Baseline intracellular ATP concentration (mM). Healthy neurons: 2-5 mM.',
            'atp_synthesis_rate': 'ATP synthesis rate via oxidative phosphorylation (nmol/cmÂ˛/s).',

            # â”€â”€ Environment descriptions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            'T_celsius':     'Experiment temperature (Â°C). Scales all kinetics via Q10.',
            'T_ref':         'Reference temperature for channel kinetics (Â°C).',
            'Q10':           'Q10 coefficient. Rate doubles per 10Â°C if Q10 = 2.',
            'T_dend_offset': 'Dendriteâ€“soma temperature difference (Â°C). Positive = dendrites warmer. Creates linear axial gradient.',

            # â”€â”€ Simulation descriptions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            't_sim':       'Total simulation time (ms).',
            'dt_eval':     'Output sample interval (ms). Does not affect solver accuracy.',
            'stim_type':   'Stimulus waveform: const / pulse / alpha / OU noise / synaptic receptors / zap chirp.',
            'Iext':        'Stimulus amplitude (ÂµA/cmÂ˛).',
            'pulse_start': 'Pulse onset time (ms).',
            'pulse_dur':   'Pulse duration (ms).',
            'alpha_tau':   'Alpha-synapse time constant (ms). Peak at t = Ď„ after onset.',
            'zap_f0_hz':   'ZAP start frequency (Hz) for chirp stimulus.',
            'zap_f1_hz':   'ZAP end frequency (Hz) for chirp stimulus.',
            'stim_comp':   'Compartment index to inject current into (0 = soma).',
            'event_times': 'Synaptic event queue (ms). Comma-separated timestamps; e.g. "10,25,40". Overrides pulse_start for AMPA/NMDA/GABA types.',
            'synaptic_train_type': 'Spike Train Generator: none / regular (fixed ISI) / poisson (random).',
            'synaptic_train_freq_hz': 'Frequency of the generated spike train (Hz).',
            'synaptic_train_duration_ms': 'Duration of the spike train (ms).',
            'stoch_gating':'Langevin gate noise via Native Hines solver. Use the STOCHASTIC button.',
            'noise_sigma': 'Additive white noise amplitude Ď (ÂµA/cmÂ˛). Added to dV/dt.',

            # â”€â”€ Analysis descriptions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            'run_mc':       'Run Monte-Carlo: parallel trials with Â±5% gNa/gK variability.',
            'mc_trials':    'Number of MC trials.',
            'run_bifurcation': 'Compute bifurcation diagram after standard run.',
            'bif_param':    'Parameter for bifurcation sweep.',
            'bif_min':      'Bifurcation range start.',
            'bif_max':      'Bifurcation range end.',
            'bif_steps':    'Number of bifurcation steps.',
            'run_sweep':    'Enable parametric sweep (use â†” SWEEP button).',
            'sweep_param':  'Parameter to vary in sweep. Options: Iext, gNa_max, gK_max, T_celsius, â€¦',
            'sweep_min':    'Sweep start value.',
            'sweep_max':    'Sweep end value.',
            'sweep_steps':  'Number of sweep points.',
            'run_sd_curve': 'Compute Strength-Duration curve (use âŹ± S-D button).',
            'run_excmap':   'Compute 2-D excitability map (use đź—ş button).',
            'excmap_I_min': 'Excitability map: minimum current (ÂµA/cmÂ˛).',
            'excmap_I_max': 'Excitability map: maximum current (ÂµA/cmÂ˛).',
            'excmap_NI':    'Excitability map: number of current steps.',
            'excmap_D_min': 'Excitability map: minimum pulse duration (ms).',
            'excmap_D_max': 'Excitability map: maximum pulse duration (ms).',
            'excmap_ND':    'Excitability map: number of duration steps.',

            # â”€â”€ Preset Mode Labels â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            'k_mode_sleep': 'Sleep (Bursts)',
            'k_mode_awake': 'Awake (Relay)',
            'l5_mode_normal': 'Standard',
            'l5_mode_ach': 'Attention (ACh ON)',

            # â”€â”€ Core Function Translations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

            # â”€â”€ ĐśĐľŃ€Ń„ĐľĐ»ĐľĐłĐ¸ŃŹ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            'single_comp':   'ĐžĐ´Đ¸Đ˝ĐľŃ‡Đ˝Đ°ŃŹ ŃĐľĐĽĐ°: Đ˝ĐµĐąŃ€ĐľĐ˝ â€” Ń‚ĐľŃ‡ĐşĐ°, Đ±ĐµĐ· Đ°ĐşŃĐľĐ˝Đ°.',
            'd_soma':        'Đ”Đ¸Đ°ĐĽĐµŃ‚Ń€ ŃĐľĐĽŃ‹ (ŃĐĽ). Đ‘ĐľĐ»ŃŚŃĐµ â†’ Đ±ĐľĐ»ŃŚŃĐµ Ń‘ĐĽĐşĐľŃŃ‚ŃŚ â†’ Ń‚Ń€ŃĐ´Đ˝ĐµĐµ Đ˛ĐľĐ·Đ±ŃĐ´Đ¸Ń‚ŃŚ.',
            'N_ais':         'ĐˇĐµĐłĐĽĐµĐ˝Ń‚Ń‹ AIS. AIS â€” Â«ĐşŃŃ€ĐľĐşÂ» Đ˝ĐµĐąŃ€ĐľĐ˝Đ°, Đ·ĐľĐ˝Đ° Đ¸Đ˝Đ¸Ń†Đ¸Đ°Ń†Đ¸Đ¸ ŃĐżĐ°ĐąĐşĐ°.',
            'd_ais':         'Đ”Đ¸Đ°ĐĽĐµŃ‚Ń€ AIS (ŃĐĽ).',
            'l_ais':         'Длина сегмента AIS (см).',
            'gNa_ais_mult':  'ĐśĐ˝ĐľĐ¶Đ¸Ń‚ĐµĐ»ŃŚ gNa Đ˛ AIS (ĐľĐ±Ń‹Ń‡Đ˝Đľ 40â€“100Ă—). Đ—Đ°Đ´Đ°Ń‘Ń‚ ĐżĐľŃ€ĐľĐł Đ˛ĐľĐ·Đ±ŃĐ´Đ¸ĐĽĐľŃŃ‚Đ¸.',
            'gK_ais_mult':   'ĐśĐ˝ĐľĐ¶Đ¸Ń‚ĐµĐ»ŃŚ gK Đ˛ AIS.',
            'gIh_ais_mult':  'ĐśĐ˝ĐľĐ¶Đ¸Ń‚ĐµĐ»ŃŚ gIh Đ˛ AIS.',
            'gCa_ais_mult':  'ĐśĐ˝ĐľĐ¶Đ¸Ń‚ĐµĐ»ŃŚ gCa Đ˛ AIS.',
            'gA_ais_mult':   'ĐśĐ˝ĐľĐ¶Đ¸Ń‚ĐµĐ»ŃŚ gA Đ˛ AIS.',
            'N_trunk':       'ĐˇĐµĐłĐĽĐµĐ˝Ń‚ĐľĐ˛ ŃŃ‚Đ˛ĐľĐ»Đ° Đ°ĐşŃĐľĐ˝Đ°.',
            'd_trunk':       'Đ”Đ¸Đ°ĐĽĐµŃ‚Ń€ ŃŃ‚Đ˛ĐľĐ»Đ° (ŃĐĽ). Đ˘ĐľĐ»Ń‰Đµ â†’ Đ±Ń‹ŃŃ‚Ń€ĐµĐµ ĐżŃ€ĐľĐ˛ĐµĐ´ĐµĐ˝Đ¸Đµ.',
            'N_b1':          'ĐˇĐµĐłĐĽĐµĐ˝Ń‚ĐľĐ˛ Đ˛ĐµŃ‚Đ˛Đ¸ 1.',
            'd_b1':          'Đ”Đ¸Đ°ĐĽĐµŃ‚Ń€ Đ˛ĐµŃ‚Đ˛Đ¸ 1 (ŃĐĽ).',
            'N_b2':          'ĐˇĐµĐłĐĽĐµĐ˝Ń‚ĐľĐ˛ Đ˛ĐµŃ‚Đ˛Đ¸ 2.',
            'd_b2':          'Đ”Đ¸Đ°ĐĽĐµŃ‚Ń€ Đ˛ĐµŃ‚Đ˛Đ¸ 2 (ŃĐĽ).',
            'Ra':            'ĐĐşŃĐ¸Đ°Đ»ŃŚĐ˝ĐľĐµ ŃĐľĐżŃ€ĐľŃ‚Đ¸Đ˛Đ»ĐµĐ˝Đ¸Đµ (ĐžĐĽÂ·ŃĐĽ). Đ’Ń‹ŃĐľĐşĐľĐµ Ra ĐłĐ°ŃĐ¸Ń‚ ŃĐ¸ĐłĐ˝Đ°Đ».',
            'dx':            'Đ”Đ»Đ¸Đ˝Đ° ĐľĐ´Đ˝ĐľĐłĐľ ŃĐµĐłĐĽĐµĐ˝Ń‚Đ° (ŃĐĽ).',

            # â”€â”€ ĐšĐ°Đ˝Đ°Đ»Ń‹ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            'Cm':        'ĐĐĽĐşĐľŃŃ‚ŃŚ ĐĽĐµĐĽĐ±Ń€Đ°Đ˝Ń‹ (ĐĽĐşĐ¤/ŃĐĽÂ˛). ĐĐ˝ĐµŃ€Ń†Đ¸ŃŹ: Đ±ĐľĐ»ŃŚŃĐ°ŃŹ Cm Đ·Đ°ĐĽĐµĐ´Đ»ŃŹĐµŃ‚ Ń€ĐµĐ°ĐşŃ†Đ¸ŃŽ.',
            'gNa_max':   'ĐśĐ°ĐşŃ. ĐżŃ€ĐľĐ˛ĐľĐ´Đ¸ĐĽĐľŃŃ‚ŃŚ Naâş. ĐžŃ‚Đ˛ĐµŃ‡Đ°ĐµŃ‚ Đ·Đ° Đ±Ń‹ŃŃ‚Ń€Ń‹Đą Ń„Ń€ĐľĐ˝Ń‚ ŃĐżĐ°ĐąĐşĐ°.',
            'gK_max':    'ĐśĐ°ĐşŃ. ĐżŃ€ĐľĐ˛ĐľĐ´Đ¸ĐĽĐľŃŃ‚ŃŚ Kâş. Đ ĐµĐżĐľĐ»ŃŹŃ€Đ¸Đ·ŃĐµŃ‚ ĐĽĐµĐĽĐ±Ń€Đ°Đ˝Ń ĐżĐľŃĐ»Đµ ŃĐżĐ°ĐąĐşĐ°.',
            'gL':        'ĐźŃ€ĐľĐ˛ĐľĐ´Đ¸ĐĽĐľŃŃ‚ŃŚ ŃŃ‚ĐµŃ‡ĐşĐ¸. Đ—Đ°Đ´Đ°Ń‘Ń‚ Đ˛Ń…ĐľĐ´Đ˝ĐľĐµ ŃĐľĐżŃ€ĐľŃ‚Đ¸Đ˛Đ»ĐµĐ˝Đ¸Đµ Đ¸ Ď„_m.',
            'ENa':       'ĐźĐľŃ‚ĐµĐ˝Ń†Đ¸Đ°Đ» Ń€ĐµĐ˛ĐµŃ€ŃĐ¸Đ¸ Naâş (ĐĽĐ’). ĐźĐ¸Đş ŃĐżĐ°ĐąĐşĐ° ŃŃ‚Ń€ĐµĐĽĐ¸Ń‚ŃŃŹ Đş ŃŤŃ‚ĐľĐĽŃ Đ·Đ˝Đ°Ń‡ĐµĐ˝Đ¸ŃŽ.',
            'EK':        'ĐźĐľŃ‚ĐµĐ˝Ń†Đ¸Đ°Đ» Ń€ĐµĐ˛ĐµŃ€ŃĐ¸Đ¸ Kâş (ĐĽĐ’). Â«Đ”Đ˝ĐľÂ» Ń€ĐµĐżĐľĐ»ŃŹŃ€Đ¸Đ·Đ°Ń†Đ¸Đ¸.',
            'EL':        'ĐźĐľŃ‚ĐµĐ˝Ń†Đ¸Đ°Đ» Ń€ĐµĐ˛ĐµŃ€ŃĐ¸Đ¸ ŃŃ‚ĐµŃ‡ĐşĐ¸ (ĐĽĐ’). ĐźĐľŃ‚ĐµĐ˝Ń†Đ¸Đ°Đ» ĐżĐľĐşĐľŃŹ.',
            'enable_Ih': 'Đ’ĐşĐ»ŃŽŃ‡Đ¸Ń‚ŃŚ Ih (ĐżĐµĐąŃĐĽĐµĐąĐşĐµŃ€Đ˝Ń‹Đą Ń‚ĐľĐş). Đ’Ń‹Đ·Ń‹Đ˛Đ°ĐµŃ‚ ŃĐżĐľĐ˝Ń‚Đ°Đ˝Đ˝Ń‹Đą Ń€Đ¸Ń‚ĐĽ.',
            'gIh_max':   'ĐśĐ°ĐşŃ. ĐżŃ€ĐľĐ˛ĐľĐ´Đ¸ĐĽĐľŃŃ‚ŃŚ Ih (ĐĽĐˇĐĽ/ŃĐĽÂ˛).',
            'E_Ih':      'ĐźĐľŃ‚ĐµĐ˝Ń†Đ¸Đ°Đ» Ń€ĐµĐ˛ĐµŃ€ŃĐ¸Đ¸ Ih (ĐĽĐ’).',
            'enable_ICa':'Đ’ĐşĐ»ŃŽŃ‡Đ¸Ń‚ŃŚ L-Ń‚Đ¸Đż CaÂ˛âş (Huguenard 1992). ĐźĐľĐ·Đ˛ĐľĐ»ŃŹĐµŃ‚ ĐżĐ»Đ°Ń‚Đľ-ĐżĐľŃ‚ĐµĐ˝Ń†Đ¸Đ°Đ»Ń‹.',
            'gCa_max':   'ĐśĐ°ĐşŃ. ĐżŃ€ĐľĐ˛ĐľĐ´Đ¸ĐĽĐľŃŃ‚ŃŚ ICa (ĐĽĐˇĐĽ/ŃĐĽÂ˛).',
            'E_Ca':      'ĐźĐľŃ‚ĐµĐ˝Ń†Đ¸Đ°Đ» Ń€ĐµĐ˛ĐµŃ€ŃĐ¸Đ¸ CaÂ˛âş (ĐĽĐ’). ĐźŃ€Đ¸ Đ˛ĐşĐ»ŃŽŃ‡Ń‘Đ˝Đ˝ĐľĐĽ Ca ĐżĐµŃ€ĐµĐ·Đ°ĐżĐ¸ŃŃ‹Đ˛Đ°ĐµŃ‚ŃŃŹ ĐťĐµŃ€Đ˝ŃŃ‚ĐľĐĽ.',
            'enable_IA': 'Đ’ĐşĐ»ŃŽŃ‡Đ¸Ń‚ŃŚ A-Ń‚ĐľĐş (Connor-Stevens). Đ—Đ°Đ´ĐµŃ€Đ¶Đ¸Đ˛Đ°ĐµŃ‚ ĐżĐµŃ€Đ˛Ń‹Đą ŃĐżĐ°ĐąĐş.',
            'gA_max':    'ĐśĐ°ĐşŃ. ĐżŃ€ĐľĐ˛ĐľĐ´Đ¸ĐĽĐľŃŃ‚ŃŚ IA (ĐĽĐˇĐĽ/ŃĐĽÂ˛).',
            'E_A':       'ĐźĐľŃ‚ĐµĐ˝Ń†Đ¸Đ°Đ» Ń€ĐµĐ˛ĐµŃ€ŃĐ¸Đ¸ IA (ĐĽĐ’).',
            'enable_SK': 'Đ’ĐşĐ»ŃŽŃ‡Đ¸Ń‚ŃŚ SK-ĐşĐ°Đ˝Đ°Đ» (CaÂ˛âş-Đ·Đ°Đ˛Đ¸ŃĐ¸ĐĽŃ‹Đą Kâş). ĐˇĐżĐ°ĐąĐşĐľĐ˛Đ°ŃŹ Đ°Đ´Đ°ĐżŃ‚Đ°Ń†Đ¸ŃŹ.',
            'gSK_max':   'ĐśĐ°ĐşŃ. ĐżŃ€ĐľĐ˛ĐľĐ´Đ¸ĐĽĐľŃŃ‚ŃŚ SK. ĐˇĐ¸Đ»Đ° ĐşĐ°Đ»ŃŚŃ†Đ¸Đą-Đ·Đ°Đ˛Đ¸ŃĐ¸ĐĽĐľĐłĐľ Ń‚ĐľŃ€ĐĽĐľĐ¶ĐµĐ˝Đ¸ŃŹ.',

            # â”€â”€ ĐšĐ°Đ»ŃŚŃ†Đ¸Đą â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            'dynamic_Ca':'Đ”Đ¸Đ˝Đ°ĐĽĐ¸Ń‡ĐµŃĐşĐ¸Đą [CaÂ˛âş]áµ˘ Ń ĐżĐµŃ€ĐµŃŃ‡Ń‘Ń‚ĐľĐĽ ĐťĐµŃ€Đ˝ŃŃ‚Đ° Đ´Đ»ŃŹ E_Ca.',
            'Ca_ext':    'Đ’Đ˝ĐµĐşĐ»ĐµŃ‚ĐľŃ‡Đ˝Ń‹Đą [CaÂ˛âş] (ĐĽĐś). ĐŃĐżĐľĐ»ŃŚĐ·ŃĐµŃ‚ŃŃŹ Đ˛ ŃŃ€Đ°Đ˛Đ˝ĐµĐ˝Đ¸Đ¸ ĐťĐµŃ€Đ˝ŃŃ‚Đ°.',
            'Ca_rest':   'ĐźĐľĐşĐľŃŹŃ‰Đ¸ĐąŃŃŹ [CaÂ˛âş]áµ˘ (ĐĽĐś). Đ‘Đ°Đ·ĐľĐ˛Ń‹Đą ŃŃ€ĐľĐ˛ĐµĐ˝ŃŚ Đ´Đľ ŃŃ‚Đ¸ĐĽŃĐ»Đ°.',
            'tau_Ca':    'Đ’Ń€ĐµĐĽŃŹ ĐľŃ‚ĐşĐ°Ń‡ĐşĐ¸ CaÂ˛âş (ĐĽŃ). Đ”Đ»Đ¸Đ˝Đ˝Ń‹Đą Ď„ â†’ Đ´ĐľĐ»ĐłĐ¸Đą SK-Ń‚ĐľĐş â†’ Đ°Đ´Đ°ĐżŃ‚Đ°Ń†Đ¸ŃŹ.',
            'B_Ca':      'ĐšĐľĐ˝Đ˛ĐµŃ€ŃĐ¸ŃŹ Ń‚ĐľĐşĐ° Đ˛ ĐşĐľĐ˝Ń†ĐµĐ˝Ń‚Ń€Đ°Ń†Đ¸ŃŽ (ĐĽĐś / (ĐĽĐşĐ/ŃĐĽÂ˛ Â· ĐĽŃ)).',

            # â”€â”€ ĐśĐµŃ‚Đ°Đ±ĐľĐ»Đ¸Đ·ĐĽ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            'enable_dynamic_atp': 'Đ’ĐşĐ»ŃŽŃ‡Đ¸Ń‚ŃŚ Đ´Đ¸Đ˝Đ°ĐĽĐ¸Ń‡ĐµŃĐşĐ¸Đą ĐżŃĐ» ĐĐ˘Đ¤. ĐŃŃ‚ĐľŃ‰ĐµĐ˝Đ¸Đµ ĐĐ˘Đ¤ ĐľŃ‚ĐşŃ€Ń‹Đ˛Đ°ĐµŃ‚ K_ATP ĐşĐ°Đ˝Đ°Đ»Ń‹, ĐłĐ¸ĐżĐµŃ€ĐżĐľĐ»ŃŹŃ€Đ¸Đ·ŃŃŹ ĐşĐ»ĐµŃ‚ĐşŃ.',
            'g_katp_max': 'ĐśĐ°ĐşŃ. ĐżŃ€ĐľĐ˛ĐľĐ´Đ¸ĐĽĐľŃŃ‚ŃŚ ATP-Ń‡ŃĐ˛ŃŃ‚Đ˛Đ¸Ń‚ĐµĐ»ŃŚĐ˝Ń‹Ń… Kâş ĐşĐ°Đ˝Đ°Đ»ĐľĐ˛ (ĐĽĐˇĐĽ/ŃĐĽÂ˛). ĐžŃ‚ĐşŃ€Ń‹Đ˛Đ°ŃŽŃ‚ŃŃŹ ĐżŃ€Đ¸ ĐĐ˘Đ¤ < 0.5 ĐĽĐś.',
            'katp_kd_atp_mM': 'ĐšĐľĐ˝Ń†ĐµĐ˝Ń‚Ń€Đ°Ń†Đ¸ŃŹ ĐĐ˘Đ¤ Đ´Đ»ŃŹ ĐżĐľĐ»ŃĐ°ĐşŃ‚Đ¸Đ˛Đ°Ń†Đ¸Đ¸ K_ATP (ĐĽĐś).',
            'atp_max_mM': 'Đ‘Đ°Đ·ĐľĐ˛Đ°ŃŹ Đ˛Đ˝ŃŃ‚Ń€Đ¸ĐşĐ»ĐµŃ‚ĐľŃ‡Đ˝Đ°ŃŹ ĐşĐľĐ˝Ń†ĐµĐ˝Ń‚Ń€Đ°Ń†Đ¸ŃŹ ĐĐ˘Đ¤ (ĐĽĐś). Đ—Đ´ĐľŃ€ĐľĐ˛Ń‹Đµ Đ˝ĐµĐąŃ€ĐľĐ˝Ń‹: 2-5 ĐĽĐś.',
            'atp_synthesis_rate': 'ĐˇĐşĐľŃ€ĐľŃŃ‚ŃŚ ŃĐ¸Đ˝Ń‚ĐµĐ·Đ° ĐĐ˘Đ¤ Ń‡ĐµŃ€ĐµĐ· ĐľĐşĐ¸ŃĐ»Đ¸Ń‚ĐµĐ»ŃŚĐ˝ĐľĐµ Ń„ĐľŃŃ„ĐľŃ€Đ¸Đ»Đ¸Ń€ĐľĐ˛Đ°Đ˝Đ¸Đµ (Đ˝ĐĽĐľĐ»ŃŚ/ŃĐĽÂ˛/Ń).',

            # â”€â”€ ĐˇŃ€ĐµĐ´Đ° â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            'T_celsius':     'Đ˘ĐµĐĽĐżĐµŃ€Đ°Ń‚ŃŃ€Đ° ŃŤĐşŃĐżĐµŃ€Đ¸ĐĽĐµĐ˝Ń‚Đ° (Â°C). ĐśĐ°ŃŃŃ‚Đ°Đ±Đ¸Ń€ŃĐµŃ‚ ĐşĐ¸Đ˝ĐµŃ‚Đ¸ĐşŃ Ń‡ĐµŃ€ĐµĐ· Q10.',
            'T_ref':         'Đ ĐµŃ„ĐµŃ€ĐµĐ˝ŃĐ˝Đ°ŃŹ Ń‚ĐµĐĽĐżĐµŃ€Đ°Ń‚ŃŃ€Đ° ĐşĐ°Đ˝Đ°Đ»ĐľĐ˛ (Â°C).',
            'Q10':           'ĐšĐľŃŤŃ„Ń„Đ¸Ń†Đ¸ĐµĐ˝Ń‚ Q10. ĐźŃ€Đ¸ Q10=3 ŃĐşĐľŃ€ĐľŃŃ‚ŃŚ Ń€Đ°ŃŃ‚Ń‘Ń‚ Đ˛ 3 Ń€Đ°Đ·Đ° Đ·Đ° 10Â°C.',
            'T_dend_offset': 'Đ Đ°Đ·Đ˝Đ¸Ń†Đ° Ń‚ĐµĐĽĐżĐµŃ€Đ°Ń‚ŃŃ€ Đ´ĐµĐ˝Đ´Ń€Đ¸Ń‚Ń‹â€“ŃĐľĐĽĐ° (Â°C). ĐźĐ»ŃŽŃ = Đ´ĐµĐ˝Đ´Ń€Đ¸Ń‚Ń‹ Ń‚ĐµĐżĐ»ĐµĐµ. Đ›Đ¸Đ˝ĐµĐąĐ˝Ń‹Đą Đ°ĐşŃĐ¸Đ°Đ»ŃŚĐ˝Ń‹Đą ĐłŃ€Đ°Đ´Đ¸ĐµĐ˝Ń‚.',

            # â”€â”€ ĐˇĐ¸ĐĽŃĐ»ŃŹŃ†Đ¸ŃŹ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            't_sim':       'Đ”Đ»Đ¸Ń‚ĐµĐ»ŃŚĐ˝ĐľŃŃ‚ŃŚ ŃĐ¸ĐĽŃĐ»ŃŹŃ†Đ¸Đ¸ (ĐĽŃ).',
            'dt_eval':     'Đ¨Đ°Đł Đ˛Ń‹Đ˛ĐľĐ´Đ° (ĐĽŃ). ĐťĐµ Đ˛Đ»Đ¸ŃŹĐµŃ‚ Đ˝Đ° Ń‚ĐľŃ‡Đ˝ĐľŃŃ‚ŃŚ Đ¸Đ˝Ń‚ĐµĐłŃ€Đ°Ń‚ĐľŃ€Đ°.',
            'stim_type':   'Đ˘Đ¸Đż ŃŃ‚Đ¸ĐĽŃĐ»Đ°: const / pulse / alpha / ŃŃĐĽ OU / ŃĐ¸Đ˝Đ°ĐżŃ‚Đ¸Ń‡ĐµŃĐşĐ¸Đµ Ń€ĐµŃ†ĐµĐżŃ‚ĐľŃ€Ń‹ / ZAP-Ń‡Đ¸Ń€Đż.',
            'Iext':        'ĐĐĽĐżĐ»Đ¸Ń‚ŃĐ´Đ° ŃŃ‚Đ¸ĐĽŃĐ»Đ° (ĐĽĐşĐ/ŃĐĽÂ˛).',
            'pulse_start': 'ĐťĐ°Ń‡Đ°Đ»Đľ Đ¸ĐĽĐżŃĐ»ŃŚŃĐ° (ĐĽŃ).',
            'pulse_dur':   'Đ”Đ»Đ¸Ń‚ĐµĐ»ŃŚĐ˝ĐľŃŃ‚ŃŚ Đ¸ĐĽĐżŃĐ»ŃŚŃĐ° (ĐĽŃ).',
            'alpha_tau':   'ĐźĐľŃŃ‚ĐľŃŹĐ˝Đ˝Đ°ŃŹ Đ˛Ń€ĐµĐĽĐµĐ˝Đ¸ alpha-ŃĐ¸Đ˝Đ°ĐżŃĐ° (ĐĽŃ). ĐźĐ¸Đş ĐżŃ€Đ¸ t = Ď„.',
            'zap_f0_hz':   'ĐťĐ°Ń‡Đ°Đ»ŃŚĐ˝Đ°ŃŹ Ń‡Đ°ŃŃ‚ĐľŃ‚Đ° ZAP-Ń‡Đ¸Ń€ĐżĐ° (Đ“Ń†).',
            'zap_f1_hz':   'ĐšĐľĐ˝ĐµŃ‡Đ˝Đ°ŃŹ Ń‡Đ°ŃŃ‚ĐľŃ‚Đ° ZAP-Ń‡Đ¸Ń€ĐżĐ° (Đ“Ń†).',
            'stim_comp':   'ĐšĐľĐĽĐżĐ°Ń€Ń‚ĐĽĐµĐ˝Ń‚ ŃŃ‚Đ¸ĐĽŃĐ»ŃŹŃ†Đ¸Đ¸ (0 = ŃĐľĐĽĐ°).',
            'event_times': 'ĐžŃ‡ĐµŃ€ĐµĐ´ŃŚ ŃĐ¸Đ˝Đ°ĐżŃ‚Đ¸Ń‡ĐµŃĐşĐ¸Ń… ŃĐľĐ±Ń‹Ń‚Đ¸Đą (ĐĽŃ). Đ’Ń€ĐµĐĽĐµĐ˝Đ˝Ń‹Đµ ĐĽĐµŃ‚ĐşĐ¸ Ń‡ĐµŃ€ĐµĐ· Đ·Đ°ĐżŃŹŃ‚ŃŃŽ: "10,25,40". Đ—Đ°ĐĽĐµŃ‰Đ°ĐµŃ‚ pulse_start Đ´Đ»ŃŹ AMPA/NMDA/GABA.',
            'synaptic_train_type': 'Đ“ĐµĐ˝ĐµŃ€Đ°Ń‚ĐľŃ€ ĐżĐľĐµĐ·Đ´ĐľĐş Đ¸ĐĽĐżŃĐ»ŃŚŃĐľĐ˛: none (ĐľŃ‚ŃŃŃ‚ŃŃ‚Đ˛ŃĐµŃ‚) / regular (Ń„Đ¸ĐşŃĐ¸Ń€ĐľĐ˛Đ°Đ˝Đ˝Ń‹Đą ĐĐźĐ’) / poisson (ŃĐ»ŃŃ‡Đ°ĐąĐ˝Ń‹Đą).',
            'synaptic_train_freq_hz': 'Đ§Đ°ŃŃ‚ĐľŃ‚Đ° ŃĐłĐµĐ˝ĐµŃ€Đ¸Ń€ĐľĐ˛Đ°Đ˝Đ˝ĐľĐłĐľ ĐżĐľĐµĐ·Đ´Đ° Đ¸ĐĽĐżŃĐ»ŃŚŃĐľĐ˛ (Đ“Ń†).',
            'synaptic_train_duration_ms': 'Đ”Đ»Đ¸Ń‚ĐµĐ»ŃŚĐ˝ĐľŃŃ‚ŃŚ ĐżĐľĐµĐ·Đ´Đ° Đ¸ĐĽĐżŃĐ»ŃŚŃĐľĐ˛ (ĐĽŃ).',
            'stoch_gating':'Шум Ланжевена в гейтах (Нативный Hines-солвер). Используйте кнопку СТОХАСТИКА.',
            'noise_sigma': 'ĐĐĽĐżĐ»Đ¸Ń‚ŃĐ´Đ° Đ±ĐµĐ»ĐľĐłĐľ ŃŃĐĽĐ° Đ˝Đ° ĐĽĐµĐĽĐ±Ń€Đ°Đ˝Ń Ď (ĐĽĐşĐ/ŃĐĽÂ˛).',

            # â”€â”€ ĐĐ˝Đ°Đ»Đ¸Đ· â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            'run_mc':       'ĐśĐľĐ˝Ń‚Đµ-ĐšĐ°Ń€Đ»Đľ: ĐżĐ°Ń€Đ°Đ»Đ»ĐµĐ»ŃŚĐ˝Ń‹Đµ Đ·Đ°ĐżŃŃĐşĐ¸ Ń Â±5% Ń€Đ°Đ·Đ±Ń€ĐľŃĐľĐĽ gNa/gK.',
            'mc_trials':    'Đ§Đ¸ŃĐ»Đľ ĐśĐš-ĐżĐľĐżŃ‹Ń‚ĐľĐş.',
            'run_bifurcation': 'Đ‘Đ¸Ń„ŃŃ€ĐşĐ°Ń†Đ¸ĐľĐ˝Đ˝Ń‹Đą Đ°Đ˝Đ°Đ»Đ¸Đ· ĐżĐľŃĐ»Đµ ĐľŃĐ˝ĐľĐ˛Đ˝ĐľĐłĐľ Đ·Đ°ĐżŃŃĐşĐ°.',
            'bif_param':    'ĐźĐ°Ń€Đ°ĐĽĐµŃ‚Ń€ Đ´Đ»ŃŹ Đ±Đ¸Ń„ŃŃ€ĐşĐ°Ń†Đ¸Đ¸.',
            'bif_min':      'ĐťĐ°Ń‡Đ°Đ»Đľ Đ´Đ¸Đ°ĐżĐ°Đ·ĐľĐ˝Đ° Đ±Đ¸Ń„ŃŃ€ĐşĐ°Ń†Đ¸Đ¸.',
            'bif_max':      'ĐšĐľĐ˝ĐµŃ† Đ´Đ¸Đ°ĐżĐ°Đ·ĐľĐ˝Đ° Đ±Đ¸Ń„ŃŃ€ĐşĐ°Ń†Đ¸Đ¸.',
            'bif_steps':    'ĐšĐľĐ»Đ¸Ń‡ĐµŃŃ‚Đ˛Đľ Ń‚ĐľŃ‡ĐµĐş Đ±Đ¸Ń„ŃŃ€ĐşĐ°Ń†Đ¸Đ¸.',
            'run_sweep':    'Đ’ĐşĐ»ŃŽŃ‡Đ¸Ń‚ŃŚ ĐżĐ°Ń€Đ°ĐĽĐµŃ‚Ń€Đ¸Ń‡ĐµŃĐşĐ¸Đą ĐżĐµŃ€ĐµĐ±ĐľŃ€ (ĐşĐ˝ĐľĐżĐşĐ° â†”).',
            'sweep_param':  'ĐźĐµŃ€ĐµĐ±Đ¸Ń€Đ°ĐµĐĽŃ‹Đą ĐżĐ°Ń€Đ°ĐĽĐµŃ‚Ń€: Iext, gNa_max, gK_max, T_celsius, â€¦',
            'sweep_min':    'ĐťĐ°Ń‡Đ°Đ»ŃŚĐ˝ĐľĐµ Đ·Đ˝Đ°Ń‡ĐµĐ˝Đ¸Đµ ĐżĐµŃ€ĐµĐ±ĐľŃ€Đ°.',
            'sweep_max':    'ĐšĐľĐ˝ĐµŃ‡Đ˝ĐľĐµ Đ·Đ˝Đ°Ń‡ĐµĐ˝Đ¸Đµ ĐżĐµŃ€ĐµĐ±ĐľŃ€Đ°.',
            'sweep_steps':  'ĐšĐľĐ»Đ¸Ń‡ĐµŃŃ‚Đ˛Đľ Ń‚ĐľŃ‡ĐµĐş ĐżĐµŃ€ĐµĐ±ĐľŃ€Đ°.',
            'run_sd_curve': 'ĐšŃ€Đ¸Đ˛Đ°ŃŹ ĐˇĐ¸Đ»Đ°-Đ”Đ»Đ¸Ń‚ĐµĐ»ŃŚĐ˝ĐľŃŃ‚ŃŚ (ĐşĐ˝ĐľĐżĐşĐ° âŹ±).',
            'run_excmap':   'ĐšĐ°Ń€Ń‚Đ° Đ˛ĐľĐ·Đ±ŃĐ´Đ¸ĐĽĐľŃŃ‚Đ¸ 2-D (ĐşĐ˝ĐľĐżĐşĐ° đź—ş).',
            'excmap_I_min': 'ĐšĐ°Ń€Ń‚Đ°: ĐĽĐ¸Đ˝. Ń‚ĐľĐş (ĐĽĐşĐ/ŃĐĽÂ˛).',
            'excmap_I_max': 'ĐšĐ°Ń€Ń‚Đ°: ĐĽĐ°ĐşŃ. Ń‚ĐľĐş (ĐĽĐşĐ/ŃĐĽÂ˛).',
            'excmap_NI':    'ĐšĐ°Ń€Ń‚Đ°: Ń‡Đ¸ŃĐ»Đľ ŃĐ°ĐłĐľĐ˛ ĐżĐľ Ń‚ĐľĐşŃ.',
            'excmap_D_min': 'ĐšĐ°Ń€Ń‚Đ°: ĐĽĐ¸Đ˝. Đ´Đ»Đ¸Ń‚ĐµĐ»ŃŚĐ˝ĐľŃŃ‚ŃŚ (ĐĽŃ).',
            'excmap_D_max': 'ĐšĐ°Ń€Ń‚Đ°: ĐĽĐ°ĐşŃ. Đ´Đ»Đ¸Ń‚ĐµĐ»ŃŚĐ˝ĐľŃŃ‚ŃŚ (ĐĽŃ).',
            'excmap_ND':    'ĐšĐ°Ń€Ń‚Đ°: Ń‡Đ¸ŃĐ»Đľ ŃĐ°ĐłĐľĐ˛ ĐżĐľ Đ´Đ»Đ¸Ń‚ĐµĐ»ŃŚĐ˝ĐľŃŃ‚Đ¸.',

            # â”€â”€ ĐśĐµŃ‚ĐşĐ¸ Ń€ĐµĐ¶Đ¸ĐĽĐľĐ˛ ĐżŃ€ĐµŃĐµŃ‚ĐľĐ˛ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            'k_mode_sleep': 'ĐˇĐľĐ˝ (Đ’ŃĐżĐ»ĐµŃĐşĐ¸)',
            'k_mode_awake': 'Đ‘ĐľĐ´Ń€ŃŃ‚Đ˛ĐľĐ˛Đ°Đ˝Đ¸Đµ (Đ ĐµĐ»Đµ)',
            'l5_mode_normal': 'ĐˇŃ‚Đ°Đ˝Đ´Đ°Ń€Ń‚',
            'l5_mode_ach': 'Đ’Đ˝Đ¸ĐĽĐ°Đ˝Đ¸Đµ (ĐCh Đ’ĐšĐ›)',

            # â”€â”€ ĐźĐµŃ€ĐµĐ˛ĐľĐ´ ĐľŃĐ˝ĐľĐ˛Đ˝Ń‹Ń… Ń„ŃĐ˝ĐşŃ†Đ¸Đą â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            'func_gax': 'ĐĐşŃĐ¸Đ°Đ»ŃŚĐ˝Đ°ŃŹ ĐżŃ€ĐľĐ˛ĐľĐ´Đ¸ĐĽĐľŃŃ‚ŃŚ ĐľĐ´Đ˝ĐľŃ€ĐľĐ´Đ˝ĐľĐłĐľ Ń†Đ¸Đ»Đ¸Đ˝Đ´Ń€Đ° (ĐĽĐˇĐĽ).',
            'func_gax_pair': 'ĐźŃ€ĐľĐ˛ĐľĐ´Đ¸ĐĽĐľŃŃ‚ŃŚ Đ˝Đ° ŃŃ‚Ń‹ĐşĐµ Đ´Đ˛ŃŃ… ĐşĐľĐĽĐżĐ°Ń€Ń‚ĐĽĐµĐ˝Ń‚ĐľĐ˛ Ń€Đ°Đ·Đ˝ĐľĐłĐľ Đ´Đ¸Đ°ĐĽĐµŃ‚Ń€Đ° (ŃĐľĐĽĐ°->Đ°ĐşŃĐľĐ˝).',
            'func_add_link': 'Đ”ĐľĐ±Đ°Đ˛Đ»ŃŹĐµŃ‚ ŃĐ˛ŃŹĐ·ŃŚ ĐĽĐµĐ¶Đ´Ń ĐşĐľĐĽĐżĐ°Ń€Ń‚ĐĽĐµĐ˝Ń‚Đ°ĐĽĐ¸ i Đ¸ j Ń ĐżŃ€ĐľĐ˛ĐľĐ´Đ¸ĐĽĐľŃŃ‚ŃŚŃŽ g.',
            'class_morphology_builder': 'ĐšĐ»Đ°ŃŃ Đ´Đ»ŃŹ ŃĐ±ĐľŃ€ĐşĐ¸ ĐłĐµĐľĐĽĐµŃ‚Ń€Đ¸Đ¸ Đ˝ĐµĐąŃ€ĐľĐ˝Đ°, ĐżĐ»ĐľŃ‰Đ°Đ´ĐµĐą ĐĽĐµĐĽĐ±Ń€Đ°Đ˝Ń‹ Đ¸ Đ›Đ°ĐżĐ»Đ°ŃĐ¸Đ°Đ˝Đ°.',
            'func_nernst_ca': 'Đ”Đ¸Đ˝Đ°ĐĽĐ¸Ń‡ĐµŃĐşĐ¸Đą ĐżĐľŃ‚ĐµĐ˝Ń†Đ¸Đ°Đ» ĐťĐµŃ€Đ˝ŃŃ‚Đ° Đ´Đ»ŃŹ ĐšĐ°Đ»ŃŚŃ†Đ¸ŃŹ (z=2).',
            'func_get_stim_current': 'ĐśĐ°Ń‚ĐµĐĽĐ°Ń‚Đ¸ĐşĐ° Đ˛ŃĐµŃ… Ń‚Đ¸ĐżĐľĐ˛ ŃŃ‚Đ¸ĐĽŃĐ»ĐľĐ˛ v10.',
            'func_am': 'ĐĐşŃ‚Đ¸Đ˛Đ°Ń†Đ¸ŃŹ Na (m). ĐžĐżŃ€ĐµĐ´ĐµĐ»ŃŹĐµŃ‚ Ń€ĐµĐ·ĐşĐ¸Đą Ń„Ń€ĐľĐ˝Ń‚ ĐżĐľŃ‚ĐµĐ˝Ń†Đ¸Đ°Đ»Đ° Đ´ĐµĐąŃŃ‚Đ˛Đ¸ŃŹ.',
            'func_bm': 'Đ”ĐµĐ°ĐşŃ‚Đ¸Đ˛Đ°Ń†Đ¸ŃŹ Na (m).',
            'func_ah': 'ĐˇĐ˝ŃŹŃ‚Đ¸Đµ Đ¸Đ˝Đ°ĐşŃ‚Đ¸Đ˛Đ°Ń†Đ¸Đ¸ Na (h).',
            'func_bh': 'ĐĐ˝Đ°ĐşŃ‚Đ¸Đ˛Đ°Ń†Đ¸ŃŹ Na (h). ĐśĐµĐ´Đ»ĐµĐ˝Đ˝ĐµĐµ m, Đ·Đ°Đ˛ĐµŃ€ŃĐ°ĐµŃ‚ ŃĐżĐ°ĐąĐş.',
            'func_an': 'ĐĐşŃ‚Đ¸Đ˛Đ°Ń†Đ¸ŃŹ K (n). ĐśĐµĐ´Đ»ĐµĐ˝Đ˝Ń‹Đą Ń‚ĐľĐş Ń€ĐµĐżĐľĐ»ŃŹŃ€Đ¸Đ·Đ°Ń†Đ¸Đ¸.',
            'func_bn': 'Đ”ĐµĐ°ĐşŃ‚Đ¸Đ˛Đ°Ń†Đ¸ŃŹ K (n).',
            'class_channel_registry': 'ĐžĐžĐź-Ń€ĐµĐµŃŃ‚Ń€ ĐşĐ°Đ˝Đ°Đ»ĐľĐ˛ Đ´Đ»ŃŹ Đ˛Đ·Đ°Đ¸ĐĽĐľĐ´ĐµĐąŃŃ‚Đ˛Đ¸ŃŹ Ń GUI Đ¸ ŃĐ±ĐľŃ€ĐşĐ¸ Đ˛ĐµĐşŃ‚ĐľŃ€Đ° ŃĐľŃŃ‚ĐľŃŹĐ˝Đ¸Đą.',
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
                "I: In Vitro Slice (Mammalian 23Â°C)": "I: In Vitro Slice (Mammalian 23Â°C)",
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
                "A: Squid Giant Axon (HH 1952)": "A: Đ“Đ¸ĐłĐ°Đ˝Ń‚ŃĐşĐ¸Đą Đ°ĐşŃĐľĐ˝ ĐşĐ°Đ»ŃŚĐĽĐ°Ń€Đ° (ĐĄĐĄ 1952)",
                "B: Pyramidal L5 (Mainen 1996)": "B: ĐźĐ¸Ń€Đ°ĐĽĐ¸Đ´Đ˝Ń‹Đą L5 (Mainen 1996)",
                "C: FS Interneuron (Wang-Buzsaki)": "C: Đ‘Ń‹ŃŃ‚Ń€Ń‹Đą Đ¸Đ˝Ń‚ĐµŃ€Đ˝ĐµĐąŃ€ĐľĐ˝ (Wang-Buzsaki)",
                "D: alpha-Motoneuron (Powers 2001)": "D: ĐĐ»ŃŚŃ„Đ°-ĐĽĐľŃ‚ĐľĐ˝ĐµĐąŃ€ĐľĐ˝ (Powers 2001)",
                "E: Cerebellar Purkinje (De Schutter)": "E: ĐšĐ»ĐµŃ‚ĐşĐ° ĐźŃŃ€ĐşĐ¸Đ˝ŃŚĐµ ĐĽĐľĐ·Đ¶ĐµŃ‡ĐşĐ° (De Schutter)",
                "F: Multiple Sclerosis (Demyelination)": "F: Đ Đ°ŃŃĐµŃŹĐ˝Đ˝Ń‹Đą ŃĐşĐ»ĐµŃ€ĐľĐ· (Đ´ĐµĐĽĐ¸ĐµĐ»Đ¸Đ˝Đ¸Đ·Đ°Ń†Đ¸ŃŹ)",
                "G: Local Anesthesia (gNa Block)": "G: ĐśĐµŃŃ‚Đ˝Đ°ŃŹ Đ°Đ˝ĐµŃŃ‚ĐµĐ·Đ¸ŃŹ (Đ±Đ»ĐľĐşĐ°Đ´Đ° gNa)",
                "H: Severe Hyperkalemia (High EK)": "H: Đ˘ŃŹĐ¶ĐµĐ»Đ°ŃŹ ĐłĐ¸ĐżĐµŃ€ĐşĐ°Đ»Đ¸ĐµĐĽĐ¸ŃŹ (Đ˛Ń‹ŃĐľĐşĐ¸Đą EK)",
                "I: In Vitro Slice (Mammalian 23Â°C)": "I: ĐˇŃ€ĐµĐ· in vitro (ĐĽĐ»ĐµĐşĐľĐżĐ¸Ń‚Đ°ŃŽŃ‰Đ¸Đµ 23Â°C)",
                "J: C-Fiber (Pain / Unmyelinated)": "J: C-Đ˛ĐľĐ»ĐľĐşĐ˝Đľ (Đ±ĐľĐ»ŃŚ / Đ±ĐµĐ·ĐĽĐ¸ĐµĐ»Đ¸Đ˝ĐľĐ˛ĐľĐµ)",
                "K: Thalamic Relay (Ih + ITCa + Burst)": "K: Đ˘Đ°Đ»Đ°ĐĽĐ¸Ń‡ĐµŃĐşĐľĐµ Ń€ĐµĐ»Đµ (Ih + ITCa + ĐżĐ°Ń‡ĐşĐ¸)",
                "L: Hippocampal CA1 Pyramidal (Adapting)": "L: Đ“Đ¸ĐżĐżĐľĐşĐ°ĐĽĐż CA1 ĐżĐ¸Ń€Đ°ĐĽĐ¸Đ´Đ˝Ń‹Đą (Đ°Đ´Đ°ĐżŃ‚Đ¸Đ˛Đ˝Ń‹Đą)",
                "M: Epilepsy (v10 SCN1A mutation)": "M: Đ­ĐżĐ¸Đ»ĐµĐżŃĐ¸ŃŹ (ĐĽŃŃ‚Đ°Ń†Đ¸ŃŹ SCN1A v10)",
                "N: Alzheimer's (v10 Calcium Toxicity)": "N: Đ‘ĐľĐ»ĐµĐ·Đ˝ŃŚ ĐĐ»ŃŚŃ†ĐłĐµĐąĐĽĐµŃ€Đ° (Ń‚ĐľĐşŃĐ¸Ń‡Đ˝ĐľŃŃ‚ŃŚ ĐşĐ°Đ»ŃŚŃ†Đ¸ŃŹ v10)",
                "O: Hypoxia (v10 ATP-pump failure)": "O: Đ“Đ¸ĐżĐľĐşŃĐ¸ŃŹ (ĐľŃ‚ĐşĐ°Đ· ĐĐ˘Đ¤-Đ˝Đ°ŃĐľŃĐ° v10)",
                "P: Thalamic Reticular Nucleus (TRN Spindles)": "P: Đ ĐµŃ‚Đ¸ĐşŃĐ»ŃŹŃ€Đ˝ĐľĐµ ŃŹĐ´Ń€Đľ Ń‚Đ°Đ»Đ°ĐĽŃŃĐ° (TRN Đ˛ĐµŃ€ĐµŃ‚Ń‘Đ˝Đ°)",
                "Q: Striatal Spiny Projection (SPN)": "Q: Đ¨Đ¸ĐżĐľĐ˛Đ°Ń‚Ń‹Đą ĐżŃ€ĐľĐµĐşŃ†Đ¸ĐľĐ˝Đ˝Ń‹Đą Đ˝ĐµĐąŃ€ĐľĐ˝ (SPN)",
                "R: Cholinergic Neuromodulation (ACh)": "R: ĐĄĐľĐ»Đ¸Đ˝ĐµŃ€ĐłĐ¸Ń‡ĐµŃĐşĐ°ŃŹ Đ˝ĐµĐąŃ€ĐľĐĽĐľĐ´ŃĐ»ŃŹŃ†Đ¸ŃŹ (ACh)",
                "S: Pathology: Dravet Syndrome (SCN1A LOF)": "S: ĐźĐ°Ń‚ĐľĐ»ĐľĐłĐ¸ŃŹ: ĐˇĐ¸Đ˝Đ´Ń€ĐľĐĽ Đ”Ń€Đ°Đ˛Đµ (SCN1A LOF)",
            }
        }
        return repair_text(preset_translations.get(self.lang, {}).get(preset_name, preset_name))


# Global translator instance â€” default English
T = Translator('EN')
