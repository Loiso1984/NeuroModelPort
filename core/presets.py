from core.models import FullModelConfig
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# UNIT CONVERSION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _soma_area_cm2(d_soma_cm: float) -> float:
    """Calculate soma surface area (sphere) from diameter in cm."""
    return np.pi * d_soma_cm ** 2

def _iext_density_to_absolute(iext_density_uA_per_cm2: float, d_soma_cm: float) -> float:
    """
    Convert current density (µA/cm²) to absolute current (nanoamperes).
    
    Formula: I_abs [nA] = I_density [µA/cm²] × A_soma [cm²] × 1000 [nA/µA]
    
    After morphology building, conductances are multiplied by area:
    gX_total [mS] = gX_density [mS/cm²] × A_soma [cm²]
    
    So Iext should also be in absolute current (after scaling by area).
    But we want to display as nanoamperes for user readability.
    
    Returns:
    --------
    Absolute injected current in nanoamperes (nA)
    """
    area = _soma_area_cm2(d_soma_cm)
    return iext_density_uA_per_cm2 * area * 1000.0  # Convert µA → nA

def get_preset_names():
    """Возвращает полный список доступных научных сценариев v10.0."""
    return [
        "A: Squid Giant Axon (HH 1952)",
        "B: Pyramidal L5 (Mainen 1996)",
        "C: FS Interneuron (Wang-Buzsaki)",
        "D: alpha-Motoneuron (Powers 2001)",
        "E: Cerebellar Purkinje (De Schutter)",
        "F: Multiple Sclerosis (Demyelination)",
        "G: Local Anesthesia (gNa Block)",
        "H: Severe Hyperkalemia (High EK)",
        "I: In Vitro Slice (Mammalian 23°C)",
        "J: C-Fiber (Pain / Unmyelinated)",
        "K: Thalamic Relay (Ih + ICa + Burst)",
        "L: Hippocampal CA1 (Theta rhythm)",
        "M: Epilepsy (v10 SCN1A mutation)",
        "N: Alzheimer's (v10 Calcium Toxicity)",
        "O: Hypoxia (v10 ATP-pump failure)",
    ]

def _reset_cfg_to_defaults(cfg: FullModelConfig) -> None:
    """
    Reset every sub-model field to its Pydantic default value.

    Called at the start of apply_preset() so that switching between presets
    never leaves stale values from a previously loaded preset.
    Object identity is preserved (GUI form references remain valid).
    """
    from core.models import (
        MorphologyParams, ChannelParams, CalciumParams,
        EnvironmentParams, SimulationParams,
        StimulationLocationParams, DendriticFilterParams, AnalysisParams,
        PresetModeParams,
    )
    _copy_defaults(cfg.morphology,        MorphologyParams())
    _copy_defaults(cfg.channels,          ChannelParams())
    _copy_defaults(cfg.calcium,           CalciumParams())
    _copy_defaults(cfg.env,               EnvironmentParams())
    _copy_defaults(cfg.stim,              SimulationParams())
    _copy_defaults(cfg.stim_location,     StimulationLocationParams())
    _copy_defaults(cfg.dendritic_filter,  DendriticFilterParams())
    _copy_defaults(cfg.analysis,          AnalysisParams())
    _copy_defaults(cfg.preset_modes,      PresetModeParams())


def _copy_defaults(target, source) -> None:
    """Copy every field from *source* (a fresh default instance) to *target*."""
    for field_name in type(source).model_fields:
        setattr(target, field_name, getattr(source, field_name))


def _apply_k_mode(cfg: FullModelConfig) -> None:
    """Apply thalamic relay mode variants."""
    if cfg.preset_modes.k_mode == "baseline":
        # Baseline: low-throughput relay mode with theta-like global rate envelope.
        cfg.stim.stim_type = "alpha"
        cfg.stim.alpha_tau = 8.0
        cfg.stim.Iext = 24.0
        cfg.channels.gIh_max = 0.02
        cfg.channels.gCa_max = 0.06
    else:
        # Activated: task-driven relay state with stronger throughput.
        cfg.stim.stim_type = "const"
        cfg.stim.Iext = 30.0
        cfg.channels.gIh_max = 0.03
        cfg.channels.gCa_max = 0.08


def _apply_alzheimer_mode(cfg: FullModelConfig) -> None:
    """Apply Alzheimer's stage variants."""
    if cfg.preset_modes.alzheimer_mode == "terminal":
        # Terminal stage: severe network failure, near-silent response.
        cfg.calcium.tau_Ca = 1200.0
        cfg.channels.gSK_max = 2.2
        cfg.channels.gCa_max = 0.06
        cfg.stim.stim_type = "alpha"
        cfg.stim.alpha_tau = 2.0
        cfg.stim.Iext = 15.0
    else:
        # Progressive stage: initial spiking followed by adaptation/decline.
        cfg.calcium.tau_Ca = 800.0
        cfg.channels.gSK_max = 1.5
        cfg.channels.gCa_max = 0.08
        cfg.stim.stim_type = "const"
        cfg.stim.Iext = 18.0


def _apply_hypoxia_mode(cfg: FullModelConfig) -> None:
    """Apply hypoxia stage variants."""
    if cfg.preset_modes.hypoxia_mode == "terminal":
        # Terminal stage: profound depolarization block and pump collapse.
        cfg.channels.EK = -45.0
        cfg.channels.EL = -40.0
        cfg.channels.gL = 0.4
        cfg.calcium.tau_Ca = 1200.0
        cfg.channels.gCa_max = 0.10
        cfg.stim.stim_type = "const"
        cfg.stim.Iext = 80.0
    else:
        # Progressive stage: short early spiking epoch, then attenuation.
        cfg.channels.EK = -60.0
        cfg.channels.EL = -50.0
        cfg.channels.gL = 0.15
        cfg.calcium.tau_Ca = 900.0
        cfg.channels.gCa_max = 0.08
        cfg.stim.stim_type = "const"
        cfg.stim.Iext = 30.0


def apply_preset(cfg: FullModelConfig, name: str):
    """Применяет полные наборы параметров (Мембрана + Морфология).

    Полностью сбрасывает все поля до значений по умолчанию перед применением
    пресета, предотвращая смешивание параметров между пресетами.
    """
    # Keep user-selected preset modes across preset reloads.
    selected_modes = cfg.preset_modes.model_copy(deep=True)

    # ПОЛНЫЙ СБРОС: все поля возвращаются к умолчаниям Pydantic
    _reset_cfg_to_defaults(cfg)
    _copy_defaults(cfg.preset_modes, selected_modes)
    
    # --- 1. КЛАССИКА: ГИГАНТСКИЙ АКСОН КАЛЬМАРА ---
    if "Squid" in name:
        cfg.dendritic_filter.enabled = False  # Squid axon has no dendrites
        cfg.channels.gNa_max, cfg.channels.gK_max, cfg.channels.gL = 120.0, 36.0, 0.3
        cfg.channels.ENa, cfg.channels.EK, cfg.channels.EL = 50.0, -77.0, -54.387
        cfg.channels.Cm = 1.0
        cfg.env.T_celsius, cfg.env.T_ref, cfg.env.Q10 = 6.3, 6.3, 3.0
        # Space-clamped preparation (HH 1952 original): single-compartment
        cfg.morphology.single_comp = True
        cfg.morphology.d_soma = 500e-4  # 500 µm giant axon diameter
        cfg.stim.Iext = 10.0  # Classic HH: ~68 Hz tonic firing, Vmax ≈ 40 mV

    # --- 2. БАЗОВЫЙ ПРЕСЕТ: ПИРАМИДНЫЙ L5 (Млекопитающие) ---
    elif "Pyramidal L5" in name:
        cfg.morphology.single_comp = False
        cfg.stim_location.location = "dendritic_filtered"
        cfg.dendritic_filter.enabled = True
        cfg.dendritic_filter.distance_um = 150.0
        cfg.dendritic_filter.space_constant_um = 150.0
        cfg.dendritic_filter.tau_dendritic_ms = 10.0
        
        # CONDUCTANCE DENSITY (mS/cm²) - SIZE INDEPENDENT
        # These values are SOMA-SPECIFIC from Mainen et al. 1996 calibration
        # Reference: Mainen & Sejnowski (1996) Nature 382: 363-366
        # Whole-cell recordings at 23°C on layer 5 pyramidal neurons
        cfg.channels.gNa_max = 56.0    # SOMA density [mS/cm²] - NOT soma+AIS average
        cfg.channels.gK_max = 6.0      # SOMA density [mS/cm²]
        cfg.channels.gL = 0.02         # SOMA leak density [mS/cm²]
        cfg.channels.Cm = 0.75         # Specific membrane capacitance [µF/cm²]
        
        cfg.channels.ENa, cfg.channels.EK, cfg.channels.EL = 50.0, -90.0, -70.0
        cfg.env.T_celsius, cfg.env.T_ref, cfg.env.Q10 = 37.0, 23.0, 2.3
        
        # MORPHOLOGY: SIZE-INDEPENDENT (DENSITY-BASED)
        # These will scale automatically with soma area - presets remain valid
        # for any soma diameter in physiological range (15-30 µm)
        cfg.morphology.d_soma = 20e-4  # Mainen et al 1996: ~20 µm ± 2 µm
        cfg.morphology.N_ais = 3       # Axon initial segment compartments
        cfg.morphology.gNa_ais_mult = 40.0  # AIS has 40× higher Na than soma
        cfg.morphology.gK_ais_mult = 5.0    # AIS has 5× higher K than soma
        cfg.morphology.d_trunk = 2e-4  # Dendrite trunk diameter
        cfg.morphology.Ra = 150.0      # Axial resistance [Ω·cm]
        
        cfg.channels.enable_SK = False  # Disabled for development, enable later with Ca-dyn
        cfg.calcium.dynamic_Ca = False  # Disabled for development, enable with SK+ICa
        cfg.channels.enable_ICa = False  # Disabled for development
        
        # CALIBRATED: Literature-based rheobase (Mainen 1996)
        # Soma mode with const stimulus: ~6 µA/cm² rheobase range
        cfg.stim.stim_type = 'const'
        cfg.stim.alpha_tau = 2.0
        cfg.stim.Iext = 6.0

    # --- 3. БЫСТРЫЙ ИНТЕРНЕЙРОН (FS) ---
    elif "FS Interneuron" in name:
        cfg.stim_location.location = "dendritic_filtered"
        cfg.dendritic_filter.enabled = True
        cfg.dendritic_filter.distance_um = 75.0
        cfg.dendritic_filter.space_constant_um = 100.0
        cfg.dendritic_filter.tau_dendritic_ms = 5.0
        # HH-standard conductances needed for full-amplitude spikes with HH kinetics at 37°C
        # Wang-Buzsaki kinetics differ from HH 1952; we compensate with standard gNa/gK
        cfg.channels.gNa_max, cfg.channels.gK_max, cfg.channels.gL = 120.0, 36.0, 0.1
        cfg.channels.ENa, cfg.channels.EK, cfg.channels.EL = 55.0, -90.0, -65.0
        cfg.channels.Cm = 1.0
        cfg.env.T_celsius, cfg.env.T_ref, cfg.env.Q10 = 37.0, 23.0, 2.3
        cfg.morphology.d_soma = 15e-4
        # IA channel: High for fast repolarization and strong adaptation
        cfg.channels.enable_IA = True
        cfg.channels.gA_max = 0.8  # High IA conductance for rapid spike termination
        cfg.channels.E_A = -77.0   # K+ reversal potential
        # Validated: 40 µA/cm² through dendritic filter (atten=0.47) → ~19 effective
        # Produces ~215 Hz tonic firing, Vmax ≈ 37 mV (physiological FS range 150-400 Hz)
        cfg.stim.stim_type = 'const'
        cfg.stim.alpha_tau = 1.0
        cfg.stim.Iext = 40.0

    # --- 4. МОТОНЕЙРОН СПИННОГО МОЗГА ---
    elif "alpha-Motoneuron" in name:
        cfg.morphology.single_comp = False
        cfg.stim_location.location = "dendritic_filtered"
        cfg.channels.gNa_max, cfg.channels.gK_max, cfg.channels.gL = 100.0, 30.0, 0.3
        cfg.channels.ENa, cfg.channels.EK, cfg.channels.EL = 50.0, -77.0, -60.0
        cfg.channels.Cm = 1.5
        cfg.env.T_celsius, cfg.env.T_ref, cfg.env.Q10 = 37.0, 23.0, 2.3
        cfg.morphology.d_soma = 60e-4
        cfg.morphology.N_ais = 5
        cfg.morphology.Ra = 70.0
        # IA channel: Moderate for frequency adaptation and spike-frequency accommodation
        cfg.channels.enable_IA = True
        cfg.channels.gA_max = 0.25  # Moderate IA for adaptation dynamics
        cfg.channels.E_A = -77.0    # K+ reversal potential
        # Alpha stimulus: represents one synaptic volley from descending pathways
        # Multi-comp with AIS produces burst of ~16 spikes, Vmax ≈ 34 mV
        cfg.stim.stim_type = 'alpha'
        cfg.stim.alpha_tau = 1.5
        cfg.stim.Iext = 50.0

    # --- 5. КЛЕТКА ПУРКИНЬЕ (МОЗЖЕЧОК) ---
    elif "Purkinje" in name:
        cfg.morphology.single_comp = False
        cfg.stim_location.location = "dendritic_filtered"
        cfg.dendritic_filter.enabled = True
        cfg.dendritic_filter.distance_um = 200.0
        cfg.dendritic_filter.space_constant_um = 180.0
        cfg.dendritic_filter.tau_dendritic_ms = 15.0
        cfg.channels.gNa_max, cfg.channels.gK_max, cfg.channels.gL = 75.0, 20.0, 0.05
        cfg.channels.ENa, cfg.channels.EK, cfg.channels.EL = 45.0, -85.0, -68.0
        cfg.env.T_celsius, cfg.env.T_ref, cfg.env.Q10 = 37.0, 22.0, 2.3
        cfg.morphology.d_soma = 25e-4
        cfg.morphology.N_ais = 3
        cfg.morphology.gNa_ais_mult = 30.0
        cfg.channels.enable_SK = False  # SK disabled - too much hyperpolarization with calcium
        cfg.calcium.dynamic_Ca = True
        cfg.calcium.tau_Ca = 150.0  # Fast extrusion (Purkinje: excellent calcium buffering)
        cfg.calcium.B_Ca = 1e-5  # Calibrated conversion: keeps spike-driven Ca transients in physiological nM-µM range
        cfg.channels.enable_ICa = True
        cfg.channels.gCa_max = 0.08  # Physiological L-type calcium conductance
        cfg.channels.gSK_max = 0.5
        # IA channel: High for complex spike dynamics and dendritic integration
        cfg.channels.enable_IA = True
        cfg.channels.gA_max = 0.4   # IA for complex spike dynamics
        cfg.channels.E_A = -77.0    # K+ reversal potential
        # Tuned to preserve tonic Purkinje spiking and avoid a silent island
        # at moderate low-drive screening points in the branch validation contour.
        cfg.stim.stim_type = 'const'
        cfg.stim.alpha_tau = 2.0
        cfg.stim.Iext = 32.0

    # --- 6. ТАЛАМИЧЕСКИЙ РЕЛЕ-НЕЙРОН (Ih + ICa + Ca-dynamics) ---
    elif "Thalamic" in name:
        # Single-compartment mode improves stable resting dynamics for this
        # reduced relay model with Ih/ICa without introducing AIS-driven artifacts.
        cfg.morphology.single_comp = True
        cfg.stim_location.location = "dendritic_filtered"
        cfg.dendritic_filter.enabled = True
        cfg.dendritic_filter.distance_um = 180.0
        cfg.dendritic_filter.space_constant_um = 120.0
        cfg.dendritic_filter.tau_dendritic_ms = 12.0
        # gNa=100 needed for full spikes through dendritic filter at 37°C
        cfg.channels.gNa_max, cfg.channels.gK_max, cfg.channels.gL = 100.0, 10.0, 0.05
        cfg.channels.ENa, cfg.channels.EK, cfg.channels.EL = 50.0, -90.0, -70.0
        cfg.env.T_celsius, cfg.env.T_ref, cfg.env.Q10 = 37.0, 23.0, 2.3
        cfg.channels.enable_Ih = True
        cfg.channels.gIh_max = 0.03
        cfg.channels.enable_ICa = True
        cfg.channels.gCa_max = 0.08  # Physiological L-type calcium conductance
        cfg.channels.enable_SK = False  # Disable SK for now
        cfg.calcium.dynamic_Ca = True
        cfg.calcium.tau_Ca = 200.0
        cfg.calcium.B_Ca = 1e-5  # Calibrated conversion: avoids unphysiological Ca overload
        cfg.morphology.d_soma = 25e-4
        cfg.stim.jacobian_mode = 'sparse_fd'
        # Validated: 30 µA/cm² through dend filter (atten=0.22) → ~6.7 effective
        # Produces ~21 spikes, 144 Hz, Vmax ≈ 41 mV with Ih + ICa + Ca dynamics
        cfg.stim.stim_type = 'const'
        cfg.stim.alpha_tau = 5.0
        cfg.stim.Iext = 30.0

    # --- 7. ПАТОЛОГИЯ: РАССЕЯННЫЙ СКЛЕРОЗ (Демиелинизация) ---
    elif "Multiple Sclerosis" in name:
        apply_preset(cfg, "alpha-Motoneuron (Powers 2001)")
        # Stronger axial resistance increase to emphasize impaired conduction.
        cfg.morphology.Ra = 450.0  # Demyelination: 70 → 450 (axial resistance ↑)
        cfg.channels.gL = 1.2  # Increased leak: 0.3 → 1.2 (exposed membrane)
        cfg.stim.jacobian_mode = 'sparse_fd'
        # Pathology signature: reduced spike amplitude (~21 vs ~34 mV) due to leak shunt
        cfg.stim.stim_type = 'alpha'
        cfg.stim.alpha_tau = 1.5
        cfg.stim.Iext = 50.0  # Same as base motoneuron

    # --- 8. ПАТОЛОГИЯ: ЭПИЛЕПСИЯ (SCN1A GAIN-OF-FUNCTION) ---
    elif "Epilepsy" in name:
        apply_preset(cfg, "FS Interneuron (Wang-Buzsaki)")
        cfg.channels.gNa_max = 200.0  # Gain-of-function: 120 → 200 (SCN1A mutation)
        cfg.channels.enable_IA = False
        # Enable calcium channels for pathological calcium dynamics
        cfg.channels.enable_ICa = True  # L-type calcium channel
        cfg.channels.gCa_max = 0.08  # Moderate ICa - PHYSIOLOGICAL range (was 1.2, too high)
        cfg.calcium.dynamic_Ca = True  # Enable calcium dynamics
        cfg.calcium.tau_Ca = 300.0  # Moderately impaired clearance
        cfg.calcium.B_Ca = 1e-5  # Calibrated conversion for pathological accumulation without runaway Ca
        # Const stim to show sustained hyperexcitability
        # Validated: ~29 spikes, 197 Hz, Vmax ≈ 48 mV (higher amp than FS base)
        cfg.stim.stim_type = 'const'
        cfg.stim.alpha_tau = 1.0
        cfg.stim.Iext = 30.0

    # --- 9. ПАТОЛОГИЯ: АЛЬЦГЕЙМЕР (CALCIUM TOXICITY - SLOW EXTRUSION) ---
    elif "Alzheimer's" in name:
        apply_preset(cfg, "Pyramidal L5 (Mainen 1996)")
        # CALCIUM DYNAMICS: Model pathological calcium accumulation
        cfg.calcium.dynamic_Ca = True  # ENABLED: Model impaired clearance
        cfg.calcium.tau_Ca = 800.0  # Slow pump: 200 → 800ms (impaired Ca clearance)
        cfg.calcium.B_Ca = 1e-5  # Calibrated conversion for slow-clearance pathology
        # Enable calcium-dependent adaptive current
        cfg.channels.enable_ICa = True  # L-type calcium channel
        cfg.channels.gCa_max = 0.08  # Moderate ICa - PHYSIOLOGICAL range (was 0.8, too high)
        cfg.channels.enable_SK = True  # SK channel activated by calcium
        cfg.channels.gSK_max = 1.5  # Stronger SK for calcium-dependent adaptation
        cfg.stim.jacobian_mode = 'sparse_fd'
        # Alpha stim: Shows reduced firing due to SK activation by calcium
        cfg.stim.stim_type = 'alpha'
        cfg.stim.alpha_tau = 2.0
        cfg.stim.Iext = 25.0  # Increased to compensate for SK hyperpolarization

    # --- 10. ПАТОЛОГИЯ: ГИПОКСИЯ (ATP-PUMP FAILURE - ION IMBALANCE + Ca overload) ---
    elif "Hypoxia" in name:
        apply_preset(cfg, "FS Interneuron (Wang-Buzsaki)")
        # Ion imbalance from pump failure
        cfg.channels.EK = -50.0  # Reduced: -90 → -50 (K+ accumulation)
        cfg.channels.EL = -45.0  # Reduced: -65 → -45 (Na+ accumulation)
        # CALCIUM OVERLOAD: Hypoxia causes Ca2+ accumulation and mitochondrial dysfunction
        cfg.calcium.dynamic_Ca = True  # ENABLED: Model calcium overload
        cfg.calcium.tau_Ca = 900.0  # Slow clearance: pump failure (fixed from 1500ms)
        cfg.calcium.B_Ca = 1e-5  # Calibrated conversion for hypoxic overload scenarios
        cfg.channels.enable_ICa = True  # Unregulated Ca entry
        cfg.channels.gCa_max = 0.08  # Elevated ICa - PHYSIOLOGICAL range (was 1.2, too high)
        cfg.channels.enable_SK = False  # SK may not work under ATP depletion
        cfg.stim.jacobian_mode = 'sparse_fd'
        # Pathology: depolarization block after 1-2 spikes due to ion imbalance + Ca overload
        cfg.stim.stim_type = 'alpha'
        cfg.stim.alpha_tau = 1.0
        cfg.stim.Iext = 80.0  # Higher stim, but fails due to block

    # --- 11. С-ВОЛОКНО (БОЛЬ / БЕЗМИЕЛИНОВОЕ) ---
    elif "C-Fiber" in name:
        cfg.morphology.single_comp = False
        cfg.stim_location.location = "dendritic_filtered"
        # gNa=80 needed for proper spike amplitude in small unmyelinated fibers
        cfg.channels.gNa_max, cfg.channels.gK_max, cfg.channels.gL = 80.0, 10.0, 0.1
        cfg.channels.ENa, cfg.channels.EK, cfg.channels.EL = 50.0, -80.0, -60.0
        cfg.env.T_celsius, cfg.env.T_ref, cfg.env.Q10 = 37.0, 23.0, 2.3
        cfg.morphology.d_soma = 10e-4
        cfg.morphology.d_trunk = 0.8e-4
        cfg.morphology.Ra = 200.0
        cfg.morphology.N_ais = 0
        # Validated: ~3 spikes per alpha volley, Vmax ≈ 33 mV (slow C-fiber response)
        cfg.stim.stim_type = 'alpha'
        cfg.stim.alpha_tau = 1.0
        cfg.stim.Iext = 300.0

    # --- 12. ГИППОКАМП CA1 (THETA RHYTHM - Ih PACEMAKER) ---
    elif "Hippocampal CA1" in name:
        # Single-compartment CA1 preset for stable baseline validation of
        # intrinsic theta-related conductances in this reduced model.
        cfg.morphology.single_comp = True
        cfg.stim_location.location = "dendritic_filtered"
        # gNa=100 for proper spike amplitude through dendritic filter
        cfg.channels.gNa_max, cfg.channels.gK_max, cfg.channels.gL = 100.0, 8.0, 0.03
        cfg.channels.ENa, cfg.channels.EK, cfg.channels.EL = 50.0, -85.0, -68.0
        cfg.env.T_celsius, cfg.env.T_ref, cfg.env.Q10 = 37.0, 23.0, 2.3
        cfg.channels.enable_Ih = True
        cfg.channels.gIh_max = 0.02
        cfg.channels.enable_IA = True
        cfg.channels.gA_max = 0.8  # IA tuned for intrinsic theta-band pacing
        cfg.channels.enable_SK = False
        cfg.channels.enable_ICa = False
        cfg.channels.gCa_max = 0.0
        cfg.calcium.dynamic_Ca = False
        cfg.morphology.d_soma = 20e-4
        # Validated: const 15 µA/cm² through dend filter → ~21 spikes, 141 Hz, Vmax ≈ 42 mV
        # Theta rhythm (4-12 Hz) requires network oscillatory input; const stim shows tonic mode
        cfg.stim.stim_type = 'const'
        cfg.stim.alpha_tau = 2.0
        cfg.stim.Iext = 3.0

    # --- 13. АНЕСТЕЗИЯ (ЛИДОКАИН) ---
    elif "Anesthesia" in name:
        apply_preset(cfg, "Squid Giant Axon (HH 1952)")
        cfg.channels.gNa_max = 12.0  # 90% blockade: 120 → 12
        # Pathology: conduction block expected — high current, no/minimal spikes
        cfg.stim.stim_type = 'const'
        cfg.stim.Iext = 20.0  # Strong stimulus, but 90% Na block prevents spiking

    # --- 14. ГИПЕРКАЛИЕМИЯ (HIGH K+ REDUCES DRIVING FORCE) ---
    elif "Hyperkalemia" in name:
        apply_preset(cfg, "Squid Giant Axon (HH 1952)")
        cfg.channels.EK = -55.0  # Elevated external K+: -77 → -55 (Nernst shift)
        # Pathology: depolarization block — elevated K+ reduces repolarization reserve
        cfg.stim.stim_type = 'const'
        cfg.stim.Iext = 15.0

    # --- 15. IN VITRO (SLICE AT 23C - SLOW KINETICS) ---
    elif "In Vitro" in name:
        apply_preset(cfg, "Pyramidal L5 (Mainen 1996)")
        cfg.morphology.single_comp = True  # Slice: isolated soma (cut dendrites)
        cfg.stim_location.location = "soma"  # Direct patch-clamp
        cfg.dendritic_filter.enabled = False  # No dendritic filtering in slice
        cfg.env.T_celsius = 23.0  # Room temperature: 37 → 23°C (phi drops to 1.0)
        # Validated: 10 µA/cm² → ~10 spikes, 65 Hz, Vmax ≈ 46 mV (slower than 37°C)
        cfg.stim.stim_type = 'const'
        cfg.stim.Iext = 10.0

    # Stage/mode overlays for selected presets.
    if "Thalamic" in name:
        _apply_k_mode(cfg)
    elif "Alzheimer's" in name:
        _apply_alzheimer_mode(cfg)
    elif "Hypoxia" in name:
        _apply_hypoxia_mode(cfg)

    # Calculate absolute current for GUI display
    _calculate_absolute_iext(cfg)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: CALCULATE ABSOLUTE CURRENT FOR GUI DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

def _calculate_absolute_iext(cfg: FullModelConfig):
    """
    Calculate absolute stimulus current (nanoamperes) from density for GUI display.

    This function should be called AFTER setting cfg.stim.Iext density to populate
    the Iext_absolute_nA field for user-friendly display.

    Parameters
    ----------
    cfg : FullModelConfig
        Configuration with set Iext and morphology
    """
    from core.unit_converter import density_to_absolute_current

    soma_diameter_cm = cfg.morphology.d_soma
    soma_area_cm2 = np.pi * soma_diameter_cm ** 2

    cfg.stim.Iext_absolute_nA = density_to_absolute_current(
        cfg.stim.Iext, soma_area_cm2
    )

# ═══════════════════════════════════════════════════════════════════════════════
# СИНАПТИЧЕСКИЕ СТИМУЛЫ: Неврофизиологически корректные расширения alpha-функции
# ═══════════════════════════════════════════════════════════════════════════════

def get_synaptic_stimulus_names():
    """
    Возвращает доступные синаптические типы стимуляции.
    Применяются поверх уже установленного нейрона через apply_synaptic_stimulus().
    """
    return [
        "SYN: AMPA-receptor (Fast Excitation, 1-3 ms)",
        "SYN: NMDA-receptor (Slow Excitation, 50-100 ms)",
        "SYN: Kainate-receptor (Intermediate, 10-15 ms)",
        "SYN: GABA-A receptor (Fast Inhibition, 3-5 ms)",
        "SYN: GABA-B receptor (Slow Inhibition, 100-300 ms)",
        "SYN: Nicotinic ACh (Fast Excitation, 5-10 ms)",
    ]


def apply_synaptic_stimulus(cfg: FullModelConfig, stimulus_type: str):
    """
    Применяет конфигурацию синаптического стимула к уже установленному нейрону.
    
    Параметры:
    -----------
    cfg : FullModelConfig
        Конфигурация нейрона (уже должна иметь membrane parameters)
    stimulus_type : str
        Тип синаптического входа (из get_synaptic_stimulus_names())
    
    Примечание:
    -----------
    Эта функция ТОЛЬКО переопределяет параметры стимуляции (alpha_tau, Iext).
    Не трогает саму морфологию и каналы нейрона.
    
    ВАЖНО: Амплитуды масштабированы для soma (soma diameter ~20-30 µm).
    Для многокомпартментных моделей нужны относительно высокие значения (~5-20 µA)
    чтобы преодолеть кабельную утечку и вызвать деполяризацию.
    """
    
    # Default fallback is alpha, but known receptor presets below
    # switch to dedicated synaptic stimulus kernels.
    cfg.stim.stim_type = 'alpha'
    cfg.stim.pulse_start = 10.0  # Стандартная задержка перед стимулом
    
    # --- AMPA receptors (быстрые возбуждающие синапсы) ---
    if "AMPA" in stimulus_type:
        cfg.stim.stim_type = 'AMPA'
        cfg.stim.alpha_tau = 1.0        # 1 ms kinetics (fast rise/decay, Otis et al 1995)
        cfg.stim.Iext = 1.5             # Current density [µA/cm²] (peak 50-100 pA / soma area)
        cfg.stim.t_sim = 100.0

    # --- NMDA receptors (медленные возбуждающие синапсы) ---
    elif "NMDA" in stimulus_type:
        cfg.stim.stim_type = 'NMDA'
        cfg.stim.alpha_tau = 70.0       # 70 ms kinetics (slow, Mg-dependent, Jahr & Stevens 1990)
        cfg.stim.Iext = 0.8             # Current density [µA/cm²] (lower than AMPA, voltage-dependent)
        cfg.stim.t_sim = 300.0

    # --- Kainate receptors (промежуточные возбуждающие синапсы) ---
    elif "Kainate" in stimulus_type:
        cfg.stim.stim_type = 'Kainate'
        cfg.stim.alpha_tau = 12.0       # 10-15 ms kinetics (Ozawa et al 1998)
        cfg.stim.Iext = 1.2             # Current density [µA/cm²] (intermediate)
        cfg.stim.t_sim = 200.0

    # --- GABA-A receptors (быстрое торможение) ---
    elif "GABA-A" in stimulus_type:
        cfg.stim.stim_type = 'GABAA'
        cfg.stim.alpha_tau = 4.0        # 3-5 ms kinetics (fast, Thalmann 1986)
        cfg.stim.Iext = -1.5            # Current density [µA/cm²] INHIBITORY (hyperpolarizing)
        cfg.stim.t_sim = 100.0

    # --- GABA-B receptors (медленное торможение) ---
    elif "GABA-B" in stimulus_type:
        cfg.stim.stim_type = 'GABAB'
        cfg.stim.alpha_tau = 150.0      # 100-300 ms kinetics (very slow, G-protein coupled)
        cfg.stim.Iext = -0.6            # Current density [µA/cm²] INHIBITORY (longer effect)
        cfg.stim.t_sim = 400.0

    # --- Nicotinic Acetylcholine receptors (быстрое возбуждение от ACh) ---
    elif "Nicotinic" in stimulus_type:
        cfg.stim.stim_type = 'Nicotinic'
        cfg.stim.alpha_tau = 7.0        # 5-10 ms kinetics (Armstrong & Gilly 1992)
        cfg.stim.Iext = 1.8             # Current density [µA/cm²] (similar to AMPA, cation channel)
        cfg.stim.t_sim = 150.0
    
    else:
        raise ValueError(f"Unknown synaptic stimulus type: {stimulus_type}")
