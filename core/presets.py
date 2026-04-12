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
        "K: Thalamic Relay (Ih + ITCa + Burst)",
        "L: Hippocampal CA1 Pyramidal (Adapting)",
        "M: Epilepsy (v10 SCN1A mutation)",
        "N: Alzheimer's (v10 Calcium Toxicity)",
        "O: Hypoxia (v10 ATP-pump failure)",
        "P: Thalamic Reticular Nucleus (TRN Spindles)",
        "Q: Striatal Spiny Projection (SPN)",
        "R: Cholinergic Neuromodulation (ACh)",
        "S: Pathology: Dravet Syndrome (SCN1A LOF)",
        "T: Passive Cable (Linear Decay)",
        "U: I-Clamp Threshold Explorer",
    ]


def _preset_code(name: str) -> str:
    """Return the stable single-letter preset code prefix (e.g. 'K')."""
    if not isinstance(name, str):
        return ""
    head, _, _ = name.partition(":")
    return head.strip()


def _restore_owned_modes(cfg: FullModelConfig, name: str, saved_modes: dict) -> None:
    """Restore only the mode fields that actually belong to the selected preset."""
    code = _preset_code(name)
    owners = {
        "B": ("l5_mode",),
        "E": ("purkinje_mode",),
        "F": ("delay_target",),
        "G": ("anesthesia_mode",),
        "K": ("k_mode",),
        "N": ("alzheimer_mode",),
        "O": ("hypoxia_mode",),
        "R": ("ach_mode",),
        "S": ("dravet_mode",),
    }
    for field in owners.get(code, ()):
        if field in saved_modes:
            setattr(cfg.preset_modes, field, saved_modes[field])


def _apply_base_preset(cfg: FullModelConfig, base_name: str, *, preserve_mode_fields: tuple[str, ...] = ()) -> None:
    """Apply another preset as an internal base while preserving selected outer mode fields."""
    preserved = {field: getattr(cfg.preset_modes, field) for field in preserve_mode_fields}
    apply_preset(cfg, base_name)
    cfg.stim.event_times = []
    cfg.stim.event_times_2 = []
    cfg.stim.synaptic_train_type = "none"
    cfg.stim.synaptic_train_freq_hz = 40.0
    cfg.stim.synaptic_train_duration_ms = 200.0
    cfg.stim.noise_sigma = 0.0
    cfg.dual_stimulation = None
    for field, value in preserved.items():
        setattr(cfg.preset_modes, field, value)

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
        PresetModeParams, MetabolismParams,
    )
    _copy_defaults(cfg.morphology,        MorphologyParams())
    _copy_defaults(cfg.channels,          ChannelParams())
    _copy_defaults(cfg.calcium,           CalciumParams())
    _copy_defaults(cfg.metabolism,       MetabolismParams())
    _copy_defaults(cfg.env,               EnvironmentParams())
    _copy_defaults(cfg.stim,              SimulationParams())
    _copy_defaults(cfg.stim_location,     StimulationLocationParams())
    _copy_defaults(cfg.dendritic_filter,  DendriticFilterParams())
    _copy_defaults(cfg.analysis,          AnalysisParams())
    _copy_defaults(cfg.preset_modes,      PresetModeParams())
    cfg.dual_stimulation = None
    cfg.notes = ""


def _copy_defaults(target, source) -> None:
    """
    Copy field values from a fresh default `source` model into an existing `target` model while preserving nested object identity.
    
    Mutates `target` in place: for each field present on `source`, if both `target` and `source` attributes look like Pydantic models (have `model_fields`), the function recurses into those nested objects to update their fields rather than replacing the nested object; otherwise it assigns the `source` attribute value onto `target`.
    
    Parameters:
        target: The existing Pydantic model instance to be updated in place.
        source: A fresh/default Pydantic model instance whose field values will be copied into `target`.
    """
    for field_name in type(source).model_fields:
        target_val = getattr(target, field_name)
        source_val = getattr(source, field_name)
        
        # If it's a nested Pydantic model, recurse to preserve identity
        if hasattr(target_val, 'model_fields') and hasattr(source_val, 'model_fields'):
            _copy_defaults(target_val, source_val)
        else:
            # Otherwise, just set the value (preserves the parent object identity)
            setattr(target, field_name, source_val)


def _apply_l5_mode(cfg: FullModelConfig) -> None:
    """
    Apply L5 pyramidal neuromodulation overlays according to cfg.preset_modes.l5_mode.
    
    If `l5_mode` is "high_ach", modifies the configuration to reflect high acetylcholine (arousal):
    - disables the muscarinic-sensitive M current (`channels.enable_IM = False`) and sets `channels.gIM_max = 0.0`;
    - reduces SK conductance (`channels.gSK_max = 0.6`);
    - increases synaptic train frequency (`stim.synaptic_train_freq_hz = 40.0`) and external drive (`stim.Iext = 80.0`).
    
    If `l5_mode` is any other value, leaves channel and stimulation defaults from the current preset unchanged. The function returns immediately if `cfg` lacks `channels` or `stim`.
     
    Parameters:
        cfg (FullModelConfig): Model configuration to modify in place.
    """
    # Validate config structure before modification
    if not hasattr(cfg, 'channels') or not hasattr(cfg, 'stim'):
        return

    if cfg.preset_modes.l5_mode == "high_ach":
        # High acetylcholine: blocks I_M (muscarinic-sensitive K+ current) → reduced adaptation
        cfg.channels.enable_IM = False  # ACh blocks I_M
        cfg.channels.gIM_max = 0.0
        cfg.channels.gSK_max = 0.6  # Reduce SK by 50% (1.2 → 0.6)
        # Increase synaptic drive to compensate for reduced adaptation and show arousal effect
        cfg.stim.synaptic_train_freq_hz = 40.0  # Higher frequency for arousal state
        cfg.stim.Iext = 80.0  # Stronger drive to overcome reduced adaptation
        # Title suffix added by GUI
    # Normal mode: use default IM (0.04) and SK (1.2) from preset


def _apply_ach_mode(cfg: FullModelConfig) -> None:
    """Apply ACh mode for preset R (sleep vs arousal)."""
    cfg.preset_modes.l5_mode = "normal"
    mode = getattr(cfg.preset_modes, "ach_mode", "sleep")
    cfg.channels.enable_IM = True
    if mode == "arousal":
        cfg.channels.gIM_max = 0.05
        cfg.channels.im_speed_multiplier = 1.0
        cfg.stim.Iext = 12.0
        cfg.stim.noise_sigma = 0.35
    else:
        cfg.channels.gIM_max = 0.5
        cfg.channels.im_speed_multiplier = 0.15
        cfg.stim.Iext = 5.0
        cfg.stim.noise_sigma = 0.10


def _apply_purkinje_mode(cfg: FullModelConfig) -> None:
    """Apply Purkinje firing mode overlays."""
    mode = getattr(cfg.preset_modes, "purkinje_mode", "tonic")
    if mode == "climbing_fiber":
        cfg.stim.stim_type = "pulse"
        cfg.stim.Iext = 150.0
        cfg.stim.pulse_dur = 2.0


def _apply_anesthesia_mode(cfg: FullModelConfig) -> None:
    """Apply partial vs. full sodium-channel block for local anesthesia."""
    mode = getattr(cfg.preset_modes, "anesthesia_mode", "full_block")
    cfg.stim.stim_type = "const"
    if mode == "partial_block":
        cfg.channels.gNa_max = 48.0
        cfg.stim.Iext = 24.0
    else:
        cfg.channels.gNa_max = 12.0
        cfg.stim.Iext = 20.0


def _apply_k_mode(cfg: FullModelConfig) -> None:
    """
    Configure the model's thalamic relay behavior according to cfg.preset_modes.k_mode.
    
    For "baseline": set a hyperpolarizing pulse to promote post-inhibitory rebound bursting (bursty/sleep-like relay).
    For "delta_oscillator": set a sustained hyperpolarizing drive with enhanced pacemaker currents to support slow (~delta) oscillations.
    For other values ("activated"): set a tonic depolarizing drive to favor tonic relay (awake) behaviour.
    """
    if cfg.preset_modes.k_mode == "baseline":
        # Sleep/drowsy: hyperpolarizing pulse de-inactivates I_T → Post-Inhibitory Rebound burst
        cfg.stim.stim_type = "pulse"
        cfg.stim.Iext = -4.0  # Gentler hyperpolarization preserves low-throughput rebound output
        cfg.stim.pulse_start = 10.0
        cfg.stim.pulse_dur = 60.0
        cfg.channels.gIh_max = 0.02
        cfg.channels.gTCa_max = 1.2
        cfg.channels.EL = -75.0  # Rest Vm = -75mV (sleep)
    elif cfg.preset_modes.k_mode == "delta_oscillator":
        # Self-sustained delta oscillations: constant hyperpolarization + strong Ih + I_T
        cfg.stim.stim_type = "const"
        cfg.stim.Iext = -3.0  # Constant hyperpolarization to drive Ih activation
        cfg.channels.gIh_max = 0.05  # Increased Ih for pacemaker activity
        cfg.channels.gTCa_max = 2.0  # I_T for burst generation
        cfg.channels.EL = -70.0  # Rest Vm slightly hyperpolarized
    else:
        # Activated: tonic relay firing (I_T mostly inactivated at depolarized Vm)
        cfg.stim.stim_type = "const"
        cfg.stim.Iext = 8.0  # Tonic drive
        cfg.channels.gIh_max = 0.02
        cfg.channels.gTCa_max = 2.0
        cfg.channels.EL = -60.0  # Rest Vm = -60mV (awake)


def _apply_alzheimer_mode(cfg: FullModelConfig) -> None:
    """Apply Alzheimer's stage variants."""
    # Enable calcium dynamics and channels for Alzheimer pathology
    cfg.calcium.dynamic_Ca = True
    cfg.channels.enable_ICa = True
    cfg.channels.enable_SK = True
    
    if cfg.preset_modes.alzheimer_mode == "terminal":
        # Terminal stage: severe network failure, near-silent response.
        cfg.calcium.tau_Ca = 1200.0
        cfg.channels.gSK_max = 2.2
        cfg.channels.gCa_max = 0.06
        cfg.stim.stim_type = "alpha"
        cfg.stim.alpha_tau = 2.0
        cfg.stim.Iext = 15.0
        cfg.stim.t_sim = 500.0
    else:
        # Progressive stage: initial spiking followed by adaptation/decline.
        cfg.calcium.tau_Ca = 800.0
        cfg.channels.gSK_max = 1.0
        cfg.channels.gCa_max = 0.08
        cfg.stim.stim_type = "const"
        cfg.stim.Iext = 30.0
        cfg.stim.t_sim = 500.0


def _apply_hypoxia_mode(cfg: FullModelConfig) -> None:
    """Apply hypoxia stage variants."""
    # Enable calcium dynamics for hypoxia pathology
    cfg.calcium.dynamic_Ca = True
    cfg.channels.enable_ICa = True
    cfg.channels.enable_SK = False  # SK may not work under ATP depletion

    # Enable dynamic ATP metabolism for hypoxia pathology
    cfg.metabolism.enable_dynamic_atp = True
    cfg.metabolism.atp_synthesis_rate = 0.1
    # Axonal failure under energy depletion: reduced effective Na conductance in trunk.
    cfg.morphology.gNa_trunk_mult = 0.5

    if cfg.preset_modes.hypoxia_mode == "terminal":
        # Terminal stage: profound depolarization block and pump collapse.
        # ATP synthesis stops completely, K_ATP fully opens, pumps fail
        cfg.morphology.gL_trunk_mult = 10.0  # Massive demyelination-like leak in trunk
        cfg.channels.EK = -45.0
        cfg.channels.EL = -40.0
        cfg.channels.gL = 0.4
        cfg.calcium.tau_Ca = 1200.0
        cfg.channels.gCa_max = 0.10
        cfg.stim.stim_type = 'const'
        cfg.stim.Iext = 80.0
    else:
        # Progressive stage: short early spiking epoch, then attenuation.
        # Reduced ATP synthesis causes gradual depletion, K_ATP opens, spiking slows
        cfg.channels.EK = -60.0
        cfg.channels.EL = -50.0
        cfg.channels.gL = 0.15
        cfg.calcium.tau_Ca = 900.0
        cfg.channels.gCa_max = 0.08
        cfg.stim.stim_type = 'const'
        cfg.stim.Iext = 95.0


def _apply_ca1_theta_protocol(cfg: FullModelConfig) -> None:
    """Single-cell CA1 theta surrogate: distal excitation + somatic inhibitory pacing."""
    from core.dual_stimulation import DualStimulationConfig

    cfg.morphology.single_comp = False
    cfg.morphology.N_trunk = 6
    cfg.stim_location.location = "dendritic_filtered"
    cfg.dendritic_filter.enabled = True
    cfg.dendritic_filter.distance_um = 180.0
    cfg.dendritic_filter.space_constant_um = 180.0
    cfg.dendritic_filter.tau_dendritic_ms = 12.0

    # Distal glutamatergic theta drive with a mild high-conductance background.
    cfg.stim.stim_type = 'AMPA'
    cfg.stim.Iext = 2.0
    cfg.stim.alpha_tau = 2.0
    cfg.stim.pulse_start = 0.0
    cfg.stim.synaptic_train_type = 'regular'
    cfg.stim.synaptic_train_freq_hz = 7.0
    cfg.stim.synaptic_train_duration_ms = 5000.0
    cfg.stim.t_sim = 5000.0
    cfg.stim.dt_eval = 0.5
    cfg.stim.noise_sigma = 0.25
    cfg.stim.jacobian_mode = 'native_hines'

    # Phase-shifted fast inhibition at the soma keeps the rhythm honest.
    cfg.dual_stimulation = DualStimulationConfig(
        enabled=True,
        secondary_location="soma",
        secondary_stim_type="GABAA",
        secondary_Iext=0.8,
        secondary_start=35.0,
        secondary_train_type="regular",
        secondary_train_freq_hz=7.0,
        secondary_train_duration_ms=5000.0,
        secondary_alpha_tau=6.0,
    )


def _apply_epilepsy_disinhibition_protocol(cfg: FullModelConfig) -> None:
    """Protocol overlay for epileptic disinhibition, separate from intrinsic excitability."""
    from core.dual_stimulation import DualStimulationConfig

    cfg.stim_location.location = "soma"
    cfg.stim.stim_type = "const"
    cfg.stim.Iext = 30.0
    cfg.stim.noise_sigma = 0.2
    cfg.dual_stimulation = DualStimulationConfig(
        enabled=True,
        secondary_location="dendritic_filtered",
        secondary_stim_type="GABAA",
        secondary_Iext=15.0,
        secondary_train_type="poisson",
        secondary_train_freq_hz=100.0,
        secondary_train_duration_ms=5000.0,
        secondary_distance_um=75.0,
        secondary_space_constant_um=100.0,
        secondary_tau_dendritic_ms=5.0,
    )


def apply_preset(cfg: FullModelConfig, name: str):
    saved_modes = cfg.preset_modes.model_dump()
    code = _preset_code(name)
    # Reset all fields to Pydantic defaults
    _reset_cfg_to_defaults(cfg)
    _restore_owned_modes(cfg, name, saved_modes)

    # --- 1. CLASSIC: SQUID GIANT AXON (HH 1952) ---
    if code == "A":
        cfg.dendritic_filter.enabled = False  # Squid axon has no dendrites
        cfg.channels.gNa_max, cfg.channels.gK_max, cfg.channels.gL = 120.0, 36.0, 0.3
        cfg.channels.ENa, cfg.channels.EK, cfg.channels.EL = 50.0, -77.0, -54.387
        cfg.channels.Cm = 1.0
        cfg.env.T_celsius, cfg.env.T_ref, cfg.env.Q10 = 6.3, 6.3, 3.0
        # Space-clamped preparation (HH 1952 original): single-compartment
        cfg.morphology.single_comp = True
        cfg.morphology.d_soma = 500e-4  # 500 µm giant axon diameter
        cfg.stim.Iext = 10.0  # Classic HH: ~68 Hz tonic firing, Vmax ≈ 40 mV
        cfg.stim.jacobian_mode = 'native_hines'  # Use native Hines for Squid Axon

    # --- 2. БАЗОВЫЙ ПРЕСЕТ: ПИРАМИДАЛЬНЫЙ L5 (Млекопитающие) ---
    elif code == "B":
        cfg.morphology.single_comp = False
        cfg.stim_location.location = "dendritic_filtered"  # v12.0: Use dendritic filtering
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
        cfg.morphology.N_trunk = 10     # Standardized reduced model
        cfg.morphology.N_ais = 3       # Axon initial segment compartments
        cfg.morphology.gNa_ais_mult = 40.0  # AIS has 40× higher Na than soma
        cfg.morphology.gK_ais_mult = 5.0    # AIS has 5× higher K than soma
        cfg.morphology.d_trunk = 2e-4  # Dendrite trunk diameter
        cfg.morphology.Ra = 150.0      # Axial resistance [Ω·cm]
        
        # CALCIUM DYNAMICS: Enable for realistic adaptation
        cfg.channels.enable_SK = True  # Enable SK for calcium-dependent adaptation
        cfg.calcium.dynamic_Ca = True  # Enable calcium dynamics
        cfg.calcium.Ca_rest = 5e-5       # 50 nM resting [Ca²⁺]ᵢ
        cfg.calcium.Ca_ext = 2.0         # 2 mM extracellular
        cfg.calcium.tau_Ca = 200.0
        cfg.calcium.B_Ca = 1e-5  # v12.0: Reverted to 1e-5 (was 1e-7)
        cfg.channels.enable_ICa = True  # L-type calcium channel
        cfg.channels.gCa_max = 0.08  # Moderate ICa for calcium dynamics
        cfg.channels.gSK_max = 1.2  # SK channel for calcium-dependent adaptation
        
        # M-TYPE K CURRENT: Enable for spike frequency adaptation
        cfg.channels.enable_IM = True  # KCNQ2/3 channels for adaptation
        cfg.channels.gIM_max = 0.04  # v12.0: Increased M-current for stronger SFA
        
        # SYNAPTIC TRAIN: Poisson 30Hz for natural cortical jitter
        cfg.stim.synaptic_train_type = 'poisson'
        cfg.stim.synaptic_train_freq_hz = 30.0
        cfg.stim.synaptic_train_duration_ms = 500.0

        # CALIBRATED: Literature-based rheobase (Mainen 1996)
        cfg.stim.stim_type = 'AMPA'  # v12.0: Fixed - 'synaptic_train' not in s_map
        cfg.stim.alpha_tau = 2.0
        cfg.stim.Iext = 60.0  # v12.0: Further increased for SFA with multiple spikes

    # Stage/mode overlays for selected presets.
    if code == "B":
        _apply_l5_mode(cfg)

    # --- 3. БЫСТРЫЙ ИНТЕРНЕЙРОН (FS) ---
    elif code == "C":
        cfg.stim_location.location = "dendritic_filtered"
        cfg.dendritic_filter.enabled = True
        cfg.dendritic_filter.distance_um = 75.0
        cfg.dendritic_filter.space_constant_um = 100.0
        cfg.dendritic_filter.tau_dendritic_ms = 5.0
        # Wang-Buzsaki neurons are non-adapting: ultra-fast repolarization
        cfg.channels.gNa_max, cfg.channels.gK_max, cfg.channels.gL = 120.0, 45.0, 0.1
        cfg.channels.ENa, cfg.channels.EK, cfg.channels.EL = 55.0, -90.0, -65.0
        cfg.channels.Cm = 1.0
        cfg.env.T_celsius, cfg.env.T_ref, cfg.env.Q10 = 37.0, 23.0, 2.3
        cfg.morphology.d_soma = 15e-4
        # RESURGENT Na: Enables sustained high-frequency firing (>150 Hz)
        cfg.channels.enable_NaR = True
        cfg.channels.gNaR_max = 0.4
        # Disable IA and SK: Wang-Buzsaki neurons are non-adapting
        cfg.channels.enable_IA = False
        cfg.channels.enable_SK = False
        # Validated: 40 µA/cm² through dendritic filter (atten=0.47) → ~19 effective
        # Produces sustained high-frequency firing >150 Hz with near-zero adaptation
        cfg.stim.stim_type = 'const'
        cfg.stim.alpha_tau = 1.0
        cfg.stim.Iext = 40.0

    # --- 4. МОТОНЕЙРОН СПИННОГО МОЗГА ---
    elif code == "D":
        cfg.morphology.single_comp = False
        cfg.stim_location.location = "dendritic_filtered"
        cfg.channels.gNa_max, cfg.channels.gK_max, cfg.channels.gL = 100.0, 30.0, 0.1
        cfg.channels.ENa, cfg.channels.EK, cfg.channels.EL = 50.0, -77.0, -60.0
        cfg.channels.Cm = 1.5
        cfg.env.T_celsius, cfg.env.T_ref, cfg.env.Q10 = 37.0, 23.0, 2.3
        cfg.morphology.d_soma = 70e-4
        cfg.morphology.N_trunk = 10     # Standardized reduced model
        cfg.morphology.d_ais = 4e-4    # Match large soma (60 μm)
        cfg.morphology.d_trunk = 5e-4  # Match large soma (60 μm)
        cfg.morphology.N_ais = 10
        cfg.morphology.l_ais = 100e-4
        cfg.morphology.Ra = 70.0
        
        # CALCIUM DYNAMICS: Enable for PICs and AHP
        cfg.calcium.dynamic_Ca = True
        cfg.calcium.Ca_rest = 5e-5       # 50 nM resting [Ca²⁺]ᵢ
        cfg.calcium.Ca_ext = 2.0         # 2 mM extracellular
        cfg.calcium.tau_Ca = 50.0  # v12.0: Fast initial buffering (was 200.0)
        cfg.calcium.B_Ca = 2e-6  # v12.0: Large volume = high dilution (was 1e-5)
        cfg.channels.enable_ICa = True  # L-type calcium channels for PICs
        cfg.channels.gCa_max = 0.08  # Moderate L-type Ca for persistent inward currents
        cfg.channels.enable_SK = True  # SK channels for AHP
        cfg.channels.gSK_max = 2.0  # v12.0: Reduced from 4.0 to 2.0 (was 1.0)
        
        # IA channel: Moderate for frequency adaptation and spike-frequency accommodation
        cfg.channels.enable_IA = True
        cfg.channels.gA_max = 0.25  # Moderate IA for adaptation dynamics
        cfg.channels.EK = -77.0    # K+ reversal potential
        
        # Alpha stimulus: represents one synaptic volley from descending pathways
        # Multi-comp with AIS produces burst of ~16 spikes, Vmax ≈ 34 mV
        cfg.stim.stim_type = 'alpha'
        cfg.stim.alpha_tau = 1.5
        cfg.stim.Iext = 50.0

    # --- 5. КЛЕТКА ПУРКИНЬЕ (МОЗЖЕЧОК) ---
    elif code == "E":
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
        cfg.morphology.N_trunk = 10     # Standardized reduced model
        cfg.morphology.N_ais = 3
        cfg.morphology.gNa_ais_mult = 15.0
        cfg.channels.enable_SK = True  # Re-enable SK - critical for Purkinje physiology
        cfg.channels.gSK_max = 1.5  # Strong SK AHP in Purkinje physiology
        
        # CALCIUM DYNAMICS: Purkinje is calcium processing machine
        cfg.calcium.dynamic_Ca = True
        cfg.calcium.Ca_rest = 5e-5       # 50 nM resting [Ca²⁺]ᵢ
        cfg.calcium.Ca_ext = 2.0         # 2 mM extracellular
        cfg.calcium.tau_Ca = 60.0
        cfg.calcium.B_Ca = 1e-5  # v12.0: Reverted to 1e-5 (was 1e-7)
        cfg.channels.enable_ICa = True  # L-type calcium channels
        cfg.channels.gCa_max = 0.08  # Physiological L-type calcium conductance
        
        # RESURGENT Na: Enables high-frequency firing >200 Hz
        cfg.channels.enable_NaR = True  # Enable resurgent Na for high-frequency bursts
        cfg.channels.gNaR_max = 0.5  # Moderate resurgent Na conductance
        
        # IA channel: High for complex spike dynamics and dendritic integration
        cfg.channels.enable_IA = True
        cfg.channels.gA_max = 0.4   # IA for complex spike dynamics
        # E_A is now a property that returns EK
        
        # Tuned to preserve tonic Purkinje spiking and avoid a silent island
        cfg.stim.stim_type = 'const'
        cfg.stim.alpha_tau = 2.0
        cfg.stim.Iext = 32.0

        # DUAL STIMULATION: Add noisy dendritic inhibition (granular layer)
        # Primary stimulus is set on cfg.stim, secondary on dual_stimulation
        from core.dual_stimulation import DualStimulationConfig
        cfg.stim_location.location = "soma"
        cfg.stim.stim_type = "const"
        cfg.stim.Iext = 30.0  # Primary stimulus
        cfg.dual_stimulation = DualStimulationConfig(
            enabled=True,
            secondary_location="dendritic_filtered",
            secondary_stim_type="GABAA",
            secondary_Iext=15.0,
            secondary_train_type="poisson",
            secondary_train_freq_hz=60.0,
            secondary_train_duration_ms=5000.0,
            secondary_distance_um=200.0,
            secondary_space_constant_um=180.0,
            secondary_tau_dendritic_ms=15.0
        )

    # --- 6. ТАЛАМИЧЕСКИЙ РЕЛЕ-НЕЙРОН (Ih + IT + Ca-dynamics) ---
    elif code == "K":
        # Thalamocortical relay neuron: T-type Ca²⁺ (I_T) drives low-threshold
        # spikes (LTS) and post-inhibitory rebound bursts.
        # Reference: Destexhe et al. 1998, J Neurosci 18:3574;
        #            McCormick & Huguenard 1992, J Neurophysiol 68:1384
        cfg.morphology.single_comp = True
        cfg.stim_location.location = "soma"
        cfg.dendritic_filter.enabled = False
        cfg.channels.gNa_max, cfg.channels.gK_max, cfg.channels.gL = 90.0, 12.0, 0.05
        cfg.channels.ENa, cfg.channels.EK, cfg.channels.EL = 50.0, -90.0, -70.0
        cfg.env.T_celsius, cfg.env.T_ref, cfg.env.Q10 = 37.0, 24.0, 2.3
        # Ih: hyperpolarization-activated cation current — essential for sag and rebound
        cfg.channels.enable_Ih = True
        cfg.channels.gIh_max = 0.03
        cfg.channels.E_Ih = -43.0  # Standardized reversal potential
        # I_T: low-threshold T-type Ca²⁺ (CaV3.x) — replaces L-type for LTS bursting
        cfg.channels.enable_ITCa = True
        cfg.channels.gTCa_max = 2.0     # Destexhe 1998: 2.0 mS/cm² somatic density
        cfg.morphology.single_comp = False
        cfg.morphology.N_trunk = 5
        cfg.morphology.gCa_ais_mult = 1.0  # Keep T-current distributed along soma+trunk
        cfg.channels.enable_ICa = False  # L-type NOT used in relay neurons
        cfg.channels.enable_SK = False
        cfg.calcium.dynamic_Ca = True
        cfg.calcium.Ca_rest = 5e-5       # 50 nM resting [Ca²⁺]ᵢ
        cfg.calcium.Ca_ext = 2.0         # 2 mM extracellular
        cfg.calcium.tau_Ca = 200.0
        cfg.calcium.B_Ca = 2e-5  # v12.0: Thalamic calcium conversion factor (was 1e-5)
        cfg.morphology.d_soma = 25e-4
        cfg.stim.jacobian_mode = 'native_hines'
        # v12.0: k_mode overlay handles stimulus (baseline=alpha burst, activated=tonic)
        cfg.stim.stim_type = 'const'
        cfg.stim.alpha_tau = 5.0
        cfg.stim.Iext = 0.0  # v12.0: No default Iext, k_mode overlay sets it

    # --- 7. ПАТОЛОГИЯ: РАССЕЯННЫЙ СКЛЕРОЗ (Демиелинизация) ---
    elif code == "F":
        _apply_base_preset(cfg, "B: Pyramidal L5 (Mainen 1996)", preserve_mode_fields=("delay_target",))
        # Authentic demyelination model: axon-specific scaling
        # Soma remains healthy (gL=0.02, Cm=0.75), but axon trunk/branches have increased leak
        # This creates "Conduction Delay/Block" where soma spikes but axon conduction is impaired
        cfg.channels.gL = 0.02  # Soma remains healthy
        cfg.channels.Cm = 0.75  # Soma remains healthy
        cfg.channels.gK_max = 6.0
        cfg.channels.gSK_max = 0.8
        cfg.morphology.gL_trunk_mult = 10.0  # Strong but not fully blocking leak in demyelinated segments
        cfg.morphology.Cm_trunk_mult = 3.0  # Moderately increased capacitance (was 10.0)
        cfg.morphology.gNa_trunk_mult = 0.8  # Partial Nav redistribution preserves weak terminal conduction
        cfg.morphology.N_ais = 10
        cfg.morphology.gNa_ais_mult = 50.0
        cfg.morphology.Ra = 500.0  # Elevated axial resistance slows conduction without complete block
        cfg.stim.jacobian_mode = 'native_hines'
        # Set delay target to Terminal to show the "Wow" effect of conduction block immediately
        cfg.preset_modes.delay_target = "Terminal"
        # Keep the same input class as control; pathology should emerge from axon-specific changes.
        cfg.stim.stim_type = 'alpha'
        cfg.stim.alpha_tau = 1.5
        cfg.stim.Iext = 50.0

    # --- 8. ПАТОЛОГИЯ: ЭПИЛЕПСИЯ (SCN1A GAIN-OF-FUNCTION) ---
    elif code == "M":
        _apply_base_preset(cfg, "C: FS Interneuron (Wang-Buzsaki)")
        cfg.channels.gNa_max = 180.0  # Gain-of-function: 120 → 180 (SCN1A mutation)
        cfg.channels.enable_IA = False
        # Enable calcium channels for pathological calcium dynamics
        cfg.channels.enable_ICa = True  # L-type calcium channel
        cfg.channels.gCa_max = 0.08  # Moderate ICa - PHYSIOLOGICAL range (was 1.2, too high)
        cfg.calcium.dynamic_Ca = True  # Enable calcium dynamics
        cfg.calcium.tau_Ca = 300.0  # Moderately impaired clearance
        cfg.calcium.B_Ca = 1e-5  # Calibrated conversion for pathological accumulation without runaway Ca
        # Chloride Shift: depolarized GABA reversal (pathological disinhibition)
        cfg.channels.e_rev_syn_secondary = -40.0  # Depolarized from -75 to -40 mV
        _apply_epilepsy_disinhibition_protocol(cfg)
# --- 9. ПАТОЛОГИЯ: АЛЬЦГЕЙМЕР (CALCIUM TOXICITY - SLOW EXTRUSION) ---
    elif code == "N":
        _apply_base_preset(cfg, "B: Pyramidal L5 (Mainen 1996)", preserve_mode_fields=("alzheimer_mode",))
        # Apply Alzheimer mode overlay immediately
        _apply_alzheimer_mode(cfg)
        cfg.stim.jacobian_mode = 'native_hines'
        # Mode overlay owns the effective stimulus profile; keep this branch from
        # re-imposing a generic alpha pulse that would erase stage differences.

    # --- 10. ПАТОЛОГИЯ: ГИПОКСИЯ (ATP-PUMP FAILURE - ION IMBALANCE + Ca overload) ---
    elif code == "O":
        _apply_base_preset(cfg, "C: FS Interneuron (Wang-Buzsaki)", preserve_mode_fields=("hypoxia_mode",))
        cfg.morphology.single_comp = False
        cfg.morphology.N_trunk = 10
        cfg.morphology.N_ais = 5
        cfg.morphology.gNa_ais_mult = 20.0
        # Ion imbalance from pump failure
        cfg.channels.EK = -50.0  # Reduced: -90 → -50 (K+ accumulation)
        cfg.channels.EL = -45.0  # Reduced: -65 → -45 (Na+ accumulation)
        # CALCIUM OVERLOAD: Hypoxia causes Ca2+ accumulation and mitochondrial dysfunction
        cfg.calcium.tau_Ca = 900.0  # Slow clearance: pump failure (fixed from 1500ms)
        cfg.calcium.B_Ca = 2e-5  # v12.0: Slower metabolic suffocation (was 1e-5)
        cfg.channels.gCa_max = 0.08  # Elevated ICa - PHYSIOLOGICAL range (was 1.2, too high)
        cfg.stim.jacobian_mode = 'native_hines'
        # Pathology: depolarization block after 1-2 spikes due to ion imbalance + Ca overload
        cfg.stim.stim_type = 'alpha'
        cfg.stim.alpha_tau = 1.0
        cfg.stim.Iext = 80.0  # Higher stim, but fails due to block
        # Enable dynamic ATP metabolism for hypoxia pathology
        cfg.metabolism.enable_dynamic_atp = True
        cfg.metabolism.atp_synthesis_rate = 0.1  # Reduced synthesis causes gradual depletion

    # --- 11. С-ВОЛОКНО (БОЛЬ / БЕЗМИЕЛИНОВОЕ) ---
    elif code == "J":
        cfg.morphology.single_comp = False
        cfg.stim_location.location = "dendritic_filtered"
        # gNa=80 needed for proper spike amplitude in small unmyelinated fibers
        cfg.channels.gNa_max, cfg.channels.gK_max, cfg.channels.gL = 80.0, 10.0, 0.1
        cfg.channels.ENa, cfg.channels.EK, cfg.channels.EL = 50.0, -80.0, -60.0
        cfg.env.T_celsius, cfg.env.T_ref, cfg.env.Q10 = 37.0, 23.0, 2.3
        cfg.morphology.d_soma = 10e-4
        cfg.morphology.N_trunk = 10     # Standardized reduced model
        cfg.morphology.d_trunk = 0.8e-4
        cfg.morphology.Ra = 200.0
        cfg.morphology.N_ais = 0
        # Validated: ~3 spikes per alpha volley, Vmax ≈ 33 mV (slow C-fiber response)
        cfg.stim.stim_type = 'alpha'
        cfg.stim.alpha_tau = 1.0
        cfg.stim.Iext = 300.0

    # --- 12. ГИППОКАМП CA1 (АДАПТИВНЫЙ ПИРАМИДНЫЙ) ---
    elif code == "L":
        # CA1 pyramidal neuron: regular-spiking adapting type.
        # v12.0: Theta rhythm (7Hz) driven by dual-drive stimulation
        # Reference: Magee 1998, J Neurosci 18:7613; Storm 1990, J Physiol 421:529
        cfg.morphology.single_comp = True
        cfg.stim_location.location = "soma"
        cfg.dendritic_filter.enabled = False
        cfg.channels.gNa_max, cfg.channels.gK_max, cfg.channels.gL = 56.0, 8.0, 0.05
        cfg.channels.ENa, cfg.channels.EK, cfg.channels.EL = 50.0, -85.0, -60.0
        cfg.env.T_celsius, cfg.env.T_ref, cfg.env.Q10 = 37.0, 23.0, 2.3
        # Ih: provides subthreshold resonance in theta band (Magee 1998)
        cfg.channels.enable_Ih = True
        cfg.channels.gIh_max = 0.02  # Increased for theta resonance
        cfg.channels.E_Ih = -43.0  # Standardized reversal potential
        # IA: physiological density for spike-frequency adaptation
        # (Storm 1990: gA ~0.3-0.5 mS/cm² somatic; NOT the unphysiological 10.0)
        cfg.channels.enable_IA = True
        cfg.channels.gA_max = 0.3  # Standard for CA1
        # v12.0: Enable calcium dynamics for proper adaptation
        cfg.channels.enable_SK = True
        cfg.channels.enable_IM = True
        cfg.channels.enable_ICa = True
        cfg.calcium.dynamic_Ca = True
        cfg.calcium.Ca_rest = 5e-5       # 50 nM resting [Ca²⁺]ᵢ
        cfg.calcium.Ca_ext = 2.0         # 2 mM extracellular
        cfg.calcium.tau_Ca = 100.0  # v12.0: CA1 calcium decay (was disabled)
        cfg.calcium.B_Ca = 1e-5  # v12.0: CA1 calcium conversion factor (was disabled)
        cfg.channels.gCa_max = 0.08  # L-type calcium for adaptation
        cfg.channels.gSK_max = 0.5  # SK for calcium-dependent adaptation (reduced to allow theta-rhythm trains)
        cfg.channels.gIM_max = 0.04  # M-type for slow adaptation
        cfg.morphology.d_soma = 20e-4
        _apply_ca1_theta_protocol(cfg)
# --- 13. АНЕСТЕЗИЯ (ЛИДОКАИН) ---
    elif code == "G":
        _apply_base_preset(cfg, "A: Squid Giant Axon (HH 1952)", preserve_mode_fields=("anesthesia_mode",))
        _apply_anesthesia_mode(cfg)

    # --- 14. ГИПЕРКАЛИЕМИЯ (HIGH K+ REDUCES DRIVING FORCE) ---
    elif code == "H":
        _apply_base_preset(cfg, "A: Squid Giant Axon (HH 1952)")
        cfg.channels.EK = -55.0  # Elevated external K+: -77 → -55 (Nernst shift)
        # Pathology: depolarization block — elevated K+ reduces repolarization reserve
        cfg.stim.stim_type = 'const'
        cfg.stim.Iext = 15.0

    # --- 15. IN VITRO (SLICE AT 23C - SLOW KINETICS) ---
    elif code == "I":
        _apply_base_preset(cfg, "B: Pyramidal L5 (Mainen 1996)")
        cfg.morphology.single_comp = True  # Slice: isolated soma (cut dendrites)
        cfg.stim_location.location = "soma"  # Direct patch-clamp
        cfg.dendritic_filter.enabled = False  # No dendritic filtering in slice
        cfg.env.T_celsius = 23.0  # Room temperature: 37 → 23°C (phi drops to 1.0)
        # Validated: 10 µA/cm² → ~10 spikes, 65 Hz, Vmax ≈ 46 mV (slower than 37°C)
        cfg.stim.stim_type = 'const'
        cfg.stim.Iext = 10.0

    # --- 16. ТАЛАМИЧЕСКОЕ РЕТИКУЛЯРНОЕ ЯДРО (TRN — СОННЫЕ ВЕРЕТЁНА) ---
    elif code == "P":
        # TRN neurons: GABAergic burst-ready cells. In a single-neuron simulator
        # this is best represented as a rebound-burst surrogate rather than a
        # full spindle generator.
        # Reference: Destexhe et al. 1996, J Neurosci 16:169;
        #            Huguenard & Prince 1992, J Neurosci 12:3804
        cfg.morphology.single_comp = True
        cfg.stim_location.location = "soma"
        cfg.dendritic_filter.enabled = False
        cfg.channels.gNa_max, cfg.channels.gK_max, cfg.channels.gL = 60.0, 10.0, 0.05
        cfg.channels.ENa, cfg.channels.EK, cfg.channels.EL = 50.0, -90.0, -76.0
        cfg.env.T_celsius, cfg.env.T_ref, cfg.env.Q10 = 37.0, 24.0, 2.3
        # I_T: very high density in TRN dendrites (Huguenard & Prince 1992)
        # Somatic density ~3-5× higher than in relay neurons
        cfg.channels.enable_ITCa = True
        cfg.channels.gTCa_max = 7.0     # Larger T-current reserve for rebound bursts
        cfg.channels.enable_Ih = True
        cfg.channels.gIh_max = 0.005    # Lower Ih than relay — less sag, more burst
        cfg.channels.E_Ih = -43.0  # Standardized reversal potential
        cfg.channels.enable_ICa = False
        cfg.channels.enable_SK = False
        cfg.calcium.dynamic_Ca = True
        cfg.calcium.Ca_rest = 5e-5
        cfg.calcium.Ca_ext = 2.0
        cfg.calcium.tau_Ca = 150.0       # Fast extrusion
        cfg.calcium.B_Ca = 1e-5
        cfg.morphology.d_soma = 20e-4
        cfg.stim.jacobian_mode = 'native_hines'
        # Hyperpolarizing pulse de-inactivates I_T; release evokes the rebound burst.
        cfg.stim.stim_type = 'pulse'
        cfg.stim.Iext = -4.0
        cfg.stim.pulse_dur = 40.0

    # --- 17. ШИПОВАТЫЙ ПРОЕКЦИОННЫЙ НЕЙРОН СТРИАТУМА (SPN) ---
    elif code == "Q":
        # Medium spiny neuron (MSN/SPN): characteristic long latency to first
        # spike due to strong I_A and hyperpolarized resting potential.
        # Reference: Nisenbaum & Wilson 1995, J Neurophysiol 74:1163;
        #            Surmeier et al. 1989, Brain Res 473:187
        cfg.morphology.single_comp = False
        cfg.morphology.N_trunk = 15
        cfg.stim_location.location = "dendritic_filtered"
        cfg.dendritic_filter.enabled = True
        cfg.dendritic_filter.distance_um = 180.0
        cfg.dendritic_filter.tau_dendritic_ms = 10.0
        cfg.channels.gNa_max, cfg.channels.gK_max, cfg.channels.gL = 80.0, 8.0, 0.04
        cfg.channels.ENa, cfg.channels.EK, cfg.channels.EL = 50.0, -90.0, -80.0  # Very hyperpolarized rest
        cfg.env.T_celsius, cfg.env.T_ref, cfg.env.Q10 = 37.0, 23.0, 2.3
        # Strong I_A: key feature — delays spike onset by hundreds of ms
        cfg.channels.enable_IA = True
        cfg.channels.gA_max = 0.8       # High but not silencing: preserves long-latency firing
        # Ih: inward rectification contributes to ramp-like depolarization
        cfg.channels.enable_Ih = True
        cfg.channels.gIh_max = 0.01
        cfg.channels.E_Ih = -43.0  # Standardized reversal potential
        cfg.channels.enable_ICa = False
        cfg.channels.enable_SK = False
        cfg.calcium.dynamic_Ca = False
        cfg.morphology.d_soma = 15e-4   # Small soma (12-20 µm)
        cfg.stim.jacobian_mode = 'native_hines'
        # Tonic depolarization acts as an up-state proxy while I_A delays recruitment.
        cfg.stim.stim_type = 'const'
        cfg.stim.Iext = 22.0

    # --- 18. ХОЛИНЕРГИЧЕСКАЯ НЕЙРОМОДУЛЯЦИЯ (ACh: Бодрствование vs Сон) ---
    elif code == "R":
        # L5 pyramidal base + I_M. ACh blocks I_M → shift from adapting to tonic.
        # With I_M enabled: adapting/bursting (sleep-like, M-current dampens excitability).
        # With I_M blocked (gIM_max→0 via GUI): tonic high-frequency (awake/attentive).
        # Reference: Brown & Adams 1980, Nature 283:673;
        #            McCormick & Prince 1986, J Physiol 375:169
        _apply_base_preset(cfg, "B: Pyramidal L5 (Mainen 1996)", preserve_mode_fields=("ach_mode",))
        cfg.morphology.single_comp = True
        cfg.stim_location.location = "soma"
        cfg.dendritic_filter.enabled = False
        cfg.channels.gNa_max = 56.0
        # I_M: muscarinic-sensitive K+ current — the ACh target
        cfg.channels.enable_IM = True
        cfg.channels.gIM_max = 0.5      # Yamada 1989 somatic density
        # Moderate Ih for subthreshold dynamics
        cfg.channels.enable_Ih = True
        cfg.channels.gIh_max = 0.01
        cfg.channels.E_Ih = -43.0  # Standardized reversal potential
        cfg.stim.jacobian_mode = 'native_hines'
        # Const stimulus: with I_M → adapting; user blocks gIM → tonic firing
        cfg.stim.stim_type = 'const'
        cfg.stim.Iext = 5.0

    # --- 19. ПАТОЛОГИЯ: СИНДРОМ ДРАВЕ (SCN1A LOSS-OF-FUNCTION) ---
    elif code == "S":
        # FS interneuron with reduced gNa (SCN1A haploinsufficiency).
        # Paradox: Na-channel loss-of-function in inhibitory neurons causes
        # network hyperexcitability (disinhibition) → epileptic seizures.
        # Reference: Yu et al. 2006, Nat Neurosci 9:1142;
        #            Ogiwara et al. 2007, J Neurosci 27:5903
        _apply_base_preset(cfg, "C: FS Interneuron (Wang-Buzsaki)", preserve_mode_fields=("dravet_mode",))
        cfg.morphology.single_comp = True
        cfg.stim_location.location = "soma"
        cfg.dendritic_filter.enabled = False
        # SCN1A haploinsufficiency: ~50% reduction in Nav1.1
        # FS interneurons depend heavily on Nav1.1 for sustained firing
        cfg.channels.gNa_max = 60.0     # Reduced: 120 → 60 (50% Nav1.1 loss)
        cfg.channels.enable_IA = True
        cfg.channels.gA_max = 0.8       # FS baseline
        cfg.stim.jacobian_mode = 'native_hines'
        # Strong const stimulus: FS interneuron fails to maintain high-freq firing
        # → reduced inhibitory output → network disinhibition
        cfg.stim.stim_type = 'const'
        cfg.stim.Iext = 40.0            # Same drive as FS base, but fewer spikes

        # Fever mode: temperature-triggered failure
        if cfg.preset_modes.dravet_mode == "febrile":
            cfg.env.T_celsius = 40.0  # Fever temperature
            cfg.channels.gNa_max = 42.0  # Further 30% reduction from baseline (60 → 42)
            cfg.env.Q10_Na = 3.0  # Thermal hyper-sensitivity: Na channels become more sensitive to heat

    # --- 20. PASSIVE CABLE (LINEAR DECAY) ---
    elif code == "T":
        # Pure passive membrane: no active channels, only leak conductance.
        # Shows exponential voltage decay V = V_0 * exp(-x/lambda) without spikes.
        # Reference: Rall 1959, Cable theory for dendrites
        cfg.morphology.single_comp = False
        cfg.stim_location.location = "dendritic_filtered"
        cfg.dendritic_filter.enabled = True
        cfg.dendritic_filter.distance_um = 200.0
        cfg.dendritic_filter.space_constant_um = 200.0
        cfg.dendritic_filter.tau_dendritic_ms = 10.0

        # Disable ALL active channels
        cfg.channels.gNa_max = 0.0
        cfg.channels.gK_max = 0.0
        cfg.channels.enable_NaR = False
        cfg.channels.enable_IM = False
        cfg.channels.enable_Ih = False
        cfg.channels.enable_IA = False
        cfg.channels.enable_SK = False
        cfg.channels.enable_ICa = False
        cfg.channels.enable_ITCa = False

        # Only leak conductance (high resistance for clear decay)
        cfg.channels.gL = 0.1
        cfg.channels.EL = -65.0

        # Morphology for cable properties
        cfg.channels.Cm = 1.0  # Standard membrane capacitance
        cfg.channels.ENa, cfg.channels.EK = 50.0, -90.0
        cfg.env.T_celsius = 37.0

        cfg.morphology.d_soma = 20e-4
        cfg.morphology.N_trunk = 10     # Standardized reduced model
        cfg.morphology.N_ais = 0
        cfg.morphology.d_trunk = 2e-4
        cfg.morphology.Ra = 150.0

        cfg.stim.jacobian_mode = 'native_hines'
        # Const stimulus at distal branch to show decay
        cfg.stim.stim_type = 'const'
        cfg.stim.Iext = 10.0

    # --- 21. I-CLAMP THRESHOLD EXPLORER ---
    elif code == "U":
        # Standard HH model with high-resolution sampling for threshold exploration.
        # Uses simulation (220ms) with standard dt (0.05ms) to capture exact rheobase.
        cfg.morphology.single_comp = True
        cfg.morphology.d_soma = 500e-4  # 500 µm giant axon diameter
        cfg.channels.gNa_max, cfg.channels.gK_max, cfg.channels.gL = 120.0, 36.0, 0.3
        cfg.channels.ENa, cfg.channels.EK, cfg.channels.EL = 50.0, -77.0, -54.387
        cfg.channels.Cm = 1.0
        cfg.env.T_celsius, cfg.env.T_ref, cfg.env.Q10 = 6.3, 6.3, 3.0
        cfg.stim_location.location = "soma"
        cfg.dendritic_filter.enabled = False
        cfg.stim.t_sim = 220.0
        cfg.stim.dt_eval = 0.05
        cfg.stim.stim_type = 'const'
        cfg.stim.Iext = 10.0  # Rheobase for HH model (slightly above threshold)
        cfg.stim.jacobian_mode = 'native_hines'

    # Stage/mode overlays for selected presets.
    if code == "K":
        _apply_k_mode(cfg)
    if code == "O":
        _apply_hypoxia_mode(cfg)
    if code == "N":
        _apply_alzheimer_mode(cfg)
    if code == "R":
        _apply_ach_mode(cfg)
    if code == "E":
        _apply_purkinje_mode(cfg)
    if code == "G":
        _apply_anesthesia_mode(cfg)



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
    """Apply a synaptic stimulus profile on top of the current neuron config."""
    cfg.stim.stim_type = 'alpha'
    cfg.stim.pulse_start = 10.0

    if "AMPA" in stimulus_type:
        cfg.stim.stim_type = 'AMPA'
        cfg.stim.alpha_tau = 1.0
        cfg.stim.Iext = 1.5
        cfg.stim.t_sim = 100.0
    elif "NMDA" in stimulus_type:
        cfg.stim.stim_type = 'NMDA'
        cfg.stim.alpha_tau = 1.0
        cfg.stim.Iext = 6.0
        cfg.stim.t_sim = 500.0
    elif "Kainate" in stimulus_type:
        cfg.stim.stim_type = 'Kainate'
        cfg.stim.alpha_tau = 1.0
        cfg.stim.Iext = 1.2
        cfg.stim.t_sim = 200.0
    elif "GABA-A" in stimulus_type:
        cfg.stim.stim_type = 'GABAA'
        cfg.stim.alpha_tau = 1.0
        cfg.stim.Iext = 1.5
        cfg.stim.t_sim = 100.0
    elif "GABA-B" in stimulus_type:
        cfg.stim.stim_type = 'GABAB'
        cfg.stim.alpha_tau = 1.0
        cfg.stim.Iext = 2.5
        cfg.stim.t_sim = 400.0
    elif "Nicotinic" in stimulus_type:
        cfg.stim.stim_type = 'Nicotinic'
        cfg.stim.alpha_tau = 1.0
        cfg.stim.Iext = 1.8
        cfg.stim.t_sim = 150.0
    else:
        raise ValueError(f"Unknown synaptic stimulus type: {stimulus_type}")
