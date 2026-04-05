from pydantic import BaseModel, Field, model_validator
from typing import Literal, List, Optional, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .dual_stimulation import DualStimulationConfig


class MorphologyParams(BaseModel):
    """Neuron geometry: soma, AIS, axon trunk and branches."""
    single_comp: bool   = Field(default=False, description="Single-compartment soma (0-D)")
    d_soma: float       = Field(default=50e-4,  gt=0, description="Soma diameter (cm)")

    # AIS (Axon Initial Segment)
    N_ais: int          = Field(default=2,   ge=0,  description="Number of AIS segments")
    d_ais: float        = Field(default=1.5e-4, gt=0, description="AIS diameter (cm)")
    gNa_ais_mult: float = Field(default=40.0, ge=1.0, description="gNa multiplier in AIS (40–100×)")
    gK_ais_mult:  float = Field(default=5.0,  ge=1.0, description="gK multiplier in AIS")
    gIh_ais_mult: float = Field(default=1.0,           description="gIh multiplier in AIS")
    gCa_ais_mult: float = Field(default=2.0,           description="gCa multiplier in AIS")
    gA_ais_mult:  float = Field(default=3.0,           description="gA multiplier in AIS")
    gM_ais_mult:  float = Field(default=1.0,           description="gM multiplier in AIS")

    # Axon trunk
    N_trunk: int   = Field(default=35, ge=0, description="Number of trunk segments")
    d_trunk: float = Field(default=2e-4,  gt=0, description="Trunk diameter (cm)")

    # Bifurcation and branches (Rall's law)
    N_b1: int  = Field(default=4,    ge=0, description="Branch 1 segments")
    d_b1: float = Field(default=1.5e-4, gt=0, description="Branch 1 diameter (cm)")
    N_b2: int  = Field(default=4,    ge=0, description="Branch 2 segments")
    d_b2: float = Field(default=1.0e-4, gt=0, description="Branch 2 diameter (cm)")

    Ra: float = Field(default=150.0, gt=0, description="Axial resistance (Ω·cm)")
    dx: float = Field(default=0.002, gt=0, description="Segment length (cm)")


class ChannelParams(BaseModel):
    """Ion channel conductances and reversal potentials.
    
    All conductances are stored as density (mS/cm²) for consistency with the 
    multi-compartment system. They are not pre-scaled by area.
    """
    Cm:       float = Field(default=1.0,      gt=0, description="Membrane capacitance (µF/cm²)")
    gNa_max:  float = Field(default=120.0,    ge=0, description="Max Na conductance (mS/cm²)")
    gK_max:   float = Field(default=36.0,     ge=0, description="Max K conductance (mS/cm²)")
    gL:       float = Field(default=0.3,      ge=0, description="Leak conductance (mS/cm²)")
    ENa:      float = Field(default=50.0,           description="Na reversal potential (mV)")
    EK:       float = Field(default=-77.0,          description="K reversal potential (mV)")
    EL:       float = Field(default=-54.387,         description="Leak reversal potential (mV)")

    enable_Ih: bool  = Field(default=False, description="Enable Ih (HCN pacemaker current)")
    gIh_max:   float = Field(default=0.02,  description="Max Ih conductance (mS/cm²)")
    E_Ih:      float = Field(default=-30.0, description="Ih reversal potential (mV)")

    enable_ICa: bool  = Field(default=False, description="Enable L-type Ca²⁺ current")
    gCa_max:    float = Field(default=1.0, description="Max Ca conductance (mS/cm²)")
    E_Ca:       float = Field(default=120.0, description="Ca reversal potential (mV) [overridden by Nernst if dynamic Ca]")

    enable_IA: bool  = Field(default=False, description="Enable A-current (transient K⁺, Connor-Stevens)")
    gA_max:    float = Field(default=0.4, description="Max A-current conductance (mS/cm²)")  # Default 0.4, NOT 10.0 (was unphysiologically high)
    E_A:       float = Field(default=-77.0, description="A-current reversal potential (mV)")

    enable_SK: bool  = Field(default=False, description="Enable SK channel (Ca-activated K⁺, spike adaptation)")
    gSK_max:   float = Field(default=2.0,  description="Max SK conductance (mS/cm²)")

    enable_ITCa: bool  = Field(default=False, description="Enable T-type Ca²⁺ current (low-threshold, CaV3.x)")
    gTCa_max:    float = Field(default=2.0,  description="Max T-type Ca conductance (mS/cm², Destexhe 1998)")

    enable_IM: bool  = Field(default=False, description="Enable M-current (KCNQ/Kv7, spike-frequency adaptation)")
    gM_max:    float = Field(default=0.02, description="Max M-current conductance (mS/cm², Yamada/Koch/Adams 1989)")


class CalciumParams(BaseModel):
    """Intracellular calcium dynamics."""
    dynamic_Ca: bool  = Field(default=False, description="Track [Ca²⁺]ᵢ with Nernst correction")
    Ca_ext:     float = Field(default=2.0,   description="Extracellular [Ca²⁺] (mM)")
    Ca_rest:    float = Field(default=50e-6, description="Resting [Ca²⁺]ᵢ (mM)")
    tau_Ca:     float = Field(default=200.0, description="Ca²⁺ pump time constant (ms)")
    B_Ca:       float = Field(default=0.001, description="Current-to-concentration conversion")


class EnvironmentParams(BaseModel):
    """Thermodynamics and temperature scaling."""
    T_celsius: float = Field(default=6.3,  ge=0, le=45.0, description="Experiment temperature (°C)")
    T_ref:     float = Field(default=6.3,              description="Reference temperature for kinetics (°C)")
    Q10:       float = Field(default=3.0,              description="Global Q10 (legacy, used as fallback)")
    # Channel-specific Q10 values (literature-based defaults)
    # Na: Hodgkin & Huxley 1952, J Physiol 117:500-544 (Q10 ~ 2.2-3.0 for m/h)
    Q10_Na:    float = Field(default=2.2,              description="Q10 for Na channels (m,h gates)")
    # K: Hodgkin & Huxley 1952 (Q10 ~ 3.0 for n gate)
    Q10_K:     float = Field(default=3.0,              description="Q10 for K channels (n gate)")
    # Ih: Magee 1998, J Neurosci 18:7613; Destexhe 1993 (Q10 ~ 4.0-4.5)
    Q10_Ih:    float = Field(default=4.0,              description="Q10 for HCN/Ih channel (r gate)")
    # Ca L-type: Coulter et al 1989, J Physiol 414:587 (Q10 ~ 2.3-3.0)
    Q10_Ca:    float = Field(default=2.5,              description="Q10 for Ca channels (s,u gates)")
    # IA: Huguenard et al 1991, J Neurophysiol 65:1271 (Q10 ~ 2.8-3.5)
    Q10_IA:    float = Field(default=3.0,              description="Q10 for A-type K channel (a,b gates)")
    # T-type Ca: Destexhe 1998, J Neurosci 18:3574 — Q10_m=5.0, Q10_h=3.0 at 24°C ref.
    # Using geometric mean ~3.9 for combined gate scaling in single-phi model.
    Q10_TCa:   float = Field(default=3.9,              description="Q10 for T-type Ca channel (m,h gates)")
    # I_M: Yamada, Koch & Adams 1989; Destexhe IM.mod (Q10 = 2.3, ref 36°C)
    Q10_M:     float = Field(default=2.3,              description="Q10 for M-current (w gate)")

    @property
    def phi(self) -> float:
        """Legacy global phi (backward compatibility)."""
        return self.Q10 ** ((self.T_celsius - self.T_ref) / 10.0)

    def phi_channel(self, q10: float) -> float:
        """Per-channel temperature scaling: q10^((T - T_ref)/10)."""
        return q10 ** ((self.T_celsius - self.T_ref) / 10.0)


class DendriticFilterParams(BaseModel):
    """Dendritic cable filtering for synaptic inputs.
    
    Instead of directly injecting current into soma, the dendritic filter mode
    applies a CURRENT SOURCE TEST where stimulus is applied at dendrites,
    then filtered through cable properties (exponential decay + low-pass temporal filter)
    before reaching the soma.
    
    Physics:
    1. Amplitude attenuation: A(d) = exp(-distance/space_constant)
       - Models cable decay with distance from stimulation site
       - Typical parameters: exp(-150/150) ≈ 0.368 (63% loss)
    
    2. Temporal filtering: tau_dendritic determines the low-pass corner frequency
       - Fast components (synaptic transients): heavily attenuated
       - Slow components (sustained input): passed through
       - Results in more realistic somatic response
    
    Default values are PHYSIOLOGICALLY REALISTIC for L5 pyramidal neurons:
    - distance_um: 150 µm (middle-range dendrite, soma to proximal/distal transition)
    - space_constant_um: 150 µm (cable space constant for soma-proximal dendrites)
    - tau_dendritic_ms: 10 ms (dendritic integration time scale)
    """
    
    enabled: bool = Field(default=True, description="Enable dendritic filtering")
    distance_um: float = Field(
        default=150.0, gt=0,
        description="Distance soma→synapse stimulation site (µm). Typical range: 50-300 µm"
    )
    space_constant_um: float = Field(
        default=150.0, gt=0,
        description="Cable space constant λ (µm). λ = sqrt(a/(4ρ*g_m)). Typical: 100-200 µm"
    )
    tau_dendritic_ms: float = Field(
        default=10.0, gt=0,
        description="Dendritic low-pass filter time constant τ (ms). Typical: 5-20 ms"
    )


class StimulationLocationParams(BaseModel):
    """Configuration of stimulus injection location and mode.
    
    THREE MODES:
    1. 'soma': Direct injection into soma (soma current clamp)
       - Laboratory condition (whole-cell patch-clamp)
       - UNPHYSIOLOGICAL for natural neuron operation
       - High current (4× rheobase) gives high-amplitude spikes
       
    2. 'ais': Direct injection into AIS (axon initial segment)
       - Very sensitive due to 40× higher Na density
       - Minimal current needed for spike
       - Models direct axonal stimulation (electrode on axon)
       
    3. 'dendritic_filtered': Injection on dendrites with cable filtering
       - PHYSIOLOGICAL mode (models synaptic inputs)
       - Current attenuated by exponential decay (cable properties)
       - Temporally filtered (low-pass)
       - Requires larger input current (~100 µA/cm²) because of attenuation
       - Results in realistic spike amplitude and firing rates
    """
    
    location: Literal['soma', 'ais', 'dendritic_filtered'] = Field(
        default='soma',
        description="Stimulus location: soma (direct), ais (axonal), or dendritic_filtered (synaptic)"
    )


class SimulationParams(BaseModel):
    """Solver settings, stimulation, and noise.

    Note: The system operates in a density-based mode for backward compatibility.
    Iext is in µA/cm² (current density). The unit_converter module provides
    display conversions for user-facing interfaces.

    For GUI display, absolute current in nanoamperes (nA) is computed as:
        I_absolute_nA = Iext * Area_soma_cm² * 1000
        Area_soma_cm² = π * (d_soma_cm)²
    """
    t_sim:      float   = Field(default=150.0, gt=0,  description="Simulation duration (ms)")
    dt_eval:    float   = Field(default=0.05,  gt=0,  description="Output time step (ms)")
    jacobian_mode: Literal['dense_fd', 'sparse_fd', 'analytic_sparse'] = Field(
        default='dense_fd',
        description="Jacobian handling for BDF: dense finite-diff, sparse finite-diff, or analytic sparse"
    )
    stim_type:  Literal[
        'const', 'pulse', 'alpha', 'ou_noise',
        'AMPA', 'NMDA', 'GABAA', 'GABAB',
        'Kainate', 'Nicotinic'
    ] = Field(
        default='const',
        description=(
            "Stimulus waveform: constant / pulse / alpha-synapse / OU noise / "
            "synaptic receptors (AMPA, NMDA, GABAA, GABAB, Kainate, Nicotinic)"
        )
    )
    Iext:       float   = Field(default=10.0,        description="Stimulus amplitude (uA/cm2 density)")
    Iext_absolute_nA: float = Field(default=0.0,     description="Stimulus absolute current (nanoamperes, for GUI display only)")
    pulse_start: float  = Field(default=10.0,        description="Pulse onset (ms)")
    pulse_dur:   float  = Field(default=1.0,         description="Pulse duration (ms)")
    alpha_tau:   float  = Field(default=2.0,         description="Alpha-synapse time constant (ms)")
    stim_comp:   int    = Field(default=0,           description="Compartment index to inject current")

    # Stochastic
    stoch_gating: bool  = Field(default=False, description="Langevin gate noise (Euler-Maruyama solver)")
    noise_sigma:  float = Field(default=0.0,   description="Additive membrane current noise sigma (uA/cm2)")


class AnalysisParams(BaseModel):
    """Analysis, bifurcation, Monte-Carlo, Sweep, S-D curve, Excitability map."""
    # Spike detection controls (used by passport/analytics and GUI)
    spike_detect_algorithm: Literal['peak_repolarization', 'threshold_crossing', 'fsm'] = Field(
        default='peak_repolarization',
        description="Spike detection algorithm: peak+repolarization, threshold-crossing, or fsm (state machine)"
    )
    spike_detect_threshold: float = Field(
        default=-20.0,
        description="Spike detection threshold (mV)"
    )
    spike_detect_prominence: float = Field(
        default=10.0,
        ge=0.0,
        description="Minimum peak prominence for peak-based detector (mV)"
    )
    spike_detect_baseline_threshold: float = Field(
        default=-50.0,
        description="Voltage that must be crossed during repolarization check (mV)"
    )
    spike_detect_repolarization_window_ms: float = Field(
        default=20.0,
        gt=0.0,
        description="Repolarization validation window (ms)"
    )
    spike_detect_refractory_ms: float = Field(
        default=1.0,
        gt=0.0,
        description="Minimum inter-spike interval for detector de-duplication (ms)"
    )

    # Monte-Carlo
    run_mc:    bool = Field(default=False, description="Run parallel Monte-Carlo")
    mc_trials: int  = Field(default=50,   ge=1, description="Number of MC trials")

    # Bifurcation
    run_bifurcation: bool  = Field(default=False)
    bif_param:       str   = Field(default="Iext",  description="Bifurcation parameter name")
    bif_min:         float = Field(default=0.0)
    bif_max:         float = Field(default=25.0)
    bif_steps:       int   = Field(default=60,  ge=2)

    # Parameter Sweep
    run_sweep:    bool  = Field(default=False, description="Run parametric sweep")
    sweep_param:  str   = Field(default="Iext", description="Parameter to sweep")
    sweep_min:    float = Field(default=0.0,   description="Sweep start value")
    sweep_max:    float = Field(default=20.0,  description="Sweep end value")
    sweep_steps:  int   = Field(default=20,   ge=2, description="Number of sweep steps")

    # Strength-Duration Curve
    run_sd_curve: bool = Field(default=False, description="Compute Strength-Duration curve")

    # Excitability Map
    run_excmap:   bool  = Field(default=False, description="Compute 2-D excitability map")
    excmap_I_min: float = Field(default=0.5,   description="Excitability map: min current (µA/cm²)")
    excmap_I_max: float = Field(default=20.0,  description="Excitability map: max current (µA/cm²)")
    excmap_NI:    int   = Field(default=15,   ge=2, description="Excitability map: current resolution")
    excmap_D_min: float = Field(default=0.1,   description="Excitability map: min duration (ms)")
    excmap_D_max: float = Field(default=5.0,   description="Excitability map: max duration (ms)")
    excmap_ND:    int   = Field(default=15,   ge=2, description="Excitability map: duration resolution")

    # Lyapunov / FTLE analysis (default OFF)
    enable_lyapunov: bool = Field(default=False, description="Enable FTLE/LLE stability analysis")
    lyapunov_embedding_dim: int = Field(default=3, ge=2, le=8, description="Delay-embedding dimension for FTLE/LLE")
    lyapunov_lag_steps: int = Field(default=2, ge=1, le=50, description="Delay-embedding lag in samples")
    lyapunov_min_separation_ms: float = Field(default=10.0, ge=0.0, description="Minimum temporal separation for neighbor search (ms)")
    lyapunov_fit_start_ms: float = Field(default=5.0, ge=0.0, description="Linear-fit window start for FTLE/LLE (ms)")
    lyapunov_fit_end_ms: float = Field(default=40.0, gt=0.0, description="Linear-fit window end for FTLE/LLE (ms)")

    # Non-FFT modulatory contribution analysis (default OFF)
    enable_modulation_decomposition: bool = Field(default=False, description="Enable phase-based modulation analysis")
    modulation_source: Literal['voltage', 'stimulus'] = Field(default='voltage', description="Signal used as low-frequency modulator proxy")
    modulation_low_hz: float = Field(default=4.0, gt=0.0, description="Low cutoff of modulatory band (Hz)")
    modulation_high_hz: float = Field(default=12.0, gt=0.0, description="High cutoff of modulatory band (Hz)")
    modulation_phase_bins: int = Field(default=18, ge=8, le=72, description="Phase-bin count for phase-rate profile")
    modulation_surrogates: int = Field(default=60, ge=0, le=500, description="Number of surrogate shuffles for significance estimate")


class PresetModeParams(BaseModel):
    """Switchable physiology modes for selected presets."""
    k_mode: Literal['activated', 'baseline'] = Field(
        default='baseline',
        description="Thalamic relay mode: activated (high drive) or baseline (low drive)"
    )
    alzheimer_mode: Literal['progressive', 'terminal'] = Field(
        default='progressive',
        description="Alzheimer preset stage: progressive (early spikes then decay) or terminal"
    )
    hypoxia_mode: Literal['progressive', 'terminal'] = Field(
        default='progressive',
        description="Hypoxia preset stage: progressive (early spikes then decay) or terminal"
    )


class FullModelConfig(BaseModel):
    """Master configuration container v10.1 - with dendritic filtering support."""
    morphology: MorphologyParams = MorphologyParams()
    channels:   ChannelParams    = ChannelParams()
    calcium:    CalciumParams     = CalciumParams()
    env:        EnvironmentParams = EnvironmentParams()
    stim:       SimulationParams  = SimulationParams()
    stim_location: StimulationLocationParams = StimulationLocationParams()
    dendritic_filter: DendriticFilterParams = DendriticFilterParams()
    dual_stimulation: Optional[Any] = None  # Optional dual stimulation config (DualStimulationConfig or None)
    analysis:   AnalysisParams    = AnalysisParams()
    preset_modes: PresetModeParams = PresetModeParams()


# Rebuild FullModelConfig after importing DualStimulationConfig (optional)
try:
    from .dual_stimulation import DualStimulationConfig
except ImportError:
    pass
