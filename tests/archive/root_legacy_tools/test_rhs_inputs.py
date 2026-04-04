"""test_rhs_inputs.py - Check what RHS receives"""

from core.solver import NeuronSolver
from core.models import FullModelConfig
from core.presets import apply_preset
import numpy as np

# Monkey-patch RHS to print first few calls
from core import rhs
original_rhs = rhs.rhs_multicompartment

call_count = [0]  # Use list to allow modification in nested function

@rhs.njit(cache=False)  # Disable cache for debug
def debug_rhs(t, y, n_comp,
    en_ih, en_ica, en_ia, en_sk, dyn_ca,
    gna_v, gk_v, gl_v, gih_v, gca_v, ga_v, gsk_v,
    ena, ek, el, eih, ea,
    cm_v, l_data, l_indices, l_indptr,
    phi, t_kelvin, ca_ext, ca_rest, tau_ca, b_ca,
    stype, iext, t0, td, atau, stim_comp, stim_mode,
    use_dfilter, dfilter_attenuation, dfilter_tau_ms,
    dual_stim_enabled,
    stype_2, iext_2, t0_2, td_2, atau_2, stim_comp_2, stim_mode_2,
    dfilter_attenuation_2, dfilter_tau_ms_2
):
    """Debug wrapper for RHS - just check parameters passed"""
    # This won't actually print in Numba, so we'll just call original
    return original_rhs(t, y, n_comp,
        en_ih, en_ica, en_ia, en_sk, dyn_ca,
        gna_v, gk_v, gl_v, gih_v, gca_v, ga_v, gsk_v,
        ena, ek, el, eih, ea,
        cm_v, l_data, l_indices, l_indptr,
        phi, t_kelvin, ca_ext, ca_rest, tau_ca, b_ca,
        stype, iext, t0, td, atau, stim_comp, stim_mode,
        use_dfilter, dfilter_attenuation, dfilter_tau_ms,
        dual_stim_enabled,
        stype_2, iext_2, t0_2, td_2, atau_2, stim_comp_2, stim_mode_2,
        dfilter_attenuation_2, dfilter_tau_ms_2
    )

# Can't easily patch Numba functions, so let's just print solver args instead

cfg = FullModelConfig()
apply_preset(cfg, 'B: Pyramidal L5 (Mainen 1996)')

print("Config after L5 preset:")
print(f"  stim.stim_type: {cfg.stim.stim_type}")
print(f"  stim.Iext: {cfg.stim.Iext}")
print(f"  stim.pulse_start: {cfg.stim.pulse_start}")
print(f"  stim.pulse_dur: {cfg.stim.pulse_dur}")
print(f"  stim.alpha_tau: {cfg.stim.alpha_tau}")
print(f"  stim.stim_comp: {cfg.stim.stim_comp}")
print(f"  stim_location.location: {cfg.stim_location.location}")
print(f"  dendritic_filter.enabled: {cfg.dendritic_filter.enabled}")
print(f"  dendritic_filter.distance_um: {cfg.dendritic_filter.distance_um}")
print(f"  dendritic_filter.space_constant_um: {cfg.dendritic_filter.space_constant_um}")
print(f"  dendritic_filter.tau_dendritic_ms: {cfg.dendritic_filter.tau_dendritic_ms}")

print(f"\nDual stim config:")
print(f"  dual_stimulation: {cfg.dual_stimulation}")

# Now let's manually check args that will be passed to RHS
from core.morphology import MorphologyBuilder
from core.channels import ChannelRegistry

morph = MorphologyBuilder.build(cfg)
n_comp = morph['N_comp']
print(f"\nMorphology:")
print(f"  n_comp: {n_comp}")
print(f"  gNa_v[0]: {morph['gNa_v'][0]}")
print(f"  gK_v[0]: {morph['gK_v'][0]}")
print(f"  gL_v[0]: {morph['gL_v'][0]}")

s_map = {
    'const': 0, 'pulse': 1, 'alpha': 2, 'ou_noise': 3,
    'AMPA': 4, 'NMDA': 5, 'GABAA': 6, 'GABAB': 7,
    'Kainate': 8, 'Nicotinic': 9,
}
stype = s_map.get(cfg.stim.stim_type, 0)
print(f"\nStimulation mapping:")
print(f"  stim.stim_type '{cfg.stim.stim_type}' -> stype: {stype}")
print(f"  cfg.stim.Iext (iext): {cfg.stim.Iext}")
print(f"  cfg.stim.pulse_start (t0): {cfg.stim.pulse_start}")
print(f"  cfg.stim.pulse_dur (td): {cfg.stim.pulse_dur}")
print(f"  cfg.stim.alpha_tau (atau): {cfg.stim.alpha_tau}")

stim_mode_map = {'soma': 0, 'ais': 1, 'dendritic_filtered': 2}
stim_mode = stim_mode_map.get(cfg.stim_location.location, 0)
use_dfilter = int(stim_mode == 2 and cfg.dendritic_filter.enabled)
print(f"\nStimulation location mapping:")
print(f"  stim_location.location '{cfg.stim_location.location}' -> stim_mode: {stim_mode}")
print(f"  use_dfilter: {use_dfilter}")

if use_dfilter == 1 and cfg.dendritic_filter.space_constant_um > 0:
    attenuation = np.exp(
        -cfg.dendritic_filter.distance_um / cfg.dendritic_filter.space_constant_um  
    )
else:
    attenuation = 1.0

print(f"  dfilter_attenuation: {attenuation}")
print(f"  dfilter_tau_ms: {cfg.dendritic_filter.tau_dendritic_ms if use_dfilter == 1 else 0.0}")
