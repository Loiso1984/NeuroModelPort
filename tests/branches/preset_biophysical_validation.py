"""
Biophysical Literature Validation for NeuroModelPort Presets

Validates preset parameters and simulation outputs against published literature data.

Usage:
    python tests/branches/preset_biophysical_validation.py

References:
    - Hodgkin & Huxley 1952 (Squid Giant Axon)
    - Mainen et al. 1996 (L5 Pyramidal)
    - Wang & Buzsaki 1996 (FS Interneurons)
    - Powers et al. 2001 (Alpha-Motoneuron)
    - De Schutter & Bower 1994 (Purkinje)
    - McCormick & Huguenard 1992 (Thalamic Relay)
    - Magee 1998 (CA1 Pyramidal)
    - Nisenbaum & Wilson 1995 (Striatal SPN)
"""

import sys
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.insert(0, 'c:\\NeuroModelPort')

from core.models import FullModelConfig
from core.presets import apply_preset, get_preset_names
from core.solver import NeuronSolver
from core.analysis import full_analysis


@dataclass
class LiteratureReference:
    """Literature reference with expected physiological ranges."""
    name: str
    citation: str
    rheobase_range: Tuple[float, float]  # µA/cm²
    spike_freq_range: Tuple[float, float]  # Hz at 2x rheobase
    ap_amplitude_range: Tuple[float, float]  # mV
    ap_width_range: Tuple[float, float]  # ms at half-amplitude
    input_resistance: Optional[Tuple[float, float]] = None  # MΩ
    resting_potential_range: Optional[Tuple[float, float]] = None  # mV


# Literature database for validation
LITERATURE_DB = {
    "A": LiteratureReference(
        name="Squid Giant Axon (HH 1952)",
        citation="Hodgkin & Huxley 1952, J Physiol 117:500-544",
        rheobase_range=(6.0, 10.0),
        spike_freq_range=(40.0, 80.0),
        ap_amplitude_range=(90.0, 110.0),
        ap_width_range=(1.0, 1.5),
        resting_potential_range=(-65.0, -60.0)
    ),
    "B": LiteratureReference(
        name="Pyramidal L5 (Mainen 1996)",
        citation="Mainen et al. 1996, Nature 382:363-366",
        rheobase_range=(4.0, 8.0),
        spike_freq_range=(10.0, 40.0),
        ap_amplitude_range=(80.0, 100.0),
        ap_width_range=(0.8, 1.5),
        input_resistance=(50.0, 150.0),
        resting_potential_range=(-75.0, -65.0)
    ),
    "C": LiteratureReference(
        name="FS Interneuron (Wang-Buzsaki)",
        citation="Wang & Buzsaki 1996, J Neurosci 16:6402-6413",
        rheobase_range=(2.0, 6.0),
        spike_freq_range=(100.0, 300.0),
        ap_amplitude_range=(70.0, 90.0),
        ap_width_range=(0.3, 0.6),
        input_resistance=(20.0, 80.0),
        resting_potential_range=(-70.0, -60.0)
    ),
    "D": LiteratureReference(
        name="Alpha-Motoneuron (Powers 2001)",
        citation="Powers et al. 2001, J Neurophysiol 85:1109-1120",
        rheobase_range=(2.0, 8.0),
        spike_freq_range=(10.0, 40.0),
        ap_amplitude_range=(75.0, 95.0),
        ap_width_range=(1.0, 2.0),
        input_resistance=(0.5, 2.0),  # Lower due to large soma
        resting_potential_range=(-70.0, -60.0)
    ),
    "E": LiteratureReference(
        name="Cerebellar Purkinje (De Schutter)",
        citation="De Schutter & Bower 1994, J Neurophysiol 71:401-419",
        rheobase_range=(10.0, 40.0),
        spike_freq_range=(30.0, 80.0),
        ap_amplitude_range=(60.0, 80.0),
        ap_width_range=(0.4, 0.8),
        input_resistance=(20.0, 60.0),
        resting_potential_range=(-65.0, -55.0)
    ),
    "K": LiteratureReference(
        name="Thalamic Relay (McCormick)",
        citation="McCormick & Huguenard 1992, J Neurophysiol 68:1384-1400",
        rheobase_range=(50.0, 150.0),  # Higher due to Ih activation
        spike_freq_range=(5.0, 25.0),
        ap_amplitude_range=(70.0, 90.0),
        ap_width_range=(0.8, 1.5),
        input_resistance=(30.0, 100.0),
        resting_potential_range=(-65.0, -55.0)
    ),
    "L": LiteratureReference(
        name="Hippocampal CA1 Pyramidal (Magee)",
        citation="Magee 1998, J Neurosci 18:7613-7625",
        rheobase_range=(20.0, 60.0),
        spike_freq_range=(5.0, 25.0),
        ap_amplitude_range=(80.0, 100.0),
        ap_width_range=(1.0, 2.0),
        input_resistance=(80.0, 200.0),
        resting_potential_range=(-70.0, -60.0)
    ),
    "Q": LiteratureReference(
        name="Striatal SPN (Nisenbaum)",
        citation="Nisenbaum & Wilson 1995, J Neurophysiol 74:1163-1177",
        rheobase_range=(50.0, 200.0),  # Very high rheobase due to strong IA
        spike_freq_range=(1.0, 15.0),
        ap_amplitude_range=(70.0, 90.0),
        ap_width_range=(1.0, 2.5),
        input_resistance=(50.0, 150.0),
        resting_potential_range=(-85.0, -75.0)
    ),
}


class PresetValidator:
    """Validates presets against literature data."""
    
    def __init__(self):
        self.results = {}
        self.warnings = []
        self.errors = []
        
    def _preset_code(self, name: str) -> str:
        """Extract preset code from name."""
        if not isinstance(name, str):
            return ""
        head, _, _ = name.partition(":")
        return head.strip()
    
    def run_simulation_and_analyze(self, preset_name: str) -> Optional[Dict]:
        """Run simulation and extract key metrics."""
        try:
            cfg = FullModelConfig()
            apply_preset(cfg, preset_name)
            
            # Run simulation
            solver = NeuronSolver(cfg)
            result = solver.run_single()
            
            if result is None or not hasattr(result, 'v'):
                return None
            
            # Extract basic metrics
            v = result.v
            t = result.t
            
            # Resting potential (first 10ms before any stimulation)
            v_rest = np.mean(v[:int(10/cfg.stim.dt_eval)]) if len(v) > 100 else v[0]
            
            # Spike detection
            spike_threshold = -20.0  # mV
            spike_indices = np.where((v[:-1] < spike_threshold) & (v[1:] >= spike_threshold))[0]
            n_spikes = len(spike_indices)
            
            # Spike frequency
            sim_duration_ms = cfg.stim.t_sim
            spike_freq = (n_spikes / sim_duration_ms) * 1000 if sim_duration_ms > 0 else 0
            
            # AP amplitude (if spikes present)
            if n_spikes > 0:
                ap_peaks = []
                for idx in spike_indices:
                    if idx + 50 < len(v):
                        peak_idx = idx + np.argmax(v[idx:idx+50])
                        ap_peaks.append(v[peak_idx])
                ap_amplitude = np.mean(ap_peaks) - v_rest if ap_peaks else 0
            else:
                ap_amplitude = 0
            
            return {
                'preset_name': preset_name,
                'v_rest': v_rest,
                'n_spikes': n_spikes,
                'spike_freq': spike_freq,
                'ap_amplitude': ap_amplitude,
                'sim_duration_ms': sim_duration_ms,
                'stim_type': cfg.stim.stim_type,
                'iext': cfg.stim.Iext,
                'gNa_max': cfg.channels.gNa_max,
                'gK_max': cfg.channels.gK_max,
                'gL': cfg.channels.gL,
                'ENa': cfg.channels.ENa,
                'EK': cfg.channels.EK,
                'EL': cfg.channels.EL,
                'Cm': cfg.channels.Cm,
                'T_celsius': cfg.env.T_celsius,
            }
        except Exception as e:
            self.errors.append(f"{preset_name}: Simulation failed - {str(e)}")
            return None
    
    def validate_preset(self, preset_name: str) -> Dict:
        """Validate a single preset against literature."""
        code = self._preset_code(preset_name)
        result = self.run_simulation_and_analyze(preset_name)
        
        if result is None:
            return {'status': 'ERROR', 'preset': preset_name}
        
        validation = {
            'preset': preset_name,
            'code': code,
            'status': 'PASS',
            'warnings': [],
            'errors': [],
            'metrics': result
        }
        
        # Check against literature if available
        if code in LITERATURE_DB:
            ref = LITERATURE_DB[code]
            
            # Validate resting potential
            if ref.resting_potential_range:
                v_rest = result['v_rest']
                if not (ref.resting_potential_range[0] <= v_rest <= ref.resting_potential_range[1]):
                    validation['warnings'].append(
                        f"Resting potential {v_rest:.1f} mV outside range {ref.resting_potential_range}"
                    )
            
            # Validate spike frequency at given stimulus
            stim_ratio = result['iext'] / ((ref.rheobase_range[0] + ref.rheobase_range[1]) / 2)
            if result['n_spikes'] == 0 and stim_ratio > 1.5:
                validation['errors'].append(
                    f"No spikes detected at Iext={result['iext']:.1f} (expected {ref.spike_freq_range[1]:.0f} Hz)"
                )
                validation['status'] = 'FAIL'
            
            # Validate AP amplitude
            if result['ap_amplitude'] > 0 and not (ref.ap_amplitude_range[0] <= result['ap_amplitude'] <= ref.ap_amplitude_range[1]):
                validation['warnings'].append(
                    f"AP amplitude {result['ap_amplitude']:.1f} mV outside range {ref.ap_amplitude_range}"
                )
        
        self.results[preset_name] = validation
        return validation
    
    def validate_all(self) -> Dict[str, Dict]:
        """Validate all presets."""
        preset_names = get_preset_names()
        
        print("=" * 80)
        print("BIOPHYSICAL LITERATURE VALIDATION")
        print("=" * 80)
        
        for preset_name in preset_names:
            code = self._preset_code(preset_name)
            print(f"\n[{code}] {preset_name}")
            
            if code not in LITERATURE_DB:
                print(f"  ℹ No literature reference (skipping detailed validation)")
            
            validation = self.validate_preset(preset_name)
            
            if validation['status'] == 'PASS':
                print(f"  ✓ PASS")
            elif validation['status'] == 'FAIL':
                print(f"  ✗ FAIL")
            else:
                print(f"  ? ERROR")
            
            # Print metrics
            if 'metrics' in validation:
                m = validation['metrics']
                print(f"    Vrest = {m['v_rest']:.1f} mV, Spikes = {m['n_spikes']}, Freq = {m['spike_freq']:.1f} Hz")
                print(f"    gNa = {m['gNa_max']:.1f}, gK = {m['gK_max']:.1f}, gL = {m['gL']:.2f} mS/cm²")
                print(f"    Iext = {m['iext']:.1f} µA/cm², {m['stim_type']}")
                if m['ap_amplitude'] > 0:
                    print(f"    AP amplitude = {m['ap_amplitude']:.1f} mV")
            
            # Print warnings/errors
            for w in validation.get('warnings', []):
                print(f"  ⚠ {w}")
            for e in validation.get('errors', []):
                print(f"  ✗ {e}")
        
        return self.results


def analyze_conductance_ratios():
    """Analyze gNa/gK ratios across presets - biophysical sanity check."""
    print("\n" + "=" * 80)
    print("CONDUCTANCE RATIO ANALYSIS")
    print("=" * 80)
    print("Physiological ranges: gNa/gK = 2-10 (typical), gNa/gL = 100-1000")
    print("-" * 80)
    
    preset_names = get_preset_names()
    
    for preset_name in preset_names:
        code = preset_name.split(":")[0].strip()
        try:
            cfg = FullModelConfig()
            apply_preset(cfg, preset_name)
            
            gNa = cfg.channels.gNa_max
            gK = cfg.channels.gK_max
            gL = cfg.channels.gL
            
            if gK > 0 and gL > 0:
                na_k_ratio = gNa / gK
                na_l_ratio = gNa / gL
                
                status = "✓" if 2 <= na_k_ratio <= 15 else "⚠"
                print(f"[{code}] {na_k_ratio:.1f} gNa/gK, {na_l_ratio:.0f} gNa/gL {status}")
                
                # Flag extreme ratios
                if na_k_ratio > 20:
                    print(f"    WARNING: Very high gNa/gK ratio ({na_k_ratio:.1f})")
                if na_l_ratio < 50:
                    print(f"    WARNING: Very low gNa/gL ratio ({na_l_ratio:.0f})")
        except Exception as e:
            print(f"[{code}] Error: {e}")


def main():
    """Main validation routine."""
    validator = PresetValidator()
    results = validator.validate_all()
    
    analyze_conductance_ratios()
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for r in results.values() if r.get('status') == 'PASS')
    failed = sum(1 for r in results.values() if r.get('status') == 'FAIL')
    errors = sum(1 for r in results.values() if r.get('status') == 'ERROR')
    
    print(f"Total presets: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")
    
    # Critical findings
    critical_issues = []
    for preset_name, validation in results.items():
        if validation.get('errors'):
            for err in validation['errors']:
                critical_issues.append(f"{preset_name}: {err}")
    
    if critical_issues:
        print("\n" + "=" * 80)
        print("CRITICAL ISSUES REQUIRING ATTENTION")
        print("=" * 80)
        for issue in critical_issues:
            print(f"  ✗ {issue}")
    
    return passed, failed, errors


if __name__ == "__main__":
    passed, failed, errors = main()
    sys.exit(0 if (failed == 0 and errors == 0) else 1)
