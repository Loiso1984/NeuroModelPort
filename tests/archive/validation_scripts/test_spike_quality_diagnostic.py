"""
DIAGNOSTIC: Spike Quality Analysis
Check for false detections, temporal patterns, realistic ISI
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver

def analyze_spike_quality(v_soma, t, rest_v, threshold_offset=30):
    """
    Detailed spike quality analysis
    Returns: (spike_times, isis, quality_metrics)
    """
    threshold = rest_v + threshold_offset
    
    # Find threshold crossings (rising phase only)
    above_threshold = v_soma > threshold
    
    # Find indices where voltage crosses threshold from below
    crossing_indices = np.where(np.diff(above_threshold.astype(int)) == 1)[0]
    spike_times = t[crossing_indices]
    
    if len(spike_times) < 2:
        return spike_times, np.array([]), {'quality': 'INSUFFICIENT_SPIKES', 'spike_count': len(spike_times)}
    
    # Calculate inter-spike intervals
    isis = np.diff(spike_times)
    
    # Quality metrics
    metrics = {
        'spike_count': len(spike_times),
        'isi_mean_ms': isis.mean(),
        'isi_std_ms': isis.std(),
        'isi_min_ms': isis.min(),
        'isi_max_ms': isis.max(),
        'cv_isi': isis.std() / isis.mean() if isis.mean() > 0 else np.inf,
        'firing_rate_hz': 1000.0 / isis.mean() if isis.mean() > 0 else 0,
        'quality': 'UNKNOWN'
    }
    
    # Quality assessment
    if metrics['isi_min_ms'] < 0.3:
        metrics['quality'] = 'SUSPICIOUS: ISI too short (< 0.3 ms)'
    elif metrics['isi_mean_ms'] < 0.5:
        metrics['quality'] = 'SUSPICIOUS: mean ISI < 0.5 ms (refractory violation?)'
    elif metrics['firing_rate_hz'] > 1000:
        metrics['quality'] = 'SUSPICIOUS: firing rate > 1000 Hz'
    elif metrics['isi_std_ms'] / metrics['isi_mean_ms'] > 2.0:
        metrics['quality'] = 'IRREGULAR: High ISI variability (CV > 2)'
    else:
        metrics['quality'] = 'REALISTIC'
    
    return spike_times, isis, metrics


print("=" * 90)
print("SPIKE QUALITY DIAGNOSTIC")
print("=" * 90)

test_presets = [
    ('A: Squid Giant Axon (HH 1952)', 10.0, 'Classic HH model - expect ~50-150 Hz'),
    ('C: FS Interneuron (Wang-Buzsaki)', 40.0, 'Fast spiking - expect ~150-400 Hz'),
    ('B: Pyramidal L5 (Mainen 1996)', 6.0, 'Regular firing - expect ~20-50 Hz'),
]

print()

for preset_name, iext, expected_behavior in test_presets:
    print(f"\n{preset_name}")
    print(f"  Stimulus: {iext} µA")
    print(f"  Expected: {expected_behavior}")
    print()
    
    cfg = FullModelConfig()
    apply_preset(cfg, preset_name)
    cfg.stim.Iext = iext
    cfg.stim.stim_type = 'const'
    
    try:
        solver = NeuronSolver(cfg)
        result = solver.run_single()
        
        v_soma = result.v_soma
        t = result.t
        rest_v = v_soma[0]
        
        spike_times, isis, metrics = analyze_spike_quality(v_soma, t, rest_v)
        
        print(f"  Simulation: {t[0]:.1f}-{t[-1]:.1f} ms ({len(t)} samples, dt={t[1]-t[0]:.3f} ms)")
        print()
        print(f"  Spike Detection:")
        print(f"    Total spikes: {metrics['spike_count']}")
        print(f"    ISI mean: {metrics['isi_mean_ms']:.2f} ms")
        print(f"    ISI std: {metrics['isi_std_ms']:.2f} ms")
        print(f"    ISI range: {metrics['isi_min_ms']:.2f}-{metrics['isi_max_ms']:.2f} ms")
        print(f"    CV (ISI): {metrics['cv_isi']:.3f}")
        print(f"    Firing rate: {metrics['firing_rate_hz']:.1f} Hz")
        print()
        print(f"  Quality Assessment: {metrics['quality']}")
        print()
        
        # Check refractory period (absolute refractory should be ~1-2 ms for HH models)
        if metrics['spike_count'] > 0:
            if metrics['isi_min_ms'] < 1.0:
                print(f"  ⚠️  WARNING: Minimum ISI {metrics['isi_min_ms']:.2f} ms is BELOW typical refractory (1-2 ms)")
                print(f"      This suggests either:")
                print(f"      1) False spike detections (threshold crossing noise)")
                print(f"      2) Abnormal model dynamics")
            elif metrics['isi_min_ms'] > 300:
                print(f"  ⚠️  EXTREME: Minimum ISI {metrics['isi_min_ms']:.2f} ms is extremely long")
            else:
                print(f"  ✓ ISI range physiologically plausible")
        
    except Exception as e:
        print(f"  ❌ ERROR: {str(e)[:100]}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 90)
print("INTERPRETATION GUIDE")
print("=" * 90)
print("""
NORMAL RANGES by neuron type:
  Squid Giant Axon (HH 1952):
    - Firing rate: 50-200 Hz
    - ISI: 5-20 ms
    - Absolute refractory: ~1 ms
    
  Fast Spiking Interneuron:
    - Firing rate: 150-400 Hz
    - ISI: 2.5-7 ms
    
  Regular Pyramidal:
    - Firing rate: 20-50 Hz
    - ISI: 20-50 ms

RED FLAGS:
  ❌ ISI < 0.3 ms: Definitely false detections or numerical instability
  ❌ ISI < 1.0 ms: Suspicious, may violate refractory biology
  ❌ Firing rate > 1000 Hz: Not possible in biological neurons
  ❌ CV > 2.0: Extreme variability (unless bursting)
  
GREEN FLAGS:
  ✓ ISI matches expected range
  ✓ CV < 0.5: Regular firing
  ✓ Firing rate in physiological range
""")
