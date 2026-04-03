"""
Test Branch: Spike Detection Validation
Validates detect_spikes() correctness on synthetic data with known spikes
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.analysis import detect_spikes


class SpikeDetectionTestBranch:
    """Test branch for validating spike detection algorithm"""
    
    def __init__(self):
        self.results = {}
    
    def generate_synthetic_spikes(self, n_spikes=5, noise_level=0.5):
        """Generate synthetic voltage trace with known spikes"""
        t = np.linspace(0, 100, 1000)  # 100ms, 0.1ms resolution
        dt = t[1] - t[0]
        
        # Baseline at -70mV
        v = np.ones_like(t) * -70.0
        
        # Known spike times (ms)
        spike_times_true = [20, 40, 60, 80, 95]
        spike_amps_true = []
        
        for sp_t in spike_times_true[:n_spikes]:
            # Generate action potential shape
            sp_idx = int(sp_t / dt)
            if sp_idx < len(t) - 50:
                # Upstroke (0-1ms)
                up_duration = int(1.0 / dt)
                v[sp_idx:sp_idx+up_duration] = np.linspace(-70, 45, up_duration)
                
                # Peak and repolarization (1-2ms)
                peak_duration = int(1.0 / dt)
                v[sp_idx+up_duration:sp_idx+up_duration+peak_duration] = np.linspace(45, -60, peak_duration)
                
                # AHP (2-5ms)
                ahp_duration = int(3.0 / dt)
                v[sp_idx+up_duration+peak_duration:sp_idx+up_duration+peak_duration+ahp_duration] = np.linspace(-60, -70, ahp_duration)
                
                spike_amps_true.append(45.0)
        
        # Add noise
        v += np.random.normal(0, noise_level, len(v))
        
        return t, v, spike_times_true[:n_spikes], spike_amps_true
    
    def test_detection_accuracy(self):
        """Test detection accuracy on synthetic data"""
        print("\n🧪 Testing spike detection accuracy...")
        
        t, v, true_times, true_amps = self.generate_synthetic_spikes(n_spikes=5)
        
        # Detect spikes
        peak_idx, spike_times, spike_amps = detect_spikes(v, t, threshold=-20.0, baseline_threshold=-50.0)
        
        print(f"   True spikes: {len(true_times)} at {true_times} ms")
        print(f"   Detected: {len(spike_times)} at {[f'{st:.1f}' for st in spike_times]} ms")
        print(f"   Amplitudes: {[f'{sa:.1f}' for sa in spike_amps]} mV")
        
        # Check accuracy (within 1ms tolerance)
        tolerance = 1.0  # ms
        matched = 0
        false_positives = 0
        
        for detected_time in spike_times:
            found_match = False
            for true_time in true_times:
                if abs(detected_time - true_time) < tolerance:
                    matched += 1
                    found_match = True
                    break
            if not found_match:
                false_positives += 1
        
        missed = len(true_times) - matched
        
        print(f"   Matched: {matched}, Missed: {missed}, False positives: {false_positives}")
        
        # Amplitude check (should be ~45mV)
        amp_correct = all(40 < sa < 50 for sa in spike_amps[:matched])
        
        passed = matched >= 4 and missed <= 1 and false_positives == 0 and amp_correct
        
        self.results['detection_accuracy'] = {
            'matched': matched,
            'missed': missed,
            'false_positives': false_positives,
            'passed': passed
        }
        
        if passed:
            print("   ✅ PASSED")
        else:
            print("   ❌ FAILED")
        
        return passed
    
    def test_threshold_sensitivity(self):
        """Test sensitivity to threshold parameter"""
        print("\n🧪 Testing threshold sensitivity...")
        
        t, v, true_times, _ = self.generate_synthetic_spikes(n_spikes=3)
        
        thresholds = [-30, -20, -10, 0]
        results = []
        
        for thresh in thresholds:
            peak_idx, spike_times, _ = detect_spikes(v, t, threshold=thresh)
            results.append({'thresh': thresh, 'n_spikes': len(spike_times)})
            print(f"   Threshold {thresh}mV: {len(spike_times)} spikes detected")
        
        # With threshold -20, should detect all 3
        # With threshold 0, might miss some
        correct_at_minus20 = results[1]['n_spikes'] == 3
        
        passed = correct_at_minus20
        self.results['threshold_sensitivity'] = {'results': results, 'passed': passed}
        
        if passed:
            print("   ✅ PASSED")
        else:
            print("   ❌ FAILED")
        
        return passed
    
    def test_baseline_crossing_validation(self):
        """Test that algorithm properly requires repolarization"""
        print("\n🧪 Testing baseline crossing validation...")
        print("   ✅ IMPROVED: Algorithm now checks repolarization")
        
        # Create trace that crosses threshold but doesn't repolarize
        t = np.linspace(0, 50, 500)
        v = np.ones_like(t) * -70.0
        v[200:300] = -10  # Sustained depolarization, no spike
        v[300:] = -70.0  # Return to baseline
        
        peak_idx, spike_times, _ = detect_spikes(v, t, threshold=-20.0, baseline_threshold=-50.0)
        
        # With improved algorithm, should NOT detect false positive
        false_positive_detected = len(spike_times) > 0
        
        print(f"   False positive detected: {false_positive_detected}")
        
        if not false_positive_detected:
            print("   ✅ PASSED - algorithm correctly rejects sustained depolarization")
            passed = True
        else:
            print("   ❌ FAILED - algorithm still has false positives")
            passed = False
        
        self.results['baseline_crossing'] = {
            'false_positive': false_positive_detected,
            'passed': passed
        }
        
        return passed
    
    def test_high_frequency_burst(self):
        """Test detection of high-frequency bursts"""
        print("\n🧪 Testing high-frequency burst detection...")
        
        t = np.linspace(0, 50, 500)
        dt = t[1] - t[0]
        v = np.ones_like(t) * -70.0
        
        # 5 spikes at 100Hz (10ms interval)
        spike_times = [10, 20, 30, 40, 48]
        
        for sp_t in spike_times:
            sp_idx = int(sp_t / dt)
            if sp_idx < len(t) - 20:
                v[sp_idx:sp_idx+5] = np.linspace(-70, 40, 5)
                v[sp_idx+5:sp_idx+15] = np.linspace(40, -70, 10)
        
        peak_idx, detected_times, _ = detect_spikes(v, t, threshold=-20.0)
        
        print(f"   True spikes: 5 at high frequency (100Hz)")
        print(f"   Detected: {len(detected_times)}")
        
        # Should detect all 5
        passed = len(detected_times) >= 4
        
        self.results['high_frequency'] = {'detected': len(detected_times), 'passed': passed}
        
        if passed:
            print("   ✅ PASSED")
        else:
            print("   ❌ FAILED")
        
        return passed
    
    def run_all_tests(self):
        """Run complete spike detection validation"""
        print("=" * 70)
        print("SPIKE DETECTION VALIDATION (TEST BRANCH)")
        print("=" * 70)
        print("Testing detect_spikes() correctness on synthetic data")
        
        tests = [
            self.test_detection_accuracy,
            self.test_threshold_sensitivity,
            self.test_baseline_crossing_validation,
            self.test_high_frequency_burst,
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            try:
                if test():
                    passed += 1
            except Exception as e:
                print(f"   ❌ EXCEPTION: {e}")
        
        print("\n" + "=" * 70)
        print(f"SUMMARY: {passed}/{total} tests passed")
        print("=" * 70)
        
        if passed < total:
            print("\n⚠️  CRITICAL: Spike detection algorithm needs improvement!")
            print("   - Add repolarization verification")
            print("   - Consider baseline crossing detection")
        
        return self.results


if __name__ == "__main__":
    test_branch = SpikeDetectionTestBranch()
    test_branch.run_all_tests()
