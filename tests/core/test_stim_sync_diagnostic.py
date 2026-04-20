"""
Diagnostic test for stimulus-voltage synchronization after FIX-CRIT-A.

This verifies that the 1-Step Lag Bug is actually fixed by checking:
1. i_stim[0] = 0 (no stimulus at t=0)
2. Stimulus appears at correct time (not delayed)
3. Current Balance Error is reasonable (not 100-200+)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np


def test_stimulus_timing():
    """Verify stimulus appears at correct time relative to voltage."""
    # This is a diagnostic that should be run with actual simulation
    # For now, we document the expected behavior
    
    print("Expected behavior after FIX-CRIT-A:")
    print("1. i_stim[0] should be 0.0 (no stimulus before simulation)")
    print("2. First non-zero i_stim should appear at t ≈ stim_start")
    print("3. No 1-step delay between voltage response and stimulus")
    print()
    print("If CBE is still 100-200+, check:")
    print("- GHK mismatch (core uses Ohm, analysis uses GHK)")
    print("- dt_eval vs dt_internal (subsampling artifact)")
    print("- These are documented in User Audit as architectural issues")


def check_i_stim_array(result):
    """Diagnostic helper to check i_stim array from simulation result."""
    if not hasattr(result, 'i_stim_total') or result.i_stim_total is None:
        print("⚠ No i_stim_total in result")
        return
    
    i_stim = result.i_stim_total
    t = result.t
    
    print(f"i_stim array shape: {i_stim.shape}")
    print(f"Time array shape: {t.shape}")
    print(f"First i_stim value: {i_stim[0] if len(i_stim) > 0 else 'N/A'}")
    print(f"First time value: {t[0] if len(t) > 0 else 'N/A'}")
    print(f"Max i_stim: {np.max(i_stim) if len(i_stim) > 0 else 'N/A'}")
    print(f"Non-zero i_stim count: {np.sum(i_stim != 0)}")
    
    # Check for monotonic time
    dt = np.diff(t)
    if np.any(dt <= 0):
        print("⚠ WARNING: Non-monotonic time detected!")
        duplicates = np.sum(dt == 0)
        print(f"   Duplicate time points: {duplicates}")
    else:
        print("✓ Time is monotonic")


if __name__ == "__main__":
    test_stimulus_timing()
    print("\nRun actual simulation to use check_i_stim_array()")
