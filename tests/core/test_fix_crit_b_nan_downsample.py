"""
Regression test for FIX-CRIT-B: NaN poisoning in _downsample_xy

Verifies that LLE data with NaN values (during transient period)
is handled correctly by the downsampling function.

Before fix:
- np.argmin/argmax on array with NaN returns index of first NaN
- This causes LLE curve to be drawn with artifacts or disappear

After fix:
- NaN values are masked and excluded from min/max calculations
- LLE curve renders correctly
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np


def _downsample_xy_original(t: np.ndarray, y: np.ndarray, max_points: int = 2000):
    """Original version without NaN handling (for comparison)."""
    n = int(len(t))
    if n <= max_points or max_points <= 0:
        return t, y

    n_chunks = max_points // 2
    chunk_size = max(1, n // n_chunks)

    indices = []
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        chunk = y[i:end]
        min_idx = i + int(np.argmin(chunk))
        max_idx = i + int(np.argmax(chunk))
        indices.append(min_idx)
        indices.append(max_idx)

    indices = np.unique(np.array(indices, dtype=np.int64))
    if indices[-1] != n - 1:
        indices = np.concatenate([indices, np.array([n - 1])])

    return t[indices], y[indices]


def test_downsample_xy_with_nan():
    """Test that _downsample_xy handles NaN values correctly."""
    from gui.analytics import _downsample_xy
    
    # Create test data with NaN in the beginning (like LLE transient)
    t = np.linspace(0, 100, 1000)
    y = np.concatenate([
        np.full(100, np.nan),  # NaN in transient period
        np.sin(t[100:]) + 0.1 * np.random.randn(900)  # Real data after
    ])
    
    # Test the fixed version
    td, yd = _downsample_xy(t, y, max_points=100)
    
    # Verify:
    # 1. No NaN in output (should be filtered)
    assert not np.any(np.isnan(yd)), "Output should not contain NaN"
    
    # 2. Output should have reasonable number of points
    assert len(td) <= 100, f"Output should be downsampled, got {len(td)} points"
    assert len(td) > 0, "Output should not be empty"
    
    # 3. Time should be monotonic
    assert np.all(np.diff(td) >= 0), "Time should be monotonic"
    
    print("✓ _downsample_xy handles NaN correctly")


def test_downsample_xy_all_nan():
    """Test behavior when all values are NaN."""
    from gui.analytics import _downsample_xy
    
    t = np.linspace(0, 100, 100)
    y = np.full(100, np.nan)
    
    # Should not crash - function should handle gracefully
    td, yd = _downsample_xy(t, y, max_points=50)
    
    # When all NaN and downsampling required, 
    # function returns early with original data (no valid points to decimate)
    # OR returns filtered result - both are acceptable
    assert len(td) > 0, "Should return non-empty result"
    assert len(td) == len(yd), "t and y should have same length"
    
    print("✓ _downsample_xy handles all-NaN case correctly (no crash)")


def test_downsample_xy_no_nan():
    """Test that normal data still works correctly."""
    from gui.analytics import _downsample_xy
    
    t = np.linspace(0, 100, 1000)
    y = np.sin(t)
    
    td, yd = _downsample_xy(t, y, max_points=100)
    
    # Should preserve peaks (min and max of sine should be present)
    assert len(td) <= 100, "Should be downsampled"
    assert np.min(yd) <= -0.9, "Should preserve minimum peaks"
    assert np.max(yd) >= 0.9, "Should preserve maximum peaks"
    
    print("✓ _downsample_xy works correctly with normal data")


if __name__ == "__main__":
    test_downsample_xy_with_nan()
    test_downsample_xy_all_nan()
    test_downsample_xy_no_nan()
    print("\nAll FIX-CRIT-B tests passed!")
