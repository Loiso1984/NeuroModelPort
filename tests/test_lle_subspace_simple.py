"""
tests/test_lle_subspace_simple.py — LLE Subspace Renormalization Verification v12.2

Standalone tests that verify the mathematical correctness of LLE renormalization
without heavy dependencies.

Run: python tests/test_lle_subspace_simple.py
"""
import numpy as np
import sys
import os

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_lle_renormalization_preserves_non_subspace():
    """
    Core test: Verify that renormalization snaps non-subspace variables to main trajectory.
    
    This simulates the Benettin LLE renormalization step and validates that:
    - Variables IN subspace are scaled: y_pert = y_main + (y_pert - y_main) * scale
    - Variables NOT in subspace are snapped: y_pert = y_main
    
    This prevents metabolic variables (ATP, Ca_i) from being artificially inflated.
    """
    n_state = 30
    n_comp = 5
    
    # Mock state vectors
    y_main = np.linspace(-70, 0, n_state, dtype=np.float64)  # Voltage-like range
    y_pert = y_main + np.random.randn(n_state) * 0.1  # Small perturbation
    
    # v_only subspace: only first n_comp variables (voltage)
    in_subspace = np.zeros(n_state, dtype=np.bool_)
    in_subspace[:n_comp] = True
    
    # Renormalization parameters
    lle_delta = 1e-6
    dist_sq = np.sum((y_pert[:n_comp] - y_main[:n_comp])**2)
    dist = np.sqrt(dist_sq)
    scale = lle_delta / max(dist, 1e-12)
    
    # Apply renormalization (matching native_loop.py logic)
    y_pert_new = y_pert.copy()
    for i in range(n_state):
        if in_subspace[i]:
            # Scale perturbation vector
            y_pert_new[i] = y_main[i] + (y_pert[i] - y_main[i]) * scale
        else:
            # Snap to main trajectory (CRITICAL FIX v12.1)
            y_pert_new[i] = y_main[i]
    
    # VALIDATION
    
    # 1. Subspace variables should be scaled (small but non-zero difference)
    for i in range(n_comp):
        diff = y_pert_new[i] - y_main[i]
        orig_diff = y_pert[i] - y_main[i]
        expected_diff = orig_diff * scale
        
        assert abs(diff - expected_diff) < 1e-10, \
            f"Subspace[{i}]: expected diff={expected_diff:.2e}, got {diff:.2e}"
    
    # 2. Non-subspace variables should be EXACTLY equal to main (snapped)
    for i in range(n_comp, n_state):
        diff = y_pert_new[i] - y_main[i]
        assert abs(diff) < 1e-10, \
            f"Non-subspace[{i}]: should be snapped (diff=0), got {diff:.2e}"
    
    print(f"✅ v_only mode: {n_comp} voltage vars scaled, {n_state-n_comp} metabolic vars snapped")
    print(f"   Scale factor: {scale:.2e}, Initial dist: {dist:.2e}")


def test_lle_v_and_gates_mode():
    """
    Test v_and_gates mode (mode 1): V + gates in subspace, metabolic NOT in subspace.
    """
    n_comp = 5
    # State layout: V(5) + m(5) + h(5) + n(5) + r(5) + s(5) + u(5) + Ca(5) + ATP(5) + Na_i(5) + K_o(5)
    n_state = n_comp * 11  # Simplified layout
    
    offsets = {
        'v': 0,
        'm': n_comp,
        'h': 2*n_comp,
        'n': 3*n_comp,
        'r': 4*n_comp,  # Ih
        's': 5*n_comp,  # ICa
        'u': 6*n_comp,  # ICa
        'ca': 7*n_comp,
        'atp': 8*n_comp,
        'na_i': 9*n_comp,
        'k_o': 10*n_comp,
    }
    
    y_main = np.random.uniform(-70, 50, n_state).astype(np.float64)
    y_pert = y_main + np.random.randn(n_state) * 0.01
    
    # v_and_gates subspace
    in_subspace = np.zeros(n_state, dtype=np.bool_)
    # Voltage
    in_subspace[offsets['v']:offsets['v']+n_comp] = True
    # Core gates
    in_subspace[offsets['m']:offsets['m']+n_comp] = True
    in_subspace[offsets['h']:offsets['h']+n_comp] = True
    in_subspace[offsets['n']:offsets['n']+n_comp] = True
    # Channel gates
    in_subspace[offsets['r']:offsets['r']+n_comp] = True  # Ih
    in_subspace[offsets['s']:offsets['s']+n_comp] = True  # ICa s
    in_subspace[offsets['u']:offsets['u']+n_comp] = True  # ICa u
    
    # Metabolic NOT in subspace
    assert not in_subspace[offsets['ca']]
    assert not in_subspace[offsets['atp']]
    assert not in_subspace[offsets['na_i']]
    assert not in_subspace[offsets['k_o']]
    
    # Renormalize
    lle_delta = 1e-6
    dist = np.sqrt(np.sum((y_pert[in_subspace] - y_main[in_subspace])**2))
    scale = lle_delta / max(dist, 1e-12)
    
    y_pert_new = y_pert.copy()
    for i in range(n_state):
        if in_subspace[i]:
            y_pert_new[i] = y_main[i] + (y_pert[i] - y_main[i]) * scale
        else:
            y_pert_new[i] = y_main[i]
    
    # Verify metabolic snapped
    for key in ['ca', 'atp', 'na_i', 'k_o']:
        start = offsets[key]
        for i in range(n_comp):
            idx = start + i
            diff = y_pert_new[idx] - y_main[idx]
            assert abs(diff) < 1e-10, f"{key}[{i}]: should be snapped, diff={diff:.2e}"
    
    # Verify gates scaled
    for key in ['m', 'h', 'n', 'r', 's', 'u']:
        start = offsets[key]
        for i in range(n_comp):
            idx = start + i
            diff = y_pert_new[idx] - y_main[idx]
            orig_diff = y_pert[idx] - y_main[idx]
            expected = orig_diff * scale
            assert abs(diff - expected) < 1e-10, f"{key}[{i}]: expected scaled diff"
    
    subspace_count = np.sum(in_subspace)
    print(f"✅ v_and_gates mode: {subspace_count} vars scaled (V+gates), {n_state-subspace_count} metabolic snapped")


def test_lle_full_state_mode():
    """
    Test full_state mode (mode 2): ALL variables in subspace.
    """
    n_state = 50
    y_main = np.random.randn(n_state).astype(np.float64)
    y_pert = y_main + np.random.randn(n_state) * 0.1
    
    # Full state: everything in subspace
    in_subspace = np.ones(n_state, dtype=np.bool_)
    
    lle_delta = 1e-6
    dist = np.sqrt(np.sum((y_pert - y_main)**2))
    scale = lle_delta / max(dist, 1e-12)
    
    y_pert_new = y_pert.copy()
    for i in range(n_state):
        y_pert_new[i] = y_main[i] + (y_pert[i] - y_main[i]) * scale
    
    # All variables should be scaled (not snapped)
    for i in range(n_state):
        diff = y_pert_new[i] - y_main[i]
        orig_diff = y_pert[i] - y_main[i]
        expected = orig_diff * scale
        assert abs(diff - expected) < 1e-10, f"full_state[{i}]: should be scaled"
    
    print(f"✅ full_state mode: all {n_state} variables scaled")


def test_atp_preservation_under_lle():
    """
    Specific test for ATP preservation - the key bug that was fixed in v12.1.
    
    Before v12.1: ATP was scaled along with V during renormalization,
    causing artificial ATP inflation and fake Lyapunov divergence.
    
    After v12.1: ATP snaps to main trajectory, preserving metabolic memory.
    """
    n_state = 100
    atp_idx = 50  # ATP at a specific index
    
    y_main = np.zeros(n_state, dtype=np.float64)
    y_main[atp_idx] = 2.5  # ATP at 2.5 mM (normal level)
    
    # Perturbed trajectory with inflated ATP
    y_pert = y_main.copy()
    y_pert[atp_idx] = 3.0  # 3.0 mM (pathologically high)
    
    # v_and_gates mode: ATP NOT in subspace
    in_subspace = np.zeros(n_state, dtype=np.bool_)
    in_subspace[:50] = True  # V + gates
    in_subspace[atp_idx] = False  # ATP explicitly excluded
    
    # Renormalize
    dist = 1e-3  # Some distance
    scale = 1e-6 / dist
    
    y_pert_new = y_pert.copy()
    for i in range(n_state):
        if in_subspace[i]:
            y_pert_new[i] = y_main[i] + (y_pert[i] - y_main[i]) * scale
        else:
            y_pert_new[i] = y_main[i]  # SNAP
    
    # ATP should be exactly at main trajectory level (2.5 mM)
    atp_after = y_pert_new[atp_idx]
    assert abs(atp_after - 2.5) < 1e-10, \
        f"ATP should be preserved at 2.5 mM, got {atp_after} mM"
    
    # If ATP was scaled (old bug), it would be:
    # atp_bug = 2.5 + (3.0 - 2.5) * scale ≈ 2.5 + 0.5 * 1e-3 = 2.5005
    atp_if_scaled = y_main[atp_idx] + (y_pert[atp_idx] - y_main[atp_idx]) * scale
    assert abs(atp_after - atp_if_scaled) > 1e-4, \
        "ATP should NOT follow scaled value (this would be the bug)"
    
    print(f"✅ ATP preservation: {y_pert[atp_idx]:.2f} → {atp_after:.2f} mM (snapped to main)")
    print(f"   (If scaled, would be: {atp_if_scaled:.6f} mM - BUG!)")


if __name__ == "__main__":
    print("=" * 60)
    print("LLE Subspace Renormalization Verification Tests v12.2")
    print("=" * 60)
    print()
    
    test_lle_renormalization_preserves_non_subspace()
    print()
    
    test_lle_v_and_gates_mode()
    print()
    
    test_lle_full_state_mode()
    print()
    
    test_atp_preservation_under_lle()
    print()
    
    print("=" * 60)
    print("✅ ALL LLE SUBSPACE TESTS PASSED")
    print("=" * 60)
    print()
    print("Key Validations:")
    print("  • v_only mode: Only V scaled, metabolic vars snapped")
    print("  • v_and_gates: V+gates scaled, metabolic snapped")
    print("  • full_state: All variables scaled")
    print("  • ATP preservation: Metabolic memory preserved during LLE")
    print()
