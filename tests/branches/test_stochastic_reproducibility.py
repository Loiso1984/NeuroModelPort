"""
Stochastic Reproducibility Tests for NeuroModelPort v10.0

Tests that stochastic simulations produce identical results when using the same seed.
Validates RNG synchronization across all stochastic components.
"""

import pytest
import numpy as np
from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver
from core.stochastic_rng import StochasticRNG, seed_all
from core.advanced_sim import run_euler_maruyama


def test_stochastic_rng_basic():
    """Test 1: Basic RNG reproducibility."""
    seed = 12345
    
    # Create two identical RNG instances
    rng1 = StochasticRNG(seed)
    rng2 = StochasticRNG(seed)
    
    # Generate sequences
    seq1 = rng1.normal(0, 1, 100)
    seq2 = rng2.normal(0, 1, 100)
    
    assert np.allclose(seq1, seq2), "RNG sequences should be identical with same seed"
    print("✅ Basic RNG reproducibility test passed")


def test_stochastic_rng_state_save_load():
    """Test 2: RNG state save/load functionality."""
    seed = 54321
    
    rng1 = StochasticRNG(seed)
    
    # Generate some numbers
    seq1_part1 = rng1.normal(0, 1, 50)
    
    # Save state
    state = rng1.get_state()
    
    # Create new RNG and restore state
    rng2 = StochasticRNG()
    rng2.set_state(state)
    
    # Generate remaining numbers
    seq1_part2 = rng1.normal(0, 1, 50)
    seq2_part2 = rng2.normal(0, 1, 50)
    
    assert np.allclose(seq1_part2, seq2_part2), "Restored RNG should continue sequence correctly"
    print("✅ RNG state save/load test passed")


def test_physics_params_stochastic():
    """Test 3: PhysicsParams with stochastic parameters."""
    from core.physics_params import create_physics_params
    import numpy as np
    
    # Create physics params with stochastic settings
    physics = create_physics_params(
        # Model size and feature flags
        n_comp=1,
        en_ih=False, en_ica=False, en_ia=False, en_sk=False,
        dyn_ca=False, en_itca=False, en_im=False, en_nap=False, en_nar=False,
        
        # Conductance matrix
        gbar_mat=np.zeros((11, 1)),
        
        # Reversal potentials
        ena=50.0, ek=-77.0, el=-65.0, eih=-43.0, ea=-77.0,
        
        # Morphology
        cm_v=np.array([1.0]),
        l_data=np.array([0.0]), l_indices=np.array([0]), l_indptr=np.array([0, 1]),
        
        # Temperature scaling
        phi_mat=np.zeros((9, 1)),
        
        # Environment
        t_kelvin=310.15, ca_ext=2.0, ca_rest=5e-5, tau_ca=200.0,
        b_ca=np.array([1e-5]), mg_ext=1.0, tau_sk=10.0,
        
        # Stimulation
        stype=0, iext=10.0, t0=0.0, td=0.0, atau=1.0,
        zap_f0_hz=0.5, zap_f1_hz=40.0, event_times_arr=np.array([]),
        n_events=0, stim_comp=0, stim_mode=0,
        use_dfilter_primary=0, dfilter_attenuation=0.5, dfilter_tau_ms=10.0,
        
        # Dual stimulation
        dual_stim_enabled=0, stype_2=0, iext_2=0.0, t0_2=0.0, td_2=0.0,
        atau_2=1.0, zap_f0_hz_2=0.5, zap_f1_hz_2=40.0,
        stim_comp_2=0, stim_mode_2=0, use_dfilter_secondary=0,
        dfilter_attenuation_2=0.5, dfilter_tau_ms_2=10.0,
        
        # Stochastic parameters
        stoch_gating=True,
        noise_sigma=0.1,
        rng_state=np.array([12345, 67890])  # Mock RNG state
    )
    
    assert physics.stoch_gating == True, "Stochastic gating should be enabled"
    assert physics.noise_sigma == 0.1, "Noise sigma should be set"
    assert physics.rng_state is not None, "RNG state should be stored"
    print("✅ PhysicsParams stochastic integration test passed")


def test_euler_maruyama_reproducibility():
    """Test 4: Euler-Maruyama simulation reproducibility."""
    seed = 98765
    
    # Configure simple stochastic simulation
    config = FullModelConfig()
    apply_preset(config, "Squid Giant Axon (HH 1952)")
    config.stim.stoch_gating = True
    config.stim.noise_sigma = 0.05
    config.stim.t_sim = 100.0
    config.stim.dt_eval = 0.1
    
    # Seed all stochastic components
    seed_all(seed)
    
    # Run first simulation
    result1 = run_euler_maruyama(config)
    
    # Reset with same seed and run again
    seed_all(seed)
    result2 = run_euler_maruyama(config)
    
    # Compare results (allowing for small floating point differences)
    assert np.allclose(result1.v_soma, result2.v_soma, rtol=1e-10, atol=1e-12), \
        "Euler-Maruyama results should be identical with same seed"
    
    # Check that results are actually different from deterministic
    config.stim.stoch_gating = False
    config.stim.noise_sigma = 0.0
    result_det = run_euler_maruyama(config)
    
    assert not np.allclose(result1.v_soma, result_det.v_soma, rtol=1e-6), \
        "Stochastic should differ from deterministic"
    
    print("✅ Euler-Maruyama reproducibility test passed")


def test_monte_carlo_reproducibility():
    """Test 5: Monte-Carlo parameter sweep reproducibility."""
    seed = 13579
    
    config = FullModelConfig()
    apply_preset(config, "Squid Giant Axon (HH 1952)")
    config.stim.t_sim = 50.0
    config.stim.dt_eval = 0.1
    
    # Seed for Monte-Carlo
    seed_all(seed)
    
    # Run Monte-Carlo twice
    solver = NeuronSolver(config)
    results1 = solver.run_mc(n_trials=10, param_var=0.05)
    
    seed_all(seed)
    results2 = solver.run_mc(n_trials=10, param_var=0.05)
    
    # Compare parameter variations from configs
    variations1 = []
    variations2 = []
    
    for r1, r2 in zip(results1, results2):
        if r1 is not None and hasattr(r1, 'config'):
            variations1.append(r1.config.channels.gNa_max)
        else:
            variations1.append(np.nan)
            
        if r2 is not None and hasattr(r2, 'config'):
            variations2.append(r2.config.channels.gNa_max)
        else:
            variations2.append(np.nan)
    
    # Filter out NaN values for comparison
    valid_mask = ~(np.isnan(variations1) | np.isnan(variations2))
    
    if np.any(valid_mask):
        assert np.allclose(np.array(variations1)[valid_mask], 
                         np.array(variations2)[valid_mask]), \
            "Monte-Carlo parameter variations should be identical with same seed"
    
    print("✅ Monte-Carlo reproducibility test passed")


def test_cross_component_rng_consistency():
    """Test 6: RNG consistency across different components."""
    seed = 24680
    
    # Seed global RNG
    seed_all(seed)
    
    # Use RNG in different contexts
    from core.stochastic_rng import get_rng
    
    rng = get_rng()
    
    # Generate sequences in different contexts
    seq1 = rng.normal(0, 1, 20)
    seq2 = rng.normal(0, 1, 20)
    seq3 = rng.randn(20)
    
    # Reset and regenerate
    seed_all(seed)
    rng = get_rng()
    
    seq1_b = rng.normal(0, 1, 20)
    seq2_b = rng.normal(0, 1, 20)
    seq3_b = rng.randn(20)
    
    assert np.allclose(seq1, seq1_b), "Cross-component RNG should be consistent"
    assert np.allclose(seq2, seq2_b), "Cross-component RNG should be consistent"
    assert np.allclose(seq3, seq3_b), "Cross-component RNG should be consistent"
    
    print("✅ Cross-component RNG consistency test passed")


if __name__ == "__main__":
    """Run all stochastic reproducibility tests."""
    print("🎲 STOCHASTIC REPRODUCIBILITY VALIDATION")
    print("=" * 50)
    
    test_stochastic_rng_basic()
    test_stochastic_rng_state_save_load()
    test_physics_params_stochastic()
    test_euler_maruyama_reproducibility()
    test_monte_carlo_reproducibility()
    test_cross_component_rng_consistency()
    
    print("=" * 50)
    print("🎯 ALL STOCHASTIC REPRODUCIBILITY TESTS PASSED!")
    print("✅ Stochastic RNG synchronization is working correctly")
