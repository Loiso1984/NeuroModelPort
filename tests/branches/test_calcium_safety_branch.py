"""
Branch test for Calcium safety bounds (Phase P1).
Tests calcium clamping, NaN/Inf prevention, and extreme condition handling.
Includes Hypoxia and Alzheimer presets with extended simulation (2000ms).
"""

import sys
import numpy as np
from pathlib import Path

# Add project path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver

# Допуск для ODE-решателя (SciPy solve_ivp имеет atol=1e-7, возможен overshoot)
SOLVER_TOLERANCE = 1e-6


def test_calcium_clamping_basic():
    """Test 1: Basic calcium clamping prevents NaN/Inf."""
    config = FullModelConfig()
    config.stim.t_sim = 100.0
    config.stim.dt_eval = 0.1

    # Enable calcium dynamics
    config.calcium.dynamic_Ca = True
    config.channels.enable_ICa = True

    solver = NeuronSolver(config)
    result = solver.run_single()

    # Check for NaN/Inf in voltage AND calcium
    assert np.all(np.isfinite(result.v_soma)), "Voltage produced NaN/Inf values"
    
    if result.ca_i is not None:
        assert np.all(np.isfinite(result.ca_i)), "Calcium dynamics produced NaN/Inf values"

        ca_min = np.min(result.ca_i)
        ca_max = np.max(result.ca_i)

        # Ослабленные проверки с учетом overshoot'а интегратора
        assert ca_min >= (1e-9 - SOLVER_TOLERANCE), f"Calcium below bound: {ca_min}"
        assert ca_max <= 10.0 + SOLVER_TOLERANCE, f"Calcium above bound: {ca_max}"
        return True
    else:
        raise AssertionError("Calcium dynamics not enabled in results")


def test_hypoxia_preset_calcium_safety():
    """Test 2: Hypoxia preset with extended simulation (2000ms)."""
    config = FullModelConfig()
    apply_preset(config, 'Hypoxia')
    config.stim.t_sim = 2000.0  # Extended simulation
    config.stim.dt_eval = 0.5   # Чуть увеличим шаг вывода для ускорения теста (не влияет на точность)
    config.stim.jacobian_mode = 'sparse_fd' # Убедимся что используется быстрый режим

    solver = NeuronSolver(config)
    result = solver.run_single()

    assert np.all(np.isfinite(result.v_soma)), "Hypoxia preset produced NaN in voltage"
    
    if result.ca_i is not None:
        assert np.all(np.isfinite(result.ca_i)), "Hypoxia preset produced NaN/Inf in calcium"

        ca_min = np.min(result.ca_i)
        ca_max = np.max(result.ca_i)

        assert ca_min >= (1e-9 - SOLVER_TOLERANCE), f"Hypoxia calcium below bound: {ca_min}"
        assert ca_max <= 10.0 + SOLVER_TOLERANCE, f"Hypoxia calcium above bound: {ca_max}"
        return True
    else:
        raise AssertionError("Hypoxia preset missing calcium dynamics")


def test_alzheimer_preset_calcium_safety():
    """Test 3: Alzheimer preset with extended simulation (2000ms)."""
    config = FullModelConfig()
    apply_preset(config, "Alzheimer's")
    config.stim.t_sim = 2000.0 
    config.stim.dt_eval = 0.5
    config.stim.jacobian_mode = 'sparse_fd'

    solver = NeuronSolver(config)
    result = solver.run_single()

    assert np.all(np.isfinite(result.v_soma)), "Alzheimer preset produced NaN in voltage"
    
    if result.ca_i is not None:
        assert np.all(np.isfinite(result.ca_i)), "Alzheimer preset produced NaN/Inf in calcium"

        ca_min = np.min(result.ca_i)
        ca_max = np.max(result.ca_i)

        assert ca_min >= (1e-9 - SOLVER_TOLERANCE), f"Alzheimer calcium below bound: {ca_min}"
        assert ca_max <= 10.0 + SOLVER_TOLERANCE, f"Alzheimer calcium above bound: {ca_max}"
        return True
    else:
        raise AssertionError("Alzheimer preset missing calcium dynamics")


def test_extreme_calcium_conditions():
    """Test 4: Extreme calcium conditions handling."""
    config = FullModelConfig()
    config.stim.t_sim = 500.0
    config.stim.dt_eval = 0.1

    # Enable calcium with extreme parameters
    config.calcium.dynamic_Ca = True
    config.channels.enable_ICa = True
    config.calcium.Ca_ext = 5.0    # High external calcium
    config.calcium.Ca_rest = 1e-3  # High resting calcium
    config.calcium.B_Ca = 0.05     # EXTREME calcium influx (was 0.01)
    config.channels.gCa_max = 5.0  # Massive calcium conductance

    solver = NeuronSolver(config)
    result = solver.run_single()

    assert np.all(np.isfinite(result.v_soma)), "Extreme conditions produced NaN in voltage"
    
    if result.ca_i is not None:
        assert np.all(np.isfinite(result.ca_i)), "Extreme conditions produced NaN/Inf in calcium"

        ca_min = np.min(result.ca_i)
        ca_max = np.max(result.ca_i)

        assert ca_min >= (1e-9 - SOLVER_TOLERANCE), f"Extreme calcium below bound: {ca_min}"
        assert ca_max <= 10.0 + SOLVER_TOLERANCE, f"Extreme calcium above bound: {ca_max}"
        return True
    else:
        raise AssertionError("Calcium dynamics missing in extreme test")


def _run_as_script() -> int:
    """Run tests manually with console output."""
    print("🔬 Phase P1: Calcium Safety (NaN/Inf Guard) Validation")
    print("=" * 60)
    
    tests = [
        test_calcium_clamping_basic,
        test_hypoxia_preset_calcium_safety,
        test_alzheimer_preset_calcium_safety,
        test_extreme_calcium_conditions,
    ]
    
    passed = 0
    for test_func in tests:
        try:
            print(f"Running {test_func.__name__}...", end=" ")
            if test_func():
                print("✅ PASS")
                passed += 1
        except Exception as e:
            print(f"❌ FAIL\n  Error: {e}")
            
    print("=" * 60)
    print(f"Results: {passed}/{len(tests)} tests passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(_run_as_script())
