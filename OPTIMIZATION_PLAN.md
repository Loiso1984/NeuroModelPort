# Computation Optimization Plan

## Current Status

### Already Implemented:
- ✅ Time estimation in solver.py (lines 60-91)
- ✅ Heavy simulation warnings (>30s)
- ✅ Actual completion time reporting
- ✅ Numba @njit decorators on kinetics functions
- ✅ ProcessPoolExecutor for Monte Carlo (existing)

### Performance Bottlenecks Identified:

1. **rhs_multicompartment** - Called at every time step
   - Multiple channel current calculations
   - Kinetic function evaluations
   - Matrix operations for compartment coupling

2. **solve_ivp integration** - BDF method
   - Adaptive step sizing
   - Frequent RHS evaluations
   - Jacobian computation (implicit)

3. **Channel kinetics** - Gate variable updates
   - Alpha/beta calculations for all channels
   - Temperature scaling (Q10)

## Optimization Strategies

### Phase 1: Python Optimization (Immediate)

#### 1.1 Optimize rhs_multicompartment
```python
# Current: Multiple function calls per step
# Optimize: Inline critical calculations, reduce function call overhead

# Strategies:
- Pre-compute channel parameters that don't change
- Vectorize gate variable updates
- Use numba @njit for the entire RHS function
- Cache kinetic evaluations
```

#### 1.2 Improve solve_ivp Settings
```python
# Current: BDF with default settings
# Optimize:
- max_step limit for stability
- rtol/atol tuning for speed vs accuracy
- dense_output=False for memory savings
- method='BDF' vs 'Radau' comparison
```

#### 1.3 Parallel Parameter Sweeps
```python
# Use existing ProcessPoolExecutor more effectively
- Batch similar simulations together
- Share initial state computations
- Parallel across parameters, not just Monte Carlo
```

### Phase 2: C Extension (Medium Term)

#### 2.1 Critical Path Translation
```c
// File: core/fast_rhs.c
// Compile: python setup.py build_ext --inplace

// Functions to translate:
- rhs_multicompartment()
- All channel current calculations
- Kinetic alpha/beta functions
- Compartment coupling matrix

// Use Cython for seamless integration
```

#### 2.2 Cython Implementation Plan
```python
# File: core/cython_rhs.pyx

# Benefits:
- Type declarations for speed
- Direct C function calls
- Memory views for arrays
- GIL release for parallel sections

# Target: 5-10x speedup for RHS evaluations
```

### Phase 3: Multithreading (Advanced)

#### 3.1 OpenMP Parallelization
```c
// Parallel regions:
- Channel current calculations (per compartment)
- Gate variable updates
- Parameter sweeps

// Avoid parallelizing solve_ivp itself (not thread-safe)
```

#### 3.2 GPU Acceleration (Future)
```cuda
// For very large simulations:
- CUDA kernels for channel calculations
- cuSOLVER for linear algebra
- Batch simulations on GPU
```

## Implementation Priority

### High Priority (This Session)
1. ✅ Time estimation and warnings - DONE
2. [ ] Tune solve_ivp parameters for speed
3. [ ] Add max_step limit to prevent hanging
4. [ ] Create Cython extension skeleton

### Medium Priority (Next Session)
5. [ ] Implement Cython version of rhs_multicompartment
6. [ ] Optimize channel kinetics calculations
7. [ ] Parallel parameter sweep utilities

### Low Priority (Future)
8. [ ] Full C extension module
9. [ ] OpenMP parallelization
10. [ ] GPU acceleration

## Expected Performance Gains

| Optimization | Speedup | Effort |
|-------------|---------|--------|
| solve_ivp tuning | 1.5-2x | Low |
| Cython RHS | 5-10x | Medium |
| Multithreading | 2-4x (multi-core) | High |
| GPU | 10-50x | Very High |

## Code Structure

```
core/
├── rhs.py                    # Current Python RHS
├── rhs_cython.pyx            # Cython version (new)
├── fast_rhs.c                # Pure C version (future)
├── solver.py                 # Updated with optimization
└── optimization/
    ├── __init__.py
    ├── cython_setup.py       # Build configuration
    └── benchmarks.py         # Performance tests
```

## Testing Plan

Create benchmark tests:
- Simple model (Na+K only): 100ms
- Medium model (Na+K+Ih+ICa): 100ms  
- Complex model (all channels): 100ms
- Compare Python vs Cython timings

## Notes

- Maintain backward compatibility
- Keep Python fallback for debugging
- Profile before optimizing (use cProfile)
- Test numerical accuracy after optimization
