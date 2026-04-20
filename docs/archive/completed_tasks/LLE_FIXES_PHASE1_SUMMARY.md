# LLE Critical Bug Fixes — Phase 1 Summary

**Date:** 2026-01-14  
**Status:** ✅ COMPLETED

---

## ✅ FIXED BUGS

### 1. Duplicate LLE Recording (CRITICAL)
**File:** `core/native_loop.py:226-230`
- **Problem:** LLE recorded at loop start (before step integration) with stale data
- **Fix:** Removed duplicate recording, kept only at final output section after all computations

### 2. Re-orthonormalization Timing (CRITICAL)
**File:** `core/native_loop.py:482-567`
- **Problem:** Re-orthonormalization checked before `t += dt`, causing early triggers
- **Fix:** Moved entire block after `t += dt`, changed to absolute time scheduling

### 3. Distance Threshold (MEDIUM)
**File:** `core/native_loop.py:559`
- **Problem:** `dist > 1e-30` too small for float64 stability
- **Fix:** Changed to `dist > 1e-12` for reasonable numerical threshold

### 4. Cross-talk Bug — VERIFIED SAFE
**File:** `core/native_loop.py:431-475`
- **Status:** Calcium and ATP dynamics already correctly inside trajectory loop
- **Verified:** Both main (y) and perturbed (y_pert) trajectories update independently

---

## 🚀 NEW FEATURES

### LLE Subspace Modes
**4 modes for flexible sensitivity analysis:**

| Mode | Includes | Use Case |
|------|----------|----------|
| `v_only` (default) | Voltage compartments only | Standard neurophysiology |
| `v_and_gates` | V + all HH gating variables | Full phase space analysis |
| `full_state` | All state variables (Ca, ATP, etc.) | Theoretical completeness |
| `custom` | User-defined mask | Targeted sensitivity hints |

### Helper Functions

```python
# Create custom mask for targeted analysis
def make_lle_subspace_mask(
    n_comp: int,
    state_offsets,
    include_v: bool = True,
    include_gates: list[str] | None = None,
    include_ca: bool = False,
    include_atp: bool = False,
    include_nai: bool = False,
    include_ko: bool = False,
) -> np.ndarray

# Auto-generate weights for mixed-scale variables
def make_lle_weights(mask, n_comp, state_offsets) -> np.ndarray
```

**Auto-weighting:**
- Voltage (~10mV): weight 0.1
- Gates (0-1): weight 1.0
- Calcium (~1e-5 mM): weight 1e5
- ATP (~5 mM): weight 0.2
- Na_i (~10 mM): weight 0.1
- K_o (~5 mM): weight 0.2

---

## 📁 FILES MODIFIED

| File | Changes |
|------|---------|
| `core/native_loop.py` | +180 lines: subspace parameters, helper functions, corrected timing |
| `core/solver.py` | +4 lines: new parameters in `run_native()` signature and call |

---

## 🔧 API USAGE

### Basic Usage (default v_only mode)
```python
solver = NeuronSolver(config)
result = solver.run_native(calc_lle=True, lle_t_evolve=5.0)
```

### Custom Subspace (sensitivity analysis)
```python
from core.native_loop import make_lle_subspace_mask, make_lle_weights

# Analyze only sodium channel contribution
mask = make_lle_subspace_mask(
    n_comp=2,
    state_offsets=offsets,
    include_v=True,
    include_gates=["m", "h"],  # Na+ activation/inactivation only
)
weights = make_lle_weights(mask, n_comp=2, state_offsets=offsets)

result = solver.run_native(
    calc_lle=True,
    lle_subspace_mode="custom",
    lle_custom_mask=mask,
    lle_weights=weights,
)
```

### Full Phase Space (HH dynamics)
```python
result = solver.run_native(
    calc_lle=True,
    lle_subspace_mode="v_and_gates",
)
```

---

## 🧪 VALIDATION

- ✅ `native_loop` compiles with Numba
- ✅ `solver` imports successfully
- ✅ Helper functions tested
- ✅ Parameter forwarding verified

---

## 🎯 NEXT PHASE (After Audit)

1. **GUI Chaos Tab:** Display `lle_convergence` curve
2. **Micro-optimizations:** Nernst caching, loop unrolling
3. **Fallback LLE:** Keep Rosenstein method for `calc_lle=False` runs
4. **Sensitivity Hints:** Auto-compare subspaces for parameter insights
