# DUAL STIM INTEGRATION - IMPLEMENTATION PLAN

**Status:** 📋 READY TO CODE  
**Approach:** Minimal safe changes (backward compatible)  
**Estimated time:** 2-3 hours

---

## 🎯 STRATEGY

**Goal:** Add dual stim support WITHOUT breaking Phase 6 single-stim baseline

**Approach:** 
- **RHS kernel:** Add optional dual stim parameters (default: None = single stim only)
- **Solver:** Detect DualStimulationConfig, pack dual params into args
- **Backward compat:** Single stim unaffected = all Phase 6 tests pass

---

## 📝 IMPLEMENTATION STEPS

### STEP 1: Modify rhs.py (RHS kernel)

**File:** `core/rhs.py`  
**Section:** `rhs_multicompartment()` function signature + stimulus calculation

**Changes:**
```python
# BEFORE (line ~90):
@njit(cache=True)
def rhs_multicompartment(
    t, y, n_comp,
    # ... existing 30+ parameters ...
    stype, iext, t0, td, atau, stim_comp, stim_mode,
    use_dfilter, dfilter_attenuation, dfilter_tau_ms
):

# AFTER:
@njit(cache=True)
def rhs_multicompartment(
    t, y, n_comp,
    # ... existing 30+ parameters ...
    stype, iext, t0, td, atau, stim_comp, stim_mode,
    use_dfilter, dfilter_attenuation, dfilter_tau_ms,
    # Dual stimulation (optional)
    dual_stim_enabled,  # 0=single, 1=dual
    stype_2, iext_2, t0_2, td_2, atau_2, stim_comp_2, stim_mode_2,
    dfilter_attenuation_2, dfilter_tau_ms_2
):
```

**Stimulus calculation (line ~160-190):**
```python
# BEFORE:
base_current = get_stim_current(t, stype, iext, t0, td, atau)
# ... apply based on stim_mode ...
i_stim[...] = base_current  # or filtered version

# AFTER:
base_current = get_stim_current(t, stype, iext, t0, td, atau)
# ... apply based on stim_mode ...  
if stim_mode == 0:
    i_stim[stim_comp] = base_current
# ... other modes ...

# IF dual_stim_enabled == 1:
if dual_stim_enabled == 1:
    base_current_2 = get_stim_current(t, stype_2, iext_2, t0_2, td_2, atau_2)
    # ... apply based on stim_mode_2 ...
    if stim_mode_2 == 0:
        i_stim[stim_comp_2] += base_current_2  # NOTE: += not =
    # ... other modes ...
```

**KEY:** Use `+=` for secondary stim so it adds to whatever's already there (primary or base)

**Test:** Single stim should work exactly as before (dual_stim_enabled=0)

---

### STEP 2: Modify solver.py (argument packing)

**File:** `core/solver.py`  
**Section:** `run_single()` method, args packing (~line 92-125)

**Changes:**
```python
# Add detection (after line ~80):
dual_stim_enabled = 0
stype_2, iext_2, t0_2, td_2, atau_2, stim_comp_2, stim_mode_2 = 0, 0, 0, 0, 0, 0, 0
dfilter_attenuation_2, dfilter_tau_ms_2 = 1.0, 0.0

# Check if dual stim config exists
if hasattr(cfg, 'dual_stimulation') and cfg.dual_stimulation is not None:
    dual_cfg = cfg.dual_stimulation
    if hasattr(dual_cfg, 'enabled') and dual_cfg.enabled:
        dual_stim_enabled = 1
        
        # Primary stim (might override defaults if needed)
        # ... secondary stim mapping ...
        stype_2 = s_map.get(dual_cfg.secondary_stim_type, 0)
        iext_2 = dual_cfg.secondary_Iext
        t0_2 = dual_cfg.secondary_start
        td_2 = dual_cfg.secondary_duration
        atau_2 = dual_cfg.secondary_alpha_tau
        
        # Secondary location
        stim_mode_2_map = {'soma': 0, 'ais': 1, 'dendritic_filtered': 2}
        stim_mode_2 = stim_mode_2_map.get(dual_cfg.secondary_location, 0)
        
        # Secondary dendritic filtering
        if stim_mode_2 == 2:
            dfilter_attenuation_2 = np.exp(
                -dual_cfg.secondary_distance_um / dual_cfg.secondary_space_constant_um
            )
            dfilter_tau_ms_2 = dual_cfg.secondary_tau_dendritic_ms

# Add to args tuple (line ~125):
args = (
    # ... existing args ...
    stype, cfg.stim.Iext,
    cfg.stim.pulse_start, cfg.stim.pulse_dur,
    cfg.stim.alpha_tau, cfg.stim.stim_comp, stim_mode,
    use_dfilter, attenuation,
    cfg.dendritic_filter.tau_dendritic_ms if use_dfilter == 1 else 0.0,
    # NEW: dual stimulation params
    dual_stim_enabled,
    stype_2, iext_2, t0_2, td_2, atau_2, stim_comp_2, stim_mode_2,
    dfilter_attenuation_2, dfilter_tau_ms_2,
)
```

**Test:** Single stim should still work (dual_stim_enabled=0 by default)

---

### STEP 3: Add DualStimulationConfig to FullModelConfig

**File:** `core/models.py`  
**Section:** FullModelConfig class definition (~line 220)

**Changes:**
```python
from .dual_stimulation import DualStimulationConfig  # Add import

class FullModelConfig(BaseModel):
    # ... existing fields ...
    dendritic_filter: DendriticFilterParams = DendriticFilterParams()
    
    # NEW: Optional dual stimulation config
    dual_stimulation: Optional[DualStimulationConfig] = None  # Will be None or DualStimulationConfig()
    
    analysis: AnalysisParams = AnalysisParams()
```

**Note:** Optional!  Not required for backward compat

---

### STEP 4: Create test file

**File:** `test_dual_stim.py`  
**Purpose:** Verify dual stim works + backward compat

```python
from core.solver import NeuronSolver
from core.models import FullModelConfig
from core.dual_stimulation import DualStimulationConfig
from core.presets import apply_preset

# TEST 1: Single stim (Phase 6 baseline)
cfg = FullModelConfig()
apply_preset(cfg, 'B: Pyramidal L5 (Mainen 1996)')
solver = NeuronSolver(cfg)
result = solver.run_single()
# Verify: L5 soma fires @ 7.5 Hz (exact match Phase 6)

# TEST 2: Soma + Dendritic (dual stim)
cfg = FullModelConfig()
apply_preset(cfg, 'B: Pyramidal L5 (Mainen 1996)')
cfg.dual_stimulation = DualStimulationConfig()
cfg.dual_stimulation.enabled = True
cfg.dual_stimulation.primary_location = 'soma'
cfg.dual_stimulation.primary_Iext = 10.0
cfg.dual_stimulation.secondary_location = 'dendritic_filtered'
cfg.dual_stimulation.secondary_Iext = -5.0
cfg.dual_stimulation.secondary_stim_type = 'GABAA'
solver = NeuronSolver(cfg)
result = solver.run_single()
# Verify: Frequency lower than soma-only (inhibition works)

# TEST 3: soma + AIS (phase offset)
# ... similar structure ...
```

---

## ⚠️ SAFETY CHECKLIST

- [ ] RHS kernel changes are **LOCALIZED** to stimulus calculation block
- [ ] Single stim = dual_stim_enabled=0 by default (zero breaking changes)
- [ ] All args parameters have sensible defaults
- [ ] No changes to membrane equation or gate dynamics
- [ ] DualStimulationConfig is Optional[...] = None (backward compat)

---

## 🚀 EXECUTION ORDER

1. **Modify rhs.py** (15-20 min)
   - Add dual stim params to signature
   - Add dual stim logic to stimulus block
   - **NO other changes**

2. **Test single stim still works** (5 min)
   - Run Phase 6 baseline test
   - Verify L5 7.5 Hz (exact)

3. **Add Optional field to FullModelConfig** (5 min)
   - Import DualStimulationConfig
   - Add dual_stimulation field

4. **Modify solver.py** (10-15 min)
   - Add dual stim detection
   - Pack into args
   - **Test single stim again** (5 min)

5. **Create test_dual_stim.py** (20 min)
   - Test 1: Soma + Dendritic (GABA-A)
   - Test 2: Soma + AIS (offset)
   - Test 3: Verify backward compat

6. **Run full test suite** (10 min)
   - Single stim tests
   - Dual stim tests
   - Phase 6 baseline

---

## 📋 SUCCESS CRITERIA

✅ Single stim works **exactly** as Фхаэ 6 (no regression)  
✅ Dual stim soma+dend works (soma frequency reduced by > 10%)  
✅ Dual stim soma+AIS offset works (clear phase separation)  
✅ Zero new bugs or exceptions  
✅ All integration tests pass  

---

**NEXT ACTION:** Begin with Step 1 (modify rhs.py)

