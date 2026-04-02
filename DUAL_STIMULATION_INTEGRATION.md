# DUAL STIMULATION INTEGRATION PLAN

**Статус:** 📋 IMPLEMENTATION ROADMAP  
**Критичность:** 🔴 HIGH - v10.1 feature  
**Дата:** 2026-04-01

---

## 📊 ТЕКУЩЕЕ СОСТОЯНИЕ

### ✅ Существует
- `core/dual_stimulation.py` - Полный модуль с `get_dual_stim_current()` функцией
- `core/dual_stimulation_presets.py` - Конфиги для dual сценариев
- `gui/dual_stimulation_widget.py` - GUI компонент
- Поддержка AMPA, NMDA, GABAA, GABAB в стим моделях

### ❌ НЕ интегрировано
- `solver.py` не использует DualStimulation
- `rhs.py` (RHS kernel) не получает dual stim токи
- `main.py` не вызывает `dual_stimulation_widget.py`

---

## 🎯 ИНТЕГРАЦИЯ - ШАГИ

### ШАГИ ИНТЕГРАЦИИ:

```
STEP 1: Modify solver.py
  ├─ Detect if dual_stim enabled in config
  ├─ If YES: use get_dual_stim_current() instead of single stim func
  ├─ Pack dual stim params into args tuple for RHS
  └─ Pass to solve_ivp()

STEP 2: Modify rhs.py (RHS kernel)
  ├─ Add dual_stim_enabled parameter
  ├─ If enabled: compute I_stim = I_primary + I_secondary
  ├─ If disabled: compute I_stim as before (backward compat)
  └─ Test with known single-stim cases

STEP 3: Test Dual Stimulation
  ├─ Test 1: Soma excite + Dendritic inhibit
  ├─ Test 2: Soma excite + AIS excite (phase offset)
  ├─ Test 3: Temporal summation (sublinear vs supralinear)
  └─ Test 4: Verify backward compat with single stim

STEP 4: GUI Integration (optional for now)
  ├─ Expose dual_stim_widget in main_window
  ├─ Allow user to configure dual scenarios
  └─ Display both stimuli in plots

STEP 5: Documentation
  ├─ Update NEURON_PRESETS_GUIDE.md
  ├─ Add examples: soma+dend, soma+ais
  └─ Add DUAL_STIMULATION_GUIDE.md
```

---

## 🔎 ИНТЕГРАЦИЯ - ДЕТАЛИ

### Где находятся стим токи в коде:

**solver.py:**
```python
# Около строки 102:
stype = s_map.get(cfg.stim.stim_type, 0)  # String → enum

# Нужно добавить:
if hasattr(cfg, 'dual_stimulation') and cfg.dual_stimulation and cfg.dual_stimulation.enabled:
    dual_stim_enabled = 1
    # Pack dual config into args
else:
    dual_stim_enabled = 0
    # Use single stim as before
```

**rhs.py:**
```python
# RHS kernel получает stimulus через args tuple
# Около строки где вычисляется I_stim:

I_stim = calculate_stimulus(t, stim_params)

# Нужно изменить на:
if dual_stim_enabled:
    I_stim = calculate_dual_stimulus(t, dual_stim_params)
else:
    I_stim = calculate_stimulus(t, stim_params)
```

---

## 🧪 ТЕСТОВЫЕ СЦЕНАРИИ

### Test 1: Soma EXCITE + Dendritic INHIBIT
```python
dual_cfg = DualStimulationConfig()
dual_cfg.primary_location = 'soma'
dual_cfg.primary_stim_type = 'const'
dual_cfg.primary_Iext = 20.0    # Excitatory

dual_cfg.secondary_location = 'dendritic_filtered'
dual_cfg.secondary_stim_type = 'GABAA'
dual_cfg.secondary_Iext = -10.0  # Inhibitory

Expected: Soma spikes reduced compared to soma-only
Verify: Frequency lower, Vpeak lower
```

### Test 2: Soma + AIS with Phase Offset
```python
dual_cfg.primary_location = 'soma'
dual_cfg.primary_Iext = 10.0
dual_cfg.primary_start = 100.0

dual_cfg.secondary_location = 'ais'
dual_cfg.secondary_Iext = 5.0
dual_cfg.secondary_start = 150.0  # 50ms delay

Expected: AIS fires first (lower latency), soma follows
Verify: Plot shows AIS spike before soma spike
```

### Test 3: Temporal Summation
```
Test case 1: Primary only
  → freq = 10 Hz

Test case 2: Secondary only (weak)
  → freq = 2 Hz

Test case 3: Both together
  → freq = ? (linear summation = 12 Hz, but may be nonlinear)

Verify: Measure actual frequency, compare to expectation
```

### Test 4: Backward Compatibility
```python
# Single stim should still work as before:
cfg = FullModelConfig()
apply_preset(cfg, 'B: Pyramidal L5 (Mainen 1996)')
# No dual_stim → should get same result as Phase 6

Expected: L5 soma mode = 7.5 Hz (like Phase 6)
Verify: Exact match
```

---

## 📝 IMPLEMENTATION CHECKLIST

### Phase 1: Core Integration
- [ ] Modify solver.py to detect dual_stim
- [ ] Modify rhs.py to support dual stimulus
- [ ] Ensure backward compatibility (test 4)

### Phase 2: Testing
- [ ] Test 1: Soma excite + Dend inhibit ✓
- [ ] Test 2: Soma + AIS phase offset ✓
- [ ] Test 3: Temporal summation ✓
- [ ] Test 4: Backward compat ✓

### Phase 3: Documentation
- [ ] Update NEURON_PRESETS_GUIDE.md with dual examples
- [ ] Create DUAL_STIMULATION_GUIDE.md
- [ ] Add to DOCUMENTATION_INDEX.md

### Phase 4: GUI (if time)
- [ ] Expose dual_stim_widget in main
- [ ] Allow user config of dual scenarios
- [ ] Test GUI interaction

---

## ⚠️ RISKS & MITIGATION

**Risk 1:** Breaking single stim (backward compat)
- **Mitigation:** Test single stim extensively (Test 4)
- **Verification:** Run all Phase 6 baseline tests

**Risk 2:** RHS kernel complexity
- **Mitigation:** Keep dual logic simple (I_stim = I1 + I2)
- **Verification:** Unit test RHS with known inputs

**Risk 3:** GUI widget untested
- **Mitigation:** Focus on solver/RHS integration first, GUI optional for Phase 7
- **Verification:** Manual testing if GUI enabled

---

## 🎬 START HERE

**Next action:**
1. Read current rhs.py structure (~200 lines)
2. Identify where stimulus current is calculated
3. Add dual stim support (minimal changes)
4. Test backward compat
5. Test dual scenarios

**Ready to implement?**

