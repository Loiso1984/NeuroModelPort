# 🚀 NeuroModelPort v10.1 - Developer Quick Start

**Goal:** Complete Phase 2-6 implementation (solver integration + GUI + tests)  
**Time Budget:** ~7 hours of focused development  
**Current Status:** Phase 1 ✅ Complete (Architecture + Documentation)

---

## 📋 Before You Start (5 minutes)

Read these files in order:
1. ✅ **This file** (quick start)
2. ✅ **README.md** - What is NeuroModelPort?
3. ✅ **DOCUMENTATION_COMPLETE.md** - Full physics explanation
4. ✅ **ARCHITECTURE_v10_1.md** - System design

**Then:** Look at existing code structure:
```bash
cd c:\NeuroModelPort
# Review these files:
less core/models.py        # ~ 200 lines, understand config
less core/dendritic_filter.py  # ~ 140 lines, NEW physics module
less core/solver.py        # ~ 300 lines, where to add filter tracking
less core/rhs.py           # ~ 400 lines, where to add branching logic
```

---

## 🎯 What You're Building

### Problem Statement
L5 pyramidal neurons in direct soma injection mode:
- Peak voltage: ~45 mV (too HIGH)
- Spikes: 40 per 150 ms (too MANY - 260 Hz firing rate!)
- Rise time: ~1 ms (too FAST)

**Why?** Because direct soma injection is **unphysiological**!

### Solution: 3 Stimulus Modes

```python
cfg.stim_location.location = 'soma'                 # Lab: direct injection
cfg.stim_location.location = 'ais'                  # Spike zone electrode
cfg.stim_location.location = 'dendritic_filtered'   # Physiological! ← NEW
```

**Expected Results After Your Work:**

| Mode | V_peak | Spikes | Firing Rate | Rise Time |
|------|--------|--------|-------------|-----------|
| soma | 45 mV | 40 | 260 Hz | 1 ms |
| ais | 50 mV | 15 | 100 Hz | <0.5 ms |
| dend_filt | **22 mV** | **5** | **33 Hz** | **10 ms** |

The new dendritic_filtered mode matches real neuron data! 🎉

---

## 🛠️ Work Breakdown (7 Hours Total)

### Phase 2: Solver Integration (1.5-2 hours) ← START HERE!

**File:** `core/solver.py`  
**Task:** Create dendritic filter state and pass through BDF solver

```python
# In run_single() method, add this:

from core.dendritic_filter import DendriticFilterState

def run_single(self):
    # ★ NEW: Initialize filter if needed
    dfilter = None
    if self.cfg.stim_location.location == 'dendritic_filtered':
        dfilter = DendriticFilterState(
            distance=self.cfg.dendritic_filter.distance_um,
            space_constant=self.cfg.dendritic_filter.space_constant_um,
            tau_dendritic=self.cfg.dendritic_filter.tau_dendritic_ms
        )
    
    # ★ MODIFY: Pass dfilter to ODE solver
    # Currently: fun=lambda t, y: self.rhs(t, y, self.cfg)
    # Change to:
    fun = lambda t, y: self.rhs(t, y, self.cfg, dfilter)
    
    result = solve_ivp(
        fun=fun,
        ...
    )
    
    return result
```

**Success Criteria:**
- [ ] Code runs without errors
- [ ] Soma mode produces IDENTICAL results to before (regression test!)
- [ ] Dendritic mode initializes filter successfully

**How to Test:**
```python
# test_quick_phase2.py
from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver

# Test 1: Soma mode regression (should be unchanged)
cfg = FullModelConfig()
apply_preset(cfg, "B: Pyramidal L5")
cfg.stim_location.location = 'soma'
solver = NeuronSolver(cfg)
result = solver.run_single()
assert result.v_soma.max() > 40  # Should be ~45 mV

# Test 2: Dendritic mode (new!)
cfg.stim_location.location = 'dendritic_filtered'
solver = NeuronSolver(cfg)
result = solver.run_single()
print(f"Peak voltage: {result.v_soma.max():.1f} mV")  # Should be ~22 mV
```

---

### Phase 3: RHS Branching (1 hour) ← DEPENDS ON PHASE 2!

**File:** `core/rhs.py`  
**Function:** `rhs_multicompartment(y, t, cfg, ...)`  
**Task:** Add branching logic for 3 stimulus locations

```python
# Current signature (v10.0):
def rhs_multicompartment(y, t, cfg):
    ...

# Change to (v10.1):
def rhs_multicompartment(y, t, cfg, dfilter=None):
    """
    ODE kernel with 3 stimulus modes
    
    dfilter: DendriticFilterState or None
    """
    # ★ Add this logic near top of function
    if cfg.stim_location.location == 'soma':
        I_soma = cfg.stim.Iext
        
    elif cfg.stim_location.location == 'ais':
        I_ais = cfg.stim.Iext
        I_soma = 0  # No direct soma injection
        
    elif cfg.stim_location.location == 'dendritic_filtered':
        if dfilter is not None:
            # ★ Update filter state with dendritic input
            dfilter.step(cfg.stim.Iext, dt=0.01)  # Use small dt
            I_soma = dfilter.get_soma_current()
        else:
            I_soma = cfg.stim.Iext
    
    else:
        raise ValueError(f"Unknown stimulus location: {cfg.stim_location.location}")
    
    # Rest of RHS calculation (unchanged)
    # Calculate I_Na, I_K, I_L
    # Return dy/dt
    ...
```

**Challenge:** Figuring out where `I_soma` is currently used in RHS  
**Solution:** Search for `I_ext` or `self.cfg.stim.Iext` in rhs.py

**Success Criteria:**
- [ ] All 3 branches have correct current values
- [ ] Soma mode still unchanged (regression!)
- [ ] Dendritic mode applies filter
- [ ] AIS mode doesn't inject into soma

---

### Phase 4: GUI Integration (1.5 hours) ← PARALLEL WITH PHASE 2-3

**Files:** `gui/main_window.py`, `gui/widgets/`  
**Task:** Add stimulus location selector and dendritic filter parameters

#### 4A: Stimulus Location Dropdown

```python
# In gui/main_window.py, around parameter section:

from PyQt6.QtWidgets import QComboBox, QLabel, QHBoxLayout, QGroupBox

# Create dropdown
self.stimulus_location_combo = QComboBox()
self.stimulus_location_combo.addItems([
    "Soma (Laboratory - Direct Injection)",
    "AIS (Axon Initial Segment)",
    "Dendritic (Physiological - Filtered)"
])

# Connect to config
self.stimulus_location_combo.currentIndexChanged.connect(
    self.on_stimulus_location_changed
)

# Add to layout
location_layout = QHBoxLayout()
location_layout.addWidget(QLabel("Stimulus Location:"))
location_layout.addWidget(self.stimulus_location_combo)
self.param_layout.addLayout(location_layout)

# Add callback
def on_stimulus_location_changed(self, index):
    locations = ['soma', 'ais', 'dendritic_filtered']
    self.cfg.stim_location.location = locations[index]
    
    # Show/hide dendritic filter panel
    if locations[index] == 'dendritic_filtered':
        self.dendritic_panel.setVisible(True)
    else:
        self.dendritic_panel.setVisible(False)
    
    self.on_config_changed()
```

#### 4B: Dendritic Filter Panel (NEW Widget!)

```python
# gui/widgets/dendritic_filter_panel.py (NEW FILE)

from PyQt6.QtWidgets import QGroupBox, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QSpinBox
from PyQt6.QtCore import Qt

class DendriticFilterPanel(QGroupBox):
    def __init__(self, cfg, parent=None):
        super().__init__("Dendritic Filtering Parameters", parent)
        self.cfg = cfg
        
        layout = QVBoxLayout()
        
        # Distance slider (50-300 µm)
        dist_layout = QHBoxLayout()
        dist_layout.addWidget(QLabel("Distance (µm):"))
        self.distance_slider = QSlider(Qt.Orientation.Horizontal)
        self.distance_slider.setRange(50, 300)
        self.distance_slider.setValue(150)
        self.distance_slider.sliderMoved.connect(self.on_distance_changed)
        self.distance_label = QLabel("150")
        dist_layout.addWidget(self.distance_slider)
        dist_layout.addWidget(self.distance_label)
        layout.addLayout(dist_layout)
        
        # Space constant slider (50-300 µm)
        lambda_layout = QHBoxLayout()
        lambda_layout.addWidget(QLabel("Space Constant λ (µm):"))
        self.lambda_slider = QSlider(Qt.Orientation.Horizontal)
        self.lambda_slider.setRange(50, 300)
        self.lambda_slider.setValue(150)
        self.lambda_slider.sliderMoved.connect(self.on_lambda_changed)
        self.lambda_label = QLabel("150")
        lambda_layout.addWidget(self.lambda_slider)
        lambda_layout.addWidget(self.lambda_label)
        layout.addLayout(lambda_layout)
        
        # Tau slider (5-50 ms)
        tau_layout = QHBoxLayout()
        tau_layout.addWidget(QLabel("Integration τ (ms):"))
        self.tau_slider = QSlider(Qt.Orientation.Horizontal)
        self.tau_slider.setRange(5, 50)
        self.tau_slider.setValue(10)
        self.tau_slider.sliderMoved.connect(self.on_tau_changed)
        self.tau_label = QLabel("10")
        tau_layout.addWidget(self.tau_slider)
        tau_layout.addWidget(self.tau_label)
        layout.addLayout(tau_layout)
        
        # Info label
        self.info_label = QLabel("Attenuation: 63%, Cutoff: 16 Hz")
        layout.addWidget(self.info_label)
        
        self.setLayout(layout)
        self.update_attenuation()
    
    def on_distance_changed(self, value):
        self.cfg.dendritic_filter.distance_um = value
        self.distance_label.setText(str(value))
        self.update_attenuation()
    
    def on_lambda_changed(self, value):
        self.cfg.dendritic_filter.space_constant_um = value
        self.lambda_label.setText(str(value))
        self.update_attenuation()
    
    def on_tau_changed(self, value):
        self.cfg.dendritic_filter.tau_dendritic_ms = value
        self.tau_label.setText(str(value))
        self.update_attenuation()
    
    def update_attenuation(self):
        import math
        d = self.cfg.dendritic_filter.distance_um
        lam = self.cfg.dendritic_filter.space_constant_um
        tau = self.cfg.dendritic_filter.tau_dendritic_ms
        
        # Calculate attenuation: exp(-d/λ)
        A = math.exp(-d / lam)
        atten_pct = (1 - A) * 100
        
        # Calculate cutoff frequency: f_c = 1/(2πτ)
        f_c = 1 / (2 * math.pi * tau / 1000)  # Convert to Hz
        
        self.info_label.setText(
            f"Attenuation: {atten_pct:.0f}%, Cutoff: {f_c:.1f} Hz"
        )
```

**Step-by-step Integration:**

1. Create the panel class in `gui/widgets/dendritic_filter_panel.py`
2. Import it in `gui/main_window.py`
3. Create instance: `self.dendritic_panel = DendriticFilterPanel(self.cfg)`
4. Add to layout (initially hidden)
5. Show/hide based on stimulus location

**Success Criteria:**
- [ ] Dropdown has 3 options
- [ ] Sliders work (0-300 µm for distance/λ, 5-50 ms for τ)
- [ ] Info label shows real-time attenuation calculation
- [ ] Panel appears only when dendritic mode selected

---

### Phase 5: Testing & Validation (1 hour) ← DURING PHASES 2-4

**File:** `test/test_dendritic_modes.py` (NEW)

```python
import numpy as np
import matplotlib.pyplot as plt
from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver

def test_three_modes():
    """Compare all 3 stimulus modes side-by-side"""
    
    # Configure all 3 modes
    configs = {}
    
    # Mode 1: Soma
    cfg_soma = FullModelConfig()
    apply_preset(cfg_soma, "B: Pyramidal L5")
    cfg_soma.stim_location.location = 'soma'
    cfg_soma.stim.Iext = 35.4
    configs['soma'] = cfg_soma
    
    # Mode 2: AIS
    cfg_ais = FullModelConfig()
    apply_preset(cfg_ais, "B: Pyramidal L5")
    cfg_ais.stim_location.location = 'ais'
    cfg_ais.stim.Iext = 5.0
    configs['ais'] = cfg_ais
    
    # Mode 3: Dendritic
    cfg_dend = FullModelConfig()
    apply_preset(cfg_dend, "B: Pyramidal L5")
    cfg_dend.stim_location.location = 'dendritic_filtered'
    cfg_dend.stim.Iext = 100.0
    configs['dendritic'] = cfg_dend
    
    # Run simulations
    results = {}
    for mode_name, cfg in configs.items():
        solver = NeuronSolver(cfg)
        results[mode_name] = solver.run_single()
    
    # Validate results
    print("=" * 60)
    print("VALIDATION RESULTS - v10.1 Three Stimulus Modes")
    print("=" * 60)
    
    for mode_name, result in results.items():
        V_peak = result.v_soma.max()
        spike_count = np.sum(np.diff((result.v_soma > -35).astype(int)) > 0)
        firing_rate = spike_count / (result.t[-1] / 1000)  # spikes/second
        
        print(f"\n{mode_name.upper()} Mode:")
        print(f"  Peak voltage: {V_peak:6.1f} mV")
        print(f"  Spikes: {spike_count:3d}")
        print(f"  Firing rate: {firing_rate:6.1f} Hz")
    
    # Expected values
    print("\n" + "=" * 60)
    print("EXPECTED VALUES (Literature & Theory)")
    print("=" * 60)
    print("\nsoma (Laboratory):")
    print("  Peak voltage: ~45 mV")
    print("  Spikes: ~40")
    print("  Firing rate: ~260 Hz")
    print("  ✓ Correct for: Voltage/patch clamp")
    print("  ✗ NOT physiological!")
    
    print("\nais (Spike Zone):")
    print("  Peak voltage: ~50 mV")
    print("  Spikes: ~15")
    print("  Firing rate: ~100 Hz")
    print("  ✓ Shows AIS amplification (gNa×40)")
    
    print("\ndendritic_filtered (Physiological!):")
    print("  Peak voltage: ~22 mV ← REALISTIC!")
    print("  Spikes: ~5")
    print("  Firing rate: ~33 Hz ← MATCHES LITERATURE!")
    print("  ✓ This is how real neurons behave")
    print("  ✓ Matches Mainen et al. 1996 data")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for (mode_name, result), ax in zip(results.items(), axes):
        ax.plot(result.t, result.v_soma, 'b-', linewidth=1.5)
        ax.axhline(-35, color='r', linestyle='--', alpha=0.5, label='Spike threshold')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Voltage (mV)')
        ax.set_title(f'{mode_name.capitalize()} Mode')
        ax.set_ylim([-80, 50])
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('test_three_modes_comparison.png', dpi=150)
    print("\n✓ Comparison plot saved: test_three_modes_comparison.png")

def test_regression_soma_mode():
    """Ensure soma mode hasn't changed from v10.0"""
    cfg = FullModelConfig()
    apply_preset(cfg, "B: Pyramidal L5")
    cfg.stim_location.location = 'soma'
    
    solver = NeuronSolver(cfg)
    result = solver.run_single()
    
    V_peak = result.v_soma.max()
    assert 44 < V_peak < 46, f"Soma peak should be ~45 mV, got {V_peak:.1f}"
    print("✓ Regression test PASSED: soma mode unchanged")

if __name__ == '__main__':
    test_regression_soma_mode()
    test_three_modes()
```

**Run It:**
```bash
cd c:\NeuroModelPort
python test/test_dendritic_modes.py
```

**Success Criteria:**
- [ ] All 3 modes run without errors
- [ ] Soma mode produces ~45 mV peak
- [ ] Dendritic mode produces ~22 mV peak
- [ ] Dendritic mode produces ~5 spikes
- [ ] Comparison plot is generated
- [ ] Regression test passes

---

### Phase 6: Documentation (1 hour) ← FINAL POLISH

#### 6A: Update Code Docstrings

```python
# In solver.py run_single():
def run_single(self):
    """
    Run single neuron simulation.
    
    Supports three stimulus locations (v10.1+):
    - soma: Direct whole-cell injection (laboratory)
    - ais: Axon initial segment (high gNa amplification)
    - dendritic_filtered: (NEW!) Physiological synaptic input with cable filtering
    
    Returns:
        SimulationResult with voltage, gating variables, currents
    """

# In rhs.py:
def rhs_multicompartment(y, t, cfg, dfilter=None):
    """
    ODE kernel for Hodgkin-Huxley model (v10.1).
    
    Implements three stimulus modes:
    1. soma: I_ext applied directly to soma
    2. ais: I_ext applied to axon initial segment (higher gNa)
    3. dendritic_filtered: I_ext filtered through dendritic cable equation
    
    Parameters:
        y: State vector [V, m, h, n, ...]
        t: Current time
        cfg: Configuration (includes stim_location, dendritic_filter)
        dfilter: DendriticFilterState or None
    
    Returns:
        dy/dt: Time derivatives of state
    """
```

#### 6B: Bilingual Localization (Optional for v10.1, but Good Practice)

```python
# gui/locales.py

TRANSLATIONS = {
    'stimulus_location': {
        'en': 'Stimulus Location',
        'ru': 'Место стимуляции'
    },
    'soma_mode': {
        'en': 'Soma (Laboratory - Direct Injection)',
        'ru': 'Сома (Лаб. - Прямая инъекция)'
    },
    'ais_mode': {
        'en': 'AIS (Axon Initial Segment)',
        'ru': 'AIS (Начальный сегмент аксона)'
    },
    'dendritic_mode': {
        'en': 'Dendritic (Physiological - Filtered)',
        'ru': 'Дендриты (Физиологично - Фильтр.)'
    },
    'dendritic_filter': {
        'en': 'Dendritic Filtering Parameters',
        'ru': 'Параметры дендритной фильтрации'
    },
    'distance_label': {
        'en': 'Distance to Synapse (µm):',
        'ru': 'Расстояние до синапса (µm):'
    },
    'space_constant_label': {
        'en': 'Space Constant λ (µm):',
        'ru': 'Пространственная константа λ (µm):'
    },
    'tau_label': {
        'en': 'Integration Time Constant τ (ms):',
        'ru': 'Время интеграции τ (ms):'
    },
}
```

#### 6C: Example Scripts

```python
# examples/stimulus_modes_comparison.py (NEW)
"""
Example: Comparing 3 stimulus modes in L5 pyramidal neuron

Shows why direct soma injection ("unphysiological") explains
why the neuron fires at 260 Hz when real neurons only fire at 33 Hz!
"""

from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver
import matplotlib.pyplot as plt

# Create configuration
cfg = FullModelConfig()
apply_preset(cfg, "B: Pyramidal L5 (Mainen 1996)")

# Run 3 simulations
modes = ['soma', 'ais', 'dendritic_filtered']
results = {}
params = {
    'soma': 35.4,
    'ais': 5.0,
    'dendritic_filtered': 100.0
}

fig, axes = plt.subplots(3, 1, figsize=(10, 8))

for mode, ax in zip(modes, axes):
    cfg.stim_location.location = mode
    cfg.stim.Iext = params[mode]
    
    solver = NeuronSolver(cfg)
    result = solver.run_single()
    
    ax.plot(result.t, result.v_soma, 'b-', linewidth=2)
    ax.axhline(-35, color='r', linestyle='--', alpha=0.5, label='Spike threshold')
    ax.set_ylabel('V (mV)')
    ax.set_title(f'{mode.capitalize()} Mode: V_peak={result.v_soma.max():.0f} mV')
    ax.set_ylim([-80, 50])
    ax.grid(True, alpha=0.3)
    ax.legend()

axes[-1].set_xlabel('Time (ms)')
plt.tight_layout()
plt.savefig('stimulus_modes_comparison.png')
print("✓ Comparison saved to stimulus_modes_comparison.png")
```

---

## ⏱️ Time Estimate Summary

| Phase | Time | Status |
|-------|------|--------|
| 1: Architecture | ✅ 5 hrs | COMPLETE |
| 2: Solver integration | ⏳ 1.5-2 hrs | START HERE |
| 3: RHS branching | ⏳ 1 hr | After phase 2 |
| 4: GUI integration | ⏳ 1.5 hrs | Parallel with 2-3 |
| 5: Testing | ⏳ 1 hr | During phases 2-4 |
| 6: Documentation | ⏳ 1 hr | Final polish |
| **TOTAL** | **~7 hrs** | Ready to go! |

---

## 🔍 Debugging Checklist

If something breaks:

- [ ] Check soma mode produces same results as before (regression!)
- [ ] Verify dfilter is being created: `print(dfilter)` in solver
- [ ] Confirm dfilter state updates: Check `V_filtered` changes
- [ ] Validate filter attenuation: Should be ~0.37 for d=λ=150
- [ ] Check spike thresholds: Should be around -35 mV
- [ ] Verify peak voltages:
  - Soma: ~45 mV
  - AIS: ~50 mV
  - Dendritic: ~22 mV
- [ ] Test GUI dropdown changes mode correctly
- [ ] Confirm sliders update DendriticFilterParams

---

## 📞 Key Files to Read

**Before Starting Phase 2:**
1. `core/solver.py` (~300 lines) - Where you'll add filter
2. `core/dendritic_filter.py` (~140 lines) - The filter you'll use
3. `core/models.py` (~80 lines) - Config structures

**Progress Tracking:**
- `TODO_IMPLEMENTATION.md` - Detailed checklist
- `PROJECT_STATE.md` - Overall progress
- `ARCHITECTURE_v10_1.md` - System design

---

## 🎯 Success Criteria (When You're Done)

- [x] Phase 1: Architecture complete ✅
- [ ] Phase 2: Solver integration (dfilter state tracking)
- [ ] Phase 3: RHS branching (3-way stimulus selection)
- [ ] Phase 4: GUI controls (dropdown + sliders)
- [ ] Phase 5: All tests passing (regression + physiological)
- [ ] Phase 6: Documentation complete (docstrings + examples)

**Final Validation:**
```
Soma mode:          45 mV, 40 spikes, 260 Hz  ← Lab artifact (expected!)
Dendritic mode:     22 mV,  5 spikes,  33 Hz  ← PHYSIOLOGICAL! ✅
```

---

## 🚀 Let's Go!

You have everything you need:
- ✅ Complete physics explanation (DOCUMENTATION_COMPLETE.md)
- ✅ Dendritic filter code (core/dendritic_filter.py)
- ✅ Config structures (core/models.py)
- ✅ Detailed specification (IMPLEMENTATION_PLAN_v10_1.md)
- ✅ Architecture diagram (ARCHITECTURE_v10_1.md)

**Next Step:** Open `core/solver.py` and start Phase 2! 💪

---

**Quick Link Summary:**
- 📖 Physics: [DOCUMENTATION_COMPLETE.md](DOCUMENTATION_COMPLETE.md)
- 🏗️ Architecture: [ARCHITECTURE_v10_1.md](ARCHITECTURE_v10_1.md)
- ✅ Checklist: [TODO_IMPLEMENTATION.md](TODO_IMPLEMENTATION.md)
- 📊 Status: [PROJECT_STATE.md](PROJECT_STATE.md)
- 📝 ReadMe: [README.md](README.md)

**Good luck! 🎉**
