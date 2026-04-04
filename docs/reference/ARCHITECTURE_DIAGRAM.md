# Implementation Architecture Diagram

## Tab Navigation Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    AnalyticsWidget (QTabWidget)             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [🧬 Passport] [📊 Traces] [⚙ Gates] [⚡ Currents] ...    │
│                                            ↑↑↑               │
│                                         NEW TAB!             │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                  Current Tab Content Area                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  When "⚡ Currents" tab is selected:                       │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ Membrane Potential (V_soma)                           │ │
│  │ ┌─────────────────────────────────────────────────┐  │ │
│  │ │  40 mV ─╱╲╱╲───────  Blue (#2060CC)           │  │ │
│  │ │        0  ────────────────────────────────     │  │ │
│  │ │ -80 mV                                         │  │ │
│  │ └─────────────────────────────────────────────────┘  │ │
│  │                                                       │ │
│  │ I_Na (pA)                                            │ │
│  │ ┌─────────────────────────────────────────────────┐  │ │
│  │ │ -200 pA ╱╱ ╲╲────  Red (#DC3232)  ← In-current  │  │ │
│  │ │          0 ────────────────────────────────     │  │ │
│  │ │ +100 pA                                         │  │ │
│  │ └─────────────────────────────────────────────────┘  │ │
│  │                                                       │ │
│  │ I_K (pA)                                             │ │
│  │ ┌─────────────────────────────────────────────────┐  │ │
│  │ │ -100 pA                                         │  │ │
│  │ │         ╱╱ ╲╲─────  Blue (#3264DC) ← Out-current │  │ │
│  │ │          0 ────────────────────────────────     │  │ │
│  │ │ +200 pA                                         │  │ │
│  │ └─────────────────────────────────────────────────┘  │ │
│  │                                                       │ │
│  │ I_Leak (pA)                                          │ │
│  │ ┌─────────────────────────────────────────────────┐  │ │
│  │ │  -50 pA ─────  Green (#32A050)                 │  │ │
│  │ │          0 ────────────────────────────────     │  │ │
│  │ │   50 pA                                         │  │ │
│  │ └─────────────────────────────────────────────────┘  │ │
│  │                                                       │ │
│  │        0       25       50       75      100         │ │
│  │        Time (ms) ─────────────────────────           │ │
│  │  [🔍 Pan | 🔄 Zoom | 💾 Save]  [Settings]          │ │
│  │  (Matplotlib toolbar)                               │ │
│  │                                                      │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Code Integration Map

```
update_analytics() [Line 200]
    │
    ├─ _update_passport()
    ├─ _update_traces()
    ├─ _update_gates()
    ├─ _update_currents() ← NEW! [Line 208]
    │   │
    │   ├─ Extract: result.currents (dict)
    │   ├─ Filter: Keep only active currents
    │   ├─ Layout: Dynamic subplot grid
    │   │
    │   ├─ Subplot 1: V_soma (reference)
    │   │   ├─ Plot: result.v_soma
    │   │   ├─ Color: #2060CC (blue)
    │   │   └─ Format: Via _configure_ax_interactive()
    │   │
    │   ├─ Subplot 2+: Individual currents
    │   │   ├─ Data: result.currents[name]
    │   │   ├─ Color: CHAN_COLORS[name]
    │   │   ├─ Zero-line: axhline(y=0)
    │   │   └─ Format: Via _configure_ax_interactive()
    │   │
    │   └─ Render: self.cvs_currents.draw()
    │
    ├─ _update_equil()
    ├─ _update_phase()
    ├─ _update_kymo()
    ├─ _update_energy()
    └─ _update_balance()
```

---

## Data Source: Simulation Result

```
┌──────────────────────────────────────────────────────────────┐
│                    SimulationResult Object                   │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  USES BY _update_currents():                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                                                      │   │
│  │  result.currents  Dict[str, ndarray]               │   │
│  │  ├─ 'Na'   → I_Na(t)   [nA/cm²] or [pA]           │   │
│  │  ├─ 'K'    → I_K(t)    [nA/cm²] or [pA]           │   │
│  │  ├─ 'Leak' → I_Leak(t) [nA/cm²] or [pA]           │   │
│  │  ├─ 'Ih'   → I_h(t)    [nA/cm²] or [pA] (opt.)   │   │
│  │  ├─ 'ICa'  → I_Ca(t)   [nA/cm²] or [pA] (opt.)   │   │
│  │  ├─ 'IA'   → I_A(t)    [nA/cm²] or [pA] (opt.)   │   │
│  │  └─ 'SK'   → I_SK(t)   [nA/cm²] or [pA] (opt.)   │   │
│  │                                                     │   │
│  │  result.v_soma  ndarray [mV]                       │   │
│  │  ├─ Shape: (N_timepoints,)                         │   │
│  │  └─ Range: typically -80 to +40 mV               │   │
│  │                                                     │   │
│  │  result.t  ndarray [ms]                            │   │
│  │  ├─ Shape: (N_timepoints,)                         │   │
│  │  └─ Range: 0 to t_sim                              │   │
│  │                                                     │   │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│            Populated by solver.py                          │
│            Lines 168-193 (_post_process_physics)           │
│                                                             │
└──────────────────────────────────────────────────────────────┘
```

---

## File Modification Summary

```
gui/analytics.py
│
├─ Line 128-132: Tab Registration
│  ├─ self.fig_currents = Figure(...)
│  ├─ self.cvs_currents = FigureCanvas(...)
│  └─ addTab(..., "⚡ Currents")
│
├─ Line 208: Update Pipeline Call
│  └─ self._update_currents(result)
│
└─ Line 401-438: New Method
   ├─ def _update_currents(self, result):
   ├─ Extract active currents
   ├─ Create dynamic layout
   ├─ Plot V_soma (row 1)
   ├─ Plot I_channel(t) for each active channel
   └─ Render figure
```

---

## Color Palette

```
Channel Color Mapping (CHAN_COLORS):

    Na (Sodium)
    ████████ #DC3232 Red
    
    K (Potassium)
    ████████ #3264DC Blue
    
    Leak (Passive)
    ████████ #32A050 Green
    
    Ih (H-current)
    ████████ #9632C8 Purple
    
    ICa (L-type Ca²⁺)
    ████████ #FA9600 Orange
    
    IA (A-type K)
    ████████ #00C8C8 Cyan
    
    SK (Small-conductance K)
    ████████ #C83296 Magenta

    Membrane Potential (reference)
    ████████ #2060CC Royal Blue
```

---

## Execution Timeline

```
┌─────────────────────────────────────────────────────────────┐
│  T = 0: User runs simulation                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  T+N ms: Solver finishes                                  │
│          └─ result = SimulationResult(...)                │
│             ├─ result.v_soma computed                     │
│             ├─ result.currents populated [Line 168-193]   │
│             └─ Post-processing complete                   │
│                                                            │
│  T+N+δ: MainWindow receives result                       │
│         └─ analytics.update_analytics(result)            │
│            ├─ _update_passport()                          │
│            ├─ _update_traces()                            │
│            ├─ _update_gates()                             │
│            ├─ _update_currents() ← NEW                   │
│            │  ├─ Extract: result.currents                │
│            │  ├─ Filter: Keep active channels            │
│            │  ├─ Plot: V_soma + I_channels              │
│            │  └─ Render: fig_currents.draw()            │
│            ├─ _update_equil()                            │
│            ├─ _update_phase()                            │
│            ├─ ...                                         │
│            └─ All tabs updated                            │
│                                                            │
│  T+N+2δ: UI refresh                                      │
│          └─ "⚡ Currents" tab shows channel currents    │
│             with professional formatting                  │
│                                                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Dependencies Graph

```
_update_currents()
    │
    ├─ Imports/Uses:
    │  ├─ result.currents     (dict from solver)
    │  ├─ result.v_soma       (1D array)
    │  ├─ result.t            (time vector)
    │  └─ numpy as np         (filtering, max/abs)
    │
    ├─ Constants Used:
    │  └─ CHAN_COLORS         (defined in module)
    │
    ├─ Helper Functions:
    │  └─ _configure_ax_interactive()  (existing helper)
    │
    ├─ Matplotlib API:
    │  ├─ Figure.clear()
    │  ├─ Figure.add_subplot()
    │  ├─ Axes.plot()
    │  ├─ Axes.axhline()
    │  └─ Canvas.draw()
    │
    └─ Module Attributes:
       ├─ self.fig_currents   (Figure)
       └─ self.cvs_currents   (FigureCanvas)
```

---

## Changes at a Glance

```
BEFORE:                          AFTER:
───────────────────────────────────────────────

[🧬 Passport]                    [🧬 Passport]
[📊 Traces]                      [📊 Traces]
[⚙ Gates]                        [⚙ Gates]
[📈 Equilibrium]        →→→→→    [⚡ Currents]  ← NEW
[🔄 Phase Plane]                 [📈 Equilibrium]
...                              [🔄 Phase Plane]
                                 ...

Tab Count: 11              Tab Count: 12
Methods: 11               Methods: 12 (+_update_currents)
Lines: ~800              Lines: ~845 (+45)
```

---

**Diagram created for implementation documentation**  
**Shows complete integration of Channel Currents tab in NeuroModel GUI v10.1**
