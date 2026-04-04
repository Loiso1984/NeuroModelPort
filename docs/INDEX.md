# NeuroModelPort v10.1 — Documentation Index

## Quick Start

```bash
cd NeuroModelPort
python main.py          # Launch GUI
pytest tests/ -v        # Run full test suite
```

## Project Structure

```
NeuroModelPort/
├── main.py                     Entry point (GUI)
├── README.md                   Project overview
├── AIDER_PLAN.md               Refactoring roadmap
│
├── core/                       Simulation engine
│   ├── models.py               FullModelConfig data structures
│   ├── kinetics.py             HH gating variables (m, h, n, r, s, u)
│   ├── channels.py             Ion channel implementations
│   ├── morphology.py           Axial conductance, cable equations
│   ├── rhs.py                  RHS ODE system (Numba @njit)
│   ├── solver.py               BDF solver with sparse Jacobian
│   ├── jacobian.py             Analytic sparse Jacobian
│   ├── presets.py              Scientific presets + synaptic receptors
│   ├── dual_stimulation.py     Dual stimulation module
│   ├── analysis.py             Spike detection, phase analysis, LLE
│   ├── advanced_sim.py         SWEEP, S-D curve, excitability map
│   ├── validation.py           Parameter validation
│   ├── errors.py               Custom exceptions
│   ├── unit_converter.py       Density/absolute unit display
│   └── dendritic_filter.py     Dendritic filtering
│
├── gui/                        PySide6 interface
│   ├── main_window.py          Main window, tabs, controls
│   ├── plots.py                Oscilloscope (voltage, currents)
│   ├── analytics.py            Spike analysis, phase plane, bifurcation
│   ├── topology.py             Neuron morphology visualization
│   ├── axon_biophysics.py      AIS visualization (4 panels)
│   ├── dual_stimulation_widget.py  Dual stim controls
│   ├── dendritic_filter_monitor.py Dendritic filter display
│   ├── locales.py              i18n (RU/EN)
│   └── bilingual_tooltips.py   Bilingual parameter tooltips
│
├── tests/                      Test suite (pytest)
│   ├── core/                   Core module tests
│   ├── presets/                Preset validation tests
│   ├── integration/            Integration tests
│   ├── stress/                 Stress tests
│   ├── validation/             Validation tests
│   ├── utils/                  Calibration & reporting utilities
│   └── archive/                Legacy/debug scripts
│
└── docs/                       Documentation
    ├── INDEX.md                This file
    ├── MASTER_BACKLOG_CONTRACT.md  Canonical project contract
    ├── CURRENT_VALIDATION_TASKS.md
    ├── PHYSIOLOGY_VALIDATION_MEMORY.md
    ├── VALIDATION_COVERAGE_STATUS.md
    ├── reference/              Architecture, guides, references
    └── archive/                Phase completion reports
```

## Documentation

| Document | Purpose |
|----------|---------|
| `MASTER_BACKLOG_CONTRACT.md` | Canonical project contract & task list |
| `CURRENT_VALIDATION_TASKS.md` | Active validation work |
| `PHYSIOLOGY_VALIDATION_MEMORY.md` | Ion channel validation notes |
| `VALIDATION_COVERAGE_STATUS.md` | Test coverage status |
| `reference/ION_CHANNELS_REFERENCE.md` | Channel parameters & literature |
| `reference/LITERATURE_CHANNEL_VALUES.md` | Literature values for validation |
| `reference/NEURON_PRESETS_GUIDE.md` | Preset descriptions & usage |
| `reference/ARCHITECTURE_v10_1.md` | Architecture overview |

## Testing

```bash
# Critical tests
pytest tests/core/test_solver.py -v
pytest tests/presets/test_squid_golden.py -v

# Full suite
pytest tests/ -v

# Specific channel validation
pytest tests/core/test_hcn_*.py -v
pytest tests/core/test_ia_*.py -v
pytest tests/core/test_calcium_channels.py -v
```
