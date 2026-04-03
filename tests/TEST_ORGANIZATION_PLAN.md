# Test Organization Plan

## Current Issues

### 1. Debug Artifacts in Root Directory
Files to move to `tests/archive/` or `debug/`:
- debug_b_ca.py
- debug_calcium.py
- debug_current_balance.py
- debug_current_direction.py
- debug_hcn_current.py
- debug_high_rates.py
- debug_preset.py

### 2. Scattered Test Files in Root
Files to organize:
- test_all_calcium_presets.py → tests/validation/
- test_axon_widget.py → tests/integration/
- test_comprehensive_integration.py → tests/integration/
- test_dendritic_filter.py → tests/core/
- test_dual_stim*.py → tests/integration/
- test_hcn_*.py → tests/branches/ or tests/core/
- test_new_analytics.py → tests/core/
- test_preset_dynamics.py → tests/validation/
- test_presets_calcium.py → tests/validation/
- test_rhs_inputs.py → tests/core/
- test_secondary_*.py → tests/core/
- test_strong_secondary.py → tests/core/

### 3. Test Categories

```
tests/
├── branches/          # Isolated channel validation (process: test before main files)
│   ├── test_hcn_isolated_branch.py    # TRUE isolated HCN (no Na/K)
│   ├── test_ia_isolated_branch.py     # TRUE isolated IA + Na/K baseline
│   ├── test_ica_isolated_branch.py    # Isolated Ca channels
│   ├── test_sk_isolated_branch.py     # Isolated SK channels
│   └── test_spike_detection_branch.py # Spike detection correctness
│
├── core/              # Core functionality tests
│   ├── test_solver.py
│   ├── test_analysis.py
│   ├── test_kinetics.py
│   └── test_morphology.py
│
├── integration/       # Multi-component integration tests
│   ├── test_dual_stimulation.py
│   ├── test_synaptic_integration.py
│   └── test_gui_integration.py
│
├── presets/           # Preset validation tests
│   ├── test_preset_dynamics.py
│   ├── test_preset_combinations.py
│   └── test_preset_channels.py
│
├── stress/            # Stress and load tests
│   ├── test_multichannel_interactions.py
│   ├── test_parameter_sweeps.py
│   └── test_long_duration.py
│
├── validation/        # Physiological validation tests
│   ├── test_hcn_physics.py
│   ├── test_ica_physics.py
│   ├── test_ia_physics.py
│   └── test_literature_compliance.py
│
└── archive/           # Deprecated/outdated tests
    └── (old test files)
```

## Process Fix Required

### BEFORE modifying core files:
1. Create test branch in `tests/branches/`
2. Test in isolation
3. Validate against literature values
4. Only THEN modify core files

### Current Violations:
- [X] presets.py modified without branch testing
- [X] models.py modified without branch testing  
- [X] kinetics.py modified without branch testing

## Spike Detection Verification Needed

Current algorithm: threshold crossing + local maxima
Problems:
- No baseline crossing verification
- No repolarization check
- May detect sustained depolarization as "spike"

Required test: Synthetic data with known spikes

## Solo Validation Needed

### HCN (Ih) Solo Test:
```python
# TRUE isolated configuration
cfg.channels.gNa_max = 0.0   # No Na
cfg.channels.gK_max = 0.0    # No K
cfg.channels.gL = 0.01       # Minimal leak
cfg.channels.gIh_max = 0.5  # Test conductance
cfg.channels.enable_Ih = True
# All other channels OFF
```

Tests:
1. V_½ activation (-78 mV Destexhe 1993)
2. Input resistance
3. Temperature scaling
4. Resting stability

### IA (A-type) Solo Test:
Baseline Na/K required for spiking, but IA tested in isolation:
- Activation curve (-40 mV)
- Inactivation curve (-60 mV)
- Spike delay effect
- Recovery from inactivation

## Action Plan

1. [ ] Move debug artifacts to archive/
2. [ ] Organize scattered tests into categories
3. [ ] Create proper isolated test branches
4. [ ] Run branch tests BEFORE any core file changes
5. [ ] Create comprehensive preset validation suite
6. [ ] Create multi-channel stress tests

## Notes

- Tests currently hang - need investigation
- May need timeout mechanisms in test framework
- Consider parallel test execution for speed
