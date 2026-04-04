# Neuron Presets Guide - NeuroModelPort v10.1

**Purpose:** Complete guide to available neuron presets and their configurations  
**Last Updated:** March 31, 2026

## 🧠 Available Neuron Presets

### **Cortical Pyramidal Neurons**

#### **B: Pyramidal L5 (Mainen 1996)**
**Literature:** Mainen & Sejnowski 1996, J. Neurosci.  
**Characteristics:** Regular spiking, thick-tufted pyramidal cells

```python
# Configuration
preset_name = "B: Pyramidal L5 (Mainen 1996)"

# Key Parameters
gNa_max = 130.0  # mS/cm²
gK_max = 40.0    # mS/cm²
gL = 0.08        # mS/cm²
Cm = 1.0         # µF/cm²

# Morphology
d_soma = 20.0    # µm
N_ais = 3        # AIS compartments
single_comp = False

# Expected Behavior
firing_frequency = 10-20 Hz  # Regular spiking
input_resistance = 100-150 MΩ
membrane_tau = 15-25 ms
```

**Stimulation Guidelines:**
- **Soma Iext:** 10-30 µA/cm² (typical: 25)
- **AIS Iext:** 20-80 µA/cm² (typical: 50)
- **Dendritic Iext:** 20-60 µA/cm² (typical: 50)

---

### **Cortical Interneurons**

#### **C: FS Interneuron (Wang-Buzsaki 1996)**
**Literature:** Wang & Buzsáki 1996, J. Neurosci.  
**Characteristics:** Fast-spiking parvalbumin-positive interneurons

```python
# Configuration
preset_name = "C: FS Interneuron (Wang-Buzsaki 1996)"

# Key Parameters
gNa_max = 120.0  # mS/cm² (high for fast spiking)
gK_max = 40.0    # mS/cm² (includes Kv3-like properties)
gL = 0.15        # mS/cm² (higher leak)
Cm = 1.0         # µF/cm²

# Morphology
d_soma = 15.0    # µm
N_ais = 2        # AIS compartments
single_comp = False

# Expected Behavior
firing_frequency = 80-200 Hz  # Fast-spiking
input_resistance = 50-100 MΩ
membrane_tau = 5-10 ms
```

**Stimulation Guidelines:**
- **Soma Iext:** 20-100 µA/cm² (typical: 50)
- **AIS Iext:** 50-200 µA/cm² (typical: 100)
- **Dendritic Iext:** 30-80 µA/cm² (typical: 30)

---

### **Cerebellar Neurons**

#### **E: Cerebellar Purkinje (De Schutter 1994)**
**Literature:** De Schutter & Bower 1994, J. Neurophysiol.  
**Characteristics:** Large GABAergic output neurons with extensive dendrites

```python
# Configuration
preset_name = "E: Cerebellar Purkinje (De Schutter 1994)"

# Key Parameters
gNa_max = 50.0   # mS/cm² (moderate sodium)
gK_max = 22.0    # mS/cm² (balanced potassium)
gL = 0.08        # mS/cm² (low leak)
Cm = 1.0         # µF/cm²

# Additional Channels
gCa_max = 0.5    # mS/cm² (calcium channels)
gIh_max = 1.5    # mS/cm² (HCN channels)
enable_ICa = True
enable_Ih = True

# Morphology
d_soma = 25.0    # µm
N_ais = 3        # AIS compartments
single_comp = False

# Expected Behavior
firing_frequency = 30-100 Hz  # Simple spikes
input_resistance = 20-50 MΩ
membrane_tau = 15-30 ms
```

**Stimulation Guidelines:**
- **Soma Iext:** 10-40 µA/cm² (typical: 20)
- **AIS Iext:** 50-250 µA/cm² (typical: 200)
- **Dendritic Iext:** 20-60 µA/cm² (typical: 40)

---

### **Thalamic Neurons**

#### **K: Thalamic Relay (McCormick & Huguenard 1992)**
**Literature:** McCormick & Huguenard 1992, J. Physiol.  
**Characteristics:** Thalamocortical relay neurons with burst capability

```python
# Configuration
preset_name = "K: Thalamic Relay (McCormick & Huguenard 1992)"

# Key Parameters
gNa_max = 80.0   # mS/cm² (moderate sodium)
gK_max = 25.0    # mS/cm² (balanced potassium)
gL = 0.10        # mS/cm² (standard leak)
Cm = 1.0         # µF/cm²

# Additional Channels
gIh_max = 1.0    # mS/cm² (HCN channels)
gCa_max = 0.3    # mS/cm² (T-type calcium)
enable_Ih = True
enable_ICa = True

# Morphology
d_soma = 25.0    # µm
N_ais = 2        # AIS compartments
single_comp = False

# Expected Behavior
firing_frequency = 5-40 Hz  # Relay mode
burst_frequency = 100-200 Hz  # Burst mode
input_resistance = 30-80 MΩ
membrane_tau = 10-20 ms
```

**Stimulation Guidelines:**
- **Soma Iext:** 10-30 µA/cm² (typical: 20)
- **AIS Iext:** 50-200 µA/cm² (typical: 200)
- **Dendritic Iext:** 20-50 µA/cm² (typical: 30)

---

## 🔧 Preset Usage Guide

### **Basic Usage**
```python
from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver

# Create configuration
cfg = FullModelConfig()

# Apply preset
apply_preset(cfg, "B: Pyramidal L5 (Mainen 1996)")

# Modify parameters if needed
cfg.stim.Iext = 25.0
cfg.stim_location.location = "soma"

# Run simulation
solver = NeuronSolver(cfg)
result = solver.run_single()
```

### **Custom Parameter Modification**
```python
# Apply preset first
apply_preset(cfg, "C: FS Interneuron (Wang-Buzsaki 1996)")

# Then modify specific parameters
cfg.channels.gNa_max = 140.0  # Increase excitability
cfg.channels.gK_max = 45.0    # Adjust repolarization
cfg.stim.Iext = 75.0         # Adjust stimulation

# Add custom channels
cfg.channels.enable_Ih = True
cfg.channels.gIh_max = 0.5
```

### **AIS Configuration**
```python
# Ensure AIS is properly configured
if cfg.morphology.single_comp and cfg.morphology.N_ais > 0:
    cfg.morphology.single_comp = False

# AIS multipliers (applied automatically)
print(f"AIS gNa multiplier: {cfg.morphology.gNa_ais_mult}×")
print(f"AIS gK multiplier: {cfg.morphology.gK_ais_mult}×")
```

---

## 📊 Preset Comparison

### **Parameter Comparison Table**
| Preset | gNa_max | gK_max | gL | Cm | Soma Size | N_ais | Typical Firing |
|--------|---------|--------|----|----|-----------|-------|----------------|
| **L5 Pyramidal** | 130 | 40 | 0.08 | 1.0 | 20 µm | 3 | 10-20 Hz |
| **FS Interneuron** | 120 | 40 | 0.15 | 1.0 | 15 µm | 2 | 80-200 Hz |
| **Purkinje** | 50 | 22 | 0.08 | 1.0 | 25 µm | 3 | 30-100 Hz |
| **Thalamic** | 80 | 25 | 0.10 | 1.0 | 25 µm | 2 | 5-40 Hz |

### **Functional Properties**
| Property | L5 Pyramidal | FS Interneuron | Purkinje | Thalamic |
|----------|--------------|---------------|----------|----------|
| **Firing Type** | Regular spiking | Fast-spiking | Simple spikes | Relay/Burst |
| **Input R** | 100-150 MΩ | 50-100 MΩ | 20-50 MΩ | 30-80 MΩ |
| **Time Constant** | 15-25 ms | 5-10 ms | 15-30 ms | 10-20 ms |
| **Special Channels** | None | Kv3-like | Ca²⁺, HCN | HCN, T-type Ca²⁺ |
| **AIS Length** | 60 µm | 40 µm | 60 µm | 40 µm |

---

## 🎯 Choosing the Right Preset

### **For Regular Spiking Cortical Neurons**
→ **B: Pyramidal L5 (Mainen 1996)**
- Use for: Layer 5 pyramidal cells, regular spiking behavior
- Applications: Cortical circuits, feedforward inhibition

### **For Fast Inhibition**
→ **C: FS Interneuron (Wang-Buzsaki 1996)**
- Use for: Fast-spiking interneurons, gamma oscillations
- Applications: Inhibitory networks, fast timing

### **For Cerebellar Processing**
→ **E: Cerebellar Purkinje (De Schutter 1994)**
- Use for: Cerebellar output, motor coordination
- Applications: Cerebellar circuitry, learning

### **For Thalamic Relay**
→ **K: Thalamic Relay (McCormick & Huguenard 1992)**
- Use for: Thalamocortical relay, sleep rhythms
- Applications: Thalamic circuits, oscillations

---

## 🔍 Troubleshooting Presets

### **Common Issues and Solutions**

#### **Problem: Neuron not firing (0 Hz)**
**Causes:**
- Iext too low
- gNa_max too low
- Temperature compensation off

**Solutions:**
```python
# Increase current
cfg.stim.Iext *= 2.0

# Increase sodium conductance
cfg.channels.gNa_max *= 1.5

# Check temperature
cfg.env.temperature = 36.0  # Set to physiological temperature
```

#### **Problem: Firing too fast**
**Causes:**
- Iext too high
- gK_max too low
- gL too low

**Solutions:**
```python
# Decrease current
cfg.stim.Iext *= 0.5

# Increase potassium conductance
cfg.channels.gK_max *= 1.5

# Increase leak conductance
cfg.channels.gL *= 1.2
```

#### **Problem: AIS not responding**
**Causes:**
- single_comp=True conflict
- AIS current too low
- Compartment coupling issues

**Solutions:**
```python
# Fix single compartment issue
if cfg.morphology.single_comp and cfg.morphology.N_ais > 0:
    cfg.morphology.single_comp = False

# Increase AIS current
cfg.stim.Iext *= 3.0  # AIS needs higher current
```

---

## 🚀 Advanced Usage

### **Creating Custom Presets**
```python
def create_custom_preset(cfg, base_preset, modifications):
    """Create custom preset based on existing one"""
    apply_preset(cfg, base_preset)
    
    for param, value in modifications.items():
        if hasattr(cfg.channels, param):
            setattr(cfg.channels, param, value)
        elif hasattr(cfg.morphology, param):
            setattr(cfg.morphology, param, value)
        elif hasattr(cfg.stim, param):
            setattr(cfg.stim, param, value)
    
    return cfg

# Usage
cfg = FullModelConfig()
cfg = create_custom_preset(cfg, "B: Pyramidal L5 (Mainen 1996)", {
    'gNa_max': 150.0,
    'gK_max': 45.0,
    'Iext': 30.0
})
```

### **Multi-Preset Comparison**
```python
presets = [
    "B: Pyramidal L5 (Mainen 1996)",
    "C: FS Interneuron (Wang-Buzsaki 1996)",
    "E: Cerebellar Purkinje (De Schutter 1994)",
    "K: Thalamic Relay (McCormick & Huguenard 1992)"
]

results = {}
for preset in presets:
    cfg = FullModelConfig()
    apply_preset(cfg, preset)
    cfg.stim.Iext = 20.0
    cfg.stim_location.location = "soma"
    
    solver = NeuronSolver(cfg)
    result = solver.run_single()
    results[preset] = result
```

---

## 📚 References

1. **Mainen & Sejnowski (1996)** - "Influence of dendritic structure on firing patterns of neocortical neurons"
2. **Wang & Buzsáki (1996)** - "Gamma oscillation by synaptic inhibition in a hippocampal interneuronal network"
3. **De Schutter & Bower (1994)** - "An active membrane model of the cerebellar Purkinje cell"
4. **McCormick & Huguenard (1992)** - "A model of the electrophysiological properties of thalamocortical relay neurons"

---

**This guide provides comprehensive information for using and customizing neuron presets in NeuroModelPort. Always refer to the literature sources for detailed implementation details.**
