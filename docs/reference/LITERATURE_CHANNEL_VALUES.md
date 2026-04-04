# Literature-Based Channel Parameter Reference

**Purpose:** Quick reference for biophysically accurate channel parameters  
**Sources:** Peer-reviewed neuroscience literature  
**Last Updated:** March 31, 2026

## 📚 Channel Parameter Summary

| Neuron Type | gNa_max (mS/cm²) | gK_max (mS/cm²) | gL (mS/cm²) | Cm (µF/cm²) | Source |
|-------------|------------------|------------------|-------------|-------------|---------|
| **L5 Pyramidal** | 120-150 | 30-50 | 0.05-0.1 | 1.0 | Mainen & Sejnowski 1996 |
| **FS Interneuron** | 80-120 | 20-40 | 0.1-0.2 | 1.0 | Wang & Buzsáki 1996 |
| **Purkinje** | 40-60 | 15-25 | 0.05-0.1 | 1.0 | De Schutter & Bower 1994 |
| **Thalamic Relay** | 60-100 | 20-35 | 0.08-0.12 | 1.0 | McCormick & Huguenard 1992 |

## 🔬 Detailed Literature Sources

### **L5 Pyramidal Neurons**
**Paper:** Mainen & Sejnowski (1996) *J. Neurosci.* 16(16): 5073-5087  
**Key Findings:**
- **Soma diameter:** 20-25 µm
- **Input resistance:** 100-150 MΩ
- **Membrane time constant:** 15-25 ms
- **Firing pattern:** Regular spiking, 10-15 Hz typical

**Recommended Parameters:**
```python
gNa_max = 130.0  # mS/cm² (mid-range of literature)
gK_max = 40.0    # mS/cm² (mid-range of literature)
gL = 0.08        # mS/cm² (low leak for excitability)
Cm = 1.0         # µF/cm² (standard)
```

### **Fast-Spiking (FS) Interneurons**
**Paper:** Wang & Buzsáki (1996) *J. Neurosci.* 16(20): 6402-6413  
**Key Findings:**
- **Soma diameter:** 12-16 µm
- **Input resistance:** 50-100 MΩ
- **Membrane time constant:** 5-10 ms
- **Firing pattern:** Fast-spiking, 80-200 Hz typical
- **Special channels:** Kv3.1b for high-frequency firing

**Recommended Parameters:**
```python
gNa_max = 120.0  # mS/cm² (high for fast spiking)
gK_max = 40.0    # mS/cm² (includes Kv3-like properties)
gL = 0.15        # mS/cm² (higher leak for stability)
Cm = 1.0         # µF/cm² (standard)
```

### **Cerebellar Purkinje Cells**
**Paper:** De Schutter & Bower (1994) *J. Neurophysiol.* 72(4): 1609-1629  
**Key Findings:**
- **Soma diameter:** 25-30 µm
- **Input resistance:** 20-50 MΩ
- **Membrane time constant:** 15-30 ms
- **Firing pattern:** Simple spikes 30-100 Hz typical
- **Special channels:** Calcium channels, HCN channels

**Recommended Parameters:**
```python
gNa_max = 50.0   # mS/cm² (moderate sodium)
gK_max = 22.0    # mS/cm² (balanced potassium)
gL = 0.08        # mS/cm² (low leak)
Cm = 1.0         # µF/cm² (standard)
gCa_max = 0.5    # mS/cm² (calcium channels)
gIh_max = 1.5    # mS/cm² (HCN channels)
```

### **Thalamic Relay Neurons**
**Paper:** McCormick & Huguenard (1992) *J. Physiol.* 453: 321-353  
**Key Findings:**
- **Soma diameter:** 20-25 µm
- **Input resistance:** 30-80 MΩ
- **Membrane time constant:** 10-20 ms
- **Firing pattern:** Relay mode 5-40 Hz, burst mode up to 200 Hz
- **Special channels:** HCN channels (Ih), T-type calcium

**Recommended Parameters:**
```python
gNa_max = 80.0   # mS/cm² (moderate sodium)
gK_max = 25.0    # mS/cm² (balanced potassium)
gL = 0.10        # mS/cm² (standard leak)
Cm = 1.0         # µF/cm² (standard)
gIh_max = 1.0    # mS/cm² (HCN channels)
gCa_max = 0.3    # mS/cm² (T-type calcium)
```

## 🎯 Current Injection (Iext) Guidelines

### **Literature-Based Current Ranges**
| Neuron Type | Soma Iext (µA/cm²) | AIS Iext (µA/cm²) | Dendritic Iext (µA/cm²) | Expected Frequency |
|-------------|-------------------|------------------|------------------------|-------------------|
| **L5 Pyramidal** | 10-30 | 20-80 | 20-60 | 10-20 Hz |
| **FS Interneuron** | 20-100 | 50-200 | 30-80 | 50-200 Hz |
| **Purkinje** | 10-40 | 50-250 | 20-60 | 30-100 Hz |
| **Thalamic** | 10-30 | 50-200 | 20-50 | 5-40 Hz |

### **Current Scaling Notes**
- **Higher gNa/gK requires higher Iext** for same effect
- **AIS stimulation needs 2-5× higher current** than soma
- **Dendritic filtering reduces effective current** by factor e^(-distance/λ)
- **Temperature compensation (phi)** affects current requirements

## 🔧 Implementation Tips

### **Temperature Compensation**
```python
# Q10 temperature scaling
phi = 3.0 ** ((temperature - 6.3) / 10.0)  # Standard Q10 formula
# Apply to all rate constants
```

### **Morphology Considerations**
- **Surface area scaling:** Larger neurons need more current
- **Compartment coupling:** Poor coupling requires higher AIS current
- **Dendritic filtering:** Distance-dependent attenuation

### **Validation Checklist**
- [ ] Parameters within literature ranges
- [ ] Firing frequencies reasonable for neuron type
- [ ] Input resistance matches literature
- [ ] Membrane time constant appropriate
- [ ] AIS responds to stimulation

## 📊 Frequency Validation

### **Expected Firing Ranges**
| Neuron Type | Resting | Typical | Maximum |
|-------------|---------|---------|---------|
| **L5 Pyramidal** | 0 Hz | 10-15 Hz | 50-100 Hz |
| **FS Interneuron** | 0 Hz | 80-150 Hz | 200-500 Hz |
| **Purkinje** | 30-50 Hz | 37±15 Hz | 200 Hz |
| **Thalamic** | 0-5 Hz | 10-25 Hz | 100-200 Hz |

### **Troubleshooting**
- **0 Hz:** Check Iext, gNa_max, or temperature
- **Too high frequency:** Reduce Iext or increase gK_max
- **Too low frequency:** Increase Iext or gNa_max
- **Irregular firing:** Check noise parameters or channel kinetics

## 🚀 Quick Reference

### **Standard Parameters (Safe Starting Point)**
```python
# L5 Pyramidal
gNa_max, gK_max, gL, Cm = 130.0, 40.0, 0.08, 1.0

# FS Interneuron  
gNa_max, gK_max, gL, Cm = 120.0, 40.0, 0.15, 1.0

# Purkinje
gNa_max, gK_max, gL, Cm = 50.0, 22.0, 0.08, 1.0

# Thalamic
gNa_max, gK_max, gL, Cm = 80.0, 25.0, 0.10, 1.0
```

### **Standard Current Injection**
```python
# Starting points (adjust as needed)
iext_soma = {'L5': 25, 'FS': 50, 'Purkinje': 20, 'Thalamic': 20}
iext_ais = {'L5': 50, 'FS': 100, 'Purkinje': 200, 'Thalamic': 200}
```

---

**This reference provides the foundation for biophysically accurate neuron modeling in NeuroModelPort. Always validate against literature when implementing new neuron types.**
