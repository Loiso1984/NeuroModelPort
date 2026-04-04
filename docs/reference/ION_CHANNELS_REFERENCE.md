# Ion Channels Reference Guide
# Справочник по ионным каналам

## 📋 Table of Contents | Содержание

1. [Sodium Channels (Na⁺)](#sodium-channels-na)
2. [Potassium Channels (K⁺)](#potassium-channels-k)
3. [Calcium Channels (Ca²⁺)](#calcium-channels-ca)
4. [HCN Channels (Ih)](#hcn-channels-ih)
5. [A-type Potassium Channels (IA)](#a-type-potassium-channels-ia)
6. [SK Channels (Ca²⁺-activated K⁺)](#sk-channels-ca-activated-k)
7. [Leak Channels (I_leak)](#leak-channels-i_leak)

---

## 🔥 Sodium Channels (Na⁺) | Натриевые каналы

### Physiology | Физиология
- **Function**: Action potential upstroke, rapid depolarization
- **Role**: Initiates and propagates action potentials
- **Kinetics**: Fast activation (τ ≈ 0.1-1 ms), slow inactivation (τ ≈ 1-10 ms)

### Reference Values | Референсные значения
```python
# Standard values for cortical pyramidal neurons
gNa_max = 120.0  # mS/cm² (maximum conductance)
ENa = 50.0       # mV (reversal potential)
V_half = -40.0   # mV (activation midpoint)
k = 7.0          # mV (slope factor)
```

### Preset Examples | Примеры пресетов
```python
# Fast-spiking interneuron
cfg.channels.gNa_max = 120.0  # High for fast spiking

# Motoneuron (Powers 2001)
cfg.channels.gNa_max = 75.0   # Moderate for regular spiking

# Thalamic relay
cfg.channels.gNa_max = 100.0  # Standard for relay neurons
```

### Validation Criteria | Критерии валидации
- **Spike amplitude**: > 80 mV peak
- **Spike width**: 0.5-2.0 ms at half-height
- **Rheobase**: 5-20 µA/cm² stimulation threshold

---

## 💚 Potassium Channels (K⁺) | Калиевые каналы

### Physiology | Физиология
- **Function**: Action potential repolarization, afterhyperpolarization
- **Role**: Returns membrane to resting potential, controls firing frequency
- **Kinetics**: Delayed rectifier, activates during depolarization

### Reference Values | Референсные значения
```python
# Standard delayed rectifier
gK_max = 36.0    # mS/cm² (maximum conductance)
EK = -77.0       # mV (reversal potential)
V_half = -55.0   # mV (activation midpoint)
k = 10.0         # mV (slope factor)
```

### Preset Examples | Примеры пресетов
```python
# Fast-spiking interneuron
cfg.channels.gK_max = 20.0   # Lower for high frequency

# Regular spiking
cfg.channels.gK_max = 36.0   # Standard

# Thalamic relay
cfg.channels.gK_max = 10.0   # Low for burst capability
```

### Validation Criteria | Критерии валидации
- **AHP amplitude**: 5-15 mV below resting
- **Firing frequency adaptation**: Present in regular spiking
- **Repolarization rate**: 10-50 V/s

---

## ⚡ Calcium Channels (Ca²⁺) | Кальциевые каналы

### Physiology | Физиология
- **Function**: Calcium influx, signaling, neurotransmitter release
- **Role**: Secondary depolarization, activates Ca²⁺-dependent processes
- **Kinetics**: L-type, slow activation (τ ≈ 10-100 ms), voltage-dependent

### Reference Values | Референсные значения
```python
# L-type calcium channels (validated)
gCa_max = 0.08   # mS/cm² (physiological range: 0.05-0.1)
ECa = 120.0      # mV (dynamic Nernst calculation)
B_Ca = 0.001     # Current-to-concentration conversion
tau_Ca = 200.0   # ms (clearance time constant)
Ca_rest = 50e-6  # M (resting concentration)
Ca_ext = 2.0     # M (extracellular concentration)
```

### Kinetics | Кинетика
```python
# Destexhe 1993 L-type kinetics
V_half_s = -5.0  # mV (s gate activation)
k_s = 7.2        # mV (slope)
V_half_u = 13.0  # mV (u gate inactivation)
k_u = 50.0       # mV (slope)
```

### Validation Results | Результаты валидации
✅ **E_Ca = 141.6 mV** (physiological range: 120-150 mV)  
✅ **Ca influx = 137 nM/spike** (physiological: 10-1000 nM)  
✅ **B_Ca = 0.001** (physiological range: 0.0005-0.01)  
✅ **All 5 calcium presets validated** (K, L, M, N, O)

### Preset Examples | Примеры пресетов
```python
# Thalamic relay (burst capable)
cfg.channels.gCa_max = 0.08
cfg.calcium.B_Ca = 0.001
cfg.calcium.tau_Ca = 200.0

# Purkinje cell (fast clearance)
cfg.channels.gCa_max = 0.08
cfg.calcium.tau_Ca = 150.0

# Alzheimer's (slow clearance)
cfg.channels.gCa_max = 0.08
cfg.calcium.tau_Ca = 800.0  # Impaired clearance
```

### Validation Criteria | Критерии валидации
- **Ca influx**: 10-1000 nM per spike
- **E_Ca**: 120-150 mV (Nernst calculated)
- **Time constant**: 150-900 ms (clearance)
- **Concentration range**: 50 nM - 100 µM

---

## 🌊 HCN Channels (Ih) | HCN каналы

### Physiology | Физиология
- **Function**: Hyperpolarization-activated cation current, pacemaker activity
- **Role**: Resting potential regulation, input resistance control, rhythmic activity
- **Kinetics**: Activated by hyperpolarization, slow (τ ≈ 100-1000 ms)

### Reference Values | Референсные значения
```python
# Destexhe 1993 HCN parameters (validated)
gIh_max = 0.02-0.03  # mS/cm² (physiological range)
E_Ih = -30.0         # mV (reversal potential)
V_half = -78.0       # mV (activation midpoint)
k = 18.0             # mV (slope factor)
tau_range = 96-1000   # ms (time constants)
```

### Kinetics | Кинетика
```python
# Destexhe 1993 HCN kinetics
ar_Ih(V) = 0.001 * exp(-(V + 78.0) / 18.0)  # Activation rate
br_Ih(V) = 0.001 / (1.0 + exp(-(V + 78.0) / 18.0))  # Deactivation rate
```

### Validation Results | Результаты валидации
✅ **V_half = -78 mV** (physiological: -75 to -85 mV)  
✅ **Time constants**: 96-1000 ms (physiological)  
✅ **2 HCN presets identified** (K: Thalamic, L: CA1)  
✅ **Temperature scaling verified**  

### Preset Examples | Примеры пресетов
```python
# Thalamic relay (burst pacemaker)
cfg.channels.enable_Ih = True
cfg.channels.gIh_max = 0.03
cfg.channels.E_Ih = -30.0

# Hippocampal CA1 (theta rhythm)
cfg.channels.enable_Ih = True
cfg.channels.gIh_max = 0.02
cfg.channels.E_Ih = -30.0
```

### Validation Criteria | Критерии валидации
- **Resting potential**: -70 to -60 mV (depolarized by Ih)
- **Input resistance**: Reduced by 20-50% with Ih active
- **Voltage dependence**: Activated below -60 mV
- **Temperature sensitivity**: Q10 ≈ 2-3

---

## ⚡ A-type Potassium Channels (IA) | A-type калиевые каналы

### Physiology | Физиология
- **Function**: Transient outward current, delays first spike
- **Role**: Spike timing control, dendritic integration, frequency filtering
- **Kinetics**: Fast activation, rapid inactivation (τ ≈ 5-50 ms)

### Reference Values | Референсные значения
```python
# Connor-Stevens A-type parameters
gA_max = 10.0    # mS/cm² (typical range: 5-20)
EA = -77.0       # mV (reversal potential, same as K)
V_half_m = -40.0 # mV (activation)
V_half_h = -60.0 # mV (inactivation)
```

### Kinetics | Кинетика
```python
# Connor-Stevens kinetics
am_A(V) = 0.02 * (V + 13.0) / (1.0 - exp(-(V + 13.0) / 10.0))
bm_A(V) = 0.0175 * exp(-(V + 65.0) / 20.0)
ah_A(V) = 0.0016 * exp(-(V + 57.0) / 40.0)
bh_A(V) = 0.05 / (1.0 + exp(-(V + 57.0) / 10.0))
```

### Preset Examples | Примеры пресетов
```python
# Pyramidal neuron (delay)
cfg.channels.enable_IA = True
cfg.channels.gA_max = 10.0

# Fast-spiking (minimal A-current)
cfg.channels.enable_IA = False
```

### Validation Criteria | Критерии валидации
- **First spike delay**: 5-50 ms with IA active
- **Frequency adaptation**: Enhanced by IA
- **Dendritic filtering**: IA blocks backpropagation

---

## 🧪 SK Channels (Ca²⁺-activated K⁺) | SK каналы

### Physiology | Физиология
- **Function**: Calcium-activated potassium current, afterhyperpolarization
- **Role**: Spike frequency adaptation, burst termination
- **Kinetics**: Activated by intracellular Ca²⁺, no voltage dependence

### Reference Values | Референсные значения
```python
# SK channel parameters
gSK_max = 2.0    # mS/cm² (typical range: 0.5-5.0)
ESK = -77.0      # mV (reversal potential)
Ca_sensitivity = 0.1-1.0  # µM (activation threshold)
```

### Preset Examples | Примеры пресетов
```python
# Alzheimer's (enhanced adaptation)
cfg.channels.enable_SK = True
cfg.channels.gSK_max = 1.5

# Regular spiking
cfg.channels.enable_SK = False
```

### Validation Criteria | Критерии валидации
- **Adaptation index**: 0.3-0.8 with SK active
- **Burst termination**: SK stops prolonged firing
- **Calcium dependence**: Requires Ca²⁺ influx

---

## 🔌 Leak Channels (I_leak) | Токи утечки

### Physiology | Физиология
- **Function**: Non-specific background conductance
- **Role**: Sets resting potential, input resistance
- **Kinetics**: Voltage-independent, constant

### Reference Values | Референсные значения
```python
# Standard leak parameters
gL = 0.3         # mS/cm² (typical range: 0.1-1.0)
EL = -54.387     # mV (reversal potential)
```

### Preset Examples | Примеры пресетов
```python
# Standard leak
cfg.channels.gL = 0.3
cfg.channels.EL = -54.387

# High resistance (sensitive)
cfg.channels.gL = 0.1

# Low resistance (stable)
cfg.channels.gL = 0.5
```

### Validation Criteria | Критерии валидации
- **Resting potential**: -70 to -50 mV
- **Input resistance**: 50-500 MΩ
- **Stability**: No spontaneous oscillations

---

## 📊 Channel Interaction Matrix | Матрица взаимодействий каналов

| Channel Pair | Interaction | Effect | Physiological Role |
|--------------|-------------|--------|-------------------|
| Na⁺ + K⁺ | Antagonistic | Spike shape | Action potential generation |
| Ca²⁺ + SK | Synergistic | Adaptation | Frequency control |
| Ih + Na⁺ | Synergistic | Excitability | Pacemaker activity |
| IA + Na⁺ | Antagonistic | Spike delay | Timing control |
| Ca²⁺ + Ih | Complex | Burst mode | Thalamic rhythms |

---

## 🔬 Validation Protocols | Протоколы валидации

### Single Channel Tests | Тесты отдельных каналов
1. **Voltage clamp**: I-V curves, reversal potentials
2. **Kinetics**: Time constants, activation curves
3. **Pharmacology**: Channel blockers, agonists
4. **Temperature**: Q10 coefficients

### Multi-channel Tests | Мультиканальные тесты
1. **Interaction studies**: Combined channel effects
2. **Stress testing**: Extreme parameter ranges
3. **Stability**: Numerical integration issues
4. **Physiological validation**: Comparison to experimental data

### Preset Validation | Валидация пресетов
1. **Firing patterns**: Regular, bursting, fast-spiking
2. **Frequency ranges**: Theta, gamma, beta rhythms
3. **Pathological models**: Epilepsy, Alzheimer's, hypoxia
4. **Species-specific**: Rodent vs human parameters

---

## 📚 References | Литература

1. **Destexhe, A., et al. (1993)** - "A model of thalamic relay neurons"
2. **Hodgkin, A.L., Huxley, A.F. (1952)** - "A quantitative description of membrane current"
3. **Connor, J.A., Stevens, C.F. (1971)** - "Predictions of repetitive firing from recovery"
4. **Mainen, Z.F., Sejnowski, T.J. (1996)** - "Influence of dendritic structure"
5. **Wang, X.J., Buzsáki, G. (1996)** - "Gamma oscillation by synaptic inhibition"

---

## 🚀 Quick Reference Card | Быстрая справка

```python
# Physiological parameter ranges
gNa_max: 50-200 mS/cm²
gK_max: 10-50 mS/cm²
gCa_max: 0.05-0.1 mS/cm²
gIh_max: 0.01-0.05 mS/cm²
gA_max: 5-20 mS/cm²
gSK_max: 0.5-5.0 mS/cm²
gL: 0.1-1.0 mS/cm²

# Reversal potentials
ENa: 50-60 mV
EK: -77 to -85 mV
ECa: 120-150 mV (Nernst)
E_Ih: -20 to -40 mV
EL: -50 to -60 mV

# Time constants
tau_Na: 0.1-10 ms
tau_K: 1-20 ms
tau_Ca: 150-900 ms
tau_Ih: 100-1000 ms
tau_A: 5-50 ms
```

---

*Last updated: April 2026*  
*Version: NeuroModelPort v10.1*
