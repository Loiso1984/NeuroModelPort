# NeuroModelPort v10.1 - Документация | Documentation

## 📚 Содержание | Table of Contents

- [🇷🇺 Русская документация](#-русская-документация)
- [🇺🇸 English Documentation](#english-documentation)

---

# 🇷🇺 Русская документация

## 🧠 Обзор | Overview

NeuroModelPort - это научно точная фреймворк для моделирования одиночного нейрона на основе модели Ходжкина-Хаксли (1952) с современными расширениями для вычислительной нейронауки.

**Особенности v10.1:**
- ✅ **Дендритная фильтрация** - Физиологически реалистичные синаптические входы
- ✅ **Многокомпартментная стимуляция** - Сома, AIS и дендритные локации
- ✅ **Параметры на основе литературы** - Валидированы против экспериментальных данных
- ✅ **Полная валидация Phase 6** - Все типы нейронов функциональны
- ✅ **Стимуляция AIS работает** - Функциональность аксонного начального сегмента

## 🎯 Ключевые особенности | Key Features

### **Валидированные модели нейронов (4 основных типа)**

| Тип нейрона | Литература | Размер сомы | Режим firing | Статус |
|-------------|------------|-----------|----------------|---------|
| **L5 Пирамидный** | Mainen & Sejnowski 1996 | 20 мкм | Регулярный спайкинг (10-20 Гц) | ✅ Валидирован |
| **Быстрый интернейрон** | Wang & Buzsáki 1996 | 15 мкм | Быстрый спайкинг (80-200 Гц) | ✅ Валидирован |
| **Клетка Пуркинье** | De Schutter & Bower 1994 | 25 мкм | Простые спайки (30-100 Гц) | ✅ Валидирован |
| **Таламическое реле** | McCormick & Huguenard 1992 | 25 мкм | Реле/Пачки (5-200 Гц) | ✅ Валидирован |

### **Режимы стимуляции**

#### **1. Инъекция в сомы (Стандарт)**
```python
cfg.stim_location.location = 'soma'
cfg.stim.Iext = 25.0  # мкА/см²
# Прямая соматическая инъекция тока
```

#### **2. Инъекция в AIS (Высокая чувствительность)**
```python
cfg.stim_location.location = 'ais'
cfg.stim.Iext = 100.0  # мкА/см² (выше из-за размера компартмента)
# Стимуляция аксонного начального сегмента
```

#### **3. Дендритная фильтрованная (Физиологическая)**
```python
cfg.stim_location.location = 'dendritic_filtered'
cfg.stim.Iext = 50.0  # мкА/см²
# Физиологически реалистичная дендритная стимуляция
```

## 📖 Руководство пользователя | User Guide

### 🚀 Быстрый старт | Quick Start

1. **Запуск GUI:**
```bash
cd c:\NeuroModelPort
python main.py
```

2. **Выбор пресета:**
   - Выберите из выпадающего списка (например, "B: Pyramidal L5")
   - Параметры автоматически применятся

3. **Запуск симуляции:**
   - Нажмите "▶ ЗАПУСТИТЬ СИМУЛЯЦИЮ"
   - Наблюдайте спайкинг на осциллографе

### 🎛️ Параметры модели | Model Parameters

#### **Морфология | Morphology**
- `d_soma`: Диаметр сомы (см) - влияет на ёмкость
- `N_ais`: Сегменты AIS - зона инициации спайка
- `gNa_ais_mult`: Множитель gNa в AIS (40-100×)
- `Ra`: Аксиальное сопротивление (Ом·см)

#### **Каналы | Channels**
- `gNa_max`: Максимальная проводимость Na⁺ (мСм/см²)
- `gK_max`: Максимальная проводимость K⁺ (мСм/см²)
- `Cm`: Ёмкость мембраны (мкФ/см²)
- `ENa, EK, EL`: Потенциалы реверсии (мВ)

#### **Среда | Environment**
- `T_celsius`: Температура эксперимента (°C)
- `Q10`: Температурный коэффициент

#### **Стимуляция | Stimulation**
- `stim_type`: Тип стимула (const/pulse/alpha)
- `Iext`: Амплитуда стимула (мкА/см²)
- `stim_comp`: Компартмент стимуляции

## 🔬 Научные пресеты | Scientific Presets

### **Физиологические нейроны | Physiological Neurons**

#### **A: Гигантский аксон кальмара (HH 1952)**
- **Температура:** 6.3°C (холоднокровный)
- **Проводимости:** gNa=120, gK=36, gL=0.3 мСм/см²
- **Применение:** Классическая модель Ходжкина-Хаксли

#### **B: Пирамидный L5 (Mainen 1996)**
- **Температура:** 37°C, T_ref=23°C, Q10=2.3
- **Проводимости:** gNa=56, gK=6, gL=0.02 мСм/см²
- **Морфология:** Сома 20 мкм + AIS (40× gNa)

#### **C: Быстрый интернейрон (Wang-Buzsaki)**
- **Температура:** 37°C
- **Проводимости:** gNa=120, gK=36, gL=0.1 мСм/см²
- **Режим:** Быстрый спайкинг 150-400 Гц

### **Патологические модели | Pathological Models**

#### **F: Рассеянный склероз (Демиелинизация)**
- **Изменения:** Ra: 70→300 Ом·см, gL: 0.3→1.2
- **Эффект:** Снижение амплитуды спайка

#### **M: Эпилепсия (SCN1A мутация)**
- **Изменения:** gNa: 120→200 мСм/см²
- **Эффект:** Гипервозбудимость

---

# 🇺🇸 English Documentation

## 🧠 Overview

NeuroModelPort is a scientifically accurate single-neuron simulation framework based on the Hodgkin-Huxley (1952) model with modern extensions for computational neuroscience research.

**v10.1 Features:**
- ✅ **Dendritic Filtering** - Physiologically realistic synaptic inputs
- ✅ **Multi-Compartment Stimulation** - Soma, AIS, and dendritic locations
- ✅ **Literature-Based Parameters** - Validated against experimental data
- ✅ **Phase 6 Complete Validation** - All neuron types functional
- ✅ **AIS Stimulation Working** - Axon Initial Segment functionality

## 🎯 Key Features

### **Validated Neuron Models (4 Core Types)**

| Neuron Type | Literature | Soma Size | Firing Pattern | Status |
|-------------|------------|-----------|----------------|---------|
| **L5 Pyramidal** | Mainen & Sejnowski 1996 | 20 µm | Regular spiking (10-20 Hz) | ✅ Validated |
| **FS Interneuron** | Wang & Buzsáki 1996 | 15 µm | Fast-spiking (80-200 Hz) | ✅ Validated |
| **Purkinje Cell** | De Schutter & Bower 1994 | 25 µm | Simple spikes (30-100 Hz) | ✅ Validated |
| **Thalamic Relay** | McCormick & Huguenard 1992 | 25 µm | Relay/Burst (5-200 Hz) | ✅ Validated |

### **Stimulation Modes**

#### **1. Soma Injection (Standard)**
```python
cfg.stim_location.location = 'soma'
cfg.stim.Iext = 25.0  # µA/cm²
# Direct somatic current injection
```

#### **2. AIS Injection (High Sensitivity)**
```python
cfg.stim_location.location = 'ais'
cfg.stim.Iext = 100.0  # µA/cm² (higher due to compartment size)
# Axon Initial Segment stimulation
```

#### **3. Dendritic Filtered (Physiological)**
```python
cfg.stim_location.location = 'dendritic_filtered'
cfg.stim.Iext = 50.0  # µA/cm²
# Physiologically realistic dendritic stimulation
```

## 📖 User Guide

### 🚀 Quick Start

1. **Launch GUI:**
```bash
cd c:\NeuroModelPort
python main.py
```

2. **Select Preset:**
   - Choose from dropdown (e.g., "B: Pyramidal L5")
   - Parameters automatically applied

3. **Run Simulation:**
   - Press "▶ RUN SIMULATION"
   - Observe spiking on oscilloscope

### 🎛️ Model Parameters

#### **Morphology**
- `d_soma`: Soma diameter (cm) - affects capacitance
- `N_ais`: AIS segments - spike initiation zone
- `gNa_ais_mult`: gNa multiplier in AIS (40-100×)
- `Ra`: Axial resistance (Ω·cm)

#### **Channels**
- `gNa_max`: Max Na⁺ conductance (mS/cm²)
- `gK_max`: Max K⁺ conductance (mS/cm²)
- `Cm`: Membrane capacitance (µF/cm²)
- `ENa, EK, EL`: Reversal potentials (mV)

#### **Environment**
- `T_celsius`: Experiment temperature (°C)
- `Q10`: Temperature coefficient

#### **Stimulation**
- `stim_type`: Stimulus type (const/pulse/alpha)
- `Iext`: Stimulus amplitude (µA/cm²)
- `stim_comp`: Stimulation compartment

## 🔬 Scientific Presets

### **Physiological Neurons**

#### **A: Squid Giant Axon (HH 1952)**
- **Temperature:** 6.3°C (cold-blooded)
- **Conductances:** gNa=120, gK=36, gL=0.3 mS/cm²
- **Application:** Classical Hodgkin-Huxley model

#### **B: Pyramidal L5 (Mainen 1996)**
- **Temperature:** 37°C, T_ref=23°C, Q10=2.3
- **Conductances:** gNa=56, gK=6, gL=0.02 mS/cm²
- **Morphology:** Soma 20 µm + AIS (40× gNa)

#### **C: FS Interneuron (Wang-Buzsaki)**
- **Temperature:** 37°C
- **Conductances:** gNa=120, gK=36, gL=0.1 mS/cm²
- **Mode:** Fast spiking 150-400 Hz

### **Pathological Models**

#### **F: Multiple Sclerosis (Demyelination)**
- **Changes:** Ra: 70→300 Ω·cm, gL: 0.3→1.2
- **Effect:** Reduced spike amplitude

#### **M: Epilepsy (SCN1A mutation)**
- **Changes:** gNa: 120→200 mS/cm²
- **Effect:** Hyperexcitability

---

## 📚 Справочник API | API Reference

### Основные классы | Core Classes

#### `FullModelConfig`
Основная конфигурация модели со всеми параметрами.

#### `NeuronSolver`
Высокопроизводительный решатель ОДУ с Numba JIT.

#### `Translator`
Двуязычная система перевода (RU/EN).

### Функции пресетов | Preset Functions

#### `apply_preset(cfg, name)`
Применяет научный пресет к конфигурации.

#### `get_preset_names()`
Возвращает список доступных пресетов.

---

## 🤝 Справка | Support

### 📧 Контакты | Contact
- **GitHub Issues:** [Repository Issues](https://github.com/your-repo/issues)
- **Documentation:** [Full Docs](https://docs.neuromodelport.org)

### 📄 Лицензия | License
MIT License - см. файл LICENSE для деталей.

---

*NeuroModelPort v10.1 - © 2024 Computational Neuroscience Laboratory*
