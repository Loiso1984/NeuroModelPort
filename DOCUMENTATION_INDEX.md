# NeuroModelPort v10.1 - Индекс документации | Documentation Index

## 📚 Полная документация | Complete Documentation

### 🏠 Главная документация | Main Documentation
- **[📖 Двуязычная документация (RU/EN)](DOCUMENTATION_BILINGUAL.md)** - Полное руководство на русском и английском

### 📂 Структура проекта | Project Structure

```
c:\NeuroModelPort/
│
├── 📄 DOCUMENTATION_BILINGUAL.md     # Основная двуязычная документация
├── 📄 README.md                     # Краткое описание (EN)
├── 📄 INDEX.md                      # Индекс проекта (EN)
│
├── 🧠 core/                         # Ядро симуляции | Simulation core
│   ├── models.py                     # Конфигурация модели | Model configuration
│   ├── kinetics.py                   # Кинетика каналов | Channel kinetics (RU+EN comments)
│   ├── channels.py                   # Ионные каналы | Ion channels
│   ├── morphology.py                 # Морфология нейрона | Neuron morphology (RU+EN comments)
│   ├── rhs.py                        # Система ОДУ | ODE system (RU+EN comments)
│   ├── solver.py                     # Решатель ОДУ | ODE solver
│   ├── presets.py                    # Научные пресеты | Scientific presets
│   └── unit_converter.py             # Конвертер единиц | Unit converter
│
├── 🖥️ gui/                           # Графический интерфейс | GUI
│   ├── main_window.py                # Главное окно | Main window
│   ├── locales.py                    # Переводы интерфейса | Interface translations (RU/EN)
│   ├── bilingual_tooltips.py          # Двуязычные подсказки | Bilingual tooltips
│   ├── plots.py                      # Осциллограф | Oscilloscope
│   ├── analytics.py                  # Аналитика | Analytics
│   └── topology.py                  # Визуализация топологии | Topology visualization
│
└── 🧪 tests/                         # Тесты | Tests
    ├── core/                         # Тесты ядра | Core tests
    ├── presets/                      # Тесты пресетов | Preset tests
    └── utils/                        # Утилиты тестирования | Testing utilities
```

### 🌐 Языковая поддержка | Language Support

#### 🇷🇺 Русский язык | Russian Language
- **Интерфейс:** Полный перевод всех элементов GUI
- **Комментарии:** Русские комментарии сохранены в коде с английскими дубликатами
- **Документация:** Полное руководство на русском языке

#### 🇺🇸 English Language
- **Interface:** Complete translation of all GUI elements
- **Comments:** English translations provided alongside Russian comments
- **Documentation:** Complete English guide

### 📖 Руководства | Guides

#### 🚀 Быстрый старт | Quick Start
1. **Запуск | Launch:**
   ```bash
   cd c:\NeuroModelPort
   python main.py
   ```

2. **Выбор пресета | Select Preset:**
   - Выберите нейрон из списка | Choose neuron from list
   - Параметры применятся автоматически | Parameters applied automatically

3. **Симуляция | Simulate:**
   - Нажмите "▶ RUN" | Press "▶ RUN"
   - Наблюдайте спайкинг | Observe spiking

#### 🎛️ Параметры | Parameters

| Категория | Category | Описание | Description |
|-----------|------------|-----------|-------------|
| **Морфология** | **Morphology** | Размеры и структура нейрона | Neuron size and structure |
| **Каналы** | **Channels** | Ионные проводимости | Ionic conductances |
| **Среда** | **Environment** | Температура, Q10 | Temperature, Q10 |
| **Стимуляция** | **Stimulation** | Тип и амплитуда стимула | Stimulus type and amplitude |

### 🔬 Научные пресеты | Scientific Presets

#### 🧪 Физиологические модели | Physiological Models

| Пресет | Описание | Литература |
|---------|-----------|-------------|
| **A: Squid Giant Axon** | Классическая модель Ходжкина-Хаксли | Hodgkin & Huxley 1952 |
| **B: Pyramidal L5** | Пирамидный нейрон коры | Mainen & Sejnowski 1996 |
| **C: FS Interneuron** | Быстрый интернейрон | Wang & Buzsáki 1996 |
| **D: Alpha-Motoneuron** | Мотонейрон спинного мозга | Powers 2001 |
| **E: Purkinje Cell** | Клетка Пуркинье мозжечка | De Schutter & Bower 1994 |

#### 🏥 Патологические модели | Pathological Models

| Пресет | Патология | Эффект |
|---------|-----------|---------|
| **F: Multiple Sclerosis** | Рассеянный склероз | Демиелинизация |
| **G: Local Anesthesia** | Местная анестезия | Блокада Na каналов |
| **H: Hyperkalemia** | Гиперкалиемия | Высокий внеклеточный K⁺ |
| **M: Epilepsy** | Эпилепсия | Мутация SCN1A |
| **N: Alzheimer's** | Болезнь Альцгеймера | Токсичность кальция |

### 🛠️ Разработка | Development

#### 📝 Добавление новых пресетов | Adding New Presets

```python
def apply_preset(cfg: FullModelConfig, name: str):
    """Применить пресет | Apply preset"""
    
    if "New Neuron" in name:
        # Морфология | Morphology
        cfg.morphology.d_soma = 20e-4  # 20 µm
        cfg.morphology.N_ais = 3
        
        # Каналы | Channels
        cfg.channels.gNa_max = 50.0   # mS/cm²
        cfg.channels.gK_max = 10.0    # mS/cm²
        cfg.channels.gL = 0.05        # mS/cm²
        
        # Среда | Environment
        cfg.env.T_celsius = 37.0
        cfg.env.Q10 = 2.3
        
        # Стимуляция | Stimulation
        cfg.stim.Iext = 15.0  # µA/cm²
```

#### 🌐 Добавление переводов | Adding Translations

1. **Интерфейс | Interface:**
   ```python
   # В gui/locales.py
   'new_param': 'New parameter description',
   'новый_параметр': 'Описание нового параметра',
   ```

2. **Подсказки | Tooltips:**
   ```python
   # В gui/bilingual_tooltips.py
   'new_param': {
       'en': 'English description',
       'ru': 'Русское описание'
   }
   ```

### 📊 Тестирование | Testing

#### 🧪 Запуск тестов | Running Tests

```bash
# Быстрая проверка | Quick check
python tests/presets/test_squid_golden.py

# Полный набор тестов | Full test suite
pytest tests/ -v

# Валидация пресетов | Preset validation
python tests/utils/validate_alpha_presets_v10_1.py
```

#### 📈 Валидация | Validation

- **✅ Физиологические диапазоны** | Physiological ranges
- **✅ Соответствие литературе** | Literature correspondence
- **✅ Устойчивость чисел** | Numerical stability

### 🤝 Сообщество | Community

#### 🐛 Сообщить об ошибке | Report Bug
- **GitHub Issues:** [Ссылка на репозиторий](https://github.com/your-repo/issues)
- **Email:** [neuromodelport@example.com](mailto:neuromodelport@example.com)

#### 💬 Обсуждение | Discussion
- **Форум:** [Форум проекта](https://forum.neuromodelport.org)
- **Discord:** [Сервер Discord](https://discord.gg/neuromodelport)

#### 📚 Публикации | Publications
Если используете NeuroModelPort в исследованиях, пожалуйста цитируйте:
If you use NeuroModelPort in research, please cite:

```bibtex
@software{neuromodelport2024,
  title={NeuroModelPort v10.1: Biophysically Accurate Neuron Modeling},
  author={Computational Neuroscience Laboratory},
  year={2024},
  url={https://github.com/your-repo/neuromodelport}
}
```

---

## 📄 Лицензия | License

**MIT License**

Copyright (c) 2024 Computational Neuroscience Laboratory

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

---

*NeuroModelPort v10.1 - Двуязычная нейросимуляционная платформа*
*NeuroModelPort v10.1 - Bilingual Neuron Simulation Platform*
