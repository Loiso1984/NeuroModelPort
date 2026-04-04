# NeuroModelPort v10.1 - Руководство по двуязычной поддержке
# NeuroModelPort v10.1 - Bilingual Support Guide

## 🎯 Цель | Goal

Создать полностью двуязычную среду (Русский/English) для:
- Create fully bilingual environment (Russian/English) for:
- Интерфейса пользователя | User Interface
- Документации кода | Code Documentation  
- Всплывающих подсказок | Tooltips
- Научной документации | Scientific Documentation

## 📋 Принципы双语ности | Bilingual Principles

### 1. Сохранение русских комментариев | Preserve Russian Comments
```python
# Русский комментарий | English comment
def function_name():
    """Русское описание | English description"""
    pass
```

### 2. Двуязычный интерфейс | Bilingual Interface
```python
# В locales.py
TEXTS = {
    'EN': {
        'param_name': 'English description',
    },
    'RU': {
        'param_name': 'Русское описание',
    }
}
```

### 3. Научные подсказки | Scientific Tooltips
```python
# В bilingual_tooltips.py
SCIENTIFIC_TOOLTIPS = {
    'param_key': {
        'en': 'English scientific explanation',
        'ru': 'Русское научное объяснение',
    }
}
```

## 🔧 Реализация | Implementation

### Файлы перевода | Translation Files

#### `gui/locales.py`
- Основные переводы интерфейса | Main interface translations
- Кнопки, меню, статусы | Buttons, menus, status
- Параметры и описания | Parameters and descriptions

#### `gui/bilingual_tooltips.py`
- Научные всплывающие подсказки | Scientific tooltips
- Форматирование значений | Value formatting
- Динамическое переключение | Dynamic switching

### Функции перевода | Translation Functions

```python
# Основные функции | Main functions
T.tr(key)           # Перевод строки интерфейса | Interface string translation
T.desc(key)         # Описание параметра | Parameter description  
T.func_desc(key)     # Описание функции | Function description

# Расширенные функции | Extended functions
TOOLTIP_MANAGER.get_tooltip(key)           # Получить подсказку | Get tooltip
get_scientific_tooltip(key, show_both)  # Научная подсказка | Scientific tooltip
format_parameter_tooltip(name, value, units)  # Форматирование | Formatting
```

## 📝 Добавление нового контента | Adding New Content

### 1. Новый параметр интерфейса | New Interface Parameter

#### Шаг 1: Добавить в locales.py
```python
# В TEXTS['EN']
'new_param': 'New parameter description',

# В TEXTS['RU'] 
'new_param': 'Описание нового параметра',
```

#### Шаг 2: Добавить научную подсказку
```python
# В SCIENTIFIC_TOOLTIPS
'new_param': {
    'en': 'Detailed English scientific explanation with units and ranges.',
    'ru': 'Подробное русское научное объяснение с единицами и диапазонами.',
}
```

#### Шаг 3: Использовать в GUI
```python
from gui.bilingual_tooltips import BilingualQWidget

class MyWidget(BilingualQWidget):
    def __init__(self):
        super().__init__()
        self.set_scientific_tooltip('new_param', current_value, 'units')
```

### 2. Новая функция с комментариями | New Function with Comments

```python
def calculate_membrane_time_constant(cm, gl):
    """
    Вычислить постоянную времени мембраны τ = Cm/gL.
    Calculate membrane time constant τ = Cm/gL.
    
    Параметры | Parameters:
    -----------
    cm : float
        Ёмкость мембраны (мкФ/см²) | Membrane capacitance (µF/cm²)
    gl : float  
        Проводимость утечки (мСм/см²) | Leak conductance (mS/cm²)
        
    Возвращает | Returns:
    --------
    float
        Постоянная времени (мс) | Time constant (ms)
    """
    return cm / gl
```

### 3. Новый пресет | New Preset

```python
def apply_preset(cfg: FullModelConfig, name: str):
    """Применить пресет | Apply preset"""
    
    if "New Neuron Type" in name:
        # Настройки с двуязычными комментариями | Settings with bilingual comments
        cfg.morphology.d_soma = 25e-4  # 25 мкм | 25 µm
        cfg.channels.gNa_max = 60.0     # мСм/см² | mS/cm²
        
        # Научное обоснование | Scientific rationale
        cfg.env.T_celsius = 37.0  # Физиологическая температура | Physiological temperature
```

## 🎨 GUI интеграция | GUI Integration

### Переключение языка | Language Switching

```python
def change_language(lang: str):
    """Переключить язык интерфейса | Switch interface language"""
    
    # 1. Обновить переводчик | Update translator
    TOOLTIP_MANAGER.set_language(lang)
    T.set_language(lang)
    
    # 2. Обновить все виджеты | Update all widgets
    update_all_tooltips_language(lang)
    
    # 3. Обновить заголовки | Update titles
    main_window.setWindowTitle(T.tr('app_title'))
```

### Двуязычные метки | Bilingual Labels

```python
from gui.bilingual_tooltips import format_parameter_tooltip

# Создание метки с подсказкой | Create label with tooltip
param_label = QLabel(T.tr('param_name'))
param_label.set_bilingual_tooltip('param_key')

# Создание метки со значением | Create label with value
value_label = QLabel(f"{current_value} {units}")
value_label.setToolTip(format_parameter_tooltip('param_key', current_value, units))
```

## 📚 Структура документации | Documentation Structure

### Основные файлы | Main Files

1. **`DOCUMENTATION_BILINGUAL.md`**
   - Полное руководство на обоих языках | Complete guide in both languages
   - Научные пресеты | Scientific presets
   - API справочник | API reference

2. **`DOCUMENTATION_INDEX.md`**
   - Индекс всей документации | Index of all documentation
   - Структура проекта | Project structure
   - Руководства разработчика | Developer guides

3. **`README.md`**
   - Краткое описание (EN) | Brief description (EN)
   - Быстрый старт | Quick start

### Форматирование | Formatting

```markdown
## 🇷🇺 Русский раздел | Russian Section

### Подзаголовок | Subheading

- **Параметр | Parameter:** Описание | Description
- **Значение | Value:** 15.0 мкА/см² | 15.0 µA/cm²

---

## 🇺🇸 English Section

### Subheading

- **Parameter:** Description
- **Value:** 15.0 µA/cm²
```

## 🧪 Тестирование переводов | Translation Testing

### Юнит-тесты | Unit Tests

```python
def test_bilingual_tooltips():
    """Тест двуязычных подсказок | Test bilingual tooltips"""
    
    TOOLTIP_MANAGER.set_language('RU')
    ru_tooltip = get_scientific_tooltip('soma_diameter')
    assert 'Диаметр сомы' in ru_tooltip
    
    TOOLTIP_MANAGER.set_language('EN') 
    en_tooltip = get_scientific_tooltip('soma_diameter')
    assert 'Soma diameter' in en_tooltip
    
def test_parameter_formatting():
    """Тест форматирования параметров | Test parameter formatting"""
    tooltip = format_parameter_tooltip('gna_max', 120.0, 'mS/cm²')
    assert '120.0 mS/cm²' in tooltip
```

### Интеграционные тесты | Integration Tests

```python
def test_language_switching():
    """Тест переключения языка | Test language switching"""
    
    # Переключить на русский | Switch to Russian
    change_language('RU')
    assert main_window.windowTitle().contains('Симулятор')
    
    # Переключить на английский | Switch to English  
    change_language('EN')
    assert main_window.windowTitle().contains('Simulator')
```

## 📋 Чек-лист双语ности | Bilingual Checklist

### ✅ Код | Code
- [ ] Русские комментарии сохранены | Russian comments preserved
- [ ] Английские дубликаты добавлены | English duplicates added
- [ ] Функции документированы | Functions documented
- [ ] Параметры описаны | Parameters described

### ✅ Интерфейс | Interface  
- [ ] Все элементы переведены | All elements translated
- [ ] Всплывающие подсказки двуязычные | Tooltips bilingual
- [ ] Переключение языка работает | Language switching works
- [ ] Значения форматированы | Values formatted

### ✅ Документация | Documentation
- [ ] Основная документация двуязычная | Main documentation bilingual
- [ ] Научные термины объяснены | Scientific terms explained
- [ ] Примеры кода двуязычные | Code examples bilingual
- [ ] API справочник полный | Complete API reference

## 🔄 Обновление контента | Content Updates

### Процесс добавления нового перевода | New Translation Addition Process

1. **Добавить в locales.py** | Add to locales.py
2. **Добавить в bilingual_tooltips.py** | Add to bilingual_tooltips.py  
3. **Обновить DOCUMENTATION_BILINGUAL.md** | Update DOCUMENTATION_BILINGUAL.md
4. **Протестировать** | Test
5. **Закоммитить** | Commit

### Автоматизация | Automation

```python
# Скрипт проверки полноты переводов | Translation completeness check script
def check_translation_completeness():
    """Проверить полноту всех переводов | Check completeness of all translations"""
    
    missing_keys = []
    
    # Проверить все ключи в EN и RU | Check all keys in EN and RU
    en_keys = set(T.TEXTS['EN'].keys())
    ru_keys = set(T.TEXTS['RU'].keys())
    
    # Найти отсутствующие переводы | Find missing translations
    missing_en = ru_keys - en_keys
    missing_ru = en_keys - ru_keys
    
    if missing_en:
        print(f"Missing English translations: {missing_en}")
    if missing_ru:
        print(f"Missing Russian translations: {missing_ru}")
        
    return len(missing_en) == 0 and len(missing_ru) == 0
```

## 📞 Поддержка | Support

### Контакты для вопросов о双语ности | Bilingual Support Contacts

- **Технические вопросы | Technical Questions:** [tech@neuromodelport.org](mailto:tech@neuromodelport.org)
- **Вопросы переводов | Translation Questions:** [i18n@neuromodelport.org](mailto:i18n@neuromodelport.org)
- **GitHub Issues:** [Repository Issues](https://github.com/your-repo/issues)

### Ресурсы по локализации | Localization Resources

- **Unicode стандарты | Unicode Standards:** [Unicode.org](https://unicode.org)
- **PySide6 i18n:** [Qt Documentation](https://doc.qt.io/qtforpython/)
- **Научная терминология | Scientific Terminology:** [Neuroscience Glossary](https://www.neuroscienceglossary.com)

---

*Данное руководство обеспечивает полную双语ную поддержку NeuroModelPort v10.1*
*This guide ensures complete bilingual support for NeuroModelPort v10.1*
