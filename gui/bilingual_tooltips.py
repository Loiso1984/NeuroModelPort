"""
gui/bilingual_tooltips.py - Двуязычная система всплывающих подсказок v10.1
Bilingual tooltip system v10.1

Provides bilingual tooltips for GUI elements with automatic language switching.
"""

from typing import Dict, Tuple, Optional
from .locales import T


class BilingualTooltipManager:
    """Управление двуязычными всплывающими подсказками | Manages bilingual tooltips."""
    
    def __init__(self):
        self.current_lang = 'EN'
        self.tooltip_cache = {}
        
    def set_language(self, lang: str):
        """Установить язык | Set language."""
        self.current_lang = lang
        T.set_language(lang)
        
    def get_tooltip(self, key: str, custom_en: str = None, custom_ru: str = None) -> str:
        """
        Получить двуязычную всплывающую подсказку | Get bilingual tooltip.
        
        Parameters:
        -----------
        key : str
            Ключ для перевода из locales.py | Translation key from locales.py
        custom_en : str, optional
            Кастомный английский текст | Custom English text
        custom_ru : str, optional
            Кастомный русский текст | Custom Russian text
            
        Returns:
        --------
        str
            Текст на текущем языке | Text in current language
        """
        if custom_en and custom_ru:
            return custom_ru if self.current_lang == 'RU' else custom_en
            
        # Используем существующий перевод из locales.py
        base_text = T.desc(key) if key else ''
        
        if not base_text and custom_en:
            return custom_ru if self.current_lang == 'RU' else custom_en
            
        return base_text
        
    def get_bilingual_tooltip(self, key: str, custom_en: str = None, custom_ru: str = None) -> str:
        """
        Получить двуязычную подсказку (оба языка) | Get bilingual tooltip (both languages).
        
        Returns formatted string with both languages.
        """
        en_text = custom_en or T.desc(key) or ''
        ru_text = custom_ru or T.desc(key) or ''
        
        if self.current_lang == 'RU':
            return f"{ru_text}\n───\n{en_text}" if en_text else ru_text
        else:
            return f"{en_text}\n───\n{ru_text}" if ru_text else en_text
            
    def get_parameter_tooltip(self, param_name: str, value: str = None) -> str:
        """
        Получить всплывающую подсказку для параметра с текущим значением | Get parameter tooltip with current value.
        """
        base_tooltip = self.get_tooltip(param_name)
        if value is not None:
            value_suffix = f"\n\n📍 Текущее значение | Current value: {value}"
            return base_tooltip + value_suffix
        return base_tooltip


# Глобальный менеджер всплывающих подсказок | Global tooltip manager
TOOLTIP_MANAGER = BilingualTooltipManager()


# Предопределенные научные подсказки | Predefined scientific tooltips
SCIENTIFIC_TOOLTIPS = {
    # Морфология | Morphology
    'soma_diameter': {
        'en': 'Soma diameter (cm). Larger soma = more capacitance = harder to excite.',
        'ru': 'Диаметр сомы (см). Больше → больше ёмкость → труднее возбудить.'
    },
    'ais_segments': {
        'en': 'AIS segments. The axon initial segment is the spike trigger zone.',
        'ru': 'Сегменты AIS. AIS — «курок» нейрона, зона инициации спайка.'
    },
    'gna_multiplier': {
        'en': 'gNa multiplier in AIS (typically 40–100×). Sets excitability threshold.',
        'ru': 'Множитель gNa в AIS (обычно 40–100×). Задаёт порог возбудимости.'
    },
    
    # Каналы | Channels
    'membrane_capacitance': {
        'en': 'Membrane capacitance (µF/cm²). Electrical inertia — higher Cm slows response.',
        'ru': 'Ёмкость мембраны (мкФ/см²). Инерция: большая Cm замедляет реакцию.'
    },
    'sodium_conductance': {
        'en': 'Max Na⁺ conductance. Drives the fast depolarising upstroke of the AP.',
        'ru': 'Макс. проводимость Na⁺. Отвечает за быстрый фронт спайка.'
    },
    'potassium_conductance': {
        'en': 'Max K⁺ conductance. Repolarises the membrane after the spike.',
        'ru': 'Макс. проводимость K⁺. Реполяризует мембрану после спайка.'
    },
    
    # Среда | Environment
    'temperature': {
        'en': 'Experiment temperature (°C). Scales all kinetics via Q10.',
        'ru': 'Температура эксперимента (°C). Масштабирует кинетику через Q10.'
    },
    'q10_coefficient': {
        'en': 'Q10 coefficient. Rate doubles per 10°C if Q10 = 2.',
        'ru': 'Коэффициент Q10. При Q10=3 скорость растёт в 3 раза за 10°C.'
    },
    
    # Стимуляция | Stimulation
    'stimulus_amplitude': {
        'en': 'Stimulus amplitude (µA/cm²). Current density for injection.',
        'ru': 'Амплитуда стимула (мкА/см²). Плотность тока для инъекции.'
    },
    'stimulus_type': {
        'en': 'Stimulus waveform: const / pulse / alpha / OU noise / synaptic receptors / zap chirp.',
        'ru': 'Тип стимула: const / pulse / alpha / шум OU / синаптические рецепторы / ZAP-чирп.'
    },
    'stimulus_location': {
        'en': 'Compartment index to inject current into (0 = soma).',
        'ru': 'Компартмент стимуляции (0 = сома).'
    }
}


def get_scientific_tooltip(key: str, show_both: bool = False) -> str:
    """
    Получить научную всплывающую подсказку | Get scientific tooltip.
    
    Parameters:
    -----------
    key : str
            Ключ научной подсказки | Scientific tooltip key
    show_both : bool
            Показать оба языка | Show both languages
            
    Returns:
    --------
    str
            Форматированная подсказка | Formatted tooltip
    """
    if key not in SCIENTIFIC_TOOLTIPS:
        return TOOLTIP_MANAGER.get_tooltip(key)
        
    tooltip_data = SCIENTIFIC_TOOLTIPS[key]
    
    if show_both:
        en_text = tooltip_data['en']
        ru_text = tooltip_data['ru']
        return f"{ru_text}\n───\n{en_text}" if TOOLTIP_MANAGER.current_lang == 'RU' else f"{en_text}\n───\n{ru_text}"
    else:
        return tooltip_data['ru'] if TOOLTIP_MANAGER.current_lang == 'RU' else tooltip_data['en']


def format_parameter_tooltip(param_name: str, current_value: any, units: str = None) -> str:
    """
    Форматировать всплывающую подсказку параметра со значением и единицами | Format parameter tooltip with value and units.
    
    Parameters:
    -----------
    param_name : str
            Имя параметра | Parameter name
    current_value : any
            Текущее значение | Current value
    units : str, optional
            Единицы измерения | Units
            
    Returns:
    --------
    str
            Отформатированная подсказка | Formatted tooltip
    """
    base_tooltip = get_scientific_tooltip(param_name)
    value_str = str(current_value)
    
    if units:
        value_str += f" {units}"
        
    suffix = f"\n\n📍 Текущее значение | Current value: {value_str}"
    
    return base_tooltip + suffix


# Классы для интеграции с PySide6 | Classes for PySide6 integration
class BilingualQWidget:
    """Базовый класс для виджетов с двуязычными подсказками | Base class for widgets with bilingual tooltips."""
    
    def set_bilingual_tooltip(self, key: str, custom_en: str = None, custom_ru: str = None):
        """Установить двуязычную всплывающую подсказку | Set bilingual tooltip."""
        tooltip = TOOLTIP_MANAGER.get_tooltip(key, custom_en, custom_ru)
        self.setToolTip(tooltip)
        
    def set_scientific_tooltip(self, param_name: str, current_value: any = None, units: str = None):
        """Установить научную всплывающую подсказку | Set scientific tooltip."""
        if current_value is not None:
            tooltip = format_parameter_tooltip(param_name, current_value, units)
        else:
            tooltip = get_scientific_tooltip(param_name)
        self.setToolTip(tooltip)


# Утилиты для динамического обновления | Utilities for dynamic updates
def update_all_tooltips_language(lang: str):
    """
    Обновить язык всех всплывающих подсказок | Update language for all tooltips.
    
    Эта функция должна вызываться при смене языка в интерфейсе.
    This function should be called when language changes in the interface.
    """
    TOOLTIP_MANAGER.set_language(lang)
    # Здесь можно добавить логику для обновления всех виджетов
    # Here you can add logic to update all widgets


if __name__ == "__main__":
    # Тестирование системы всплывающих подсказок | Test tooltip system
    TOOLTIP_MANAGER.set_language('RU')
    print("Русский язык | Russian language:")
    print(get_scientific_tooltip('soma_diameter'))
    print(get_scientific_tooltip('soma_diameter', show_both=True))
    
    TOOLTIP_MANAGER.set_language('EN')
    print("\nАнглийский язык | English language:")
    print(get_scientific_tooltip('soma_diameter'))
    print(get_scientific_tooltip('soma_diameter', show_both=True))
