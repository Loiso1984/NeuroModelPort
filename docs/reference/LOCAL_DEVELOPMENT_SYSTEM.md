# 🌿 Локальная система разработки NeuroModelPort v10.1
# Local Development System for NeuroModelPort v10.1

## 🎯 **Концепция**

Избегаем сложности с Git и создаем удобную локальную систему для:
- Создания "виртуальных веток" фичей
- Изолированного тестирования изменений  
- Безопасной интеграции в основную кодовую базу
- Полного аудита всех модификаций

## 📁 **Структура системы**

```
c:\NeuroModelPort/
├── scripts/                          # 🛠️ Скрипты разработки
│   ├── feature_manager.py            # Управление фичами
│   ├── test_runner.py               # Универсальный тестовый раннер  
│   └── development_workflow.py      # Полный рабочий процесс
├── _features/                      # 🌿 Виртуальные ветки фичей
│   ├── spike-detection-config/      # Пример фичи
│   │   ├── metadata.json           # Метаданные фичи
│   │   ├── snapshots/             # Снимки измененных файлов
│   │   └── test_results.json     # Результаты тестов
│   └── [другие фичи]/
├── _backups/                       # 💾 Бэкапы основного кода
│   ├── backup_20240401_180000/   # Timestamped бэкапы
│   └── ...
└── _sessions/                     # 📋 Сессии разработки
    ├── spike-detection-config.json  # Активные сессии
    └── ...
```

## 🚀 **Быстрый старт**

### **1. Начать новую фичу:**
```bash
cd c:\NeuroModelPort
python scripts/development_workflow.py start --feature "spike-detection-config" --description "Configurable spike detection thresholds"
```

### **2. Вносить изменения:**
```python
# В коде工作时
from scripts.development_workflow import make_change

# Сохранить изменение
make_change(
    file_path="gui/analytics.py",
    description="Add configurable spike detection threshold",
    content=new_file_content
)
```

### **3. Тестировать:**
```bash
# Быстрые тесты
python scripts/development_workflow.py test --test-type quick

# Полные тесты  
python scripts/development_workflow.py test --test-type all

# Только тесты каналов
python scripts/development_workflow.py test --test-type channels
```

### **4. Просмотреть прогресс:**
```bash
python scripts/development_workflow.py review
```

### **5. Интегрировать если готово:**
```bash
python scripts/development_workflow.py integrate
```

## 🧪 **Типы тестов**

### **🔍 Базовые тесты (Quick):**
- Синтаксис Python файлов
- Базовые юнит-тесты
- Валидация пресетов

### **🧬 Специализированные тесты каналов:**
- **Calcium:** Nernst стабильность, диапазоны концентраций
- **HCN (Ih):** Кривые активации, sag потенциалы
- **A-current:** Кинетика инактивации, влияние на спайкинг
- **Multi-channel:** Стресс-тесты взаимодействий

### **🔬 Комплексные тесты:**
- Производительность симуляции
- GUI функциональность
- Аналитика и детекция спайков
- Интеграционные тесты

## 📊 **Отчетность и аудит**

### **Автоматическое сохранение:**
- 📝 Все изменения отслеживаются
- 🧪 Результаты тестов сохраняются
- 📋 Прогресс сессии записывается
- 💾 Бэкапы создаются автоматически

### **Визуализация прогресса:**
```bash
# Показать все активные фичи
python scripts/feature_manager.py

# Показать детали текущей работы
python scripts/development_workflow.py review
```

## 🎯 **Пример рабочего процесса**

### **Разработка настраиваемой детекции спайков:**

```bash
# 1. Начать фичу
python scripts/development_workflow.py start --feature "spike-config" --description "Add configurable spike detection"

# 2. Внести изменения в GUI
# (работаем в IDE, изменения автоматически отслеживаются)

# 3. Быстрая проверка
python scripts/development_workflow.py test --test-type quick

# 4. Тесты каналов (важно!)
python scripts/development_workflow.py test --test-type channels

# 5. Полная валидация  
python scripts/development_workflow.py test --test-type all

# 6. Просмотр результатов
python scripts/development_workflow.py review

# 7. Интеграция если все ок
python scripts/development_workflow.py integrate
```

## 🛡️ **Безопасность**

### **Автоматические бэкапы:**
- Перед каждой новой фичой создается бэкап
- Сохраняются все важные директории (core, gui, tests)
- Timestamped для истории

### **Валидация перед интеграцией:**
- Все тесты должны быть пройдены
- Проверяется синтаксис и функциональность
- Только после успешной валидации - интеграция

### **Откат изменений:**
```bash
# Отменить текущую фичу
python scripts/development_workflow.py abort

# Система вернется к последнему бэкапу
```

## 📈 **Преимущества подхода**

### **🎯 Простота:**
- Никаких Git команд
- Все через Python скрипты
- Понятная структура директорий

### **🔒 Безопасность:**
- Изолированная разработка
- Автоматические бэкапы
- Валидация перед интеграцией

### **📊 Аудит:**
- Полная история изменений
- Результаты всех тестов
- Прогресс отслеживается

### **🚀 Скорость:**
- Быстрое переключение между фичами
- Локальное тестирование
- Мгновенная интеграция

## 🎨 **Интеграция с IDE**

### **Для VSCode / PyCharm:**
```python
# Добавить в начало рабочего файла
from scripts.development_workflow import make_change, test_my_feature

# При изменении файла:
make_change(
    file_path=__file__,
    description="Add new feature",
    content=new_content
)
```

### **Автоматические тесты:**
```python
# Можно добавить в __main__ для автотестирования
if __name__ == "__main__":
    test_my_feature("quick")
```

---

## 📞 **Использование**

### **Начать работу:**
1. Выберите фичу из нашего списка
2. Запустите `python scripts/development_workflow.py start --feature "название"`
3. Вносите изменения как обычно
4. Периодически запускайте тесты
5. После готовности - интегрируйте

### **Текущие фичи в планах:**
1. **spike-detection-config** - Настраиваемая детекция спайков
2. **refactor-dual-stim** - Рефакторинг двойной стимуляции  
3. **error-handling** - Система обработки ошибок
4. **calcium-validation** - Валидация кальциевых каналов

Эта система даст нам безопасность Git-подхода с простотой локальной разработки!
