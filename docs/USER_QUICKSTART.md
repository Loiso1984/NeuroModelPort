# NeuroModelPort — User Quickstart (EN/RU)

## 1) Launch / Запуск

### EN
1. Install dependencies from `requirements.txt`.
2. Start the app:
   ```bash
   python main.py
   ```
3. In the main window choose a preset and click **Run**.

### RU
1. Установите зависимости из `requirements.txt`.
2. Запустите приложение:
   ```bash
   python main.py
   ```
3. В главном окне выберите пресет и нажмите **Run**.

---

## 2) Typical workflow / Типовой сценарий

### EN
1. **Setup:** choose preset + stimulation parameters (`Iext`, duration, temperature).
2. **Run simulation** and inspect:
   - Oscilloscope (voltage/gates/currents),
   - Topology (compartment map),
   - Analytics (impedance, spectrogram, phase metrics, etc.).
3. Adjust parameters and rerun.

### RU
1. **Настройка:** выберите пресет и параметры стимула (`Iext`, длительность, температура).
2. **Запустите симуляцию** и анализируйте:
   - Oscilloscope (напряжение/ворота/токи),
   - Topology (карта компартментов),
   - Analytics (импеданс, спектрограмма, phase-метрики и т.д.).
3. Меняйте параметры и повторяйте запуск.

---

## 3) Reading core outputs / Как читать ключевые результаты

### EN
- **Spike count / firing rate:** first sanity marker of excitability regime.
- **Delay (soma → target):** propagation quality indicator.
- **Current balance:** identifies dominant ionic contributors.
- **WARN/FAIL in stress reports:** inspect preset envelope and parameter sanity notes.

### RU
- **Spike count / firing rate:** базовый индикатор режима возбудимости.
- **Delay (soma → target):** показатель качества проведения.
- **Current balance:** показывает доминирующие ионные токи.
- **WARN/FAIL в stress-отчетах:** смотрите комментарии по диапазонам и sanity-проверкам.

---

## 4) Validation utilities / Утилиты валидации

### EN
- Extended pathology validation:
  ```bash
  python tests/utils/run_f_conduction_extended.py --target-ratio 0.3
  ```
- Preset stress sweep:
  ```bash
  PYTHONPATH=. python tests/utils/run_preset_stress_validation.py \
    --out tests/artifacts/preset_stress_validation.json \
    --report-md tests/artifacts/preset_stress_validation.md \
    --reference tests/utils/preset_reference_ranges.json
  ```

### RU
- Расширенная патологическая валидация:
  ```bash
  python tests/utils/run_f_conduction_extended.py --target-ratio 0.3
  ```
- Стресс-валидация пресетов:
  ```bash
  PYTHONPATH=. python tests/utils/run_preset_stress_validation.py \
    --out tests/artifacts/preset_stress_validation.json \
    --report-md tests/artifacts/preset_stress_validation.md \
    --reference tests/utils/preset_reference_ranges.json
  ```

---

## 5) Notes / Примечания

### EN
- In minimal CI/dev environments, some runtime deps (e.g. `pydantic`) may be unavailable.
- In such cases stress harnesses return a dependency diagnostic instead of raw traceback.

### RU
- В минимальных CI/dev окружениях часть runtime-зависимостей (например, `pydantic`) может отсутствовать.
- В этом случае stress-harness возвращает диагностическое сообщение о зависимости вместо «сырого» traceback.
