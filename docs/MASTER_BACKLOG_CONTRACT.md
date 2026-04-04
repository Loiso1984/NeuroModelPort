# Master Backlog Contract

This document is the canonical, verbatim-preserved project contract and must be treated as immutable source context between sessions.

## Verbatim Backlog + Rules (User Source)

```text
Мы работаем с папкой NeuroModelPort
Изучи архитектуру приложения. Пока мы работаем с ним локально, не отвлекайся на гитхаб
Обрати внимание что в папке уже есть раздел с тестами и тестовой веткой.
Но тесты могут быть некорректными.

Правила!!!
Проверяй в процессе все исходные параметры ионных каналов и прочего на соответствие реальным. Можешь использовать поиск или брать данные из таких программ как Neuron и научной литературы
Добавь необходимость этих проверок в память
Все параметры пресетов должны максимально соответствовать физиологическим, а результаты тестов - показывать те пики и частоты что ожидаются физиологически от данного пресета (тета ритм итд)
Внося изменения которые касаются не интерфеса а значений presets или логики рассчётов МЫ ВСЕГДА сначала вносим их в тестовые ветки и тестируем и лишь потом меняем в основным файлах
Работая с тестами ВСЕГДА проверять их корректность. Как они считают факт спайка (пересечение baseline) и сверяем результаты теста с ожидаемыми физиологичесими параметрами (число спайков, напряжение и параметры каналов и импульса)

Полный список работ. В данный момент валидируем HCN, потом валидируем IA и всё вместе. Физиологию, тестовые пресеты, их флаги, широкие диапазоны обкатки на соответствие биологической реальности.

Список работ (если ты обнаружишь что задание уже было выполнено, можешь его пропустить)
0.Группировка тестов в отдельную папку по категориям. Очистка директории от ненужных артефактов дебага
1. Калибровка кальциевых каналов и Nernst динамики.
Проблема: Nernst потенциал давал нереалистичные значения
Тесты: диапазон [Ca²⁺]ᵢ, валидация E_Ca, стабильность при разных температурах
Фокус: presets K, L, M, N, O
Проверить пресет альцгеймера  особенно

Ca influx
E_Ca
B_Ca  gCa_max
2. Валидация HCN каналов и стабильности покоя
Проблема: Ih может вызывать нестабильность покоя
Тесты: V_½ активации, влияние на входное сопротивление, температурная чувствительность
Валидация iA каналов. Валидация пресетов и широкая серия Стресс тестов симуляции не только по пресетам по и по Sweep разных параметров. Цель-максимально физиологичные ответы нейрона на нормативных и не нормативных симуляциях. Корректная работа всех типов каналов по отдельности и вместе.
Также учитывай что не всем пресетам нужны все типы каналов, если они не нужны-можешь выключать их для пресета. Также ни один из пресетов по умолчанию не должен иметь Double Stimulation
2.1 Проверь корректную работу Double Stim в вычислениях и GUI и корректное функционирование на простых пресетах.
3. Настраиваемая детекция спайков в GUI
Проблема: Детекция "зашита" без контроля пользователя
Фичи: настраиваемый порог, выбор алгоритма, визуализация. Нужны общие улучшения информативности и настраиваемости в GUI
4. Рефакторинг двойной стимуляции
Проблема: Код добавлен напрямую в rhs.py
Решение: отдельный модуль core/dual_stimulation.py , отключение двойной стимуляции в пресетах по умолчанию.
5. Система обработки ошибок и логирования
Проблема: Отсутствует валидация параметров
Решение: кастомные исключения, logging система, вывод пользователю предупреждений, расчёт времени симуляции, предупреждение о краевых параметрах
6. Стресс-тесты мульти-канальных взаимодействий
Сценарии: Ih+ICa, IA+SK, все каналы включены
Цель: численная стабильность. Стресс тест пресетов а также поведения виртуального нейрона в разных нефизиологических и физиологических ситуациях (по литературным данным)
После всех тестов - все изменения можно и НУЖНО вносить в основную версию программы
7. Улучшение графиков и читаемости
Проблемы: малые шрифты, плохие цвета, недостаточная интерактивность
Фичи: настройка цветов, толщины, экспорт. Также на графике потенциалов должно быть видно спайк в соме и его отставание в терминальной версии аксона к примеру (или настраиваемо на развилке, или в произвольной).
8. Паспорт нейрона с ML классификацией
Подход: гибридная система (правила + простые классификаторы)
Признаки: частота, CV, адаптация, морфология. Улучшение информативности паспорта нейрона, добавление информации о поведении каналов, кинетики и прочее что покажется важным
📈 СРЕДНИЙ ПРИОРИТЕТ:
9. Визуализация топологии
Улучшение наглядности и читаемости, доработка анализа распространения нервного импульса по аксону (и его интенсивности). Наглядность приоритетна.
Возможность подавать первичный импульс не только на сому, AIS или дендритный фильтр, но и на произвольный сегмент аксона (если есть)
10. Real-time настройка параметров
Слайдеры с немедленным пересчетом
11. Экспорт графиков
PDF, PNG, SVG форматы
12. Проверка, группировка, дополнение и переписывание документации и для пользователя и для разработчика
13. Оптимизация вычислений (мультиканальные взаимодействия очень тяжеловесны) возможно перевод части вычислений на С или мультипоток
14. После всего сделанного-чистка старой документации, составление новой двуязычной документации, чистка дебаг-элементов, проверка словарей и полной двуязычности подсказок пользователю, добавление новых подсказок пользователю, отладка программы
15. Если в ходе работы ты поймёшь что программу можно ещё как-то улучшить - говори, обсудим
16. Что в модели целесообразно визуализировать или обсчитать с помощью формулы Ляпунова? Что это даст? Разумеется это должен быть выключенный по умолчанию флаг, тем не менее мне интересно где её можно применить к модели.
Также вопрос, было бы селесообразно также сделать простенькую визуализацию стимулирующего сигнала (к примеру альфа тока) для наглядности?
```

## Additional Mandatory Requirement (Newest)

If Jacobian optimization is confirmed to accelerate simulations without introducing numerical or physiological errors, Jacobian mode must be exposed in NeuronSolver GUI as a user-selectable option (and evaluated as a possible default for computationally heavy presets), not only in test runs.

## Additional Exploratory Item 17

17. Можем ли мы сделать что-то навроде фурье анализа (но не фурье анализ) для симуляций. Также настройка с флагом, чтобы для высокочастотных спайков узнать вклад в них модулирующего ритма (тета) основного от пресинаптической стимуляции итд?

Working interpretation:
1. phase-locking and preferred phase of spikes relative to low-frequency modulatory rhythm,
2. phase-dependent firing-rate/burst probability estimates,
3. surrogate-based significance checks to avoid false coupling.

## Additional Exploratory Item 18

18. Expanded ion/channel causality view on spike analytics:
1. add an optional analytics view that shows per-spike evolution of key ionic currents/channels and Ca load,
2. expose an interpretable explanation for spike attenuation/block tendency (e.g., Ca accumulation, SK/K increase, Na drive reduction),
3. keep this as analysis-layer diagnostics (default non-invasive for core solver behavior).

## Additional Exploratory Item 19

19. Extended ion/channel dynamics on spike analytics graph:
1. add a richer time-resolved panel (not only per-spike points) for each enabled ion current/channel activity,
2. emphasize causality of spike attenuation and conduction weakening (e.g., Ca accumulation -> SK growth -> spike shrinkage/block),
3. keep it behind analysis-layer controls and default-safe behavior.

## Integration Note for Item 16 (Lyapunov / LLE-FTLE)

Use a practical Lyapunov-based stability module (not symbolic Lyapunov function search):
1. compute FTLE/LLE from nearby trajectory divergence with periodic renormalization,
2. keep this analysis behind a default-OFF flag (`analysis.enable_lyapunov`),
3. report `FTLE_mean`, `FTLE_time_series`, and qualitative class:
   - stable (`<0`), limit-cycle-like (`~0`), unstable/chaotic (`>0`),
4. use only as stability descriptor together with physiological metrics (not as standalone realism criterion),
5. include dedicated synthetic tests where expected FTLE sign is known.

## Additional Process Directive (Newest)

After full physiology validation is explicitly closed, tasks focused on analytics/plots/GUI/documentation may be implemented directly in the main contour without tedious transfer flow; branch-first remains mandatory for any change to presets, channel parameters, or core physiological calculation logic.
