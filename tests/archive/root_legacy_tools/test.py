from core.models import FullModelConfig
from core.solver import NeuronSolver
import time

# Создаем конфигурацию: Пирамидный нейрон (L5) с Alpha-синапсом и SK-каналом
config = FullModelConfig()

# Настраиваем морфологию (многокомпартментная)
config.morphology.single_comp = False
config.morphology.N_ais = 2
config.morphology.N_trunk = 10

# Настраиваем стимул
config.stim.stim_type = 'alpha'
config.stim.Iext = 5.0
config.stim.pulse_start = 10.0

# Включаем спайковую адаптацию!
config.channels.enable_SK = True
config.channels.gSK_max = 2.0
config.calcium.dynamic_Ca = True
config.channels.enable_ICa = True  # Кальций должен входить, чтобы SK работал
config.stim.t_sim = 100.0
config.stim.dt_eval = 0.1

# Запуск
solver = NeuronSolver(config)

print("Начинаем симуляцию...")
start_time = time.time()
result = solver.run_single()
print(f"Готово за {time.time() - start_time:.3f} сек!")

# Получение потенциала сомы:
V_soma = result.v_soma
print(f"Пиковый потенциал: {V_soma.max():.2f} мВ")