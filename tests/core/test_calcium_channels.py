"""
tests/core/test_calcium_channels.py - Специализированные тесты кальциевых каналов
Specialized calcium channel validation tests

Тестирует:
1. Nernst динамику и стабильность
2. Диапазоны концентраций кальция
3. Баланс токов (Ca втекает, K вытекает)
4. Температурную зависимость
5. Влияние на порог возбудимости
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Добавить путь к проекту
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.models import FullModelConfig
from core.presets import apply_preset
from core.morphology import MorphologyBuilder
from core.solver import NeuronSolver
from core.analysis import detect_spikes


class CalciumChannelValidator:
    """Валидатор кальциевых каналов"""
    
    def __init__(self):
        self.test_results = {}
        
    def test_nernst_stability(self) -> dict:
        """Тест 1: Стабильность Nernst потенциала"""
        print("🧪 Testing Nernst stability...")
        
        results = {
            'test_name': 'Nernst Stability',
            'passed': True,
            'details': []
        }
        
        # Тестируем разные концентрации кальция
        ca_concentrations = [50e-6, 100e-6, 200e-6, 500e-6, 1e-3]  # 50nM to 1µM
        ca_ext = 2.0  # 2mM external
        t_kelvin = 310.15  # 37°C
        
        eca_values = []
        for ca_i in ca_concentrations:
            # Используем ту же функцию что в rhs.py
            from core.rhs import nernst_ca_ion
            eca = nernst_ca_ion(ca_i, ca_ext, t_kelvin)
            eca_values.append(eca)
        
        # Проверяем физическую реализуемость
        eca_min, eca_max = min(eca_values), max(eca_values)
        
        # Nernst для Ca должен быть в диапазоне 100-150 мВ
        if eca_min < 80 or eca_max > 180:
            results['passed'] = False
            results['details'].append(f"E_Ca range unrealistic: {eca_min:.1f} to {eca_max:.1f} mV")
        
        # Проверяем монотонность (больше Ca_i -> меньше E_Ca)
        if not all(eca_values[i] >= eca_values[i+1] for i in range(len(eca_values)-1)):
            results['passed'] = False
            results['details'].append("E_Ca not monotonic with Ca_i concentration")
        
        results['eca_values'] = eca_values
        results['ca_concentrations'] = ca_concentrations
        
        print(f"   E_Ca range: {eca_min:.1f} to {eca_max:.1f} mV")
        print(f"   Monotonic: {'✅' if all(eca_values[i] >= eca_values[i+1] for i in range(len(eca_values)-1)) else '❌'}")
        
        return results
    
    def test_concentration_ranges(self) -> dict:
        """Тест 2: Диапазоны концентраций кальция"""
        print("🧪 Testing calcium concentration ranges...")
        
        results = {
            'test_name': 'Calcium Concentration Ranges',
            'passed': True,
            'details': []
        }
        
        # Тестируем пресеты с кальциевыми каналами
        calcium_presets = [
            "K: Thalamic Relay (Ih + ICa + Burst)",
            "L: Hippocampal CA1 (Theta rhythm)",
            "M: Epilepsy (v10 SCN1A mutation)",
            "N: Alzheimer's (v10 Calcium Toxicity)",
            "O: Hypoxia (v10 ATP-pump failure)"
        ]
        
        for preset_name in calcium_presets:
            try:
                cfg = FullModelConfig()
                apply_preset(cfg, preset_name)
                
                # Проверяем включен ли кальций
                if not cfg.channels.enable_ICa:
                    results['details'].append(f"{preset_name}: ICa not enabled")
                    continue
                
                # Проверяем параметры кальция
                ca_params = cfg.calcium
                
                # Физиологические диапазоны
                if not (1e-9 <= ca_params.Ca_rest <= 1e-3):  # 1nM to 1µM
                    results['passed'] = False
                    results['details'].append(f"{preset_name}: Ca_rest out of range: {ca_params.Ca_rest}")
                
                if not (0.1 <= ca_params.tau_Ca <= 1000):  # 0.1ms to 1s
                    results['passed'] = False
                    results['details'].append(f"{preset_name}: tau_Ca out of range: {ca_params.tau_Ca}")
                
                if not (0.0005 <= cfg.calcium.B_Ca <= 0.01):  # Physiological range
                    results['passed'] = False
                    results['details'].append(f"{preset_name}: B_Ca out of range: {cfg.calcium.B_Ca}")
                
                print(f"   {preset_name}: ✅ Ca_rest={ca_params.Ca_rest:.2e}, tau={ca_params.tau_Ca:.1f}")
                
            except Exception as e:
                results['passed'] = False
                results['details'].append(f"{preset_name}: Error - {e}")
        
        return results
    
    def test_current_balance(self) -> dict:
        """Тест 3: Баланс кальциевых и калиевых токов"""
        print("🧪 Testing calcium-potassium current balance...")
        
        results = {
            'test_name': 'Calcium-Potassium Current Balance',
            'passed': True,
            'details': []
        }
        
        # Тестируем Thalamic relay (у него есть и Ca, и K)
        cfg = FullModelConfig()
        apply_preset(cfg, "K: Thalamic Relay (Ih + ICa + Burst)")
        
        # Запускаем симуляцию
        try:
            solver = NeuronSolver(cfg)
            result = solver.run_single()
            
            # Анализируем токи
            v = result.v_soma
            t = result.t
            
            # Детектируем спайки
            spike_indices, spike_times, _ = detect_spikes(v, t, threshold=-20.0)
            
            if len(spike_indices) > 5:  # Нужно достаточно спайков
                # Анализируем изменение внутриклеточного кальция
                if hasattr(result, 'ca_i') and result.ca_i is not None:
                    ca_i_trace = result.ca_i  # Shape: (n_comp, n_timepoints)
                    
                    # Находим типичный спайк
                    spike_idx = spike_indices[2]  # третий спайк
                    
                    # Берем трейс кальция из сомы (первый компартмент)
                    soma_ca_trace = ca_i_trace[0, :]  # Сома
                    
                    pre_spike_ca = soma_ca_trace[max(0, spike_idx-20)]
                    post_spike_ca = soma_ca_trace[min(len(soma_ca_trace)-1, spike_idx+20)]
                    
                    ca_increase = post_spike_ca - pre_spike_ca
                    
                    # Проверяем баланс: Ca должен входить во время спайка
                    if ca_increase < 1e-9:  # меньше 1nM
                        results['passed'] = False
                        results['details'].append(f"Insufficient calcium influx during spike: {ca_increase:.2e}")
                    
                    print(f"   Ca influx per spike: {ca_increase:.2e} M")
                else:
                    results['details'].append("No calcium dynamics in simulation results")
            
        except Exception as e:
            results['passed'] = False
            results['details'].append(f"Simulation error: {e}")
        
        return results
    
    def test_temperature_scaling(self) -> dict:
        """Тест 4: Температурная зависимость"""
        print("🧪 Testing temperature scaling...")
        
        results = {
            'test_name': 'Calcium Temperature Scaling',
            'passed': True,
            'details': []
        }
        
        # Тестируем при разных температурах
        temperatures = [23.0, 30.0, 37.0, 40.0]  # °C
        
        baseline_freq = None
        
        for temp in temperatures:
            try:
                cfg = FullModelConfig()
                apply_preset(cfg, "K: Thalamic Relay (Ih + ICa + Burst)")
                cfg.env.T_celsius = temp
                
                # Запускаем быструю симуляцию
                solver = NeuronSolver(cfg)
                result = solver.run_single()
                
                # Детектируем спайки
                spike_indices, _, _ = detect_spikes(result.v_soma, result.t, threshold=-20.0)
                firing_freq = len(spike_indices) / (result.t[-1] / 1000.0)
                
                if baseline_freq is None:
                    baseline_freq = firing_freq
                
                # Проверяем разумность температурной зависимости
                temp_ratio = firing_freq / baseline_freq if baseline_freq > 0 else 1.0
                
                # При повышении температуры частота должна расти (Q10 effect)
                if temp > 23.0 and temp_ratio < 0.8:  # не должно сильно падать
                    results['passed'] = False
                    results['details'].append(f"Unrealistic temperature scaling at {temp}°C: ratio={temp_ratio:.2f}")
                
                print(f"   {temp}°C: {firing_freq:.1f} Hz (ratio: {temp_ratio:.2f})")
                
            except Exception as e:
                results['passed'] = False
                results['details'].append(f"Temperature {temp}°C error: {e}")
        
        return results
    
    def test_excitability_thresholds(self) -> dict:
        """Тест 5: Влияние на порог возбудимости"""
        print("🧪 Testing excitability thresholds...")
        
        results = {
            'test_name': 'Calcium Excitability Thresholds',
            'passed': True,
            'details': []
        }
        
        # Сравниваем пороги с и без Ca каналов
        test_preset = "B: Pyramidal L5 (Mainen 1996)"
        
        thresholds = {}
        
        for enable_ca in [False, True]:
            cfg = FullModelConfig()
            apply_preset(cfg, test_preset)
            cfg.channels.enable_ICa = enable_ca
            
            # Находим минимальный ток для спайкинга
            min_current = None
            test_currents = np.linspace(5.0, 25.0, 9)  # 5-25 µA/cm²
            
            for current in test_currents:
                cfg.stim.Iext = current
                try:
                    solver = NeuronSolver(cfg)
                    result = solver.run_single()
                    
                    spike_indices, _, _ = detect_spikes(result.v_soma, result.t, threshold=-20.0)
                    
                    if len(spike_indices) > 2:  # хотя бы 2 спайка
                        min_current = current
                        break
                        
                except:
                    continue
            
            thresholds[f'Ca_{enable_ca}'] = min_current
        
        if thresholds['Ca_True'] and thresholds['Ca_False']:
            ratio = thresholds['Ca_True'] / thresholds['Ca_False']
            
            # Ca каналы обычно СНИЖАЮТ порог (бутстропная инактивация)
            if ratio > 1.2:  # Ca повышает порог более чем на 20%
                results['passed'] = False
                results['details'].append(f"Calcium channels increase threshold too much: ratio={ratio:.2f}")
            
            print(f"   Threshold without Ca: {thresholds['Ca_False']:.1f} µA/cm²")
            print(f"   Threshold with Ca: {thresholds['Ca_True']:.1f} µA/cm²")
            print(f"   Ratio (Ca/NoCa): {ratio:.2f}")
        
        return results
    
    def run_all_tests(self) -> dict:
        """Запустить все тесты кальциевых каналов"""
        print("🧪 CALCIUM CHANNEL VALIDATION SUITE")
        print("=" * 50)
        
        all_results = {}
        
        # Запускаем все тесты
        test_methods = [
            self.test_nernst_stability,
            self.test_concentration_ranges,
            self.test_current_balance,
            self.test_temperature_scaling,
            self.test_excitability_thresholds
        ]
        
        for test_method in test_methods:
            try:
                result = test_method()
                all_results[result['test_name']] = result
                
                status = '✅ PASSED' if result['passed'] else '❌ FAILED'
                print(f"\n{result['test_name']}: {status}")
                
                if result['details']:
                    for detail in result['details']:
                        print(f"   ⚠️ {detail}")
                        
            except Exception as e:
                error_result = {
                    'test_name': test_method.__name__.replace('test_', ''),
                    'passed': False,
                    'details': [f"Test execution error: {e}"]
                }
                all_results[error_result['test_name']] = error_result
                print(f"\n{error_result['test_name']}: ❌ ERROR - {e}")
        
        # Итоговый результат
        passed_count = sum(1 for r in all_results.values() if r['passed'])
        total_count = len(all_results)
        
        overall_status = 'PASSED' if passed_count == total_count else 'FAILED'
        
        print(f"\n{'='*50}")
        print(f"📊 CALCIUM VALIDATION SUMMARY")
        print(f"{'='*50}")
        print(f"Overall Status: {'✅ PASSED' if overall_status == 'PASSED' else '❌ FAILED'}")
        print(f"Tests Passed: {passed_count}/{total_count}")
        
        return {
            'overall_status': overall_status,
            'individual_tests': all_results,
            'passed_count': passed_count,
            'total_count': total_count
        }


def main():
    """Основная функция для запуска тестов"""
    validator = CalciumChannelValidator()
    results = validator.run_all_tests()
    
    # Сохранить результаты
    import json
    from datetime import datetime
    
    output_file = Path("c:/NeuroModelPort/_test_results/calcium_validation.json")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return results['overall_status'] == 'PASSED'


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
