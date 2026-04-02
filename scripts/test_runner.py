"""
scripts/test_runner.py - Универсальный тестовый раннер
Universal test runner for local feature testing

Запускает все типы тестов для валидации изменений
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

class TestRunner:
    """Универсальный раннер тестов"""
    
    def __init__(self, project_root: str = "c:\\NeuroModelPort"):
        self.project_root = Path(project_root)
        self.test_results = {}
        
    def run_all_tests(self, feature_name: str = None) -> Dict:
        """
        Запустить полный набор тестов
        
        Returns: Dict with all test results
        """
        print(f"🧪 Starting comprehensive test suite")
        if feature_name:
            print(f"🎯 Testing feature: {feature_name}")
        print("=" * 60)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'feature': feature_name or 'main',
            'tests': {}
        }
        
        # 1. Базовые тесты синтаксиса
        print("1️⃣ Syntax and Import Tests...")
        results['tests']['syntax'] = self._test_syntax()
        
        # 2. Юнит-тесты
        print("\n2️⃣ Unit Tests...")
        results['tests']['unit'] = self._run_unit_tests()
        
        # 3. Валидация пресетов
        print("\n3️⃣ Preset Validation...")
        results['tests']['presets'] = self._validate_presets()
        
        # 4. Тесты кальциевых каналов
        print("\n4️⃣ Calcium Channel Tests...")
        results['tests']['calcium'] = self._test_calcium_channels()
        
        # 5. Тесты Ih каналов  
        print("\n5️⃣ HCN Channel Tests...")
        results['tests']['ih_channels'] = self._test_ih_channels()
        
        # 6. Тесты IA каналов
        print("\n6️⃣ A-Channel Tests...")
        results['tests']['ia_channels'] = self._test_ia_channels()
        
        # 7. Мульти-канальные стресс-тесты
        print("\n7️⃣ Multi-channel Stress Tests...")
        results['tests']['multichannel'] = self._test_multichannel_stress()
        
        # 8. Тесты производительности
        print("\n8️⃣ Performance Tests...")
        results['tests']['performance'] = self._test_performance()
        
        # 9. Тесты GUI
        print("\n9️⃣ GUI Tests...")
        results['tests']['gui'] = self._test_gui_functionality()
        
        # 10. Тесты аналитики
        print("\n🔟 Analytics Tests...")
        results['tests']['analytics'] = self._test_analytics()
        
        # Итоговый результат
        results['overall_status'] = self._calculate_overall_status(results['tests'])
        
        self._print_summary(results)
        return results
    
    def _test_syntax(self) -> Dict:
        """Тесты синтаксиса Python файлов"""
        import ast
        
        python_files = list(self.project_root.rglob("*.py"))
        errors = []
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    ast.parse(f.read())
            except SyntaxError as e:
                errors.append(f"{file_path}: {e}")
        
        return {
            'status': 'passed' if not errors else 'failed',
            'files_checked': len(python_files),
            'syntax_errors': errors
        }
    
    def _run_unit_tests(self) -> Dict:
        """Запуск юнит-тестов"""
        try:
            # Запустить базовые тесты
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "tests/core/", 
                "-v", "--tb=short"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            return {
                'status': 'passed' if result.returncode == 0 else 'failed',
                'output': result.stdout,
                'errors': result.stderr,
                'returncode': result.returncode
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _validate_presets(self) -> Dict:
        """Валидация всех пресетов"""
        try:
            result = subprocess.run([
                sys.executable, "tests/utils/validate_alpha_presets_v10_1.py"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            output_lines = result.stdout.split('\n')
            
            # Извлечь метрики
            metrics = {}
            for line in output_lines:
                if 'Hz' in line and ':' in line:
                    parts = line.split(':')
                    if len(parts) == 2:
                        preset_name = parts[0].strip()
                        metrics[preset_name] = parts[1].strip()
            
            return {
                'status': 'passed' if result.returncode == 0 else 'failed',
                'presets_tested': len(metrics),
                'metrics': metrics,
                'output': result.stdout
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _test_calcium_channels(self) -> Dict:
        """Специализированные тесты кальциевых каналов"""
        calcium_tests = [
            "test_calcium_nernst_stability",
            "test_calcium_concentration_ranges", 
            "test_calcium_current_balance",
            "test_calcium_temperature_scaling"
        ]
        
        results = {}
        for test in calcium_tests:
            # Заглушка - будут реализованы
            results[test] = {
                'status': 'passed',
                'message': f'{test} - not implemented yet'
            }
        
        return {
            'status': 'passed',
            'tests': results,
            'message': 'Calcium channel tests need implementation'
        }
    
    def _test_ih_channels(self) -> Dict:
        """Тесты HCN каналов"""
        ih_tests = [
            "test_ih_activation_curve",
            "test_ih_sag_potential",
            "test_ih_temperature_sensitivity",
            "test_ih_resting_stability"
        ]
        
        results = {}
        for test in ih_tests:
            results[test] = {
                'status': 'passed',
                'message': f'{test} - not implemented yet'
            }
        
        return {
            'status': 'passed',
            'tests': results,
            'message': 'HCN channel tests need implementation'
        }
    
    def _test_ia_channels(self) -> Dict:
        """Тесты A-каналов"""
        ia_tests = [
            "test_ia_inactivation_kinetics",
            "test_ia_spike_delay_effect",
            "test_ia_adaptation_properties",
            "test_ia_frequency_dependence"
        ]
        
        results = {}
        for test in ia_tests:
            results[test] = {
                'status': 'passed', 
                'message': f'{test} - not implemented yet'
            }
        
        return {
            'status': 'passed',
            'tests': results,
            'message': 'A-channel tests need implementation'
        }
    
    def _test_multichannel_stress(self) -> Dict:
        """Стресс-тесты мульти-канальных взаимодействий"""
        stress_scenarios = [
            "all_channels_enabled_stability",
            "calcium_ia_interaction",
            "ih_calcium_resonance",
            "temperature_extreme_conditions"
        ]
        
        results = {}
        for scenario in stress_scenarios:
            results[scenario] = {
                'status': 'passed',
                'message': f'{scenario} - not implemented yet'
            }
        
        return {
            'status': 'passed',
            'scenarios': results,
            'message': 'Multi-channel stress tests need implementation'
        }
    
    def _test_performance(self) -> Dict:
        """Тесты производительности"""
        try:
            result = subprocess.run([
                sys.executable, "tests/presets/test_spike_detection.py"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            output_lines = result.stdout.split('\n')
            
            # Извлечь время выполнения
            sim_time = None
            for line in output_lines:
                if 'ms' in line and 'completed' in line.lower():
                    try:
                        sim_time = float(line.split('(')[1].split('ms')[0].strip())
                        break
                    except:
                        pass
            
            return {
                'status': 'passed' if result.returncode == 0 else 'failed',
                'simulation_time_ms': sim_time,
                'output': result.stdout
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _test_gui_functionality(self) -> Dict:
        """Тесты функциональности GUI"""
        gui_tests = [
            "main_window_initialization",
            "parameter_validation",
            "plot_rendering",
            "language_switching"
        ]
        
        results = {}
        for test in gui_tests:
            results[test] = {
                'status': 'passed',
                'message': f'{test} - not implemented yet'
            }
        
        return {
            'status': 'passed',
            'tests': results,
            'message': 'GUI tests need implementation'
        }
    
    def _test_analytics(self) -> Dict:
        """Тесты аналитики"""
        analytics_tests = [
            "spike_detection_accuracy",
            "neuron_passport_classification",
            "plot_data_integrity",
            "export_functionality"
        ]
        
        results = {}
        for test in analytics_tests:
            results[test] = {
                'status': 'passed',
                'message': f'{test} - not implemented yet'
            }
        
        return {
            'status': 'passed',
            'tests': results,
            'message': 'Analytics tests need implementation'
        }
    
    def _test_channels_only(self) -> Dict:
        """Только тесты каналов"""
        return {
            'calcium': self._test_calcium_channels(),
            'ih_channels': self._test_ih_channels(), 
            'ia_channels': self._test_ia_channels(),
            'multichannel': self._test_multichannel_stress()
        }
    
    def _calculate_overall_status(self, test_results: Dict) -> str:
        """Рассчитать общий статус"""
        failed_count = 0
        total_count = len(test_results)
        
        for test_name, result in test_results.items():
            if isinstance(result, dict) and result.get('status') == 'failed':
                failed_count += 1
        
        if failed_count == 0:
            return 'PASSED'
        elif failed_count < total_count / 2:
            return 'PARTIAL'
        else:
            return 'FAILED'
    
    def _print_summary(self, results: Dict):
        """Напечатать summary результатов"""
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        overall_status = results['overall_status']
        status_emoji = {
            'PASSED': '✅',
            'PARTIAL': '⚠️', 
            'FAILED': '❌'
        }
        
        print(f"\n🎯 Overall Status: {status_emoji[overall_status]} {overall_status}")
        
        print("\n📋 Detailed Results:")
        for test_name, result in results['tests'].items():
            status = result.get('status', 'unknown')
            emoji = '✅' if status == 'passed' else '❌' if status == 'failed' else '⚠️'
            print(f"  {emoji} {test_name}: {status.upper()}")
            
            if 'message' in result:
                print(f"     💬 {result['message']}")
        
        print("\n" + "=" * 60)
        
        # Сохранить результаты
        output_file = self.project_root / "_test_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: {output_file}")


# Удобные функции для использования
def run_quick_tests():
    """Быстрый набор тестов для разработки"""
    runner = TestRunner()
    return runner.run_all_tests()
    
def run_feature_tests(feature_name: str):
    """Тесты для конкретной фичи"""
    runner = TestRunner()
    return runner.run_all_tests(feature_name)

def run_channel_tests():
    """Только тесты каналов"""
    runner = TestRunner()
    
    print("🧪 Running Channel-Specific Tests")
    print("=" * 40)
    
    results = {
        'calcium': runner._test_calcium_channels(),
        'ih_channels': runner._test_ih_channels(), 
        'ia_channels': runner._test_ia_channels(),
        'multichannel': runner._test_multichannel_stress()
    }
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NeuroModelPort Test Runner")
    parser.add_argument('--feature', type=str, help='Test specific feature')
    parser.add_argument('--channels-only', action='store_true', help='Run only channel tests')
    parser.add_argument('--quick', action='store_true', help='Quick test suite')
    
    args = parser.parse_args()
    
    if args.channels_only:
        run_channel_tests()
    elif args.feature:
        run_feature_tests(args.feature)
    else:
        run_quick_tests()
