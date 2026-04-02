"""
scripts/development_workflow.py - Рабочий процесс разработки
Development workflow manager for NeuroModelPort

Организует полный цикл разработки: создание фичи → тестирование → интеграция
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from feature_manager import LocalFeatureManager
from test_runner import TestRunner

class DevelopmentWorkflow:
    """Менеджер рабочего процесса разработки"""
    
    def __init__(self, project_root: str = "c:\\NeuroModelPort"):
        self.project_root = Path(project_root)
        self.current_session = None
        self.feature_manager = LocalFeatureManager()
        self.test_runner = TestRunner()
        
    def start_development_session(self, feature_name: str, description: str = "") -> Dict:
        """
        Начать сессию разработки новой фичи
        
        Returns session info
        """
        print(f"🚀 Starting Development Session")
        print(f"🎯 Feature: {feature_name}")
        if description:
            print(f"📝 Description: {description}")
        print("=" * 50)
        
        # 1. Создать фичу
        if not self.feature_manager.create_feature(feature_name, description):
            return {'status': 'error', 'message': 'Failed to create feature'}
        
        # 2. Запустить базовые тесты (чтобы убедиться что система работает)
        print("\n🔍 Running baseline tests...")
        baseline_results = self.test_runner.run_all_tests()
        
        # 3. Создать сессию
        session = {
            'feature_name': feature_name,
            'description': description,
            'started_at': datetime.now().isoformat(),
            'baseline_tests': baseline_results,
            'changes_made': [],
            'test_results': [],
            'status': 'active'
        }
        
        self.current_session = session
        
        # Сохранить сессию
        self._save_session(session)
        
        print(f"\n✅ Development session started for '{feature_name}'")
        print(f"📁 Feature directory: {self.project_root}/_features/{feature_name}")
        
        return session
    
    def add_change(self, file_path: str, change_description: str, content: str):
        """
        Добавить изменение в текущую сессию
        
        Parameters:
        -----------
        file_path : str
            Путь к измененному файлу
        change_description : str  
            Описание изменения
        content : str
            Новое содержимое файла
        """
        if not self.current_session:
            print("❌ No active development session")
            return
        
        # 1. Отследить изменение в feature manager
        self.feature_manager.save_file_snapshot(file_path, content)
        
        # 2. Добавить в сессию
        change = {
            'file': file_path,
            'description': change_description,
            'timestamp': datetime.now().isoformat(),
            'type': 'modification'
        }
        
        self.current_session['changes_made'].append(change)
        
        # 3. Сохранить сессию
        self._save_session(self.current_session)
        
        print(f"📝 Change tracked: {change_description}")
        print(f"📄 File: {file_path}")
    
    def run_development_tests(self, test_type: str = "all") -> Dict:
        """
        Запустить тесты для текущей фичи
        
        test_type: 'all', 'channels', 'quick', 'gui'
        """
        if not self.current_session:
            print("❌ No active development session")
            return {}
        
        feature_name = self.current_session['feature_name']
        print(f"\n🧪 Running {test_type} tests for '{feature_name}'")
        print("=" * 50)
        
        # Запустить соответствующие тесты
        if test_type == "channels":
            results = self.test_runner._test_channels_only()
        elif test_type == "quick":
            # Базовые тесты для быстрой проверки
            results = {
                'syntax': self.test_runner._test_syntax(),
                'unit': self.test_runner._run_unit_tests(),
                'presets': self.test_runner._validate_presets()
            }
        else:
            # Полный набор тестов
            results = self.test_runner.run_all_tests(feature_name)
        
        # Добавить в сессию
        test_result = {
            'timestamp': datetime.now().isoformat(),
            'type': test_type,
            'results': results
        }
        
        self.current_session['test_results'].append(test_result)
        self._save_session(self.current_session)
        
        return results
    
    def review_changes(self) -> Dict:
        """
        Показать все изменения в текущей сессии
        """
        # Попробовать загрузить текущую сессию
        if not self.current_session:
            # Попробовать найти последнюю сессию
            sessions_dir = self.project_root / "_sessions"
            if sessions_dir.exists():
                session_files = list(sessions_dir.glob("*.json"))
                if session_files:
                    # Загрузить последнюю сессию
                    latest_file = max(session_files, key=lambda x: x.stat().st_mtime)
                    with open(latest_file, 'r') as f:
                        self.current_session = json.load(f)
                    
        if not self.current_session:
            print("❌ No active development session")
            return {}
        
        print(f"\n📋 Development Session Review")
        print(f"🎯 Feature: {self.current_session['feature_name']}")
        print(f"📅 Started: {self.current_session['started_at']}")
        print("=" * 50)
        
        print(f"\n📝 Changes Made ({len(self.current_session['changes_made'])}):")
        for i, change in enumerate(self.current_session['changes_made'], 1):
            print(f"{i}. {change['description']}")
            print(f"   📄 {change['file']}")
            print(f"   🕐 {change['timestamp']}")
        
        print(f"\n🧪 Test Runs ({len(self.current_session['test_results'])}):")
        for i, test in enumerate(self.current_session['test_results'], 1):
            status = test['results'].get('overall_status', 'UNKNOWN')
            emoji = '✅' if status == 'PASSED' else '❌' if status == 'FAILED' else '⚠️'
            print(f"{i}. {test['type']} test - {emoji} {status}")
            print(f"   🕐 {test['timestamp']}")
        
        return self.current_session
    
    def integrate_feature_if_ready(self) -> bool:
        """
        Интегрировать фичу если тесты пройдены
        """
        if not self.current_session:
            print("❌ No active development session")
            return False
        
        # Проверить последние результаты тестов
        if not self.current_session['test_results']:
            print("❌ No tests run yet")
            return False
        
        latest_test = self.current_session['test_results'][-1]
        test_results = latest_test['results']
        
        overall_status = test_results.get('overall_status', 'FAILED')
        
        if overall_status != 'PASSED':
            print(f"❌ Tests not passed ({overall_status})")
            print("🔧 Fix issues before integration")
            
            # Показать что сломалось
            self._show_failed_tests(test_results)
            return False
        
        # Интегрировать
        feature_name = self.current_session['feature_name']
        if self.feature_manager.integrate_feature(feature_name):
            self.current_session['status'] = 'integrated'
            self.current_session['integrated_at'] = datetime.now().isoformat()
            self._save_session(self.current_session)
            
            print(f"✅ Feature '{feature_name}' successfully integrated!")
            self.current_session = None
            return True
        
        return False
    
    def abort_session(self):
        """Прервать текущую сессию разработки"""
        if not self.current_session:
            print("❌ No active development session")
            return
        
        self.current_session['status'] = 'aborted'
        self.current_session['aborted_at'] = datetime.now().isoformat()
        self._save_session(self.current_session)
        
        print(f"⏹️ Development session aborted")
        self.current_session = None
    
    def _save_session(self, session: Dict):
        """Сохранить сессию в файл"""
        sessions_dir = self.project_root / "_sessions"
        sessions_dir.mkdir(exist_ok=True)
        
        session_file = sessions_dir / f"{session['feature_name']}.json"
        with open(session_file, 'w') as f:
            json.dump(session, f, indent=2)
    
    def _show_failed_tests(self, test_results: Dict):
        """Показать детали проваленных тестов"""
        print("\n❌ Failed Tests Details:")
        print("-" * 30)
        
        for test_name, result in test_results.get('tests', {}).items():
            if isinstance(result, dict) and result.get('status') == 'failed':
                print(f"🔴 {test_name}: {result.get('message', 'Unknown error')}")
                
                if 'errors' in result:
                    for error in result['errors']:
                        print(f"   💥 {error}")
    
    def list_active_sessions(self) -> List[Dict]:
        """Показать все активные сессии"""
        sessions_dir = self.project_root / "_sessions"
        if not sessions_dir.exists():
            return []
        
        sessions = []
        for session_file in sessions_dir.glob("*.json"):
            with open(session_file, 'r') as f:
                session = json.load(f)
                sessions.append(session)
        
        return sessions


# Глобальный менеджер рабочего процесса
WORKFLOW = DevelopmentWorkflow()

# Удобные функции для использования
def start_feature(feature_name: str, description: str = "") -> Dict:
    """Начать разработку фичи"""
    return WORKFLOW.start_development_session(feature_name, description)

def make_change(file_path: str, description: str, content: str):
    """Сделать изменение в текущей фиче"""
    WORKFLOW.add_change(file_path, description, content)

def test_my_feature(test_type: str = "all") -> Dict:
    """Протестировать текущую фичу"""
    return WORKFLOW.run_development_tests(test_type)

def review_my_work():
    """Посмотреть что сделано"""
    return WORKFLOW.review_changes()

def integrate_my_feature() -> bool:
    """Интегрировать фичу если готова"""
    return WORKFLOW.integrate_feature_if_ready()

def abort_feature():
    """Отменить текущую фичу"""
    WORKFLOW.abort_session()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NeuroModelPort Development Workflow")
    parser.add_argument('command', choices=['start', 'test', 'review', 'integrate', 'abort', 'list'])
    parser.add_argument('--feature', type=str, help='Feature name')
    parser.add_argument('--description', type=str, help='Feature description')
    parser.add_argument('--test-type', choices=['all', 'channels', 'quick'], default='all')
    parser.add_argument('--file', type=str, help='File to modify')
    parser.add_argument('--change-desc', type=str, help='Change description')
    
    args = parser.parse_args()
    
    if args.command == 'start':
        start_feature(args.feature or input("Feature name: "), 
                   args.description or input("Description: "))
    
    elif args.command == 'test':
        test_my_feature(args.test_type)
    
    elif args.command == 'review':
        review_my_work()
    
    elif args.command == 'integrate':
        integrate_my_feature()
    
    elif args.command == 'abort':
        abort_feature()
    
    elif args.command == 'list':
        sessions = WORKFLOW.list_active_sessions()
        print("\n🌿 Active Development Sessions:")
        print("=" * 40)
        for session in sessions:
            status = session.get('status', 'unknown')
            print(f"🎯 {session['feature_name']} - {status}")
            print(f"   📅 Started: {session['started_at']}")
            print()
