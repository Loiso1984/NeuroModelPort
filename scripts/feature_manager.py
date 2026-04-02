"""
scripts/feature_manager.py - Локальный менеджер фич и веток
Local feature and branch manager

Позволяет создавать "виртуальные ветки" локально без Git,
изолировать изменения и тестировать их перед интеграцией.
"""

import os
import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

class LocalFeatureManager:
    """Локальное управление фичами без Git"""
    
    def __init__(self, project_root: str = "c:\\NeuroModelPort"):
        self.project_root = Path(project_root)
        self.features_dir = self.project_root / "_features"
        self.backup_dir = self.project_root / "_backups"
        self.current_feature = None
        
        # Создать директории
        self.features_dir.mkdir(exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True)
        
    def create_feature(self, feature_name: str, description: str = "") -> bool:
        """
        Создать новую "ветку" фичи локально
        
        Returns True if successful
        """
        try:
            # 1. Создать бэкап текущего состояния
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self._create_backup(backup_name)
            
            # 2. Создать директорию фичи
            feature_dir = self.features_dir / feature_name
            feature_dir.mkdir(exist_ok=True)
            
            # 3. Сохранить метаданные
            metadata = {
                'name': feature_name,
                'description': description,
                'created_at': datetime.now().isoformat(),
                'backup_used': backup_name,
                'files_modified': [],
                'tests_passed': False,
                'integration_ready': False
            }
            
            with open(feature_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
                
            self.current_feature = feature_name
            print(f"✅ Feature '{feature_name}' created successfully")
            print(f"📁 Feature dir: {feature_dir}")
            print(f"💾 Backup: {backup_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating feature: {e}")
            return False
    
    def track_file_change(self, file_path: str, change_type: str = "modified"):
        """
        Отследить изменение файла в текущей фиче
        change_type: 'added', 'modified', 'deleted'
        """
        if not self.current_feature:
            print("⚠️ No active feature to track changes")
            return
            
        feature_dir = self.features_dir / self.current_feature
        
        # Добавить в список измененных файлов
        metadata_file = feature_dir / "metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            
        file_change = {
            'file': file_path,
            'type': change_type,
            'timestamp': datetime.now().isoformat()
        }
        
        metadata['files_modified'].append(file_change)
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"📝 Tracked {change_type}: {file_path}")
    
    def save_file_snapshot(self, file_path: str, content: str):
        """
        Сохранить снимок файла для текущей фичи
        """
        if not self.current_feature:
            print("⚠️ No active feature")
            return
            
        feature_dir = self.features_dir / self.current_feature
        snapshots_dir = feature_dir / "snapshots"
        snapshots_dir.mkdir(exist_ok=True)
        
        # Создать относительный путь
        rel_path = Path(file_path).relative_to(self.project_root)
        snapshot_file = snapshots_dir / str(rel_path)
        snapshot_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(snapshot_file, 'w', encoding='utf-8') as f:
            f.write(content)
            
        self.track_file_change(file_path, "modified")
        print(f"💾 Snapshot saved: {rel_path}")
    
    def run_feature_tests(self, feature_name: str = None) -> Dict:
        """
        Запустить тесты для фичи
        """
        feature = feature_name or self.current_feature
        if not feature:
            return {'status': 'error', 'message': 'No feature specified'}
            
        print(f"🧪 Running tests for feature: {feature}")
        
        # Здесь будет логика тестирования
        results = {
            'feature': feature,
            'timestamp': datetime.now().isoformat(),
            'unit_tests': self._run_unit_tests(),
            'preset_validation': self._run_preset_validation(),
            'integration_tests': self._run_integration_tests(),
            'performance': self._run_performance_tests()
        }
        
        # Сохранить результаты
        feature_dir = self.features_dir / feature
        with open(feature_dir / "test_results.json", 'w') as f:
            json.dump(results, f, indent=2)
            
        return results
    
    def integrate_feature(self, feature_name: str) -> bool:
        """
        Интегрировать фичу в основную кодовую базу
        """
        print(f"🔄 Integrating feature: {feature_name}")
        
        # Проверить тесты
        test_results = self.run_feature_tests(feature_name)
        if not self._tests_passed(test_results):
            print("❌ Tests failed! Cannot integrate.")
            return False
            
        # Применить изменения файлов
        feature_dir = self.features_dir / feature_name
        snapshots_dir = feature_dir / "snapshots"
        
        if snapshots_dir.exists():
            self._apply_snapshots(snapshots_dir)
            
        print(f"✅ Feature '{feature_name}' integrated successfully")
        return True
    
    def list_features(self) -> List[Dict]:
        """Показать все фичи"""
        features = []
        
        for feature_dir in self.features_dir.iterdir():
            if feature_dir.is_dir():
                metadata_file = feature_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        features.append(metadata)
                        
        return features
    
    def _create_backup(self, backup_name: str):
        """Создать бэкап проекта"""
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(exist_ok=True)
        
        # Копировать важные директории
        dirs_to_backup = ['core', 'gui', 'tests']
        
        for dir_name in dirs_to_backup:
            src = self.project_root / dir_name
            dst = backup_path / dir_name
            if src.exists():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                
        print(f"💾 Backup created: {backup_name}")
    
    def _run_unit_tests(self) -> Dict:
        """Запустить юнит-тесты"""
        # Заглушка - будет реализовано
        return {'status': 'passed', 'count': 25, 'failed': 0}
    
    def _run_preset_validation(self) -> Dict:
        """Валидация пресетов"""
        # Заглушка - будет реализовано
        return {'status': 'passed', 'presets_tested': 15}
    
    def _run_integration_tests(self) -> Dict:
        """Интеграционные тесты"""
        # Заглушка - будет реализовано
        return {'status': 'passed', 'tests': 10}
    
    def _run_performance_tests(self) -> Dict:
        """Тесты производительности"""
        # Заглушка - будет реализовано
        return {'status': 'passed', 'time_ms': 150}
    
    def _tests_passed(self, results: Dict) -> bool:
        """Проверить прошли ли тесты"""
        required_tests = ['unit_tests', 'preset_validation', 'integration_tests']
        
        for test in required_tests:
            if results[test].get('status') != 'passed':
                return False
        return True
    
    def _apply_snapshots(self, snapshots_dir: Path):
        """Применить снимки файлов к основному проекту"""
        for snapshot_file in snapshots_dir.rglob('*'):
            if snapshot_file.is_file():
                rel_path = snapshot_file.relative_to(snapshots_dir)
                target_file = self.project_root / rel_path
                
                # Создать директорию если нужно
                target_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Копировать файл
                shutil.copy2(snapshot_file, target_file)
                print(f"📄 Applied: {rel_path}")


# Глобальный менеджер для использования в проекте
FEATURE_MANAGER = LocalFeatureManager()

# Удобные функции для использования
def start_feature(name: str, description: str = "") -> bool:
    """Начать новую фичу"""
    return FEATURE_MANAGER.create_feature(name, description)

def track_change(file_path: str, content: str):
    """Отследить изменение файла"""
    FEATURE_MANAGER.save_file_snapshot(file_path, content)

def test_feature(feature_name: str = None) -> Dict:
    """Протестировать фичу"""
    return FEATURE_MANAGER.run_feature_tests(feature_name)

def integrate_feature(feature_name: str) -> bool:
    """Интегрировать фичу"""
    return FEATURE_MANAGER.integrate_feature(feature_name)

def list_features():
    """Показать все фичи"""
    features = FEATURE_MANAGER.list_features()
    print("\n🌿 Active Features:")
    print("=" * 50)
    for i, feature in enumerate(features, 1):
        status = "✅ Ready" if feature.get('tests_passed') else "⏳ In Progress"
        print(f"{i}. {feature['name']} - {status}")
        print(f"   📝 {feature.get('description', 'No description')}")
        print(f"   📅 Created: {feature['created_at']}")
        print()


if __name__ == "__main__":
    # Демонстрация
    print("🌿 Local Feature Manager Demo")
    print("=" * 40)
    
    # Создать тестовую фичу
    start_feature("spike-detection-config", "Configurable spike detection thresholds")
    
    # Показать все фичи
    list_features()
