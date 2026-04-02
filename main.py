import sys
import os

# Добавляем корень проекта в пути поиска, чтобы Python видел папки gui и core
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow

def main():
    # Создаем экземпляр приложения Qt
    app = QApplication(sys.argv)
    
    # Добавляем современный стиль
    app.setStyle("Fusion")
    
    # Создаем и показываем главное окно
    try:
        window = MainWindow()
        window.show()
    except Exception as e:
        print(f"Критическая ошибка при запуске окна: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Запускаем цикл обработки событий
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

    