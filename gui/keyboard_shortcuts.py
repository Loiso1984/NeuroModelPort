"""
gui/keyboard_shortcuts.py - Keyboard Shortcuts Manager v11.7

Provides convenient keyboard shortcuts for power users.
Improves researcher workflow by reducing mouse dependency.

Shortcuts:
- Ctrl+R: Run simulation
- Ctrl+S: Stochastic simulation
- Ctrl+1-5: Quick preset selection
- Ctrl+E: Export results
- Ctrl+T: Toggle language (EN/RU)
- F5: Run (same as Ctrl+R)
- F11: Fullscreen oscilloscope
"""

from PySide6.QtWidgets import QMainWindow
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtCore import QObject, Signal
from typing import Callable, Optional

from gui.locales import T


class ShortcutManager(QObject):
    """Manages keyboard shortcuts for MainWindow."""
    
    # Signals for shortcut actions
    run_simulation = Signal()
    run_stochastic = Signal()
    export_results = Signal()
    toggle_language = Signal()
    preset_requested = Signal(int)  # 1-5 for presets 1-5
    fullscreen_oscilloscope = Signal()
    
    def __init__(self, parent: Optional[QMainWindow] = None):
        super().__init__(parent)
        self._parent = parent
        self._shortcuts: list = []
        self._setup_shortcuts()
    
    def _setup_shortcuts(self):
        """Create all keyboard shortcuts."""
        # Run simulation: Ctrl+R or F5
        self._add_shortcut("Ctrl+R", self.run_simulation.emit)
        self._add_shortcut("F5", self.run_simulation.emit)
        
        # Stochastic: Ctrl+S
        self._add_shortcut("Ctrl+S", self.run_stochastic.emit)
        
        # Export: Ctrl+E
        self._add_shortcut("Ctrl+E", self.export_results.emit)
        
        # Language toggle: Ctrl+T
        self._add_shortcut("Ctrl+T", self.toggle_language.emit)
        
        # Quick presets: Ctrl+1 through Ctrl+5
        for i in range(1, 6):
            self._add_shortcut(f"Ctrl+{i}", lambda idx=i: self.preset_requested.emit(idx))
        
        # Fullscreen oscilloscope: F11
        self._add_shortcut("F11", self.fullscreen_oscilloscope.emit)
    
    def _add_shortcut(self, key: str, callback: Callable):
        """Add a keyboard shortcut."""
        if not self._parent:
            return
            
        shortcut = QShortcut(QKeySequence(key), self._parent)
        shortcut.activated.connect(callback)
        self._shortcuts.append(shortcut)
    
    def enable_all(self):
        """Enable all shortcuts."""
        for sc in self._shortcuts:
            sc.setEnabled(True)
    
    def disable_all(self):
        """Disable all shortcuts (e.g., during simulation)."""
        for sc in self._shortcuts:
            sc.setEnabled(False)
    
    def get_shortcuts_help(self) -> str:
        """Return formatted help text for shortcuts."""
        return """
╔══════════════════════════════════════════════════════════════╗
║              KEYBOARD SHORTCUTS v11.7                        ║
╠══════════════════════════════════════════════════════════════╣
║  Ctrl+R / F5    →  Run simulation                            ║
║  Ctrl+S         →  Stochastic simulation                     ║
║  Ctrl+E         →  Export results to CSV                     ║
║  Ctrl+T         →  Toggle language (EN/RU)                 ║
║  Ctrl+1..5      →  Quick preset 1-5                        ║
║  F11            →  Fullscreen oscilloscope                 ║
╚══════════════════════════════════════════════════════════════╝
        """.strip()


def show_shortcuts_dialog(parent=None):
    """Show keyboard shortcuts help dialog."""
    from PySide6.QtWidgets import QMessageBox
    
    manager = ShortcutManager()
    help_text = manager.get_shortcuts_help()
    
    msg = QMessageBox(parent)
    msg.setWindowTitle(T.tr("shortcuts_title", "Keyboard Shortcuts"))
    msg.setTextFormat(Qt.PlainText)
    msg.setText(help_text)
    msg.setIcon(QMessageBox.Information)
    msg.exec()
