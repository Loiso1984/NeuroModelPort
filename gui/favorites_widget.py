"""
gui/favorites_widget.py - Quick Access Favorites Widget v11.7

Provides one-click access to frequently used neuron presets.
Improves researcher workflow by eliminating dropdown navigation.
"""

import json
import os
from pathlib import Path
from typing import List, Callable, Optional

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QMenu, QDialog,
    QVBoxLayout, QListWidget, QListWidgetItem, QInputDialog,
    QLabel, QLineEdit
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from gui.locales import T


DEFAULT_FAVORITES = [
    "A: Squid Giant Axon (HH 1952)",
    "B: Cortical Regular Spiking",
    "C: Thalamic Burst Mode",
]

FAVORITES_FILE = ".favorites.json"


class FavoritesWidget(QWidget):
    """Horizontal bar of favorite preset buttons with management features."""
    
    preset_selected = Signal(str)
    
    def __init__(self, parent=None, on_preset_load: Optional[Callable[[str], None]] = None):
        super().__init__(parent)
        self.on_preset_load = on_preset_load
        self._favorites: List[str] = []
        self._buttons: List[QPushButton] = []
        self._active_preset: str = ""
        
        self._setup_ui()
        self._load_favorites()
        self._refresh_buttons()
    
    def _setup_ui(self):
        """Create the widget layout."""
        self._layout = QHBoxLayout(self)
        self._layout.setSpacing(4)
        self._layout.setContentsMargins(0, 0, 0, 0)
        
        # Label
        self._label = QLabel("★ " + T.tr("favorites_label", "Favorites:") + " ")
        self._label.setStyleSheet("font-weight: bold; color: #89B4FA;")
        self._layout.addWidget(self._label)
        
        # Manage button with dropdown menu
        self._manage_btn = QPushButton("⚙️")
        self._manage_btn.setToolTip(T.tr("favorites_manage", "Manage favorites"))
        self._manage_btn.setFixedWidth(32)
        self._manage_btn.setStyleSheet("""
            QPushButton {
                background: #313244;
                border: 1px solid #45475A;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover { background: #45475A; }
        """)
        
        # Create menu
        self._menu = QMenu(self)
        self._menu.addAction(T.tr("favorites_add_current", "Add current preset"), self._add_current)
        self._menu.addAction(T.tr("favorites_edit", "Edit favorites..."), self._edit_favorites)
        self._menu.addSeparator()
        self._menu.addAction(T.tr("favorites_reset", "Reset to defaults"), self._reset_defaults)
        self._manage_btn.setMenu(self._menu)
        
        self._layout.addWidget(self._manage_btn)
        self._layout.addStretch()
    
    def _refresh_buttons(self):
        """Update favorite buttons display."""
        # Remove old buttons
        for btn in self._buttons:
            self._layout.removeWidget(btn)
            btn.deleteLater()
        self._buttons.clear()
        
        # Add buttons for each favorite
        for preset_name in self._favorites:
            btn = QPushButton(self._truncate_name(preset_name))
            btn.setToolTip(preset_name)
            
            # Highlight active preset
            is_active = preset_name == self._active_preset
            btn.setStyleSheet(self._button_style(is_active))
            
            btn.clicked.connect(lambda checked, name=preset_name: self._on_preset_clicked(name))
            
            # Insert before the stretch
            self._layout.insertWidget(self._layout.count() - 1, btn)
            self._buttons.append(btn)
    
    def _button_style(self, active: bool) -> str:
        """Generate button style based on active state."""
        if active:
            return """
                QPushButton {
                    background: #A6E3A1;
                    color: #1E1E2E;
                    border: 1px solid #A6E3A1;
                    border-radius: 4px;
                    padding: 4px 12px;
                    font-weight: bold;
                    font-size: 11px;
                }
            """
        return """
            QPushButton {
                background: #313244;
                color: #CDD6F4;
                border: 1px solid #45475A;
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 11px;
            }
            QPushButton:hover { background: #45475A; }
        """
    
    def _truncate_name(self, name: str, max_len: int = 20) -> str:
        """Truncate long preset names for display."""
        if len(name) <= max_len:
            return name
        return name[:max_len-3] + "..."
    
    def _on_preset_clicked(self, preset_name: str):
        """Handle favorite button click."""
        self._active_preset = preset_name
        self._refresh_buttons()
        self.preset_selected.emit(preset_name)
        if self.on_preset_load:
            self.on_preset_load(preset_name)
    
    def set_active_preset(self, preset_name: str):
        """Update active preset highlighting."""
        self._active_preset = preset_name
        self._refresh_buttons()
    
    def _load_favorites(self):
        """Load favorites from JSON file."""
        if os.path.exists(FAVORITES_FILE):
            try:
                with open(FAVORITES_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._favorites = data.get('favorites', DEFAULT_FAVORITES.copy())
            except Exception:
                self._favorites = DEFAULT_FAVORITES.copy()
        else:
            self._favorites = DEFAULT_FAVORITES.copy()
    
    def _save_favorites(self):
        """Save favorites to JSON file."""
        try:
            with open(FAVORITES_FILE, 'w', encoding='utf-8') as f:
                json.dump({'favorites': self._favorites}, f, indent=2)
        except Exception as e:
            print(f"Error saving favorites: {e}")
    
    def _add_current(self):
        """Add current active preset to favorites."""
        if self._active_preset and self._active_preset not in self._favorites:
            self._favorites.append(self._active_preset)
            self._save_favorites()
            self._refresh_buttons()
    
    def _edit_favorites(self):
        """Open dialog to edit favorites list."""
        dialog = EditFavoritesDialog(self, self._favorites)
        if dialog.exec():
            self._favorites = dialog.get_favorites()
            self._save_favorites()
            self._refresh_buttons()
    
    def _reset_defaults(self):
        """Reset favorites to default list."""
        self._favorites = DEFAULT_FAVORITES.copy()
        self._save_favorites()
        self._refresh_buttons()


class EditFavoritesDialog(QDialog):
    """Dialog for editing favorites list."""
    
    def __init__(self, parent=None, favorites: List[str] = None):
        super().__init__(parent)
        self.setWindowTitle(T.tr("favorites_edit_title", "Edit Favorites"))
        self.setMinimumWidth(400)
        self._favorites = favorites.copy() if favorites else []
        
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Instruction
        layout.addWidget(QLabel(T.tr("favorites_edit_hint", "Drag to reorder. Double-click to edit.")))
        
        # List widget
        self._list = QListWidget()
        self._list.setDragDropMode(QListWidget.InternalMove)
        for fav in self._favorites:
            item = QListWidgetItem(fav)
            self._list.addItem(item)
        layout.addWidget(self._list)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        add_btn = QPushButton(T.tr("favorites_add", "Add..."))
        add_btn.clicked.connect(self._add_item)
        btn_layout.addWidget(add_btn)
        
        remove_btn = QPushButton(T.tr("favorites_remove", "Remove"))
        remove_btn.clicked.connect(self._remove_selected)
        btn_layout.addWidget(remove_btn)
        
        btn_layout.addStretch()
        
        ok_btn = QPushButton(T.tr("dialog_ok", "OK"))
        ok_btn.clicked.connect(self.accept)
        btn_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton(T.tr("dialog_cancel", "Cancel"))
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)
    
    def _add_item(self):
        """Add new favorite via input dialog."""
        text, ok = QInputDialog.getText(
            self, 
            T.tr("favorites_add_title", "Add Favorite"),
            T.tr("favorites_add_prompt", "Enter preset name:")
        )
        if ok and text:
            self._list.addItem(QListWidgetItem(text))
    
    def _remove_selected(self):
        """Remove selected item from list."""
        for item in self._list.selectedItems():
            self._list.takeItem(self._list.row(item))
    
    def get_favorites(self) -> List[str]:
        """Return current favorites list."""
        return [self._list.item(i).text() for i in range(self._list.count())]
