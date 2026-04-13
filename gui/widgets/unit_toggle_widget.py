from PySide6.QtWidgets import QWidget, QHBoxLayout, QRadioButton, QButtonGroup
from PySide6.QtCore import Signal


class UnitToggleWidget(QWidget):
    unit_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._soma_area_cm2: float = 1.0
        self._current_density_val: float = 0.0
        
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        
        self._button_group = QButtonGroup(self)
        
        self._density_radio = QRadioButton("Density (µA/cm²)", self)
        self._density_radio.setChecked(True)
        self._button_group.addButton(self._density_radio)
        self._layout.addWidget(self._density_radio)
        
        self._absolute_radio = QRadioButton("Absolute (nA)", self)
        self._button_group.addButton(self._absolute_radio)
        self._layout.addWidget(self._absolute_radio)
        
        self._button_group.buttonClicked.connect(self._on_unit_changed)
    
    def _on_unit_changed(self, button):
        if button == self._density_radio:
            self.unit_changed.emit('density')
        elif button == self._absolute_radio:
            self.unit_changed.emit('absolute')
    
    def set_soma_area(self, area_cm2: float) -> None:
        self._soma_area_cm2 = area_cm2
    
    def get_converted_value(self, value_in_current_unit: float) -> float:
        if self._density_radio.isChecked():
            return value_in_current_unit
        else:
            return value_in_current_unit / (self._soma_area_cm2 * 1000.0)
    
    def get_display_value(self, density_val: float) -> float:
        if self._density_radio.isChecked():
            return density_val
        else:
            return density_val * self._soma_area_cm2 * 1000.0
