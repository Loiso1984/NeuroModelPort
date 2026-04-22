from PySide6.QtWidgets import QWidget, QVBoxLayout, QDoubleSpinBox, QLabel, QHBoxLayout
from PySide6.QtCore import Signal, QObject
import numpy as np
import logging

from gui.widgets.form_generator import PydanticFormWidget
from gui.widgets.unit_toggle_widget import UnitToggleWidget


class StimFormWithUnits(QWidget):
    """Primary Stimulus form with unit toggle for Iext field."""
    
    config_changed = Signal(str, object)  # field_name, value

    def __init__(self, stim_config, soma_diameter_cm: float, parent=None, on_change=None):
        super().__init__(parent)
        
        self.stim_config = stim_config
        self.soma_diameter_cm = soma_diameter_cm
        self.on_change = on_change
        
        # Calculate soma area
        self.soma_area_cm2 = np.pi * (self.soma_diameter_cm ** 2)
        
        # Setup layout
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)
        
        # Create unit toggle row for Iext (place above form)
        self._setup_unit_toggle_row()
        
        # Create the standard form (without Iext_absolute_nA)
        self.form_stim = PydanticFormWidget(
            self.stim_config,
            "Primary Stimulus",
            on_change=self._on_form_field_changed,
            hidden_fields={"Iext_absolute_nA"},
        )
        self._layout.addWidget(self.form_stim)
        
        # Expose widgets_map and labels_map for compatibility with existing code
        self.widgets_map = self.form_stim.widgets_map
        self.labels_map = self.form_stim.labels_map
        
        # Store reference to Iext widget for direct manipulation
        self._iext_widget = None
        self._iext_original_handler = None
        self._find_and_hook_iext_widget()
    
    def _setup_unit_toggle_row(self):
        """Create the unit toggle row for Iext field."""
        unit_row = QWidget()
        unit_layout = QHBoxLayout(unit_row)
        unit_layout.setContentsMargins(0, 0, 0, 0)
        unit_layout.setSpacing(8)
        
        # Unit toggle widget
        self.unit_toggle = UnitToggleWidget(self)
        self.unit_toggle.set_soma_area(self.soma_area_cm2)
        self.unit_toggle.unit_changed.connect(self._on_unit_changed)
        
        unit_layout.addWidget(QLabel("Iext Units:"))
        unit_layout.addWidget(self.unit_toggle)
        unit_layout.addStretch()
        
        self._layout.addWidget(unit_row)
    
    def _find_and_hook_iext_widget(self):
        """Find the Iext spinbox in the form and hook its valueChanged signal."""
        if 'Iext' in self.form_stim.widgets_map:
            self._iext_widget = self.form_stim.widgets_map['Iext']
            # Replace the form's default handler with our custom unit-aware handler
            try:
                # Block signals, disconnect form's handler, connect ours
                self._iext_widget.blockSignals(True)
                try:
                    self._iext_widget.valueChanged.disconnect()
                except (TypeError, RuntimeError):
                    # No connections to disconnect or already disconnected
                    pass
                self._iext_widget.valueChanged.connect(self._on_iext_value_changed)
                self._iext_widget.blockSignals(False)
            except Exception as e:
                logging.warning(f"Could not hook Iext widget signal: {e}")
            # Initialize display value based on current unit (density by default)
            self._update_display_value()
    
    def _on_form_field_changed(self, field_name: str, value):
        """Handle field changes from the form (non-Iext fields)."""
        if field_name != 'Iext':
            if self.on_change:
                self.on_change(field_name, value)
            self.config_changed.emit(field_name, value)
    
    def _on_iext_value_changed(self, value):
        """Handle Iext value changes with unit conversion."""
        if self.unit_toggle._density_radio.isChecked():
            # In density mode, value is already in µA/cm²
            density_value = value
        else:
            # In absolute mode, convert from nA to µA/cm²
            density_value = value / (self.soma_area_cm2 * 1000.0)
        
        # Update the config
        setattr(self.stim_config, 'Iext', density_value)
        
        if self.on_change:
            self.on_change('Iext', density_value)
        self.config_changed.emit('Iext', density_value)
    
    def _on_unit_changed(self, unit: str):
        """Handle unit toggle changes."""
        if self._iext_widget is None:
            self._find_and_hook_iext_widget()
            if self._iext_widget is None:
                logging.warning("Iext widget not found in form")
                return
        
        # Get current canonical density value from config
        current_density = self.stim_config.Iext
        
        # Convert to display value based on new unit
        if unit == 'density':
            display_value = current_density
        else:  # absolute
            display_value = current_density * self.soma_area_cm2 * 1000.0
        
        # Block signals to prevent triggering change event
        self._iext_widget.blockSignals(True)
        self._iext_widget.setValue(display_value)
        self._iext_widget.blockSignals(False)
    
    def _update_display_value(self):
        """Update the Iext display based on current unit setting."""
        if self._iext_widget is None:
            return
        
        current_density = self.stim_config.Iext
        
        if self.unit_toggle._density_radio.isChecked():
            display_value = current_density
        else:
            display_value = current_density * self.soma_area_cm2 * 1000.0
        
        self._iext_widget.blockSignals(True)
        self._iext_widget.setValue(display_value)
        self._iext_widget.blockSignals(False)
    
    def refresh(self):
        """Refresh the form from the config."""
        # Update soma diameter from config in case it changed
        from core.models import FullModelConfig
        if hasattr(self.stim_config, '__dict__'):
            # Try to get the parent config to access morphology
            # This is a workaround since we only have stim_config
            pass
        self.form_stim.refresh()
        self._find_and_hook_iext_widget()
        self._update_display_value()
    
    def update_soma_diameter(self, diameter_cm: float):
        """Update soma diameter and recalculate area."""
        self.soma_diameter_cm = diameter_cm
        self.soma_area_cm2 = np.pi * (diameter_cm ** 2)
        self.unit_toggle.set_soma_area(self.soma_area_cm2)
        self._update_display_value()

    def set_priority_filter(self, priority: str) -> None:
        if hasattr(self.form_stim, "set_priority_filter"):
            self.form_stim.set_priority_filter(priority)

    def set_search_filter(self, text: str) -> None:
        if hasattr(self.form_stim, "set_search_filter"):
            self.form_stim.set_search_filter(text)
