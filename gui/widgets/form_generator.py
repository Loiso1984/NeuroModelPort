from PySide6.QtWidgets import (
    QWidget, QFormLayout, QDoubleSpinBox, QSpinBox,
    QCheckBox, QComboBox, QGroupBox, QVBoxLayout, QLabel, QLineEdit
)
from PySide6.QtCore import Qt
from typing import Literal, get_args, get_origin, Iterable
import logging
from pydantic import BaseModel


class PydanticFormWidget(QWidget):
    def __init__(self, pydantic_instance: BaseModel, title: str = "", parent=None, on_change=None, hidden_fields: Iterable[str] | None = None):
        super().__init__(parent)
        self.instance = pydantic_instance
        self.on_change = on_change
        self.widgets_map = {}
        self.labels_map = {}
        self.hidden_fields = set(hidden_fields or ())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.group_box = QGroupBox(title)
        self.form_layout = QFormLayout(self.group_box)
        layout.addWidget(self.group_box)
        self._build_form()

    def _build_form(self):
        for field_name, field_info in self.instance.model_fields.items():
            if field_name in self.hidden_fields:
                continue
            field_type = field_info.annotation
            val = getattr(self.instance, field_name)
            origin = get_origin(field_type)

            if field_type is bool:
                w = QCheckBox()
                w.setChecked(val)
                w.toggled.connect(lambda v, n=field_name: self._set_field(n, v))
            elif origin is Literal:
                w = QComboBox()
                w.addItems([str(c) for c in get_args(field_type)])
                w.setCurrentText(str(val))
                w.currentTextChanged.connect(lambda v, n=field_name: self._set_field(n, v))
            elif origin is list:
                w = QLineEdit()
                # Handle empty list case
                if val and len(val) > 0:
                    w.setText(", ".join(str(x) for x in val))
                else:
                    w.setText("")
                
                # Create a local closure to handle text parsing
                def text_edited(text_val, n=field_name, widget=w):
                    try:
                        # Parse comma-separated string back to List[float]
                        if text_val.strip():
                            parsed_list = [float(x.strip()) for x in text_val.split(",") if x.strip()]
                            # Validate parsed values are within reasonable range
                            if any(abs(x) > 1e6 for x in parsed_list):
                                raise ValueError(f"Values exceed reasonable range: {parsed_list}")
                        else:
                            parsed_list = []
                        # Only set field if validation passes
                        self._set_field(n, parsed_list)
                        # Clear any error styling
                        widget.setStyleSheet("")
                    except ValueError as e:
                        # Provide visual feedback for invalid input
                        widget.setStyleSheet("background-color: #ffe6e6; border: 1px solid #ff6666;")
                        logging.warning(f"Invalid input for field '{n}': {text_val}. Error: {e}")
                        # Don't set the field with invalid data validation passes
                    except Exception as e:
                        # Handle any other unexpected errors
                        widget.setStyleSheet("background-color: #ffe6e6; border: 1px solid #ff6666;")
                        logging.error(f"Unexpected error processing field '{n}': {e}")
                        # Don't set the field with invalid data
                        
                w.textChanged.connect(text_edited)
            elif field_type in (int, float):
                if field_type is int:
                    w = QSpinBox()
                    w.setRange(0, 1000000)
                else:
                    w = QDoubleSpinBox()
                    w.setRange(-5000.0, 5000.0)
                    w.setDecimals(6)
                    w.setSingleStep(0.1 if abs(val) > 0.1 else 0.0001)
                w.setValue(val)
                w.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
                w.setKeyboardTracking(True)
                w.valueChanged.connect(lambda v, n=field_name: self._set_field(n, v))
            else:
                continue

            w.setToolTip(field_info.description or field_name)
            label = QLabel(field_name)
            self.form_layout.addRow(label, w)
            self.labels_map[field_name] = label
            self.widgets_map[field_name] = w

    def _set_field(self, field_name, value):
        setattr(self.instance, field_name, value)
        if self.on_change is not None:
            self.on_change(field_name, value)

    def refresh(self):
        """Sync widgets from model (needed after preset load)."""
        from gui.locales import T
        for name, widget in self.widgets_map.items():
            try:
                val = getattr(self.instance, name)
                widget.blockSignals(True)
                try:
                    if isinstance(widget, QCheckBox):
                        widget.setChecked(val)
                    elif isinstance(widget, QComboBox):
                        widget.setCurrentText(str(val))
                    elif isinstance(widget, QLineEdit):
                        if isinstance(val, list):
                            if val and len(val) > 0:
                                widget.setText(", ".join(str(x) for x in val))
                            else:
                                widget.setText("")
                        else:
                            widget.setText(str(val))
                    else:
                        widget.setValue(val)
                    tooltip = T.desc(name)
                    if tooltip and tooltip != name:
                        widget.setToolTip(tooltip)
                finally:
                    widget.blockSignals(False)
            except Exception as e:
                logging.error(f"Error refreshing widget for field '{name}': {e}")
                # Ensure signals are re-enabled even if there's an error
                widget.blockSignals(False)
        # Safety check for hidden_fields set
        if not isinstance(self.hidden_fields, set):
            self.hidden_fields = set(self.hidden_fields)
