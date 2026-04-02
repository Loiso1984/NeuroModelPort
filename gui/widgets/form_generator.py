from PySide6.QtWidgets import (
    QWidget, QFormLayout, QDoubleSpinBox, QSpinBox,
    QCheckBox, QComboBox, QGroupBox, QVBoxLayout
)
from PySide6.QtCore import Qt
from typing import Literal, get_args, get_origin
from pydantic import BaseModel


class PydanticFormWidget(QWidget):
    def __init__(self, pydantic_instance: BaseModel, title: str = "", parent=None):
        super().__init__(parent)
        self.instance = pydantic_instance
        self.widgets_map = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.group_box = QGroupBox(title)
        self.form_layout = QFormLayout(self.group_box)
        layout.addWidget(self.group_box)
        self._build_form()

    def _build_form(self):
        for field_name, field_info in self.instance.model_fields.items():
            field_type = field_info.annotation
            val = getattr(self.instance, field_name)
            origin = get_origin(field_type)

            if field_type is bool:
                w = QCheckBox()
                w.setChecked(val)
                w.toggled.connect(lambda v, n=field_name: setattr(self.instance, n, v))
            elif origin is Literal:
                w = QComboBox()
                w.addItems([str(c) for c in get_args(field_type)])
                w.setCurrentText(str(val))
                w.currentTextChanged.connect(lambda v, n=field_name: setattr(self.instance, n, v))
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
                w.valueChanged.connect(lambda v, n=field_name: setattr(self.instance, n, v))
            else:
                continue

            w.setToolTip(field_info.description or field_name)
            self.form_layout.addRow(field_name, w)
            self.widgets_map[field_name] = w

    def refresh(self):
        """Sync widgets from model (needed after preset load)."""
        from gui.locales import T
        for name, widget in self.widgets_map.items():
            val = getattr(self.instance, name)
            widget.blockSignals(True)
            if isinstance(widget, QCheckBox):
                widget.setChecked(val)
            elif isinstance(widget, QComboBox):
                widget.setCurrentText(str(val))
            else:
                widget.setValue(val)
            tooltip = T.desc(name)
            if tooltip and tooltip != name:
                widget.setToolTip(tooltip)
            widget.blockSignals(False)
