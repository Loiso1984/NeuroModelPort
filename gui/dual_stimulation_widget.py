"""
gui/dual_stimulation_widget.py - Dual Stimulation GUI Widget

Provides user interface for configuring dual stimulation scenarios.
Includes preset selection, parameter configuration, and real-time preview.

Features:
- Dual preset selection with descriptions
- Independent configuration for primary/secondary stimuli
- Visual timing diagram
- Biological parameter validation
- Real-time E/I balance display
"""

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox, QTextEdit,
    QGridLayout, QFrame, QPushButton
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from core.dual_stimulation_presets import (
    get_dual_preset_names, apply_dual_preset, get_preset_description,
    validate_dual_preset
)
from core.dual_stimulation import DualStimulationConfig


class DualStimulationWidget(QWidget):
    """Widget for configuring dual stimulation parameters."""
    
    # Signal emitted when configuration changes
    config_changed = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = DualStimulationConfig()
        self.setup_ui()
        self.load_default_preset()
    
    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # ── Preset Selection ─────────────────────────────────────
        preset_group = QGroupBox("Dual Stimulation Presets")
        preset_layout = QVBoxLayout(preset_group)
        
        preset_select_layout = QHBoxLayout()
        preset_select_layout.addWidget(QLabel("Preset:"))
        self.combo_presets = QComboBox()
        self.combo_presets.addItems(["— Select preset —"] + get_dual_preset_names())
        self.combo_presets.currentTextChanged.connect(self.on_preset_changed)
        preset_select_layout.addWidget(self.combo_presets)
        
        self.btn_load_preset = QPushButton("Load")
        self.btn_load_preset.clicked.connect(self.load_preset)
        preset_select_layout.addWidget(self.btn_load_preset)
        preset_select_layout.addStretch()
        
        preset_layout.addLayout(preset_select_layout)
        
        self.txt_description = QTextEdit()
        self.txt_description.setMaximumHeight(80)
        self.txt_description.setReadOnly(True)
        self.txt_description.setStyleSheet("""
            QTextEdit {
                background: #2A2A2A; color: #A0A0A0;
                border: 1px solid #45475A; border-radius: 4px;
                padding: 8px; font-family: monospace; font-size: 11px;
            }
        """)
        preset_layout.addWidget(self.txt_description)
        
        layout.addWidget(preset_group)
        
        # ── Primary Stimulus ───────────────────────────────────────
        primary_group = QGroupBox("Primary Stimulus (Excitatory)")
        primary_layout = QGridLayout(primary_group)
        
        # Location and type
        primary_layout.addWidget(QLabel("Location:"), 0, 0)
        self.combo_primary_location = QComboBox()
        self.combo_primary_location.addItems(["soma", "ais", "dendritic_filtered"])
        self.combo_primary_location.currentTextChanged.connect(self.on_config_changed)
        primary_layout.addWidget(self.combo_primary_location, 0, 1)
        
        primary_layout.addWidget(QLabel("Type:"), 0, 2)
        self.combo_primary_type = QComboBox()
        self.combo_primary_type.addItems(["const", "pulse", "alpha", "ou_noise", "AMPA", "NMDA", "GABAA", "GABAB", "Kainate", "Nicotinic", "zap"])
        self.combo_primary_type.currentTextChanged.connect(self.on_config_changed)
        primary_layout.addWidget(self.combo_primary_type, 0, 3)
        
        # Current and timing
        primary_layout.addWidget(QLabel("Current (µA/cm²):"), 1, 0)
        self.spin_primary_current = QDoubleSpinBox()
        self.spin_primary_current.setRange(-100, 100)
        self.spin_primary_current.setSingleStep(0.5)
        self.spin_primary_current.setDecimals(1)
        self.spin_primary_current.valueChanged.connect(self.on_config_changed)
        primary_layout.addWidget(self.spin_primary_current, 1, 1)
        
        primary_layout.addWidget(QLabel("Start (ms):"), 1, 2)
        self.spin_primary_start = QDoubleSpinBox()
        self.spin_primary_start.setRange(0, 200)
        self.spin_primary_start.setSingleStep(1)
        self.spin_primary_start.valueChanged.connect(self.on_config_changed)
        primary_layout.addWidget(self.spin_primary_start, 1, 3)
        
        primary_layout.addWidget(QLabel("Duration (ms):"), 2, 0)
        self.spin_primary_duration = QDoubleSpinBox()
        self.spin_primary_duration.setRange(0.1, 100)
        self.spin_primary_duration.setSingleStep(0.5)
        self.spin_primary_duration.valueChanged.connect(self.on_config_changed)
        primary_layout.addWidget(self.spin_primary_duration, 2, 1)
        
        primary_layout.addWidget(QLabel("Alpha τ (ms):"), 2, 2)
        self.spin_primary_alpha = QDoubleSpinBox()
        self.spin_primary_alpha.setRange(0.1, 20)
        self.spin_primary_alpha.setSingleStep(0.1)
        self.spin_primary_alpha.setDecimals(1)
        self.spin_primary_alpha.valueChanged.connect(self.on_config_changed)
        primary_layout.addWidget(self.spin_primary_alpha, 2, 3)
        
        # Event times
        primary_layout.addWidget(QLabel("Event Times (ms):"), 3, 0)
        from PySide6.QtWidgets import QLineEdit
        self.line_primary_event_times = QLineEdit()
        self.line_primary_event_times.setPlaceholderText("e.g., 10, 20, 30")
        self.line_primary_event_times.setToolTip("Comma-separated event timestamps (ms). Overrides pulse_start for synaptic types.")
        self.line_primary_event_times.textChanged.connect(self.on_config_changed)
        primary_layout.addWidget(self.line_primary_event_times, 3, 1, 1, 3)
        
        layout.addWidget(primary_group)
        
        # ── Secondary Stimulus ─────────────────────────────────────
        secondary_group = QGroupBox("Secondary Stimulus (Inhibitory)")
        secondary_layout = QGridLayout(secondary_group)
        
        # Location and type
        secondary_layout.addWidget(QLabel("Location:"), 0, 0)
        self.combo_secondary_location = QComboBox()
        self.combo_secondary_location.addItems(["soma", "ais", "dendritic_filtered"])
        self.combo_secondary_location.currentTextChanged.connect(self.on_config_changed)
        secondary_layout.addWidget(self.combo_secondary_location, 0, 1)
        
        secondary_layout.addWidget(QLabel("Type:"), 0, 2)
        self.combo_secondary_type = QComboBox()
        self.combo_secondary_type.addItems(["const", "pulse", "alpha", "ou_noise", "AMPA", "NMDA", "GABAA", "GABAB", "Kainate", "Nicotinic", "zap"])
        self.combo_secondary_type.currentTextChanged.connect(self.on_config_changed)
        secondary_layout.addWidget(self.combo_secondary_type, 0, 3)
        
        # Current and timing
        secondary_layout.addWidget(QLabel("Current (µA/cm²):"), 1, 0)
        self.spin_secondary_current = QDoubleSpinBox()
        self.spin_secondary_current.setRange(-100, 100)
        self.spin_secondary_current.setSingleStep(0.5)
        self.spin_secondary_current.setDecimals(1)
        self.spin_secondary_current.valueChanged.connect(self.on_config_changed)
        secondary_layout.addWidget(self.spin_secondary_current, 1, 1)
        
        secondary_layout.addWidget(QLabel("Start (ms):"), 1, 2)
        self.spin_secondary_start = QDoubleSpinBox()
        self.spin_secondary_start.setRange(0, 200)
        self.spin_secondary_start.setSingleStep(1)
        self.spin_secondary_start.valueChanged.connect(self.on_config_changed)
        secondary_layout.addWidget(self.spin_secondary_start, 1, 3)
        
        secondary_layout.addWidget(QLabel("Duration (ms):"), 2, 0)
        self.spin_secondary_duration = QDoubleSpinBox()
        self.spin_secondary_duration.setRange(0.1, 200)
        self.spin_secondary_duration.setSingleStep(0.5)
        self.spin_secondary_duration.valueChanged.connect(self.on_config_changed)
        secondary_layout.addWidget(self.spin_secondary_duration, 2, 1)
        
        secondary_layout.addWidget(QLabel("Alpha τ (ms):"), 2, 2)
        self.spin_secondary_alpha = QDoubleSpinBox()
        self.spin_secondary_alpha.setRange(0.1, 20)
        self.spin_secondary_alpha.setSingleStep(0.1)
        self.spin_secondary_alpha.setDecimals(1)
        self.spin_secondary_alpha.valueChanged.connect(self.on_config_changed)
        secondary_layout.addWidget(self.spin_secondary_alpha, 2, 3)
        
        # Event times
        secondary_layout.addWidget(QLabel("Event Times (ms):"), 3, 0)
        self.line_secondary_event_times = QLineEdit()
        self.line_secondary_event_times.setPlaceholderText("e.g., 30, 40, 50")
        self.line_secondary_event_times.setToolTip("Comma-separated event timestamps (ms). Overrides pulse_start for synaptic types.")
        self.line_secondary_event_times.textChanged.connect(self.on_config_changed)
        secondary_layout.addWidget(self.line_secondary_event_times, 3, 1, 1, 3)
        
        # Train generator (ephemeral)
        secondary_layout.addWidget(QLabel("Train Type:"), 4, 0)
        self.combo_secondary_train_type = QComboBox()
        self.combo_secondary_train_type.addItems(["none", "regular", "poisson"])
        self.combo_secondary_train_type.setToolTip("Auto-generate spike train (ephemeral, does not mutate manual event_times)")
        self.combo_secondary_train_type.currentTextChanged.connect(self.on_config_changed)
        secondary_layout.addWidget(self.combo_secondary_train_type, 4, 1)
        
        secondary_layout.addWidget(QLabel("Train Freq (Hz):"), 4, 2)
        self.spin_secondary_train_freq = QDoubleSpinBox()
        self.spin_secondary_train_freq.setRange(0.1, 500.0)
        self.spin_secondary_train_freq.setSingleStep(1.0)
        self.spin_secondary_train_freq.setDecimals(1)
        self.spin_secondary_train_freq.setValue(40.0)
        self.spin_secondary_train_freq.setToolTip("Spike frequency for auto-generated train")
        self.spin_secondary_train_freq.valueChanged.connect(self.on_config_changed)
        secondary_layout.addWidget(self.spin_secondary_train_freq, 4, 3)
        
        secondary_layout.addWidget(QLabel("Train Dur (ms):"), 5, 0)
        self.spin_secondary_train_duration = QDoubleSpinBox()
        self.spin_secondary_train_duration.setRange(1.0, 5000.0)
        self.spin_secondary_train_duration.setSingleStep(10.0)
        self.spin_secondary_train_duration.setValue(200.0)
        self.spin_secondary_train_duration.setToolTip("Duration of auto-generated train")
        self.spin_secondary_train_duration.valueChanged.connect(self.on_config_changed)
        secondary_layout.addWidget(self.spin_secondary_train_duration, 5, 1, 1, 3)
        
        layout.addWidget(secondary_group)
        
        # ── Dendritic Parameters ───────────────────────────────────
        dendritic_group = QGroupBox("Dendritic Parameters (Secondary)")
        dendritic_layout = QGridLayout(dendritic_group)
        
        dendritic_layout.addWidget(QLabel("Distance (µm):"), 0, 0)
        self.spin_dendritic_distance = QDoubleSpinBox()
        self.spin_dendritic_distance.setRange(10, 500)
        self.spin_dendritic_distance.setSingleStep(10)
        self.spin_dendritic_distance.valueChanged.connect(self.on_config_changed)
        dendritic_layout.addWidget(self.spin_dendritic_distance, 0, 1)
        
        dendritic_layout.addWidget(QLabel("Space Constant (µm):"), 0, 2)
        self.spin_dendritic_lambda = QDoubleSpinBox()
        self.spin_dendritic_lambda.setRange(50, 300)
        self.spin_dendritic_lambda.setSingleStep(10)
        self.spin_dendritic_lambda.valueChanged.connect(self.on_config_changed)
        dendritic_layout.addWidget(self.spin_dendritic_lambda, 0, 3)
        
        dendritic_layout.addWidget(QLabel("Filter τ (ms):"), 1, 0)
        self.spin_dendritic_tau = QDoubleSpinBox()
        self.spin_dendritic_tau.setRange(1, 50)
        self.spin_dendritic_tau.setSingleStep(1)
        self.spin_dendritic_tau.valueChanged.connect(self.on_config_changed)
        dendritic_layout.addWidget(self.spin_dendritic_tau, 1, 1)
        
        # Attenuation display
        dendritic_layout.addWidget(QLabel("Attenuation:"), 1, 2)
        self.lbl_attenuation = QLabel("0.000")
        self.lbl_attenuation.setStyleSheet("font-weight: bold; color: #89DCEB;")
        dendritic_layout.addWidget(self.lbl_attenuation, 1, 3)
        
        layout.addWidget(dendritic_group)
        
        # ── Status Display ─────────────────────────────────────────
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        
        # E/I Balance
        ei_layout = QHBoxLayout()
        ei_layout.addWidget(QLabel("E/I Ratio:"))
        self.lbl_ei_ratio = QLabel("0.00")
        self.lbl_ei_ratio.setStyleSheet("""
            QLabel {
                font-weight: bold; font-size: 14px; padding: 4px;
                background: #2A2A2A; border-radius: 4px;
                color: #A0A0A0;
            }
        """)
        ei_layout.addWidget(self.lbl_ei_ratio)
        ei_layout.addStretch()
        status_layout.addLayout(ei_layout)
        
        # Validation status
        self.lbl_validation = QLabel("✓ Configuration valid")
        self.lbl_validation.setStyleSheet("color: #A6E3A1; font-weight: bold;")
        status_layout.addWidget(self.lbl_validation)
        
        layout.addWidget(status_group)
        
        # ── Enable Checkbox ─────────────────────────────────────────
        self.check_enabled = QCheckBox("Enable Dual Stimulation")
        self.check_enabled.stateChanged.connect(self.on_enabled_changed)
        layout.addWidget(self.check_enabled)
        
        layout.addStretch()
    
    def on_preset_changed(self, preset_name):
        """Handle preset selection change."""
        if preset_name and preset_name != "— Select preset —":
            description = get_preset_description(preset_name)
            self.txt_description.setPlainText(description)
    
    def load_preset(self):
        """Load selected preset — single atomic update, no signal cascade."""
        preset_name = self.combo_presets.currentText()
        if not preset_name or preset_name == "— Select preset —":
            return
        apply_dual_preset(self.config, preset_name)
        self._refresh_ui()          # block signals, then emit once

    def load_default_preset(self):
        """Reset to safe default: dual-stim disabled until user explicitly enables it."""
        self.config = DualStimulationConfig()
        # Block the combo signal so it doesn't trigger on_preset_changed -> description
        self.combo_presets.blockSignals(True)
        self.combo_presets.setCurrentText("— Select preset —")
        self.combo_presets.blockSignals(False)
        self.txt_description.setPlainText(
            "Dual stimulation is disabled by default.\n"
            "Select a dual preset and click Load, then enable the checkbox if needed."
        )
        self._refresh_ui()

    def _refresh_ui(self):
        """
        Push config → all widgets with signals blocked, then do one emit.
        Prevents the N-widget-change cascade that caused multiple Load fires.
        """
        _all_widgets = [
            self.combo_primary_location, self.combo_primary_type,
            self.spin_primary_current, self.spin_primary_start,
            self.spin_primary_duration, self.spin_primary_alpha,
            self.line_primary_event_times,
            self.combo_secondary_location, self.combo_secondary_type,
            self.spin_secondary_current, self.spin_secondary_start,
            self.spin_secondary_duration, self.spin_secondary_alpha,
            self.line_secondary_event_times,
            self.combo_secondary_train_type, self.spin_secondary_train_freq,
            self.spin_secondary_train_duration,
            self.spin_dendritic_distance, self.spin_dendritic_lambda,
            self.spin_dendritic_tau, self.check_enabled,
        ]
        for w in _all_widgets:
            w.blockSignals(True)

        # Primary stimulus
        self.combo_primary_location.setCurrentText(self.config.primary_location)
        self.combo_primary_type.setCurrentText(self.config.primary_stim_type)
        self.spin_primary_current.setValue(self.config.primary_Iext)
        self.spin_primary_start.setValue(self.config.primary_start)
        self.spin_primary_duration.setValue(self.config.primary_duration)
        self.spin_primary_alpha.setValue(self.config.primary_alpha_tau)
        if self.config.primary_event_times and len(self.config.primary_event_times) > 0:
            self.line_primary_event_times.setText(", ".join(str(x) for x in self.config.primary_event_times)
            )
        else:
            self.line_primary_event_times.setText("")

        # Secondary stimulus
        self.combo_secondary_location.setCurrentText(self.config.secondary_location)
        self.combo_secondary_type.setCurrentText(self.config.secondary_stim_type)
        self.spin_secondary_current.setValue(self.config.secondary_Iext)
        self.spin_secondary_start.setValue(self.config.secondary_start)
        self.spin_secondary_duration.setValue(self.config.secondary_duration)
        self.spin_secondary_alpha.setValue(self.config.secondary_alpha_tau)
        if self.config.secondary_event_times and len(self.config.secondary_event_times) > 0:
            self.line_secondary_event_times.setText(", ".join(str(x) for x in self.config.secondary_event_times)
            )
        else:
            self.line_secondary_event_times.setText("")
        
        # Train generator
        self.combo_secondary_train_type.setCurrentText(self.config.secondary_train_type)
        self.spin_secondary_train_freq.setValue(self.config.secondary_train_freq_hz)
        self.spin_secondary_train_duration.setValue(self.config.secondary_train_duration_ms)

        # Dendritic parameters
        self.spin_dendritic_distance.setValue(self.config.secondary_distance_um)
        self.spin_dendritic_lambda.setValue(self.config.secondary_space_constant_um)
        self.spin_dendritic_tau.setValue(self.config.secondary_tau_dendritic_ms)

        # Enable checkbox
        self.check_enabled.setChecked(self.config.enabled)

        for w in _all_widgets:
            w.blockSignals(False)

        self.update_displays()
        self.config_changed.emit()

    def update_ui_from_config(self):
        """Backward-compat alias → _refresh_ui."""
        self._refresh_ui()
    
    def update_config_from_ui(self):
        """Update configuration from UI elements."""
        # Primary stimulus
        self.config.primary_location = self.combo_primary_location.currentText()
        self.config.primary_stim_type = self.combo_primary_type.currentText()
        self.config.primary_Iext = self.spin_primary_current.value()
        self.config.primary_start = self.spin_primary_start.value()
        self.config.primary_duration = self.spin_primary_duration.value()
        self.config.primary_alpha_tau = self.spin_primary_alpha.value()
        
        # Parse primary event times
        event_text = self.line_primary_event_times.text().strip()
        if event_text:
            try:
                self.config.primary_event_times = [float(x.strip()) for x in event_text.split(",") if x.strip()]
                self.line_primary_event_times.setStyleSheet("")  # Clear error style
            except ValueError:
                self.config.primary_event_times = []
                self.line_primary_event_times.setStyleSheet("border: 2px solid #F38BA8;")  # Red border for error
        else:
            self.config.primary_event_times = []
            self.line_primary_event_times.setStyleSheet("")  # Clear error style
        
        # Secondary stimulus
        self.config.secondary_location = self.combo_secondary_location.currentText()
        self.config.secondary_stim_type = self.combo_secondary_type.currentText()
        self.config.secondary_Iext = self.spin_secondary_current.value()
        self.config.secondary_start = self.spin_secondary_start.value()
        self.config.secondary_duration = self.spin_secondary_duration.value()
        self.config.secondary_alpha_tau = self.spin_secondary_alpha.value()
        
        # Parse secondary event times
        event_text = self.line_secondary_event_times.text().strip()
        if event_text:
            try:
                self.config.secondary_event_times = [float(x.strip()) for x in event_text.split(",") if x.strip()]
                self.line_secondary_event_times.setStyleSheet("")  # Clear error style
            except ValueError:
                self.config.secondary_event_times = []
                self.line_secondary_event_times.setStyleSheet("border: 2px solid #F38BA8;")  # Red border for error
        else:
            self.config.secondary_event_times = []
            self.line_secondary_event_times.setStyleSheet("")  # Clear error style
        
        # Train generator
        self.config.secondary_train_type = self.combo_secondary_train_type.currentText()
        self.config.secondary_train_freq_hz = self.spin_secondary_train_freq.value()
        self.config.secondary_train_duration_ms = self.spin_secondary_train_duration.value()
        
        # Dendritic parameters
        self.config.secondary_distance_um = self.spin_dendritic_distance.value()
        self.config.secondary_space_constant_um = self.spin_dendritic_lambda.value()
        self.config.secondary_tau_dendritic_ms = self.spin_dendritic_tau.value()
        
        self.config.enabled = self.check_enabled.isChecked()
    
    def update_displays(self):
        """Update status displays."""
        # Calculate attenuation
        if self.config.secondary_location == "dendritic_filtered":
            atten = np.exp(-self.config.secondary_distance_um / self.config.secondary_space_constant_um)
            self.lbl_attenuation.setText(f"{atten:.3f}")
        else:
            self.lbl_attenuation.setText("N/A")
        
        # Calculate E/I ratio
        excitation = abs(self.config.primary_Iext)
        inhibition = 0.0
        
        if self.config.secondary_stim_type in ["GABAA", "GABAB"]:
            inhibition = abs(self.config.secondary_Iext)
        elif self.config.primary_stim_type in ["GABAA", "GABAB"]:
            inhibition = abs(self.config.primary_Iext)
            excitation = abs(self.config.secondary_Iext)
        
        if inhibition > 0:
            ei_ratio = excitation / inhibition
            self.lbl_ei_ratio.setText(f"{ei_ratio:.2f}")
            
            # Color code based on physiological range
            if 0.5 <= ei_ratio <= 3.0:
                self.lbl_ei_ratio.setStyleSheet("""
                    QLabel {
                        font-weight: bold; font-size: 14px; padding: 4px;
                        background: #2A2A2A; border-radius: 4px;
                        color: #A6E3A1;
                    }
                """)
            else:
                self.lbl_ei_ratio.setStyleSheet("""
                    QLabel {
                        font-weight: bold; font-size: 14px; padding: 4px;
                        background: #2A2A2A; border-radius: 4px;
                        color: #F38BA8;
                    }
                """)
        else:
            self.lbl_ei_ratio.setText("N/A")
    
    def on_config_changed(self):
        """Handle any configuration change."""
        self.update_config_from_ui()
        self.update_displays()
        self.config_changed.emit()
    
    def on_enabled_changed(self):
        """Handle enable/disable checkbox."""
        self.config.enabled = self.check_enabled.isChecked()
        self.config_changed.emit()
    
    def get_config(self) -> DualStimulationConfig:
        """Get current dual stimulation configuration."""
        return self.config


if __name__ == '__main__':
    import sys
    from PySide6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    widget = DualStimulationWidget()
    widget.show()
    sys.exit(app.exec())
