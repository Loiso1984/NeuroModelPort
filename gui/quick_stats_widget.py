"""
gui/quick_stats_widget.py - Quick Simulation Stats Widget v11.7

Displays key simulation metrics in a compact, always-visible panel.
Provides immediate feedback to researchers without switching to Analytics tab.

Features:
- Real-time spike count, firing rate, min/max voltage
- ATP level indicator (when metabolic simulation active)
- Quick export button
- Bilingual support
"""

from typing import Optional, Dict, Any
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QPushButton, QFrame, QToolTip
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from gui.locales import T


class QuickStatsWidget(QFrame):
    """Compact panel showing key simulation statistics."""
    
    export_requested = Signal()  # Emitted when user clicks export
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.setStyleSheet("""
            QFrame {
                background: #1E1E2E;
                border: 1px solid #313244;
                border-radius: 6px;
            }
            QLabel {
                color: #CDD6F4;
            }
            QLabel#value_label {
                color: #A6E3A1;
                font-weight: bold;
                font-size: 12px;
            }
            QLabel#alert_label {
                color: #F38BA8;
                font-weight: bold;
            }
            QLabel#warning_label {
                color: #F9E2AF;
                font-weight: bold;
            }
        """)
        
        self._setup_ui()
        self._reset_stats()
    
    def _setup_ui(self):
        """Create the stats display layout."""
        self._layout = QHBoxLayout(self)
        self._layout.setSpacing(12)
        self._layout.setContentsMargins(10, 6, 10, 6)
        
        # Spike count
        self._add_stat("spikes", T.tr("stat_spikes", "Spikes:"), "0")
        
        # Firing rate
        self._add_stat("rate", T.tr("stat_rate", "Rate:"), "0 Hz")
        
        # Min voltage
        self._add_stat("vmin", T.tr("stat_vmin", "Vmin:"), "-70 mV")
        
        # Max voltage
        self._add_stat("vmax", T.tr("stat_vmax", "Vmax:"), "+40 mV")
        
        # ATP level (hidden by default)
        self._atp_container, self._atp_label = self._add_stat(
            "atp", T.tr("stat_atp", "ATP:"), "2.0 mM", visible=False
        )
        
        # Separator
        self._layout.addSpacing(20)
        
        # Status indicator
        self._status_label = QLabel(T.tr("stat_ready", "Ready"))
        self._status_label.setStyleSheet("color: #6C7086; font-style: italic;")
        self._layout.addWidget(self._status_label)
        
        self._layout.addStretch()
        
        # Quick export button
        self._export_btn = QPushButton("📊 " + T.tr("btn_quick_export", "Quick Export"))
        self._export_btn.setStyleSheet("""
            QPushButton {
                background: #313244;
                color: #89B4FA;
                border: 1px solid #45475A;
                border-radius: 4px;
                padding: 3px 10px;
                font-size: 11px;
            }
            QPushButton:hover { background: #45475A; }
            QPushButton:disabled { color: #6C7086; }
        """)
        self._export_btn.setToolTip(T.tr("tt_quick_export", "Export current results to CSV"))
        self._export_btn.clicked.connect(self.export_requested.emit)
        self._export_btn.setEnabled(False)
        self._layout.addWidget(self._export_btn)
    
    def _add_stat(self, name: str, label: str, default: str, visible: bool = True):
        """Add a stat field to the layout."""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setSpacing(4)
        layout.setContentsMargins(0, 0, 0, 0)
        
        name_label = QLabel(label)
        name_label.setStyleSheet("color: #6C7086; font-size: 11px;")
        
        value_label = QLabel(default)
        value_label.setObjectName("value_label")
        value_label.setToolTip(T.tr(f"tt_stat_{name}", f"Click to copy {name}"))
        
        layout.addWidget(name_label)
        layout.addWidget(value_label)
        
        if visible:
            self._layout.addWidget(container)
        
        setattr(self, f"_{name}_value", value_label)
        return container, value_label
    
    def _reset_stats(self):
        """Reset all stats to default values."""
        self._spikes_value.setText("0")
        self._rate_value.setText("0 Hz")
        self._vmin_value.setText("-70 mV")
        self._vmax_value.setText("+40 mV")
        self._atp_label.setText("2.0 mM")
        self._atp_container.setVisible(False)
        self._status_label.setText(T.tr("stat_ready", "Ready"))
        self._export_btn.setEnabled(False)
        
        # Reset styles
        self._rate_value.setObjectName("value_label")
        self._atp_label.setObjectName("value_label")
        self._rate_value.setStyleSheet("")
        self._atp_label.setStyleSheet("")
    
    def update_from_result(self, result: Any, stats: Optional[Dict[str, Any]] = None):
        """Update stats from simulation result."""
        if result is None:
            self._reset_stats()
            return
        
        # Spike count
        n_spikes = getattr(result, 'n_spikes', 0)
        if n_spikes == 0 and hasattr(result, 'spike_times'):
            n_spikes = len(result.spike_times)
        self._spikes_value.setText(str(n_spikes))
        
        # Voltage stats
        if hasattr(result, 'v_soma') and len(result.v_soma) > 0:
            v = np.asarray(result.v_soma)
            vmin = np.min(v)
            vmax = np.max(v)
            self._vmin_value.setText(f"{vmin:.1f} mV")
            self._vmax_value.setText(f"{vmax:.1f} mV")
        
        # Firing rate
        rate_hz = 0.0
        if stats:
            rate_hz = stats.get('f_initial_hz', 0) or stats.get('f_steady_hz', 0)
        elif hasattr(result, 'firing_rate_hz'):
            rate_hz = result.firing_rate_hz
        
        self._rate_value.setText(f"{rate_hz:.1f} Hz")
        
        # Color code firing rate
        if rate_hz > 100:
            self._rate_value.setObjectName("alert_label")
            self._rate_value.setStyleSheet("color: #F38BA8; font-weight: bold; font-size: 12px;")
        elif rate_hz > 50:
            self._rate_value.setObjectName("warning_label")
            self._rate_value.setStyleSheet("color: #F9E2AF; font-weight: bold; font-size: 12px;")
        else:
            self._rate_value.setObjectName("value_label")
            self._rate_value.setStyleSheet("color: #A6E3A1; font-weight: bold; font-size: 12px;")
        
        # ATP level
        if hasattr(result, 'atp_level') and result.atp_level is not None:
            atp_arr = np.asarray(result.atp_level)
            atp_min = np.min(atp_arr)
            self._atp_label.setText(f"{atp_min:.2f} mM")
            self._atp_container.setVisible(True)
            
            # Color code ATP
            if atp_min < 0.2:
                self._atp_label.setObjectName("alert_label")
                self._atp_label.setStyleSheet("color: #F38BA8; font-weight: bold; font-size: 12px;")
            elif atp_min < 0.5:
                self._atp_label.setObjectName("warning_label")
                self._atp_label.setStyleSheet("color: #F9E2AF; font-weight: bold; font-size: 12px;")
            else:
                self._atp_label.setObjectName("value_label")
                self._atp_label.setStyleSheet("color: #A6E3A1; font-weight: bold; font-size: 12px;")
        
        # Status
        if n_spikes > 0:
            self._status_label.setText(T.tr("stat_active", "Active"))
            self._status_label.setStyleSheet("color: #A6E3A1; font-style: italic;")
        else:
            self._status_label.setText(T.tr("stat_silent", "Silent"))
            self._status_label.setStyleSheet("color: #F9E2AF; font-style: italic;")
        
        self._export_btn.setEnabled(True)
    
    def set_computing(self):
        """Show computing state."""
        self._status_label.setText(T.tr("status_computing", "Computing..."))
        self._status_label.setStyleSheet("color: #89B4FA; font-style: italic;")
        self._export_btn.setEnabled(False)
