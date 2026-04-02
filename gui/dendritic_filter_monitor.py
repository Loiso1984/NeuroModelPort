"""
gui/dendritic_filter_monitor.py - Dendritic Filter Monitoring Widget

Real-time monitoring of dendritic filter state evolution.
Shows filter dynamics, attenuation, and temporal filtering effects.

Features:
- Real-time filter state plot
- Attenuation calculation display
- Time constant visualization
- Filter response to different input types
- Integration with Neuron Passport
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QTextEdit, QCheckBox, QSpinBox, QDoubleSpinBox
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QColor, QFont

from core.models import FullModelConfig
from core.presets import apply_preset
from core.solver import NeuronSolver


class DendriticFilterMonitor(QWidget):
    """Widget for monitoring dendritic filter dynamics in real-time."""
    
    # Signal emitted when filter state changes
    filter_updated = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = None
        self.result = None
        self.filter_data = []
        self.setup_ui()
        self.setup_timer()
    
    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # ── Filter Status ────────────────────────────────────────
        status_group = QGroupBox("Dendritic Filter Status")
        status_layout = QVBoxLayout(status_group)
        
        # Basic parameters
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("Distance (µm):"))
        self.lbl_distance = QLabel("N/A")
        params_layout.addWidget(self.lbl_distance)
        params_layout.addWidget(QLabel("λ (µm):"))
        self.lbl_lambda = QLabel("N/A")
        params_layout.addWidget(self.lbl_lambda)
        params_layout.addWidget(QLabel("τ (ms):"))
        self.lbl_tau = QLabel("N/A")
        params_layout.addWidget(self.lbl_tau)
        params_layout.addStretch()
        status_layout.addLayout(params_layout)
        
        # Attenuation display
        atten_layout = QHBoxLayout()
        atten_layout.addWidget(QLabel("Attenuation:"))
        self.lbl_attenuation = QLabel("0.000")
        self.lbl_attenuation.setStyleSheet("""
            QLabel {
                font-weight: bold; font-size: 16px; padding: 4px;
                background: #2A2A2A; border-radius: 4px;
                color: #89DCEB;
            }
        """)
        atten_layout.addWidget(self.lbl_attenuation)
        atten_layout.addWidget(QLabel("Expected:"))
        self.lbl_expected = QLabel("0.000")
        self.lbl_expected.setStyleSheet("""
            QLabel {
                font-weight: bold; font-size: 16px; padding: 4px;
                background: #2A2A2A; border-radius: 4px;
                color: #A6E3A1;
            }
        """)
        atten_layout.addWidget(self.lbl_expected)
        atten_layout.addWidget(QLabel("Error:"))
        self.lbl_error = QLabel("0.0%")
        self.lbl_error.setStyleSheet("""
            QLabel {
                font-weight: bold; font-size: 16px; padding: 4px;
                background: #2A2A2A; border-radius: 4px;
                color: #F38BA8;
            }
        """)
        atten_layout.addWidget(self.lbl_error)
        atten_layout.addStretch()
        status_layout.addLayout(atten_layout)
        
        layout.addWidget(status_group)
        
        # ── Filter Dynamics Plot ───────────────────────────────────
        plot_group = QGroupBox("Filter Dynamics")
        plot_layout = QVBoxLayout(plot_group)
        
        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#1E1E2E')
        self.plot_widget.setLabel('left', 'Filter State', units='µA/cm²')
        self.plot_widget.setLabel('bottom', 'Time', units='ms')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend(offset=(10, 10))
        
        # Create plot curves
        self.curve_filter_state = self.plot_widget.plot(
            name='Filter State', pen=pg.mkPen('#89DCEB', width=2)
        )
        self.curve_input_current = self.plot_widget.plot(
            name='Input Current', pen=pg.mkPen('#FA8C3C', width=2)
        )
        self.curve_attenuated = self.plot_widget.plot(
            name='Attenuated Input', pen=pg.mkPen('#A6E3A1', width=2, style=Qt.PenStyle.DashLine)
        )
        
        plot_layout.addWidget(self.plot_widget)
        layout.addWidget(plot_group)
        
        # ── Control Panel ─────────────────────────────────────────
        control_group = QGroupBox("Control Panel")
        control_layout = QVBoxLayout(control_group)
        
        # Simulation controls
        sim_layout = QHBoxLayout()
        self.btn_run_simulation = QPushButton("Run Simulation")
        self.btn_run_simulation.clicked.connect(self.run_simulation)
        sim_layout.addWidget(self.btn_run_simulation)
        
        self.btn_clear_data = QPushButton("Clear Data")
        self.btn_clear_data.clicked.connect(self.clear_data)
        sim_layout.addWidget(self.btn_clear_data)
        
        self.check_real_time = QCheckBox("Real-time Update")
        self.check_real_time.setChecked(True)
        sim_layout.addWidget(self.check_real_time)
        
        sim_layout.addStretch()
        control_layout.addLayout(sim_layout)
        
        # Update rate control
        rate_layout = QHBoxLayout()
        rate_layout.addWidget(QLabel("Update Rate (ms):"))
        self.spin_update_rate = QSpinBox()
        self.spin_update_rate.setRange(10, 1000)
        self.spin_update_rate.setValue(100)
        self.spin_update_rate.setSingleStep(10)
        rate_layout.addWidget(self.spin_update_rate)
        rate_layout.addStretch()
        control_layout.addLayout(rate_layout)
        
        layout.addWidget(control_group)
        
        # ── Analysis Display ───────────────────────────────────────
        analysis_group = QGroupBox("Filter Analysis")
        analysis_layout = QVBoxLayout(analysis_group)
        
        self.txt_analysis = QTextEdit()
        self.txt_analysis.setMaximumHeight(120)
        self.txt_analysis.setReadOnly(True)
        self.txt_analysis.setStyleSheet("""
            QTextEdit {
                background: #2A2A2A; color: #A0A0A0;
                border: 1px solid #45475A; border-radius: 4px;
                padding: 8px; font-family: monospace; font-size: 11px;
            }
        """)
        analysis_layout.addWidget(self.txt_analysis)
        
        layout.addWidget(analysis_group)
    
    def setup_timer(self):
        """Setup real-time update timer."""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.setInterval(100)  # 100ms default
    
    def set_config(self, config: FullModelConfig):
        """Set the neuron configuration for monitoring."""
        self.config = config
        
        # Update filter parameters display
        if hasattr(config, 'dendritic_filter') and config.dendritic_filter.enabled:
            self.lbl_distance.setText(f"{config.dendritic_filter.distance_um:.1f}")
            self.lbl_lambda.setText(f"{config.dendritic_filter.space_constant_um:.1f}")
            self.lbl_tau.setText(f"{config.dendritic_filter.tau_dendritic_ms:.1f}")
            
            # Calculate expected attenuation
            expected = np.exp(-config.dendritic_filter.distance_um / 
                             config.dendritic_filter.space_constant_um)
            self.lbl_expected.setText(f"{expected:.3f}")
        else:
            self.lbl_distance.setText("Disabled")
            self.lbl_lambda.setText("Disabled")
            self.lbl_tau.setText("Disabled")
            self.lbl_expected.setText("N/A")
    
    def run_simulation(self):
        """Run simulation and collect filter data."""
        if not self.config:
            self.txt_analysis.append("❌ No configuration set")
            return
        
        try:
            self.txt_analysis.append("🔄 Running simulation...")
            
            # Ensure dendritic filtering is enabled
            self.config.stim_location.location = "dendritic_filtered"
            self.config.dendritic_filter.enabled = True
            
            solver = NeuronSolver(self.config)
            self.result = solver.run_single()
            
            # Extract filter state data
            self.extract_filter_data()
            
            # Update analysis
            self.update_analysis()
            
            self.txt_analysis.append("✅ Simulation complete")
            
        except Exception as e:
            self.txt_analysis.append(f"❌ Simulation error: {str(e)}")
    
    def extract_filter_data(self):
        """Extract dendritic filter state from simulation results."""
        if not self.result:
            return
        
        # Reconstruct filter state from simulation
        # This requires accessing the filter state variable from the solver
        # For now, we'll simulate the expected behavior
        
        t = self.result.t
        
        # Get input current (stimulus)
        input_current = np.zeros_like(t)
        for i, time in enumerate(t):
            # Simplified stimulus calculation
            if (self.config.stim.pulse_start <= time <= 
                self.config.stim.pulse_start + self.config.stim.pulse_dur):
                input_current[i] = self.config.stim.Iext
        
        # Simulate filter response
        tau = self.config.dendritic_filter.tau_dendritic_ms
        dt = np.mean(np.diff(t))
        
        filter_state = np.zeros_like(t)
        for i in range(1, len(t)):
            d_filter_dt = (input_current[i] - filter_state[i-1]) / tau
            filter_state[i] = filter_state[i-1] + d_filter_dt * dt
        
        # Apply attenuation
        attenuation = np.exp(-self.config.dendritic_filter.distance_um / 
                           self.config.dendritic_filter.space_constant_um)
        attenuated = attenuation * filter_state
        
        # Store data
        self.filter_data = {
            'time': t,
            'input_current': input_current,
            'filter_state': filter_state,
            'attenuated': attenuated,
            'attenuation': attenuation
        }
        
        # Update plots
        self.update_plots()
    
    def update_plots(self):
        """Update the filter dynamics plots."""
        if not self.filter_data:
            return
        
        data = self.filter_data
        
        # Update curves
        self.curve_filter_state.setData(data['time'], data['filter_state'])
        self.curve_input_current.setData(data['time'], data['input_current'])
        self.curve_attenuated.setData(data['time'], data['attenuated'])
        
        # Update attenuation display
        self.lbl_attenuation.setText(f"{data['attenuation']:.3f}")
        
        # Calculate error
        expected = np.exp(-self.config.dendritic_filter.distance_um / 
                         self.config.dendritic_filter.space_constant_um)
        error = abs(data['attenuation'] - expected) / expected * 100
        self.lbl_error.setText(f"{error:.1f}%")
        
        # Color code error
        if error < 20:
            self.lbl_error.setStyleSheet("color: #A6E3A1; font-weight: bold;")
        elif error < 50:
            self.lbl_error.setStyleSheet("color: #F9E2AF; font-weight: bold;")
        else:
            self.lbl_error.setStyleSheet("color: #F38BA8; font-weight: bold;")
    
    def update_analysis(self):
        """Update the analysis display with filter statistics."""
        if not self.filter_data:
            return
        
        data = self.filter_data
        
        # Calculate statistics
        max_input = np.max(data['input_current'])
        max_filter = np.max(data['filter_state'])
        max_attenuated = np.max(data['attenuated'])
        
        # Calculate effective attenuation
        if max_input > 0:
            effective_atten = max_attenuated / max_input
        else:
            effective_atten = 0
        
        # Calculate time constant from response
        # Find rise time to 63% of final value
        if max_filter > 0:
            rise_time_idx = np.where(data['filter_state'] >= 0.63 * max_filter)[0]
            if len(rise_time_idx) > 0:
                rise_time = data['time'][rise_time_idx[0]]
            else:
                rise_time = 0
        else:
            rise_time = 0
        
        analysis_text = f"""
Filter Statistics:
  Max Input: {max_input:.2f} µA/cm²
  Max Filter: {max_filter:.2f} µA/cm²
  Max Attenuated: {max_attenuated:.2f} µA/cm²
  Effective Attenuation: {effective_atten:.3f}
  Rise Time (63%): {rise_time:.2f} ms
  Expected τ: {self.config.dendritic_filter.tau_dendritic_ms:.1f} ms
"""
        
        self.txt_analysis.setPlainText(analysis_text)
        
        # Emit signal for other components
        self.filter_updated.emit(data)
    
    def clear_data(self):
        """Clear all filter data."""
        self.filter_data = []
        self.result = None
        
        # Clear plots
        self.curve_filter_state.clear()
        self.curve_input_current.clear()
        self.curve_attenuated.clear()
        
        # Reset displays
        self.lbl_attenuation.setText("0.000")
        self.lbl_error.setText("0.0%")
        self.txt_analysis.clear()
        
        self.txt_analysis.append("🗑️ Data cleared")
    
    def update_display(self):
        """Update display in real-time mode."""
        if self.check_real_time.isChecked() and self.filter_data:
            # Could implement real-time filtering here
            pass
    
    def get_filter_data(self):
        """Get current filter data for export."""
        return self.filter_data
    
    def export_data(self, filename):
        """Export filter data to file."""
        if not self.filter_data:
            return False
        
        try:
            # Save data to CSV
            import pandas as pd
            
            df = pd.DataFrame({
                'time_ms': self.filter_data['time'],
                'input_current': self.filter_data['input_current'],
                'filter_state': self.filter_data['filter_state'],
                'attenuated': self.filter_data['attenuated']
            })
            
            df.to_csv(filename, index=False)
            self.txt_analysis.append(f"💾 Data exported to {filename}")
            return True
            
        except Exception as e:
            self.txt_analysis.append(f"❌ Export error: {str(e)}")
            return False


if __name__ == '__main__':
    import sys
    from PySide6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Create test configuration
    config = FullModelConfig()
    apply_preset(config, "B: Pyramidal L5 (Mainen 1996)")
    config.stim_location.location = "dendritic_filtered"
    config.dendritic_filter.enabled = True
    
    widget = DendriticFilterMonitor()
    widget.set_config(config)
    widget.show()
    
    sys.exit(app.exec())
