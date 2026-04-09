"""
gui/axon_biophysics.py — Axonal Segment Biophysics Viewer v10.0

Visualizes the first AIS segment:
  - Membrane potential along the segment  
  - Ion channel currents (Na, K, Leak, Ih, ICa, IA, SK) as heatmap
  - Channel density distribution
  - Activation/inactivation dynamics
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from pyqtgraph import ViewBox


class AxonBiophysicsWidget(QWidget):
    """Visualize first axonal segment (AIS) biophysics in detail."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Top control bar
        ctrl_bar = QHBoxLayout()
        
        lbl_channel = QLabel("Channel View:")
        self.combo_channel = QComboBox()
        self.combo_channel.addItems([
            "All Ion Currents (Heatmap)",
            "Sodium (Na)",
            "Potassium (K)",
            "Leak (L)",
            "Inward Rectifier (Ih) [if enabled]",
            "Calcium (ICa) [if enabled]",
            "A-type (IA) [if enabled]",
            "SK Calcium-activated [if enabled]"
        ])
        self.combo_channel.currentIndexChanged.connect(self._on_channel_changed)
        
        ctrl_bar.addWidget(lbl_channel)
        ctrl_bar.addWidget(self.combo_channel)
        ctrl_bar.addStretch()
        layout.addLayout(ctrl_bar)

        # Create graphics layout for 3-panel view
        self._glw = pg.GraphicsLayoutWidget()  # type: ignore
        self._glw.setBackground('#0D1117')
        layout.addWidget(self._glw, stretch=1)

        # Panel 1: Membrane potential along segment
        self._p_voltage = self._glw.addPlot(row=0, col=0)  # type: ignore
        self._p_voltage.setLabel('left', 'Voltage (mV)', color='#CDD6F4')
        self._p_voltage.setLabel('bottom', 'Position along AIS (µm)', color='#CDD6F4')
        self._p_voltage.setTitle('Membrane Potential', color='#89B4FA')
        self._p_voltage.getViewBox().setMouseEnabled(x=True, y=True)
        self._p_voltage.addLegend(offset=(10, 10), labelTextColor='#CDD6F4')

        # Panel 2: Ion channel heatmap / currents
        self._p_heatmap = self._glw.addPlot(row=0, col=1)  # type: ignore
        self._p_heatmap.setLabel('left', 'Channel Type', color='#CDD6F4')
        self._p_heatmap.setLabel('bottom', 'Time (ms)', color='#CDD6F4')
        self._p_heatmap.setTitle('ion Currents (Heatmap)', color='#89B4FA')

        # Panel 3: Channel conductance profile
        self._p_conduct = self._glw.addPlot(row=1, col=0)  # type: ignore
        self._p_conduct.setLabel('left', 'Conductance Density (S/cm²)', color='#CDD6F4')
        self._p_conduct.setLabel('bottom', 'Channel Type', color='#CDD6F4')
        self._p_conduct.setTitle('Channel Density Profile', color='#89B4FA')
        self._p_conduct.addLegend(offset=(10, 10), labelTextColor='#CDD6F4')

        # Panel 4: Gate dynamics (m, h, n, r, s, u, a, b)
        self._p_gates = self._glw.addPlot(row=1, col=1)  # type: ignore
        self._p_gates.setLabel('left', 'Gate Position (0-1)', color='#CDD6F4')
        self._p_gates.setLabel('bottom', 'Time (ms)', color='#CDD6F4')
        self._p_gates.setTitle('Ion Channel Gating Dynamics', color='#A6ADC8')
        self._p_gates.addLegend(offset=(10, 10), labelTextColor='#CDD6F4')

        # Info label
        self._info = QLabel("")
        self._info.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._info.setStyleSheet("color:#A6E3A1; font-size:10px; padding:4px;")
        layout.addWidget(self._info)

        self._current_channel = 0
        self._last_result = None
        self._last_config = None
        
    def clear_cached_data(self):
        """Clear cached simulation data to prevent stale display."""
        self._last_result = None
        self._last_config = None
        self._clear_plots()

    def _on_channel_changed(self):
        """User changed channel view selection."""
        self._current_channel = self.combo_channel.currentIndex()
        # Re-render immediately if we have data
        if self._last_result is not None and self._last_config is not None:
            self.plot_axon_data(self._last_result, self._last_config)

    # ─────────────────────────────────────────────────────────────────
    def plot_axon_data(self, result, config):
        """
        Render the AIS visualization given a simulation result.
        
        Args:
            result: SimulationResult from solver
            config: FullModelConfig with morphology and channel info
        """
        # Store for re-rendering on channel change
        self._last_result = result
        self._last_config = config
        
        self._clear_plots()

        t = result.t
        y = result.y
        n_comp = result.n_comp
        cfg = config

        if n_comp < 2:
            self._info.setText("Single-compartment model: no axonal segments to visualize")
            return

        # Extract first segment voltages and gating variables
        # y layout: [V (n), m(n), h(n), n(n), [r(n)], [s(n),u(n)], [a(n),b(n)], [Ca(n)]]
        v_all = y[0:n_comp, :]  # Voltages for all compartments
        v_ais = v_all[0, :]      # First AIS segment

        cursor = n_comp * 4  # After V, m, h, n

        m = y[n_comp:2*n_comp, :]
        h = y[2*n_comp:3*n_comp, :]
        n_gate = y[3*n_comp:4*n_comp, :]

        gates_dict = {
            'm': m[0, :],
            'h': h[0, :],
            'n': n_gate[0, :],
        }

        # ────── Panel 1: Voltage trace ──────────────────────────────
        self._p_voltage.plot(
            t, v_ais,
            pen=pg.mkPen(QColor('#FA8C3C'), width=2),
            name='AIS Voltage'
        )
        
        # Resting potential line
        rest = v_ais[0]
        self._p_voltage.addLine(y=rest, pen=pg.mkPen('#585B70', style=Qt.PenStyle.DashLine))

        # ────── Panel 2: Ion currents heatmap ──────────────────────
        if hasattr(result, 'currents') and result.currents:
            currents_list = []
            current_names = []
            channel_map = {
                0: ['Na', 'K', 'Leak', 'Ih', 'ICa', 'IA', 'SK'],  # All
                1: ['Na'],  # Sodium
                2: ['K'],  # Potassium
                3: ['Leak'],  # Leak
                4: ['Ih'],  # Ih
                5: ['ICa'],  # Calcium
                6: ['IA'],  # A-type
                7: ['SK'],  # SK
            }
            
            channels_to_show = channel_map.get(self._current_channel, channel_map[0])

            for ch in channels_to_show:
                if ch in result.currents:
                    currents_list.append(result.currents[ch])
                    current_names.append(f'I_{ch}')

            if currents_list:
                # Stack currents into 2D array [channels × time]
                I_matrix = np.vstack(currents_list)  # shape: (n_channels, n_timepoints)

                # Create heatmap
                img = pg.ImageItem(
                    image=I_matrix,
                    levels=(I_matrix.min(), I_matrix.max())
                )
                # Apply viridis colormap if available
                cm = pg.colormap.get('viridis')
                if cm is not None:
                    img.setColorMap(cm)
                self._p_heatmap.addItem(img)

                # Set Y-axis labels (channel names)
                ax = self._p_heatmap.getAxis('left')
                ticks = [(i, name) for i, name in enumerate(current_names)]
                ax.setTicks([ticks])

                # Set position and scale to match time/channel dimensions
                # ImageItem rect: (x, y, width, height) in data coordinates
                rect = pg.QtCore.QRectF(0, 0, t[-1], len(current_names))
                img.setRect(rect)

        # ────── Panel 3: Channel conductance profile ─────────────────
        if hasattr(result, 'morph') and result.morph:
            morph = result.morph
            conduct_data = {}

            if 'gNa_v' in morph:
                conduct_data['gNa'] = morph['gNa_v'][0]
            if 'gK_v' in morph:
                conduct_data['gK'] = morph['gK_v'][0]
            if 'gL_v' in morph:
                conduct_data['gL'] = morph['gL_v'][0]
            if 'gIh_v' in morph and cfg.channels.enable_Ih:
                conduct_data['gIh'] = morph['gIh_v'][0]
            if 'gCa_v' in morph and cfg.channels.enable_ICa:
                conduct_data['gCa'] = morph['gCa_v'][0]
            if 'gA_v' in morph and cfg.channels.enable_IA:
                conduct_data['gA'] = morph['gA_v'][0]
            if 'gSK_v' in morph and cfg.channels.enable_SK:
                conduct_data['gSK'] = morph['gSK_v'][0]

            if conduct_data:
                names = list(conduct_data.keys())
                values = list(conduct_data.values())
                x_pos = np.arange(len(names))

                # Bar plot
                brush_colors = {
                    'gNa': '#FA8C3C',   # Orange
                    'gK': '#89B4FA',    # Blue
                    'gL': '#A6ADC8',    # Gray
                    'gIh': '#F5A97F',   # Light orange
                    'gCa': '#F38BA8',   # Pink
                    'gA': '#A6E3A1',    # Green
                    'gSK': '#EBA0AC',   # Red
                }

                for i, (name, val) in enumerate(zip(names, values)):
                    color = brush_colors.get(name, '#CDD6F4')
                    self._p_conduct.plot(
                        [x_pos[i]], [val],
                        pen=None,
                        symbol='s',
                        symbolBrush=pg.mkBrush(QColor(color)),
                        symbolSize=12,
                        name=name
                    )

                # Set X-axis labels
                ax = self._p_conduct.getAxis('bottom')
                ticks = [(i, name) for i, name in enumerate(names)]
                ax.setTicks([ticks])

        # ────── Panel 4: Gating dynamics ────────────────────────────
        colors_gates = {
            'm': '#FA8C3C',   # Na activation (orange)
            'h': '#F38BA8',   # Na inactivation (pink)
            'n': '#89B4FA',   # K activation (blue)
            'r': '#F5A97F',   # Ih activation (light orange)
            's': '#EBA0AC',   # Ca activation (red)
            'u': '#DBA0F2',   # Ca inactivation (purple)
            'a': '#A6E3A1',   # A activation (green)
            'b': '#94E2D5',   # A inactivation (cyan)
        }

        for gate_name, gate_vals in gates_dict.items():
            if gate_vals is not None and len(gate_vals) > 0:
                color = colors_gates.get(gate_name, '#CDD6F4')
                self._p_gates.plot(
                    t, gate_vals,
                    pen=pg.mkPen(QColor(color), width=2),
                    name=f'{gate_name}'
                )

        # ────── Info text ──────────────────────────────────────────
        info_text = (
            f"AIS Segment #0  |  "
            f"V_peak = {v_ais.max():.1f} mV  |  "
            f"V_min = {v_ais.min():.1f} mV  |  "
            f"Spikes: {(v_ais > v_ais[0] + 30).sum()}"
        )
        self._info.setText(info_text)

    def _clear_plots(self):
        """Clear all plot panels."""
        self._p_voltage.clear()
        self._p_heatmap.clear()
        self._p_conduct.clear()
        self._p_gates.clear()
