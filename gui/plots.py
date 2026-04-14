"""
gui/plots.py — Real-time Oscilloscope Widget v10.3

Upper pane : V(t) for soma, AIS, axon terminal — with spike markers + threshold line
Middle pane: Gate variables m, h, n (toggleable)
Lower pane : Ion currents with fill-to-zero + filtered stimulus if available

v10.3 changes:
- Spike markers (vertical dotted lines) at each AP peak in V(t)
- Horizontal threshold line at -20 mV in V(t)
- Spike count + firing rate shown in V(t) title
- Filtered dendritic stimulus trace in currents pane (when dendritic filter active)
- Subplot size ratios: V(t) gets more vertical space (3:2:2)
"""
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                                QCheckBox, QGroupBox, QComboBox, QDoubleSpinBox, QLabel, QSpinBox, QPushButton, QMainWindow, QScrollArea, QFormLayout)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPageSize, QPdfWriter
from .delay_target import junction_index, resolve_delay_target

# Colour scheme (matches analytics.py)
CHAN_COLORS = {
    'Na':   (220, 50,  50,  200),
    'K':    (50,  100, 220, 200),
    'Leak': (50,  160, 50,  150),
    'Ih':   (160, 50,  200, 180),
    'ICa':  (250, 150, 0,   200),
    'KATP': (249, 226, 175, 200),
    'PumpNaK': (229, 200, 144, 200),
    'IA':   (0,   200, 200, 180),
    'SK':   (200, 50,  160, 180),
}
GATE_COLORS = {
    'm': (255, 80,  80,  220),
    'h': (80,  120, 255, 220),
    'n': (80,  200, 80,  220),
}
COMP_COLORS = {
    'Soma':     'b',
    'AIS':      'r',
    'Terminal': (0, 180, 80),
}

_THRESHOLD_MV = -20.0   # AP detection threshold (mV)
_MAX_SPIKE_MARKERS = 150  # cap markers to avoid clutter at very high firing rates
_MAX_PLOT_POINTS = 2500

_OSC_LAYOUT_PROFILES = {
    "Compact": (3.5, 1.0, 2.0, 0.8),
    "Balanced": (4.0, 1.4, 3.0, 1.0),
    "Voltage Focus": (6.0, 1.0, 2.2, 0.8),
    "Currents Focus": (3.0, 1.0, 5.0, 0.8),
    "Expanded": (5.0, 1.8, 4.5, 1.2),
}

_OSC_HEIGHT_PROFILES = {
    "Fit": 720,
    "Tall": 980,
    "Very Tall": 1280,
}


def _plot_point_budget(plot, *, min_points: int = 800, max_points: int = 6000, oversample: float = 2.0) -> int:
    """Estimate draw-point budget from current viewport width."""
    width_px = float(plot.getViewBox().sceneBoundingRect().width())
    if width_px <= 0.0:
        return _MAX_PLOT_POINTS
    budget = int(width_px * oversample)
    return max(min_points, min(max_points, budget))


def _downsample_xy(t: np.ndarray, y: np.ndarray, max_points: int = _MAX_PLOT_POINTS) -> tuple[np.ndarray, np.ndarray]:
    """Peak-preserving Min-Max Decimation for high-fidelity spike rendering.

    Min-Max Decimation ensures every spike peak is visible by capturing both
    min and max values within each chunk. For AP traces with 1ms peaks,
    this prevents aliasing that occurs with naive stride sampling [::step].
    
    Supports both 1D (single compartment) and 2D (all compartments) arrays.
    For 2D, finds global min/max across ALL compartments for each time chunk,
    ensuring distal compartment spikes are preserved.
    """
    # Normalize inputs
    t = np.asarray(t)
    y = np.asarray(y)
    n = t.shape[0]
    
    # Validate dimensions
    if y.ndim > 2:
        raise ValueError(f"_downsample_xy supports 1D or 2D arrays, got {y.ndim}D")
    
    is_2d = y.ndim == 2
    
    # Validate shape consistency for 2D (time must be axis=1)
    if is_2d and y.shape[1] != n:
        raise ValueError(f"Shape mismatch: t has {n} points, y has {y.shape[1]} time points (shape {y.shape})")
    if not is_2d and y.shape[0] != n:
        raise ValueError(f"Shape mismatch: t has {n} points, y has {y.shape[0]} points")
    
    # Handle empty or trivial cases (return normalized arrays)
    if n == 0 or n <= max_points or max_points <= 0:
        return t, y

    # Each chunk contributes 2 points (min and max)
    n_chunks = max_points // 2
    chunk_size = max(1, n // n_chunks)

    # Collect min and max indices from each chunk
    indices = []
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        if is_2d:
            # 2D: (n_comp, n_time) - find global min/max across ALL compartments
            chunk = y[:, i:end]  # Shape: (n_comp, chunk_time)
            # Find where global min and max occur (across all compartments)
            flat_min_idx = int(np.argmin(chunk))
            flat_max_idx = int(np.argmax(chunk))
            # Convert flat index to (compartment, time) - safe for any memory order
            min_time_idx = np.unravel_index(flat_min_idx, chunk.shape)[1]
            max_time_idx = np.unravel_index(flat_max_idx, chunk.shape)[1]
            min_idx = i + min_time_idx
            max_idx = i + max_time_idx
        else:
            # 1D: original logic
            chunk = y[i:end]
            min_idx = i + int(np.argmin(chunk))
            max_idx = i + int(np.argmax(chunk))
        indices.append(min_idx)
        indices.append(max_idx)

    # Sort indices to maintain temporal order
    indices = np.unique(np.array(indices, dtype=np.int64))

    # Always include the last point for trace continuity
    if indices[-1] != n - 1:
        indices = np.concatenate([indices, np.array([n - 1])])

    # Return appropriately sliced array
    if is_2d:
        return t[indices], y[:, indices]  # Keep all compartments, downsample time
    else:
        return t[indices], y[indices]


PLOT_THEMES = {
    "Default": {
        "soma": "#4080FF",
        "ais": "#FF4040",
        "terminal": (0, 200, 100),
        "threshold": "#F9E2AF",
        "spike": "#F38BA8",
        "stim_input": "#F5C2E7",
        "stim_filtered": "#89B4FA",
        "custom": "#CBA6F7",  # Purple for custom compartment
    },
    "High Contrast": {
        "soma": "#00D1FF",
        "ais": "#FF2A2A",
        "terminal": "#00FF7F",
        "threshold": "#FFD700",
        "spike": "#FF66CC",
        "stim_input": "#FFC0CB",
        "stim_filtered": "#8EC5FF",
        "custom": "#E0A0FF",  # Bright purple
    },
    "Colorblind Friendly": {
        "soma": "#0072B2",
        "ais": "#D55E00",
        "terminal": "#009E73",
        "threshold": "#F0E442",
        "spike": "#CC79A7",
        "stim_input": "#E69F00",
        "stim_filtered": "#56B4E9",
        "custom": "#AA4499",  # Purple distinguishable for colorblind
    },
}


class OscilloscopeWidget(QWidget):
    """
    Three-pane oscilloscope with checkboxes.
    - Top   : V(t) per compartment, spike markers, threshold line
    - Middle: gate dynamics m, h, n
    - Bottom: ionic currents (fill to zero) + optional filtered stim trace
    """

    delay_target_changed = Signal(str, int)
    time_highlighted = Signal(float)

    def __init__(self, parent=None):
        """
        Initialize the OscilloscopeWidget, build its UI, and initialize visual and stateful plot elements.
        
        Sets default view and rendering settings (theme, line width, title font size, grid alpha, presentation and scale-bar modes), delay-target defaults, last-result caches, fullscreen window tracker, and containers for persistent and transient plot items. Builds the UI and adds a persistent, initially hidden reference voltage curve and its label to the voltage plot.
        
        Parameters:
            parent: Optional parent widget for the widget.
        """
        super().__init__(parent)
        self._theme_name = "Default"
        self._line_width_scale = 1.0
        self._title_font_px = 14
        self._grid_alpha = 0.15
        self._presentation_mode = False
        self._scale_bar_mode = False
        self._layout_profile = "Balanced"
        self._height_profile = "Fit"
        self._delay_target_name = "Terminal"
        self._delay_custom_index = 1
        self._last_result = None
        self._last_mc_results = None
        self._fullscreen_windows = []
        self._build_ui()
        self._curves_v:    dict = {}
        self._curves_gate: dict = {}
        self._curves_i:    dict = {}
        self._curves_ca:   dict = {}
        self._transient_items: list = []
        # Persistent reference curve — created once, never removed from the plot.
        # ignoreBounds=True prevents it from affecting autoRange.
        _ref_pen = pg.mkPen(color=(150, 150, 150, 102), width=1.5, style=Qt.PenStyle.DashLine)  # Opacity 0.4 = 102/255
        self._ref_curve_v = pg.PlotDataItem([], [], pen=_ref_pen, name="Reference")
        self._p_v.addItem(self._ref_curve_v, ignoreBounds=True)
        self._ref_curve_v.setVisible(False)
        # Reference label
        self._ref_label = pg.TextItem("", color=(150, 150, 150, 180), anchor=(0, 1))
        self._ref_label.setPos(0.02, 0.98)
        self._ref_label.setVisible(False)
        self._p_v.addItem(self._ref_label, ignoreBounds=True)
        # External time marker line (for TopologyWidget sync)
        self._time_marker_line = pg.InfiniteLine(
            pos=0, angle=90,
            pen=pg.mkPen('#F9E2AF', width=2, style=Qt.PenStyle.DashLine)
        )
        self._time_marker_line.setVisible(False)
        self._p_v.addItem(self._time_marker_line, ignoreBounds=True)
        # Ghost currents storage and curves (for Comparative Biophysics)
        self._ref_currents: dict = {}  # Full currents dictionary reference
        self._ghost_curves: dict = {}  # Ghost curve items for currents pane

    # ─────────────────────────────────────────────────────────────────
    def _title_html(self, text: str, color: str) -> str:
        return f"<span style='color:{color}; font-size:{self._title_font_px}px'>{text}</span>"

    def _apply_grid_alpha(self):
        self._p_v.showGrid(x=True, y=True, alpha=self._grid_alpha)
        self._p_g.showGrid(x=True, y=True, alpha=self._grid_alpha)
        self._p_i.showGrid(x=True, y=True, alpha=self._grid_alpha)
        self._p_ca.showGrid(x=True, y=True, alpha=self._grid_alpha)

    def _set_default_titles(self):
        self._p_v.setTitle(self._title_html("Membrane Potential  V(t)", "#89B4FA"))
        self._p_g.setTitle(self._title_html("Gate Variables  m, h, n", "#A6E3A1"))
        self._p_i.setTitle(self._title_html("Ion Currents  (soma)", "#FAB387"))

    def _target_plot_height(self) -> int:
        base = _OSC_HEIGHT_PROFILES.get(self._height_profile, _OSC_HEIGHT_PROFILES["Fit"])
        if self._presentation_mode:
            base = int(base * 1.1)
        return base

    def _apply_plot_layout_profile(self) -> None:
        weights = _OSC_LAYOUT_PROFILES.get(
            self._layout_profile,
            _OSC_LAYOUT_PROFILES["Balanced"],
        )
        row_visible = (
            True,
            self._p_g.isVisible(),
            self._p_i.isVisible(),
            self._p_ca.isVisible(),
        )
        for row_i, (weight, visible) in enumerate(zip(weights, row_visible)):
            self._win.ci.layout.setRowStretchFactor(row_i, float(weight) if visible else 0.01)

        target_height = self._target_plot_height()
        if hasattr(self, "_plot_container"):
            self._plot_container.setMinimumHeight(target_height)
            self._plot_container.setFixedHeight(target_height)
        if hasattr(self, "_win"):
            self._win.setMinimumHeight(target_height)
            self._win.setFixedHeight(target_height)
        if hasattr(self, "_plot_scroll"):
            self._plot_scroll.setMinimumHeight(min(target_height + 16, 1200))
            self._plot_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    # ─────────────────────────────────────────────────────────────────
    def _mouse_moved(self, evt):
        """Handle mouse movement for crosshair display."""
        pos = evt[0]
        if self._p_v.sceneBoundingRect().contains(pos):
            mousePoint = self._p_v.getViewBox().mapSceneToView(pos)
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())
            self.crosshair_label.setText(f"t={mousePoint.x():.2f} ms\nV={mousePoint.y():.2f} mV")
            self.crosshair_label.setPos(mousePoint.x(), mousePoint.y())
            self.time_highlighted.emit(mousePoint.x())

    def set_time_marker(self, t_ms: float):
        """Set a vertical time marker line (called from TopologyWidget)."""
        if not np.isfinite(t_ms):
            self._time_marker_line.setVisible(False)
            return
        self._time_marker_line.setPos(float(t_ms))
        self._time_marker_line.setVisible(True)

    def set_delay_compartment(self, comp_idx: int):
        """Set the delay target compartment index (called from TopologyWidget)."""
        # Switch to Custom Compartment mode and set the index
        self._combo_delay_target.blockSignals(True)
        self._combo_delay_target.setCurrentText("Custom Compartment")
        self._combo_delay_target.blockSignals(False)
        # Update the spinbox
        max_idx = self._spin_delay_comp.maximum()
        if 1 <= comp_idx <= max_idx:
            self._spin_delay_comp.blockSignals(True)
            self._spin_delay_comp.setValue(comp_idx)
            self._spin_delay_comp.blockSignals(False)
            self._delay_target_name = "Custom Compartment"
            self._delay_custom_index = comp_idx
            self._spin_delay_comp.setEnabled(True)
            # Emit signal to notify MainWindow
            self.delay_target_changed.emit("Custom Compartment", comp_idx)
            # Refresh plots if we have a result
            if self._last_result is not None:
                self.update_plots(self._last_result)
            elif self._last_mc_results is not None:
                self.update_plots_mc(self._last_mc_results)

    def cleanup(self):
        """Clean up resources to prevent memory leaks."""
        if hasattr(self, 'proxy') and self.proxy is not None:
            self.proxy.disconnect()
            self.proxy = None

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()

    def _build_ui(self):
        """
        Construct and lay out the oscilloscope plotting area and its accompanying control panel.
        
        Creates the main graphics widget with four stacked plots (membrane potential, gate variables, ion currents, intracellular calcium), adds crosshair items and a mouse-move signal proxy for the voltage plot, and configures plot appearance (labels, legends, grid, row stretch factors, and view linking). Builds a scrollable right-hand control panel containing grouped checkboxes for voltage/gate/current/calcium traces and a "View" group with theme, line width, title font, grid alpha, reference toggle, presentation/scale-bar mode, spike/delay controls, delay-target selection and custom compartment selector, and a fullscreen button. Connects UI controls to the widget's handlers, stores commonly referenced widgets on self (plots, checkboxes, controls, proxy, reference widgets), sets initial splitter sizes, and updates row visibility.
        """
        from PySide6.QtWidgets import QSplitter
        
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        
        # Use splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)
        splitter.setStyleSheet("QSplitter::handle { background: #45475A; }")

        # ── Plot area ─────────────────────────────────────────────────
        self._win = pg.GraphicsLayoutWidget()
        self._win.setBackground('#0D1117')

        # V(t) — row 0
        self._p_v = self._win.addPlot(
            title=self._title_html("Membrane Potential  V(t)", "#89B4FA"))
        self._p_v.setLabel('left', 'V', units='mV', color='#CDD6F4')
        self._p_v.showGrid(x=True, y=True, alpha=self._grid_alpha)
        self._p_v.addLegend(offset=(10, 10), labelTextColor='#CDD6F4')
        self._p_v.setMouseEnabled(x=True, y=True)  # Enable zoom/pan
        
        # Crosshair for V(t) plot
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#A6ADC8', style=Qt.PenStyle.DashLine))
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('#A6ADC8', style=Qt.PenStyle.DashLine))
        self._p_v.addItem(self.vLine, ignoreBounds=True)
        self._p_v.addItem(self.hLine, ignoreBounds=True)
        self.crosshair_label = pg.TextItem(text="", color='#CBA6F7', fill='#1E1E2E88')
        self._p_v.addItem(self.crosshair_label, ignoreBounds=True)
        
        self.proxy = pg.SignalProxy(self._p_v.scene().sigMouseMoved, rateLimit=60, slot=self._mouse_moved)
        
        self._win.nextRow()

        # Gate variables — row 1
        self._p_g = self._win.addPlot(
            title=self._title_html("Gate Variables  m, h, n", "#A6E3A1"))
        self._p_g.setLabel('left', 'probability  [0–1]', color='#CDD6F4')
        self._p_g.showGrid(x=True, y=True, alpha=self._grid_alpha)
        self._p_g.addLegend(offset=(10, 10), labelTextColor='#CDD6F4')
        self._p_g.setXLink(self._p_v)
        self._p_g.setYRange(-0.05, 1.05)
        self._p_g.setMouseEnabled(x=True, y=True)  # Enable zoom/pan
        self._win.nextRow()

        # Currents — row 2
        self._p_i = self._win.addPlot(
            title=self._title_html("Ion Currents  (soma)", "#FAB387"))
        self._p_i.setLabel('left', 'I', units='µA/cm²', color='#CDD6F4')
        self._p_i.setLabel('bottom', 'Time', units='ms', color='#CDD6F4')
        self._p_i.showGrid(x=True, y=True, alpha=self._grid_alpha)
        self._p_i.addLegend(offset=(10, 10), labelTextColor='#CDD6F4')
        self._p_i.setXLink(self._p_v)
        self._p_i.setMouseEnabled(x=True, y=True)  # Enable zoom/pan
        self._win.nextRow()

        # Calcium — row 3
        self._p_ca = self._win.addPlot(
            title=self._title_html("Intracellular Calcium [Ca²+]_i", "#F38BA8"))
        self._p_ca.setLabel('left', '[Ca²+]_i', units='nM', color='#CDD6F4')
        self._p_ca.setLabel('bottom', 'Time', units='ms', color='#CDD6F4')
        self._p_ca.showGrid(x=True, y=True, alpha=self._grid_alpha)
        self._p_ca.addLegend(offset=(10, 10), labelTextColor='#CDD6F4')
        self._p_ca.setXLink(self._p_v)
        self._p_ca.setMouseEnabled(x=True, y=True)  # Enable zoom/pan

        # Row stretch: V(t) is tallest (3), gates (1.5), currents (4) - enlarged, calcium (1)
        self._win.ci.layout.setRowStretchFactor(0, 3)
        self._win.ci.layout.setRowStretchFactor(1, 1.5)
        self._win.ci.layout.setRowStretchFactor(2, 4)
        self._win.ci.layout.setRowStretchFactor(3, 1)

        self._plot_container = QWidget()
        plot_layout = QVBoxLayout(self._plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(0)
        plot_layout.addWidget(self._win)

        self._plot_scroll = QScrollArea()
        self._plot_scroll.setWidgetResizable(True)
        self._plot_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._plot_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._plot_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._plot_scroll.setStyleSheet("QScrollArea { background: #0D1117; border: none; }")
        self._plot_scroll.setWidget(self._plot_container)

        splitter.addWidget(self._plot_scroll)
        splitter.setStretchFactor(0, 10)

        # ── Scrollable Checkbox panel ─────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(180)
        scroll.setMaximumWidth(500)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: #1E1E2E; border: none; }")

        cb_widget = QWidget()
        cb_widget.setStyleSheet("background:#1E1E2E;")
        cb_layout = QVBoxLayout(cb_widget)
        cb_layout.setContentsMargins(4, 4, 4, 4)
        cb_layout.setSpacing(4)
        cb_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Voltage traces
        grp_v = QGroupBox("Voltage")
        grp_v.setStyleSheet("QGroupBox { color: #89B4FA; font-size:11px; }")
        vl = QVBoxLayout(grp_v)
        self._cb_v = {}
        for name in ('Soma', 'AIS', 'Terminal'):
            cb = QCheckBox(name)
            cb.setChecked(True)
            cb.setStyleSheet("color:#CDD6F4; font-size:11px;")
            cb.stateChanged.connect(lambda _, n=name: self._toggle_v(n))
            vl.addWidget(cb)
            self._cb_v[name] = cb
        cb_layout.addWidget(grp_v)

        # Gate checkboxes
        grp_g = QGroupBox("Gates")
        grp_g.setStyleSheet("QGroupBox { color: #A6E3A1; font-size:11px; }")
        gl = QVBoxLayout(grp_g)
        self._cb_g = {}
        for name in ('m', 'h', 'n'):
            cb = QCheckBox(name)
            cb.setChecked(True)
            cb.setStyleSheet("color:#CDD6F4; font-size:11px;")
            cb.stateChanged.connect(lambda _, n=name: self._toggle_g(n))
            gl.addWidget(cb)
            self._cb_g[name] = cb
        cb_layout.addWidget(grp_g)

        # Current checkboxes
        grp_i = QGroupBox("Currents")
        grp_i.setStyleSheet("QGroupBox { color: #FAB387; font-size:11px; }")
        il = QVBoxLayout(grp_i)
        self._cb_i = {}
        for name in list(CHAN_COLORS.keys()) + ['Stim_input', 'Stim_filtered']:
            if name == 'Stim_filtered':
                label = 'Stim(filt)'
            elif name == 'Stim_input':
                label = 'Stim(input)'
            else:
                label = name
            cb = QCheckBox(label)
            cb.setChecked(True)
            cb.setStyleSheet("color:#CDD6F4; font-size:11px;")
            cb.stateChanged.connect(lambda _, n=name: self._toggle_i(n))
            il.addWidget(cb)
            self._cb_i[name] = cb
        cb_layout.addWidget(grp_i)

        # Calcium checkbox
        grp_ca = QGroupBox("Calcium")
        grp_ca.setStyleSheet("QGroupBox { color: #F38BA8; font-size:11px; }")
        cal = QVBoxLayout(grp_ca)
        self._cb_ca = {}
        cb_ca = QCheckBox("Show [Ca²+]_i")
        cb_ca.setChecked(True)
        cb_ca.setStyleSheet("color:#CDD6F4; font-size:11px;")
        cb_ca.stateChanged.connect(self._toggle_ca)
        cal.addWidget(cb_ca)
        self._cb_ca['calcium'] = cb_ca
        cb_layout.addWidget(grp_ca)

        # View controls
        grp_view = QGroupBox("View")
        grp_view.setStyleSheet("QGroupBox { color: #94E2D5; font-size:11px; }")
        vl2 = QFormLayout(grp_view)
        vl2.setContentsMargins(4, 8, 4, 4)
        vl2.setVerticalSpacing(2)

        self._combo_theme = QComboBox()
        self._combo_theme.addItems(list(PLOT_THEMES.keys()))
        self._combo_theme.setCurrentText(self._theme_name)
        self._combo_theme.currentTextChanged.connect(self._on_view_settings_changed)
        vl2.addRow("Theme", self._combo_theme)

        self._spin_line_width = QDoubleSpinBox()
        self._spin_line_width.setRange(0.7, 3.0)
        self._spin_line_width.setSingleStep(0.1)
        self._spin_line_width.setDecimals(1)
        self._spin_line_width.setValue(self._line_width_scale)
        self._spin_line_width.valueChanged.connect(self._on_view_settings_changed)
        vl2.addRow("Line Width", self._spin_line_width)

        self._spin_title_px = QSpinBox()
        self._spin_title_px.setRange(10, 24)
        self._spin_title_px.setValue(self._title_font_px)
        self._spin_title_px.valueChanged.connect(self._on_view_settings_changed)
        vl2.addRow("Title Font", self._spin_title_px)

        self._spin_grid_alpha = QDoubleSpinBox()
        self._spin_grid_alpha.setRange(0.05, 0.60)
        self._spin_grid_alpha.setSingleStep(0.05)
        self._spin_grid_alpha.setDecimals(2)
        self._spin_grid_alpha.setValue(self._grid_alpha)
        self._spin_grid_alpha.valueChanged.connect(self._on_view_settings_changed)
        vl2.addRow("Grid Alpha", self._spin_grid_alpha)

        self._combo_layout_profile = QComboBox()
        self._combo_layout_profile.addItems(list(_OSC_LAYOUT_PROFILES.keys()))
        self._combo_layout_profile.setCurrentText(self._layout_profile)
        self._combo_layout_profile.currentTextChanged.connect(self._on_view_settings_changed)
        vl2.addRow("Layout", self._combo_layout_profile)

        self._combo_height_profile = QComboBox()
        self._combo_height_profile.addItems(list(_OSC_HEIGHT_PROFILES.keys()))
        self._combo_height_profile.setCurrentText(self._height_profile)
        self._combo_height_profile.currentTextChanged.connect(self._on_view_settings_changed)
        vl2.addRow("Plot Height", self._combo_height_profile)

        self._cb_keep_reference = QCheckBox("Keep as Reference")
        self._cb_keep_reference.setChecked(False)
        self._cb_keep_reference.setToolTip("Freeze current soma trace as a grey reference line for comparison")
        self._cb_keep_reference.setStyleSheet("color:#A6ADC8; font-size:11px;")
        vl2.addRow(self._cb_keep_reference)

        # Ghost metrics display (Comparative Biophysics metrics)
        self._lbl_ghost_metrics = QLabel("")
        self._lbl_ghost_metrics.setStyleSheet(
            "color:#89B4FA; font-size:10px; font-family:monospace; padding:4px;"
        )
        self._lbl_ghost_metrics.setWordWrap(True)
        self._lbl_ghost_metrics.setToolTip("Quantitative comparison between current and reference traces")
        vl2.addRow(self._lbl_ghost_metrics)

        self._cb_presentation = QCheckBox("Presentation Mode")
        self._cb_presentation.setChecked(False)
        self._cb_presentation.setStyleSheet("color:#CDD6F4; font-size:11px;")
        self._cb_presentation.stateChanged.connect(self._on_view_settings_changed)
        vl2.addRow(self._cb_presentation)

        self._cb_scale_bars = QCheckBox("Publication Mode (Scale Bars)")
        self._cb_scale_bars.setChecked(False)
        self._cb_scale_bars.setToolTip("Hide axes and show scale bars for publication-ready figures")
        self._cb_scale_bars.setStyleSheet("color:#CBA6F7; font-size:11px;")
        self._cb_scale_bars.stateChanged.connect(self._on_view_settings_changed)
        vl2.addRow(self._cb_scale_bars)

        self._cb_show_spike_markers = QCheckBox("Show spike markers")
        self._cb_show_spike_markers.setChecked(True)
        self._cb_show_spike_markers.setStyleSheet("color:#CDD6F4; font-size:11px;")
        self._cb_show_spike_markers.stateChanged.connect(self._on_view_settings_changed)
        vl2.addRow(self._cb_show_spike_markers)

        self._cb_dense_spike_mode = QCheckBox("Dense-spike simplification")
        self._cb_dense_spike_mode.setChecked(True)
        self._cb_dense_spike_mode.setToolTip("Automatically simplify spike markers for long, high-rate traces")
        self._cb_dense_spike_mode.setStyleSheet("color:#CDD6F4; font-size:11px;")
        self._cb_dense_spike_mode.stateChanged.connect(self._on_view_settings_changed)
        vl2.addRow(self._cb_dense_spike_mode)

        self._cb_show_delay = QCheckBox("Show soma delay overlay")
        self._cb_show_delay.setChecked(True)
        self._cb_show_delay.setStyleSheet("color:#CDD6F4; font-size:11px;")
        self._cb_show_delay.stateChanged.connect(self._on_view_settings_changed)
        vl2.addRow(self._cb_show_delay)

        self._combo_delay_target = QComboBox()
        self._combo_delay_target.addItems(
            ["Terminal", "AIS", "Trunk Junction", "Custom Compartment"]
        )
        self._combo_delay_target.setCurrentText(self._delay_target_name)
        self._combo_delay_target.currentTextChanged.connect(self._on_view_settings_changed)
        vl2.addRow("Delay Target", self._combo_delay_target)

        self._spin_delay_comp = QSpinBox()
        self._spin_delay_comp.setRange(1, 1)
        self._spin_delay_comp.setValue(self._delay_custom_index)
        self._spin_delay_comp.setEnabled(False)
        self._spin_delay_comp.valueChanged.connect(self._on_view_settings_changed)
        vl2.addRow("Custom Comp", self._spin_delay_comp)

        self._btn_fullscreen = QPushButton("Full Screen")
        self._btn_fullscreen.setToolTip("Open oscilloscope plots in a maximized window")
        self._btn_fullscreen.clicked.connect(self.open_fullscreen)
        vl2.addRow(self._btn_fullscreen)

        cb_layout.addWidget(grp_view)
        cb_layout.addStretch()

        scroll.setWidget(cb_widget)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([800, 240])  # Initial sizes
        
        root.addWidget(splitter)
        self._update_row_visibility()
        self._apply_plot_layout_profile()

    def _copy_view_state_to(self, other: "OscilloscopeWidget"):
        """Copy current visual toggles/settings into another oscilloscope widget."""
        other._combo_theme.setCurrentText(self._combo_theme.currentText())
        other._spin_line_width.setValue(self._spin_line_width.value())
        other._spin_title_px.setValue(self._spin_title_px.value())
        other._spin_grid_alpha.setValue(self._spin_grid_alpha.value())
        other._combo_layout_profile.setCurrentText(self._combo_layout_profile.currentText())
        other._combo_height_profile.setCurrentText(self._combo_height_profile.currentText())
        other._cb_show_spike_markers.setChecked(self._cb_show_spike_markers.isChecked())
        other._cb_dense_spike_mode.setChecked(self._cb_dense_spike_mode.isChecked())
        other._cb_show_delay.setChecked(self._cb_show_delay.isChecked())
        other._combo_delay_target.setCurrentText(self._combo_delay_target.currentText())
        other._spin_delay_comp.setValue(self._spin_delay_comp.value())
        for name, cb in self._cb_v.items():
            if name in other._cb_v:
                other._cb_v[name].setChecked(cb.isChecked())
        for name, cb in self._cb_g.items():
            if name in other._cb_g:
                other._cb_g[name].setChecked(cb.isChecked())
        for name, cb in self._cb_i.items():
            if name in other._cb_i:
                other._cb_i[name].setChecked(cb.isChecked())

    def open_fullscreen(self):
        """Open an interactive, maximized oscilloscope clone for detailed inspection."""
        win = QMainWindow(self)
        win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        win.setWindowTitle("NeuroModelPort — Oscilloscope (Full Screen)")
        full = OscilloscopeWidget()
        self._copy_view_state_to(full)
        if self._last_result is not None:
            full.update_plots(self._last_result)
        elif self._last_mc_results is not None:
            full.update_plots_mc(self._last_mc_results)
        win.setCentralWidget(full)
        win.showMaximized()
        self._fullscreen_windows.append(win)

        def _cleanup(*_):
            self._fullscreen_windows = [w for w in self._fullscreen_windows if w is not win]

        win.destroyed.connect(_cleanup)

    # ─────────────────────────────────────────────────────────────────
    #  TOGGLE VISIBILITY
    # ─────────────────────────────────────────────────────────────────
    def _toggle_v(self, name):
        if name in self._curves_v:
            self._curves_v[name].setVisible(self._cb_v[name].isChecked())
        self._update_row_visibility()

    def _toggle_g(self, name):
        if name in self._curves_gate:
            self._curves_gate[name].setVisible(self._cb_g[name].isChecked())
        self._update_row_visibility()

    def _toggle_i(self, name):
        if name in self._curves_i:
            self._curves_i[name].setVisible(self._cb_i[name].isChecked())
        self._update_row_visibility()

    def _toggle_ca(self):
        if 'calcium' in self._curves_ca:
            self._curves_ca['calcium'].setVisible(self._cb_ca['calcium'].isChecked())
        self._update_row_visibility()

    def _update_row_visibility(self):
        """Collapse plot rows entirely if all their checkboxes are disabled."""
        any_g = any(cb.isChecked() for cb in self._cb_g.values())
        self._p_g.setVisible(any_g)

        any_i = any(cb.isChecked() for cb in self._cb_i.values())
        self._p_i.setVisible(any_i)

        any_ca = self._cb_ca['calcium'].isChecked() if 'calcium' in self._cb_ca else False
        self._p_ca.setVisible(any_ca)
        self._apply_plot_layout_profile()

    def _on_view_settings_changed(self, *_):
        """
        Apply the current view-related UI controls to the widget's internal state and refresh the displayed plots.
        
        Updates theme, line width scale, title font size, grid alpha, presentation and scale-bar modes, and the delay-target selection (including enabling the custom-index control). Emits the `delay_target_changed` signal with the new selection, reapplies grid transparency, and re-renders either the last single-result or Monte-Carlo plots if available; otherwise restores default plot titles.
        """
        self._theme_name = self._combo_theme.currentText()
        line_scale_base = float(self._spin_line_width.value())
        title_px_base = int(self._spin_title_px.value())
        grid_alpha_base = float(self._spin_grid_alpha.value())
        self._presentation_mode = bool(self._cb_presentation.isChecked())
        self._scale_bar_mode = bool(self._cb_scale_bars.isChecked())
        self._layout_profile = self._combo_layout_profile.currentText()
        self._height_profile = self._combo_height_profile.currentText()
        if self._presentation_mode:
            self._line_width_scale = line_scale_base * 1.35
            self._title_font_px = title_px_base + 2
            self._grid_alpha = min(0.6, grid_alpha_base + 0.08)
        else:
            self._line_width_scale = line_scale_base
            self._title_font_px = title_px_base
            self._grid_alpha = grid_alpha_base
        self._delay_target_name = self._combo_delay_target.currentText()
        self._delay_custom_index = int(self._spin_delay_comp.value())
        self._spin_delay_comp.setEnabled(
            self._delay_target_name == "Custom Compartment"
        )
        self.delay_target_changed.emit(
            self._delay_target_name,
            self._delay_custom_index,
        )
        self._apply_grid_alpha()
        self._apply_plot_layout_profile()
        if self._last_result is not None:
            self.update_plots(self._last_result)
        elif self._last_mc_results is not None:
            self.update_plots_mc(self._last_mc_results)
        else:
            self._set_default_titles()

    def get_delay_target_selection(self) -> tuple[str, int]:
        return self._delay_target_name, self._delay_custom_index

    @staticmethod
    def _first_crossing_time(v: np.ndarray, t: np.ndarray, threshold: float) -> float:
        idx = np.where((v[:-1] < threshold) & (v[1:] >= threshold))[0]
        if len(idx) == 0:
            return float("nan")
        return float(t[idx[0] + 1])

    def _resolve_delay_target(self, result):
        n = int(result.n_comp)
        mc = result.config.morphology
        return resolve_delay_target(
            target_name=self._delay_target_name,
            custom_index=self._delay_custom_index,
            n_comp=n,
            n_ais=int(mc.N_ais),
            n_trunk=int(mc.N_trunk),
            terminal_idx=n - 1,
        )

    def _sync_delay_controls_core(self, n: int, mc):
        self._cb_show_delay.setEnabled(n > 1)
        self._combo_delay_target.setEnabled(n > 1)
        max_idx = max(1, n - 1)
        self._spin_delay_comp.setRange(1, max_idx)
        if self._spin_delay_comp.value() > max_idx:
            self._spin_delay_comp.setValue(max_idx)
        has_ais = n > 1 and int(mc.N_ais) > 0
        j = junction_index(n, int(mc.N_ais), int(mc.N_trunk))
        has_junction = 1 <= j < n
        model = self._combo_delay_target.model()
        for i in range(self._combo_delay_target.count()):
            name = self._combo_delay_target.itemText(i)
            enabled = True
            if name == "AIS":
                enabled = has_ais
            elif name == "Trunk Junction":
                enabled = has_junction
            elif name == "Custom Compartment":
                enabled = n > 1
            if hasattr(model, "item"):
                item = model.item(i)
                if item is not None:
                    item.setEnabled(enabled)
        if self._delay_target_name == "AIS" and not has_ais:
            self._combo_delay_target.setCurrentText("Terminal")
        if self._delay_target_name == "Trunk Junction" and not has_junction:
            self._combo_delay_target.setCurrentText("Terminal")
        if self._delay_target_name == "Custom Compartment" and n <= 1:
            self._combo_delay_target.setCurrentText("Terminal")
        self._spin_delay_comp.setEnabled(
            n > 1 and self._combo_delay_target.currentText() == "Custom Compartment"
        )

    def _sync_delay_controls(self, result):
        self._sync_delay_controls_core(int(result.n_comp), result.config.morphology)

    def sync_delay_controls_for_config(self, config):
        mc = config.morphology
        if bool(mc.single_comp):
            n = 1
        else:
            n = int(1 + mc.N_ais + mc.N_trunk + mc.N_b1 + mc.N_b2)
        self._sync_delay_controls_core(n, mc)

    # ─────────────────────────────────────────────────────────────────
    #  CLEAR
    # ─────────────────────────────────────────────────────────────────
    def clear(self):
        for item in self._transient_items:
            for plot in (self._p_v, self._p_g, self._p_i, self._p_ca):
                try:
                    plot.removeItem(item)
                except (RuntimeError, AttributeError, ValueError) as e:
                    # Log specific cleanup errors for debugging
                    import logging
                    logging.debug(f"Failed to remove plot item {type(item).__name__}: {e}")
                    # Continue cleanup even if one item fails
                except Exception as e:
                    # Catch-all for unexpected errors but don't crash
                    logging.warning(f"Unexpected error during plot cleanup: {e}")
        self._transient_items = []
        # Hide and clear all persistent curve data (do NOT remove from plot)
        for c in self._curves_v.values():
            c.setData([], [])
            c.setVisible(False)
        for c in self._curves_gate.values():
            c.setData([], [])
            c.setVisible(False)
        for c in self._curves_i.values():
            c.setData([], [])
            c.setVisible(False)
        for c in self._curves_ca.values():
            c.setData([], [])
            c.setVisible(False)
        # Reference curve is always initialised; just hide it
        self._ref_curve_v.setData([], [])
        self._ref_curve_v.setVisible(False)
        self._ref_label.setVisible(False)
        # Hide ghost curves
        for c in self._ghost_curves.values():
            c.setData([], [])
            c.setVisible(False)
        # Hide time marker line
        self._time_marker_line.setVisible(False)

        self._apply_grid_alpha()
        self._set_default_titles()

    # ─────────────────────────────────────────────────────────────────
    #  UPDATE — single result
    # ─────────────────────────────────────────────────────────────────
    def update_plots(self, result):
        """
        Update all oscilloscope panes to display data from a single simulation result.
        
        Parameters:
            result: An object representing a single simulation result. Expected attributes:
                - t: 1D array of time points (ms).
                - v_soma: 1D array of soma membrane potential values.
                - v_all: 2D array of membrane potentials per compartment (shape: n_comp × n_time).
                - v_dendritic_filtered: 1D array or None for a filtered dendritic/stimulus trace.
                - currents: dict mapping current names to 1D or 2D arrays (if 2D, summed across compartments).
                - ca_i: 2D array or None for calcium concentrations (units: M); first row is used.
                - atp_estimate: numeric ATP estimate used for currents pane title.
                - n_comp: integer number of compartments.
                - config.morphology: morphology object used to resolve delay targets.
        
        Behavior:
            - Stores the given result for later reference and refreshes the voltage, gate, current,
              and calcium plots to reflect its data.
            - Preserves or updates a persistent reference voltage trace when the "keep reference"
              option is enabled.
            - Downsamples traces to the current viewport budgets, updates persistent plot items,
              and manages transient plot items (threshold line, spike markers, delay markers,
              Monte-Carlo cloud/mean bands are not affected by this function).
            - Updates checkbox visibility and delay-control state based on the result's contents.
            - Applies or restores publication-style scale bars when publication mode is active.
            - Updates plot titles (spike count, firing rate, delay tag, ATP estimate) and autoscaling.
        
        Notes:
            - This function mutates widget state and plot items; it does not return a value.
        """
        self._last_result = result
        self._last_mc_results = None

        # --- Handle Reference Trace (Comparison Mode) ---
        # Capture reference data BEFORE clearing current plots.
        # Uses setData/setVisible on the persistent curve — never removeItem/plot.
        if self._cb_keep_reference.isChecked():
            if "Soma" in self._curves_v:
                old_t, old_v = self._curves_v["Soma"].getData()
                if old_t is not None and old_v is not None and len(old_t) > 0:
                    self._ref_curve_v.setData(old_t, old_v)
                    self._ref_curve_v.setVisible(True)
            # Store full currents dictionary for Ghost Currents feature
            if hasattr(result, 'currents') and result.currents:
                self._ref_currents = dict(result.currents)
                # Store time array for reference
                if hasattr(result, 't'):
                    self._ref_t = np.asarray(result.t)
        else:
            self._ref_curve_v.setData([], [])
            self._ref_curve_v.setVisible(False)
            self._ref_currents = {}
            self._ref_t = None

        self.clear()
        self._sync_delay_controls(result)
        theme = PLOT_THEMES.get(self._theme_name, PLOT_THEMES["Default"])
        
        # --- Reset All Active Curves (The Overlap Bug Fix) ---
        # Before drawing new data, hide and clear ALL existing dynamic curves
        all_curves = (list(self._curves_v.values()) + 
                      list(self._curves_gate.values()) + 
                      list(self._curves_i.values()) + 
                      list(self._curves_ca.values()))
        
        for curve in all_curves:
            curve.setVisible(False)
            curve.setData([], [])
        
        lw = self._line_width_scale
        t   = result.t
        n   = result.n_comp
        mc  = result.config.morphology
        budget_v = _plot_point_budget(self._p_v)
        budget_g = _plot_point_budget(self._p_g)
        budget_i = _plot_point_budget(self._p_i)

        # ── Voltage traces ────────────────────────────────────────────
        soma_pen = pg.mkPen(color=theme["soma"], width=2.0 * lw)
        t_soma, v_soma = _downsample_xy(t, result.v_soma, max_points=budget_v)
        c_soma = self._curves_v.get("Soma")
        if c_soma is None:
            c_soma = self._p_v.plot([], [], pen=soma_pen, name="Soma")
            self._curves_v["Soma"] = c_soma
        else:
            c_soma.setPen(soma_pen)
        c_soma.setData(t_soma, v_soma)
        c_soma.setVisible(self._cb_v["Soma"].isChecked())  # Respect user checkbox

        if n > 1 and mc.N_ais > 0:
            ais_pen = pg.mkPen(color=theme["ais"], width=1.5 * lw,
                               style=Qt.PenStyle.DashLine)
            t_ais, v_ais = _downsample_xy(t, result.v_all[1, :], max_points=budget_v)
            c_ais = self._curves_v.get("AIS")
            if c_ais is None:
                c_ais = self._p_v.plot([], [], pen=ais_pen, name="AIS")
                self._curves_v["AIS"] = c_ais
            else:
                c_ais.setPen(ais_pen)
            c_ais.setData(t_ais, v_ais)
            c_ais.setVisible(self._cb_v["AIS"].isChecked())  # Respect user checkbox
        elif "AIS" in self._curves_v:
            self._curves_v["AIS"].setData([], [])
            self._curves_v["AIS"].setVisible(False)

        if n > 2:
            term_pen = pg.mkPen(color=theme["terminal"], width=1.2 * lw,
                                style=Qt.PenStyle.DotLine)
            t_term, v_term = _downsample_xy(t, result.v_all[-1, :], max_points=budget_v)
            c_term = self._curves_v.get("Terminal")
            if c_term is None:
                c_term = self._p_v.plot([], [], pen=term_pen, name="Terminal")
                self._curves_v["Terminal"] = c_term
            else:
                c_term.setPen(term_pen)
            c_term.setData(t_term, v_term)
            c_term.setVisible(self._cb_v["Terminal"].isChecked())  # Respect user checkbox
        elif "Terminal" in self._curves_v:
            self._curves_v["Terminal"].setData([], [])
            self._curves_v["Terminal"].setVisible(False)

        # ── Threshold line at _THRESHOLD_MV ──────────────────────────
        thresh_line = pg.InfiniteLine(
            pos=_THRESHOLD_MV, angle=0,
            pen=pg.mkPen(QColor(theme["threshold"]), width=max(1.0, 1.0 * lw),
                         style=Qt.PenStyle.DashLine),
            label=f'{_THRESHOLD_MV:+.0f} mV',
            labelOpts={'position': 0.02, 'color': theme["threshold"],
                       'anchors': [(0, 1), (0, 1)]}
        )
        self._p_v.addItem(thresh_line)
        self._transient_items.append(thresh_line)

        # ── Spike markers ─────────────────────────────────────────────
        from core.analysis import detect_spikes
        pks, sp_t, _sp_amp = detect_spikes(result.v_soma, t,
                                            threshold=_THRESHOLD_MV)
        n_spikes = len(sp_t)
        spike_pen = pg.mkPen(QColor(theme["spike"]), width=max(1.0, 1.0 * lw),
                              style=Qt.PenStyle.DotLine)
        dense_mode = bool(self._cb_dense_spike_mode.isChecked())
        suppress_markers = dense_mode and (n_spikes > 80 or (n_spikes > 0 and len(t) > 12000))
        if self._cb_show_spike_markers.isChecked() and not suppress_markers:
            # Cap markers to avoid visual clutter at very high firing rates
            markers_t = sp_t if n_spikes <= _MAX_SPIKE_MARKERS else sp_t[::max(1, n_spikes // _MAX_SPIKE_MARKERS)]
            for t_sp in markers_t:
                spike_line = pg.InfiniteLine(pos=t_sp, angle=90, pen=spike_pen)
                self._p_v.addItem(spike_line)
                self._transient_items.append(spike_line)

        delay_tag = ""
        if self._cb_show_delay.isChecked() and n > 1:
            target_idx, target_label, target_color_key = self._resolve_delay_target(result)
            t_soma = self._first_crossing_time(result.v_soma, t, _THRESHOLD_MV)
            if target_idx is not None:
                t_target = self._first_crossing_time(result.v_all[target_idx, :], t, _THRESHOLD_MV)
            else:
                t_target = float("nan")
            if np.isfinite(t_soma) and np.isfinite(t_target) and t_target >= t_soma:
                delay_ms = t_target - t_soma
                delay_tag = f" | delay soma->{target_label} {delay_ms:.2f} ms"
                soma_delay_line = pg.InfiniteLine(
                    pos=t_soma,
                    angle=90,
                    pen=pg.mkPen(theme["soma"], width=max(1.0, 1.0 * lw), style=Qt.PenStyle.DashLine),
                )
                target_delay_line = pg.InfiniteLine(
                    pos=t_target,
                    angle=90,
                    pen=pg.mkPen(theme[target_color_key], width=max(1.0, 1.0 * lw), style=Qt.PenStyle.DashLine),
                )
                self._p_v.addItem(soma_delay_line)
                self._p_v.addItem(target_delay_line)
                self._transient_items.append(soma_delay_line)
                self._transient_items.append(target_delay_line)

        # ── V(t) title with spike stats ───────────────────────────────
        duration_s = t[-1] / 1000.0
        if n_spikes > 0 and duration_s > 0:
            rate_hz   = n_spikes / duration_s
            title_tag = f"{n_spikes} spikes  |  {rate_hz:.1f} Hz"
        else:
            title_tag = "no spikes"
        if suppress_markers:
            title_tag += "  |  dense trace mode"
        self._p_v.setTitle(
            self._title_html(f"Membrane Potential  V(t)  |  {title_tag}{delay_tag}", "#89B4FA")
        )

        # ── Gate dynamics ─────────────────────────────────────────────
        from core.analysis import extract_gate_traces
        gates = extract_gate_traces(result)
        visible_gates: set[str] = set()
        for name in ('m', 'h', 'n'):
            if name in gates:
                col  = GATE_COLORS.get(name, (180, 180, 180, 200))
                pen  = pg.mkPen(color=col[:3], width=1.5 * lw)
                tg, yg = _downsample_xy(t, gates[name], max_points=budget_g)
                c = self._curves_gate.get(name)
                if c is None:
                    c = self._p_g.plot([], [], pen=pen, name=name)
                    self._curves_gate[name] = c
                else:
                    c.setPen(pen)
                c.setData(tg, yg)
                c.setVisible(self._cb_g[name].isChecked())
                visible_gates.add(name)
        for name, c in self._curves_gate.items():
            if name not in visible_gates:
                c.setData([], [])
                c.setVisible(False)

        # ── Ion currents ──────────────────────────────────────────────
        visible_currents: set[str] = set()
        for name, curr in result.currents.items():
            # Handle 2D current arrays (n_comp, n_time) - sum across compartments
            curr_arr = np.asarray(curr, dtype=float)
            if curr_arr.ndim == 2:
                curr_arr = np.sum(curr_arr, axis=0)
            
            col   = CHAN_COLORS.get(name, (120, 120, 120, 150))
            pen   = pg.mkPen(color=col[:3], width=1.5 * lw)
            brush = pg.mkBrush(col)
            tc, yc = _downsample_xy(t, curr_arr, max_points=budget_i)
            c = self._curves_i.get(name)
            if c is None:
                c = self._p_i.plot([], [], pen=pen, fillLevel=0.0, fillBrush=brush, name=f"I_{name}")
                self._curves_i[name] = c
            else:
                c.setPen(pen)
            c.setData(tc, yc)
            c.setVisible(self._cb_i.get(name, QCheckBox()).isChecked())
            visible_currents.add(name)

        # ── Reconstructed stimulus input trace ─────────────────────────
        from core.analysis import reconstruct_stimulus_trace
        stim_input = reconstruct_stimulus_trace(result)
        stim_input_pen = pg.mkPen(
            color=theme["stim_input"],
            width=1.8 * lw,
            style=Qt.PenStyle.SolidLine,
        )
        t_si, y_si = _downsample_xy(t, stim_input, max_points=budget_i)
        c_si = self._curves_i.get("Stim_input")
        if c_si is None:
            c_si = self._p_i.plot([], [], pen=stim_input_pen, name="I_stim(input)")
            self._curves_i["Stim_input"] = c_si
        else:
            c_si.setPen(stim_input_pen)
        c_si.setData(t_si, y_si)
        c_si.setVisible(self._cb_i['Stim_input'].isChecked())
        visible_currents.add("Stim_input")

        # ── Filtered stimulus current ─────────────────────────────────
        # Show the post-filter state so user can compare with reconstructed input.
        if result.v_dendritic_filtered is not None:
            filt_curr_pen = pg.mkPen(color=theme["stim_filtered"], width=1.5 * lw,
                                      style=Qt.PenStyle.DashLine)
            t_sf, y_sf = _downsample_xy(t, result.v_dendritic_filtered, max_points=budget_i)
            c_sf = self._curves_i.get("Stim_filtered")
            if c_sf is None:
                c_sf = self._p_i.plot([], [], pen=filt_curr_pen, name="I_stim(filt)")
                self._curves_i["Stim_filtered"] = c_sf
            else:
                c_sf.setPen(filt_curr_pen)
            c_sf.setData(t_sf, y_sf)
            c_sf.setVisible(self._cb_i['Stim_filtered'].isChecked())
            visible_currents.add("Stim_filtered")
        elif "Stim_filtered" in self._curves_i:
            self._curves_i["Stim_filtered"].setData([], [])
            self._curves_i["Stim_filtered"].setVisible(False)

        # ── Ghost Currents (Comparative Biophysics) ───────────────────
        # Draw reference current traces as thin, semi-transparent dashed lines
        if self._ref_currents and self._cb_keep_reference.isChecked():
            for name, ref_curr in self._ref_currents.items():
                if name in ('Stim_input', 'Stim_filtered'):
                    continue
                # Get color from CHAN_COLORS with alpha 0.3
                base_col = CHAN_COLORS.get(name, (120, 120, 120, 150))
                ghost_col = (base_col[0], base_col[1], base_col[2], 77)  # 77 ≈ 0.3 * 255
                ghost_pen = pg.mkPen(
                    color=ghost_col,
                    width=1.0 * lw,
                    style=Qt.PenStyle.DashLine
                )
                # Handle 2D arrays
                ref_arr = np.asarray(ref_curr, dtype=float)
                if ref_arr.ndim == 2:
                    ref_arr = np.sum(ref_arr, axis=0)
                # Use stored reference time or interpolate
                if hasattr(self, '_ref_t') and self._ref_t is not None:
                    t_ref = self._ref_t
                    if len(t_ref) == len(ref_arr):
                        t_ghost, y_ghost = _downsample_xy(t_ref, ref_arr, max_points=budget_i)
                    else:
                        continue
                else:
                    continue
                # Get or create ghost curve
                ghost_name = f"ghost_{name}"
                c_ghost = self._ghost_curves.get(ghost_name)
                if c_ghost is None:
                    c_ghost = self._p_i.plot([], [], pen=ghost_pen, name=f"Ref_{name}")
                    self._ghost_curves[ghost_name] = c_ghost
                else:
                    c_ghost.setPen(ghost_pen)
                c_ghost.setData(t_ghost, y_ghost)
                # Show ghost only if corresponding main current is visible
                c_ghost.setVisible(self._cb_i.get(name, QCheckBox()).isChecked())

        # Hide unused ghost curves
        for name, c in self._ghost_curves.items():
            base_name = name.replace("ghost_", "")
            if base_name not in visible_currents or not self._cb_keep_reference.isChecked():
                c.setData([], [])
                c.setVisible(False)

        # ── Ghost Currents Metrics (Quantitative Comparison) ──────────
        self._update_ghost_metrics(result)

        for name, c in self._curves_i.items():
            if name not in visible_currents:
                c.setData([], [])
                c.setVisible(False)

        # Sync checkbox visibility
        self._sync_checkboxes(result)

        # -- Calcium trace (persistent artist) --
        if result.ca_i is not None:
            ca_pen = pg.mkPen(color="#F38BA8", width=2.0 * lw)
            t_ca, ca_nM = _downsample_xy(t, result.ca_i[0, :] * 1e6, max_points=budget_i)
            c_ca = self._curves_ca.get('calcium')
            if c_ca is None:
                c_ca = self._p_ca.plot([], [], pen=ca_pen, name="[Ca²+]_i")
                self._curves_ca['calcium'] = c_ca
            else:
                c_ca.setPen(ca_pen)
            c_ca.setData(t_ca, ca_nM)
            c_ca.setVisible(self._cb_ca['calcium'].isChecked())
            self._p_ca.autoRange()
        else:
            if 'calcium' in self._curves_ca:
                self._curves_ca['calcium'].setData([], [])
                self._curves_ca['calcium'].setVisible(False)

        # ── Currents title with ATP estimate ──────────────────────────
        self._p_i.setTitle(
            self._title_html(
                f"Ion Currents (soma)  |  ATP ≈ {result.atp_estimate:.2e} nmol/cm²",
                "#FAB387",
            )
        )

        # Apply scale bar mode
        if self._scale_bar_mode:
            self._apply_scale_bar_mode(result)
        else:
            self._restore_normal_mode()

        self._p_v.enableAutoRange(axis=pg.ViewBox.XAxis)
        self._p_g.enableAutoRange(axis=pg.ViewBox.XAxis)
        self._p_i.enableAutoRange(axis=pg.ViewBox.XAxis)
        self._p_ca.enableAutoRange(axis=pg.ViewBox.XAxis)
        # Keep Y autoscale for new peaks.
        self._p_v.autoRange()
        self._p_g.autoRange()
        self._p_i.autoRange()
        self._update_row_visibility()
    
    def _apply_scale_bar_mode(self, result):
        """
        Hide axis ticks/labels on voltage, gate, and current plots and add publication-style scale bars to the voltage plot.
        
        Adds a horizontal time scale bar representing 50 ms and a vertical voltage scale bar representing 20 mV. Bars are placed near the lower-right region of the voltage plot using the provided result's time range; any previously added scale-bar items are removed before adding new ones.
        
        Parameters:
            result: An object with a `t` attribute (array-like, time in milliseconds) used to determine the horizontal placement of the scale bars.
        """
        # Hide axes
        for plot in [self._p_v, self._p_g, self._p_i]:
            plot.hideAxis('left')
            plot.hideAxis('bottom')
            plot.hideAxis('right')
            plot.hideAxis('top')
            plot.getPlotItem().hideAxis('left')
            plot.getPlotItem().hideAxis('bottom')
        
        # Add scale bars to voltage plot (bottom-right corner)
        if hasattr(self, '_scale_bar_h') and self._scale_bar_h:
            self._p_v.removeItem(self._scale_bar_h)
            self._scale_bar_h = None
        if hasattr(self, '_scale_bar_v') and self._scale_bar_v:
            self._p_v.removeItem(self._scale_bar_v)
            self._scale_bar_v = None
        
        # Time scale bar (50 ms)
        t = result.t
        if len(t) > 0:
            t_range = t[-1] - t[0]
            bar_x_start = t_range * 0.7
            bar_x_end = bar_x_start + 50.0  # 50 ms
            bar_y = -90  # Near bottom of voltage range
            
            self._scale_bar_h = pg.PlotCurveItem(
                [bar_x_start, bar_x_end], [bar_y, bar_y],
                pen=pg.mkPen(color='white', width=3)
            )
            self._p_v.addItem(self._scale_bar_h)
            
            # Voltage scale bar (20 mV)
            bar_y_start = -80
            bar_y_end = bar_y_start + 20.0  # 20 mV
            bar_x = bar_x_start
            
            self._scale_bar_v = pg.PlotCurveItem(
                [bar_x, bar_x], [bar_y_start, bar_y_end],
                pen=pg.mkPen(color='white', width=3)
            )
            self._p_v.addItem(self._scale_bar_v)
    
    def _restore_normal_mode(self):
        """
        Restore regular plot axes and remove any scale-bar items added to publication (scale-bar) mode.
        
        This re-shows the left and bottom axes for the voltage, gates, and currents plots and, if horizontal or vertical scale-bar items exist on the voltage plot, removes them and clears their references.
        """
        for plot in [self._p_v, self._p_g, self._p_i]:
            plot.showAxis('left')
            plot.showAxis('bottom')
        
        # Remove scale bars
        if hasattr(self, '_scale_bar_h') and self._scale_bar_h:
            self._p_v.removeItem(self._scale_bar_h)
            self._scale_bar_h = None
        if hasattr(self, '_scale_bar_v') and self._scale_bar_v:
            self._p_v.removeItem(self._scale_bar_v)
            self._scale_bar_v = None

    # ─────────────────────────────────────────────────────────────────
    #  UPDATE — Monte-Carlo cloud
    # ─────────────────────────────────────────────────────────────────
    def update_plots_mc(self, results_list):
        self._last_result = None
        self._last_mc_results = results_list
        self.clear()
        theme = PLOT_THEMES.get(self._theme_name, PLOT_THEMES["Default"])
        lw = self._line_width_scale
        budget_v = _plot_point_budget(self._p_v)

        def _with_alpha(color_val, alpha):
            if isinstance(color_val, str):
                q = QColor(color_val)
            elif isinstance(color_val, tuple):
                q = QColor(*color_val[:3])
            else:
                q = QColor(180, 180, 180)
            q.setAlpha(alpha)
            return q

        cloud_color = _with_alpha(theme["soma"], 40)
        mean_color = theme["ais"]
        band_color = _with_alpha(theme["ais"], 110)

        for res in results_list:
            t_cloud, v_cloud = _downsample_xy(res.t, res.v_soma, max_points=budget_v)
            cloud_line = self._p_v.plot(t_cloud, v_cloud,
                                        pen=pg.mkPen(cloud_color, width=1.0 * lw))
            self._transient_items.append(cloud_line)

        all_v  = np.array([r.v_soma for r in results_list])
        mean_v = np.mean(all_v, axis=0)
        std_v  = np.std(all_v, axis=0)
        t      = results_list[0].t

        # Mean ± std band
        t_mean, y_mean = _downsample_xy(t, mean_v, max_points=budget_v)
        t_plus, y_plus = _downsample_xy(t, mean_v + std_v, max_points=budget_v)
        t_minus, y_minus = _downsample_xy(t, mean_v - std_v, max_points=budget_v)
        mean_line = self._p_v.plot(t_mean, y_mean,
                                   pen=pg.mkPen(mean_color, width=2.5 * lw), name="Mean V(t)")
        plus_line = self._p_v.plot(t_plus, y_plus,
                                   pen=pg.mkPen(band_color, width=1.0 * lw,
                                                style=Qt.PenStyle.DashLine),
                                   name="Mean ± σ")
        minus_line = self._p_v.plot(t_minus, y_minus,
                                    pen=pg.mkPen(band_color, width=1.0 * lw,
                                                 style=Qt.PenStyle.DashLine))
        self._transient_items.extend([mean_line, plus_line, minus_line])

        # Threshold line
        thresh_line = pg.InfiniteLine(
            pos=_THRESHOLD_MV, angle=0,
            pen=pg.mkPen(QColor(theme["threshold"]), width=max(1.0, 1.0 * lw),
                         style=Qt.PenStyle.DashLine),
            label=f'{_THRESHOLD_MV:+.0f} mV',
            labelOpts={'position': 0.02, 'color': theme["threshold"],
                       'anchors': [(0, 1), (0, 1)]}
        )
        self._p_v.addItem(thresh_line)
        self._transient_items.append(thresh_line)

        n = len(results_list)
        self._p_v.setTitle(self._title_html(f"Monte-Carlo: {n} trials — mean ± σ", "#89B4FA"))
        self._p_i.setTitle(self._title_html(f"Monte-Carlo: {n} trials — mean ± σ", "#FAB387"))
        self._p_v.autoRange()
        self._p_ca.autoRange()

    # ─────────────────────────────────────────────────────────────────
    def _update_ghost_metrics(self, result):
        """Compute and display quantitative ghost currents comparison metrics."""
        if not hasattr(self, '_lbl_ghost_metrics'):
            return

        # Only show metrics if reference exists and checkbox is checked
        if not self._cb_keep_reference.isChecked() or not self._ref_currents:
            self._lbl_ghost_metrics.setText("")
            return

        import numpy as np

        metrics_lines = []
        correlations = []
        rms_errors = []

        for name, ref_curr in self._ref_currents.items():
            if name in ('Stim_input', 'Stim_filtered'):
                continue

            # Get current data
            if not hasattr(result, 'currents') or name not in result.currents:
                continue

            curr_data = result.currents[name]
            ref_arr = np.asarray(ref_curr, dtype=float)
            curr_arr = np.asarray(curr_data, dtype=float)

            # Handle 2D arrays
            if ref_arr.ndim == 2:
                ref_arr = np.sum(ref_arr, axis=0)
            if curr_arr.ndim == 2:
                curr_arr = np.sum(curr_arr, axis=0)

            # Ensure same length
            min_len = min(len(ref_arr), len(curr_arr))
            if min_len < 2:
                continue

            ref_arr = ref_arr[:min_len]
            curr_arr = curr_arr[:min_len]

            # Compute metrics
            try:
                # Pearson correlation
                if np.std(ref_arr) > 0 and np.std(curr_arr) > 0:
                    corr = np.corrcoef(ref_arr, curr_arr)[0, 1]
                else:
                    corr = 0.0

                # RMS error (normalized by reference amplitude)
                ref_max = np.max(np.abs(ref_arr))
                if ref_max > 0:
                    rms = np.sqrt(np.mean((curr_arr - ref_arr)**2)) / ref_max * 100
                else:
                    rms = 0.0

                correlations.append(corr)
                rms_errors.append(rms)

                # Only show individual metrics for visible currents
                if name in self._cb_i and self._cb_i[name].isChecked():
                    metrics_lines.append(f"{name:6s}: r={corr:5.2f}  RMS={rms:5.1f}%")
            except Exception:
                continue

        # Summary statistics
        if correlations and rms_errors:
            avg_corr = np.mean(correlations)
            avg_rms = np.mean(rms_errors)

            summary = f"Ghost Comparison (vs Reference):\n"
            summary += f"Avg Correlation: {avg_corr:.3f}  |  Avg RMS: {avg_rms:.1f}%\n"
            summary += "─" * 35 + "\n"
            summary += "\n".join(metrics_lines[:5])  # Show top 5

            self._lbl_ghost_metrics.setText(summary)
        else:
            self._lbl_ghost_metrics.setText("")

    # ─────────────────────────────────────────────────────────────────
    def _sync_checkboxes(self, result):
        """Show only checkboxes for channels / traces present in result."""
        self._sync_delay_controls(result)
        for name, cb in self._cb_i.items():
            if name == 'Stim_input':
                cb.setVisible(True)
            elif name == 'Stim_filtered':
                cb.setVisible(result.v_dendritic_filtered is not None)
            else:
                cb.setVisible(name in result.currents)

        for name, cb in self._cb_v.items():
            if name == 'AIS':
                cb.setVisible(result.n_comp > 1 and result.config.morphology.N_ais > 0)
            elif name == 'Terminal':
                cb.setVisible(result.n_comp > 2)

    def export_plot(self, path: str) -> tuple[bool, str]:
        """
        Export current oscilloscope widget view.

        Supported:
        - PNG/JPG/BMP (raster snapshot)
        - SVG (vector via QSvgGenerator)
        - PDF (vector-like Qt render to PDF page)
        """
        suffix = path.lower().rsplit(".", 1)[-1] if "." in path else ""
        if suffix in {"png", "jpg", "jpeg", "bmp"}:
            ok = self._win.grab().save(path)
            return (ok, "" if ok else "Failed to save raster image.")

        if suffix == "svg":
            try:
                from PySide6.QtSvg import QSvgGenerator
            except Exception:
                return False, "SVG export is unavailable (QtSvg module missing)."
            gen = QSvgGenerator()
            gen.setFileName(path)
            gen.setSize(self._win.size())
            gen.setViewBox(self._win.rect())
            gen.setTitle("NeuroModelPort Oscilloscope")
            painter = QPainter(gen)
            try:
                self._win.render(painter)
            finally:
                painter.end()
            return True, ""

        if suffix == "pdf":
            writer = QPdfWriter(path)
            writer.setResolution(300)
            writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
            painter = QPainter(writer)
            try:
                self._win.render(painter)
            finally:
                painter.end()
            return True, ""

        return False, f"Unsupported format: {suffix}. Use PNG, SVG, or PDF."
