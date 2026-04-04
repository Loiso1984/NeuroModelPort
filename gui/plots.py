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
                                QCheckBox, QGroupBox, QComboBox, QDoubleSpinBox, QLabel, QSpinBox, QPushButton, QMainWindow)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPageSize, QPdfWriter

# Colour scheme (matches analytics.py)
CHAN_COLORS = {
    'Na':   (220, 50,  50,  200),
    'K':    (50,  100, 220, 200),
    'Leak': (50,  160, 50,  150),
    'Ih':   (160, 50,  200, 180),
    'ICa':  (250, 150, 0,   200),
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

PLOT_THEMES = {
    "Default": {
        "soma": "#4080FF",
        "ais": "#FF4040",
        "terminal": (0, 200, 100),
        "threshold": "#F9E2AF",
        "spike": "#F38BA8",
        "stim_input": "#F5C2E7",
        "stim_filtered": "#89B4FA",
    },
    "High Contrast": {
        "soma": "#00D1FF",
        "ais": "#FF2A2A",
        "terminal": "#00FF7F",
        "threshold": "#FFD700",
        "spike": "#FF66CC",
        "stim_input": "#FFC0CB",
        "stim_filtered": "#8EC5FF",
    },
    "Colorblind Friendly": {
        "soma": "#0072B2",
        "ais": "#D55E00",
        "terminal": "#009E73",
        "threshold": "#F0E442",
        "spike": "#CC79A7",
        "stim_input": "#E69F00",
        "stim_filtered": "#56B4E9",
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

    def __init__(self, parent=None):
        super().__init__(parent)
        self._theme_name = "Default"
        self._line_width_scale = 1.0
        self._grid_alpha = 0.2
        self._title_font_px = 13
        self._delay_target_name = "Terminal"
        self._delay_custom_index = 1
        self._last_result = None
        self._last_mc_results = None
        self._fullscreen_windows = []
        self._build_ui()
        self._curves_v:    dict = {}
        self._curves_gate: dict = {}
        self._curves_i:    dict = {}

    # ─────────────────────────────────────────────────────────────────
    def _title_html(self, text: str, color: str) -> str:
        return f"<span style='color:{color}; font-size:{self._title_font_px}px'>{text}</span>"

    def _apply_grid_alpha(self):
        self._p_v.showGrid(x=True, y=True, alpha=self._grid_alpha)
        self._p_g.showGrid(x=True, y=True, alpha=self._grid_alpha)
        self._p_i.showGrid(x=True, y=True, alpha=self._grid_alpha)

    def _set_default_titles(self):
        self._p_v.setTitle(self._title_html("Membrane Potential  V(t)", "#89B4FA"))
        self._p_g.setTitle(self._title_html("Gate Variables  m, h, n", "#A6E3A1"))
        self._p_i.setTitle(self._title_html("Ion Currents  (soma)", "#FAB387"))

    # ─────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        # ── Plot area ─────────────────────────────────────────────────
        self._win = pg.GraphicsLayoutWidget()
        self._win.setBackground('#0D1117')

        # V(t) — row 0
        self._p_v = self._win.addPlot(
            title=self._title_html("Membrane Potential  V(t)", "#89B4FA"))
        self._p_v.setLabel('left', 'V', units='mV', color='#CDD6F4')
        self._p_v.showGrid(x=True, y=True, alpha=self._grid_alpha)
        self._p_v.addLegend(offset=(10, 10), labelTextColor='#CDD6F4')
        self._win.nextRow()

        # Gate variables — row 1
        self._p_g = self._win.addPlot(
            title=self._title_html("Gate Variables  m, h, n", "#A6E3A1"))
        self._p_g.setLabel('left', 'probability  [0–1]', color='#CDD6F4')
        self._p_g.showGrid(x=True, y=True, alpha=self._grid_alpha)
        self._p_g.addLegend(offset=(10, 10), labelTextColor='#CDD6F4')
        self._p_g.setXLink(self._p_v)
        self._p_g.setYRange(-0.05, 1.05)
        self._win.nextRow()

        # Currents — row 2
        self._p_i = self._win.addPlot(
            title=self._title_html("Ion Currents  (soma)", "#FAB387"))
        self._p_i.setLabel('left', 'I', units='µA/cm²', color='#CDD6F4')
        self._p_i.setLabel('bottom', 'Time', units='ms', color='#CDD6F4')
        self._p_i.showGrid(x=True, y=True, alpha=self._grid_alpha)
        self._p_i.addLegend(offset=(10, 10), labelTextColor='#CDD6F4')
        self._p_i.setXLink(self._p_v)

        # Row stretch: V(t) is tallest (3 units), gates and currents share remaining (2 each)
        self._win.ci.layout.setRowStretchFactor(0, 3)
        self._win.ci.layout.setRowStretchFactor(1, 2)
        self._win.ci.layout.setRowStretchFactor(2, 2)

        root.addWidget(self._win, stretch=10)

        # ── Checkbox panel ────────────────────────────────────────────
        cb_widget = QWidget()
        cb_widget.setMaximumWidth(210)
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

        # View controls
        grp_view = QGroupBox("View")
        grp_view.setStyleSheet("QGroupBox { color: #94E2D5; font-size:11px; }")
        vl2 = QVBoxLayout(grp_view)

        lbl_theme = QLabel("Theme")
        lbl_theme.setStyleSheet("color:#CDD6F4; font-size:11px;")
        self._combo_theme = QComboBox()
        self._combo_theme.addItems(list(PLOT_THEMES.keys()))
        self._combo_theme.setCurrentText(self._theme_name)
        self._combo_theme.currentTextChanged.connect(self._on_view_settings_changed)
        vl2.addWidget(lbl_theme)
        vl2.addWidget(self._combo_theme)

        lbl_lw = QLabel("Line Width Scale")
        lbl_lw.setStyleSheet("color:#CDD6F4; font-size:11px;")
        self._spin_line_width = QDoubleSpinBox()
        self._spin_line_width.setRange(0.7, 3.0)
        self._spin_line_width.setSingleStep(0.1)
        self._spin_line_width.setDecimals(1)
        self._spin_line_width.setValue(self._line_width_scale)
        self._spin_line_width.valueChanged.connect(self._on_view_settings_changed)
        vl2.addWidget(lbl_lw)
        vl2.addWidget(self._spin_line_width)

        lbl_title = QLabel("Title Font (px)")
        lbl_title.setStyleSheet("color:#CDD6F4; font-size:11px;")
        self._spin_title_px = QSpinBox()
        self._spin_title_px.setRange(10, 24)
        self._spin_title_px.setValue(self._title_font_px)
        self._spin_title_px.valueChanged.connect(self._on_view_settings_changed)
        vl2.addWidget(lbl_title)
        vl2.addWidget(self._spin_title_px)

        lbl_grid = QLabel("Grid Alpha")
        lbl_grid.setStyleSheet("color:#CDD6F4; font-size:11px;")
        self._spin_grid_alpha = QDoubleSpinBox()
        self._spin_grid_alpha.setRange(0.05, 0.60)
        self._spin_grid_alpha.setSingleStep(0.05)
        self._spin_grid_alpha.setDecimals(2)
        self._spin_grid_alpha.setValue(self._grid_alpha)
        self._spin_grid_alpha.valueChanged.connect(self._on_view_settings_changed)
        vl2.addWidget(lbl_grid)
        vl2.addWidget(self._spin_grid_alpha)

        self._cb_show_spike_markers = QCheckBox("Show spike markers")
        self._cb_show_spike_markers.setChecked(True)
        self._cb_show_spike_markers.setStyleSheet("color:#CDD6F4; font-size:11px;")
        self._cb_show_spike_markers.stateChanged.connect(self._on_view_settings_changed)
        vl2.addWidget(self._cb_show_spike_markers)

        self._cb_show_delay = QCheckBox("Show soma delay overlay")
        self._cb_show_delay.setChecked(True)
        self._cb_show_delay.setStyleSheet("color:#CDD6F4; font-size:11px;")
        self._cb_show_delay.stateChanged.connect(self._on_view_settings_changed)
        vl2.addWidget(self._cb_show_delay)

        lbl_delay_target = QLabel("Delay Target")
        lbl_delay_target.setStyleSheet("color:#CDD6F4; font-size:11px;")
        self._combo_delay_target = QComboBox()
        self._combo_delay_target.addItems(
            ["Terminal", "AIS", "Trunk Junction", "Custom Compartment"]
        )
        self._combo_delay_target.setCurrentText(self._delay_target_name)
        self._combo_delay_target.currentTextChanged.connect(self._on_view_settings_changed)
        vl2.addWidget(lbl_delay_target)
        vl2.addWidget(self._combo_delay_target)

        lbl_delay_comp = QLabel("Custom Comp Index")
        lbl_delay_comp.setStyleSheet("color:#CDD6F4; font-size:11px;")
        self._spin_delay_comp = QSpinBox()
        self._spin_delay_comp.setRange(1, 1)
        self._spin_delay_comp.setValue(self._delay_custom_index)
        self._spin_delay_comp.setEnabled(False)
        self._spin_delay_comp.valueChanged.connect(self._on_view_settings_changed)
        vl2.addWidget(lbl_delay_comp)
        vl2.addWidget(self._spin_delay_comp)

        self._btn_fullscreen = QPushButton("Full Screen")
        self._btn_fullscreen.setToolTip("Open oscilloscope plots in a maximized window")
        self._btn_fullscreen.clicked.connect(self.open_fullscreen)
        vl2.addWidget(self._btn_fullscreen)

        cb_layout.addWidget(grp_view)
        cb_layout.addStretch()

        root.addWidget(cb_widget, stretch=1)

    def _copy_view_state_to(self, other: "OscilloscopeWidget"):
        """Copy current visual toggles/settings into another oscilloscope widget."""
        other._combo_theme.setCurrentText(self._combo_theme.currentText())
        other._spin_line_width.setValue(self._spin_line_width.value())
        other._spin_title_px.setValue(self._spin_title_px.value())
        other._spin_grid_alpha.setValue(self._spin_grid_alpha.value())
        other._cb_show_spike_markers.setChecked(self._cb_show_spike_markers.isChecked())
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

    def _toggle_g(self, name):
        if name in self._curves_gate:
            self._curves_gate[name].setVisible(self._cb_g[name].isChecked())

    def _toggle_i(self, name):
        if name in self._curves_i:
            self._curves_i[name].setVisible(self._cb_i[name].isChecked())

    def _on_view_settings_changed(self, *_):
        self._theme_name = self._combo_theme.currentText()
        self._line_width_scale = float(self._spin_line_width.value())
        self._title_font_px = int(self._spin_title_px.value())
        self._grid_alpha = float(self._spin_grid_alpha.value())
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
        if n <= 1:
            return None, "", "terminal"

        target = self._delay_target_name
        if target == "AIS":
            if n > 1 and int(mc.N_ais) > 0:
                return 1, "AIS", "ais"
            return None, "", "ais"
        if target == "Trunk Junction":
            if int(mc.N_trunk) > 0:
                j = 1 + int(mc.N_ais) + int(mc.N_trunk) - 1
            elif int(mc.N_ais) > 0:
                j = int(mc.N_ais)
            else:
                j = 0
            if 1 <= j < n:
                return j, "junction", "terminal"
            return None, "", "terminal"
        if target == "Custom Compartment":
            idx = int(self._delay_custom_index)
            idx = max(1, min(n - 1, idx))
            return idx, f"comp[{idx}]", "terminal"
        return n - 1, "terminal", "terminal"

    def _sync_delay_controls_core(self, n: int, mc):
        self._cb_show_delay.setEnabled(n > 1)
        self._combo_delay_target.setEnabled(n > 1)
        max_idx = max(1, n - 1)
        self._spin_delay_comp.setRange(1, max_idx)
        if self._spin_delay_comp.value() > max_idx:
            self._spin_delay_comp.setValue(max_idx)
        has_ais = n > 1 and int(mc.N_ais) > 0
        if int(mc.N_trunk) > 0:
            j = 1 + int(mc.N_ais) + int(mc.N_trunk) - 1
        elif int(mc.N_ais) > 0:
            j = int(mc.N_ais)
        else:
            j = 0
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
        self._p_v.clear()
        self._p_g.clear()
        self._p_i.clear()
        self._curves_v.clear()
        self._curves_gate.clear()
        self._curves_i.clear()
        self._p_v.addLegend(offset=(10, 10), labelTextColor='#CDD6F4')
        self._p_g.addLegend(offset=(10, 10), labelTextColor='#CDD6F4')
        self._p_i.addLegend(offset=(10, 10), labelTextColor='#CDD6F4')
        self._apply_grid_alpha()
        self._set_default_titles()

    # ─────────────────────────────────────────────────────────────────
    #  UPDATE — single result
    # ─────────────────────────────────────────────────────────────────
    def update_plots(self, result):
        self._last_result = result
        self._last_mc_results = None
        self.clear()
        self._sync_delay_controls(result)
        theme = PLOT_THEMES.get(self._theme_name, PLOT_THEMES["Default"])
        lw = self._line_width_scale
        t   = result.t
        n   = result.n_comp
        mc  = result.config.morphology

        # ── Voltage traces ────────────────────────────────────────────
        soma_pen = pg.mkPen(color=theme["soma"], width=2.0 * lw)
        c_soma   = self._p_v.plot(t, result.v_soma, pen=soma_pen, name="Soma")
        self._curves_v['Soma'] = c_soma

        if n > 1 and mc.N_ais > 0:
            ais_pen = pg.mkPen(color=theme["ais"], width=1.5 * lw,
                               style=Qt.PenStyle.DashLine)
            c_ais   = self._p_v.plot(t, result.v_all[1, :], pen=ais_pen, name="AIS")
            self._curves_v['AIS'] = c_ais

        if n > 2:
            term_pen = pg.mkPen(color=theme["terminal"], width=1.2 * lw,
                                style=Qt.PenStyle.DotLine)
            c_term   = self._p_v.plot(t, result.v_all[-1, :],
                                       pen=term_pen, name="Terminal")
            self._curves_v['Terminal'] = c_term

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

        # ── Spike markers ─────────────────────────────────────────────
        from core.analysis import detect_spikes
        pks, sp_t, _sp_amp = detect_spikes(result.v_soma, t,
                                            threshold=_THRESHOLD_MV)
        n_spikes = len(sp_t)
        spike_pen = pg.mkPen(QColor(theme["spike"]), width=max(1.0, 1.0 * lw),
                              style=Qt.PenStyle.DotLine)
        if self._cb_show_spike_markers.isChecked():
            # Cap markers to avoid visual clutter at very high firing rates
            markers_t = sp_t if n_spikes <= _MAX_SPIKE_MARKERS else sp_t[::max(1, n_spikes // _MAX_SPIKE_MARKERS)]
            for t_sp in markers_t:
                self._p_v.addItem(pg.InfiniteLine(pos=t_sp, angle=90, pen=spike_pen))

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
                self._p_v.addItem(
                    pg.InfiniteLine(
                        pos=t_soma,
                        angle=90,
                        pen=pg.mkPen(theme["soma"], width=max(1.0, 1.0 * lw), style=Qt.PenStyle.DashLine),
                    )
                )
                self._p_v.addItem(
                    pg.InfiniteLine(
                        pos=t_target,
                        angle=90,
                        pen=pg.mkPen(theme[target_color_key], width=max(1.0, 1.0 * lw), style=Qt.PenStyle.DashLine),
                    )
                )

        # ── V(t) title with spike stats ───────────────────────────────
        duration_s = t[-1] / 1000.0
        if n_spikes > 0 and duration_s > 0:
            rate_hz   = n_spikes / duration_s
            title_tag = f"{n_spikes} spikes  |  {rate_hz:.1f} Hz"
        else:
            title_tag = "no spikes"
        self._p_v.setTitle(
            self._title_html(f"Membrane Potential  V(t)  |  {title_tag}{delay_tag}", "#89B4FA")
        )

        # ── Gate dynamics ─────────────────────────────────────────────
        from core.analysis import extract_gate_traces
        gates = extract_gate_traces(result)
        for name in ('m', 'h', 'n'):
            if name in gates:
                col  = GATE_COLORS.get(name, (180, 180, 180, 200))
                pen  = pg.mkPen(color=col[:3], width=1.5 * lw)
                c    = self._p_g.plot(t, gates[name], pen=pen, name=name)
                self._curves_gate[name] = c
                c.setVisible(self._cb_g[name].isChecked())

        # ── Ion currents ──────────────────────────────────────────────
        for name, curr in result.currents.items():
            col   = CHAN_COLORS.get(name, (120, 120, 120, 150))
            pen   = pg.mkPen(color=col[:3], width=1.5 * lw)
            brush = pg.mkBrush(col)
            c     = self._p_i.plot(t, curr, pen=pen,
                                    fillLevel=0.0, fillBrush=brush,
                                    name=f"I_{name}")
            self._curves_i[name] = c
            c.setVisible(self._cb_i.get(name, QCheckBox()).isChecked())

        # ── Reconstructed stimulus input trace ─────────────────────────
        from core.analysis import reconstruct_stimulus_trace
        stim_input = reconstruct_stimulus_trace(result)
        stim_input_pen = pg.mkPen(
            color=theme["stim_input"],
            width=1.8 * lw,
            style=Qt.PenStyle.SolidLine,
        )
        c_si = self._p_i.plot(t, stim_input, pen=stim_input_pen, name="I_stim(input)")
        self._curves_i['Stim_input'] = c_si
        c_si.setVisible(self._cb_i['Stim_input'].isChecked())

        # ── Filtered stimulus current ─────────────────────────────────
        # Show the post-filter state so user can compare with reconstructed input.
        if result.v_dendritic_filtered is not None:
            filt_curr_pen = pg.mkPen(color=theme["stim_filtered"], width=1.5 * lw,
                                      style=Qt.PenStyle.DashLine)
            c_sf = self._p_i.plot(t, result.v_dendritic_filtered,
                                   pen=filt_curr_pen, name="I_stim(filt)")
            self._curves_i['Stim_filtered'] = c_sf
            c_sf.setVisible(self._cb_i['Stim_filtered'].isChecked())

        # Sync checkbox visibility
        self._sync_checkboxes(result)

        # ── Currents title with ATP estimate ──────────────────────────
        self._p_i.setTitle(
            self._title_html(
                f"Ion Currents (soma)  |  ATP ≈ {result.atp_estimate:.2e} nmol/cm²",
                "#FAB387",
            )
        )

        self._p_v.autoRange()
        self._p_g.autoRange()
        self._p_i.autoRange()

    # ─────────────────────────────────────────────────────────────────
    #  UPDATE — Monte-Carlo cloud
    # ─────────────────────────────────────────────────────────────────
    def update_plots_mc(self, results_list):
        self._last_result = None
        self._last_mc_results = results_list
        self.clear()
        theme = PLOT_THEMES.get(self._theme_name, PLOT_THEMES["Default"])
        lw = self._line_width_scale

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
            self._p_v.plot(res.t, res.v_soma,
                           pen=pg.mkPen(cloud_color, width=1.0 * lw))

        all_v  = np.array([r.v_soma for r in results_list])
        mean_v = np.mean(all_v, axis=0)
        std_v  = np.std(all_v, axis=0)
        t      = results_list[0].t

        # Mean ± std band
        self._p_v.plot(t, mean_v,
                       pen=pg.mkPen(mean_color, width=2.5 * lw), name="Mean V(t)")
        self._p_v.plot(t, mean_v + std_v,
                       pen=pg.mkPen(band_color, width=1.0 * lw,
                                    style=Qt.PenStyle.DashLine),
                       name="Mean ± σ")
        self._p_v.plot(t, mean_v - std_v,
                       pen=pg.mkPen(band_color, width=1.0 * lw,
                                    style=Qt.PenStyle.DashLine))

        # Threshold line
        self._p_v.addItem(pg.InfiniteLine(
            pos=_THRESHOLD_MV, angle=0,
            pen=pg.mkPen(QColor(theme["threshold"]), width=max(1.0, 1.0 * lw),
                         style=Qt.PenStyle.DashLine),
            label=f'{_THRESHOLD_MV:+.0f} mV',
            labelOpts={'position': 0.02, 'color': theme["threshold"],
                       'anchors': [(0, 1), (0, 1)]}
        ))

        n = len(results_list)
        self._p_v.setTitle(self._title_html(f"Monte-Carlo: {n} trials — mean ± σ", "#89B4FA"))
        self._p_i.setTitle(self._title_html(f"Monte-Carlo: {n} trials — mean ± σ", "#FAB387"))
        self._p_v.autoRange()

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
