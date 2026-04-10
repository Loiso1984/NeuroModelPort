"""
gui/topology.py - Neuron Morphology Viewer v10.3

Draws a schematic of the multi-compartment neuron model:
Soma -> AIS -> Trunk/Fork -> Branch 1/2.

v10.3 additions:
- Delay-target focus synced with Oscilloscope delay target selector.
- Highlighted target compartment and index in morphology + info bar.
- Index mapping aligned with morphology builder indexing (fork = last trunk comp).
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QPushButton,
    QMainWindow,
    QCheckBox,
    QDoubleSpinBox,
    QSlider,
)
from .delay_target import resolve_delay_target


class TopologyWidget(QWidget):
    """Visual morphology schematic, updated after each run."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._win = pg.GraphicsLayoutWidget()
        self._win.setBackground("#0D1117")
        layout.addWidget(self._win, stretch=1)

        self._plot = self._win.addPlot()
        self._plot.setAspectLocked(False)
        self._plot.hideAxis("bottom")
        self._plot.hideAxis("left")
        self._plot.getViewBox().setMouseEnabled(x=True, y=True)
        self._plot.addLegend(offset=(10, 10), labelTextColor="#CDD6F4")

        self._info = QLabel("")
        self._info.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._info.setWordWrap(True)
        self._info.setStyleSheet(
            "color:#D9F99D; font-size:11px; padding:4px;"
        )
        layout.addWidget(self._info)

        self._delay_target_name = "Terminal"
        self._delay_custom_index = 1
        self._last_config = None
        self._last_dual_config = None
        self._fullscreen_windows = []
        self._show_labels = True
        self._show_indices = True
        self._high_contrast = False
        self._line_scale = 1.0
        self._font_scale = 1.0
        self._draw_items: list = []

        controls = QHBoxLayout()
        controls.setSpacing(6)

        self._cb_labels = QCheckBox("Labels")
        self._cb_labels.setChecked(True)
        self._cb_labels.toggled.connect(self._on_view_controls_changed)
        controls.addWidget(self._cb_labels)

        self._cb_indices = QCheckBox("Indices")
        self._cb_indices.setChecked(True)
        self._cb_indices.toggled.connect(self._on_view_controls_changed)
        controls.addWidget(self._cb_indices)

        self._cb_contrast = QCheckBox("High Contrast")
        self._cb_contrast.setChecked(False)
        self._cb_contrast.toggled.connect(self._on_view_controls_changed)
        controls.addWidget(self._cb_contrast)

        self._spin_line = QDoubleSpinBox()
        self._spin_line.setRange(0.8, 2.4)
        self._spin_line.setSingleStep(0.1)
        self._spin_line.setDecimals(1)
        self._spin_line.setValue(1.0)
        self._spin_line.setPrefix("Line×")
        self._spin_line.valueChanged.connect(self._on_view_controls_changed)
        controls.addWidget(self._spin_line)

        self._spin_font = QDoubleSpinBox()
        self._spin_font.setRange(0.8, 2.0)
        self._spin_font.setSingleStep(0.1)
        self._spin_font.setDecimals(1)
        self._spin_font.setValue(1.0)
        self._spin_font.setPrefix("Font×")
        self._spin_font.valueChanged.connect(self._on_view_controls_changed)
        controls.addWidget(self._spin_font)

        self._btn_reset_view = QPushButton("Reset View")
        self._btn_reset_view.clicked.connect(self._on_reset_view)
        controls.addWidget(self._btn_reset_view)

        self._btn_fullscreen = QPushButton("Full Screen")
        self._btn_fullscreen.setToolTip("Open topology view in a maximized window")
        self._btn_fullscreen.clicked.connect(self.open_fullscreen)
        controls.addWidget(self._btn_fullscreen)
        controls.addStretch(1)
        layout.addLayout(controls)

        # Time scrubber for heatmap
        self._slider_layout = QHBoxLayout()
        self._lbl_time = QLabel("t = 0.0 ms")
        self._lbl_time.setStyleSheet("color:#89B4FA; font-weight:bold; min-width:80px;")

        self._time_slider = QSlider(Qt.Orientation.Horizontal)
        self._time_slider.setEnabled(False)
        self._time_slider.valueChanged.connect(self._on_time_scrub)

        self._slider_layout.addWidget(self._lbl_time)
        self._slider_layout.addWidget(self._time_slider)
        layout.addLayout(self._slider_layout)

        self._heatmap_data = None
        self._heatmap_time = None
        self._soma_item = None   # direct reference for scrubber coloring
        
        # Enable mouse clicks for compartment selection
        self._plot.scene().sigMouseClicked.connect(self._on_plot_clicked)

    def _on_reset_view(self):
        self._plot.autoRange()

    def _on_view_controls_changed(self, *_):
        self._show_labels = bool(self._cb_labels.isChecked())
        self._show_indices = bool(self._cb_indices.isChecked())
        self._high_contrast = bool(self._cb_contrast.isChecked())
        self._line_scale = float(self._spin_line.value())
        self._font_scale = float(self._spin_font.value())
        self._win.setBackground("#070B12" if self._high_contrast else "#0D1117")
        if self._high_contrast:
            self._info.setStyleSheet(
                "color:#ECFCCB; font-size:12px; padding:4px;"
            )
        else:
            self._info.setStyleSheet(
                "color:#D9F99D; font-size:11px; padding:4px;"
            )
        if self._last_config is not None:
            self.draw_neuron(self._last_config, dual_config=self._last_dual_config)

    def open_fullscreen(self):
        """Open topology clone in a maximized window preserving last rendered state."""
        win = QMainWindow(self)
        win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        win.setWindowTitle("NeuroModelPort — Topology (Full Screen)")
        full = TopologyWidget()
        full._cb_labels.setChecked(self._cb_labels.isChecked())
        full._cb_indices.setChecked(self._cb_indices.isChecked())
        full._cb_contrast.setChecked(self._cb_contrast.isChecked())
        full._spin_line.setValue(self._spin_line.value())
        full._spin_font.setValue(self._spin_font.value())
        full.set_delay_focus(self._delay_target_name, self._delay_custom_index)
        if self._last_config is not None:
            full.draw_neuron(
                self._last_config,
                dual_config=self._last_dual_config,
                delay_target_name=self._delay_target_name,
                delay_custom_index=self._delay_custom_index,
            )
        win.setCentralWidget(full)
        win.showMaximized()
        self._fullscreen_windows.append(win)

        def _cleanup(*_):
            self._fullscreen_windows = [w for w in self._fullscreen_windows if w is not win]

        win.destroyed.connect(_cleanup)

    def set_delay_focus(self, target_name: str, custom_index: int = 1):
        """Update delay focus target and redraw if a morphology is already shown."""
        target_name = str(target_name or "Terminal")
        custom_index = max(1, int(custom_index))
        changed = (
            target_name != self._delay_target_name
            or custom_index != self._delay_custom_index
        )
        self._delay_target_name = target_name
        self._delay_custom_index = custom_index
        if changed and self._last_config is not None:
            self.draw_neuron(self._last_config, dual_config=self._last_dual_config)

    def _resolve_delay_target(
        self,
        *,
        n_comp_total: int,
        mc,
        idx_fork: int,
        idx_terminal: int,
    ):
        idx, label, _ = resolve_delay_target(
            target_name=self._delay_target_name,
            custom_index=self._delay_custom_index,
            n_comp=n_comp_total,
            n_ais=int(mc.N_ais),
            n_trunk=int(mc.N_trunk),
            terminal_idx=idx_terminal,
        )
        if idx is None:
            return None, "n/a"
        if label == "junction":
            return max(0, min(idx_fork, idx_terminal)), "fork"
        return idx, label

    def draw_neuron(
        self,
        config,
        dual_config=None,
        delay_target_name: str | None = None,
        delay_custom_index: int | None = None,
        result=None,
    ):
        """Draw neuron morphology with optional dual-stim overlay and delay focus."""
        if delay_target_name is not None:
            self._delay_target_name = str(delay_target_name)
        if delay_custom_index is not None:
            self._delay_custom_index = max(1, int(delay_custom_index))
        self._last_config = config
        self._last_dual_config = dual_config

        for item in self._draw_items:
            self._plot.removeItem(item)
        self._draw_items = []
        self._soma_item = None
        mc = config.morphology
        ch = config.channels

        from core.analysis import membrane_time_constant, space_constant

        lam_um = space_constant(mc.d_soma, mc.Ra, ch.gL) * 1e4
        tau_m = membrane_time_constant(ch.Cm, ch.gL)

        n_comp_total = 1 if mc.single_comp else int(
            1 + mc.N_ais + mc.N_trunk + mc.N_b1 + mc.N_b2
        )
        idx_soma = 0
        idx_ais_start = 1
        idx_ais_end = idx_ais_start + max(0, int(mc.N_ais)) - 1
        idx_trunk_start = 1 + int(mc.N_ais)
        idx_trunk_end = idx_trunk_start + max(0, int(mc.N_trunk)) - 1
        if int(mc.N_trunk) > 0:
            idx_fork = idx_trunk_end
        elif int(mc.N_ais) > 0:
            idx_fork = idx_ais_end
        else:
            idx_fork = 0
        idx_b1_start = idx_fork + 1
        idx_b1_end = idx_b1_start + max(0, int(mc.N_b1)) - 1
        idx_b2_start = idx_b1_start + max(0, int(mc.N_b1))
        idx_b2_end = idx_b2_start + max(0, int(mc.N_b2)) - 1
        idx_terminal = max(0, n_comp_total - 1)

        soma_r = max(4.0, mc.d_soma * 1e4 * 0.4)

        def _scaled_font(px: int) -> int:
            return max(6, int(round(px * self._font_scale)))

        def _scaled_width(w: float) -> float:
            return max(0.8, float(w) * self._line_scale)

        def _map_color(col):
            q = QColor(col)
            if self._high_contrast:
                q = q.lighter(135)
                q.setAlpha(min(255, int(q.alpha() * 1.1)))
            return q

        def _txt(text, x, y, color="#CDD6F4", anchor=(0.5, 0.5), size=8):
            if (not self._show_labels) and (not str(text).startswith("idx ")):
                return
            if str(text).startswith("idx ") and not self._show_indices:
                return
            q_col = _map_color(color)
            t = pg.TextItem(str(text), color=q_col, anchor=anchor)
            t.setFont(pg.Qt.QtGui.QFont("Segoe UI", _scaled_font(size)))
            t.setPos(x, y)
            self._plot.addItem(t)
            self._draw_items.append(t)

        def _line(x0, y0, x1, y1, color, width=2, style=Qt.PenStyle.SolidLine):
            item = pg.PlotCurveItem(
                [x0, x1],
                [y0, y1],
                pen=pg.mkPen(_map_color(color), width=_scaled_width(width), style=style),
            )
            self._plot.addItem(item)
            self._draw_items.append(item)

        def _glow(x0, y0, x1, y1, color, gw=12, alpha=60):
            rgba = _map_color(color)
            rgba.setAlpha(alpha)
            item = pg.PlotCurveItem(
                [x0, x1],
                [y0, y1],
                pen=pg.mkPen(rgba, width=_scaled_width(gw)),
            )
            self._plot.addItem(item)
            self._draw_items.append(item)

        def _seg(x0, y0, x1, y1, color, comp_idx, width=2, style=Qt.PenStyle.SolidLine):
            """Draw a segment tagged with comp_idx for voltage heatmap scrubbing."""
            w = _scaled_width(width)
            item = pg.PlotCurveItem(
                [x0, x1], [y0, y1],
                pen=pg.mkPen(_map_color(color), width=w, style=style),
            )
            item.comp_idx = comp_idx
            item._heat_pen_width = w
            self._plot.addItem(item)
            self._draw_items.append(item)
            return item

        def _marker(
            x,
            y,
            color,
            label,
            label_dx=1.2,
            label_dy=1.2,
            symbol="t",
            size=11,
        ):
            item = pg.ScatterPlotItem(
                x=[x],
                y=[y],
                size=max(5.0, size * self._line_scale),
                brush=pg.mkBrush(_map_color(color)),
                pen=pg.mkPen("#11111B", width=_scaled_width(1.5)),
                symbol=symbol,
            )
            self._plot.addItem(item)
            self._draw_items.append(item)
            _txt(label, x + label_dx, y + label_dy, color=color, anchor=(0.0, 0.0))

        def _stim_location_coords(loc, stim_amp, bif_x_val):
            color = "#F9E2AF" if stim_amp >= 0 else "#F38BA8"
            if loc == "ais" and int(mc.N_ais) > 0:
                ais_len_local = max(10, int(mc.N_ais) * 3)
                return soma_r / 2.0 + ais_len_local * 0.75, 3.5, "@ AIS", color
            if loc == "dendritic_filtered":
                return bif_x_val + 8.0, 9.0, "@ Dendrite", color
            return 0.0, soma_r + 2.0, "@ Soma", color

        soma_item = pg.ScatterPlotItem(
            x=[0],
            y=[0],
            size=max(8.0, soma_r * 2 * self._line_scale),
            brush=pg.mkBrush(_map_color("#FA8C3C")),
            pen=pg.mkPen(_map_color("#DC5A10"), width=_scaled_width(2)),
            symbol="o",
            name=f"Soma {mc.d_soma * 1e4:.0f}um",
        )
        self._soma_item = soma_item  # keep direct reference for heatmap scrubber
        soma_item.comp_idx = 0      # voltage index: soma is always compartment 0
        self._plot.addItem(soma_item)
        self._draw_items.append(soma_item)
        _txt(
            f"Soma\nCm={ch.Cm}uF/cm2\ntm={tau_m:.1f}ms",
            0,
            -soma_r - 2,
            color="#FA8C3C",
            anchor=(0.5, 1.0),
        )
        _txt(
            f"idx {idx_soma}",
            0,
            soma_r + 2.0,
            color="#F9E2AF",
            anchor=(0.5, 0.0),
            size=7,
        )

        if mc.single_comp:
            self._draw_single_comp(
                config,
                dual_config,
                soma_r,
                lam_um,
                tau_m,
                ch,
                _txt,
                _line,
                _marker,
            )
            self._plot.autoRange()
            if result is not None:
                self._heatmap_data = result.v_all
                self._heatmap_time = result.t
                self._time_slider.setRange(0, len(result.t) - 1)
                self._time_slider.setEnabled(True)
                self._on_time_scrub(self._time_slider.value())
            else:
                self._heatmap_data = None
                self._time_slider.setEnabled(False)
            return

        ais_len = max(10, int(mc.N_ais) * 3) if int(mc.N_ais) > 0 else 0.0
        trunk_len = max(15, int(mc.N_trunk) * 1.8) if int(mc.N_trunk) > 0 else 0.0
        b1_len = max(10, int(mc.N_b1) * 2.8) if int(mc.N_b1) > 0 else 0.0
        b2_len = max(10, int(mc.N_b2) * 2.8) if int(mc.N_b2) > 0 else 0.0

        x_ais_start = soma_r / 2.0
        x = x_ais_start
        bif_x = x
        b1_x = b1_y = np.nan
        b2_x = b2_y = np.nan

        if int(mc.N_ais) > 0:
            _glow(x, 0, x + ais_len, 0, "#FF1414", gw=20, alpha=80)
            _glow(x, 0, x + ais_len, 0, "#FF1414", gw=12, alpha=50)
            _n_ais = int(mc.N_ais)
            _sw = ais_len / _n_ais
            for _i in range(_n_ais):
                _seg(x + _i * _sw, 0, x + (_i + 1) * _sw, 0,
                     "#FF3030", idx_ais_start + _i, width=7)
            _line(x + ais_len * 0.3, 0, x + ais_len * 0.6, 0, "#FFD060", width=3)
            _txt(
                "Axon",
                x + ais_len * 0.45,
                4.5,
                color="#FFD060",
                anchor=(0.5, 0.0),
                size=7,
            )
            _txt(
                f"AIS ({mc.N_ais}seg)\nNa x{mc.gNa_ais_mult:.0f}  K x{mc.gK_ais_mult:.0f}",
                x + ais_len / 2,
                -6,
                color="#FF6060",
                anchor=(0.5, 1.0),
                size=8,
            )
            _txt(
                f"idx {idx_ais_start}..{idx_ais_end}",
                x + ais_len / 2,
                2.6,
                color="#F9E2AF",
                anchor=(0.5, 0.0),
                size=7,
            )
            x += ais_len

        x_trunk_start = x
        if int(mc.N_trunk) > 0:
            w_px = max(3.0, mc.d_trunk / mc.d_soma * 5.0)
            _glow(x, 0, x + trunk_len, 0, "#4080DC", gw=8, alpha=30)
            _n_trunk = int(mc.N_trunk)
            _sw = trunk_len / _n_trunk
            for _i in range(_n_trunk):
                _seg(x + _i * _sw, 0, x + (_i + 1) * _sw, 0,
                     "#5090DC", idx_trunk_start + _i, width=w_px)
            _txt(
                f"Trunk ({mc.N_trunk}seg)\nd={mc.d_trunk * 1e4:.1f}um  Ra={mc.Ra:.0f} ohm*cm",
                x + trunk_len / 2,
                -5,
                color="#7AAAE0",
                anchor=(0.5, 1.0),
                size=8,
            )
            _txt(
                f"idx {idx_trunk_start}..{idx_trunk_end}",
                x + trunk_len / 2,
                2.2,
                color="#F9E2AF",
                anchor=(0.5, 0.0),
                size=7,
            )
            x += trunk_len

        bif_x = x
        fork_item = pg.ScatterPlotItem(
            x=[bif_x],
            y=[0],
            size=max(6.0, 8 * self._line_scale),
            brush=pg.mkBrush(_map_color("#FAF060")),
            pen=pg.mkPen(_map_color("#C0A020"), width=_scaled_width(1.5)),
            symbol="d",
            name="Fork",
        )
        fork_item.comp_idx = idx_fork
        self._plot.addItem(fork_item)
        self._draw_items.append(fork_item)
        _txt("Fork", bif_x, 2.5, color="#FAF060", anchor=(0.5, 0.0), size=7)
        _txt(
            f"idx {idx_fork}",
            bif_x,
            -2.8,
            color="#F9E2AF",
            anchor=(0.5, 1.0),
            size=7,
        )

        if int(mc.N_b1) > 0:
            b1_w = max(2.0, mc.d_b1 / mc.d_trunk * 3.5)
            b1_x, b1_y = bif_x + b1_len * 0.85, b1_len * 0.5
            _glow(bif_x, 0, b1_x, b1_y, "#40CC60", gw=6, alpha=40)
            _n_b1 = int(mc.N_b1)
            for _i in range(_n_b1):
                _f0, _f1 = _i / _n_b1, (_i + 1) / _n_b1
                _seg(bif_x + _f0 * (b1_x - bif_x), _f0 * b1_y,
                     bif_x + _f1 * (b1_x - bif_x), _f1 * b1_y,
                     "#40CC60", idx_b1_start + _i, width=b1_w)
            b1_item = pg.ScatterPlotItem(
                x=[b1_x],
                y=[b1_y],
                size=max(6.0, 7 * self._line_scale),
                brush=pg.mkBrush(_map_color("#40CC60")),
                pen=pg.mkPen(_map_color("#208040"), width=_scaled_width(2)),
                symbol="o",
            )
            b1_item.comp_idx = idx_b1_end
            self._plot.addItem(b1_item)
            self._draw_items.append(b1_item)
            _txt(
                f"B1 ({mc.N_b1})\nd={mc.d_b1 * 1e4:.1f}um",
                b1_x + 2,
                b1_y,
                color="#50DD70",
                anchor=(0.0, 0.5),
                size=8,
            )
            _txt(
                f"idx {idx_b1_start}..{idx_b1_end}",
                b1_x + 2,
                b1_y - 2.0,
                color="#F9E2AF",
                anchor=(0.0, 1.0),
                size=7,
            )

        if int(mc.N_b2) > 0:
            b2_w = max(2.0, mc.d_b2 / mc.d_trunk * 3.5)
            b2_x, b2_y = bif_x + b2_len * 0.85, -b2_len * 0.5
            _glow(bif_x, 0, b2_x, b2_y, "#B040DC", gw=6, alpha=40)
            _n_b2 = int(mc.N_b2)
            for _i in range(_n_b2):
                _f0, _f1 = _i / _n_b2, (_i + 1) / _n_b2
                _seg(bif_x + _f0 * (b2_x - bif_x), _f0 * b2_y,
                     bif_x + _f1 * (b2_x - bif_x), _f1 * b2_y,
                     "#B040DC", idx_b2_start + _i, width=b2_w)
            b2_item = pg.ScatterPlotItem(
                x=[b2_x],
                y=[b2_y],
                size=max(6.0, 7 * self._line_scale),
                brush=pg.mkBrush(_map_color("#B040DC")),
                pen=pg.mkPen(_map_color("#702090"), width=_scaled_width(2)),
                symbol="o",
            )
            b2_item.comp_idx = idx_b2_end
            self._plot.addItem(b2_item)
            self._draw_items.append(b2_item)
            _txt(
                f"B2 ({mc.N_b2})\nd={mc.d_b2 * 1e4:.1f}um",
                b2_x + 2,
                b2_y,
                color="#D060FF",
                anchor=(0.0, 0.5),
                size=8,
            )
            _txt(
                f"idx {idx_b2_start}..{idx_b2_end}",
                b2_x + 2,
                b2_y + 2.0,
                color="#F9E2AF",
                anchor=(0.0, 0.0),
                size=7,
            )

        lam_plot = lam_um / max(mc.dx * 1e4, 1e-12)
        y_lam = -soma_r - 6
        _line(0, y_lam, lam_plot, y_lam, "#89B4FA", width=1.5, style=Qt.PenStyle.DashLine)
        for xv in (0, lam_plot):
            _line(xv, y_lam - 1, xv, y_lam + 1, "#89B4FA", width=1.5)
        _txt(f"lambda = {lam_um:.0f} um", lam_plot / 2, y_lam - 2, color="#89B4FA", anchor=(0.5, 1.0))

        stim_loc = config.stim_location.location
        stim_amp = config.stim.Iext
        stim_type = config.stim.stim_type
        mx, my, mlabel, mcol = _stim_location_coords(stim_loc, stim_amp, bif_x)
        if stim_loc == "dendritic_filtered":
            df = config.dendritic_filter
            att = np.exp(-df.distance_um / max(df.space_constant_um, 1e-9))
            _line(mx, my, 0, 0, mcol, width=1.5, style=Qt.PenStyle.DashLine)
            _txt(
                f"atten={att:.3f}\nt={df.tau_dendritic_ms:.0f}ms",
                mx + 1,
                my + 1.5,
                color="#A6E3A1",
                anchor=(0.0, 0.0),
                size=7,
            )
        _marker(mx, my, mcol, f"PRI {mlabel}\n{stim_type}  I={stim_amp:.1f} uA/cm2", size=13)

        if dual_config is not None and dual_config.enabled:
            s_loc = dual_config.secondary_location
            s_amp = dual_config.secondary_Iext
            s_type = dual_config.secondary_stim_type
            s_col = "#F38BA8" if s_amp < 0 or s_type in ("GABAA", "GABAB") else "#CBA6F7"
            sx, sy, slabel, _ = _stim_location_coords(s_loc, s_amp, bif_x)
            sy -= 3.5

            if s_loc == "dendritic_filtered":
                s_att = np.exp(
                    -dual_config.secondary_distance_um
                    / max(dual_config.secondary_space_constant_um, 1e-9)
                )
                _line(sx, sy, 0, 0, s_col, width=1.5, style=Qt.PenStyle.DotLine)
                _txt(
                    f"atten={s_att:.3f}\nt={dual_config.secondary_tau_dendritic_ms:.0f}ms",
                    sx + 1,
                    sy - 2,
                    color=s_col,
                    anchor=(0.0, 1.0),
                    size=7,
                )

            _marker(
                sx,
                sy,
                s_col,
                f"SEC {slabel}\n{s_type}  I={s_amp:.1f} uA/cm2",
                label_dy=-3.5,
                size=11,
                symbol="s",
            )

            e_curr = abs(dual_config.primary_Iext)
            i_curr = abs(dual_config.secondary_Iext) if s_type in ("GABAA", "GABAB") else 0
            if i_curr > 0:
                ei = e_curr / i_curr
                ei_col = "#A6E3A1" if 0.5 <= ei <= 3.0 else "#F38BA8"
                _txt(f"E/I = {ei:.2f}", -soma_r - 2, soma_r + 3, color=ei_col, anchor=(1.0, 0.0), size=8)

        def _coord_for_index(idx: int):
            idx = int(max(0, min(idx_terminal, idx)))
            if idx == 0:
                return 0.0, 0.0
            if int(mc.N_ais) > 0 and idx_ais_start <= idx <= idx_ais_end:
                frac = (idx - idx_ais_start + 0.5) / max(1, int(mc.N_ais))
                return x_ais_start + frac * ais_len, 0.0
            if int(mc.N_trunk) > 0 and idx_trunk_start <= idx <= idx_trunk_end:
                frac = (idx - idx_trunk_start + 0.5) / max(1, int(mc.N_trunk))
                return x_trunk_start + frac * trunk_len, 0.0
            if idx == idx_fork:
                return bif_x, 0.0
            if int(mc.N_b1) > 0 and idx_b1_start <= idx <= idx_b1_end:
                frac = (idx - idx_b1_start + 0.5) / max(1, int(mc.N_b1))
                return bif_x + frac * (b1_x - bif_x), frac * b1_y
            if int(mc.N_b2) > 0 and idx_b2_start <= idx <= idx_b2_end:
                frac = (idx - idx_b2_start + 0.5) / max(1, int(mc.N_b2))
                return bif_x + frac * (b2_x - bif_x), frac * b2_y
            return bif_x, 0.0

        delay_idx, delay_label = self._resolve_delay_target(
            n_comp_total=n_comp_total,
            mc=mc,
            idx_fork=idx_fork,
            idx_terminal=idx_terminal,
        )
        if delay_idx is not None:
            tx, ty = _coord_for_index(delay_idx)
            delay_item = pg.ScatterPlotItem(
                x=[tx],
                y=[ty],
                size=max(10.0, 18 * self._line_scale),
                brush=pg.mkBrush(0, 0, 0, 0),
                pen=pg.mkPen(_map_color("#F9E2AF"), width=_scaled_width(2.2)),
                symbol="o",
            )
            self._plot.addItem(delay_item)
            self._draw_items.append(delay_item)
            _txt(
                f"Delay target: {delay_label}\nidx {delay_idx}",
                tx + 1.2,
                ty + 1.2,
                color="#F9E2AF",
                anchor=(0.0, 0.0),
                size=7,
            )

        active = [f"Na{ch.gNa_max:.0f}", f"K{ch.gK_max:.0f}", f"L{ch.gL}"]
        if ch.enable_Ih:
            active.append(f"Ih{ch.gIh_max}")
        if ch.enable_ICa:
            active.append(f"Ca{ch.gCa_max}")
        if ch.enable_IA:
            active.append(f"IA{ch.gA_max}")
        if ch.enable_SK:
            active.append(f"SK{ch.gSK_max}")

        info_parts = [
            f"{n_comp_total} comp",
            f"tm={tau_m:.2f}ms",
            f"lam={lam_um:.0f}um",
            f"Rin={1.0 / ch.gL:.1f}kOhm*cm2",
            f"gL={ch.gL}  gNa={ch.gNa_max}  gK={ch.gK_max}",
            f"Stim: {stim_loc}/{stim_type}  I={stim_amp:.1f}",
            f"idx: soma={idx_soma}, fork={idx_fork}, term={idx_terminal}",
        ]
        if delay_idx is not None:
            info_parts.append(f"Delay target: {delay_label} (idx {delay_idx})")

        df = config.dendritic_filter
        if stim_loc == "dendritic_filtered" and df.enabled:
            att = np.exp(-df.distance_um / max(df.space_constant_um, 1e-9))
            info_parts.append(
                f"Dend d={df.distance_um:.0f}um  lam={df.space_constant_um:.0f}um  "
                f"tau={df.tau_dendritic_ms:.0f}ms  atten={att:.3f}"
            )
        if dual_config is not None and dual_config.enabled:
            info_parts.append(f"DUAL: {config.stim_location.location}+{dual_config.secondary_location}")

        self._info.setText("  |  ".join(info_parts))
        self._plot.autoRange()

        if result is not None:
            self._heatmap_data = result.v_all
            self._heatmap_time = result.t
            self._time_slider.setRange(0, len(result.t) - 1)
            self._time_slider.setEnabled(True)
            self._on_time_scrub(self._time_slider.value())
        else:
            self._heatmap_data = None
            self._time_slider.setEnabled(False)

    @staticmethod
    def _get_heat_color(v: float) -> QColor:
        """Map membrane voltage (mV) to a blue→yellow→red heatmap color."""
        v_norm = max(0.0, min(1.0, (v + 80.0) / 120.0))
        if v_norm < 0.5:
            r = int(2 * v_norm * 250)
            g = int(2 * v_norm * 200)
            b = 250
        else:
            r = 250
            g = int(2 * (1.0 - v_norm) * 200)
            b = int(2 * (1.0 - v_norm) * 250)
        return QColor(r, g, b)

    def _on_time_scrub(self, idx: int):
        if self._heatmap_data is None or self._heatmap_time is None:
            return
        idx = max(0, min(idx, self._heatmap_data.shape[1] - 1))
        t_val = self._heatmap_time[idx]
        self._lbl_time.setText(f"t = {t_val:.1f} ms")

        v_data = self._heatmap_data[:, idx]
        n_comp = v_data.shape[0]

        for item in self._draw_items:
            if not hasattr(item, 'comp_idx'):
                continue
            ci = int(item.comp_idx)
            if ci < 0 or ci >= n_comp:
                continue
            color = TopologyWidget._get_heat_color(float(v_data[ci]))
            if isinstance(item, pg.ScatterPlotItem):
                item.setBrush(pg.mkBrush(color))
                item.update()
            elif isinstance(item, pg.PlotCurveItem):
                w = getattr(item, '_heat_pen_width', 2.0)
                item.setPen(pg.mkPen(color, width=w))
    
    def _on_plot_clicked(self, event):
        """Handle mouse clicks on topology plot to select compartment."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Get click position in plot coordinates
            pos = event.scenePos()
            mouse_point = self._plot.vb.mapSceneToView(pos)
            mx, my = mouse_point.x(), mouse_point.y()
            
            # Find nearest compartment with comp_idx attribute
            nearest_idx = None
            min_dist = float('inf')
            
            for item in self._draw_items:
                if not hasattr(item, 'comp_idx'):
                    continue
                # For segments, check midpoint
                if isinstance(item, pg.PlotCurveItem):
                    x_data = item.getData()[0]
                    y_data = item.getData()[1]
                    if len(x_data) >= 2:
                        mid_x = (x_data[0] + x_data[1]) / 2
                        mid_y = (y_data[0] + y_data[1]) / 2
                        dist = ((mx - mid_x)**2 + (my - mid_y)**2)**0.5
                        if dist < min_dist:
                            min_dist = dist
                            nearest_idx = item.comp_idx
                # For scatter points (soma, fork, branch ends)
                elif isinstance(item, pg.ScatterPlotItem):
                    x_data = item.getData()[0]
                    y_data = item.getData()[1]
                    if len(x_data) > 0:
                        dist = ((mx - x_data[0])**2 + (my - y_data[0])**2)**0.5
                        if dist < min_dist:
                            min_dist = dist
                            nearest_idx = item.comp_idx
            
            # Emit signal if a compartment was found within reasonable distance
            if nearest_idx is not None and min_dist < 5.0:  # 5 unit tolerance
                self.compartment_selected.emit(int(nearest_idx))

    def _draw_single_comp(
        self,
        config,
        dual_config,
        soma_r,
        lam_um,
        tau_m,
        ch,
        _txt,
        _line,
        _marker,
    ):
        """Single-compartment mode: soma + optional dendritic filter stub."""
        stim_loc = config.stim_location.location
        stim_amp = config.stim.Iext
        stim_type = config.stim.stim_type
        df = config.dendritic_filter

        if stim_loc == "dendritic_filtered" and df.enabled:
            att = np.exp(-df.distance_um / max(df.space_constant_um, 1e-9))
            dend_x, dend_y = -soma_r - 12, soma_r + 3
            _line(dend_x, dend_y, 0, 0, "#A6E3A1", width=2, style=Qt.PenStyle.DashLine)
            _txt(
                f"Dendrite\nd={df.distance_um:.0f}um\n"
                f"lam={df.space_constant_um:.0f}um\n"
                f"tau={df.tau_dendritic_ms:.1f}ms\natten={att:.3f}",
                dend_x,
                dend_y + 1.5,
                color="#A6E3A1",
                anchor=(0.5, 0.0),
                size=8,
            )
            _marker(
                dend_x,
                dend_y,
                "#F9E2AF",
                f"PRI @ Dendrite\n{stim_type}  I={stim_amp:.1f} uA/cm2",
                label_dx=-1,
                label_dy=-3.5,
                size=13,
            )
        else:
            col = "#F9E2AF" if stim_amp >= 0 else "#F38BA8"
            _marker(
                0,
                soma_r + 2.2,
                col,
                f"PRI @ {stim_loc}\n{stim_type}  I={stim_amp:.1f} uA/cm2",
                label_dx=0.5,
                size=13,
            )

        if dual_config is not None and dual_config.enabled:
            s_loc = dual_config.secondary_location
            s_amp = dual_config.secondary_Iext
            s_type = dual_config.secondary_stim_type
            s_col = "#F38BA8" if s_amp < 0 or s_type in ("GABAA", "GABAB") else "#CBA6F7"

            if s_loc == "dendritic_filtered" and df.enabled:
                s_att = np.exp(
                    -dual_config.secondary_distance_um
                    / max(dual_config.secondary_space_constant_um, 1e-9)
                )
                sx, sy = -soma_r - 12, -soma_r - 3
                _line(sx, sy, 0, 0, s_col, width=1.5, style=Qt.PenStyle.DotLine)
                _txt(
                    f"Dendrite2\natten={s_att:.3f}",
                    sx,
                    sy - 1.5,
                    color=s_col,
                    anchor=(0.5, 1.0),
                    size=7,
                )
                _marker(
                    sx,
                    sy,
                    s_col,
                    f"SEC @ Dendrite\n{s_type}  I={s_amp:.1f} uA/cm2",
                    label_dx=1,
                    label_dy=-3,
                    size=11,
                    symbol="s",
                )
            else:
                _marker(
                    soma_r * 0.6,
                    soma_r + 2.2,
                    s_col,
                    f"SEC @ {s_loc}\n{s_type}  I={s_amp:.1f} uA/cm2",
                    label_dx=0.5,
                    size=11,
                    symbol="s",
                )

        _txt(f"Single-comp  lam={lam_um:.0f}um", 0, soma_r + 6, color="#89B4FA", anchor=(0.5, 0.0), size=7)

        info_parts = [
            "Single compartment",
            f"tm={tau_m:.2f}ms",
            f"lam={lam_um:.0f}um",
            f"Rin={1.0 / ch.gL:.1f}kOhm*cm2",
            "idx: soma=0, term=0",
            "Delay target: n/a (single-comp)",
        ]
        if stim_loc == "dendritic_filtered" and df.enabled:
            att = np.exp(-df.distance_um / max(df.space_constant_um, 1e-9))
            info_parts.append(f"Dend d={df.distance_um:.0f}um atten={att:.3f}")
        if dual_config is not None and dual_config.enabled:
            info_parts.append(f"DUAL: {config.stim_location.location}+{dual_config.secondary_location}")
        self._info.setText("  |  ".join(info_parts))
