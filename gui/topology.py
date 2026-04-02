"""
gui/topology.py — Neuron Morphology Viewer v10.2

Renders a schematic of the multi-compartment neuron model:
  Soma (circle) → AIS (glowing red) → Trunk → Bifurcation → Branch 1/2

v10.2 changes:
- Full dual stimulation visualization (both primary + secondary sources)
- Dendritic filter cable shown as annotated stub (single-comp & multi-comp)
- Channel badges on AIS segment
- Cleaner info bar with all passive properties
"""
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor


class TopologyWidget(QWidget):
    """Visual morphology schematic, updated after each run."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._win = pg.GraphicsLayoutWidget()
        self._win.setBackground('#0D1117')
        layout.addWidget(self._win)

        self._plot = self._win.addPlot()
        self._plot.setAspectLocked(False)
        self._plot.hideAxis('bottom')
        self._plot.hideAxis('left')
        self._plot.getViewBox().setMouseEnabled(x=True, y=True)
        self._plot.addLegend(offset=(10, 10), labelTextColor='#CDD6F4')

        self._info = QLabel("")
        self._info.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._info.setStyleSheet("color:#A6E3A1; font-size:11px; padding:4px;")
        layout.addWidget(self._info)

    # ─────────────────────────────────────────────────────────────────
    def draw_neuron(self, config, dual_config=None):
        """Draw neuron morphology with optional dual stimulation overlay."""

        self._plot.clear()
        self._plot.addLegend(offset=(10, 10), labelTextColor='#CDD6F4')
        mc = config.morphology
        ch = config.channels

        from core.analysis import space_constant, membrane_time_constant
        lam_um = space_constant(mc.d_soma, mc.Ra, ch.gL) * 1e4
        tau_m  = membrane_time_constant(ch.Cm, ch.gL)

        soma_r = max(4.0, mc.d_soma * 1e4 * 0.4)

        # ── helpers ──────────────────────────────────────────────────
        def _txt(text, x, y, color='#CDD6F4', anchor=(0.5, 0.5), size=8):
            t = pg.TextItem(text, color=color, anchor=anchor)
            t.setFont(pg.Qt.QtGui.QFont("Segoe UI", size))
            t.setPos(x, y)
            self._plot.addItem(t)

        def _line(x0, y0, x1, y1, color, width=2, style=Qt.PenStyle.SolidLine):
            self._plot.addItem(pg.PlotCurveItem(
                [x0, x1], [y0, y1],
                pen=pg.mkPen(QColor(color), width=width, style=style)
            ))

        def _glow(x0, y0, x1, y1, color, gw=12, alpha=60):
            rgba = QColor(color)
            rgba.setAlpha(alpha)
            self._plot.addItem(pg.PlotCurveItem(
                [x0, x1], [y0, y1],
                pen=pg.mkPen(rgba, width=gw)
            ))

        def _marker(x, y, color, label, label_dx=1.2, label_dy=1.2,
                    symbol='t', size=11):
            self._plot.addItem(pg.ScatterPlotItem(
                x=[x], y=[y], size=size,
                brush=pg.mkBrush(color),
                pen=pg.mkPen('#11111B', width=1.5),
                symbol=symbol
            ))
            _txt(label, x + label_dx, y + label_dy,
                 color=color, anchor=(0.0, 0.0))

        def _stim_location_coords(loc, stim_amp, bif_x_val):
            """Return (x, y, label) for a stimulus location string."""
            color = '#F9E2AF' if stim_amp >= 0 else '#F38BA8'
            if loc == 'ais' and mc.N_ais > 0:
                ais_len = max(10, mc.N_ais * 3)
                return soma_r / 2.0 + ais_len * 0.75, 3.5, "@ AIS", color
            elif loc == 'dendritic_filtered':
                return bif_x_val + 8.0, 9.0, "@ Dendrite", color
            else:  # soma
                return 0.0, soma_r + 2.0, "@ Soma", color

        # ── SOMA ────────────────────────────────────────────────────
        self._plot.addItem(pg.ScatterPlotItem(
            x=[0], y=[0], size=soma_r * 2,
            brush=pg.mkBrush(QColor('#FA8C3C')),
            pen=pg.mkPen(QColor('#DC5A10'), width=2),
            symbol='o', name=f"Soma {mc.d_soma*1e4:.0f}µm"
        ))
        _txt(f"Soma\nCm={ch.Cm}µF/cm²\ntm={tau_m:.1f}ms",
             0, -soma_r - 2, color='#FA8C3C', anchor=(0.5, 1.0))

        # ── SINGLE-COMPARTMENT ──────────────────────────────────────
        if mc.single_comp:
            self._draw_single_comp(
                config, dual_config, soma_r, lam_um, tau_m, ch,
                _txt, _line, _glow, _marker
            )
            self._plot.autoRange()
            return

        # ── MULTI-COMPARTMENT GEOMETRY ──────────────────────────────
        x = soma_r / 2.0
        bif_x = x  # will be updated

        # AIS
        if mc.N_ais > 0:
            ais_len = max(10, mc.N_ais * 3)
            _glow(x, 0, x + ais_len, 0, '#FF1414', gw=20, alpha=80)
            _glow(x, 0, x + ais_len, 0, '#FF1414', gw=12, alpha=50)
            _line(x, 0, x + ais_len, 0, '#FF3030', width=7)
            # Direction arrow
            ax1, ax2 = x + ais_len * 0.3, x + ais_len * 0.6
            _line(ax1, 0, ax2, 0, '#FFD060', width=3)
            _txt("Axon", ax1 + (ax2 - ax1) / 2, 4.5, color='#FFD060',
                 anchor=(0.5, 0.0), size=7)
            _txt(
                f"AIS ({mc.N_ais}seg)\nNa×{mc.gNa_ais_mult:.0f}  K×{mc.gK_ais_mult:.0f}",
                x + ais_len / 2, -6, color='#FF6060', anchor=(0.5, 1.0), size=8
            )
            x += ais_len

        # Trunk
        if mc.N_trunk > 0:
            trunk_len = max(15, mc.N_trunk * 1.8)
            w_px = max(3.0, mc.d_trunk / mc.d_soma * 5.0)
            _glow(x, 0, x + trunk_len, 0, '#4080DC', gw=8, alpha=30)
            _line(x, 0, x + trunk_len, 0, '#5090DC', width=w_px)
            _txt(
                f"Trunk ({mc.N_trunk}seg)\nd={mc.d_trunk*1e4:.1f}µm  Ra={mc.Ra:.0f}Ω·cm",
                x + trunk_len / 2, -5, color='#7AAAE0', anchor=(0.5, 1.0), size=8
            )
            x += trunk_len

        bif_x = x
        self._plot.addItem(pg.ScatterPlotItem(
            x=[bif_x], y=[0], size=8,
            brush=pg.mkBrush('#FAF060'),
            pen=pg.mkPen('#C0A020', width=1.5),
            symbol='d', name="Fork"
        ))
        _txt("Fork", bif_x, 2.5, color='#FAF060', anchor=(0.5, 0.0), size=7)

        # Branch 1 (up)
        if mc.N_b1 > 0:
            b1_len = max(10, mc.N_b1 * 2.8)
            b1_w   = max(2.0, mc.d_b1 / mc.d_trunk * 3.5)
            b1_x, b1_y = bif_x + b1_len * 0.85, b1_len * 0.5
            _glow(bif_x, 0, b1_x, b1_y, '#40CC60', gw=6, alpha=40)
            _line(bif_x, 0, b1_x, b1_y, '#40CC60', width=b1_w)
            self._plot.addItem(pg.ScatterPlotItem(
                x=[b1_x], y=[b1_y], size=7,
                brush=pg.mkBrush('#40CC60'),
                pen=pg.mkPen('#208040', width=2), symbol='o'
            ))
            _txt(f"B1 ({mc.N_b1})\nd={mc.d_b1*1e4:.1f}µm",
                 b1_x + 2, b1_y, color='#50DD70', anchor=(0.0, 0.5), size=8)

        # Branch 2 (down)
        if mc.N_b2 > 0:
            b2_len = max(10, mc.N_b2 * 2.8)
            b2_w   = max(2.0, mc.d_b2 / mc.d_trunk * 3.5)
            b2_x, b2_y = bif_x + b2_len * 0.85, -b2_len * 0.5
            _glow(bif_x, 0, b2_x, b2_y, '#B040DC', gw=6, alpha=40)
            _line(bif_x, 0, b2_x, b2_y, '#B040DC', width=b2_w)
            self._plot.addItem(pg.ScatterPlotItem(
                x=[b2_x], y=[b2_y], size=7,
                brush=pg.mkBrush('#B040DC'),
                pen=pg.mkPen('#702090', width=2), symbol='o'
            ))
            _txt(f"B2 ({mc.N_b2})\nd={mc.d_b2*1e4:.1f}µm",
                 b2_x + 2, b2_y, color='#D060FF', anchor=(0.0, 0.5), size=8)

        # ── λ bar ────────────────────────────────────────────────────
        lam_plot = lam_um / (mc.dx * 1e4)
        y_lam = -soma_r - 6
        _line(0, y_lam, lam_plot, y_lam, '#89B4FA', width=1.5,
              style=Qt.PenStyle.DashLine)
        for xv in (0, lam_plot):
            _line(xv, y_lam - 1, xv, y_lam + 1, '#89B4FA', width=1.5)
        _txt(f"λ = {lam_um:.0f} µm", lam_plot / 2, y_lam - 2,
             color='#89B4FA', anchor=(0.5, 1.0))

        # ── PRIMARY STIMULUS ──────────────────────────────────────────
        stim_loc = config.stim_location.location
        stim_amp = config.stim.Iext
        stim_type = config.stim.stim_type
        mx, my, mlabel, mcol = _stim_location_coords(stim_loc, stim_amp, bif_x)

        if stim_loc == 'dendritic_filtered':
            df = config.dendritic_filter
            att = np.exp(-df.distance_um / max(df.space_constant_um, 1e-9))
            _line(mx, my, 0, 0, mcol, width=1.5, style=Qt.PenStyle.DashLine)
            _txt(
                f"atten={att:.3f}\nt={df.tau_dendritic_ms:.0f}ms",
                mx + 1, my + 1.5, color='#A6E3A1', anchor=(0.0, 0.0), size=7
            )

        _marker(mx, my, mcol,
                f"PRI {mlabel}\n{stim_type}  I={stim_amp:.1f} µA/cm²",
                size=13)

        # ── SECONDARY STIMULUS (dual stim) ────────────────────────────
        if dual_config is not None and dual_config.enabled:
            s_loc  = dual_config.secondary_location
            s_amp  = dual_config.secondary_Iext
            s_type = dual_config.secondary_stim_type
            s_col  = '#F38BA8' if s_amp < 0 or s_type in ('GABAA', 'GABAB') else '#CBA6F7'

            sx, sy, slabel, _ = _stim_location_coords(s_loc, s_amp, bif_x)
            # Offset secondary marker slightly so it doesn't overlap primary
            sy -= 3.5

            if s_loc == 'dendritic_filtered':
                s_att = np.exp(
                    -dual_config.secondary_distance_um /
                    max(dual_config.secondary_space_constant_um, 1e-9)
                )
                _line(sx, sy, 0, 0, s_col, width=1.5,
                      style=Qt.PenStyle.DotLine)
                _txt(
                    f"atten={s_att:.3f}\nt={dual_config.secondary_tau_dendritic_ms:.0f}ms",
                    sx + 1, sy - 2, color=s_col, anchor=(0.0, 1.0), size=7
                )

            _marker(sx, sy, s_col,
                    f"SEC {slabel}\n{s_type}  I={s_amp:.1f} µA/cm²",
                    label_dy=-3.5, size=11, symbol='s')

            # E/I balance annotation
            e_curr = abs(dual_config.primary_Iext)
            i_curr = abs(dual_config.secondary_Iext) if s_type in ('GABAA', 'GABAB') else 0
            if i_curr > 0:
                ei = e_curr / i_curr
                ei_col = '#A6E3A1' if 0.5 <= ei <= 3.0 else '#F38BA8'
                _txt(f"E/I = {ei:.2f}", -soma_r - 2, soma_r + 3,
                     color=ei_col, anchor=(1.0, 0.0), size=8)

        # ── INFO BAR ─────────────────────────────────────────────────
        active = [f"Na{ch.gNa_max:.0f}", f"K{ch.gK_max:.0f}", f"L{ch.gL}"]
        if ch.enable_Ih:  active.append(f"Ih{ch.gIh_max}")
        if ch.enable_ICa: active.append(f"Ca{ch.gCa_max}")
        if ch.enable_IA:  active.append(f"IA{ch.gA_max}")
        if ch.enable_SK:  active.append(f"SK{ch.gSK_max}")

        n_comp_total = 1 + mc.N_ais + mc.N_trunk + mc.N_b1 + mc.N_b2
        info_parts = [
            f"{n_comp_total} comp",
            f"tm={tau_m:.2f}ms",
            f"lam={lam_um:.0f}µm",
            f"Rin={1.0/ch.gL:.1f}kOhm·cm²",
            f"gL={ch.gL}  gNa={ch.gNa_max}  gK={ch.gK_max}",
            f"Stim: {stim_loc}/{stim_type}  I={stim_amp:.1f}",
        ]
        df = config.dendritic_filter
        if stim_loc == 'dendritic_filtered' and df.enabled:
            att = np.exp(-df.distance_um / max(df.space_constant_um, 1e-9))
            info_parts.append(
                f"Dend d={df.distance_um:.0f}µm  "
                f"lam={df.space_constant_um:.0f}µm  "
                f"tau={df.tau_dendritic_ms:.0f}ms  atten={att:.3f}"
            )
        if dual_config is not None and dual_config.enabled:
            info_parts.append(
                f"DUAL: {dual_config.primary_location}+{dual_config.secondary_location}"
            )
        self._info.setText("  |  ".join(info_parts))
        self._plot.autoRange()

    # ─────────────────────────────────────────────────────────────────
    def _draw_single_comp(self, config, dual_config, soma_r, lam_um, tau_m, ch,
                          _txt, _line, _glow, _marker):
        """Single-compartment mode: soma + optional dendritic filter stub."""
        mc = config.morphology
        stim_loc = config.stim_location.location
        stim_amp = config.stim.Iext
        stim_type = config.stim.stim_type
        df = config.dendritic_filter

        if stim_loc == 'dendritic_filtered' and df.enabled:
            # Draw a schematic dendrite stub to the left of soma
            att = np.exp(-df.distance_um / max(df.space_constant_um, 1e-9))
            dend_x, dend_y = -soma_r - 12, soma_r + 3
            _line(dend_x, dend_y, 0, 0, '#A6E3A1', width=2,
                  style=Qt.PenStyle.DashLine)
            _txt(
                f"Dendrite\nd={df.distance_um:.0f}µm\n"
                f"lam={df.space_constant_um:.0f}µm\n"
                f"tau={df.tau_dendritic_ms:.1f}ms\natten={att:.3f}",
                dend_x, dend_y + 1.5, color='#A6E3A1', anchor=(0.5, 0.0), size=8
            )
            _marker(dend_x, dend_y,
                    '#F9E2AF',
                    f"PRI @ Dendrite\n{stim_type}  I={stim_amp:.1f} µA/cm²",
                    label_dx=-1, label_dy=-3.5, size=13)
        else:
            marker_y = soma_r + 2.2
            col = '#F9E2AF' if stim_amp >= 0 else '#F38BA8'
            _marker(0, marker_y, col,
                    f"PRI @ {stim_loc}\n{stim_type}  I={stim_amp:.1f} µA/cm²",
                    label_dx=0.5, size=13)

        # Secondary (dual) stim on single-comp
        if dual_config is not None and dual_config.enabled:
            s_loc  = dual_config.secondary_location
            s_amp  = dual_config.secondary_Iext
            s_type = dual_config.secondary_stim_type
            s_col  = '#F38BA8' if s_amp < 0 or s_type in ('GABAA', 'GABAB') else '#CBA6F7'

            if s_loc == 'dendritic_filtered' and df.enabled:
                s_att = np.exp(
                    -dual_config.secondary_distance_um /
                    max(dual_config.secondary_space_constant_um, 1e-9)
                )
                sx, sy = -soma_r - 12, -soma_r - 3
                _line(sx, sy, 0, 0, s_col, width=1.5,
                      style=Qt.PenStyle.DotLine)
                _txt(
                    f"Dendrite2\natten={s_att:.3f}",
                    sx, sy - 1.5, color=s_col, anchor=(0.5, 1.0), size=7
                )
                _marker(sx, sy, s_col,
                        f"SEC @ Dendrite\n{s_type}  I={s_amp:.1f} µA/cm²",
                        label_dx=1, label_dy=-3, size=11, symbol='s')
            else:
                _marker(soma_r * 0.6, soma_r + 2.2, s_col,
                        f"SEC @ {s_loc}\n{s_type}  I={s_amp:.1f} µA/cm²",
                        label_dx=0.5, size=11, symbol='s')

        _txt(f"Single-comp  lam={lam_um:.0f}µm", 0, soma_r + 6,
             color='#89B4FA', anchor=(0.5, 0.0), size=7)

        # Info bar
        info_parts = [
            "Single compartment",
            f"tm={tau_m:.2f}ms",
            f"lam={lam_um:.0f}µm",
            f"Rin={1.0/ch.gL:.1f}kOhm·cm²",
        ]
        if stim_loc == 'dendritic_filtered' and df.enabled:
            att = np.exp(-df.distance_um / max(df.space_constant_um, 1e-9))
            info_parts.append(f"Dend d={df.distance_um:.0f}µm atten={att:.3f}")
        if dual_config is not None and dual_config.enabled:
            info_parts.append(
                f"DUAL: {dual_config.primary_location}+{dual_config.secondary_location}"
            )
        self._info.setText("  |  ".join(info_parts))
