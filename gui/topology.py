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
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class TopologyWidget(QWidget):
    """Visual morphology schematic, updated after each run."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._win = pg.GraphicsLayoutWidget()
        self._win.setBackground("#0D1117")
        layout.addWidget(self._win)

        self._plot = self._win.addPlot()
        self._plot.setAspectLocked(False)
        self._plot.hideAxis("bottom")
        self._plot.hideAxis("left")
        self._plot.getViewBox().setMouseEnabled(x=True, y=True)
        self._plot.addLegend(offset=(10, 10), labelTextColor="#CDD6F4")

        self._info = QLabel("")
        self._info.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._info.setStyleSheet("color:#A6E3A1; font-size:11px; padding:4px;")
        layout.addWidget(self._info)

        self._delay_target_name = "Terminal"
        self._delay_custom_index = 1
        self._last_config = None
        self._last_dual_config = None

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
        if n_comp_total <= 1:
            return None, "n/a"

        target = self._delay_target_name
        if target == "AIS":
            if int(mc.N_ais) > 0:
                return 1, "AIS"
            return idx_terminal, "terminal"
        if target == "Trunk Junction":
            return max(0, min(idx_fork, idx_terminal)), "fork"
        if target == "Custom Compartment":
            idx = max(1, min(idx_terminal, int(self._delay_custom_index)))
            return idx, f"comp[{idx}]"
        return idx_terminal, "terminal"

    def draw_neuron(
        self,
        config,
        dual_config=None,
        delay_target_name: str | None = None,
        delay_custom_index: int | None = None,
    ):
        """Draw neuron morphology with optional dual-stim overlay and delay focus."""
        if delay_target_name is not None:
            self._delay_target_name = str(delay_target_name)
        if delay_custom_index is not None:
            self._delay_custom_index = max(1, int(delay_custom_index))
        self._last_config = config
        self._last_dual_config = dual_config

        self._plot.clear()
        self._plot.addLegend(offset=(10, 10), labelTextColor="#CDD6F4")
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

        def _txt(text, x, y, color="#CDD6F4", anchor=(0.5, 0.5), size=8):
            t = pg.TextItem(text, color=color, anchor=anchor)
            t.setFont(pg.Qt.QtGui.QFont("Segoe UI", size))
            t.setPos(x, y)
            self._plot.addItem(t)

        def _line(x0, y0, x1, y1, color, width=2, style=Qt.PenStyle.SolidLine):
            self._plot.addItem(
                pg.PlotCurveItem(
                    [x0, x1],
                    [y0, y1],
                    pen=pg.mkPen(QColor(color), width=width, style=style),
                )
            )

        def _glow(x0, y0, x1, y1, color, gw=12, alpha=60):
            rgba = QColor(color)
            rgba.setAlpha(alpha)
            self._plot.addItem(
                pg.PlotCurveItem(
                    [x0, x1],
                    [y0, y1],
                    pen=pg.mkPen(rgba, width=gw),
                )
            )

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
            self._plot.addItem(
                pg.ScatterPlotItem(
                    x=[x],
                    y=[y],
                    size=size,
                    brush=pg.mkBrush(color),
                    pen=pg.mkPen("#11111B", width=1.5),
                    symbol=symbol,
                )
            )
            _txt(label, x + label_dx, y + label_dy, color=color, anchor=(0.0, 0.0))

        def _stim_location_coords(loc, stim_amp, bif_x_val):
            color = "#F9E2AF" if stim_amp >= 0 else "#F38BA8"
            if loc == "ais" and int(mc.N_ais) > 0:
                ais_len_local = max(10, int(mc.N_ais) * 3)
                return soma_r / 2.0 + ais_len_local * 0.75, 3.5, "@ AIS", color
            if loc == "dendritic_filtered":
                return bif_x_val + 8.0, 9.0, "@ Dendrite", color
            return 0.0, soma_r + 2.0, "@ Soma", color

        self._plot.addItem(
            pg.ScatterPlotItem(
                x=[0],
                y=[0],
                size=soma_r * 2,
                brush=pg.mkBrush(QColor("#FA8C3C")),
                pen=pg.mkPen(QColor("#DC5A10"), width=2),
                symbol="o",
                name=f"Soma {mc.d_soma * 1e4:.0f}um",
            )
        )
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
            _line(x, 0, x + ais_len, 0, "#FF3030", width=7)
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
            _line(x, 0, x + trunk_len, 0, "#5090DC", width=w_px)
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
        self._plot.addItem(
            pg.ScatterPlotItem(
                x=[bif_x],
                y=[0],
                size=8,
                brush=pg.mkBrush("#FAF060"),
                pen=pg.mkPen("#C0A020", width=1.5),
                symbol="d",
                name="Fork",
            )
        )
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
            _line(bif_x, 0, b1_x, b1_y, "#40CC60", width=b1_w)
            self._plot.addItem(
                pg.ScatterPlotItem(
                    x=[b1_x],
                    y=[b1_y],
                    size=7,
                    brush=pg.mkBrush("#40CC60"),
                    pen=pg.mkPen("#208040", width=2),
                    symbol="o",
                )
            )
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
            _line(bif_x, 0, b2_x, b2_y, "#B040DC", width=b2_w)
            self._plot.addItem(
                pg.ScatterPlotItem(
                    x=[b2_x],
                    y=[b2_y],
                    size=7,
                    brush=pg.mkBrush("#B040DC"),
                    pen=pg.mkPen("#702090", width=2),
                    symbol="o",
                )
            )
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
            self._plot.addItem(
                pg.ScatterPlotItem(
                    x=[tx],
                    y=[ty],
                    size=18,
                    brush=pg.mkBrush(0, 0, 0, 0),
                    pen=pg.mkPen("#F9E2AF", width=2.2),
                    symbol="o",
                )
            )
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
            info_parts.append(f"DUAL: {dual_config.primary_location}+{dual_config.secondary_location}")

        self._info.setText("  |  ".join(info_parts))
        self._plot.autoRange()

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
            info_parts.append(f"DUAL: {dual_config.primary_location}+{dual_config.secondary_location}")
        self._info.setText("  |  ".join(info_parts))

