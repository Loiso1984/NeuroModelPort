"""
Modern axon biophysics diagnostics.

This widget focuses on what matters in the current codebase:
- propagation from soma through AIS/trunk/distal tips,
- spatiotemporal heatmaps for voltage or a selected current,
- conductance-density profiles across compartments,
- a focused trace for one compartment of interest.
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QSpinBox,
)


_SIGNAL_OPTIONS = [
    ("Voltage propagation", "voltage"),
    ("Sodium current (Na)", "Na"),
    ("Potassium current (K)", "K"),
    ("Leak current", "Leak"),
    ("Ih current", "Ih"),
    ("ICa current", "ICa"),
    ("ITCa current", "ITCa"),
    ("IA current", "IA"),
    ("IM current", "IM"),
    ("SK current", "SK"),
    ("NaP current", "NaP"),
    ("NaR current", "NaR"),
]

_TRACE_COLORS = [
    "#89B4FA",
    "#F38BA8",
    "#A6E3A1",
    "#FAB387",
    "#CBA6F7",
    "#74C7EC",
]

_GBAR_FIELDS = [
    ("gNa", "gNa_v", "#F38BA8"),
    ("gK", "gK_v", "#89B4FA"),
    ("gL", "gL_v", "#A6ADC8"),
    ("gIh", "gIh_v", "#F9E2AF"),
    ("gICa", "gCa_v", "#FAB387"),
    ("gITCa", "gTCa_v", "#F5C2E7"),
    ("gIA", "gA_v", "#A6E3A1"),
    ("gIM", "gIM_v", "#94E2D5"),
    ("gSK", "gSK_v", "#CBA6F7"),
    ("gNaP", "gNaP_v", "#F2CDCD"),
    ("gNaR", "gNaR_v", "#EBA0AC"),
]


class AxonBiophysicsWidget(QWidget):
    """Propagation- and conductance-oriented axon diagnostics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_result = None
        self._last_config = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        ctrl_bar = QHBoxLayout()
        ctrl_bar.setSpacing(6)

        ctrl_bar.addWidget(QLabel("Signal map:"))
        self.combo_signal = QComboBox()
        for label, key in _SIGNAL_OPTIONS:
            self.combo_signal.addItem(label, key)
        self.combo_signal.currentIndexChanged.connect(self._rerender)
        ctrl_bar.addWidget(self.combo_signal)

        ctrl_bar.addSpacing(10)
        ctrl_bar.addWidget(QLabel("Focus compartment:"))
        self.combo_focus = QComboBox()
        self.combo_focus.currentIndexChanged.connect(self._on_focus_changed)
        ctrl_bar.addWidget(self.combo_focus)

        self.spin_custom_comp = QSpinBox()
        self.spin_custom_comp.setMinimum(0)
        self.spin_custom_comp.setMaximum(0)
        self.spin_custom_comp.setEnabled(False)
        self.spin_custom_comp.valueChanged.connect(self._rerender)
        ctrl_bar.addWidget(self.spin_custom_comp)
        ctrl_bar.addStretch()
        layout.addLayout(ctrl_bar)

        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground("#0D1117")
        layout.addWidget(self._glw, stretch=1)

        self._p_trace = self._glw.addPlot(row=0, col=0)
        self._p_trace.setTitle("Propagation Overview", color="#89B4FA")
        self._p_trace.setLabel("left", "Voltage (mV)", color="#CDD6F4")
        self._p_trace.setLabel("bottom", "Time (ms)", color="#CDD6F4")
        self._p_trace.showGrid(x=True, y=True, alpha=0.2)
        self._p_trace.addLegend(offset=(10, 10), labelTextColor="#CDD6F4")

        self._p_heatmap = self._glw.addPlot(row=0, col=1)
        self._p_heatmap.setTitle("Spatiotemporal Map", color="#89B4FA")
        self._p_heatmap.setLabel("left", "Compartment index", color="#CDD6F4")
        self._p_heatmap.setLabel("bottom", "Time (ms)", color="#CDD6F4")

        self._p_gbar = self._glw.addPlot(row=1, col=0)
        self._p_gbar.setTitle("Conductance Density Along Cable", color="#A6E3A1")
        self._p_gbar.setLabel("left", "gbar (mS/cm^2)", color="#CDD6F4")
        self._p_gbar.setLabel("bottom", "Distance from soma (um)", color="#CDD6F4")
        self._p_gbar.showGrid(x=True, y=True, alpha=0.2)
        self._p_gbar.addLegend(offset=(10, 10), labelTextColor="#CDD6F4")

        self._p_focus = self._glw.addPlot(row=1, col=1)
        self._p_focus.setTitle("Focused Compartment Trace", color="#CBA6F7")
        self._p_focus.setLabel("left", "Signal", color="#CDD6F4")
        self._p_focus.setLabel("bottom", "Time (ms)", color="#CDD6F4")
        self._p_focus.showGrid(x=True, y=True, alpha=0.2)
        self._p_focus.addLegend(offset=(10, 10), labelTextColor="#CDD6F4")

        self._info = QLabel("")
        self._info.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._info.setStyleSheet("color:#A6E3A1; font-size:10px; padding:4px;")
        layout.addWidget(self._info)

    def clear_cached_data(self):
        self._last_result = None
        self._last_config = None
        self._clear_plots()

    def _rerender(self, *_):
        if self._last_result is not None and self._last_config is not None:
            self.plot_axon_data(self._last_result, self._last_config)

    def _on_focus_changed(self, *_):
        custom = self.combo_focus.currentData() == -1
        self.spin_custom_comp.setEnabled(custom)
        self._rerender()

    @staticmethod
    def _first_crossing_time(v: np.ndarray, t: np.ndarray, threshold: float = -20.0) -> float:
        above = np.flatnonzero(v >= threshold)
        return float(t[above[0]]) if above.size else float("nan")

    @staticmethod
    def _unique_pairs(pairs):
        seen = set()
        out = []
        for label, idx in pairs:
            idx = int(idx)
            if idx not in seen:
                seen.add(idx)
                out.append((label, idx))
        return out

    def _key_compartments(self, cfg, n_comp: int):
        mc = cfg.morphology
        pairs = [("Soma", 0)]
        if n_comp <= 1:
            return pairs
        if mc.N_ais > 0:
            pairs.append(("AIS start", 1))
            pairs.append(("AIS end", mc.N_ais))
        if mc.N_trunk > 0:
            trunk_end = 1 + mc.N_ais + mc.N_trunk - 1
            pairs.append(("Trunk end", trunk_end))
        if mc.N_b1 > 0:
            b1_tip = 1 + mc.N_ais + mc.N_trunk + mc.N_b1 - 1
            pairs.append(("B1 tip", b1_tip))
        if mc.N_b2 > 0:
            pairs.append(("B2 tip", n_comp - 1))
        elif n_comp > 1:
            pairs.append(("Distal tip", n_comp - 1))
        return self._unique_pairs([(label, min(max(idx, 0), n_comp - 1)) for label, idx in pairs])

    def _populate_focus_choices(self, cfg, n_comp: int):
        current_text = self.combo_focus.currentText()
        self.combo_focus.blockSignals(True)
        self.combo_focus.clear()
        for label, idx in self._key_compartments(cfg, n_comp):
            self.combo_focus.addItem(label, idx)
        self.combo_focus.addItem("Custom", -1)
        restore_idx = self.combo_focus.findText(current_text)
        if restore_idx < 0:
            restore_idx = self.combo_focus.findText("AIS end")
        if restore_idx < 0:
            restore_idx = self.combo_focus.findText("Distal tip")
        if restore_idx < 0:
            restore_idx = 0
        self.combo_focus.setCurrentIndex(restore_idx)
        self.combo_focus.blockSignals(False)
        self.spin_custom_comp.setMaximum(max(0, n_comp - 1))
        self.spin_custom_comp.setEnabled(self.combo_focus.currentData() == -1)

    def _focus_index(self) -> int:
        idx = self.combo_focus.currentData()
        if idx == -1:
            return int(self.spin_custom_comp.value())
        return int(idx if idx is not None else 0)

    def _signal_key(self) -> str:
        key = self.combo_signal.currentData()
        return str(key) if key is not None else "voltage"

    def _resolve_signal_matrix(self, result):
        key = self._signal_key()
        if key == "voltage":
            return np.asarray(result.v_all, dtype=float), "Voltage (mV)", False
        if not hasattr(result, "currents") or key not in result.currents:
            return None, f"{key} not available in this preset", True
        arr = np.asarray(result.currents[key], dtype=float)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        return arr, f"{key} current (uA/cm^2)", True

    def _plot_key_traces(self, t, v_all, cfg):
        self._p_trace.clear()
        for color, (label, idx) in zip(_TRACE_COLORS, self._key_compartments(cfg, v_all.shape[0])):
            self._p_trace.plot(
                t,
                v_all[idx, :],
                pen=pg.mkPen(QColor(color), width=2),
                name=label,
            )
        self._p_trace.addLine(y=-20.0, pen=pg.mkPen("#585B70", style=Qt.PenStyle.DashLine))

    def _plot_heatmap(self, t, matrix, cfg, signal_label: str, diverging: bool):
        self._p_heatmap.clear()
        if matrix is None:
            self._p_heatmap.setTitle(signal_label, color="#F38BA8")
            return

        img = pg.ImageItem(matrix)
        cmap_name = "CET-D1" if diverging else "viridis"
        cmap = pg.colormap.get(cmap_name) or pg.colormap.get("viridis")
        if cmap is not None:
            img.setColorMap(cmap)

        if diverging:
            vmax = float(np.nanmax(np.abs(matrix))) if matrix.size else 1.0
            vmax = max(vmax, 1e-9)
            img.setLevels((-vmax, vmax))
        else:
            vmin = float(np.nanmin(matrix)) if matrix.size else -80.0
            vmax = float(np.nanmax(matrix)) if matrix.size else 40.0
            if np.isclose(vmin, vmax):
                vmax = vmin + 1.0
            img.setLevels((vmin, vmax))

        width = float(t[-1] - t[0]) if len(t) > 1 else 1.0
        img.setRect(QRectF(float(t[0]), -0.5, max(width, 1e-6), float(matrix.shape[0])))
        self._p_heatmap.addItem(img)
        self._p_heatmap.setTitle(signal_label, color="#89B4FA")

        key_ticks = [(idx, label) for label, idx in self._key_compartments(cfg, matrix.shape[0])]
        if key_ticks:
            self._p_heatmap.getAxis("left").setTicks([key_ticks])

    def _plot_conductance_profile(self, dist_um, morph):
        self._p_gbar.clear()
        for label, key, color in _GBAR_FIELDS:
            if key not in morph:
                continue
            vals = np.asarray(morph[key], dtype=float)
            if vals.ndim != 1 or vals.size == 0 or np.nanmax(np.abs(vals)) <= 0.0:
                continue
            self._p_gbar.plot(
                dist_um,
                vals,
                pen=pg.mkPen(QColor(color), width=2),
                name=label,
            )

    def _plot_focus_trace(self, t, result, signal_matrix, signal_label, focus_idx, *, is_voltage_signal: bool):
        self._p_focus.clear()
        self._p_focus.setTitle(f"Focused Trace: comp {focus_idx}", color="#CBA6F7")
        self._p_focus.setLabel("bottom", "Time (ms)", color="#CDD6F4")

        if signal_matrix is None:
            self._p_focus.setLabel("left", "Signal", color="#CDD6F4")
            return

        focus_idx = min(max(int(focus_idx), 0), signal_matrix.shape[0] - 1)
        self._p_focus.setLabel("left", signal_label, color="#CDD6F4")

        self._p_focus.plot(
            t,
            signal_matrix[focus_idx, :],
            pen=pg.mkPen(QColor("#CBA6F7"), width=2.2),
            name=f"Comp {focus_idx}",
        )

        if is_voltage_signal:
            self._p_focus.plot(
                t,
                result.v_soma,
                pen=pg.mkPen(QColor("#89B4FA"), width=1.5, style=Qt.PenStyle.DashLine),
                name="Soma",
            )
            self._p_focus.addLine(y=-20.0, pen=pg.mkPen("#585B70", style=Qt.PenStyle.DashLine))
        elif signal_matrix.shape[0] > 1:
            self._p_focus.plot(
                t,
                signal_matrix[0, :],
                pen=pg.mkPen(QColor("#89B4FA"), width=1.3, style=Qt.PenStyle.DashLine),
                name="Soma ref",
            )
        self._p_focus.addLine(y=0.0, pen=pg.mkPen("#45475A", style=Qt.PenStyle.DotLine))

    def plot_axon_data(self, result, config):
        self._last_result = result
        self._last_config = config
        self._clear_plots()

        if result is None or config is None:
            self._info.setText("")
            return

        t = np.asarray(result.t, dtype=float)
        v_all = np.asarray(result.v_all, dtype=float)
        n_comp = int(result.n_comp)
        if n_comp < 2:
            self._info.setText("Single-compartment model: axonal propagation diagnostics require >=2 compartments.")
            return

        self._populate_focus_choices(config, n_comp)
        focus_idx = min(max(self._focus_index(), 0), n_comp - 1)
        dist_um = np.arange(n_comp, dtype=float) * float(config.morphology.dx) * 1.0e4

        signal_matrix, signal_label, diverging = self._resolve_signal_matrix(result)
        display_matrix = signal_matrix if signal_matrix is not None else v_all
        display_label = signal_label if signal_matrix is not None else "Voltage (fallback: selected signal unavailable)"
        display_diverging = diverging if signal_matrix is not None else False
        self._plot_key_traces(t, v_all, config)
        self._plot_heatmap(t, display_matrix, config, display_label, display_diverging)
        self._plot_conductance_profile(dist_um, result.morph or {})
        self._plot_focus_trace(
            t,
            result,
            display_matrix,
            display_label,
            focus_idx,
            is_voltage_signal=(display_matrix is v_all),
        )

        soma_cross = self._first_crossing_time(result.v_soma, t)
        focus_cross = self._first_crossing_time(v_all[focus_idx, :], t)
        delay_ms = focus_cross - soma_cross if np.isfinite(soma_cross) and np.isfinite(focus_cross) else np.nan
        peak_v = float(np.nanmax(v_all[focus_idx, :]))
        trough_v = float(np.nanmin(v_all[focus_idx, :]))
        self._info.setText(
            f"Focus comp={focus_idx} | distance={dist_um[focus_idx]:.1f} um | "
            f"peak={peak_v:.1f} mV | trough={trough_v:.1f} mV | "
            f"soma->focus delay={delay_ms:.2f} ms"
            if np.isfinite(delay_ms)
            else f"Focus comp={focus_idx} | distance={dist_um[focus_idx]:.1f} um | "
                 f"peak={peak_v:.1f} mV | trough={trough_v:.1f} mV | no threshold crossing"
        )

    def _clear_plots(self):
        self._p_trace.clear()
        self._p_heatmap.clear()
        self._p_gbar.clear()
        self._p_focus.clear()
