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
                                QCheckBox, QGroupBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

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


class OscilloscopeWidget(QWidget):
    """
    Three-pane oscilloscope with checkboxes.
    - Top   : V(t) per compartment, spike markers, threshold line
    - Middle: gate dynamics m, h, n
    - Bottom: ionic currents (fill to zero) + optional filtered stim trace
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._curves_v:    dict = {}
        self._curves_gate: dict = {}
        self._curves_i:    dict = {}

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
            title="<span style='color:#89B4FA'>Membrane Potential  V(t)</span>")
        self._p_v.setLabel('left', 'V', units='mV', color='#CDD6F4')
        self._p_v.showGrid(x=True, y=True, alpha=0.2)
        self._p_v.addLegend(offset=(10, 10), labelTextColor='#CDD6F4')
        self._win.nextRow()

        # Gate variables — row 1
        self._p_g = self._win.addPlot(
            title="<span style='color:#A6E3A1'>Gate Variables  m, h, n</span>")
        self._p_g.setLabel('left', 'probability  [0–1]', color='#CDD6F4')
        self._p_g.showGrid(x=True, y=True, alpha=0.2)
        self._p_g.addLegend(offset=(10, 10), labelTextColor='#CDD6F4')
        self._p_g.setXLink(self._p_v)
        self._p_g.setYRange(-0.05, 1.05)
        self._win.nextRow()

        # Currents — row 2
        self._p_i = self._win.addPlot(
            title="<span style='color:#FAB387'>Ion Currents  (soma)</span>")
        self._p_i.setLabel('left', 'I', units='µA/cm²', color='#CDD6F4')
        self._p_i.setLabel('bottom', 'Time', units='ms', color='#CDD6F4')
        self._p_i.showGrid(x=True, y=True, alpha=0.2)
        self._p_i.addLegend(offset=(10, 10), labelTextColor='#CDD6F4')
        self._p_i.setXLink(self._p_v)

        # Row stretch: V(t) is tallest (3 units), gates and currents share remaining (2 each)
        self._win.ci.layout.setRowStretchFactor(0, 3)
        self._win.ci.layout.setRowStretchFactor(1, 2)
        self._win.ci.layout.setRowStretchFactor(2, 2)

        root.addWidget(self._win, stretch=10)

        # ── Checkbox panel ────────────────────────────────────────────
        cb_widget = QWidget()
        cb_widget.setMaximumWidth(160)
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
        for name in list(CHAN_COLORS.keys()) + ['Stim_filtered']:
            label = name if name != 'Stim_filtered' else 'Stim(filt)'
            cb = QCheckBox(label)
            cb.setChecked(True)
            cb.setStyleSheet("color:#CDD6F4; font-size:11px;")
            cb.stateChanged.connect(lambda _, n=name: self._toggle_i(n))
            il.addWidget(cb)
            self._cb_i[name] = cb
        cb_layout.addWidget(grp_i)
        cb_layout.addStretch()

        root.addWidget(cb_widget, stretch=1)

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

    # ─────────────────────────────────────────────────────────────────
    #  UPDATE — single result
    # ─────────────────────────────────────────────────────────────────
    def update_plots(self, result):
        self.clear()
        t   = result.t
        n   = result.n_comp
        mc  = result.config.morphology

        # ── Voltage traces ────────────────────────────────────────────
        soma_pen = pg.mkPen(color='#4080FF', width=2)
        c_soma   = self._p_v.plot(t, result.v_soma, pen=soma_pen, name="Soma")
        self._curves_v['Soma'] = c_soma

        if n > 1 and mc.N_ais > 0:
            ais_pen = pg.mkPen(color='#FF4040', width=1.5,
                               style=Qt.PenStyle.DashLine)
            c_ais   = self._p_v.plot(t, result.v_all[1, :], pen=ais_pen, name="AIS")
            self._curves_v['AIS'] = c_ais

        if n > 2:
            term_pen = pg.mkPen(color=(0, 200, 100), width=1.2,
                                style=Qt.PenStyle.DotLine)
            c_term   = self._p_v.plot(t, result.v_all[-1, :],
                                       pen=term_pen, name="Terminal")
            self._curves_v['Terminal'] = c_term

        # ── Threshold line at _THRESHOLD_MV ──────────────────────────
        thresh_line = pg.InfiniteLine(
            pos=_THRESHOLD_MV, angle=0,
            pen=pg.mkPen(QColor('#F9E2AF'), width=1,
                         style=Qt.PenStyle.DashLine),
            label=f'{_THRESHOLD_MV:+.0f} mV',
            labelOpts={'position': 0.02, 'color': '#F9E2AF',
                       'anchors': [(0, 1), (0, 1)]}
        )
        self._p_v.addItem(thresh_line)

        # ── Spike markers ─────────────────────────────────────────────
        from core.analysis import detect_spikes
        pks, sp_t, _sp_amp = detect_spikes(result.v_soma, t,
                                            threshold=_THRESHOLD_MV)
        n_spikes = len(sp_t)
        spike_pen = pg.mkPen(QColor('#F38BA8'), width=1,
                              style=Qt.PenStyle.DotLine)
        # Cap markers to avoid visual clutter at very high firing rates
        markers_t = sp_t if n_spikes <= _MAX_SPIKE_MARKERS else sp_t[::max(1, n_spikes // _MAX_SPIKE_MARKERS)]
        for t_sp in markers_t:
            self._p_v.addItem(pg.InfiniteLine(pos=t_sp, angle=90, pen=spike_pen))

        # ── V(t) title with spike stats ───────────────────────────────
        duration_s = t[-1] / 1000.0
        if n_spikes > 0 and duration_s > 0:
            rate_hz   = n_spikes / duration_s
            title_tag = f"{n_spikes} spikes  |  {rate_hz:.1f} Hz"
        else:
            title_tag = "no spikes"
        self._p_v.setTitle(
            f"<span style='color:#89B4FA'>"
            f"Membrane Potential  V(t)  |  {title_tag}"
            f"</span>"
        )

        # ── Gate dynamics ─────────────────────────────────────────────
        from core.analysis import extract_gate_traces
        gates = extract_gate_traces(result)
        for name in ('m', 'h', 'n'):
            if name in gates:
                col  = GATE_COLORS.get(name, (180, 180, 180, 200))
                pen  = pg.mkPen(color=col[:3], width=1.5)
                c    = self._p_g.plot(t, gates[name], pen=pen, name=name)
                self._curves_gate[name] = c
                c.setVisible(self._cb_g[name].isChecked())

        # ── Ion currents ──────────────────────────────────────────────
        for name, curr in result.currents.items():
            col   = CHAN_COLORS.get(name, (120, 120, 120, 150))
            pen   = pg.mkPen(color=col[:3], width=1.5)
            brush = pg.mkBrush(col)
            c     = self._p_i.plot(t, curr, pen=pen,
                                    fillLevel=0.0, fillBrush=brush,
                                    name=f"I_{name}")
            self._curves_i[name] = c
            c.setVisible(self._cb_i.get(name, QCheckBox()).isChecked())

        # ── Filtered stimulus current ─────────────────────────────────
        # Show the post-filter injected current in the currents pane so the
        # user can see how much the dendritic cable attenuated the stimulus.
        if result.v_dendritic_filtered is not None:
            filt_curr_pen = pg.mkPen(color='#89B4FA', width=1.5,
                                      style=Qt.PenStyle.DashLine)
            c_sf = self._p_i.plot(t, result.v_dendritic_filtered,
                                   pen=filt_curr_pen, name="I_stim(filt)")
            self._curves_i['Stim_filtered'] = c_sf
            c_sf.setVisible(self._cb_i['Stim_filtered'].isChecked())

        # Sync checkbox visibility
        self._sync_checkboxes(result)

        # ── Currents title with ATP estimate ──────────────────────────
        self._p_i.setTitle(
            f"<span style='color:#FAB387'>Ion Currents (soma)  |  "
            f"ATP ≈ {result.atp_estimate:.2e} nmol/cm²</span>"
        )

        self._p_v.autoRange()
        self._p_g.autoRange()
        self._p_i.autoRange()

    # ─────────────────────────────────────────────────────────────────
    #  UPDATE — Monte-Carlo cloud
    # ─────────────────────────────────────────────────────────────────
    def update_plots_mc(self, results_list):
        self.clear()
        for res in results_list:
            self._p_v.plot(res.t, res.v_soma,
                           pen=pg.mkPen((70, 130, 255, 40), width=1))

        all_v  = np.array([r.v_soma for r in results_list])
        mean_v = np.mean(all_v, axis=0)
        std_v  = np.std(all_v, axis=0)
        t      = results_list[0].t

        # Mean ± std band
        self._p_v.plot(t, mean_v,
                       pen=pg.mkPen('#FF5050', width=2.5), name="Mean V(t)")
        self._p_v.plot(t, mean_v + std_v,
                       pen=pg.mkPen((200, 80, 80, 100), width=1,
                                    style=Qt.PenStyle.DashLine),
                       name="Mean ± σ")
        self._p_v.plot(t, mean_v - std_v,
                       pen=pg.mkPen((200, 80, 80, 100), width=1,
                                    style=Qt.PenStyle.DashLine))

        # Threshold line
        self._p_v.addItem(pg.InfiniteLine(
            pos=_THRESHOLD_MV, angle=0,
            pen=pg.mkPen(QColor('#F9E2AF'), width=1,
                         style=Qt.PenStyle.DashLine),
            label=f'{_THRESHOLD_MV:+.0f} mV',
            labelOpts={'position': 0.02, 'color': '#F9E2AF',
                       'anchors': [(0, 1), (0, 1)]}
        ))

        n = len(results_list)
        self._p_v.setTitle(
            f"<span style='color:#89B4FA'>Monte-Carlo: {n} trials — mean ± σ</span>"
        )
        self._p_i.setTitle(
            f"<span style='color:#FAB387'>Monte-Carlo: {n} trials — mean ± σ</span>"
        )
        self._p_v.autoRange()

    # ─────────────────────────────────────────────────────────────────
    def _sync_checkboxes(self, result):
        """Show only checkboxes for channels / traces present in result."""
        for name, cb in self._cb_i.items():
            if name == 'Stim_filtered':
                cb.setVisible(result.v_dendritic_filtered is not None)
            else:
                cb.setVisible(name in result.currents)

        for name, cb in self._cb_v.items():
            if name == 'AIS':
                cb.setVisible(result.n_comp > 1 and result.config.morphology.N_ais > 0)
            elif name == 'Terminal':
                cb.setVisible(result.n_comp > 2)
