"""
gui/analytics.py — Full Scientific Analytics Suite v10.0

10 analytical tabs using matplotlib embedded in Qt:
  0. Neuron Passport     — rich biophysical report
  1. Oscilloscope detail — multi-compartment traces
  2. Gate Dynamics       — m, h, n, r, s, u vs time
  3. Equilibrium Curves  — x_inf(V), τ(V) for all gates
  4. Phase Plane         — V vs n + nullclines
  5. Kymograph           — spatiotemporal V(x,t) heatmap
  6. Current Balance     — Cm·dV/dt − (I_stim − I_ion + I_ax)
  7. Energy / Power      — cumulative charge & instantaneous power
  8. Bifurcation         — spike peaks vs parameter
  9. Sweep               — traces + f-I curve
 10. S-D Curve           — strength-duration + Weiss fit
 11. Excitability Map    — 2-D heatmap (I × duration)
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QTabWidget,
                                QLabel, QTextEdit, QHBoxLayout,
                                QSizePolicy, QScrollArea, QPushButton, QMainWindow)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# ── colour palette (matches plots.py) ──────────────────────────────
CHAN_COLORS = {
    'Na':   '#DC3232', 'K':   '#3264DC', 'Leak': '#32A050',
    'Ih':   '#9632C8', 'ICa': '#FA9600', 'IA':   '#00C8C8',
    'SK':   '#C83296',
}
GATE_COLORS = {
    'm': '#FF4040', 'h': '#4080FF', 'n': '#40C040',
    'r': '#A040FF', 's': '#FF9000', 'u': '#009090',
    'a': '#FF40A0', 'b': '#80C0FF',
}


def _mpl_fig(nrows=1, ncols=1, tight=True, **kwargs) -> tuple:
    """Create a matplotlib Figure + FigureCanvas pair."""
    fig = Figure(figsize=(8, 4 * nrows), dpi=90, **kwargs)
    if tight:
        fig.set_tight_layout(True)
    canvas = FigureCanvas(fig)
    return fig, canvas


def _tab_with_toolbar(canvas) -> QWidget:
    """Wrap a canvas in a QWidget with a matplotlib navigation toolbar."""
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.addWidget(NavToolbar(canvas, w))
    lay.addWidget(canvas)
    return w


def _configure_ax_interactive(ax, title: str = '', xlabel: str = '', ylabel: str = '',
                               show_legend: bool = True, grid_alpha: float = 0.2):
    """
    Configure a matplotlib axis for better interactivity and readability.
    
    Improvements (Phase 7.1):
    - Professional grid with better visibility
    - Smart legend placement (outside plot area when possible)
    - Proper spacing and labels
    - Better font sizes
    """
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    
    # Improved grid
    ax.grid(True, alpha=grid_alpha, linestyle='-', linewidth=0.6, color='gray')
    ax.set_axisbelow(True)
    
    # Smart legend placement
    if show_legend and ax.get_legend_handles_labels()[0]:
        ax.legend(loc='best', fontsize=8, framealpha=0.95, 
                  edgecolor='gray', fancybox=True, shadow=False)
    
    # Better formatting
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#666666')


def _spike_detect_kwargs_from_analysis(ana) -> dict:
    return {
        "algorithm": getattr(ana, "spike_detect_algorithm", "peak_repolarization"),
        "threshold": float(getattr(ana, "spike_detect_threshold", -20.0)),
        "prominence": float(getattr(ana, "spike_detect_prominence", 10.0)),
        "baseline_threshold": float(getattr(ana, "spike_detect_baseline_threshold", -50.0)),
        "repolarization_window_ms": float(
            getattr(ana, "spike_detect_repolarization_window_ms", 20.0)
        ),
        "refractory_ms": float(getattr(ana, "spike_detect_refractory_ms", 1.0)),
    }


def _spike_detect_kwargs_from_stats(stats: dict) -> dict:
    return {
        "algorithm": stats.get("spike_detect_algorithm", "peak_repolarization"),
        "threshold": float(stats.get("spike_detect_threshold", -20.0)),
        "prominence": float(stats.get("spike_detect_prominence", 10.0)),
        "baseline_threshold": float(stats.get("spike_detect_baseline_threshold", -50.0)),
        "repolarization_window_ms": float(
            stats.get("spike_detect_repolarization_window_ms", 20.0)
        ),
        "refractory_ms": float(stats.get("spike_detect_refractory_ms", 1.0)),
    }


# ════════════════════════════════════════════════════════════════════
class AnalyticsWidget(QTabWidget):
    """Main analytics widget — updated by MainWindow after each run."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_result = None
        self._last_bif_data = None
        self._last_bif_param_name = None
        self._last_sweep_results = None
        self._last_sweep_param_name = None
        self._last_sd = None
        self._last_exc = None
        self._fullscreen_windows = []
        self._build_tabs()

    # ─────────────────────────────────────────────────────────────────
    #  TAB CONSTRUCTION
    # ─────────────────────────────────────────────────────────────────
    def _build_tabs(self):
        self._btn_fullscreen = QPushButton("Full Screen")
        self._btn_fullscreen.setToolTip("Open analytics in a maximized window")
        self._btn_fullscreen.clicked.connect(self.open_fullscreen)
        self.setCornerWidget(self._btn_fullscreen, Qt.Corner.TopRightCorner)

        # 0 — Passport (text)
        self.passport_view = QTextEdit()
        self.passport_view.setReadOnly(True)
        self.passport_view.setFont(QFont("Consolas", 10))
        self.passport_view.setStyleSheet(
            "background:#0D1117; color:#C9D1D9; border:none;"
        )
        self.addTab(self.passport_view, "🧬 Passport")

        # 1 — Traces (pyqtgraph for interactivity)
        self._build_traces_tab()

        # 2 — Gate Dynamics
        self.fig_gates, cvs = _mpl_fig(3, 1)
        self.addTab(_tab_with_toolbar(cvs), "⚙ Gates")
        self.cvs_gates = cvs

        # 2.5 — Channel Currents (NEW)
        self.fig_currents, cvs = _mpl_fig(3, 1)
        self.addTab(_tab_with_toolbar(cvs), "⚡ Currents")
        self.cvs_currents = cvs

        # 2.6 — Spike Mechanism (why spikes attenuate)
        self.fig_spike_mech, cvs = _mpl_fig(4, 1)
        self.ax_spike_mech = [self.fig_spike_mech.add_subplot(4, 1, k) for k in range(1, 5)]
        self.fig_spike_mech.set_tight_layout({'pad': 2.5})
        self.addTab(_tab_with_toolbar(cvs), "🧪 Spike Mechanism")
        self.cvs_spike_mech = cvs

        # 3 — Equilibrium Curves
        self.fig_equil, cvs = _mpl_fig(2, 2)
        self.ax_equil = [self.fig_equil.add_subplot(2, 2, k) for k in range(1, 5)]
        self.fig_equil.set_tight_layout({'pad': 3.0})
        self.addTab(_tab_with_toolbar(cvs), "📈 Equilibrium")
        self.cvs_equil = cvs

        # 4 — Phase Plane + Nullclines
        self.fig_phase, cvs = _mpl_fig(1, 1)
        self.ax_phase = self.fig_phase.add_subplot(1, 1, 1)
        self.addTab(_tab_with_toolbar(cvs), "🔄 Phase Plane")
        self.cvs_phase = cvs

        # 5 — Kymograph (dynamic layout — 1 or 2 subplots depending on n_comp)
        self.fig_kymo, cvs = _mpl_fig(1, 1)
        self.addTab(_tab_with_toolbar(cvs), "🌊 Kymograph")
        self.cvs_kymo = cvs

        # 6 — Current Balance
        self.fig_balance, cvs = _mpl_fig(2, 1)
        self.ax_balance = [self.fig_balance.add_subplot(2, 1, k) for k in range(1, 3)]
        self.addTab(_tab_with_toolbar(cvs), "⚖ Balance")
        self.cvs_balance = cvs

        # 7 — Energy
        self.fig_energy, cvs = _mpl_fig(2, 1)
        self.ax_energy = [self.fig_energy.add_subplot(2, 1, k) for k in range(1, 3)]
        self.addTab(_tab_with_toolbar(cvs), "⚡ Energy")
        self.cvs_energy = cvs

        # 8 — Bifurcation
        self.fig_bif, cvs = _mpl_fig(2, 2)
        self.ax_bif = [self.fig_bif.add_subplot(2, 2, k) for k in range(1, 5)]
        self.tab_bif = _tab_with_toolbar(cvs)
        self.addTab(self.tab_bif, "🔀 Bifurcation")
        self.cvs_bif = cvs

        # 9 — Sweep (uses colorbar — needs fig.clear() for cleanup)
        self.fig_sweep, cvs = _mpl_fig(2, 2)
        self.tab_sweep = _tab_with_toolbar(cvs)
        self.addTab(self.tab_sweep, "↔ Sweep")
        self.cvs_sweep = cvs

        # 10 — S-D Curve
        self.fig_sd, cvs = _mpl_fig(1, 2)
        self.ax_sd = [self.fig_sd.add_subplot(1, 2, k) for k in range(1, 3)]
        self.tab_sd = _tab_with_toolbar(cvs)
        self.addTab(self.tab_sd, "⏱ S-D Curve")
        self.cvs_sd = cvs

        # 11 — Excitability Map (uses colorbar — needs fig.clear() for cleanup)
        self.fig_excmap, cvs = _mpl_fig(1, 2)
        self.tab_excmap = _tab_with_toolbar(cvs)
        self.addTab(self.tab_excmap, "🗺 Excit. Map")
        self.cvs_excmap = cvs

    def _build_traces_tab(self):
        """pyqtgraph multi-compartment detail traces tab."""
        win = pg.GraphicsLayoutWidget()
        win.setBackground('w')
        self.p_detail_v = win.addPlot(title="Membrane Potential — All Key Compartments")
        self.p_detail_v.setLabel('left', 'V', units='mV')
        self.p_detail_v.showGrid(x=True, y=True, alpha=0.3)
        self.p_detail_v.addLegend(offset=(10, 10))
        win.nextRow()
        self.p_detail_ca = win.addPlot(title="Intracellular [Ca²⁺]ᵢ (soma)")
        self.p_detail_ca.setLabel('left', '[Ca²⁺]', units='nM')
        self.p_detail_ca.setLabel('bottom', 'Time', units='ms')
        self.p_detail_ca.showGrid(x=True, y=True, alpha=0.3)
        self.p_detail_ca.setXLink(self.p_detail_v)
        self.addTab(win, "📊 Traces")

    # ─────────────────────────────────────────────────────────────────
    #  MAIN UPDATE ENTRY POINT
    # ─────────────────────────────────────────────────────────────────
    def update_analytics(self, result):
        """Update all standard tabs from a SimulationResult."""
        self._last_result = result
        from core.analysis import (full_analysis, compute_equilibrium_curves,
                                    compute_optional_equilibrium,
                                    compute_nullclines, compute_current_balance,
                                    extract_gate_traces)
        stats = full_analysis(result)
        self._update_passport(result, stats)
        self._update_traces(result)
        self._update_gates(result)
        self._update_currents(result)
        self._update_spike_mechanism(result, stats)
        self._update_equil(result)
        self._update_phase(result, stats)
        self._update_kymo(result)
        self._update_energy(result)
        if result.morph:
            self._update_balance(result)

    # ─────────────────────────────────────────────────────────────────
    #  0 — NEURON PASSPORT
    # ─────────────────────────────────────────────────────────────────
    def _update_passport(self, result, stats: dict):
        cfg = result.config
        ch  = cfg.channels
        mc  = cfg.morphology

        ns   = stats['n_spikes']
        V_th = stats['V_threshold']
        V_pk = stats['V_peak']
        V_ah = stats['V_ahp']
        hw   = stats['halfwidth_ms']
        fi   = stats['f_initial_hz']
        fs   = stats['f_steady_hz']
        AI   = stats['adaptation_index']
        nt   = stats['neuron_type']
        nt_rule = stats.get('neuron_type_rule', nt)
        nt_ml = stats.get('neuron_type_ml', '—')
        nt_ml_conf = stats.get('neuron_type_ml_confidence', np.nan)
        nt_hybrid = stats.get('neuron_type_hybrid', nt)
        nt_source = stats.get('neuron_type_hybrid_source', 'rule_only')
        nt_hybrid_conf = stats.get('neuron_type_hybrid_confidence', np.nan)
        cv   = stats['conduction_vel_ms']
        tau  = stats['tau_m_ms']
        Rin  = stats['Rin_kohm_cm2']
        lam  = stats['lambda_um']
        Q    = stats['Q_per_channel']
        atp  = stats['atp_nmol_cm2']
        
        # ── NEW: Advanced firing analysis (Phase 7.1) ──
        isi_mean = stats.get('isi_mean_ms', np.nan)
        isi_std  = stats.get('isi_std_ms', np.nan)
        isi_min  = stats.get('isi_min_ms', np.nan)
        isi_max  = stats.get('isi_max_ms', np.nan)
        cv_isi   = stats.get('cv_isi', np.nan)
        lat_1st  = stats.get('first_spike_latency_ms', np.nan)
        refr_per = stats.get('refractory_period_ms', np.nan)
        firing_rel = stats.get('firing_reliability', np.nan)
        lyap_class = stats.get('lyapunov_class', 'disabled')
        lyap_lle_s = stats.get('lle_per_s', np.nan)
        lyap_pairs = int(stats.get('lyapunov_valid_pairs', 0) or 0)
        modulation_valid = bool(stats.get('modulation_valid', False))
        modulation_source = stats.get('modulation_source', '—')
        modulation_plv = stats.get('modulation_plv', np.nan)
        modulation_phase_deg = stats.get('modulation_preferred_phase_deg', np.nan)
        modulation_depth = stats.get('modulation_depth', np.nan)
        modulation_index = stats.get('modulation_index', np.nan)
        modulation_p = stats.get('modulation_p_value', np.nan)
        modulation_z = stats.get('modulation_z_score', np.nan)
        modulation_spikes_used = int(stats.get('modulation_spikes_used', 0) or 0)
        modulation_low_hz = stats.get('modulation_band_low_hz', np.nan)
        modulation_high_hz = stats.get('modulation_band_high_hz', np.nan)
        dt_val = float(np.mean(np.diff(result.t))) if len(result.t) > 1 else 0.0
        current_stats = {}
        for name, curr in result.currents.items():
            if curr is None or len(curr) == 0:
                continue
            i_min = float(np.min(curr))
            i_max = float(np.max(curr))
            q_abs = float(np.sum(np.abs(curr)) * dt_val) if dt_val > 0 else np.nan
            current_stats[name] = (i_min, i_max, q_abs)
        dominant_current = "—"
        if current_stats:
            dominant_current = max(
                current_stats.items(),
                key=lambda kv: kv[1][2] if np.isfinite(kv[1][2]) else -1.0,
            )[0]

        def _first_crossing_ms(v_trace: np.ndarray, threshold: float = -20.0) -> float:
            if len(v_trace) < 2:
                return np.nan
            idx = np.where((v_trace[:-1] < threshold) & (v_trace[1:] >= threshold))[0]
            if len(idx) == 0:
                return np.nan
            return float(result.t[idx[0] + 1])

        delay_junction_ms = np.nan
        delay_terminal_ms = np.nan
        if result.n_comp > 1:
            t_soma = _first_crossing_ms(result.v_soma)
            if np.isfinite(t_soma):
                if result.n_comp > 2:
                    if mc.N_trunk > 0:
                        j_idx = min(1 + mc.N_ais + mc.N_trunk - 1, result.n_comp - 1)
                    elif mc.N_ais > 0:
                        j_idx = min(mc.N_ais, result.n_comp - 1)
                    else:
                        j_idx = 0
                    t_j = _first_crossing_ms(result.v_all[j_idx, :])
                    if np.isfinite(t_j) and t_j >= t_soma:
                        delay_junction_ms = t_j - t_soma
                t_t = _first_crossing_ms(result.v_all[-1, :])
                if np.isfinite(t_t) and t_t >= t_soma:
                    delay_terminal_ms = t_t - t_soma

        def _fmt(v, fmt='.2f', unit=''):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return '—'
            return f"{v:{fmt}} {unit}".strip()

        lines = [
            "╔══════════════════════════════════════════════════════════════════╗",
            "║                    NEURON PASSPORT  v10.1                       ║",
            "╠══════════════════════════════════════════════════════════════════╣",
            f"║  Preset: {cfg.channels.__class__.__name__:<20}  "
            f"T = {cfg.env.T_celsius:.1f}°C  φ = {cfg.env.phi:.3f}          ║",
            f"║  Channels: " + " ".join(
                c for c, en in [('Na', True), ('K', True), ('Leak', True),
                                 ('Ih', ch.enable_Ih), ('ICa', ch.enable_ICa),
                                 ('IA', ch.enable_IA), ('SK', ch.enable_SK)]
                if en
            ) + " " * 30,
            "╠══════════════════════════════════════════════════════════════════╣",
            "║  PASSIVE MEMBRANE PROPERTIES                                    ║",
            f"║    τ_m   = {_fmt(tau, '.3f', 'ms'):<12}  "
            f"Rin   = {_fmt(Rin, '.3f', 'kΩ·cm²'):<16}  λ = {_fmt(lam, '.1f', 'µm')}  ║",
            "╠══════════════════════════════════════════════════════════════════╣",
            f"║  SPIKE COUNT: {ns:<3}  {'(no spikes)' if ns == 0 else ''}",
        ]

        if ns > 0:
            lines += [
                f"║    Threshold  = {_fmt(V_th, '+.1f', 'mV'):<12}  "
                f"Peak  = {_fmt(V_pk, '+.1f', 'mV'):<12}  "
                f"AHP   = {_fmt(V_ah, '+.1f', 'mV')}  ║",
                f"║    Halfwidth  = {_fmt(hw, '.3f', 'ms'):<12}  "
                f"dV/dt = +{_fmt(stats['dvdt_max'], '.0f', 'mV/ms')} / "
                f"{_fmt(stats['dvdt_min'], '.0f', 'mV/ms')}  ║",
            ]
        if ns > 1:
            lines += [
                f"║    f_initial  = {_fmt(fi, '.1f', 'Hz'):<12}  "
                f"f_steady = {_fmt(fs, '.1f', 'Hz'):<12}  "
                f"AI = {_fmt(AI, '+.3f')}  ║",
                f"║    Type (rule): {nt_rule:<28}  ║",
                f"║    Type (ML): {nt_ml:<13} conf={_fmt(nt_ml_conf, '.2f'):<8} source={nt_source:<10} ║",
                f"║    Type (hybrid): {nt_hybrid:<19} conf={_fmt(nt_hybrid_conf, '.2f')}        ║",
            ]
        if cv > 0:
            lines.append(
                f"║    Cond. vel. = {_fmt(cv, '.3f', 'm/s'):<12}  ║"
            )

        # ──────────────────────────────────────────────────────────────
        # NEW SECTION: FIRING DYNAMICS (Phase 7.1)
        # ──────────────────────────────────────────────────────────────
        if ns > 1:
            lines += [
                "╠══════════════════════════════════════════════════════════════════╣",
                "║  FIRING DYNAMICS (Advanced Analysis)                            ║",
                f"║    1st spike latency = {_fmt(lat_1st, '.2f', 'ms'):<20}  "
                f"Refr. period = {_fmt(refr_per, '.3f', 'ms')}  ║",
                f"║    ISI (mean ± std)  = {_fmt(isi_mean, '.3f', 'ms'):<8} ± "
                f"{_fmt(isi_std, '.3f', 'ms'):<8}  CV = {_fmt(cv_isi, '.3f')}  ║",
                f"║    ISI range: [{_fmt(isi_min, '.3f', 'ms'):<8}, "
                f"{_fmt(isi_max, '.3f', 'ms'):<8}]   "
                f"Reliability = {_fmt(firing_rel, '.3f')}  ║",
            ]

        lines += [
            "╠══════════════════════════════════════════════════════════════════╣",
            "║  DYNAMICAL STABILITY (LLE/FTLE)                                 ║",
        ]
        if lyap_class == 'disabled':
            lines.append("║    Analysis disabled (enable_lyapunov=False)                    ║")
        else:
            lines += [
                f"║    Class = {lyap_class:<21}  LLE = {_fmt(lyap_lle_s, '+.4f', '1/s'):<14}  ║",
                f"║    Valid trajectory pairs = {lyap_pairs:<5}                              ║",
            ]

        lines += [
            "╠══════════════════════════════════════════════════════════════════╣",
            "║  MODULATION DECOMPOSITION (NON-FFT)                             ║",
        ]
        if not modulation_valid:
            lines.append("║    Disabled or insufficient spikes for robust estimate           ║")
        else:
            lines += [
                f"║    Source={modulation_source:<9} Band={_fmt(modulation_low_hz, '.1f', 'Hz')}..{_fmt(modulation_high_hz, '.1f', 'Hz'):<10}  ║",
                f"║    PLV={_fmt(modulation_plv, '.3f'):<10} Phase={_fmt(modulation_phase_deg, '.1f', 'deg'):<14} Nsp={modulation_spikes_used:<5}  ║",
                f"║    Depth={_fmt(modulation_depth, '.3f'):<10} MI={_fmt(modulation_index, '.3f'):<10} p={_fmt(modulation_p, '.3f'):<9} z={_fmt(modulation_z, '.2f')}  ║",
            ]

        lines += [
            "╠══════════════════════════════════════════════════════════════════╣",
            "║  CHANNEL ENGAGEMENT                                              ║",
            f"║    Dominant |Q| channel: {dominant_current:<10}                        ║",
        ]
        top_channels = sorted(
            current_stats.items(),
            key=lambda kv: kv[1][2] if np.isfinite(kv[1][2]) else -1.0,
            reverse=True,
        )[:4]
        for name, (i_min, i_max, q_abs) in top_channels:
            lines.append(
                f"║    {name:<5} Imin={_fmt(i_min, '.2f', 'uA/cm²'):<14} "
                f"Imax={_fmt(i_max, '.2f', 'uA/cm²'):<14} Qabs={_fmt(q_abs, '.2f', 'nC/cm²')}  ║"
            )
        if result.n_comp > 1:
            lines.append(
                f"║    Delay soma->junction={_fmt(delay_junction_ms, '.2f', 'ms'):<10} "
                f"soma->terminal={_fmt(delay_terminal_ms, '.2f', 'ms')}  ║"
            )

        lines += [
            "╠══════════════════════════════════════════════════════════════════╣",
            "║  ENERGY                                                         ║",
        ]
        for name, q in Q.items():
            lines.append(f"║    Q_{name:<5} = {q:.2f} nC/cm²" + " " * 30 + "║")
        lines += [
            f"║    ATP est. = {atp:.4e} nmol/cm²" + " " * 25 + "║",
            "╚══════════════════════════════════════════════════════════════════╝",
        ]

        self.passport_view.setPlainText("\n".join(lines))

    # ─────────────────────────────────────────────────────────────────
    #  1 — TRACES (pyqtgraph)
    # ─────────────────────────────────────────────────────────────────
    def _update_traces(self, result):
        self.p_detail_v.clear()
        self.p_detail_ca.clear()
        self.p_detail_v.addLegend(offset=(10, 10))

        t = result.t
        mc = result.config.morphology
        n  = result.n_comp

        comp_colors = ['b', 'r', 'm', 'g', 'c']
        labels = ['Soma']
        key_indices = [0]

        if n > 1 and mc.N_ais > 0:
            key_indices.append(1)
            labels.append('AIS')
        if mc.N_trunk > 0:
            junc = 1 + mc.N_ais + mc.N_trunk - 1
        elif mc.N_ais > 0:
            junc = mc.N_ais
        else:
            junc = 0
        if 1 <= junc < n:
            key_indices.append(min(junc, n-1))
            labels.append('Junction')
        if n > 2:
            key_indices.append(n - 1)
            labels.append('Terminal')

        for i, (idx, lbl) in enumerate(zip(key_indices, labels)):
            color = comp_colors[i % len(comp_colors)]
            self.p_detail_v.plot(t, result.v_all[idx, :],
                                  pen=pg.mkPen(color, width=2), name=lbl)

        if result.ca_i is not None:
            self.p_detail_ca.plot(t, result.ca_i[0, :] * 1e6,
                                   pen=pg.mkPen('b', width=2), name="[Ca²⁺]ᵢ soma")
            self.p_detail_ca.show()
        else:
            self.p_detail_ca.hide()

        self.p_detail_v.autoRange()

    # ─────────────────────────────────────────────────────────────────
    #  2 — GATE DYNAMICS
    # ─────────────────────────────────────────────────────────────────
    def _update_gates(self, result):
        from core.analysis import extract_gate_traces
        gates = extract_gate_traces(result)
        t = result.t

        self.fig_gates.clear()
        self.fig_gates.set_tight_layout({'pad': 2.5})

        n_rows = max(2, len(gates) + 1)
        ax_v = self.fig_gates.add_subplot(n_rows, 1, 1)
        ax_v.plot(t, result.v_soma, color='#2060CC', linewidth=2.5, alpha=0.9)
        _configure_ax_interactive(ax_v, title='Membrane Potential (V_soma)',
                                  xlabel='', ylabel='V (mV)', show_legend=False)
        ax_v.tick_params(labelbottom=False)

        for i, (name, trace) in enumerate(gates.items(), start=2):
            ax = self.fig_gates.add_subplot(n_rows, 1, i)
            color = GATE_COLORS.get(name, '#888888')
            ax.plot(t, trace, color=color, lw=2.5, label=f'{name} activation', alpha=0.9)
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel(f'{name}(t)', fontsize=10, fontweight='bold')
            ax.tick_params(labelbottom=(i == n_rows))
            if i < n_rows:
                ax.tick_params(labelbottom=False)
            _configure_ax_interactive(ax, show_legend=True, grid_alpha=0.15)

        self.fig_gates.axes[-1].set_xlabel('Time (ms)', fontsize=10, fontweight='bold')
        self.cvs_gates.draw_idle()

    # ─────────────────────────────────────────────────────────────────
    #  2.5 — CHANNEL CURRENTS (NEW)
    # ─────────────────────────────────────────────────────────────────
    def _update_currents(self, result):
        """Plot channel currents with membrane potential overlay."""
        t = result.t
        
        self.fig_currents.clear()
        self.fig_currents.set_tight_layout({'pad': 2.5})

        # Count non-zero current traces
        currents = {name: curr for name, curr in result.currents.items() 
                   if np.max(np.abs(curr)) > 1e-9}
        
        n_rows = max(2, len(currents) + 1)
        
        # Row 1: Membrane potential
        ax_v = self.fig_currents.add_subplot(n_rows, 1, 1)
        ax_v.plot(t, result.v_soma, color='#2060CC', lw=2.5, alpha=0.9)
        _configure_ax_interactive(ax_v, title='Membrane Potential (V_soma)',
                                  xlabel='', ylabel='V (mV)', show_legend=False)
        ax_v.tick_params(labelbottom=False)

        # Remaining rows: Individual currents
        for i, (name, curr) in enumerate(currents.items(), start=2):
            ax = self.fig_currents.add_subplot(n_rows, 1, i)
            color = CHAN_COLORS.get(name, '#888888')
            ax.plot(t, curr, color=color, lw=2.5, label=f'I_{name}', alpha=0.9)
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.2, linewidth=0.8)
            ax.set_ylabel(f'I_{name} (pA)', fontsize=10, fontweight='bold')
            ax.tick_params(labelbottom=(i == n_rows))
            if i < n_rows:
                ax.tick_params(labelbottom=False)
            _configure_ax_interactive(ax, show_legend=True, grid_alpha=0.15)

        self.fig_currents.axes[-1].set_xlabel('Time (ms)', fontsize=10, fontweight='bold')
        self.cvs_currents.draw_idle()

    def _update_spike_mechanism(self, result, stats: dict):
        """
        Explain spike attenuation using per-spike ion/channel dynamics.
        """
        from core.analysis import detect_spikes

        t = np.asarray(result.t, dtype=float)
        v = np.asarray(result.v_soma, dtype=float)
        ax1, ax2, ax3, ax4 = self.ax_spike_mech
        for ax in self.ax_spike_mech:
            ax.cla()

        kwargs = _spike_detect_kwargs_from_stats(stats)
        peak_idx, spike_times, _ = detect_spikes(v, t, **kwargs)
        n_sp = len(spike_times)

        ax1.plot(t, v, color="#2060CC", lw=2.0, label="V_soma")
        if n_sp > 0:
            ax1.scatter(
                t[peak_idx],
                v[peak_idx],
                c=np.arange(n_sp),
                cmap="plasma",
                s=26,
                label="spike peaks",
                zorder=4,
            )
        _configure_ax_interactive(
            ax1,
            title=f"Spike Peaks Timeline (N={n_sp})",
            xlabel="Time (ms)",
            ylabel="V (mV)",
            show_legend=True,
        )

        if n_sp < 2:
            ax2.text(
                0.02,
                0.55,
                "Need at least 2 spikes for attenuation diagnostics.",
                transform=ax2.transAxes,
                fontsize=10,
                color="#444444",
            )
            ax2.set_axis_off()
            ax3.set_axis_off()
            ax4.set_axis_off()
            self.cvs_spike_mech.draw_idle()
            return

        sp_no = np.arange(1, n_sp + 1)
        peak_v = v[peak_idx]
        ax2.plot(sp_no, peak_v, "o-", color="#1F77B4", lw=1.8, label="V_peak")
        ax2.set_ylabel("Peak V (mV)", fontsize=10, fontweight="bold")
        ax2.grid(True, alpha=0.25)

        if result.ca_i is not None and len(result.ca_i) > 0:
            ca_nM = np.asarray(result.ca_i[0, :], dtype=float) * 1e6
            ca_sp = ca_nM[peak_idx]
            ax2b = ax2.twinx()
            ax2b.plot(sp_no, ca_sp, "s--", color="#D62728", lw=1.4, label="Ca_i@spike")
            ax2b.set_ylabel("Ca_i (nM)", fontsize=10, fontweight="bold", color="#D62728")
            ax2b.tick_params(axis="y", labelcolor="#D62728")

        _configure_ax_interactive(
            ax2,
            title="Per-Spike Amplitude and Calcium Load",
            xlabel="Spike #",
            ylabel="Peak V (mV)",
            show_legend=True,
        )

        curr_candidates = ["Na", "K", "ICa", "IA", "SK", "Ih", "Leak"]
        traces = {k: np.asarray(result.currents[k], dtype=float) for k in curr_candidates if k in result.currents}
        for name, tr in traces.items():
            ax3.plot(sp_no, tr[peak_idx], marker=".", lw=1.2, label=name, color=CHAN_COLORS.get(name, "#555555"))
        _configure_ax_interactive(
            ax3,
            title="Currents Sampled at Spike Peaks",
            xlabel="Spike #",
            ylabel="I_channel (uA/cm2)",
            show_legend=True,
        )

        # Time-resolved channel/Ca activity to explain attenuation causes.
        dt = float(np.mean(np.diff(t))) if len(t) > 1 else 0.1
        smooth_pts = max(3, int(round(2.0 / max(dt, 1e-6))))
        if smooth_pts % 2 == 0:
            smooth_pts += 1

        def _smooth(x: np.ndarray, n: int) -> np.ndarray:
            if n <= 1:
                return x
            kernel = np.ones(n, dtype=float) / float(n)
            return np.convolve(x, kernel, mode="same")

        plotted = 0
        for name in ("Na", "K", "ICa", "IA", "SK", "Ih", "Leak"):
            if name not in traces:
                continue
            y_abs = np.abs(_smooth(traces[name], smooth_pts))
            ymax = float(np.max(y_abs))
            if ymax <= 1e-12:
                continue
            ax4.plot(
                t,
                y_abs / ymax,
                lw=1.3,
                alpha=0.9,
                label=f"|I_{name}| norm",
                color=CHAN_COLORS.get(name, "#666666"),
            )
            plotted += 1

        if result.ca_i is not None and len(result.ca_i) > 0:
            ca_nM = np.asarray(result.ca_i[0, :], dtype=float) * 1e6
            ca_s = _smooth(ca_nM, smooth_pts)
            ca_rng = float(np.max(ca_s) - np.min(ca_s))
            if ca_rng > 1e-12:
                ca_norm = (ca_s - np.min(ca_s)) / ca_rng
                ax4.plot(t, ca_norm, "--", lw=1.4, color="#D62728", label="Ca_i norm")
                plotted += 1

        if plotted == 0:
            ax4.text(
                0.02,
                0.55,
                "No channel activity traces available for time-resolved causality view.",
                transform=ax4.transAxes,
                fontsize=9.5,
                color="#444444",
            )
            ax4.set_axis_off()
        else:
            _configure_ax_interactive(
                ax4,
                title="Time-Resolved Channel/Ca Activity (normalized, smoothed)",
                xlabel="Time (ms)",
                ylabel="Norm activity",
                show_legend=True,
            )

        # Heuristic explanation block
        peak_drop = float(peak_v[-1] - peak_v[0])
        reasons = []
        if peak_drop < -5.0:
            reasons.append(f"Peak attenuation detected: ΔV_peak={peak_drop:.1f} mV")

        if "Na" in traces:
            na_mag = np.abs(traces["Na"][peak_idx])
            if na_mag[0] > 1e-9:
                na_rel = float((na_mag[-1] - na_mag[0]) / na_mag[0])
                if na_rel < -0.20:
                    reasons.append(f"Na drive decreased at peaks ({na_rel*100:.1f}%) -> possible Na inactivation")

        k_like = None
        if "SK" in traces:
            k_like = np.abs(traces["SK"][peak_idx])
            k_name = "SK"
        elif "K" in traces:
            k_like = np.abs(traces["K"][peak_idx])
            k_name = "K"
        else:
            k_name = None
        if k_like is not None and k_like[0] > 1e-9:
            k_rel = float((k_like[-1] - k_like[0]) / k_like[0])
            if k_rel > 0.20:
                reasons.append(f"{k_name} outward component increased ({k_rel*100:.1f}%)")

        if result.ca_i is not None and len(result.ca_i) > 0:
            ca_nM = np.asarray(result.ca_i[0, :], dtype=float) * 1e6
            ca_sp = ca_nM[peak_idx]
            if ca_sp[0] > 1e-9:
                ca_rel = float((ca_sp[-1] - ca_sp[0]) / ca_sp[0])
                if ca_rel > 0.25:
                    reasons.append(f"Ca_i accumulation at peaks ({ca_rel*100:.1f}%) may promote adaptation/block")

        if not reasons:
            reasons = ["No dominant attenuation driver from peak-sampled metrics; inspect full-current traces."]

        ax3.text(
            0.01,
            0.02,
            " | ".join(reasons[:3]),
            transform=ax3.transAxes,
            fontsize=8.5,
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#F8F8F8", edgecolor="#CCCCCC", alpha=0.9),
        )

        self.cvs_spike_mech.draw_idle()

    # ─────────────────────────────────────────────────────────────────
    #  3 — EQUILIBRIUM CURVES
    # ─────────────────────────────────────────────────────────────────
    def _update_equil(self, result):
        from core.analysis import (compute_equilibrium_curves,
                                    compute_optional_equilibrium)
        cfg   = result.config
        phi   = cfg.env.phi
        V_rng = np.linspace(-100, 60, 500)
        eq    = compute_equilibrium_curves(V_rng, phi)
        opt   = compute_optional_equilibrium(V_rng, cfg, phi)

        ax1, ax2, ax3, ax4 = self.ax_equil
        for ax in self.ax_equil:
            ax.cla()

        # x_inf(V) — improved layout
        ax1.plot(V_rng, eq['m_inf'], color=GATE_COLORS['m'], lw=2.5, label='m∞ (Na act)', alpha=0.9)
        ax1.plot(V_rng, eq['h_inf'], color=GATE_COLORS['h'], lw=2.5, label='h∞ (Na inact)', alpha=0.9)
        ax1.plot(V_rng, eq['n_inf'], color=GATE_COLORS['n'], lw=2.5, label='n∞ (K act)', alpha=0.9)
        for k in ('r_inf', 's_inf', 'u_inf', 'a_inf', 'b_inf'):
            if k in opt:
                lbl = k.replace('_inf', '∞')
                ax1.plot(V_rng, opt[k], lw=1.8, ls='--', label=lbl, alpha=0.8)
        ax1.set_ylim(-0.05, 1.05)
        _configure_ax_interactive(ax1, title='Steady-state gating (x∞)', 
                                  xlabel='V (mV)', ylabel='x∞', show_legend=True)

        # τ(V) — main gating time constants
        ax2.plot(V_rng, eq['tau_m'], color=GATE_COLORS['m'], lw=2.5, label='τₘ', alpha=0.9)
        ax2.plot(V_rng, eq['tau_h'], color=GATE_COLORS['h'], lw=2.5, label='τₕ', alpha=0.9)
        ax2.plot(V_rng, eq['tau_n'], color=GATE_COLORS['n'], lw=2.5, label='τₙ', alpha=0.9)
        for k in ('tau_r', 'tau_s', 'tau_u', 'tau_a', 'tau_b'):
            if k in opt:
                ax2.plot(V_rng, opt[k], lw=1.8, ls='--', label=k, alpha=0.8)
        _configure_ax_interactive(ax2, title=f'Time constants (φ = {phi:.2f})',
                                  xlabel='V (mV)', ylabel='τ (ms)', show_legend=True)

        # Phase portrait V-m and V-h
        ax3.plot(result.v_soma, result.y[result.n_comp, :],
                 color=GATE_COLORS['m'], lw=2, label='m (Na act)', alpha=0.9)
        ax3.plot(result.v_soma, result.y[2 * result.n_comp, :],
                 color=GATE_COLORS['h'], lw=2, label='h (Na inact)', alpha=0.9)
        _configure_ax_interactive(ax3, title='V – Gate Phase Portraits',
                                  xlabel='V (mV)', ylabel='Gate value', show_legend=True)

        # gNa_eff and gK_eff over time
        m_t  = result.y[result.n_comp, :]
        h_t  = result.y[2 * result.n_comp, :]
        n_t  = result.y[3 * result.n_comp, :]
        t    = result.t
        g_Na = result.config.channels.gNa_max * (m_t ** 3) * h_t
        g_K  = result.config.channels.gK_max  * (n_t ** 4)
        ax4.plot(t, g_Na, color=GATE_COLORS['m'], lw=2.5, label='g_Na(t)', alpha=0.9)
        ax4.plot(t, g_K,  color=GATE_COLORS['n'], lw=2.5, label='g_K(t)', alpha=0.9)
        _configure_ax_interactive(ax4, title='Effective Conductances',
                                  xlabel='Time (ms)', ylabel='g (mS/cm²)', show_legend=True)

        self.cvs_equil.draw_idle()

    # ─────────────────────────────────────────────────────────────────
    #  4 — PHASE PLANE + NULLCLINES
    # ─────────────────────────────────────────────────────────────────
    def _update_phase(self, result, stats: dict):
        from core.analysis import compute_nullclines

        t     = result.t
        V     = result.v_soma
        n_t   = result.y[3 * result.n_comp, :]   # n gate
        cfg   = result.config
        I_stm = cfg.stim.Iext if cfg.stim.stim_type == 'const' else 0.0

        V_rng               = np.linspace(-100, 60, 500)
        n_V_null, n_n_null  = compute_nullclines(V_rng, cfg, I_stm)

        ax = self.ax_phase
        ax.cla()

        # Trajectory
        ax.plot(V, n_t, color='#F0B020', lw=1.5, zorder=3, label='AP trajectory')
        ax.plot(V[0], n_t[0], 'go', ms=8, zorder=5, label='Resting state')

        # Spike detection markers
        if stats['n_spikes'] > 0:
            from core.analysis import detect_spikes
            pk_idx, _, _ = detect_spikes(V, t, **_spike_detect_kwargs_from_stats(stats))
            ax.plot(V[pk_idx], n_t[pk_idx], 'r*', ms=12, zorder=6, label='Spike peaks')

        # Nullclines
        ax.plot(V_rng, n_n_null, color='#40CC40', lw=2, ls='--', label='dn/dt = 0  (n∞)')
        valid = ~np.isnan(n_V_null)
        ax.plot(V_rng[valid], n_V_null[valid], color='#CC4040', lw=2,
                ls='--', label='dV/dt = 0')

        ax.set_xlabel('V (mV)',  fontsize=11)
        ax.set_ylabel('n  [K⁺ activation]', fontsize=11)
        ax.set_title('Phase Plane  (V – n)  with Nullclines', fontsize=12)
        ax.legend(fontsize=9);  ax.grid(alpha=0.3)
        ax.set_xlim(-100, 60);  ax.set_ylim(-0.05, 1.05)

        if cfg.channels.enable_Ih or cfg.channels.enable_ICa or cfg.channels.enable_IA:
            ax.text(0.01, 0.01,
                    "⚠ Nullclines include Na+K+Leak only",
                    transform=ax.transAxes, fontsize=8, color='gray')

        self.fig_phase.tight_layout()
        self.cvs_phase.draw_idle()

    # ─────────────────────────────────────────────────────────────────
    #  5 — KYMOGRAPH
    # ─────────────────────────────────────────────────────────────────
    def _update_kymo(self, result):
        self.fig_kymo.clear()
        n = result.n_comp

        if n < 2:
            ax = self.fig_kymo.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'Single-compartment mode\n(no kymograph)',
                    ha='center', va='center', fontsize=14, color='gray',
                    transform=ax.transAxes)
            self.cvs_kymo.draw_idle()
            return

        mc = result.config.morphology
        t  = result.t
        V  = result.v_all      # shape (N_comp, N_time)

        # Build two axonal paths: soma → Branch1 tip, soma → Branch2 tip
        i_ais_s  = 1
        i_tr_s   = 1 + mc.N_ais
        i_branch = i_tr_s + mc.N_trunk
        i_b1s    = i_branch + 1
        i_b2s    = i_b1s + mc.N_b1

        path1 = list(range(0, min(i_b1s + mc.N_b1, n)))
        path2 = list(range(0, min(i_b2s + mc.N_b2, n)))

        ax1 = self.fig_kymo.add_subplot(2, 1, 1)
        im1 = ax1.imshow(
            V[path1, :], aspect='auto', origin='lower',
            extent=[t[0], t[-1], 0, len(path1)],
            cmap='plasma', vmin=V.min(), vmax=V.max()
        )
        self.fig_kymo.colorbar(im1, ax=ax1, label='V (mV)')
        ax1.set_ylabel('Compartment (soma → B1 tip)')
        ax1.set_title('Kymograph — Path to Branch 1')

        ax2 = self.fig_kymo.add_subplot(2, 1, 2)
        im2 = ax2.imshow(
            V[path2, :], aspect='auto', origin='lower',
            extent=[t[0], t[-1], 0, len(path2)],
            cmap='plasma', vmin=V.min(), vmax=V.max()
        )
        self.fig_kymo.colorbar(im2, ax=ax2, label='V (mV)')
        ax2.set_xlabel('Time (ms)');  ax2.set_ylabel('Compartment (soma → B2 tip)')
        ax2.set_title('Kymograph — Path to Branch 2')

        self.fig_kymo.tight_layout()
        self.cvs_kymo.draw_idle()

    # ─────────────────────────────────────────────────────────────────
    #  6 — CURRENT BALANCE
    # ─────────────────────────────────────────────────────────────────
    def _update_balance(self, result):
        from core.analysis import compute_current_balance
        try:
            I_bal = compute_current_balance(result, result.morph)
        except Exception as e:
            return

        t  = result.t
        err = float(np.max(np.abs(I_bal)))

        ax1, ax2 = self.ax_balance
        ax1.cla()
        ax2.cla()
        ax1.plot(t, I_bal, color='#DC3232', lw=1)
        ax1.axhline(0, color='k', lw=0.8, ls='--')
        ax1.set_ylabel('I_balance (µA/cm²)')
        ax1.set_title(f'Current Balance (soma) — max|error| = {err:.5f} µA/cm²  '
                      f'{"✓ Good" if err < 0.05 else "⚠ Check solver settings"}')
        ax1.grid(alpha=0.3)

        ax2.semilogy(t, np.abs(I_bal) + 1e-12, color='#3264DC', lw=1)
        ax2.set_xlabel('Time (ms)');  ax2.set_ylabel('|Error|  log scale')
        ax2.set_title('Absolute balance error (log)')
        ax2.grid(alpha=0.3)

        self.fig_balance.tight_layout()
        self.cvs_balance.draw_idle()

    # ─────────────────────────────────────────────────────────────────
    #  7 — ENERGY
    # ─────────────────────────────────────────────────────────────────
    def _update_energy(self, result):
        t   = result.t
        dt  = float(t[1] - t[0]) if len(t) > 1 else 0.05

        ax1, ax2 = self.ax_energy
        ax1.cla()
        ax2.cla()

        P_total = np.zeros_like(t)
        for name, curr in result.currents.items():
            color = CHAN_COLORS.get(name, '#888888')
            E_rev = _get_E_rev(name, result.config.channels)
            Q_cum = np.cumsum(np.abs(curr)) * dt
            ax1.plot(t, Q_cum, color=color, lw=1.5, label=f'Q_{name}')
            P = np.abs(curr * (result.v_soma - E_rev))
            ax2.plot(t, P, color=color, lw=1, alpha=0.8, label=f'P_{name}')
            P_total += P

        ax2.plot(t, P_total, 'k-', lw=2, label='Total', zorder=5)

        ax1.set_ylabel('Cumulative charge (nC/cm²)')
        ax1.set_title('Energy — Cumulative ionic charge transfer')
        ax1.legend(fontsize=8);  ax1.grid(alpha=0.3)

        ax2.set_xlabel('Time (ms)');  ax2.set_ylabel('Power (µW/cm²)')
        ax2.set_title(f'Instantaneous power   ATP ≈ {result.atp_estimate:.3e} nmol/cm²')
        ax2.legend(fontsize=8);  ax2.grid(alpha=0.3)

        self.fig_energy.tight_layout()
        self.cvs_energy.draw_idle()

    # ─────────────────────────────────────────────────────────────────
    #  8 — BIFURCATION
    # ─────────────────────────────────────────────────────────────────
    def update_bifurcation(self, bif_data: list, param_name: str):
        self._last_bif_data = bif_data
        self._last_bif_param_name = param_name
        vals   = np.array([d['val']   for d in bif_data])
        vmax   = np.array([d['max']   for d in bif_data])
        vmin   = np.array([d['min']   for d in bif_data])
        freq   = np.array([d['freq']  for d in bif_data])
        n_sp   = np.array([d['n_sp']  for d in bif_data])

        ax1, ax2, ax3, ax4 = self.ax_bif
        for ax in self.ax_bif:
            ax.cla()

        for d in bif_data:
            pks = d.get('peaks', [])
            if pks:
                ax1.plot([d['val']] * len(pks), pks, 'b.', ms=4)
            else:
                ax1.plot(d['val'], d['max'], 'r.', ms=4)

        ax1.set_xlabel(param_name);  ax1.set_ylabel('V peaks (mV)')
        ax1.set_title('Bifurcation diagram');  ax1.grid(alpha=0.3)

        ax2.plot(vals, vmax, 'r-', lw=1.5, label='Vmax')
        ax2.plot(vals, vmin, 'b-', lw=1.5, label='Vmin')
        ax2.set_xlabel(param_name);  ax2.set_ylabel('V (mV)')
        ax2.set_title('Vmax / Vmin');  ax2.legend();  ax2.grid(alpha=0.3)

        ax3.plot(vals, freq, 'g.-', lw=1.5)
        ax3.set_xlabel(param_name);  ax3.set_ylabel('f (Hz)')
        ax3.set_title('Firing frequency');  ax3.grid(alpha=0.3)

        ax4.plot(vals, n_sp, 'b.-', lw=1.5)
        ax4.set_xlabel(param_name);  ax4.set_ylabel('N spikes')
        ax4.set_title('Spike count');  ax4.grid(alpha=0.3)

        self.fig_bif.tight_layout()
        self.cvs_bif.draw_idle()
        self.setCurrentWidget(self.tab_bif)

    # ─────────────────────────────────────────────────────────────────
    #  9 — SWEEP
    # ─────────────────────────────────────────────────────────────────
    def update_sweep(self, sweep_results: list, param_name: str):
        """sweep_results: list of (param_value, SimulationResult|None)"""
        self._last_sweep_results = sweep_results
        self._last_sweep_param_name = param_name
        from core.analysis import detect_spikes

        self.fig_sweep.clear()
        ax1 = self.fig_sweep.add_subplot(2, 2, 1)
        ax2 = self.fig_sweep.add_subplot(2, 2, 2)
        ax3 = self.fig_sweep.add_subplot(2, 2, 3)
        ax4 = self.fig_sweep.add_subplot(2, 2, 4)

        n = len(sweep_results)
        cmap = plt.colormaps['plasma'](np.linspace(0.1, 0.9, n))

        param_vals, peaks, freqs, n_sps = [], [], [], []

        for i, (val, res) in enumerate(sweep_results):
            if res is None:
                continue
            param_vals.append(val)
            ax1.plot(res.t, res.v_soma, color=cmap[i], lw=1, alpha=0.8)
            pks, sp_t, sp_amp = detect_spikes(
                res.v_soma, res.t, **_spike_detect_kwargs_from_analysis(res.config.analysis)
            )
            peaks.append(float(np.max(res.v_soma)))
            n_sps.append(len(pks))
            freqs.append(1000.0 / float(np.mean(np.diff(sp_t))) if len(sp_t) > 1 else 0.0)

        ax1.set_xlabel('Time (ms)');  ax1.set_ylabel('V (mV)')
        ax1.set_title(f'Sweep traces  [{param_name}]')

        sm = plt.cm.ScalarMappable(cmap='plasma',
                                    norm=plt.Normalize(sweep_results[0][0],
                                                        sweep_results[-1][0]))
        self.fig_sweep.colorbar(sm, ax=ax1, label=param_name)

        ax2.plot(param_vals, peaks, 'r.-', lw=1.5)
        ax2.set_xlabel(param_name);  ax2.set_ylabel('V_peak (mV)')
        ax2.set_title('Peak voltage vs parameter');  ax2.grid(alpha=0.3)

        ax3.plot(param_vals, freqs, 'g.-', lw=1.5)
        ax3.set_xlabel(param_name);  ax3.set_ylabel('f (Hz)')
        ax3.set_title('Firing rate (f-I curve)');  ax3.grid(alpha=0.3)

        ax4.plot(param_vals, n_sps, 'b.-', lw=1.5)
        ax4.set_xlabel(param_name);  ax4.set_ylabel('N spikes')
        ax4.set_title('Spike count');  ax4.grid(alpha=0.3)

        self.fig_sweep.tight_layout()
        self.cvs_sweep.draw_idle()
        self.setCurrentWidget(self.tab_sweep)

    # ─────────────────────────────────────────────────────────────────
    #  10 — S-D CURVE
    # ─────────────────────────────────────────────────────────────────
    def update_sd_curve(self, sd: dict):
        self._last_sd = sd
        dur  = sd['durations']
        I_th = sd['I_threshold']
        I_rh = sd['rheobase']
        t_ch = sd['chronaxie']
        weiss = sd['weiss_fit']
        Q_th  = sd['Q_threshold']

        ax1, ax2 = self.ax_sd
        ax1.cla()
        ax2.cla()

        ax1.plot(dur, I_th, 'b.-', lw=2, ms=8, label='I_threshold')
        if weiss is not None:
            ax1.plot(dur, weiss, 'r--', lw=1.5, label="Weiss fit")
        ax1.axhline(I_rh, color='gray', ls=':', lw=1.5,
                    label=f'Rheobase = {I_rh:.2f} µA/cm²')
        if not np.isnan(t_ch):
            ax1.axvline(t_ch, color='orange', ls='--', lw=1.5,
                        label=f'Chronaxie = {t_ch:.2f} ms')
            ax1.plot(t_ch, 2 * I_rh, 'go', ms=10, zorder=5)
        ax1.set_xlabel('Pulse duration (ms)');  ax1.set_ylabel('I threshold (µA/cm²)')
        ax1.set_title('Strength-Duration Curve');  ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

        ax2.plot(dur, Q_th, 'm.-', lw=2, ms=8, label='Q = I·t')
        ax2.set_xlabel('Pulse duration (ms)');  ax2.set_ylabel('Charge threshold (nC/cm²)')
        ax2.set_title('Minimum charge vs duration')
        ax2.legend();  ax2.grid(alpha=0.3)

        self.fig_sd.tight_layout()
        self.cvs_sd.draw_idle()
        self.setCurrentWidget(self.tab_sd)

    # ─────────────────────────────────────────────────────────────────
    #  11 — EXCITABILITY MAP
    # ─────────────────────────────────────────────────────────────────
    def update_excmap(self, exc: dict):
        self._last_exc = exc
        I_r  = exc['I_range']
        d_r  = exc['dur_range']
        S    = exc['spike_matrix']
        F    = exc['freq_matrix']

        self.fig_excmap.clear()
        ax1 = self.fig_excmap.add_subplot(1, 2, 1)
        ax2 = self.fig_excmap.add_subplot(1, 2, 2)

        im1 = ax1.pcolormesh(d_r, I_r, S, cmap='Blues', shading='auto')
        self.fig_excmap.colorbar(im1, ax=ax1, label='N spikes')
        ax1.set_xlabel('Duration (ms)');  ax1.set_ylabel('I_ext (µA/cm²)')
        ax1.set_title('Spike count map')

        # Mask zero-frequency cells
        F_masked = np.where(F > 0, F, np.nan)
        im2 = ax2.pcolormesh(d_r, I_r, F_masked, cmap='hot', shading='auto')
        self.fig_excmap.colorbar(im2, ax=ax2, label='f (Hz)')
        ax2.set_xlabel('Duration (ms)');  ax2.set_ylabel('I_ext (µA/cm²)')
        ax2.set_title('Mean frequency map')

        self.fig_excmap.tight_layout()
        self.cvs_excmap.draw_idle()
        self.setCurrentWidget(self.tab_excmap)

    def open_fullscreen(self):
        """Open analytics clone in a maximized window preserving current tab/data."""
        idx = int(self.currentIndex())
        win = QMainWindow(self)
        win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        win.setWindowTitle("NeuroModelPort — Analytics (Full Screen)")
        full = AnalyticsWidget()
        win.setCentralWidget(full)

        if self._last_result is not None:
            full.update_analytics(self._last_result)
        if self._last_bif_data is not None and self._last_bif_param_name is not None:
            full.update_bifurcation(self._last_bif_data, self._last_bif_param_name)
        if self._last_sweep_results is not None and self._last_sweep_param_name is not None:
            full.update_sweep(self._last_sweep_results, self._last_sweep_param_name)
        if self._last_sd is not None:
            full.update_sd_curve(self._last_sd)
        if self._last_exc is not None:
            full.update_excmap(self._last_exc)

        full.setCurrentIndex(max(0, min(idx, full.count() - 1)))
        win.showMaximized()
        self._fullscreen_windows.append(win)

        def _cleanup(*_):
            self._fullscreen_windows = [w for w in self._fullscreen_windows if w is not win]

        win.destroyed.connect(_cleanup)


# ─────────────────────────────────────────────────────────────────────
#  HELPER
# ─────────────────────────────────────────────────────────────────────
def _get_E_rev(name: str, ch) -> float:
    mapping = {
        'Na':   ch.ENa, 'K': ch.EK, 'Leak': ch.EL,
        'Ih':   ch.E_Ih, 'ICa': ch.E_Ca,
        'IA':   ch.E_A,  'SK':  ch.EK,
    }
    return mapping.get(name, 0.0)
