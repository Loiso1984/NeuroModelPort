"""
gui/analytics.py — Full Scientific Analytics Suite v10.1

Analytical tabs using matplotlib embedded in Qt:
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
 12. Spectrogram         — STFT of soma Vm
 13. Impedance Z(f)      — membrane frequency response (|Z|, phase)

Lyapunov computation is explicit via `Compute LLE` action.
"""

import logging
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QTabWidget,
                                QLabel, QTextEdit, QHBoxLayout,
                                QSizePolicy, QScrollArea, QPushButton, QMainWindow, QComboBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Import plot themes for global color consistency
try:
    from .plots import PLOT_THEMES
except ImportError:
    # Fallback if circular import
    PLOT_THEMES = {
        "Default": {
            "soma": "#4080FF",
            "ais": "#FF4040",
            "terminal": (0, 200, 100),
            "threshold": "#F9E2AF",
        }
    }

# ── colour palette (matches plots.py) ──────────────────────────────
CHAN_COLORS = {
    'Na':   '#DC3232', 'K':   '#3264DC', 'Leak': '#32A050',
    'Ih':   '#9632C8', 'ICa': '#FA9600', 'IA':   '#00C8C8',
    'SK':   '#C83296', 'KATP': '#F9E2AF',
}
GATE_COLORS = {
    'm': '#FF4040', 'h': '#4080FF', 'n': '#40C040',
    'r': '#A040FF', 's': '#FF9000', 'u': '#009090',
    'a': '#FF40A0', 'b': '#80C0FF',
}


def _mpl_fig(nrows=1, ncols=1, tight=True, **kwargs) -> tuple:
    """Create a matplotlib Figure + FigureCanvas pair."""
    # Extract figsize from kwargs if provided, otherwise use default
    figsize = kwargs.pop('figsize', (8, 4 * nrows))
    fig = Figure(figsize=figsize, dpi=90, **kwargs)
    if tight:
        fig.set_tight_layout(True)
    canvas = FigureCanvas(fig)
    return fig, canvas


def _ensure_shape_compatible(arr, t, name="array"):
    """Ensure array has compatible shape with time array t.
    
    Returns array with same shape as t, or None if incompatible.
    """
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 1:
        arr = arr.flatten()
    if arr.size != t.size:
        if arr.size > t.size:
            arr = arr[:t.size]
        else:
            arr_padded = np.zeros_like(t)
            arr_padded[:arr.size] = arr
            arr = arr_padded
    if arr.shape != t.shape:
        logging.warning(f"{name} shape {arr.shape} doesn't match t {t.shape}, skipping")
        return None
    return arr


def _tab_with_toolbar(canvas, fullscreen_callback=None, extra_widget=None) -> QWidget:
    """Wrap a canvas in a QWidget with a matplotlib navigation toolbar.

    Args:
        canvas: matplotlib FigureCanvas
        fullscreen_callback: Optional callback for fullscreen button
        extra_widget: Optional widget to add below the canvas (e.g., sliders, checkboxes)
    """
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)

    # Toolbar row with fullscreen button
    toolbar_row = QHBoxLayout()
    toolbar_row.setContentsMargins(0, 0, 0, 0)
    toolbar_row.setSpacing(4)

    toolbar = NavToolbar(canvas, w)
    toolbar_row.addWidget(toolbar)

    if fullscreen_callback is not None:
        btn_fullscreen = QPushButton("⛶ Fullscreen")
        btn_fullscreen.setToolTip("Open plot in fullscreen window with crosshair")
        btn_fullscreen.setMaximumWidth(120)
        btn_fullscreen.clicked.connect(fullscreen_callback)
        toolbar_row.addWidget(btn_fullscreen)

    toolbar_row.addStretch()
    lay.addLayout(toolbar_row)
    lay.addWidget(canvas, 1)
    if extra_widget is not None:
        lay.addWidget(extra_widget, 0)
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


def _downsample_xy(t: np.ndarray, y: np.ndarray, max_points: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Fast stride downsampling for interactive plotting paths."""
    n = int(len(t))
    if n <= max_points or max_points <= 0:
        return t, y
    step = max(1, n // max_points)
    t_ds = t[::step]
    y_ds = y[::step]
    if len(t_ds) == 0 or t_ds[-1] != t[-1]:
        t_ds = np.concatenate((t_ds, np.array([t[-1]])))
        y_ds = np.concatenate((y_ds, np.array([y[-1]])))
    return t_ds, y_ds


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


class _LazyPlaceholder(QWidget):
    """Sentinel widget occupying a tab slot until the user first visits it.
    Detected by isinstance() in _on_tab_changed to trigger lazy construction."""

    def __init__(self, spec_key: int, parent=None):
        super().__init__(parent)
        self._spec_key = spec_key  # Store original spec key for lookup
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl = QLabel("Click this tab to load the view")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(
            "color:#555; font-size:13px; font-style:italic;"
        )
        lay.addWidget(lbl)


class FullscreenPlotViewer(QMainWindow):
    """Fullscreen matplotlib plot viewer with crosshair functionality.
    
    Provides detailed analysis view with:
    - Crosshair cursor showing coordinates
    - Interactive zoom/pan
    - Copy data to clipboard
    """
    
    def __init__(self, fig: Figure, title: str = "Plot Viewer", parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowTitle(f"NeuroModelPort — {title}")
        self.fig = fig
        self.canvas = FigureCanvas(fig)
        self.crosshair_lines = []
        self.crosshair_text = None
        
        self._build_ui()
        self._setup_crosshair()
        
    def _build_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Toolbar
        toolbar = NavToolbar(self.canvas, central)
        layout.addWidget(toolbar)
        
        # Canvas
        layout.addWidget(self.canvas)
        
        self.setCentralWidget(central)
        self.showMaximized()
        
    def _setup_crosshair(self):
        """Setup crosshair cursor on all axes."""
        for ax in self.fig.axes:
            # Create crosshair lines
            v_line = ax.axvline(color='#A6ADC8', linestyle='--', linewidth=1, alpha=0.7)
            h_line = ax.axhline(color='#A6ADC8', linestyle='--', linewidth=1, alpha=0.7)
            self.crosshair_lines.append((ax, v_line, h_line))
        
        # Connect mouse events
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.canvas.mpl_connect('button_press_event', self._on_click)
        
    def _on_mouse_move(self, event):
        """Handle mouse movement for crosshair display."""
        if event.inaxes is None:
            return
        
        ax = event.inaxes
        x, y = event.xdata, event.ydata
        
        # Update crosshair lines for this axis
        for axis, v_line, h_line in self.crosshair_lines:
            if axis == ax:
                v_line.set_xdata([x, x])
                h_line.set_ydata([y, y])
            else:
                # Hide crosshair on other axes
                v_line.set_xdata([])
                h_line.set_ydata([])
        
        # Update title with coordinates
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        title = f"{xlabel}={x:.3f}, {ylabel}={y:.3f}"
        ax.set_title(title, fontsize=10, color='#CBA6F7', pad=5)
        
        self.canvas.draw_idle()
        
    def _on_click(self, event):
        """Handle click to copy coordinates to clipboard."""
        if event.inaxes is None:
            return
        if event.button == 1:  # Left click
            x, y = event.xdata, event.ydata
            from PySide6.QtGui import QClipboard
            from PySide6.QtWidgets import QApplication
            clipboard = QApplication.clipboard()
            clipboard.setText(f"{x:.6f}, {y:.6f}")
            
    def closeEvent(self, event):
        """Cleanup on close."""
        self.canvas.mpl_disconnect('motion_notify_event')
        self.canvas.mpl_disconnect('button_press_event')
        super().closeEvent(event)


# ════════════════════════════════════════════════════════════════════
class AnalyticsWidget(QTabWidget):
    """Main analytics widget — updated by MainWindow after each run."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_result = None
        self._last_stats = None
        self._last_bif_data = None
        self._last_bif_param_name = None
        self._last_sweep_results = None
        self._last_sweep_param_name = None
        self._last_sd = None
        self._last_exc = None
        self._fullscreen_windows = []
        self._building_lazy_tab = False  # re-entrancy guard for _on_tab_changed
        self._active_category = 'All'  # Current category filter
        self._category_mapping = {  # Map tab indices to categories
            1: 'Single', 2: 'Single', 3: 'Single', 4: 'Single', 16: 'Single',
            5: 'Spectral', 6: 'Spectral', 7: 'Spectral',
            8: 'Sweep', 9: 'Sweep', 10: 'Sweep', 11: 'Sweep',
            12: 'Physics', 13: 'Physics', 14: 'Physics', 15: 'Physics', 17: 'Physics',
        }
        self._all_tab_specs = {}  # Store all tab specs for rebuilding
        self._tab_figures = {}  # Store figures for fullscreen access
        self._time_marker = None  # Store vertical line marker for linked cursor
        self._build_tabs()
    
    def highlight_time(self, t_ms: float):
        """Show a vertical marker on analytics plots at the specified time.
        
        This enables cross-tab cursor synchronization with the Oscilloscope.
        The marker is shown on time-domain plots (Single category).
        Handles Currents, Gates, and Spike Shape tabs.
        """
        # Remove existing marker if any
        if self._time_marker is not None:
            try:
                self._time_marker.remove()
            except:
                pass
            self._time_marker = None
        
        current_idx = self.currentIndex()
        if current_idx not in self._category_mapping:
            return
        
        category = self._category_mapping[current_idx]
        tab_name = self.tabText(current_idx)
        
        # Handle Spike Shape tab differently (highlight closest spike)
        if "Spike Shape" in tab_name:
            self._highlight_spike_shape(t_ms)
            return
        
        # Only show marker on time-domain plots (Single category)
        if category != 'Single':
            return
        
        if tab_name not in self._tab_figures:
            return
        
        fig = self._tab_figures[tab_name]
        if fig is None:
            return
        
        # Add vertical line marker at the specified time
        for ax in fig.axes:
            # Only add to axes with x-axis representing time (check label or data)
            xlabel = ax.get_xlabel().lower()
            if 'time' in xlabel or 't (ms)' in xlabel or 't_ms' in xlabel:
                self._time_marker = ax.axvline(x=t_ms, color='#89B4FA', linestyle='--', linewidth=1.5, alpha=0.8)
                break
        
        # Force redraw
        fig.canvas.draw()
    
    def _highlight_spike_shape(self, t_ms: float):
        """Highlight the spike closest to t_ms in Spike Shape tab."""
        if not hasattr(self, '_last_result') or self._last_result is None:
            return
        
        result = self._last_result
        t = result.t
        v = result.v_soma
        
        # Find spikes
        from core.analysis import detect_spikes
        peak_idx, spike_times, spike_amps = detect_spikes(v, t, threshold=-20.0)
        
        if len(spike_times) == 0:
            return
        
        # Find spike closest to t_ms
        closest_idx = np.argmin(np.abs(spike_times - t_ms))
        closest_spike_time = spike_times[closest_idx]
        
        # Highlight this spike in the plot
        tab_name = self.tabText(self.currentIndex())
        if tab_name not in self._tab_figures:
            return
        
        fig = self._tab_figures[tab_name]
        if fig is None:
            return
        
        # Add vertical marker at closest spike time
        for ax in fig.axes:
            xlabel = ax.get_xlabel().lower()
            if 'time' in xlabel or 't (ms)' in xlabel:
                self._time_marker = ax.axvline(x=closest_spike_time, color='#F9E2AF', linestyle='--', linewidth=2.0, alpha=0.9)
                break
        
        fig.canvas.draw()

    # ─────────────────────────────────────────────────────────────────
    #  TAB CONSTRUCTION
    # ─────────────────────────────────────────────────────────────────
    def _open_fullscreen_plot(self, tab_name: str):
        """Open a fullscreen viewer for the specified tab's plot."""
        if tab_name not in self._tab_figures:
            return

        fig = self._tab_figures[tab_name]
        if fig is None:
            return

        # Copy figure to avoid shared reference conflicts
        import pickle
        import io
        try:
            buf = io.BytesIO()
            pickle.dump(fig, buf)
            buf.seek(0)
            fig_copy = pickle.load(buf)
        except Exception:
            # Fallback: if pickle fails, use original (may have conflicts)
            fig_copy = fig

        viewer = FullscreenPlotViewer(fig_copy, title=f"Analytics — {tab_name}", parent=self)
        self._fullscreen_windows.append(viewer)
        
        def _cleanup(*_):
            self._fullscreen_windows = [w for w in self._fullscreen_windows if w is not viewer]
        
        viewer.destroyed.connect(_cleanup)
    
    def _build_tabs(self):
        # Corner widget with category filter buttons and actions
        corner = QWidget()
        corner_l = QHBoxLayout(corner)
        corner_l.setContentsMargins(0, 0, 0, 0)
        corner_l.setSpacing(4)

        # Category filter dropdown
        lbl_cat = QLabel("Filter:")
        lbl_cat.setStyleSheet("color:#CDD6F4; font-size:11px;")
        corner_l.addWidget(lbl_cat)

        self._combo_category = QComboBox()
        self._combo_category.addItems(['All', 'Single', 'Spectral', 'Sweep', 'Physics'])
        self._combo_category.setCurrentText('All')
        self._combo_category.setStyleSheet("""
            QComboBox {
                background:#313244; color:#CDD6F4; border:1px solid #45475A;
                padding:2px; font-size:11px; min-width:80px;
            }
        """)
        self._combo_category.currentTextChanged.connect(self._filter_category)
        corner_l.addWidget(self._combo_category)

        corner_l.addSpacing(10)

        self._btn_compute_lle = QPushButton("Compute LLE")
        self._btn_compute_lle.setToolTip("Run Lyapunov/LLE analysis explicitly for the latest simulation")
        self._btn_compute_lle.setEnabled(False)
        self._btn_compute_lle.clicked.connect(self._compute_lle_now)
        corner_l.addWidget(self._btn_compute_lle)

        self._btn_fullscreen = QPushButton("Full Screen")
        self._btn_fullscreen.setToolTip("Open analytics in a maximized window")
        self._btn_fullscreen.clicked.connect(self.open_fullscreen)
        corner_l.addWidget(self._btn_fullscreen)

        self.setCornerWidget(corner, Qt.Corner.TopRightCorner)

        # ── Tab 0: Passport — always built (pure text, zero MPL cost) ──
        self.passport_view = QTextEdit()
        self.passport_view.setReadOnly(True)
        self.passport_view.setFont(QFont("Consolas", 10))
        self.passport_view.setMinimumWidth(400)
        self.passport_view.setStyleSheet(
            "background:#0D1117; color:#C9D1D9; border:none;"
        )
        self.addTab(self.passport_view, "Passport")

        # ── Tabs 1–15: lazy placeholders — MPL canvas built on first visit ──
        # Each entry: builder (creates attrs + returns QWidget), updater (may be None),
        # needs_stats (updater takes (result, stats)), needs_morph (skip if no morph).
        self._all_tab_specs: dict[int, dict] = {
            1:  {'builder': '_build_tab_spike_mech', 'updater': '_update_spike_mechanism', 'title': '🧪 Spike Mechanism', 'needs_stats': True},
            2:  {'builder': '_build_tab_phase',      'updater': '_update_phase',           'title': '🔄 Phase Plane', 'needs_stats': True},
            3:  {'builder': '_build_tab_chaos',      'updater': '_update_chaos',           'title': '🦋 Chaos & LLE', 'needs_stats': True},
            4:  {'builder': '_build_tab_kymo',       'updater': '_update_kymo',            'title': '🌊 Kymograph'},
            5:  {'builder': '_build_tab_spectro',    'updater': '_update_spectrogram',     'title': '🌈 Spectrogram'},
            6:  {'builder': '_build_tab_impedance',  'updater': '_update_impedance',       'title': '🧲 Impedance'},
            7:  {'builder': '_build_tab_modulation', 'updater': '_update_modulation',      'title': '🎡 Phase-Locking', 'needs_stats': True},
            8:  {'builder': '_build_tab_sweep',      'updater': None,                      'title': '↔ f-I Curve'},
            9:  {'builder': '_build_tab_sd',         'updater': None,                      'title': '⏱ S-D Curve'},
            10: {'builder': '_build_tab_excmap',     'updater': None,                      'title': '🗺 Excit. Map'},
            11: {'builder': '_build_tab_bif',        'updater': None,                      'title': '🔀 Bifurcation'},
            12: {'builder': '_build_tab_currents',   'updater': '_update_currents',        'title': '⚡ Currents'},
            13: {'builder': '_build_tab_gates',      'updater': '_update_gates',           'title': '⚙ Gates'},
            14: {'builder': '_build_tab_energy_balance', 'updater': '_update_energy_balance', 'title': '🔋 Energy & Balance', 'needs_morph': True},
            15: {'builder': '_build_tab_spike_shape', 'updater': '_update_spike_shape', 'title': '📊 Spike Shape', 'needs_stats': True},
            16: {'builder': '_build_tab_poincare',    'updater': '_update_poincare',       'title': '🔵 Poincaré (ISI)', 'needs_stats': True},
        }
        self._tab_specs = self._all_tab_specs.copy()  # Current visible tabs
        for idx, spec in self._tab_specs.items():
            ph = _LazyPlaceholder(idx)
            self.addTab(ph, spec['title'])

        self.currentChanged.connect(self._on_tab_changed)

    def _filter_category(self, category: str):
        """Filter visible tabs based on selected category using removeTab/insertTab."""
        self._active_category = category

        # Save current index
        current_idx = self.currentIndex()
        current_widget = self.widget(current_idx) if current_idx >= 0 else None

        # Remove all tabs except Passport (index 0)
        while self.count() > 1:
            self.removeTab(1)

        # Rebuild tabs based on category and update _tab_specs
        self._tab_specs = {}
        for idx, spec in self._all_tab_specs.items():
            tab_cat = self._category_mapping.get(idx)
            if category == 'All' or tab_cat == category:
                ph = _LazyPlaceholder(idx)
                self.addTab(ph, spec['title'])
                self._tab_specs[idx] = spec

        # Restore current widget if it still exists
        if current_widget is not None and current_widget != self.passport_view:
            for i in range(self.count()):
                if self.widget(i) == current_widget:
                    self.setCurrentIndex(i)
                    break

    # ─────────────────────────────────────────────────────────────────
    #  LAZY TAB ACTIVATION
    # ─────────────────────────────────────────────────────────────────
    def _show_missing_data_message(self, tab_title: str, missing_data: str):
        """Show error message on tab when required data is missing."""
        # Map tab title to figure name
        tab_to_fig = {
            '🧪 Spike Mechanism': 'Spike Mechanism',
            '🔋 Energy & Balance': 'Energy & Balance',
        }
        fig_name = tab_to_fig.get(tab_title)
        if fig_name and fig_name in self._tab_figures:
            fig = self._tab_figures[fig_name]
            if fig:
                # Add text annotation to the figure
                for ax in fig.axes:
                    ax.clear()
                    ax.text(0.5, 0.5, f"⚠ {missing_data} required", 
                            transform=ax.transAxes, ha='center', va='center',
                            fontsize=14, fontweight='bold', color='#dc3545',
                            bbox=dict(boxstyle='round', facecolor='#f8d7da', alpha=0.8))
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.axis('off')
                fig.canvas.draw_idle()
    
    def _show_updater_error_message(self, tab_title: str, error: str):
        """Show error message on tab when updater fails."""
        tab_to_fig = {
            '🧪 Spike Mechanism': 'Spike Mechanism',
            '🔋 Energy & Balance': 'Energy & Balance',
        }
        fig_name = tab_to_fig.get(tab_title)
        if fig_name and fig_name in self._tab_figures:
            fig = self._tab_figures[fig_name]
            if fig:
                # Add text annotation to the figure
                for ax in fig.axes:
                    ax.clear()
                    ax.text(0.5, 0.5, f"⚠ Update failed:\n{error}", 
                            transform=ax.transAxes, ha='center', va='center',
                            fontsize=12, fontweight='bold', color='#dc3545',
                            bbox=dict(boxstyle='round', facecolor='#f8d7da', alpha=0.8))
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.axis('off')
                fig.canvas.draw_idle()
    
    def _on_tab_changed(self, index: int):
        """Build an MPL canvas the first time a lazy tab is visited."""
        if self._building_lazy_tab:
            return  # re-entrancy guard: removeTab/insertTab fire currentChanged too
        widget = self.widget(index)
        if widget is None:
            return
        if not isinstance(widget, _LazyPlaceholder):
            return
        spec_key = widget._spec_key
        spec = self._tab_specs.get(spec_key)
        if spec is None:
            return

        # Build the real canvas, swap out the placeholder, restore focus
        self._building_lazy_tab = True
        title = spec['title']
        try:
            new_widget = getattr(self, spec['builder'])()
            self.removeTab(index)
            self.insertTab(index, new_widget, title)
            self.setCurrentIndex(index)   # restore — removeTab may shift focus
            
            # Force a geometric refresh and draw
            new_widget.show()
            
            # Helper to recursively find FigureCanvas inside nested layouts (like QScrollArea)
            def _force_draw(w):
                if hasattr(w, 'draw_idle') and callable(w.draw_idle):
                    w.draw_idle()
                for child in w.children():
                    if hasattr(child, 'draw_idle') and callable(child.draw_idle):
                        child.draw_idle()
                    elif isinstance(child, QWidget):
                        _force_draw(child)
                        
            _force_draw(new_widget)
        except Exception as e:
            # Show error in a simple label widget
            from PySide6.QtWidgets import QLabel
            error_widget = QLabel(f"⚠ Tab build failed:\n{str(e)}")
            error_widget.setStyleSheet("QLabel { color: #dc3545; padding: 20px; }")
            error_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.removeTab(index)
            self.insertTab(index, error_widget, title)
            self.setCurrentIndex(index)
            import traceback
            traceback.print_exc()
            widget.deleteLater()
            self._building_lazy_tab = False
            return
        finally:
            self._building_lazy_tab = False
        widget.deleteLater()

        # Immediately populate with current data if available
        if self._last_result is None:
            return
        updater_name = spec.get('updater')
        if updater_name is None:
            return
        
        # Check for missing required data and show error message
        if spec.get('needs_morph') and not self._last_result.morph:
            self._show_missing_data_message(spec['title'], 'multi-compartment morphology (single-compartment simulation)')
            return
        if spec.get('needs_stats') and self._last_stats is None:
            self._show_missing_data_message(spec['title'], 'spike statistics (run simulation with spike detection)')
            return
        
        updater = getattr(self, updater_name)
        try:
            if spec.get('needs_stats'):
                updater(self._last_result, self._last_stats)
            else:
                updater(self._last_result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            # Show error on the figure if possible
            self._show_updater_error_message(spec['title'], str(e))

    def highlight_time(self, t_ms: float):
        """Highlight a specific time point across analytics tabs (linked cursor).
        
        Draws a yellow dot on the Phase Plane trajectory at the specified time.
        """
        # Check if Phase Plane tab has been built
        if not hasattr(self, 'fig_phase') or self.fig_phase is None:
            return
        
        # Check if we have simulation data
        if self._last_result is None:
            return
        
        # Find the index closest to the requested time
        t = self._last_result.t
        idx = np.argmin(np.abs(t - t_ms))
        
        # Get the phase plane data (V vs n for soma)
        # V is already in result.v_soma, need to extract n gate
        n_comp = self._last_result.n_comp
        if n_comp == 0:
            return
            
        # Extract n gate from state vector
        # State layout: V(n_comp), m(n_comp), h(n_comp), n_K(n_comp), ...
        off_n = 3 * n_comp
        y = self._last_result.y
        v_soma = y[0, :]
        n_soma = y[off_n, :]
        
        # Remove existing highlight dot if any
        if hasattr(self, '_phase_highlight_dot'):
            try:
                self._phase_highlight_dot.remove()
            except:
                pass
        
        # Add yellow dot at the specified time point
        self._phase_highlight_dot = self.ax_phase.plot(
            v_soma[idx], n_soma[idx], 
            'o', color='#F9E2AF', markersize=10, 
            markeredgecolor='black', markeredgewidth=1.5,
            zorder=10, label=f't={t_ms:.1f}ms'
        )
        
        # Redraw the canvas
        self.fig_phase.canvas.draw_idle()

    # ─────────────────────────────────────────────────────────────────
    #  PER-TAB BUILDER METHODS  (called once on first visit)
    # ─────────────────────────────────────────────────────────────────
    def _build_tab_gates(self) -> QWidget:
        # Refactored: Single large plot with checkboxes for toggling individual gates
        self.fig_gates, cvs = _mpl_fig(1, 1)
        self.ax_gates = self.fig_gates.add_subplot(1, 1, 1)
        self.fig_gates.set_tight_layout({'pad': 2.5})
        self.cvs_gates = cvs
        self._tab_figures['Gate Dynamics'] = self.fig_gates

        # Checkbox container
        from PySide6.QtWidgets import QCheckBox, QScrollArea, QWidget, QVBoxLayout
        self._gates_checkboxes: dict[str, QCheckBox] = {}
        self._gates_visibility: dict[str, bool] = {}

        # Initialize plot artists
        self._gates_line_v = None
        self._gates_lines: dict[str, object] = {}
        self._gates_signature: tuple[str, ...] = ()

        # Create checkbox container
        checkbox_widget = QWidget()
        checkbox_layout = QVBoxLayout(checkbox_widget)
        checkbox_layout.setContentsMargins(0, 0, 0, 0)
        checkbox_layout.setSpacing(2)

        # Scroll area for checkboxes
        scroll_area = QScrollArea()
        scroll_area.setWidget(checkbox_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(200)  # Increased from 150 to reduce clutter
        scroll_area.setStyleSheet("QScrollArea { border: none; }")

        # Main layout: plot on top, checkboxes below
        from PySide6.QtWidgets import QVBoxLayout
        main_layout = QVBoxLayout()
        main_layout.addWidget(cvs, 1)
        main_layout.addWidget(scroll_area, 0)

        main_widget = QWidget()
        main_widget.setLayout(main_layout)

        # Store references
        self._gates_checkbox_container = checkbox_widget
        self._gates_checkbox_layout = checkbox_layout

        return _tab_with_toolbar(cvs, fullscreen_callback=lambda: self._open_fullscreen_plot('Gate Dynamics'), extra_widget=scroll_area)

    def _build_tab_spike_mech(self) -> QWidget:
        # Create wider and larger figure for better space utilization and to prevent label overlapping
        self.fig_spike_mech, cvs = _mpl_fig(3, 1, figsize=(16, 18))
        self.ax_spike_mech = [self.fig_spike_mech.add_subplot(3, 1, k) for k in range(1, 4)]
        # Enable navigation for all subplots
        for ax in self.ax_spike_mech:
            ax.set_navigate(True)
        self.fig_spike_mech.set_tight_layout({'pad': 4.0, 'h_pad': 1.0, 'w_pad': 1.0})
        self.cvs_spike_mech = cvs
        self._tab_figures['Spike Mechanism'] = self.fig_spike_mech
        self._spike_mech_init_done = False
        self._spike_mech_lines: dict[str, object] = {}
        self._spike_mech_curr_lines: dict[str, object] = {}
        self._spike_mech_norm_lines: dict[str, object] = {}
        self._spike_mech_texts: dict[str, object] = {}
        self._spike_mech_ax2b = None
        self._spike_mech_ax3 = None  # For 3rd panel (threshold contributions)
        self._spike_mech_bar_artists: dict[str, object] = {}  # For stacked bar chart

        # Add spike zoomer spinbox and vertical scroll control
        from PySide6.QtWidgets import QSpinBox, QHBoxLayout, QLabel, QWidget, QScrollArea, QVBoxLayout
        control_widget = QWidget()
        control_layout = QHBoxLayout(control_widget)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(5)

        # Spike zoomer
        zoom_label = QLabel("Zoom to Spike #:")
        zoom_label.setStyleSheet("color:#CDD6F4; font-size:11px;")
        control_layout.addWidget(zoom_label)

        self._spike_zoomer = QSpinBox()
        self._spike_zoomer.setRange(1, 9999)
        self._spike_zoomer.setValue(1)
        self._spike_zoomer.setStyleSheet("""
            QSpinBox {
                background:#313244; color:#CDD6F4; border:1px solid #45475A;
                padding:2px; font-size:11px; min-width:60px;
            }
        """)
        self._spike_zoomer.valueChanged.connect(self._on_spike_zoomer_changed)
        control_layout.addWidget(self._spike_zoomer)

        control_layout.addStretch()

        # Store control widget reference
        self._spike_zoomer_widget = control_widget
        # Disconnect signal when widget is destroyed to prevent C++ object deletion error
        # Use weakref to avoid reference cycles
        import weakref
        self_ref = weakref.ref(self)
        def _cleanup_zoomer():
            try:
                obj = self_ref()
                if obj is not None and hasattr(obj, '_spike_zoomer'):
                    try:
                        obj._spike_zoomer.valueChanged.disconnect(obj._on_spike_zoomer_changed)
                    except (RuntimeError, TypeError, SystemError):
                        pass  # Widget already destroyed or signal not connected
            except Exception:
                pass  # Ignore any errors during cleanup
        control_widget.destroyed.connect(_cleanup_zoomer)

        # Store spike data for zoomer
        self._spike_zoomer_data = None  # Will store (t, v, spike_times)

        return _tab_with_toolbar(cvs, fullscreen_callback=lambda: self._open_fullscreen_plot('Spike Mechanism'), extra_widget=control_widget)

    def _build_tab_currents(self) -> QWidget:
        # Refactored: Single large plot with checkboxes for toggling individual currents
        self.fig_currents, cvs = _mpl_fig(1, 1)
        self.ax_currents = self.fig_currents.add_subplot(1, 1, 1)
        self.fig_currents.set_tight_layout({'pad': 2.5})
        self.cvs_currents = cvs
        self._tab_figures['Currents'] = self.fig_currents

        # Checkbox container
        from PySide6.QtWidgets import QCheckBox, QScrollArea, QWidget, QVBoxLayout
        self._currents_checkboxes: dict[str, QCheckBox] = {}
        self._currents_visibility: dict[str, bool] = {}

        # Initialize plot artists
        self._currents_line_v = None
        self._currents_lines: dict[str, object] = {}
        self._currents_signature = ()

        # Create checkbox container
        checkbox_widget = QWidget()
        checkbox_layout = QVBoxLayout(checkbox_widget)
        checkbox_layout.setContentsMargins(0, 0, 0, 0)
        checkbox_layout.setSpacing(2)

        # Scroll area for checkboxes
        scroll_area = QScrollArea()
        scroll_area.setWidget(checkbox_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(200)  # Increased from 150 to reduce clutter
        scroll_area.setStyleSheet("QScrollArea { border: none; }")

        # Main layout: plot on top, checkboxes below
        from PySide6.QtWidgets import QVBoxLayout
        main_layout = QVBoxLayout()
        main_layout.addWidget(cvs, 1)
        main_layout.addWidget(scroll_area, 0)

        main_widget = QWidget()
        main_widget.setLayout(main_layout)

        # Store references
        self._currents_checkbox_container = checkbox_widget
        self._currents_checkbox_layout = checkbox_layout

        return _tab_with_toolbar(cvs, fullscreen_callback=lambda: self._open_fullscreen_plot('Currents'), extra_widget=scroll_area)

    def _build_tab_phase(self) -> QWidget:
        self.fig_phase, cvs = _mpl_fig(1, 1)
        self.ax_phase = self.fig_phase.add_subplot(1, 1, 1)
        self.cvs_phase = cvs
        self._tab_figures['Phase Plane'] = self.fig_phase
        self._phase_lines: dict[str, object] = {}
        self._phase_warning_text = None

        # Add time slider for trajectory evolution
        from PySide6.QtWidgets import QSlider, QHBoxLayout, QLabel, QWidget
        slider_widget = QWidget()
        slider_layout = QHBoxLayout(slider_widget)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.setSpacing(5)

        slider_label = QLabel("Time Window (ms):")
        slider_label.setStyleSheet("color:#CDD6F4; font-size:11px;")
        slider_layout.addWidget(slider_label)

        self._phase_time_slider = QSlider(Qt.Orientation.Horizontal)
        self._phase_time_slider.setRange(0, 100)
        self._phase_time_slider.setValue(100)
        self._phase_time_slider.setStyleSheet("""
            QSlider {
                background:#313244; border:1px solid #45475A;
                padding:2px; height:20px;
            }
            QSlider::handle:horizontal {
                background:#89B4FA; width:16px; margin:-4px 0; border-radius:8px;
            }
        """)
        self._phase_time_slider.valueChanged.connect(self._on_phase_time_slider_changed)
        slider_layout.addWidget(self._phase_time_slider, 1)

        self._phase_time_label = QLabel("All")
        self._phase_time_label.setStyleSheet("color:#CBA6F7; font-size:11px;")
        self._phase_time_label.setFixedWidth(50)
        slider_layout.addWidget(self._phase_time_label)

        # Store slider widget reference
        self._phase_slider_widget = slider_widget
        # Disconnect signal when widget is destroyed to prevent C++ object deletion error
        # Use weakref to avoid reference cycles
        import weakref
        self_ref = weakref.ref(self)
        def _cleanup_slider():
            try:
                obj = self_ref()
                if obj is not None and hasattr(obj, '_phase_time_slider'):
                    try:
                        obj._phase_time_slider.valueChanged.disconnect(obj._on_phase_time_slider_changed)
                    except (RuntimeError, TypeError, SystemError):
                        pass  # Widget already destroyed or signal not connected
            except Exception:
                pass  # Ignore any errors during cleanup
        slider_widget.destroyed.connect(_cleanup_slider)

        # Store full trajectory data for slider
        self._phase_full_data = None  # Will store (t, V, n_t)

        return _tab_with_toolbar(cvs, fullscreen_callback=lambda: self._open_fullscreen_plot('Phase Plane'), extra_widget=slider_widget)

    def _build_tab_kymo(self) -> QWidget:
        self.fig_kymo, cvs = _mpl_fig(2, 1)
        self._kymo_axes = [self.fig_kymo.add_subplot(2, 1, k) for k in range(1, 3)]
        self.fig_kymo.set_tight_layout({'pad': 2.5})
        self._kymo_cbar1 = None
        self._kymo_cbar2 = None
        self._kymo_im1 = None
        self._kymo_im2 = None
        self._kymo_empty_text = None
        self.cvs_kymo = cvs
        self._tab_figures['Kymograph'] = self.fig_kymo
        return _tab_with_toolbar(cvs, fullscreen_callback=lambda: self._open_fullscreen_plot('Kymograph'))

    def _build_tab_energy_balance(self) -> QWidget:
        import matplotlib.gridspec as gridspec
        # Create larger figure for taller/wider graphs
        self.fig_energy, cvs = _mpl_fig(1, 1, figsize=(12, 12))
        # v11.19: Updated layout - 4 time-series plots on left, pie chart in top-right corner
        gs = gridspec.GridSpec(4, 2, width_ratios=[8, 2], height_ratios=[1, 1, 1, 1],
                               figure=self.fig_energy, hspace=0.35, wspace=0.25)
        # Time-series plots in left column (all 4 rows)
        self.ax_energy = [
            self.fig_energy.add_subplot(gs[0, 0]),  # Balance Error
            self.fig_energy.add_subplot(gs[1, 0]),  # Cumulative Charge
            self.fig_energy.add_subplot(gs[2, 0]),  # Power
            self.fig_energy.add_subplot(gs[3, 0]),  # ATP Pool (new)
            self.fig_energy.add_subplot(gs[0, 1]),  # Pie chart in top-right corner
        ]
        self.cvs_energy = cvs
        self._tab_figures['Energy & Balance'] = self.fig_energy
        self._energy_lines: dict[str, object] = {}
        self._balance_lines: dict[str, object] = {}
        self._atp_line = None
        self._atp_threshold_line = None
        self._pie_chart = None
        return _tab_with_toolbar(cvs, fullscreen_callback=lambda: self._open_fullscreen_plot('Energy & Balance'))

    def _build_tab_spike_shape(self) -> QWidget:
        self.fig_spike_shape, cvs = _mpl_fig(1, 1)
        self.ax_spike_shape = self.fig_spike_shape.add_subplot(1, 1, 1)
        self.fig_spike_shape.set_tight_layout({'pad': 2.5})
        self.cvs_spike_shape = cvs
        self._tab_figures['Spike Shape'] = self.fig_spike_shape
        self._spike_shape_init_done = False
        self._spike_shape_lines: dict[str, object] = {}

        # Add spike selection controls for handling hundreds of spikes
        from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QSpinBox, QComboBox, QCheckBox
        selection_widget = QWidget()
        selection_layout = QHBoxLayout(selection_widget)
        selection_layout.setContentsMargins(0, 0, 0, 0)
        selection_layout.setSpacing(5)

        # Spike range selection
        range_label = QLabel("Show spikes:")
        range_label.setStyleSheet("color:#CDD6F4; font-size:11px;")
        selection_layout.addWidget(range_label)

        self._spike_shape_start = QSpinBox()
        self._spike_shape_start.setRange(1, 9999)
        self._spike_shape_start.setValue(1)
        self._spike_shape_start.setMinimumWidth(60)
        self._spike_shape_start.setStyleSheet("""
            QSpinBox {
                background:#313244; color:#CDD6F4; border:1px solid #45475A;
                padding:2px; font-size:11px;
            }
        """)
        selection_layout.addWidget(self._spike_shape_start)

        to_label = QLabel("to")
        to_label.setStyleSheet("color:#CDD6F4; font-size:11px;")
        selection_layout.addWidget(to_label)

        self._spike_shape_end = QSpinBox()
        self._spike_shape_end.setRange(1, 9999)
        self._spike_shape_end.setValue(50)
        self._spike_shape_end.setMinimumWidth(60)
        self._spike_shape_end.setStyleSheet("""
            QSpinBox {
                background:#313244; color:#CDD6F4; border:1px solid #45475A;
                padding:2px; font-size:11px;
            }
        """)
        selection_layout.addWidget(self._spike_shape_end)

        of_label = QLabel("of")
        of_label.setStyleSheet("color:#CDD6F4; font-size:11px;")
        selection_layout.addWidget(of_label)

        self._spike_shape_total = QLabel("0")
        self._spike_shape_total.setStyleSheet("color:#CBA6F7; font-size:11px;")
        self._spike_shape_total.setFixedWidth(40)
        selection_layout.addWidget(self._spike_shape_total)

        # Quick selection dropdown
        quick_label = QLabel("Quick:")
        quick_label.setStyleSheet("color:#CDD6F4; font-size:11px;")
        selection_layout.addWidget(quick_label)

        self._spike_shape_quick = QComboBox()
        self._spike_shape_quick.addItems(["Custom", "First 10", "First 50", "Last 10", "Last 50", "Every 10th", "Every 5th"])
        self._spike_shape_quick.setStyleSheet("""
            QComboBox {
                background:#313244; color:#CDD6F4; border:1px solid #45475A;
                padding:2px; font-size:11px; min-width:100px;
            }
        """)
        self._spike_shape_quick.currentTextChanged.connect(self._on_spike_shape_quick_changed)
        selection_layout.addWidget(self._spike_shape_quick)

        # Color coding option
        self._spike_shape_color_by_index = QCheckBox("Color by index")
        self._spike_shape_color_by_index.setChecked(True)
        self._spike_shape_color_by_index.setStyleSheet("color:#CDD6F4; font-size:11px;")
        selection_layout.addWidget(self._spike_shape_color_by_index)

        selection_layout.addStretch()

        # Connect signals
        self._spike_shape_start.valueChanged.connect(self._on_spike_shape_range_changed)
        self._spike_shape_end.valueChanged.connect(self._on_spike_shape_range_changed)
        self._spike_shape_color_by_index.stateChanged.connect(self._on_spike_shape_options_changed)

        # Store references
        self._spike_shape_selection_widget = selection_widget
        self._spike_shape_data = None  # Will store (t, v, spike_times, peak_idx)

        return _tab_with_toolbar(cvs, fullscreen_callback=lambda: self._open_fullscreen_plot('Spike Shape'), extra_widget=selection_widget)

    def _on_spike_shape_quick_changed(self, text: str):
        """Handle quick selection dropdown change."""
        if self._spike_shape_data is None:
            return
        n_sp = len(self._spike_shape_data[2])  # spike_times

        if text == "Custom":
            return  # Don't modify user's custom selection
        elif text == "First 10":
            self._spike_shape_start.setValue(1)
            self._spike_shape_end.setValue(min(10, n_sp))
        elif text == "First 50":
            self._spike_shape_start.setValue(1)
            self._spike_shape_end.setValue(min(50, n_sp))
        elif text == "Last 10":
            self._spike_shape_start.setValue(max(1, n_sp - 9))
            self._spike_shape_end.setValue(n_sp)
        elif text == "Last 50":
            self._spike_shape_start.setValue(max(1, n_sp - 49))
            self._spike_shape_end.setValue(n_sp)
        elif text == "Every 10th":
            self._spike_shape_start.setValue(1)
            self._spike_shape_end.setValue(n_sp)
            # Store the step for later use
            self._spike_shape_step = 10
        elif text == "Every 5th":
            self._spike_shape_start.setValue(1)
            self._spike_shape_end.setValue(n_sp)
            self._spike_shape_step = 5

        # Reset to Custom after applying
        self._spike_shape_quick.blockSignals(True)
        self._spike_shape_quick.setCurrentText("Custom")
        self._spike_shape_quick.blockSignals(False)

        if hasattr(self, '_last_result') and self._last_result is not None:
            self._update_spike_shape(self._last_result, self._last_stats)

    def _on_spike_shape_range_changed(self):
        """Handle range spinbox change."""
        if hasattr(self, '_last_result') and self._last_result is not None:
            self._update_spike_shape(self._last_result, self._last_stats)

    def _on_spike_shape_options_changed(self):
        """Handle checkbox option change."""
        if hasattr(self, '_last_result') and self._last_result is not None:
            self._update_spike_shape(self._last_result, self._last_stats)

    def _build_tab_poincare(self) -> QWidget:
        self.fig_poincare, cvs = _mpl_fig(1, 1)
        self.ax_poincare = self.fig_poincare.add_subplot(1, 1, 1)
        self.fig_poincare.set_tight_layout({'pad': 2.5})
        self.cvs_poincare = cvs
        self._tab_figures['Poincaré (ISI)'] = self.fig_poincare
        self._poincare_init_done = False
        self._poincare_lines: dict[str, object] = {}
        return _tab_with_toolbar(cvs, fullscreen_callback=lambda: self._open_fullscreen_plot('Poincaré (ISI)'))

    def _build_tab_bif(self) -> QWidget:
        self.fig_bif, cvs = _mpl_fig(2, 2)
        self.ax_bif = [self.fig_bif.add_subplot(2, 2, k) for k in range(1, 5)]
        self.tab_bif = _tab_with_toolbar(cvs, fullscreen_callback=lambda: self._open_fullscreen_plot('Bifurcation'))
        self.cvs_bif = cvs
        self._tab_figures['Bifurcation'] = self.fig_bif
        self._bif_lines: dict[str, object] = {}
        self._bif_peak_scatter = None
        return self.tab_bif

    def _build_tab_sweep(self) -> QWidget:
        self.fig_sweep, cvs = _mpl_fig(2, 2)
        self.ax_sweep = [self.fig_sweep.add_subplot(2, 2, k) for k in range(1, 5)]
        self.tab_sweep = _tab_with_toolbar(cvs, fullscreen_callback=lambda: self._open_fullscreen_plot('Sweep'))
        self.cvs_sweep = cvs
        self._tab_figures['Sweep'] = self.fig_sweep
        self._sweep_cbar = None
        self._sweep_trace_lines: list = []
        self._sweep_trace_max = 64
        self._sweep_metric_lines: dict[str, object] = {}
        return self.tab_sweep

    def _build_tab_sd(self) -> QWidget:
        self.fig_sd, cvs = _mpl_fig(1, 2)
        self.ax_sd = [self.fig_sd.add_subplot(1, 2, k) for k in range(1, 3)]
        self.tab_sd = _tab_with_toolbar(cvs, fullscreen_callback=lambda: self._open_fullscreen_plot('S-D Curve'))
        self.cvs_sd = cvs
        self._tab_figures['S-D Curve'] = self.fig_sd
        self._sd_lines: dict[str, object] = {}
        return self.tab_sd

    def _build_tab_excmap(self) -> QWidget:
        self.fig_excmap, cvs = _mpl_fig(1, 2)
        self.ax_excmap = [self.fig_excmap.add_subplot(1, 2, k) for k in range(1, 3)]
        self.tab_excmap = _tab_with_toolbar(cvs, fullscreen_callback=lambda: self._open_fullscreen_plot('Excitability Map'))
        self.cvs_excmap = cvs
        self._tab_figures['Excitability Map'] = self.fig_excmap
        self._excmap_mesh = {"spikes": None, "freq": None}
        self._excmap_cbar = {"spikes": None, "freq": None}
        return self.tab_excmap

    def _build_tab_spectro(self) -> QWidget:
        self.fig_spectro, cvs = _mpl_fig(2, 1)
        self.ax_spectro = [self.fig_spectro.add_subplot(2, 1, k) for k in range(1, 3)]
        self.fig_spectro.set_tight_layout({'pad': 2.5})
        self.cvs_spectro = cvs
        self._tab_figures['Spectrogram'] = self.fig_spectro
        self._spectro_cbar = None
        self._spectro_vm_line = None
        self._spectro_mesh = None
        self._spectro_fail_text = None
        return _tab_with_toolbar(cvs, fullscreen_callback=lambda: self._open_fullscreen_plot('Spectrogram'))

    def _build_tab_impedance(self) -> QWidget:
        self.fig_impedance, cvs = _mpl_fig(2, 1)
        self.ax_impedance = [self.fig_impedance.add_subplot(2, 1, k) for k in range(1, 3)]
        self.fig_impedance.set_tight_layout({'pad': 2.5})
        self.cvs_impedance = cvs
        self._tab_figures['Impedance'] = self.fig_impedance
        self._impedance_lines: dict[str, object] = {}
        self._impedance_texts: dict[str, object] = {}
        return _tab_with_toolbar(cvs, fullscreen_callback=lambda: self._open_fullscreen_plot('Impedance'))

    def _build_tab_chaos(self) -> QWidget:
        self.fig_chaos, cvs = _mpl_fig(1, 1)
        self.ax_chaos = self.fig_chaos.add_subplot(1, 1, 1)
        self.cvs_chaos = cvs
        self._tab_figures['Chaos & LLE'] = self.fig_chaos
        self._chaos_lines = {}
        self._chaos_texts = {}
        return _tab_with_toolbar(cvs, fullscreen_callback=lambda: self._open_fullscreen_plot('Chaos & LLE'))

    def _build_tab_modulation(self) -> QWidget:
        self.fig_mod, cvs = _mpl_fig(1, 1)
        # Use polar projection for phase-locking
        self.ax_mod = self.fig_mod.add_subplot(1, 1, 1, polar=True)
        self.cvs_mod = cvs
        self._tab_figures['Phase-Locking'] = self.fig_mod
        # Initialize bar stubs with default 18 bins (will be updated dynamically based on config)
        self._mod_bars = self.ax_mod.bar(np.zeros(18), np.zeros(18), width=0.1, alpha=0.7, color='#89B4FA', edgecolor='#74C7EC')
        # Initialize PLV vector line
        self._mod_vec, = self.ax_mod.plot([], [], color='#F38BA8', lw=3)
        # Initialize error text
        self._mod_error_text = self.ax_mod.text(0.5, 0.5, '', ha='center', va='center', 
                                                 transform=self.ax_mod.transAxes, fontsize=12, color='#666666', visible=False)
        return _tab_with_toolbar(cvs, fullscreen_callback=lambda: self._open_fullscreen_plot('Phase-Locking'))

    # ─────────────────────────────────────────────────────────────────
    #  MAIN UPDATE ENTRY POINT
    # ─────────────────────────────────────────────────────────────────
    def update_analytics(self, result):
        """Update all already-built tabs from a SimulationResult.

        Lazy tabs that haven't been visited yet are skipped — their guard
        at the top of each _update_* method returns immediately if the
        corresponding figure attribute doesn't exist yet.  They will be
        populated in _on_tab_changed when the user first clicks the tab.
        """
        self._last_result = result
        from core.analysis import full_analysis
        # LLE computation is now only triggered by the Compute LLE button, not from config
        stats = full_analysis(result, compute_lyapunov=False)
        self._last_stats = stats
        self._btn_compute_lle.setEnabled(True)

        # Tab 0 — always built
        self._update_passport(result, stats)

        # Tabs 1–17 — each guard-checks hasattr(self, 'fig_*') and returns
        # early if the canvas hasn't been created yet.
        self._update_spike_mechanism(result, stats)
        self._update_phase(result, stats)
        self._update_chaos(result, stats)
        self._update_kymo(result)
        self._update_spectrogram(result)
        self._update_impedance(result)
        self._update_modulation(result, stats)
        self._update_currents(result)
        self._update_gates(result)
        self._update_equil(result)
        if result.morph:
            self._update_energy_balance(result)
        self._update_spike_shape(result, stats)
        self._update_poincare(result, stats)


    def _compute_lle_now(self):
        """Explicit Lyapunov action (separate from simulation parameter toggles)."""
        if self._last_result is None:
            return
        from core.analysis import full_analysis

        stats = full_analysis(self._last_result, compute_lyapunov=True)
        self._last_stats = stats
        self._update_passport(self._last_result, stats)

        # Force-build chaos tab if not visited yet, using the robust helper
        self._ensure_built('_build_tab_chaos')

        self._update_chaos(self._last_result, stats)

        # Force synchronous repaint so the chart appears immediately
        if hasattr(self, 'cvs_chaos'):
            self.cvs_chaos.draw()

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
        
        # Safety check for currents dictionary
        if not hasattr(result, 'currents') or not isinstance(result.currents, dict):
            logging.error("SimulationResult missing or invalid currents attribute")
            return
            
        for name, curr in result.currents.items():
            if curr is None or len(curr) == 0:
                continue
            # Handle 2D current arrays (n_comp, n_time) - sum across compartments
            curr_arr = np.asarray(curr, dtype=float)
            if curr_arr.ndim == 2:
                curr_arr = np.sum(curr_arr, axis=0)
            # Safety check for empty arrays
            if curr_arr.size == 0:
                continue
            i_min = float(np.min(curr_arr))
            i_max = float(np.max(curr_arr))
            q_abs = float(np.sum(np.abs(curr_arr)) * dt_val) if dt_val > 0 else np.nan
            current_stats[name] = (i_min, i_max, q_abs)
        dominant_current = "—"
        # Fix np capture in lambda
        isfinite = np.isfinite
        if current_stats:
            dominant_current = max(
                current_stats.items(),
                key=lambda kv: kv[1][2] if isfinite(kv[1][2]) else -1.0,
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
            lines.append("║    LLE not computed. Use `Compute LLE` button.                  ║")
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
        # Fix np capture in lambda
        isfinite = np.isfinite
        top_channels = sorted(
            current_stats.items(),
            key=lambda kv: kv[1][2] if isfinite(kv[1][2]) else -1.0,
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
        atp_bd = stats.get('atp_breakdown', {})
        atp_na_s = f"{atp_bd.get('Na_pump', 0.0):.3e}" if atp_bd else "—"
        atp_ca_s = f"{atp_bd.get('Ca_pump', 0.0):.3e}" if atp_bd else "—"
        atp_bl_s = f"{atp_bd.get('baseline', 0.0):.3e}" if atp_bd else "—"
        lines += [
            f"║    ATP total  = {atp:.4e} nmol/cm²" + " " * 22 + "║",
            f"║      Na⁺ pump = {atp_na_s} nmol/cm²" + " " * 23 + "║",
            f"║      Ca²⁺pump = {atp_ca_s} nmol/cm²" + " " * 23 + "║",
            f"║      baseline = {atp_bl_s} nmol/cm²" + " " * 23 + "║",
            "╚══════════════════════════════════════════════════════════════════╝",
        ]

        self.passport_view.setPlainText("\n".join(lines))

    def _update_impedance(self, result):
        """Update membrane impedance magnitude/phase panels from latest run."""
        if not hasattr(self, 'fig_impedance'):
            return  # tab not yet visited
        from core.analysis import reconstruct_stimulus_trace, compute_membrane_impedance

        ax_mag, ax_phase = self.ax_impedance
        if not self._impedance_lines:
            self._impedance_lines["zmag"] = ax_mag.plot([], [], color="#2E86DE", lw=1.8, label="|Z(f)|")[0]
            self._impedance_lines["fres"] = ax_mag.axvline(0.0, color="#E67E22", ls="--", lw=1.2, visible=False)
            self._impedance_lines["zph"] = ax_phase.plot([], [], color="#8E44AD", lw=1.5, label="∠Z(f)")[0]
            self._impedance_lines["zero"] = ax_phase.axhline(0.0, color="#7f8c8d", lw=0.8, ls=":")

        if not self._impedance_texts:
            self._impedance_texts["mag"] = ax_mag.text(
                0.5, 0.5, "", ha="center", va="center", transform=ax_mag.transAxes, visible=False
            )
            self._impedance_texts["phase"] = ax_phase.text(
                0.5, 0.5, "", ha="center", va="center", transform=ax_phase.transAxes, visible=False
            )
        self._impedance_texts["mag"].set_visible(False)
        self._impedance_texts["phase"].set_visible(False)

        i_stim = reconstruct_stimulus_trace(result)
        imp = compute_membrane_impedance(result.t, result.v_soma, i_stim)

        if not imp.get("valid", False):
            self._impedance_lines["zmag"].set_data([], [])
            self._impedance_lines["zph"].set_data([], [])
            self._impedance_lines["fres"].set_visible(False)
            self._impedance_texts["mag"].set_text("Insufficient data for Z(f)")
            self._impedance_texts["mag"].set_visible(True)
            self._impedance_texts["phase"].set_text("Need dynamic stimulus content")
            self._impedance_texts["phase"].set_visible(True)
            _configure_ax_interactive(ax_mag, title="Impedance Magnitude", xlabel="Frequency (Hz)", ylabel="|Z| (kΩ·cm²)", show_legend=False)
            _configure_ax_interactive(ax_phase, title="Impedance Phase", xlabel="Frequency (Hz)", ylabel="Phase (deg)", show_legend=False)
            self.cvs_impedance.draw_idle()
            return

        f = imp["freq_hz"]
        zmag = imp["z_mag_kohm_cm2"]
        zph = imp["z_phase_deg"]
        fres = imp.get("f_res_hz", np.nan)
        zres = imp.get("z_res_kohm_cm2", np.nan)

        self._impedance_lines["zmag"].set_data(f, zmag)
        if np.isfinite(fres):
            self._impedance_lines["fres"].set_xdata([fres, fres])
            self._impedance_lines["fres"].set_visible(True)
            self._impedance_lines["fres"].set_label(f"f_res={fres:.2f} Hz")
        else:
            self._impedance_lines["fres"].set_visible(False)
        ax_mag.relim()
        ax_mag.autoscale_view()
        _configure_ax_interactive(
            ax_mag,
            title=f"Membrane Impedance |Z(f)|  (peak={zres:.2f} kΩ·cm² @ {fres:.2f} Hz)" if np.isfinite(fres) else "Membrane Impedance |Z(f)|",
            xlabel="Frequency (Hz)",
            ylabel="|Z| (kΩ·cm²)",
            show_legend=True,
        )

        self._impedance_lines["zph"].set_data(f, zph)
        ax_phase.relim()
        ax_phase.autoscale_view()
        _configure_ax_interactive(
            ax_phase,
            title="Impedance Phase",
            xlabel="Frequency (Hz)",
            ylabel="Phase (deg)",
            show_legend=True,
        )
        self.cvs_impedance.draw_idle()

    def _update_chaos(self, result, stats: dict):
        """Update Lyapunov/Chaos tab with trajectory divergence plot."""
        if not hasattr(self, 'fig_chaos'):
            return  # tab not yet visited

        t = stats.get('ftle_time_ms', [])
        div = stats.get('ftle_log_divergence', [])

        # Handle disabled/empty state: only block when there is truly no data
        _no_data = (len(t) == 0 or len(div) == 0
                    or stats.get('lyapunov_class') in ('disabled', None))
        if _no_data:
            msg = ("LLE not computed — run ≥1000 ms simulation, then click 'Compute LLE'."
                   if stats.get('lyapunov_class') != 'disabled'
                   else "LLE not computed. Click 'Compute LLE' button.")
            if 'div' not in self._chaos_texts:
                self._chaos_texts['div'] = self.ax_chaos.text(
                    0.5, 0.5, msg,
                    ha='center', va='center', transform=self.ax_chaos.transAxes,
                    fontsize=11, color='#89B4FA'
                )
            else:
                self._chaos_texts['div'].set_text(msg)
            self._chaos_texts['div'].set_visible(True)
            if 'div' in self._chaos_lines:
                self._chaos_lines['div'].set_data([], [])
            self.ax_chaos.set_xlim(0, 1)
            self.ax_chaos.set_ylim(0, 1)
            _configure_ax_interactive(self.ax_chaos, title='Lyapunov Exponent (Trajectory Separation)',
                                    xlabel='Time (ms)', ylabel='Log Divergence ln(d)', show_legend=False)
            self.cvs_chaos.draw_idle()
            return

        # Hide error message if present
        if 'div' in self._chaos_texts:
            self._chaos_texts['div'].set_visible(False)

        # Create persistent line if needed
        if 'div' not in self._chaos_lines:
            self._chaos_lines['div'] = self.ax_chaos.plot([], [], color='#89B4FA', lw=2.5, label='Trajectory divergence')[0]
            self._chaos_lines['trend'] = self.ax_chaos.plot([], [], color='#F38BA8', lw=1.5, ls='--', label='Linear trend')[0]

        # Plot trajectory divergence
        div_safe = _ensure_shape_compatible(div, t, "div")
        if div_safe is not None:
            self._chaos_lines['div'].set_data(t, div_safe)
        else:
            self._chaos_lines['div'].set_data([], [])

        # Add linear trend line
        lle_per_ms = stats.get('lle_per_ms', np.nan)
        if np.isfinite(lle_per_ms) and len(t) > 0 and div_safe is not None:
            t_trend = np.array([t[0], t[-1]])
            # Use the first point as intercept and lle_per_ms as slope
            div_trend = div_safe[0] + lle_per_ms * (t_trend - t[0])
            self._chaos_lines['trend'].set_data(t_trend, div_trend)
            self._chaos_lines['trend'].set_visible(True)
        else:
            self._chaos_lines['trend'].set_visible(False)

        self.ax_chaos.relim()
        self.ax_chaos.autoscale_view()
        _configure_ax_interactive(self.ax_chaos, title='Lyapunov Exponent (Trajectory Separation)',
                                    xlabel='Time (ms)', ylabel='Log Divergence ln(d)', show_legend=True)
        self.cvs_chaos.draw_idle()

    def _update_modulation(self, result, stats: dict):
        """Update Phase-Locking (Modulation) tab with polar plot."""
        if not hasattr(self, 'fig_mod'):
            return  # tab not yet visited
        centers = stats.get('modulation_phase_bin_centers_rad', [])
        rates = stats.get('modulation_phase_rate_hz', [])
        plv = stats.get('modulation_plv', np.nan)

        # Reset error text visibility to clean state
        self._mod_error_text.set_visible(False)

        # Handle invalid state
        if not stats.get('modulation_valid', False) or len(centers) == 0 or len(rates) == 0:
            # Hide bars and vector, show error text
            for bar in self._mod_bars:
                bar.set_height(0.0)
            self._mod_vec.set_data([], [])
            self._mod_error_text.set_text("Modulation analysis disabled or insufficient spikes.")
            self._mod_error_text.set_visible(True)
            self.cvs_mod.draw_idle()
            return

        # Hide error text
        self._mod_error_text.set_visible(False)

        # Rebuild bars if count changed (dynamic bin count from config)
        n_expected = len(centers)
        if len(self._mod_bars) != n_expected:
            # Remove old bars
            try:
                for bar in self._mod_bars:
                    bar.remove()
            except Exception:
                pass
            # Create new bars with correct count
            width = 2 * np.pi / n_expected if n_expected > 0 else 0.1
            self._mod_bars = self.ax_mod.bar(np.zeros(n_expected), np.zeros(n_expected), width=width, alpha=0.7, color='#89B4FA', edgecolor='#74C7EC')
        else:
            width = 2 * np.pi / n_expected if n_expected > 0 else 0.1

        # Update bar positions and heights
        for i, bar in enumerate(self._mod_bars):
            if i < len(centers):
                bar.set_x(centers[i] - width / 2)
                bar.set_width(width)
                bar.set_height(rates[i])

        # Update PLV vector line
        if np.isfinite(plv) and 'modulation_preferred_phase_rad' in stats:
            preferred_phase = stats['modulation_preferred_phase_rad']
            max_rate = np.max(rates) if len(rates) > 0 else 1.0
            self._mod_vec.set_data([0, preferred_phase], [0, max_rate * plv])
            self._mod_vec.set_label(f"PLV: {plv:.3f}")
            # Clear previous legend to prevent accumulation
            self.ax_mod.legend_.remove() if self.ax_mod.legend_ is not None else None
            self.ax_mod.legend(loc='upper right', fontsize=8, bbox_to_anchor=(1.3, 1.1))
        else:
            self._mod_vec.set_data([], [])

        self.ax_mod.set_title("Spike Phase-Locking (Firing Rate vs LFP Phase)", fontsize=11, fontweight='bold', pad=15)
        self.cvs_mod.draw_idle()

    # ─────────────────────────────────────────────────────────────────
    #  2 — GATE DYNAMICS
    # ─────────────────────────────────────────────────────────────────
    def _update_gates(self, result):
        """Plot gate dynamics with membrane potential overlay on single plot with checkboxes."""
        if not hasattr(self, 'fig_gates'):
            return  # tab not yet visited
        from core.analysis import extract_gate_traces
        gates = extract_gate_traces(result)
        t = result.t
        gates_signature = tuple(gates.keys())

        # Rebuild checkboxes if gates changed
        if self._gates_signature != gates_signature:
            self._gates_signature = gates_signature
            self._gates_lines = {}
            self._gates_visibility = {name: True for name in gates.keys()}

            # Clear old checkboxes
            for cb in self._gates_checkboxes.values():
                try:
                    cb.setParent(None)
                    cb.deleteLater()
                except Exception:
                    pass
            self._gates_checkboxes.clear()

            # Create new checkboxes
            from PySide6.QtWidgets import QCheckBox
            for name in gates.keys():
                cb = QCheckBox(name)
                cb.setChecked(True)
                cb.stateChanged.connect(lambda state, n=name: self._on_gates_checkbox_changed(n, state))
                self._gates_checkbox_layout.addWidget(cb)
                self._gates_checkboxes[name] = cb

        ax = self.ax_gates

        # Plot membrane potential (always visible on secondary axis)
        soma_color = PLOT_THEMES.get("Default", PLOT_THEMES["Default"]).get("soma", "#4080FF")
        if self._gates_line_v is None:
            self._gates_line_v = ax.plot([], [], color=soma_color, lw=2.0, alpha=0.7, label='V_soma')[0]
        v_soma_safe = _ensure_shape_compatible(result.v_soma, t, "v_soma")
        if v_soma_safe is not None:
            self._gates_line_v.set_data(t, v_soma_safe)
        else:
            self._gates_line_v.set_data([], [])
        self._gates_line_v.set_visible(True)

        # Plot gates on secondary axis
        if not hasattr(self, '_gates_ax2'):
            self._gates_ax2 = ax.twinx()

        # Plot each gate
        visible_names: set[str] = set()
        for name, trace in gates.items():
            if name not in self._gates_lines:
                color = GATE_COLORS.get(name, '#888888')
                self._gates_lines[name] = self._gates_ax2.plot([], [], color=color, lw=1.5, label=f'{name}', alpha=0.8)[0]

            line = self._gates_lines[name]
            is_visible = self._gates_visibility.get(name, True)
            trace_safe = _ensure_shape_compatible(trace, t, f"gate_{name}")
            if trace_safe is not None:
                line.set_data(t, trace_safe)
            else:
                line.set_data([], [])
            line.set_visible(is_visible)
            if is_visible:
                visible_names.add(name)

        # Hide invisible lines
        for name, line in self._gates_lines.items():
            if name not in visible_names:
                line.set_visible(False)

        ax.relim()
        ax.autoscale_view()
        self._gates_ax2.relim()
        self._gates_ax2.autoscale_view()
        self._gates_ax2.set_ylim(-0.05, 1.05)

        _configure_ax_interactive(
            ax,
            title="Gate Dynamics (toggle with checkboxes below)",
            xlabel="Time (ms)",
            ylabel="V (mV)",
            show_legend=True,
        )
        self._gates_ax2.set_ylabel("Gate Variable (0-1)", fontsize=10, fontweight="bold")

        self.cvs_gates.draw_idle()

    def _on_gates_checkbox_changed(self, name: str, state: int):
        """Handle checkbox state change for gate visibility."""
        self._gates_visibility[name] = (state != 0)
        if hasattr(self, '_last_result') and self._last_result is not None:
            self._update_gates(self._last_result)

    # ─────────────────────────────────────────────────────────────────
    #  2.5 — CHANNEL CURRENTS (NEW)
    # ─────────────────────────────────────────────────────────────────
    def _update_currents(self, result):
        """Plot channel currents with membrane potential overlay on single plot with checkboxes."""
        if not hasattr(self, 'fig_currents'):
            return  # tab not yet visited
        t = result.t

        # Safety check for currents dictionary
        if not hasattr(result, 'currents') or not isinstance(result.currents, dict):
            logging.error("SimulationResult missing or invalid currents attribute")
            return

        # Count non-zero current traces (handle 2D arrays)
        currents = {}
        for name, curr in result.currents.items():
            curr_arr = np.asarray(curr, dtype=float)
            if curr_arr.ndim == 2:
                curr_arr = np.sum(curr_arr, axis=0)
            if np.max(np.abs(curr_arr)) > 1e-9:
                currents[name] = curr_arr
        currents_signature = tuple(currents.keys())

        # Rebuild checkboxes if currents changed
        if self._currents_signature != currents_signature:
            self._currents_signature = currents_signature
            self._currents_lines = {}
            self._currents_visibility = {name: True for name in currents.keys()}

            # Clear old checkboxes
            for cb in self._currents_checkboxes.values():
                try:
                    cb.setParent(None)
                    cb.deleteLater()
                except Exception:
                    pass
            self._currents_checkboxes.clear()

            # Create new checkboxes
            from PySide6.QtWidgets import QCheckBox
            for name in currents.keys():
                cb = QCheckBox(name)
                cb.setChecked(True)
                cb.stateChanged.connect(lambda state, n=name: self._on_currents_checkbox_changed(n, state))
                self._currents_checkbox_layout.addWidget(cb)
                self._currents_checkboxes[name] = cb

        ax = self.ax_currents

        # Plot membrane potential (always visible on secondary axis)
        soma_color = PLOT_THEMES.get("Default", PLOT_THEMES["Default"]).get("soma", "#4080FF")
        if self._currents_line_v is None:
            self._currents_line_v = ax.plot([], [], color=soma_color, lw=2.0, alpha=0.7, label='V_soma')[0]
        v_soma_safe = _ensure_shape_compatible(result.v_soma, t, "v_soma")
        if v_soma_safe is not None:
            self._currents_line_v.set_data(t, v_soma_safe)
        else:
            self._currents_line_v.set_data([], [])
        self._currents_line_v.set_visible(True)

        # Plot currents on secondary axis
        if not hasattr(self, '_currents_ax2'):
            self._currents_ax2 = ax.twinx()

        # Plot each current
        visible_names: set[str] = set()
        for name, curr in currents.items():
            if name not in self._currents_lines:
                color = CHAN_COLORS.get(name, '#888888')
                self._currents_lines[name] = self._currents_ax2.plot([], [], color=color, lw=1.5, label=f'I_{name}', alpha=0.8)[0]

            line = self._currents_lines[name]
            is_visible = self._currents_visibility.get(name, True)
            curr_safe = _ensure_shape_compatible(curr, t, f"current_{name}")
            if curr_safe is not None:
                line.set_data(t, curr_safe)
            else:
                line.set_data([], [])
            line.set_visible(is_visible)
            if is_visible:
                visible_names.add(name)

        # Hide invisible lines
        for name, line in self._currents_lines.items():
            if name not in visible_names:
                line.set_visible(False)

        ax.relim()
        ax.autoscale_view()
        self._currents_ax2.relim()
        self._currents_ax2.autoscale_view()

        _configure_ax_interactive(
            ax,
            title="Channel Currents (toggle with checkboxes below)",
            xlabel="Time (ms)",
            ylabel="V (mV)",
            show_legend=True,
        )
        self._currents_ax2.set_ylabel("Current (µA/cm²)", fontsize=10, fontweight="bold")

        self.cvs_currents.draw_idle()

    def _on_currents_checkbox_changed(self, name: str, state: int):
        """Handle checkbox state change for current visibility."""
        self._currents_visibility[name] = (state != 0)
        if hasattr(self, '_last_result') and self._last_result is not None:
            self._update_currents(self._last_result)

    def _init_spike_mechanism_artists(self) -> None:
        if self._spike_mech_init_done:
            return
        ax1, ax2, ax3 = self.ax_spike_mech

        # Row 1: Voltage trace with peaks and calcium overlay
        soma_color = PLOT_THEMES.get("Default", PLOT_THEMES["Default"]).get("soma", "#4080FF")
        self._spike_mech_lines["ax1_vm"] = ax1.plot([], [], color=soma_color, lw=2.0, label="V_soma")[0]
        self._spike_mech_lines["ax1_peaks"] = ax1.plot([], [], linestyle="None", marker="o", markersize=4.0, color="#AA3377", label="spike peaks")[0]
        self._spike_mech_lines["ax1_eca"] = ax1.plot([], [], ":", color="#FF7F0E", lw=1.5, label="E_Ca (Nernst)", alpha=0.8)[0]
        self._spike_mech_ax2b = ax1.twinx()
        self._spike_mech_lines["ax1_ca"] = self._spike_mech_ax2b.plot([], [], "s--", color="#D62728", lw=1.4, label="Ca_i@spike")[0]

        # Row 2: Channel activity with explanation
        self._spike_mech_lines["ax2_inet"] = ax2.plot([], [], color="k", lw=1.0, alpha=0.3, label="I_net")[0]
        self._spike_mech_texts["ax2_reasons"] = ax2.text(
            0.01, 0.02, "", transform=ax2.transAxes, fontsize=8.5, color="#333333",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#F8F8F8", edgecolor="#CCCCCC", alpha=0.9),
        )
        # Initialize centered error text for ax2
        self._spike_mech_texts["ax2_error"] = ax2.text(
            0.5, 0.5, "", ha='center', va='center', transform=ax2.transAxes, fontsize=12,
            visible=False
        )

        # Row 3: Current contribution at threshold (stacked bar chart)
        self._spike_mech_texts["ax3_error"] = ax3.text(
            0.5, 0.5, "", ha='center', va='center', transform=ax3.transAxes, fontsize=12,
            visible=False
        )

        self._spike_mech_init_done = True

    def _update_spike_mechanism(self, result, stats: dict):
        """Explain spike attenuation using per-spike ion/channel dynamics."""
        if not hasattr(self, 'fig_spike_mech'):
            return  # tab not yet visited
        from core.analysis import detect_spikes

        self._init_spike_mechanism_artists()
        t = np.asarray(result.t, dtype=float)
        v = np.asarray(result.v_soma, dtype=float)
        ax1, ax2, ax3 = self.ax_spike_mech

        for ax in self.ax_spike_mech:
            ax.set_axis_on()
            ax.set_visible(True)

        kwargs = _spike_detect_kwargs_from_stats(stats)
        peak_idx, spike_times, _ = detect_spikes(v, t, **kwargs)
        n_sp = len(spike_times)

        # Store spike data for zoomer
        self._spike_zoomer_data = (t, v, spike_times)
        # Update zoomer range
        if hasattr(self, '_spike_zoomer'):
            self._spike_zoomer.setRange(1, max(1, n_sp))

        # Row 1: Voltage trace with peaks and calcium overlay
        v_safe = _ensure_shape_compatible(v, t, "v_soma")
        if v_safe is not None:
            self._spike_mech_lines["ax1_vm"].set_data(t, v_safe)
        else:
            self._spike_mech_lines["ax1_vm"].set_data([], [])
        if n_sp > 0 and v_safe is not None:
            self._spike_mech_lines["ax1_peaks"].set_data(t[peak_idx], v_safe[peak_idx])
            self._spike_mech_lines["ax1_peaks"].set_visible(True)
        else:
            self._spike_mech_lines["ax1_peaks"].set_data([], [])
            self._spike_mech_lines["ax1_peaks"].set_visible(False)

        # Add continuous calcium trace on secondary axis
        if (result.ca_i is not None and
            len(result.ca_i) > 0 and
            result.ca_i.shape[0] > 0):
            ca_nM = np.asarray(result.ca_i[0, :], dtype=float) * 1e6
            ca_nM_safe = _ensure_shape_compatible(ca_nM, t, "ca_nM")
            if ca_nM_safe is not None:
                self._spike_mech_lines["ax1_ca"].set_data(t, ca_nM_safe)
                self._spike_mech_lines["ax1_ca"].set_visible(True)
                self._spike_mech_ax2b.set_ylabel("[Ca²+]_i (nM)", fontsize=10, fontweight="bold", color="#D62728")
                self._spike_mech_ax2b.tick_params(axis="y", labelcolor="#D62728")
                self._spike_mech_ax2b.relim()
                self._spike_mech_ax2b.autoscale_view()
            else:
                self._spike_mech_lines["ax1_ca"].set_data([], [])
                self._spike_mech_lines["ax1_ca"].set_visible(False)

            # Compute dynamic E_Ca Nernst potential
            # E_Ca = (RT/zF) * ln([Ca]_out / [Ca]_in)
            # At 37°C: RT/F = 26.7 mV, for Ca2+ (z=2): RT/(zF) = 13.35 mV
            ca_in_M = np.asarray(result.ca_i[0, :], dtype=float)
            ca_out_M = result.config.calcium.Ca_ext if hasattr(result.config.calcium, 'Ca_ext') else 2.0
            RT_over_zF = 13.35  # mV at 37°C for Ca2+
            e_ca = RT_over_zF * np.log(ca_out_M / (ca_in_M + 1e-12))
            e_ca_safe = _ensure_shape_compatible(e_ca, t, "e_ca")
            if e_ca_safe is not None:
                self._spike_mech_lines["ax1_eca"].set_data(t, e_ca_safe)
                self._spike_mech_lines["ax1_eca"].set_visible(True)
            else:
                self._spike_mech_lines["ax1_eca"].set_data([], [])
                self._spike_mech_lines["ax1_eca"].set_visible(False)
        else:
            self._spike_mech_lines["ax1_ca"].set_data([], [])
            self._spike_mech_lines["ax1_ca"].set_visible(False)
            self._spike_mech_ax2b.set_visible(False)
            self._spike_mech_lines["ax1_eca"].set_data([], [])
            self._spike_mech_lines["ax1_eca"].set_visible(False)

        _configure_ax_interactive(
            ax1,
            title=f"Spike Peaks Timeline (N={n_sp})",
            xlabel="Time (ms)",
            ylabel="V (mV)",
            show_legend=True,
        )

        # Row 2: Channel activity with explanation
        if n_sp < 2:
            self._spike_mech_texts["ax2_reasons"].set_text("Channel activity shown below.\n(Attenuation analysis requires ≥2 spikes)")
            self._spike_mech_texts["ax2_error"].set_visible(False)
            # Continue to plot channel activity even with < 2 spikes

        # Hide error text when we have enough spikes
        self._spike_mech_texts["ax2_error"].set_visible(False)

        # Time-resolved channel activity to explain attenuation causes
        dt = float(np.mean(np.diff(t))) if len(t) > 1 else 0.1
        smooth_pts = max(3, int(round(2.0 / max(dt, 1e-6))))
        if smooth_pts % 2 == 0:
            smooth_pts += 1

        # Cache kernel for efficiency
        _smooth_kernel = np.ones(smooth_pts) / smooth_pts

        def _smooth(x: np.ndarray) -> np.ndarray:
            """Optimized smoothing using pre-computed kernel."""
            if len(x) < smooth_pts or smooth_pts <= 1:
                return x
            return np.convolve(x, _smooth_kernel, mode='same')

        # Plot normalized channel activity
        curr_candidates = ["Na", "K", "ICa", "IA", "SK", "Ih", "Leak"]
        
        # Safety check for currents dictionary
        if not hasattr(result, 'currents') or not isinstance(result.currents, dict):
            logging.error("SimulationResult missing or invalid currents attribute")
            return
            
        # Handle 2D current arrays (n_comp, n_time) - sum across compartments
        traces = {}
        for k in curr_candidates:
            if k in result.currents:
                tr = np.asarray(result.currents[k], dtype=float)
                # If 2D array, sum across spatial dimension to get total current
                if tr.ndim == 2:
                    tr = np.sum(tr, axis=0)
                traces[k] = tr
        
        for line in self._spike_mech_norm_lines.values():
            line.set_data([], [])
            line.set_visible(False)
        
        for name, tr in traces.items():
            if name not in self._spike_mech_norm_lines:
                color = CHAN_COLORS.get(name, '#888888')
                self._spike_mech_norm_lines[name] = ax2.plot([], [], color=color, lw=1.5, alpha=0.8, label=name)[0]
            # Normalize and smooth
            tr_norm = np.abs(tr) / (np.max(np.abs(tr)) + 1e-12)
            tr_smooth = _smooth(tr_norm)
            tr_smooth_safe = _ensure_shape_compatible(tr_smooth, t, f"trace_{name}")
            if tr_smooth_safe is not None:
                self._spike_mech_norm_lines[name].set_data(t, tr_smooth_safe)
                self._spike_mech_norm_lines[name].set_visible(True)
            else:
                self._spike_mech_norm_lines[name].set_data([], [])
                self._spike_mech_norm_lines[name].set_visible(False)

        # Compute and plot net current: I_net = sum of all ionic currents
        if traces:
            i_net = np.zeros_like(t)
            for tr in traces.values():
                i_net += tr
            i_net_safe = _ensure_shape_compatible(i_net, t, "i_net")
            if i_net_safe is not None:
                self._spike_mech_lines["ax2_inet"].set_data(t, i_net_safe)
                self._spike_mech_lines["ax2_inet"].set_visible(True)
            else:
                self._spike_mech_lines["ax2_inet"].set_data([], [])
                self._spike_mech_lines["ax2_inet"].set_visible(False)

        _configure_ax_interactive(
            ax2,
            title="Time-Resolved Channel Activity (Normalized)",
            xlabel="Time (ms)",
            ylabel="|I|/max(|I|)",
            show_legend=True,
        )

        # Add explanation text - data-driven based on actual channels
        active_channels = list(traces.keys())
        if n_sp >= 2:
            # Compare first vs last spike
            v_first_peak = v[peak_idx[0]]
            v_last_peak = v[peak_idx[-1]]
            attenuation_pct = 100.0 * (v_first_peak - v_last_peak) / abs(v_first_peak) if v_first_peak != 0 else 0.0
            explanation = (f"Spike attenuation: {attenuation_pct:.1f}% (first={v_first_peak:.1f}mV, last={v_last_peak:.1f}mV)\n"
                          f"Active channels: {', '.join(active_channels)}\n"
                          f"Attenuation mechanisms: Na+ inactivation, K+ activation")
            if 'ICa' in active_channels or 'SK' in active_channels:
                explanation += ", Ca²+-dependent SK"
            if 'IA' in active_channels:
                explanation += ", A-type K+ adaptation"
        else:
            explanation = (f"Single spike detected ({n_sp} spike).\n"
                          f"Active channels: {', '.join(active_channels)}\n"
                          f"Run longer simulation for attenuation analysis.")
        self._spike_mech_texts["ax2_reasons"].set_text(explanation)

        # Row 3: Current contribution at threshold (stacked bar chart)
        ax3.clear()
        if n_sp > 0 and traces:
            # K-channel candidates for outward current analysis
            k_channels = [name for name in ["K", "IA", "IM", "SK"] if name in traces]

            if k_channels:
                contributions = []
                spike_indices = []

                # For each spike, find time point where V = V_threshold
                # Use -40mV as typical threshold or use spike onset
                threshold_v = -40.0

                for i, peak_i in enumerate(peak_idx):
                    # Find threshold crossing before peak
                    start_idx = max(0, peak_i - 20)
                    threshold_idx = None

                    for j in range(start_idx, peak_i):
                        if v[j] >= threshold_v:
                            threshold_idx = j
                            break

                    if threshold_idx is not None:
                        # Calculate outward current contributions at threshold
                        outward_currents = {}
                        total_outward = 0.0

                        for name in k_channels:
                            current = traces[name][threshold_idx]
                            if current > 0:  # Outward (positive) current
                                outward_currents[name] = current
                                total_outward += current

                        # Calculate percentages
                        if total_outward > 0:
                            percentages = {name: 100.0 * val / total_outward for name, val in outward_currents.items()}
                            contributions.append(percentages)
                            spike_indices.append(i + 1)

                # Plot stacked bar chart
                if contributions:
                    colors_list = [CHAN_COLORS.get(name, "#888888") for name in k_channels]

                    # Prepare data for stacked bar chart
                    n_bars = len(contributions)
                    bottom = np.zeros(n_bars)

                    for name in k_channels:
                        heights = [contrib.get(name, 0) for contrib in contributions]
                        ax3.bar(range(n_bars), heights, bottom=bottom, label=name, color=CHAN_COLORS.get(name, "#888888"), alpha=0.8)
                        bottom += heights

                    ax3.set_xticks(range(n_bars))
                    ax3.set_xticklabels([f"#{idx}" for idx in spike_indices], rotation=45, ha='right')
                    ax3.set_ylabel('Outward Current %')
                    ax3.set_title('K-Channel Contribution at Threshold')
                    ax3.legend(loc='upper right', fontsize=8)
                    ax3.set_ylim(0, 100)
                    ax3.grid(alpha=0.2, axis='y')
                else:
                    ax3.text(0.5, 0.5, 'No threshold crossings detected', ha='center', va='center',
                           transform=ax3.transAxes, fontsize=10)
            else:
                ax3.text(0.5, 0.5, 'No K-channels available', ha='center', va='center',
                       transform=ax3.transAxes, fontsize=10)
        else:
            ax3.text(0.5, 0.5, 'Need spikes for threshold analysis', ha='center', va='center',
                   transform=ax3.transAxes, fontsize=10)

        # ── ATP Overlay (Metabolic Status) ───────────────────────────────
        if hasattr(result, 'config') and result.config.metabolism.enable_dynamic_atp:
            # Extract ATP from state vector
            atp_data = None
            if hasattr(result, 'y_all') and result.config.metabolism.enable_dynamic_atp:
                n_comp = result.n_comp
                if n_comp == 1:
                    atp_data = result.y_all[-1, :]  # Last row is ATP for single comp
                else:
                    atp_data = result.y_all[-n_comp:, :]  # Last n_comp rows are ATP
                    atp_data = atp_data[0, :]  # Soma ATP

            if atp_data is not None and len(atp_data) > 0:
                # Get final ATP value
                atp_final = float(atp_data[-1])
                
                # Determine color and warning
                atp_color = '#FF6B6B' if atp_final < 0.5 else '#A6E3A1'
                atp_text = f"Current ATP: {atp_final:.3f} mM"
                if atp_final < 0.5:
                    atp_text += "\n⚠ METABOLIC EXHAUSTION"
                
                # Add text overlay in corner of ax1 (voltage trace)
                if 'atp_overlay' not in self._spike_mech_texts:
                    self._spike_mech_texts['atp_overlay'] = ax1.text(
                        0.98, 0.02, atp_text,
                        ha='right', va='bottom',
                        transform=ax1.transAxes,
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor=atp_color)
                    )
                else:
                    self._spike_mech_texts['atp_overlay'].set_text(atp_text)
                    self._spike_mech_texts['atp_overlay'].set_color(atp_color)

        self.cvs_spike_mech.draw_idle()

    def _on_spike_zoomer_changed(self, spike_num: int):
        """Handle spike zoomer spinbox change to zoom to specific spike."""
        if self._spike_zoomer_data is None:
            return

        t, v, spike_times = self._spike_zoomer_data
        n_sp = len(spike_times)

        if spike_num < 1 or spike_num > n_sp:
            return

        spike_time = spike_times[spike_num - 1]
        # Zoom to [spike_time - 2ms, spike_time + 5ms]
        t_min = spike_time - 2.0
        t_max = spike_time + 5.0

        ax1, ax2, ax3 = self.ax_spike_mech
        ax1.set_xlim(t_min, t_max)
        ax2.set_xlim(t_min, t_max)
        ax3.set_xlim(t_min, t_max) if n_sp > 0 else None

        self.cvs_spike_mech.draw_idle()

    # ─────────────────────────────────────────────────────────────────
    #  3 — EQUILIBRIUM CURVES
    # ─────────────────────────────────────────────────────────────────
    def _update_equil(self, result):
        if not hasattr(self, 'fig_equil'):
            return  # tab not yet visited
        from core.analysis import (compute_equilibrium_curves,
                                    compute_optional_equilibrium)
        cfg   = result.config
        phi   = cfg.env.phi
        V_rng = np.linspace(-100, 60, 500)
        eq    = compute_equilibrium_curves(V_rng, phi)
        opt   = compute_optional_equilibrium(V_rng, cfg, phi)

        ax1, ax2, ax3, ax4 = self.ax_equil
        if not self._equil_init_done:
            self._equil_lines["m_inf"] = ax1.plot([], [], color=GATE_COLORS['m'], lw=2.5, label='m∞ (Na act)', alpha=0.9)[0]
            self._equil_lines["h_inf"] = ax1.plot([], [], color=GATE_COLORS['h'], lw=2.5, label='h∞ (Na inact)', alpha=0.9)[0]
            self._equil_lines["n_inf"] = ax1.plot([], [], color=GATE_COLORS['n'], lw=2.5, label='n∞ (K act)', alpha=0.9)[0]

            self._equil_lines["tau_m"] = ax2.plot([], [], color=GATE_COLORS['m'], lw=2.5, label='τₘ', alpha=0.9)[0]
            self._equil_lines["tau_h"] = ax2.plot([], [], color=GATE_COLORS['h'], lw=2.5, label='τₕ', alpha=0.9)[0]
            self._equil_lines["tau_n"] = ax2.plot([], [], color=GATE_COLORS['n'], lw=2.5, label='τₙ', alpha=0.9)[0]

            self._equil_lines["phase_m"] = ax3.plot([], [], color=GATE_COLORS['m'], lw=2, label='m (Na act)', alpha=0.9)[0]
            self._equil_lines["phase_h"] = ax3.plot([], [], color=GATE_COLORS['h'], lw=2, label='h (Na inact)', alpha=0.9)[0]

            self._equil_lines["g_na"] = ax4.plot([], [], color=GATE_COLORS['m'], lw=2.5, label='g_Na(t)', alpha=0.9)[0]
            self._equil_lines["g_k"] = ax4.plot([], [], color=GATE_COLORS['n'], lw=2.5, label='g_K(t)', alpha=0.9)[0]
            self._equil_init_done = True

        # x_inf(V) — improved layout
        self._equil_lines["m_inf"].set_data(V_rng, eq['m_inf'])
        self._equil_lines["h_inf"].set_data(V_rng, eq['h_inf'])
        self._equil_lines["n_inf"].set_data(V_rng, eq['n_inf'])
        active_opt_inf = set()
        for k in ('r_inf', 's_inf', 'u_inf', 'a_inf', 'b_inf'):
            if k in opt:
                lbl = k.replace('_inf', '∞')
                key = f"opt_{k}"
                if key not in self._equil_lines:
                    self._equil_lines[key] = ax1.plot([], [], lw=1.8, ls='--', label=lbl, alpha=0.8)[0]
                self._equil_lines[key].set_data(V_rng, opt[k])
                active_opt_inf.add(key)
        for key in ("opt_r_inf", "opt_s_inf", "opt_u_inf", "opt_a_inf", "opt_b_inf"):
            if key in self._equil_lines and key not in active_opt_inf:
                self._equil_lines[key].set_data([], [])
        ax1.relim()
        ax1.autoscale_view()
        ax1.set_ylim(-0.05, 1.05)
        _configure_ax_interactive(ax1, title='Steady-state gating (x∞)', 
                                  xlabel='V (mV)', ylabel='x∞', show_legend=True)

        # τ(V) — main gating time constants
        self._equil_lines["tau_m"].set_data(V_rng, eq['tau_m'])
        self._equil_lines["tau_h"].set_data(V_rng, eq['tau_h'])
        self._equil_lines["tau_n"].set_data(V_rng, eq['tau_n'])
        active_opt_tau = set()
        for k in ('tau_r', 'tau_s', 'tau_u', 'tau_a', 'tau_b'):
            if k in opt:
                key = f"opt_{k}"
                if key not in self._equil_lines:
                    self._equil_lines[key] = ax2.plot([], [], lw=1.8, ls='--', label=k, alpha=0.8)[0]
                self._equil_lines[key].set_data(V_rng, opt[k])
                active_opt_tau.add(key)
        for key in ("opt_tau_r", "opt_tau_s", "opt_tau_u", "opt_tau_a", "opt_tau_b"):
            if key in self._equil_lines and key not in active_opt_tau:
                self._equil_lines[key].set_data([], [])
        ax2.relim()
        ax2.autoscale_view()
        _configure_ax_interactive(ax2, title=f'Time constants (φ = {phi:.2f})',
                                  xlabel='V (mV)', ylabel='τ (ms)', show_legend=True)

        # Phase portrait V-m and V-h
        v_soma_safe = _ensure_shape_compatible(result.v_soma, result.y[result.n_comp, :], "v_soma_phase")
        if v_soma_safe is not None:
            m_safe = _ensure_shape_compatible(result.y[result.n_comp, :], v_soma_safe, "m_phase")
            h_safe = _ensure_shape_compatible(result.y[2 * result.n_comp, :], v_soma_safe, "h_phase")
            if m_safe is not None:
                self._equil_lines["phase_m"].set_data(v_soma_safe, m_safe)
            else:
                self._equil_lines["phase_m"].set_data([], [])
            if h_safe is not None:
                self._equil_lines["phase_h"].set_data(v_soma_safe, h_safe)
            else:
                self._equil_lines["phase_h"].set_data([], [])
        else:
            self._equil_lines["phase_m"].set_data([], [])
            self._equil_lines["phase_h"].set_data([], [])
        ax3.relim()
        ax3.autoscale_view()
        _configure_ax_interactive(ax3, title='V – Gate Phase Portraits',
                                  xlabel='V (mV)', ylabel='Gate variable', show_legend=True)

        # Effective conductances over time
        t    = result.t
        m_t  = result.y[result.n_comp, :]
        h_t  = result.y[2 * result.n_comp, :]
        n_t  = result.y[3 * result.n_comp, :]
        g_Na = result.config.channels.gNa_max * (m_t ** 3) * h_t
        g_K  = result.config.channels.gK_max  * (n_t ** 4)
        g_Na_safe = _ensure_shape_compatible(g_Na, t, "g_Na")
        g_K_safe = _ensure_shape_compatible(g_K, t, "g_K")
        if g_Na_safe is not None:
            self._equil_lines["g_na"].set_data(t, g_Na_safe)
        else:
            self._equil_lines["g_na"].set_data([], [])
        if g_K_safe is not None:
            self._equil_lines["g_k"].set_data(t, g_K_safe)
        else:
            self._equil_lines["g_k"].set_data([], [])
        ax4.relim()
        ax4.autoscale_view()
        _configure_ax_interactive(ax4, title='Effective Conductances',
                                  xlabel='Time (ms)', ylabel='g (mS/cm²)', show_legend=True)

        self.cvs_equil.draw_idle()

    # ─────────────────────────────────────────────────────────────────
    #  4 — PHASE PLANE + NULLCLINES
    # ─────────────────────────────────────────────────────────────────
    def _update_phase(self, result, stats: dict):
        if not hasattr(self, 'fig_phase'):
            return  # tab not yet visited
        from core.analysis import compute_nullclines

        t     = result.t
        V     = result.v_soma
        n_t   = result.y[3 * result.n_comp, :]   # n gate
        cfg   = result.config
        I_stm = cfg.stim.Iext if cfg.stim.stim_type == 'const' else 0.0

        # Store full trajectory data for slider
        self._phase_full_data = (t, V, n_t, cfg, I_stm, stats)

        # Update slider range based on simulation duration
        if hasattr(self, '_phase_time_slider') and len(t) > 0:
            max_time = t[-1]
            self._phase_time_slider.blockSignals(True)
            self._phase_time_slider.setRange(0, int(max_time))
            self._phase_time_slider.setValue(int(max_time))  # Default to full range
            self._phase_time_slider.blockSignals(False)
            self._phase_time_label.setText(f"{int(max_time)} ms")

        # Get current time window from slider
        time_window = None
        if hasattr(self, '_phase_time_slider') and hasattr(self, '_phase_full_data'):
            time_window = self._phase_time_slider.value()
            if time_window == self._phase_time_slider.maximum():
                time_window = None  # Show all

        # Apply time window if set
        if time_window is not None and time_window > 0:
            idx_end = np.searchsorted(t, time_window)
            V = V[:idx_end]
            n_t = n_t[:idx_end]
            t = t[:idx_end]

        # v12.0: Plot 3 distinct segments to visualize trajectory collapse during adaptation
        V_rng               = np.linspace(-100, 60, 500)
        n_V_null, n_n_null  = compute_nullclines(V_rng, cfg, I_stm)

        ax = self.ax_phase
        if not self._phase_lines:
            self._phase_lines["traj_initial"] = ax.plot([], [], color='#2060CC', lw=1.5, zorder=3, label='Initial Spike (blue)')[0]
            self._phase_lines["traj_middle"] = ax.plot([], [], color='#888888', lw=1.5, zorder=3, label='Middle (gray)')[0]
            self._phase_lines["traj_final"] = ax.plot([], [], color='#DC5A10', lw=1.5, zorder=3, label='Final Spike (red)')[0]
            self._phase_lines["rest"] = ax.plot([], [], 'go', ms=8, zorder=5, label='Resting state')[0]
            self._phase_lines["spikes"] = ax.plot([], [], 'r*', ms=12, zorder=6, label='Spike peaks')[0]
            self._phase_lines["n_null"] = ax.plot([], [], color='#40CC40', lw=2, ls='--', label='dn/dt = 0  (n\u221e)')[0]
            self._phase_lines["v_null"] = ax.plot([], [], color='#CC4040', lw=2, ls='--', label='dV/dt = 0')[0]

        # Divide trajectory into 3 segments based on time
        if len(t) > 0 and t[-1] > 100:
            # Initial segment: first 100ms or first third
            t_end_initial = min(100, t[-1] / 3)
            idx_initial = np.searchsorted(t, t_end_initial)
            V_initial = V[:idx_initial]
            n_initial = n_t[:idx_initial]

            # Middle segment: middle third
            t_start_middle = t_end_initial
            t_end_middle = t_start_middle + (t[-1] - t_start_middle) / 3
            idx_start_middle = np.searchsorted(t, t_start_middle)
            idx_end_middle = np.searchsorted(t, t_end_middle)
            V_middle = V[idx_start_middle:idx_end_middle]
            n_middle = n_t[idx_start_middle:idx_end_middle]

            # Final segment: last third
            idx_final = idx_end_middle
            V_final = V[idx_final:]
            n_final = n_t[idx_final:]
        else:
            # Short simulation: just plot all as initial, hide middle/final lines
            V_initial, n_initial = V, n_t
            self._phase_lines["traj_middle"].set_visible(False)
            self._phase_lines["traj_final"].set_visible(False)
            V_middle, n_middle = V[0:0], n_t[0:0]
            V_final, n_final = V[0:0], n_t[0:0]

        # Plot the three segments
        self._phase_lines["traj_initial"].set_data(V_initial, n_initial)
        self._phase_lines["traj_middle"].set_data(V_middle, n_middle)
        self._phase_lines["traj_final"].set_data(V_final, n_final)
        self._phase_lines["rest"].set_data([V[0]], [n_t[0]])

        # Ensure middle/final lines are visible for long simulations
        if len(t) > 0 and t[-1] > 100:
            self._phase_lines["traj_middle"].set_visible(True)
            self._phase_lines["traj_final"].set_visible(True)

        # Spike detection markers (all spikes)
        if stats['n_spikes'] > 0:
            from core.analysis import detect_spikes
            pk_idx, _, _ = detect_spikes(V, t, **_spike_detect_kwargs_from_stats(stats))
            self._phase_lines["spikes"].set_data(V[pk_idx], n_t[pk_idx])
        else:
            self._phase_lines["spikes"].set_data([], [])

        # Nullclines (unchanged)
        self._phase_lines["n_null"].set_data(V_rng, n_n_null)
        valid = ~np.isnan(n_V_null)
        self._phase_lines["v_null"].set_data(V_rng[valid], n_V_null[valid])

        ax.set_xlabel('V (mV)',  fontsize=11)
        ax.set_ylabel('n  [K\u207a activation]', fontsize=11)

        title_suffix = f" (0-{time_window} ms)" if time_window is not None else ""
        ax.set_title(f'Phase Plane Trajectory{title_suffix}', fontsize=12, fontweight='bold')

        self.cvs_phase.draw_idle()

    def _on_phase_time_slider_changed(self, value: int):
        """Handle time slider change to update phase plane plot with selected time window."""
        if self._phase_full_data is None:
            return

        t, V, n_t, cfg, I_stm, stats = self._phase_full_data

        # Update label
        if hasattr(self, '_phase_time_label'):
            if value == self._phase_time_slider.maximum():
                self._phase_time_label.setText("All")
            else:
                self._phase_time_label.setText(f"{value} ms")

        # Apply time window
        if value > 0 and value < self._phase_time_slider.maximum():
            idx_end = np.searchsorted(t, value)
            V_window = V[:idx_end]
            n_t_window = n_t[:idx_end]
            t_window = t[:idx_end]
        else:
            V_window = V
            n_t_window = n_t
            t_window = t

        # Update plot with time window
        from core.analysis import compute_nullclines
        V_rng = np.linspace(-100, 60, 500)
        n_V_null, n_n_null = compute_nullclines(V_rng, cfg, I_stm)

        ax = self.ax_phase

        # Divide trajectory into 3 segments based on time window
        if len(t_window) > 0 and t_window[-1] > 100:
            t_end_initial = min(100, t_window[-1] / 3)
            idx_initial = np.searchsorted(t_window, t_end_initial)
            V_initial = V_window[:idx_initial]
            n_initial = n_t_window[:idx_initial]

            t_start_middle = t_end_initial
            t_end_middle = t_start_middle + (t_window[-1] - t_start_middle) / 3
            idx_start_middle = np.searchsorted(t_window, t_start_middle)
            idx_end_middle = np.searchsorted(t_window, t_end_middle)
            V_middle = V_window[idx_start_middle:idx_end_middle]
            n_middle = n_t_window[idx_start_middle:idx_end_middle]

            idx_final = idx_end_middle
            V_final = V_window[idx_final:]
            n_final = n_t_window[idx_final:]
        else:
            V_initial, n_initial = V_window, n_t_window
            self._phase_lines["traj_middle"].set_visible(False)
            self._phase_lines["traj_final"].set_visible(False)
            V_middle, n_middle = V_window[0:0], n_t_window[0:0]
            V_final, n_final = V_window[0:0], n_t_window[0:0]

        # Update trajectory lines
        self._phase_lines["traj_initial"].set_data(V_initial, n_initial)
        self._phase_lines["traj_middle"].set_data(V_middle, n_middle)
        self._phase_lines["traj_final"].set_data(V_final, n_final)
        self._phase_lines["rest"].set_data([V_window[0]], [n_t_window[0]])

        if len(t_window) > 0 and t_window[-1] > 100:
            self._phase_lines["traj_middle"].set_visible(True)
            self._phase_lines["traj_final"].set_visible(True)

        # Update spike markers (only within time window)
        if stats['n_spikes'] > 0:
            from core.analysis import detect_spikes
            pk_idx, _, _ = detect_spikes(V_window, t_window, **_spike_detect_kwargs_from_stats(stats))
            self._phase_lines["spikes"].set_data(V_window[pk_idx], n_t_window[pk_idx])
        else:
            self._phase_lines["spikes"].set_data([], [])

        # Nullclines (unchanged)
        self._phase_lines["n_null"].set_data(V_rng, n_n_null)
        valid = ~np.isnan(n_V_null)
        self._phase_lines["v_null"].set_data(V_rng[valid], n_V_null[valid])

        title_suffix = f" (0-{value} ms)" if value < self._phase_time_slider.maximum() else ""
        ax.set_title(f'Phase Plane Trajectory{title_suffix}', fontsize=12, fontweight='bold')
        ax.set_xlim(-100, 60)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

        self.cvs_phase.draw_idle()

    # ─────────────────────────────────────────────────────────────────
    #  5 — KYMOGRAPH
    # ─────────────────────────────────────────────────────────────────
    def _update_kymo(self, result):
        if not hasattr(self, 'fig_kymo'):
            return  # tab not yet visited
        n = result.n_comp
        ax1, ax2 = self._kymo_axes

        if n < 2:
            ax2.set_visible(False)
            if self._kymo_empty_text is None:
                self._kymo_empty_text = ax1.text(
                    0.5, 0.5, 'Single-compartment mode\n(no kymograph)',
                    ha='center', va='center', fontsize=14, color='gray',
                    transform=ax1.transAxes
                )
            if self._kymo_im1 is not None:
                self._kymo_im1.set_data(np.zeros((1, 1)))
            if self._kymo_im2 is not None:
                self._kymo_im2.set_data(np.zeros((1, 1)))
            self.cvs_kymo.draw_idle()
            return

        if self._kymo_empty_text is not None:
            try:
                self._kymo_empty_text.remove()
            except Exception:
                pass
            self._kymo_empty_text = None

        ax2.set_visible(True)
        mc = result.config.morphology
        t  = result.t
        V  = result.v_all      # shape (N_comp, N_time)

        # Build two axonal paths: soma → Branch1 tip, soma → Branch2 tip
        i_tr_s   = 1 + mc.N_ais
        i_branch = i_tr_s + mc.N_trunk
        i_b1s    = i_branch + 1
        i_b2s    = i_b1s + mc.N_b1

        path1 = list(range(0, min(i_b1s + mc.N_b1, n)))
        path2 = list(range(0, min(i_b2s + mc.N_b2, n)))
        data1 = V[path1, :]
        data2 = V[path2, :]
        vmin = float(V.min())
        vmax = float(V.max())

        if self._kymo_im1 is None:
            self._kymo_im1 = ax1.imshow(
                data1, aspect='auto', origin='lower',
                extent=[t[0], t[-1], 0, len(path1)],
                cmap='plasma', vmin=vmin, vmax=vmax
            )
            self._kymo_cbar1 = self.fig_kymo.colorbar(self._kymo_im1, ax=ax1, label='V (mV)')
        else:
            self._kymo_im1.set_data(data1)
            self._kymo_im1.set_extent([t[0], t[-1], 0, len(path1)])
            self._kymo_im1.set_clim(vmin=vmin, vmax=vmax)
            if self._kymo_cbar1 is not None:
                self._kymo_cbar1.update_normal(self._kymo_im1)
        ax1.set_ylabel('Compartment (soma → B1 tip)')
        ax1.set_title('Kymograph — Path to Branch 1')

        if self._kymo_im2 is None:
            self._kymo_im2 = ax2.imshow(
                data2, aspect='auto', origin='lower',
                extent=[t[0], t[-1], 0, len(path2)],
                cmap='plasma', vmin=vmin, vmax=vmax
            )
            self._kymo_cbar2 = self.fig_kymo.colorbar(self._kymo_im2, ax=ax2, label='V (mV)')
        else:
            self._kymo_im2.set_data(data2)
            self._kymo_im2.set_extent([t[0], t[-1], 0, len(path2)])
            self._kymo_im2.set_clim(vmin=vmin, vmax=vmax)
            if self._kymo_cbar2 is not None:
                self._kymo_cbar2.update_normal(self._kymo_im2)
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Compartment (soma → B2 tip)')
        ax2.set_title('Kymograph — Path to Branch 2')

        self.fig_kymo.tight_layout(pad=2.5)
        self.cvs_kymo.draw_idle()

    def _update_energy_balance(self, result):
        """Combined Energy & Balance tab with 4 rows: Balance Error, Cumulative Charge, Power, ATP Pool."""
        if not hasattr(self, 'fig_energy'):
            return  # tab not yet visited
        t   = result.t
        dt  = float(t[1] - t[0]) if len(t) > 1 else 0.05

        ax1, ax2, ax3, ax4, ax5 = self.ax_energy

        # ── Row 1: Current Balance Error (semilog) ─────────────────────
        from core.analysis import compute_current_balance
        try:
            I_bal = compute_current_balance(result, result.morph)
            # Ensure I_bal is 1D and matches t shape
            I_bal = np.asarray(I_bal, dtype=float)
            if I_bal.ndim != 1:
                I_bal = I_bal.flatten()
            # If I_bal size doesn't match t, truncate or pad
            if I_bal.size != t.size:
                if I_bal.size > t.size:
                    I_bal = I_bal[:t.size]
                else:
                    I_bal_padded = np.zeros_like(t)
                    I_bal_padded[:I_bal.size] = I_bal
                    I_bal = I_bal_padded
            err = float(np.max(np.abs(I_bal)))
        except Exception as e:
            I_bal = np.zeros_like(t)
            err = 0.0

        if "abs_err" not in self._balance_lines:
            self._balance_lines["abs_err"] = ax1.semilogy([], [], color='#3264DC', lw=1)[0]
        # Final shape check before set_data
        if I_bal.shape != t.shape:
            logging.warning(f"I_bal shape {I_bal.shape} doesn't match t {t.shape}, skipping energy balance error plot")
            self._balance_lines["abs_err"].set_data([], [])
        else:
            self._balance_lines["abs_err"].set_data(t, np.abs(I_bal) + 1e-12)
        ax1.set_ylabel('|Error| (µA/cm²)')
        ax1.set_title(f'Current Balance Error (log) — max|error| = {err:.5f} µA/cm²  '
                      f'{"✓ Good" if err < 0.05 else "⚠ Check solver"}')
        ax1.grid(alpha=0.3)
        ax1.relim()
        ax1.autoscale_view()
        ax1.tick_params(labelbottom=False)

        # ── Row 2: Cumulative Charge Q ─────────────────────────────────
        P_total = np.zeros_like(t)
        active_q = set()
        active_p = set()

        # Safety check for currents dictionary
        if not hasattr(result, 'currents') or not isinstance(result.currents, dict):
            logging.error("SimulationResult missing or invalid currents attribute")
            return

        for name, curr in result.currents.items():
            # Handle 2D current arrays (n_comp, n_time) - sum across compartments
            curr_arr = np.asarray(curr, dtype=float)
            if curr_arr.ndim == 2:
                curr_arr = np.sum(curr_arr, axis=0)
            
            # Ensure curr_arr is 1D and matches t shape
            if curr_arr.ndim != 1:
                curr_arr = curr_arr.flatten()
            
            # If curr_arr size doesn't match t, try to truncate or reshape
            if curr_arr.size != t.size:
                # If curr_arr is larger, truncate to match t
                if curr_arr.size > t.size:
                    curr_arr = curr_arr[:t.size]
                # If curr_arr is smaller, pad with zeros
                else:
                    curr_arr_padded = np.zeros_like(t)
                    curr_arr_padded[:curr_arr.size] = curr_arr
                    curr_arr = curr_arr_padded
            
            # Final shape check - skip if still mismatched
            if curr_arr.shape != t.shape:
                logging.warning(f"Skipping current {name}: shape {curr_arr.shape} doesn't match t {t.shape}")
                continue
            
            color = CHAN_COLORS.get(name, '#888888')
            E_rev = _get_E_rev(name, result.config.channels)
            Q_cum = np.cumsum(np.abs(curr_arr)) * dt
            q_key = f"Q_{name}"
            p_key = f"P_{name}"
            active_q.add(q_key)
            active_p.add(p_key)
            if q_key not in self._energy_lines:
                self._energy_lines[q_key] = ax2.plot([], [], color=color, lw=1.5, label=f'Q_{name}')[0]
            if p_key not in self._energy_lines:
                self._energy_lines[p_key] = ax3.plot([], [], color=color, lw=1, alpha=0.8, label=f'P_{name}')[0]
            Q_cum_safe = _ensure_shape_compatible(Q_cum, t, f"Q_cum_{name}")
            if Q_cum_safe is not None:
                self._energy_lines[q_key].set_data(t, Q_cum_safe)
            else:
                self._energy_lines[q_key].set_data([], [])
            # Ensure v_soma is 1D array compatible with curr_arr
            v_soma = np.asarray(result.v_soma, dtype=float)
            if v_soma.ndim != 1:
                v_soma = v_soma.flatten()
            # Ensure v_soma matches t shape
            if v_soma.shape != t.shape:
                if v_soma.size == t.size:
                    v_soma = v_soma.reshape(t.shape)
                elif v_soma.size > t.size:
                    v_soma = v_soma[:t.size]
                else:
                    v_soma_padded = np.zeros_like(t)
                    v_soma_padded[:v_soma.size] = v_soma
                    v_soma = v_soma_padded
            P = np.abs(curr_arr * (v_soma - E_rev))
            P_safe = _ensure_shape_compatible(P, t, f"P_{name}")
            if P_safe is not None:
                self._energy_lines[p_key].set_data(t, P_safe)
                P_total += P_safe
            else:
                self._energy_lines[p_key].set_data([], [])

        for key, line in self._energy_lines.items():
            if key.startswith("Q_") and key not in active_q:
                line.set_data([], [])
            if key.startswith("P_") and key not in active_p and key != "P_total":
                line.set_data([], [])

        ax2.set_ylabel('Cumulative charge (nC/cm²)')
        ax2.set_title('Energy — Cumulative ionic charge transfer')
        ax2.legend(fontsize=7, loc='upper left', bbox_to_anchor=(0, 1), framealpha=0.8)
        ax2.grid(alpha=0.3)
        ax2.relim()
        ax2.autoscale_view()
        ax2.tick_params(labelbottom=False)

        # ── Row 3: Instantaneous Power P ────────────────────────────────
        if "P_total" not in self._energy_lines:
            self._energy_lines["P_total"] = ax3.plot([], [], 'k-', lw=2, label='Total', zorder=5)[0]
        P_total_safe = _ensure_shape_compatible(P_total, t, "P_total")
        if P_total_safe is not None:
            self._energy_lines["P_total"].set_data(t, P_total_safe)
        else:
            self._energy_lines["P_total"].set_data([], [])

        ax3.set_xlabel('Time (ms)')
        ax3.set_ylabel('Power (µW/cm²)')
        ax3.set_title(f'Instantaneous power   ATP ≈ {result.atp_estimate:.3e} nmol/cm²')
        ax3.legend(fontsize=7, loc='upper left', bbox_to_anchor=(0, 1), framealpha=0.8)
        ax3.grid(alpha=0.3)
        ax3.relim()
        ax3.autoscale_view()
        ax3.tick_params(labelbottom=False)

        # ── Row 4: ATP Pool Time Series ─────────────────────────────────
        # Extract ATP from state vector if dynamic ATP is enabled
        atp_data = None
        if hasattr(result, 'y_all') and result.config.metabolism.enable_dynamic_atp:
            # Find ATP offset (last variable in state vector)
            n_comp = result.n_comp
            # ATP is the last variable, so it's at the end of y_all
            # For soma-only, it's the last element; for multi-comp, it's the last n_comp elements
            if n_comp == 1:
                atp_data = result.y_all[-1, :]  # Last row is ATP for single comp
            else:
                # Use soma ATP (index 0 of the last n_comp elements)
                atp_data = result.y_all[-n_comp:, :]  # Last n_comp rows are ATP
                atp_data = atp_data[0, :]  # Soma ATP

        if atp_data is not None:
            if self._atp_line is None:
                self._atp_line = ax4.plot([], [], color='#A6E3A1', lw=2, label='[ATP]i')[0]
            if self._atp_threshold_line is None:
                self._atp_threshold_line = ax4.axhline(y=0.5, color='#FF6B6B', linestyle='--', lw=1.5, label='Ischemic Threshold')

            atp_data_safe = _ensure_shape_compatible(atp_data, t, "atp_data")
            if atp_data_safe is not None:
                self._atp_line.set_data(t, atp_data_safe)
                ax4.set_ylim(0, max(3.0, np.max(atp_data_safe) * 1.1))
            else:
                self._atp_line.set_data([], [])
            ax4.set_ylabel('[ATP]i (mM)')
            ax4.set_xlabel('Time (ms)')
            ax4.set_title('Intracellular ATP Pool (Metabolic Breath)')
            ax4.legend(fontsize=8, loc='upper right')
            ax4.grid(alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Enable dynamic ATP in config\nto see metabolic dynamics',
                     ha='center', va='center', transform=ax4.transAxes, fontsize=10, color='gray')
            ax4.set_xlabel('Time (ms)')
            ax4.set_ylabel('[ATP]i (mM)')
            ax4.set_title('Intracellular ATP Pool (Disabled)')
        
        try:
            self.fig_energy.tight_layout()
        except Exception:
            pass  # Skip tight_layout if axes are incompatible
        self.cvs_energy.draw_idle()

        # Add crosshair and zoom to time-series axes only (not pie chart)
        for i, ax in enumerate(self.ax_energy[:4]):  # First 4 axes (time-series plots)
            if hasattr(ax, 'crosshair'):
                continue  # Already added
            # Add crosshair cursor
            from matplotlib.widgets import Cursor
            ax.crosshair = Cursor(ax, useblit=True, color='red', linewidth=0.5, linestyle='--')
            # Enable zoom
            ax.set_navigate(True)

        # ── Row 5: ATP Breakdown Pie Chart (top-right corner) ───────
        atp_bd = getattr(result, 'atp_breakdown', None)
        if atp_bd is None or not isinstance(atp_bd, dict):
            atp_bd = {}
        na_pump = atp_bd.get('Na_pump', 0.0)
        ca_pump = atp_bd.get('Ca_pump', 0.0)
        baseline = atp_bd.get('baseline', 0.0)
        total = atp_bd.get('total', 0.0)

        # Use standard channel colors, highlight Ca_pump in red if > 30% (metabolic stress)
        ca_ratio = ca_pump / total if total > 0 else 0.0
        if ca_ratio > 0.3:
            # Metabolic stress: highlight Ca2+ pump in bright red
            colors = [CHAN_COLORS.get('Na', '#FF6B6B'), '#FF0000', '#888888']
        else:
            colors = [CHAN_COLORS.get('Na', '#FF6B6B'), CHAN_COLORS.get('Ca', '#4ECDC4'), '#888888']
        labels = ['Na+ Pump', 'Ca2+ Pump', 'Resting Leakage']
        sizes = [na_pump, ca_pump, baseline]

        # Clear previous pie chart
        if self._pie_chart is not None:
            self._pie_chart.remove()
            self._pie_chart = None

        # Create new pie chart in top-right corner
        if total > 0:
            title_suffix = ' (METABOLIC STRESS)' if ca_ratio > 0.3 else ''
            self._pie_chart = ax5.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                      startangle=90, textprops={'fontsize': 9})[0]
        else:
            ax5.text(0.5, 0.5, 'No ATP data', ha='center', va='center', transform=ax5.transAxes)

        ax5.set_title(f'ATP Breakdown (Total: {total:.3e} nmol/cm²){title_suffix}', fontsize=10)
        ax5.axis('equal')

        try:
            self.fig_energy.tight_layout()
        except Exception:
            pass  # Skip tight_layout if axes are incompatible
        self.cvs_energy.draw_idle()

    # ─────────────────────────────────────────────────────────────────
    #  16 — SPIKE SHAPE OVERLAY
    # ─────────────────────────────────────────────────────────────────
    def _update_spike_shape(self, result, stats: dict):
        """Overlay selected spikes with color coding to show spike shape evolution."""
        if not hasattr(self, 'fig_spike_shape'):
            return  # tab not yet visited
        from core.analysis import detect_spikes

        t = np.asarray(result.t, dtype=float)
        v = np.asarray(result.v_soma, dtype=float)
        ax = self.ax_spike_shape

        kwargs = _spike_detect_kwargs_from_stats(stats)
        peak_idx, spike_times, _ = detect_spikes(v, t, **kwargs)
        n_sp = len(spike_times)

        # Store spike data for selection controls
        self._spike_shape_data = (t, v, spike_times, peak_idx)

        # Update total spike count label
        self._spike_shape_total.setText(str(n_sp))
        self._spike_shape_total.setStyleSheet("color:#CBA6F7; font-size:11px;")

        # Update range spinbox limits
        self._spike_shape_start.blockSignals(True)
        self._spike_shape_end.blockSignals(True)
        self._spike_shape_start.setMaximum(max(1, n_sp))
        self._spike_shape_end.setMaximum(max(1, n_sp))
        if n_sp > 0:
            self._spike_shape_end.setValue(min(self._spike_shape_end.value(), n_sp))
        self._spike_shape_start.blockSignals(False)
        self._spike_shape_end.blockSignals(False)

        if n_sp == 0:
            ax.clear()
            ax.text(0.5, 0.5, 'No spikes detected', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            self.cvs_spike_shape.draw_idle()
            return

        # Get selection range
        start_idx = self._spike_shape_start.value() - 1  # Convert to 0-based
        end_idx = self._spike_shape_end.value()  # Exclusive
        start_idx = max(0, min(start_idx, n_sp - 1))
        end_idx = max(start_idx + 1, min(end_idx, n_sp))

        # Check for step selection (Every Nth)
        quick_text = self._spike_shape_quick.currentText()
        if quick_text == "Every 10th":
            step = getattr(self, '_spike_shape_step', 10)
            selected_indices = list(range(start_idx, end_idx, step))
        elif quick_text == "Every 5th":
            step = getattr(self, '_spike_shape_step', 5)
            selected_indices = list(range(start_idx, end_idx, step))
        else:
            selected_indices = list(range(start_idx, end_idx))

        # Extract spike windows: -2ms to +5ms around each peak
        window_ms = 7.0
        pre_ms = 2.0
        dt = float(np.mean(np.diff(t))) if len(t) > 1 else 0.1
        window_samples = int(window_ms / dt)
        pre_samples = int(pre_ms / dt)

        spikes = []
        spike_indices = []
        for i in selected_indices:
            idx = peak_idx[i]
            start = max(0, idx - pre_samples)
            end = min(len(t), idx + (window_samples - pre_samples))
            spike_t = t[start:end] - t[idx]
            spike_v = v[start:end]
            spikes.append((spike_t, spike_v))
            spike_indices.append(i + 1)  # 1-based for display

        n_selected = len(spikes)
        color_by_index = self._spike_shape_color_by_index.isChecked()

        ax.clear()
        if color_by_index:
            # Color by absolute spike index (shows evolution)
            colors = plt.cm.viridis(np.linspace(0, 1, n_sp))
            for i, (spike_t, spike_v) in enumerate(spikes):
                abs_idx = spike_indices[i] - 1
                ax.plot(spike_t, spike_v, color=colors[abs_idx], lw=1.5, alpha=0.7, label=f'Spike {spike_indices[i]}')
        else:
            # Single color for all selected spikes
            soma_color = PLOT_THEMES.get("Default", PLOT_THEMES["Default"]).get("soma", "#4080FF")
            for i, (spike_t, spike_v) in enumerate(spikes):
                ax.plot(spike_t, spike_v, color=soma_color, lw=1.5, alpha=0.7, label=f'Spike {spike_indices[i]}')

        _configure_ax_interactive(
            ax,
            title=f'Spike Shape Overlay (Showing {n_selected} of {n_sp})',
            xlabel='Time relative to peak (ms)',
            ylabel='V (mV)',
            show_legend=False if n_selected > 10 else True,
        )
        ax.grid(alpha=0.2)

        self.cvs_spike_shape.draw_idle()

    # ─────────────────────────────────────────────────────────────────
    #  17 — POINCARÉ PLOT (ISI DYNAMICS)
    # ─────────────────────────────────────────────────────────────────
    def _update_poincare(self, result, stats: dict):
        """Poincaré plot of ISI dynamics: ISI[n+1] vs ISI[n]."""
        if not hasattr(self, 'fig_poincare'):
            return  # tab not yet visited
        from core.analysis import detect_spikes

        t = np.asarray(result.t, dtype=float)
        v = np.asarray(result.v_soma, dtype=float)
        ax = self.ax_poincare

        kwargs = _spike_detect_kwargs_from_stats(stats)
        peak_idx, spike_times, _ = detect_spikes(v, t, **kwargs)
        n_sp = len(spike_times)

        if n_sp < 3:
            ax.clear()
            ax.text(0.5, 0.5, 'Need ≥3 spikes for Poincaré plot', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            self.cvs_poincare.draw_idle()
            return

        # Calculate ISIs
        isi = np.diff(spike_times)

        # Poincaré plot: ISI[n+1] vs ISI[n]
        isi_n = isi[:-1]
        isi_n_plus_1 = isi[1:]

        ax.clear()
        ax.scatter(isi_n, isi_n_plus_1, c=range(len(isi_n)), cmap='viridis', s=30, alpha=0.7, edgecolors='k', linewidth=0.5)

        # Diagonal line (ISI[n+1] = ISI[n])
        isi_min = min(isi_n.min(), isi_n_plus_1.min())
        isi_max = max(isi_n.max(), isi_n_plus_1.max())
        ax.plot([isi_min, isi_max], [isi_min, isi_max], 'r--', lw=1.5, alpha=0.5, label='ISI[n+1] = ISI[n]')

        _configure_ax_interactive(
            ax,
            title=f'Poincaré Plot (ISI Dynamics, N={n_sp})',
            xlabel='ISI[n] (ms)',
            ylabel='ISI[n+1] (ms)',
            show_legend=True,
        )
        ax.grid(alpha=0.2)
        ax.set_aspect('equal', adjustable='datalim')

        self.cvs_poincare.draw_idle()

    # ─────────────────────────────────────────────────────────────────
    #  12 — SPECTROGRAM  (STFT of soma Vm)
    # ─────────────────────────────────────────────────────────────────
    def _update_spectrogram(self, result):
        if not hasattr(self, 'fig_spectro'):
            return  # tab not yet visited
        from scipy.signal import stft
        t  = result.t
        v  = result.v_soma
        ax_v, ax_sp = self.ax_spectro

        # Top: raw Vm trace
        if self._spectro_vm_line is None:
            self._spectro_vm_line = ax_v.plot([], [], color='#2060C0', lw=0.8)[0]
        v_safe = _ensure_shape_compatible(v, t, "v_soma_spectro")
        if v_safe is not None:
            self._spectro_vm_line.set_data(t, v_safe)
        else:
            self._spectro_vm_line.set_data([], [])
        ax_v.set_ylabel('V (mV)', fontsize=9)
        ax_v.set_title('Membrane potential — soma', fontsize=10)
        ax_v.grid(alpha=0.25)
        ax_v.set_xlim(t[0], t[-1])
        ax_v.relim()
        ax_v.autoscale_view()

        # STFT — fs in Hz, dt in ms → fs = 1000/dt
        dt_ms = float(t[1] - t[0]) if len(t) > 1 else 0.05
        fs_hz = 1000.0 / dt_ms

        # Adaptive window: ~10 ms or 256 samples, whichever is smaller
        n_seg = min(256, max(32, int(10.0 / dt_ms)))
        n_overlap = n_seg * 3 // 4

        try:
            freqs, times_stft, Zxx = stft(v, fs=fs_hz, nperseg=n_seg,
                                           noverlap=n_overlap, window='hann')
            # Convert STFT output times to simulation time axis
            t_stft = t[0] + times_stft * 1000.0  # seconds → ms

            power_db = 10.0 * np.log10(np.abs(Zxx) ** 2 + 1e-12)

            # Limit display to biologically relevant range: 0–1000 Hz
            f_max_disp = min(1000.0, freqs[-1])
            freq_mask = freqs <= f_max_disp

            x = t_stft
            y = freqs[freq_mask]
            z = power_db[freq_mask, :]
            vmin = float(np.percentile(z, 5))
            vmax = float(np.percentile(z, 99))
            if self._spectro_fail_text is None:
                self._spectro_fail_text = ax_sp.text(
                    0.5, 0.5, '', ha='center', va='center', transform=ax_sp.transAxes, visible=False
                )
            self._spectro_fail_text.set_visible(False)
            if self._spectro_mesh is None:
                self._spectro_mesh = ax_sp.imshow(
                    z,
                    aspect='auto',
                    origin='lower',
                    extent=[x[0], x[-1], y[0], y[-1]],
                    cmap='inferno',
                    vmin=vmin,
                    vmax=vmax,
                )
                self._spectro_cbar = self.fig_spectro.colorbar(
                    self._spectro_mesh, ax=ax_sp, label='Power (dB)', pad=0.02
                )
            else:
                self._spectro_mesh.set_data(z)
                self._spectro_mesh.set_extent([x[0], x[-1], y[0], y[-1]])
                self._spectro_mesh.set_clim(vmin=vmin, vmax=vmax)
                if self._spectro_cbar is not None:
                    self._spectro_cbar.update_normal(self._spectro_mesh)
            ax_sp.set_ylabel('Frequency (Hz)', fontsize=9)
        except Exception:
            if self._spectro_fail_text is None:
                self._spectro_fail_text = ax_sp.text(
                    0.5, 0.5, 'STFT unavailable\n(scipy missing?)',
                    ha='center', va='center', transform=ax_sp.transAxes
                )
            else:
                self._spectro_fail_text.set_text('STFT unavailable\n(scipy missing?)')
            self._spectro_fail_text.set_visible(True)

        ax_sp.set_xlabel('Time (ms)', fontsize=9)
        ax_sp.set_title('STFT spectrogram (V_soma)', fontsize=10)
        ax_sp.set_xlim(t[0], t[-1])

        self.fig_spectro.tight_layout(pad=2.5)
        self.cvs_spectro.draw_idle()

    # ─────────────────────────────────────────────────────────────────
    #  8 — BIFURCATION
    # ─────────────────────────────────────────────────────────────────
    def _ensure_built(self, builder_name: str):
        """Force initialization of a lazy tab by its builder name."""
        for i in range(self.count()):
            widget = self.widget(i)
            if isinstance(widget, _LazyPlaceholder):
                title = self.tabText(i)
                for spec in self._all_tab_specs.values():
                    if spec['title'] == title and spec['builder'] == builder_name:
                        self._on_tab_changed(i)
                        return

    def update_bifurcation(self, bif_data: list, param_name: str):
        self._ensure_built('_build_tab_bif')
        self._last_bif_data = bif_data
        self._last_bif_param_name = param_name
        vals   = np.array([d['val']   for d in bif_data])
        vmax   = np.array([d['max']   for d in bif_data])
        vmin   = np.array([d['min']   for d in bif_data])
        freq   = np.array([d['freq']  for d in bif_data])
        n_sp   = np.array([d['n_sp']  for d in bif_data])

        ax1, ax2, ax3, ax4 = self.ax_bif
        if not self._bif_lines:
            self._bif_lines["max_fallback"] = ax1.plot([], [], 'r.', ms=4)[0]
            self._bif_lines["peaks"] = ax1.plot([], [], linestyle='None', marker='.', color='b', ms=4)[0]
            self._bif_lines["vmax"] = ax2.plot([], [], 'r-', lw=1.5, label='Vmax')[0]
            self._bif_lines["vmin"] = ax2.plot([], [], 'b-', lw=1.5, label='Vmin')[0]
            self._bif_lines["freq"] = ax3.plot([], [], 'g.-', lw=1.5)[0]
            self._bif_lines["n_sp"] = ax4.plot([], [], 'b.-', lw=1.5)[0]

        peak_x, peak_y = [], []
        fallback_x, fallback_y = [], []
        for d in bif_data:
            pks = d.get('peaks', [])
            if pks:
                peak_x.extend([d['val']] * len(pks))
                peak_y.extend(pks)
            else:
                fallback_x.append(d['val'])
                fallback_y.append(d['max'])

        self._bif_lines["peaks"].set_data(peak_x, peak_y)
        self._bif_lines["max_fallback"].set_data(fallback_x, fallback_y)

        ax1.set_xlabel(param_name);  ax1.set_ylabel('V peaks (mV)')
        ax1.set_title('Bifurcation diagram');  ax1.grid(alpha=0.3)
        ax1.relim(); ax1.autoscale_view()

        self._bif_lines["vmax"].set_data(vals, vmax)
        self._bif_lines["vmin"].set_data(vals, vmin)
        ax2.set_xlabel(param_name);  ax2.set_ylabel('V (mV)')
        ax2.set_title('Vmax / Vmin');  ax2.legend();  ax2.grid(alpha=0.3)
        ax2.relim(); ax2.autoscale_view()

        self._bif_lines["freq"].set_data(vals, freq)
        ax3.set_xlabel(param_name);  ax3.set_ylabel('f (Hz)')
        ax3.set_title('Firing frequency');  ax3.grid(alpha=0.3)
        ax3.relim(); ax3.autoscale_view()

        self._bif_lines["n_sp"].set_data(vals, n_sp)
        ax4.set_xlabel(param_name);  ax4.set_ylabel('N spikes')
        ax4.set_title('Spike count');  ax4.grid(alpha=0.3)
        ax4.relim(); ax4.autoscale_view()

        self.fig_bif.tight_layout()
        self.cvs_bif.draw_idle()
        self.setCurrentWidget(self.tab_bif)

    # ─────────────────────────────────────────────────────────────────
    #  9 — SWEEP
    # ─────────────────────────────────────────────────────────────────
    def update_sweep(self, sweep_results: list, param_name: str):
        """sweep_results: list of (param_value, SimulationResult|None)"""
        self._ensure_built('_build_tab_sweep')
        self._last_sweep_results = sweep_results
        self._last_sweep_param_name = param_name
        from core.analysis import detect_spikes

        ax1, ax2, ax3, ax4 = self.ax_sweep

        n = len(sweep_results)
        cmap = plt.colormaps['plasma'](np.linspace(0.1, 0.9, n))

        param_vals, peaks, freqs, n_sps = [], [], [], []
        trace_count = min(n, self._sweep_trace_max)
        while len(self._sweep_trace_lines) < trace_count:
            self._sweep_trace_lines.append(ax1.plot([], [], lw=1, alpha=0.8)[0])

        used_trace_count = 0
        for i, (val, res) in enumerate(sweep_results):
            if res is None:
                continue
            param_vals.append(val)
            if used_trace_count < trace_count:
                trace_line = self._sweep_trace_lines[used_trace_count]
                trace_line.set_color(cmap[i])
                trace_line.set_data(res.t, res.v_soma)
                trace_line.set_visible(True)
                used_trace_count += 1
            pks, sp_t, sp_amp = detect_spikes(
                res.v_soma, res.t, **_spike_detect_kwargs_from_analysis(res.config.analysis)
            )
            peaks.append(float(np.max(res.v_soma)))
            n_sps.append(len(pks))
            freqs.append(1000.0 / float(np.mean(np.diff(sp_t))) if len(sp_t) > 1 else 0.0)
        for i in range(used_trace_count, len(self._sweep_trace_lines)):
            self._sweep_trace_lines[i].set_data([], [])
            self._sweep_trace_lines[i].set_visible(False)

        ax1.set_xlabel('Time (ms)');  ax1.set_ylabel('V (mV)')
        ax1.set_title(f'Sweep traces  [{param_name}]')

        if sweep_results:
            v0 = sweep_results[0][0]
            v1 = sweep_results[-1][0]
            if self._sweep_cbar is None:
                sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(v0, v1))
                self._sweep_cbar = self.fig_sweep.colorbar(sm, ax=ax1, label=param_name)
            else:
                self._sweep_cbar.mappable.set_norm(plt.Normalize(v0, v1))
                self._sweep_cbar.set_label(param_name)
                self._sweep_cbar.update_normal(self._sweep_cbar.mappable)

        if "peaks" not in self._sweep_metric_lines:
            self._sweep_metric_lines["peaks"] = ax2.plot([], [], 'r.-', lw=1.5)[0]
            self._sweep_metric_lines["freqs"] = ax3.plot([], [], 'g.-', lw=1.5)[0]
            self._sweep_metric_lines["n_sps"] = ax4.plot([], [], 'b.-', lw=1.5)[0]
        self._sweep_metric_lines["peaks"].set_data(param_vals, peaks)
        ax2.set_xlabel(param_name);  ax2.set_ylabel('V_peak (mV)')
        ax2.set_title('Peak voltage vs parameter');  ax2.grid(alpha=0.3)
        ax2.relim(); ax2.autoscale_view()

        self._sweep_metric_lines["freqs"].set_data(param_vals, freqs)
        ax3.set_xlabel(param_name);  ax3.set_ylabel('f (Hz)')
        ax3.set_title('Firing rate (f-I curve)');  ax3.grid(alpha=0.3)
        ax3.relim(); ax3.autoscale_view()

        self._sweep_metric_lines["n_sps"].set_data(param_vals, n_sps)
        ax4.set_xlabel(param_name);  ax4.set_ylabel('N spikes')
        ax4.set_title('Spike count');  ax4.grid(alpha=0.3)
        ax4.relim(); ax4.autoscale_view()

        self.fig_sweep.tight_layout()
        self.cvs_sweep.draw_idle()
        self.setCurrentWidget(self.tab_sweep)

    # ─────────────────────────────────────────────────────────────────
    #  10 — S-D CURVE
    # ─────────────────────────────────────────────────────────────────
    def update_sd_curve(self, sd: dict):
        self._ensure_built('_build_tab_sd')
        self._last_sd = sd
        dur  = sd['durations']
        I_th = sd['I_threshold']
        I_rh = sd['rheobase']
        t_ch = sd['chronaxie']
        weiss = sd['weiss_fit']
        Q_th  = sd['Q_threshold']

        ax1, ax2 = self.ax_sd
        if not self._sd_lines:
            self._sd_lines["ith"] = ax1.plot([], [], 'b.-', lw=2, ms=8, label='I_threshold')[0]
            self._sd_lines["weiss"] = ax1.plot([], [], 'r--', lw=1.5, label="Weiss fit")[0]
            self._sd_lines["rh"] = ax1.axhline(I_rh, color='gray', ls=':', lw=1.5,
                                               label=f'Rheobase = {I_rh:.2f} µA/cm²')
            self._sd_lines["ch"] = ax1.axvline(0.0, color='orange', ls='--', lw=1.5)
            self._sd_lines["ch_point"] = ax1.plot([], [], 'go', ms=10, zorder=5)[0]
            self._sd_lines["qth"] = ax2.plot([], [], 'm.-', lw=2, ms=8, label='Q = I·t')[0]

        self._sd_lines["ith"].set_data(dur, I_th)
        if weiss is not None:
            self._sd_lines["weiss"].set_data(dur, weiss)
        else:
            self._sd_lines["weiss"].set_data([], [])
        self._sd_lines["rh"].set_ydata([I_rh, I_rh])
        if not np.isnan(t_ch):
            self._sd_lines["ch"].set_xdata([t_ch, t_ch])
            self._sd_lines["ch_point"].set_data([t_ch], [2 * I_rh])
            self._sd_lines["ch"].set_visible(True)
            self._sd_lines["ch_point"].set_visible(True)
        else:
            self._sd_lines["ch"].set_visible(False)
            self._sd_lines["ch_point"].set_visible(False)
        ax1.set_xlabel('Pulse duration (ms)');  ax1.set_ylabel('I threshold (µA/cm²)')
        ax1.set_title('Strength-Duration Curve');  ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)
        ax1.relim(); ax1.autoscale_view()

        self._sd_lines["qth"].set_data(dur, Q_th)
        ax2.set_xlabel('Pulse duration (ms)');  ax2.set_ylabel('Charge threshold (nC/cm²)')
        ax2.set_title('Minimum charge vs duration')
        ax2.legend();  ax2.grid(alpha=0.3)
        ax2.relim(); ax2.autoscale_view()

        self.fig_sd.tight_layout()
        self.cvs_sd.draw_idle()
        self.setCurrentWidget(self.tab_sd)

    # ─────────────────────────────────────────────────────────────────
    #  11 — EXCITABILITY MAP
    # ─────────────────────────────────────────────────────────────────
    def update_excmap(self, exc: dict):
        self._ensure_built('_build_tab_excmap')
        self._last_exc = exc
        I_r  = exc['I_range']
        d_r  = exc['dur_range']
        S    = exc['spike_matrix']
        F    = exc['freq_matrix']

        ax1, ax2 = self.ax_excmap

        if self._excmap_mesh["spikes"] is None:
            self._excmap_mesh["spikes"] = ax1.pcolormesh(d_r, I_r, S, cmap='Blues', shading='auto')
            self._excmap_cbar["spikes"] = self.fig_excmap.colorbar(
                self._excmap_mesh["spikes"], ax=ax1, label='N spikes'
            )
        else:
            self._excmap_mesh["spikes"].set_array(S.ravel())
            if self._excmap_cbar["spikes"] is not None:
                self._excmap_cbar["spikes"].update_normal(self._excmap_mesh["spikes"])
        ax1.set_xlabel('Duration (ms)');  ax1.set_ylabel('I_ext (µA/cm²)')
        ax1.set_title('Spike count map')

        # Mask zero-frequency cells
        F_masked = np.where(F > 0, F, np.nan)
        if self._excmap_mesh["freq"] is None:
            self._excmap_mesh["freq"] = ax2.pcolormesh(d_r, I_r, F_masked, cmap='hot', shading='auto')
            self._excmap_cbar["freq"] = self.fig_excmap.colorbar(
                self._excmap_mesh["freq"], ax=ax2, label='f (Hz)'
            )
        else:
            self._excmap_mesh["freq"].set_array(F_masked.ravel())
            if self._excmap_cbar["freq"] is not None:
                self._excmap_cbar["freq"].update_normal(self._excmap_mesh["freq"])
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
