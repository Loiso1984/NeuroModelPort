"""
gui/analytics.py - Full Scientific Analytics Suite v10.1

Analytical tabs using matplotlib embedded in Qt:
  0. Neuron Passport     - rich biophysical report
  1. Oscilloscope detail - multi-compartment traces
  2. Gate Dynamics       - m, h, n, r, s, u vs time
  3. Equilibrium Curves  - x_inf(V), tau(V) for all gates
  4. Phase Plane         - V vs n + nullclines
  5. Kymograph           - spatiotemporal V(x,t) heatmap
  6. Current Balance     - Cm*dV/dt - (I_stim - I_ion + I_ax)
  7. Energy / Power      - cumulative charge & instantaneous power
  8. Bifurcation         - spike peaks vs parameter
  9. Sweep               - traces + f-I curve
 10. S-D Curve           - strength-duration + Weiss fit
 11. Excitability Map    - 2-D heatmap (I x duration)
 12. Spectrogram         - STFT of soma Vm
 13. Impedance Z(f)      - membrane frequency response (|Z|, phase)

Lyapunov computation is explicit via `Compute LLE` action.
"""

import logging
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QTabWidget,
                                QLabel, QTextEdit, QHBoxLayout,
                                QSizePolicy, QScrollArea, QPushButton, QMainWindow, QComboBox)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

import matplotlib
matplotlib.use('QtAgg')

# Configure matplotlib for dark theme (matching application)
matplotlib.rcParams['figure.facecolor'] = '#0D1117'
matplotlib.rcParams['axes.facecolor'] = '#0D1117'
matplotlib.rcParams['axes.edgecolor'] = '#313244'
matplotlib.rcParams['axes.labelcolor'] = '#CDD6F4'
matplotlib.rcParams['text.color'] = '#CDD6F4'
matplotlib.rcParams['xtick.color'] = '#CDD6F4'
matplotlib.rcParams['ytick.color'] = '#CDD6F4'
matplotlib.rcParams['grid.color'] = '#313244'
matplotlib.rcParams['legend.facecolor'] = '#1E1E2E'
matplotlib.rcParams['legend.edgecolor'] = '#313244'

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import griddata
from scipy.stats import expon
from scipy.signal import stft
from gui.text_sanitize import repair_text, repair_widget_tree

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
    'SK':   '#C83296', 'KATP': '#F9E2AF', 'PumpNaK': '#E5C890',
}
GATE_COLORS = {
    'm': '#FF4040', 'h': '#4080FF', 'n': '#40C040',
    'r': '#A040FF', 's': '#FF9000', 'u': '#009090',
    'a': '#FF40A0', 'b': '#80C0FF',
}


def _mpl_fig(nrows=1, ncols=1, tight=True, **kwargs) -> tuple:
    """Create a matplotlib Figure + FigureCanvas pair with dark theme."""
    # Extract figsize from kwargs if provided, otherwise use default
    figsize = kwargs.pop('figsize', (8, 4 * nrows))
    
    # Dark theme colors matching the application
    dark_bg = '#0D1117'
    dark_fg = '#CDD6F4'
    dark_grid = '#313244'
    
    fig = Figure(figsize=figsize, dpi=90, facecolor=dark_bg, edgecolor=dark_bg, **kwargs)
    
    # Apply dark theme to all axes
    def _apply_dark_theme(ax):
        ax.set_facecolor(dark_bg)
        ax.tick_params(colors=dark_fg, which='both')
        ax.xaxis.label.set_color(dark_fg)
        ax.yaxis.label.set_color(dark_fg)
        ax.title.set_color(dark_fg)
        for spine in ax.spines.values():
            spine.set_color(dark_grid)
    
    # Store the theme function for later use
    fig._apply_dark_theme = _apply_dark_theme
    
    if tight:
        try:
            fig.set_layout_engine('tight')
        except Exception:
            fig.set_tight_layout(True)
    canvas = FigureCanvas(fig)
    return fig, canvas


def _set_tight_layout_engine(fig, **kwargs):
    """Matplotlib layout helper that avoids deprecated set_tight_layout usage when possible."""
    try:
        fig.set_layout_engine('tight', **kwargs)
    except Exception:
        fig.set_tight_layout(kwargs or True)


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


def _set_line_data(line, x=None, y=None, *, name: str = "line") -> bool:
    """Safely update a matplotlib Line2D with shape-matched finite arrays."""
    if line is None:
        return False
    if x is None or y is None:
        line.set_data([], [])
        return False
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if x_arr.size == 0 or y_arr.size == 0:
        line.set_data([], [])
        return False
    if x_arr.size != y_arr.size:
        n = min(x_arr.size, y_arr.size)
        logging.warning("%s received mismatched data shapes x=%s y=%s; truncating to %d", name, x_arr.shape, y_arr.shape, n)
        x_arr = x_arr[:n]
        y_arr = y_arr[:n]
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    if not np.any(finite):
        line.set_data([], [])
        return False
    line.set_data(x_arr[finite], y_arr[finite])
    return True


def _set_constant_line(line, value: float, axis: str = "y") -> bool:
    """Safely update an axhline/axvline artist."""
    if line is None or not np.isfinite(value):
        if line is not None:
            line.set_visible(False)
        return False
    if axis == "x":
        line.set_data([value, value], [0.0, 1.0])
    else:
        line.set_data([0.0, 1.0], [value, value])
    line.set_visible(True)
    return True


def _axis_message(ax, cache: dict, key: str, message: str, *, title: str = "", xlabel: str = "", ylabel: str = ""):
    """Persistent empty-state message without clearing the whole axis object."""
    text = cache.get(key)
    if text is None:
        text = ax.text(
            0.5, 0.5, message,
            ha='center', va='center', transform=ax.transAxes,
            fontsize=11, color='#89B4FA',
            bbox=dict(boxstyle='round', facecolor='#1E1E2E', alpha=0.88, edgecolor='#45475A'),
        )
        cache[key] = text
    else:
        text.set_text(message)
        text.set_visible(True)
    _configure_ax_interactive(ax, title=title, xlabel=xlabel, ylabel=ylabel, show_legend=False)


def _hide_axis_message(cache: dict, key: str):
    text = cache.get(key)
    if text is not None:
        text.set_visible(False)


def _set_canvas_margins(fig, *, left=0.08, right=0.97, top=0.95, bottom=0.08, hspace=0.38, wspace=0.24):
    """Prefer manual subplot spacing over repeated tight_layout calls on complex figures."""
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, hspace=hspace, wspace=wspace)


def _tab_with_toolbar(
    canvas,
    fullscreen_callback=None,
    extra_widget=None,
    *,
    scroll_canvas: bool = False,
    min_canvas_height: int | None = None,
    extra_widget_position: str = "below",
) -> QWidget:
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
        btn_fullscreen = QPushButton("Fullscreen")
        btn_fullscreen.setToolTip("Open plot in fullscreen window with crosshair")
        btn_fullscreen.setMaximumWidth(120)
        btn_fullscreen.clicked.connect(fullscreen_callback)
        toolbar_row.addWidget(btn_fullscreen)

    toolbar_row.addStretch()
    lay.addLayout(toolbar_row)

    if min_canvas_height is not None:
        canvas.setMinimumHeight(int(min_canvas_height))
    canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    min_canvas_width = None
    fig = getattr(canvas, "figure", None)
    if fig is not None:
        try:
            min_canvas_width = int(fig.get_figwidth() * fig.dpi)
        except Exception:
            min_canvas_width = None
    if min_canvas_width:
        canvas.setMinimumWidth(min_canvas_width)

    content = QWidget()
    content_lay = QVBoxLayout(content)
    content_lay.setContentsMargins(0, 0, 0, 0)
    content_lay.setSpacing(4)

    if extra_widget is not None and extra_widget_position == "above":
        content_lay.addWidget(extra_widget, 0)
    content_lay.addWidget(canvas, 1 if not scroll_canvas else 0)
    if extra_widget is not None and extra_widget_position != "above":
        content_lay.addWidget(extra_widget, 0)

    if scroll_canvas:
        content_lay.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        if min_canvas_width:
            content.setMinimumWidth(min_canvas_width + 24)
        scroll.setWidget(content)
        lay.addWidget(scroll, 1)
    else:
        lay.addWidget(content, 1)
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
    """Peak-preserving Min-Max Decimation for high-fidelity spike rendering.

    Min-Max Decimation ensures every spike peak is visible by capturing both
    min and max values within each chunk. For AP traces with 1ms peaks,
    this prevents aliasing that occurs with naive stride sampling [::step].
    """
    n = int(len(t))
    if n <= max_points or max_points <= 0:
        return t, y

    # Each chunk contributes 2 points (min and max)
    n_chunks = max_points // 2
    chunk_size = max(1, n // n_chunks)

    # Collect min and max indices from each chunk
    indices = []
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
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

    return t[indices], y[indices]


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
        lbl = QLabel("Click this tab to load the view\nLazy-loaded to keep simulation UI responsive")
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
        self.setWindowTitle(repair_text(f"NeuroModelPort - {title}"))
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
            v_line.set_visible(False)
            h_line.set_visible(False)
            self.crosshair_lines.append((ax, v_line, h_line))
        
        # Connect mouse events and store connection IDs for cleanup
        self._crosshair_cids = [
            self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move),
            self.canvas.mpl_connect('button_press_event', self._on_click),
        ]
        
    def _on_mouse_move(self, event):
        """Handle mouse movement for crosshair display."""
        if event.inaxes is None:
            for _, v_line, h_line in self.crosshair_lines:
                v_line.set_visible(False)
                h_line.set_visible(False)
            self.canvas.draw_idle()
            return
        
        ax = event.inaxes
        x, y = event.xdata, event.ydata
        
        # Update crosshair lines for this axis
        for axis, v_line, h_line in self.crosshair_lines:
            if axis == ax:
                _set_constant_line(v_line, x, axis="x")
                _set_constant_line(h_line, y, axis="y")
            else:
                v_line.set_visible(False)
                h_line.set_visible(False)
        
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
        # Disconnect using stored connection IDs
        if hasattr(self, '_crosshair_cids'):
            for cid in self._crosshair_cids:
                try:
                    self.canvas.mpl_disconnect(cid)
                except Exception:
                    pass


class AnalyticsWidget(QTabWidget):
    """Main analytics widget — updated by MainWindow after each run."""
    
    # v12.2: Signal emitted when user requests LLE computation with full rerun
    computeLleRequested = Signal()

    def __init__(self, parent=None):
        """
        Initialize the AnalyticsWidget instance, setting internal state and constructing the tab UI.
        
        This sets placeholders for the most recent simulation data and cached analysis results, prepares bookkeeping for lazy tab construction and fullscreen viewers, initializes the category-to-tab mapping and figure registry used for fullscreen copying, and creates the initial tab layout by calling `_build_tabs()`.
        
        Parameters:
            parent (QObject | None): Optional Qt parent widget for this widget.
        """
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
            1: 'Single', 2: 'Single', 3: 'Single', 4: 'Single', 16: 'Single', 17: 'Single',
            5: 'Spectral', 6: 'Spectral', 7: 'Spectral',
            8: 'Sweep', 9: 'Sweep', 10: 'Sweep', 11: 'Sweep',
            12: 'Physics', 13: 'Physics', 14: 'Physics', 15: 'Physics', 18: 'Physics',
            19: 'Debug',
        }
        self._all_tab_specs = {}  # Store all tab specs for rebuilding
        self._tab_figures = {}  # Store figures for fullscreen access
        self._time_marker = None  # Store vertical line marker for linked cursor
        self._build_tabs()
        repair_widget_tree(self)
    
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

        viewer = FullscreenPlotViewer(fig_copy, title=f"Analytics - {tab_name}", parent=self)
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

        self._btn_fullscreen = QPushButton("Fullscreen")
        self._btn_fullscreen.setToolTip("Open analytics in a maximized window")
        self._btn_fullscreen.clicked.connect(self.open_fullscreen)
        corner_l.addWidget(self._btn_fullscreen)

        self.setCornerWidget(corner, Qt.Corner.TopRightCorner)

        # â”€â”€ Tab 0: Passport Dashboard â€" Visual biophysical report with radar chart â”€â”€
        self._passport_widget = self._build_tab_passport()
        self.addTab(self._passport_widget, "Passport")

        # â”€â”€ Tabs 1–15: lazy placeholders — MPL canvas built on first visit â”€â”€
        # Each entry: builder (creates attrs + returns QWidget), updater (may be None),
        # needs_stats (updater takes (result, stats)), needs_morph (skip if no morph).
        self._all_tab_specs: dict[int, dict] = {
            1:  {'builder': '_build_tab_spike_mech', 'updater': '_update_spike_mechanism', 'title': 'Spike Mechanism', 'needs_stats': True},
            2:  {'builder': '_build_tab_phase',      'updater': '_update_phase',           'title': 'Phase Plane', 'needs_stats': True},
            3:  {'builder': '_build_tab_chaos',      'updater': '_update_chaos',           'title': 'Chaos & LLE', 'needs_stats': True},
            4:  {'builder': '_build_tab_kymo',       'updater': '_update_kymo',            'title': 'Kymograph'},
            5:  {'builder': '_build_tab_spectro',    'updater': '_update_spectrogram',     'title': 'Spectrogram'},
            6:  {'builder': '_build_tab_impedance',  'updater': '_update_impedance',       'title': 'Impedance'},
            7:  {'builder': '_build_tab_modulation', 'updater': '_update_modulation',      'title': 'Phase-Locking', 'needs_stats': True},
            8:  {'builder': '_build_tab_sweep',      'updater': None,                      'title': 'f-I Curve'},
            9:  {'builder': '_build_tab_sd',         'updater': None,                      'title': 'S-D Curve'},
            10: {'builder': '_build_tab_excmap',     'updater': None,                      'title': 'Excit. Map'},
            11: {'builder': '_build_tab_bif',        'updater': None,                      'title': 'Bifurcation'},
            12: {'builder': '_build_tab_currents',   'updater': '_update_currents',        'title': 'Currents'},
            13: {'builder': '_build_tab_gates',      'updater': '_update_gates',           'title': 'Gates'},
            14: {'builder': '_build_tab_energy_balance', 'updater': '_update_energy_balance', 'title': 'Energy & Balance', 'needs_morph': True},
            15: {'builder': '_build_tab_spike_shape', 'updater': '_update_spike_shape', 'title': 'Spike Shape', 'needs_stats': True},
            16: {'builder': '_build_tab_poincare',    'updater': '_update_poincare',       'title': 'Poincare (ISI)', 'needs_stats': True},
            17: {'builder': '_build_tab_isi_dist',    'updater': '_update_isi_dist',       'title': 'ISI Distribution', 'needs_stats': True},
            18: {'builder': '_build_tab_fingerprint', 'updater': '_update_fingerprint',   'title': 'Fingerprint', 'needs_stats': True},
            19: {'builder': '_build_tab_csd',         'updater': '_update_csd',            'title': 'CSD', 'needs_morph': True},
            20: {'builder': '_build_tab_debug',       'updater': '_update_debug',          'title': 'Debug LTE'},
            21: {'builder': '_build_tab_metabolic',   'updater': '_update_metabolic',      'title': 'Metabolic', 'needs_stats': True},
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
            'Spike Mechanism': 'Spike Mechanism',
            'Energy & Balance': 'Energy & Balance',
        }
        fig_name = tab_to_fig.get(tab_title)
        if fig_name and fig_name in self._tab_figures:
            fig = self._tab_figures[fig_name]
            if fig:
                if not hasattr(self, '_tab_error_texts'):
                    self._tab_error_texts = {}
                for idx, ax in enumerate(fig.axes):
                    key = f"{fig_name}_{idx}_missing"
                    _axis_message(
                        ax,
                        self._tab_error_texts,
                        key,
                        f"Warning: {missing_data} required",
                        title="Data unavailable",
                    )
                fig.canvas.draw_idle()
    
    def _show_updater_error_message(self, tab_title: str, error: str):
        """Show error message on tab when updater fails."""
        tab_to_fig = {
            'Spike Mechanism': 'Spike Mechanism',
            'Energy & Balance': 'Energy & Balance',
        }
        fig_name = tab_to_fig.get(tab_title)
        if fig_name and fig_name in self._tab_figures:
            fig = self._tab_figures[fig_name]
            if fig:
                if not hasattr(self, '_tab_error_texts'):
                    self._tab_error_texts = {}
                for idx, ax in enumerate(fig.axes):
                    key = f"{fig_name}_{idx}_update_error"
                    _axis_message(
                        ax,
                        self._tab_error_texts,
                        key,
                        f"Warning: update failed\n{error}",
                        title="Updater error",
                    )
                fig.canvas.draw_idle()
    
    def _on_tab_changed(self, index: int):
        """
        Builds and inserts a real tab widget the first time a lazy placeholder tab is selected.
        
        If the widget at `index` is a `_LazyPlaceholder`, call its configured builder to create
        the real tab, replace the placeholder in-place, force redraw of any nested
        FigureCanvas widgets, and (if available) run the tab's updater using the most
        recent simulation result and stats. Guards against re-entrancy, displays a
        user-facing error widget if the builder fails, and shows figure-level messages
        when required data (morphology or stats) is missing before running the updater.
        
        Parameters:
            index (int): The tab index that was selected.
        """
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
            self.setCurrentIndex(index)
            repair_widget_tree(new_widget)
            
            # Force a geometric refresh and draw
            new_widget.show()
            
            # Helper to recursively find FigureCanvas inside nested layouts (like QScrollArea)
            def _force_draw(w):
                """
                Recursively call `draw_idle` on a widget and its descendant widgets that expose that method.
                
                Parameters:
                    w (QWidget): Root widget (or object) to refresh. For `w` and each descendant that implements a callable `draw_idle`, this function invokes `draw_idle` to schedule a repaint.
                """
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
            error_widget = QLabel(f"Tab build failed:\n{str(e)}")
            error_widget.setStyleSheet("QLabel { color: #dc3545; padding: 20px; }")
            error_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
            repair_widget_tree(error_widget)
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
        """Highlight a time-point across built analytics tabs (linked cursor)."""
        if self._last_result is None:
            return

        t_arr = np.asarray(self._last_result.t, dtype=float)
        if len(t_arr) == 0:
            return
        idx = int(np.argmin(np.abs(t_arr - t_ms)))

        # ── Phase Plane: yellow ghost dot ────────────────────────────
        if hasattr(self, 'ax_phase') and hasattr(self, 'fig_phase'):
            # Use the same gate trace that the phase plane is currently showing
            phase_data = getattr(self, '_phase_full_data', None)
            if phase_data is not None:
                ph_t, ph_V, ph_gate, _cfg, _I, _st, _gk = phase_data
                v_val = np.interp(t_ms, ph_t, ph_V)
                g_val = np.interp(t_ms, ph_t, ph_gate)
                if hasattr(self, '_phase_highlight_dot') and self._phase_highlight_dot is not None:
                    try:
                        self._phase_highlight_dot[0].remove()
                    except Exception:
                        pass
                self._phase_highlight_dot = self.ax_phase.plot(
                    v_val, g_val,
                    'o', color='#F9E2AF', markersize=10,
                    markeredgecolor='black', markeredgewidth=1.5,
                    zorder=10,
                )
                if hasattr(self, 'cvs_phase'):
                    self.cvs_phase.draw_idle()

        # ── Currents tab: vertical line ──────────────────────────────
        if hasattr(self, 'ax_currents'):
            if hasattr(self, '_currents_time_marker') and self._currents_time_marker is not None:
                try:
                    self._currents_time_marker.remove()
                except Exception:
                    pass
            self._currents_time_marker = self.ax_currents.axvline(
                x=t_ms, color='#89B4FA', linestyle='--', linewidth=1.4, alpha=0.85
            )
            if hasattr(self, 'cvs_currents'):
                self.cvs_currents.draw_idle()

        # ── Gates tab: vertical line ─────────────────────────────────
        if hasattr(self, 'ax_gates'):
            if hasattr(self, '_gates_time_marker') and self._gates_time_marker is not None:
                try:
                    self._gates_time_marker.remove()
                except Exception:
                    pass
            self._gates_time_marker = self.ax_gates.axvline(
                x=t_ms, color='#89B4FA', linestyle='--', linewidth=1.4, alpha=0.85
            )
            if hasattr(self, 'cvs_gates'):
                self.cvs_gates.draw_idle()

        # ── Spike Shape tab: nearest spike vertical marker ───────────
        if hasattr(self, 'ax_spike_shape'):
            try:
                from core.analysis import detect_spikes
                v = self._last_result.v_soma
                peak_idx, spike_times, _ = detect_spikes(v, t_arr, threshold=-20.0)
                if len(spike_times) > 0:
                    nearest_t = float(spike_times[int(np.argmin(np.abs(spike_times - t_ms)))])
                    if hasattr(self, '_spike_shape_time_marker') and self._spike_shape_time_marker is not None:
                        try:
                            self._spike_shape_time_marker.remove()
                        except Exception:
                            pass
                    self._spike_shape_time_marker = self.ax_spike_shape.axvline(
                        x=nearest_t, color='#F9E2AF', linestyle='--', linewidth=2.0, alpha=0.9
                    )
                    if hasattr(self, 'cvs_spike_shape'):
                        self.cvs_spike_shape.draw_idle()
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────
    #  PER-TAB BUILDER METHODS  (called once on first visit)
    # ─────────────────────────────────────────────────────────────────
    def _build_tab_gates(self) -> QWidget:
        # Scrollable dashboard with persistent plot + toggle controls.
        self.fig_gates, cvs = _mpl_fig(1, 1)
        self.ax_gates = self.fig_gates.add_subplot(1, 1, 1)
        _set_tight_layout_engine(self.fig_gates, pad=2.5)
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

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(4)
        hint = QLabel("How to read: compare gate opening/closure against V_soma. Fast gates shape the upstroke; slow gates shape adaptation and rebound.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#A6ADC8; font-size:11px;")
        controls_layout.addWidget(hint)
        controls_layout.addWidget(scroll_area)

        # Store references
        self._gates_checkbox_container = checkbox_widget
        self._gates_checkbox_layout = checkbox_layout

        return _tab_with_toolbar(
            cvs,
            fullscreen_callback=lambda: self._open_fullscreen_plot('Gate Dynamics'),
            extra_widget=controls,
            extra_widget_position="above",
            scroll_canvas=True,
            min_canvas_height=980,
        )

    def _build_tab_spike_mech(self) -> QWidget:
        self.fig_spike_mech, cvs = _mpl_fig(5, 1, figsize=(11, 24), tight=False)
        self.ax_spike_mech = [self.fig_spike_mech.add_subplot(5, 1, k) for k in range(1, 6)]
        for ax in self.ax_spike_mech:
            ax.set_navigate(True)
        _set_canvas_margins(self.fig_spike_mech, left=0.08, right=0.97, top=0.97, bottom=0.05, hspace=0.55, wspace=0.28)
        self.cvs_spike_mech = cvs
        self._tab_figures['Spike Mechanism'] = self.fig_spike_mech
        self._spike_mech_selected = 0
        self._spike_mech_peak_idx = np.array([], dtype=int)
        self._spike_mech_spike_times = np.array([], dtype=float)
        self._spike_mech_last_result = None
        self._spike_mech_last_stats = None
        self._spike_mech_click_cid = self.cvs_spike_mech.mpl_connect('button_press_event', self._on_spike_mech_clicked)

        from PySide6.QtWidgets import QSpinBox, QComboBox, QCheckBox, QHBoxLayout, QLabel, QWidget
        controls = QWidget()
        layout = QHBoxLayout(controls)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        layout.addWidget(QLabel('Spike'))
        self._spike_zoomer = QSpinBox()
        self._spike_zoomer.setRange(1, 1)
        self._spike_zoomer.valueChanged.connect(self._on_spike_zoomer_changed)
        layout.addWidget(self._spike_zoomer)

        layout.addWidget(QLabel('Currents'))
        self._spike_mech_current_scope = QComboBox()
        self._spike_mech_current_scope.addItems(['Top 6', 'Top 10', 'All'])
        self._spike_mech_current_scope.currentTextChanged.connect(self._on_spike_mech_controls_changed)
        layout.addWidget(self._spike_mech_current_scope)

        self._spike_mech_normalize = QCheckBox('Normalize currents')
        self._spike_mech_normalize.setChecked(True)
        self._spike_mech_normalize.stateChanged.connect(self._on_spike_mech_controls_changed)
        layout.addWidget(self._spike_mech_normalize)

        layout.addStretch(1)
        hint = QLabel('Click a spike in the overview to drive all lower panels.')
        hint.setStyleSheet('color:#A6ADC8; font-size:11px;')
        layout.addWidget(hint)

        return _tab_with_toolbar(
            cvs,
            fullscreen_callback=lambda: self._open_fullscreen_plot('Spike Mechanism'),
            extra_widget=controls,
            scroll_canvas=True,
            min_canvas_height=1600,
        )

    def _build_tab_currents(self) -> QWidget:
        # Scrollable dashboard with persistent plot + toggle controls.
        self.fig_currents, cvs = _mpl_fig(1, 1)
        self.ax_currents = self.fig_currents.add_subplot(1, 1, 1)
        _set_tight_layout_engine(self.fig_currents, pad=2.5)
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

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(4)
        hint = QLabel("How to read: inward and outward channel currents compete to shape threshold, repolarization, burst support, and energetic cost.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#A6ADC8; font-size:11px;")
        controls_layout.addWidget(hint)
        controls_layout.addWidget(scroll_area)

        # Store references
        self._currents_checkbox_container = checkbox_widget
        self._currents_checkbox_layout = checkbox_layout

        return _tab_with_toolbar(
            cvs,
            fullscreen_callback=lambda: self._open_fullscreen_plot('Currents'),
            extra_widget=controls,
            extra_widget_position="above",
            scroll_canvas=True,
            min_canvas_height=980,
        )

    def _build_tab_phase(self) -> QWidget:
        self.fig_phase, cvs = _mpl_fig(1, 1, figsize=(9.4, 6.6))
        self.ax_phase = self.fig_phase.add_subplot(1, 1, 1)
        self.cvs_phase = cvs
        self._tab_figures['Phase Plane'] = self.fig_phase
        self._phase_lines: dict[str, object] = {}
        self._phase_warning_text = None

        # Add time slider for trajectory evolution
        from PySide6.QtWidgets import QSlider, QHBoxLayout, QLabel, QWidget, QComboBox, QVBoxLayout
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(4)
        self._phase_intro_label = QLabel(
            "How to read: the trajectory shows how membrane voltage and a chosen gate evolve together. "
            "A clipped window helps isolate threshold approach, rebound, burst loops, or late adaptation."
        )
        self._phase_intro_label.setWordWrap(True)
        self._phase_intro_label.setStyleSheet("color:#A6ADC8; font-size:11px;")
        controls_layout.addWidget(self._phase_intro_label)
        slider_widget = QWidget()
        slider_layout = QHBoxLayout(slider_widget)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.setSpacing(5)

        # Add Y-axis selector for Phase Plane
        lbl_y_axis = QLabel("Y-Axis:")
        lbl_y_axis.setStyleSheet("color:#CDD6F4; font-size:11px;")
        slider_layout.insertWidget(0, lbl_y_axis)

        self._phase_y_combo = QComboBox()
        self._phase_y_combo.addItems([
            "n (K act)", "h (Na inact)", "m (Na act)",
            "q (ITCa inact)", "r (Ih act)", "w (IM act)", "a (IA act)"
        ])
        self._phase_y_combo.setStyleSheet("""
            QComboBox {
                background:#313244; color:#CDD6F4; border:1px solid #45475A;
                padding:2px; font-size:11px; min-width:90px;
            }
        """)
        self._phase_y_combo.currentTextChanged.connect(self._on_phase_y_changed)
        slider_layout.insertWidget(1, self._phase_y_combo)
        slider_layout.insertSpacing(2, 15)

        slider_label = QLabel("Window:")
        slider_label.setStyleSheet("color:#CDD6F4; font-size:11px;")
        slider_layout.addWidget(slider_label)

        self._phase_time_start = QSlider(Qt.Orientation.Horizontal)
        self._phase_time_end = QSlider(Qt.Orientation.Horizontal)
        for slider, accent in ((self._phase_time_start, "#F9E2AF"), (self._phase_time_end, "#89B4FA")):
            slider.setRange(0, 100)
            slider.setValue(0 if slider is self._phase_time_start else 100)
            slider.setStyleSheet(f"""
                QSlider {{
                    background:#313244; border:1px solid #45475A;
                    padding:2px; height:20px;
                }}
                QSlider::handle:horizontal {{
                    background:{accent}; width:16px; margin:-4px 0; border-radius:8px;
                }}
            """)
            slider.valueChanged.connect(self._on_phase_time_slider_changed)

        slider_layout.addWidget(QLabel("Start"))
        slider_layout.addWidget(self._phase_time_start, 1)
        slider_layout.addWidget(QLabel("End"))
        slider_layout.addWidget(self._phase_time_end, 1)

        self._phase_time_label = QLabel("All")
        self._phase_time_label.setStyleSheet("color:#CBA6F7; font-size:11px;")
        self._phase_time_label.setMinimumWidth(120)
        slider_layout.addWidget(self._phase_time_label)
        self._phase_window_source_label = QLabel("Source: full trace")
        self._phase_window_source_label.setStyleSheet("color:#94E2D5; font-size:11px;")
        self._phase_window_source_label.setMinimumWidth(140)
        slider_layout.addWidget(self._phase_window_source_label)

        # Store slider widget reference
        self._phase_slider_widget = slider_widget
        # Disconnect signal when widget is destroyed to prevent C++ object deletion error
        # Use weakref to avoid reference cycles
        import weakref
        self_ref = weakref.ref(self)
        def _cleanup_slider():
            try:
                obj = self_ref()
                if obj is not None and hasattr(obj, '_phase_time_start') and hasattr(obj, '_phase_time_end'):
                    try:
                        obj._phase_time_start.valueChanged.disconnect(obj._on_phase_time_slider_changed)
                    except (RuntimeError, TypeError, SystemError):
                        pass  # Widget already destroyed or signal not connected
                    try:
                        obj._phase_time_end.valueChanged.disconnect(obj._on_phase_time_slider_changed)
                    except (RuntimeError, TypeError, SystemError):
                        pass
            except Exception:
                pass  # Ignore any errors during cleanup
        slider_widget.destroyed.connect(_cleanup_slider)

        # Store full trajectory data for slider
        self._phase_full_data = None  # Will store (t, V, n_t)
        self._phase_warning_text = self.ax_phase.text(
            0.02, 0.98, '', transform=self.ax_phase.transAxes,
            ha='left', va='top', color='#F38BA8', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='#1E1E2E', alpha=0.8)
        )

        # Add Vector Field overlay checkbox
        from PySide6.QtWidgets import QCheckBox
        self._cb_vector_field = QCheckBox("Show Vector Field")
        self._cb_vector_field.setChecked(False)
        self._cb_vector_field.setStyleSheet("color:#CDD6F4; font-size:11px;")
        self._cb_vector_field.stateChanged.connect(self._on_phase_vector_field_changed)
        slider_layout.addWidget(self._cb_vector_field)

        # Initialize vector field storage
        self._vector_field_quiver = None
        self._vector_field_grid_V = None
        self._vector_field_grid_gate = None

        self._phase_explain_label = QLabel("")
        self._phase_explain_label.setWordWrap(True)
        self._phase_explain_label.setStyleSheet("color:#BAC2DE; font-size:11px;")
        controls_layout.addWidget(slider_widget)
        controls_layout.addWidget(self._phase_explain_label)

        return _tab_with_toolbar(
            cvs,
            fullscreen_callback=lambda: self._open_fullscreen_plot('Phase Plane'),
            extra_widget=controls,
            scroll_canvas=True,
            min_canvas_height=760,
        )

    def _build_tab_kymo(self) -> QWidget:
        self.fig_kymo, cvs = _mpl_fig(2, 1)
        self._kymo_axes = [self.fig_kymo.add_subplot(2, 1, k) for k in range(1, 3)]
        _set_tight_layout_engine(self.fig_kymo, pad=2.5)
        self._kymo_cbar1 = None
        self._kymo_cbar2 = None
        self._kymo_im1 = None
        self._kymo_im2 = None
        self._kymo_empty_text = None
        self.cvs_kymo = cvs
        self._tab_figures['Kymograph'] = self.fig_kymo
        return _tab_with_toolbar(
            cvs,
            fullscreen_callback=lambda: self._open_fullscreen_plot('Kymograph'),
            scroll_canvas=True,
            min_canvas_height=900,
        )

    def _build_tab_energy_balance(self) -> QWidget:
        import matplotlib.gridspec as gridspec
        # Create larger figure for taller/wider graphs
        self.fig_energy, cvs = _mpl_fig(1, 1, figsize=(10.5, 12), tight=False)
        # v11.20: 4 time-series + horizontal stacked bar (replaces pie chart)
        gs = gridspec.GridSpec(4, 2, width_ratios=[8, 2], height_ratios=[1, 1, 1, 1],
                               figure=self.fig_energy, hspace=0.35, wspace=0.25)
        # Time-series plots in left column (all 4 rows)
        self.ax_energy = [
            self.fig_energy.add_subplot(gs[0, 0]),  # Balance Error
            self.fig_energy.add_subplot(gs[1, 0]),  # Cumulative Charge
            self.fig_energy.add_subplot(gs[2, 0]),  # Power
            self.fig_energy.add_subplot(gs[3, 0]),  # ATP Pool
            self.fig_energy.add_subplot(gs[0, 1]),  # ATP Breakdown (stacked bar)
        ]
        self.cvs_energy = cvs
        self._tab_figures['Energy & Balance'] = self.fig_energy
        self._energy_lines: dict[str, object] = {}
        self._balance_lines: dict[str, object] = {}
        self._energy_texts: dict[str, object] = {}
        self._atp_line = None
        self._atp_threshold_line = None
        # Pre-allocate horizontal stacked bar artists (created once, updated in-place)
        ax5 = self.ax_energy[4]
        bar_colors = [
            CHAN_COLORS.get('Na', '#FF6B6B'),
            CHAN_COLORS.get('ICa', '#FA9600'),
            '#888888',
        ]
        self._atp_bar_patches = []
        left = 0.0
        for c in bar_colors:
            bar = ax5.barh(0, 0.0, left=left, height=0.5, color=c, edgecolor='none')
            self._atp_bar_patches.append(bar[0])
        ax5.set_xlim(0, 100)
        ax5.set_yticks([])
        ax5.set_xlabel('%', fontsize=9)
        ax5.set_title('ATP Breakdown', fontsize=10)
        self._atp_bar_labels = ['Na+ Pump', 'Ca2+ Pump', 'Resting']
        self._atp_bar_no_data_text = ax5.text(
            0.5, 0.5, 'No ATP data', ha='center', va='center',
            transform=ax5.transAxes, fontsize=10, color='#BAC2DE', visible=False,
        )
        return _tab_with_toolbar(
            cvs,
            fullscreen_callback=lambda: self._open_fullscreen_plot('Energy & Balance'),
            scroll_canvas=True,
            min_canvas_height=1080,
        )

    def _build_tab_spike_shape(self) -> QWidget:
        self.fig_spike_shape, cvs = _mpl_fig(1, 1)
        self.ax_spike_shape = self.fig_spike_shape.add_subplot(1, 1, 1)
        _set_tight_layout_engine(self.fig_spike_shape, pad=2.5)
        self.cvs_spike_shape = cvs
        self._tab_figures['Spike Shape'] = self.fig_spike_shape
        self._spike_shape_init_done = False
        self._spike_shape_lines: dict[str, object] = {}
        self._spike_shape_dynamic_artists: list[object] = []
        self._spike_shape_empty_text = self.ax_spike_shape.text(
            0.5, 0.5, '', ha='center', va='center', transform=self.ax_spike_shape.transAxes,
            fontsize=12, color='#89B4FA', visible=False,
            bbox=dict(boxstyle='round', facecolor='#1E1E2E', alpha=0.85, edgecolor='#45475A')
        )

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

        return _tab_with_toolbar(
            cvs,
            fullscreen_callback=lambda: self._open_fullscreen_plot('Spike Shape'),
            extra_widget=selection_widget,
            scroll_canvas=True,
            min_canvas_height=920,
        )

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
        _set_tight_layout_engine(self.fig_poincare, pad=2.5)
        self.cvs_poincare = cvs
        self._tab_figures['Poincare (ISI)'] = self.fig_poincare
        self._poincare_init_done = False
        self._poincare_lines: dict[str, object] = {}
        self._poincare_lines["diag"] = self.ax_poincare.plot([], [], 'r--', lw=1.5, alpha=0.5, label='ISI[n+1] = ISI[n]')[0]
        self._poincare_lines["scatter"] = self.ax_poincare.scatter([], [], c=[], cmap='viridis', s=30, alpha=0.7, edgecolors='k', linewidth=0.5)
        self._poincare_lines["msg"] = self.ax_poincare.text(
            0.5, 0.5, '', ha='center', va='center', transform=self.ax_poincare.transAxes,
            fontsize=12, color='#89B4FA', visible=False,
            bbox=dict(boxstyle='round', facecolor='#1E1E2E', alpha=0.85, edgecolor='#45475A')
        )
        return _tab_with_toolbar(cvs, fullscreen_callback=lambda: self._open_fullscreen_plot('Poincare (ISI)'))

    def _build_tab_isi_dist(self) -> QWidget:
        """Build the ISI Distribution tab with histogram and Shannon entropy."""
        self.fig_isi_dist, cvs = _mpl_fig(2, 1, figsize=(10.0, 9.0), tight=False)
        self._isi_dist_axes = [self.fig_isi_dist.add_subplot(2, 1, k) for k in range(1, 3)]
        _set_canvas_margins(self.fig_isi_dist, left=0.10, right=0.90, top=0.94, bottom=0.08, hspace=0.38, wspace=0.20)
        self.cvs_isi_dist = cvs
        self._tab_figures['ISI Distribution'] = self.fig_isi_dist

        # Initialize plot elements
        ax_hist, ax_metrics = self._isi_dist_axes
        self._isi_hist_bars = None
        self._isi_fit_line = None
        self._isi_metrics_text = None

        ax_hist.set_xlabel('ISI (ms)')
        ax_hist.set_ylabel('Count')
        ax_hist.set_title('ISI Distribution Histogram')

        ax_metrics.set_xlim(0, 1)
        ax_metrics.set_ylim(0, 1)
        ax_metrics.axis('off')

        return _tab_with_toolbar(
            cvs,
            fullscreen_callback=lambda: self._open_fullscreen_plot('ISI Distribution'),
            scroll_canvas=True,
            min_canvas_height=920,
        )

    def _build_tab_bif(self) -> QWidget:
        self.fig_bif, cvs = _mpl_fig(2, 2, figsize=(10.2, 9.2), tight=False)
        self.ax_bif = [self.fig_bif.add_subplot(2, 2, k) for k in range(1, 5)]
        from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(4)
        hint = QLabel(
            "How to read: sweep one control parameter and watch where the cell leaves rest, enters tonic spiking, switches into burst-prone behavior, or falls into depolarization block. "
            "Peak branches show the geometric structure, while rate and spike-count panels tell you whether the branch represents sustained firing or only isolated events."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#A6ADC8; font-size:11px;")
        self._bif_summary_label = QLabel("Run a bifurcation scan to map rest → tonic → burst/block transitions.")
        self._bif_summary_label.setWordWrap(True)
        self._bif_summary_label.setStyleSheet("color:#BAC2DE; font-size:11px;")
        controls_layout.addWidget(hint)
        controls_layout.addWidget(self._bif_summary_label)
        self.tab_bif = _tab_with_toolbar(
            cvs,
            fullscreen_callback=lambda: self._open_fullscreen_plot('Bifurcation'),
            extra_widget=controls,
            extra_widget_position="above",
            scroll_canvas=True,
            min_canvas_height=900,
        )
        self.cvs_bif = cvs
        self._tab_figures['Bifurcation'] = self.fig_bif
        self._bif_lines: dict[str, object] = {}
        self._bif_peak_scatter = None
        return self.tab_bif

    def _build_tab_sweep(self) -> QWidget:
        self.fig_sweep, cvs = _mpl_fig(2, 2, figsize=(10.2, 9.2), tight=False)
        self.ax_sweep = [self.fig_sweep.add_subplot(2, 2, k) for k in range(1, 5)]
        from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(4)
        hint = QLabel(
            "How to read: the upper-left panel shows representative traces, while the lower panels turn the same sweep into peak-voltage, firing-rate, and spike-count summaries. "
            "For an f-I run, focus on the first spiking sample (approximate rheobase), the slope of the firing-rate curve, and whether the spike count keeps growing or collapses."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#A6ADC8; font-size:11px;")
        self._sweep_summary_label = QLabel(
            "Run a sweep or dedicated f-I scan to populate this dashboard. "
            "For stim.Iext, the first active sample is your practical threshold estimate."
        )
        self._sweep_summary_label.setWordWrap(True)
        self._sweep_summary_label.setStyleSheet("color:#BAC2DE; font-size:11px;")
        controls_layout.addWidget(hint)
        controls_layout.addWidget(self._sweep_summary_label)
        self.tab_sweep = _tab_with_toolbar(
            cvs,
            fullscreen_callback=lambda: self._open_fullscreen_plot('Sweep'),
            extra_widget=controls,
            extra_widget_position="above",
            scroll_canvas=True,
            min_canvas_height=900,
        )
        self.cvs_sweep = cvs
        self._tab_figures['Sweep'] = self.fig_sweep
        self._tab_figures['f-I Curve'] = self.fig_sweep
        self._sweep_cbar = None
        self._sweep_trace_lines: list = []
        self._sweep_trace_max = 64
        self._sweep_metric_lines: dict[str, object] = {}
        return self.tab_sweep

    def _build_tab_sd(self) -> QWidget:
        self.fig_sd, cvs = _mpl_fig(1, 2, figsize=(10.0, 7.6), tight=False)
        self.ax_sd = [self.fig_sd.add_subplot(1, 2, k) for k in range(1, 3)]
        self.tab_sd = _tab_with_toolbar(
            cvs,
            fullscreen_callback=lambda: self._open_fullscreen_plot('S-D Curve'),
            scroll_canvas=True,
            min_canvas_height=760,
        )
        self.cvs_sd = cvs
        self._tab_figures['S-D Curve'] = self.fig_sd
        self._sd_lines: dict[str, object] = {}
        return self.tab_sd

    def _build_tab_excmap(self) -> QWidget:
        self.fig_excmap, cvs = _mpl_fig(1, 2, figsize=(10.0, 7.6), tight=False)
        self.ax_excmap = [self.fig_excmap.add_subplot(1, 2, k) for k in range(1, 3)]
        self.tab_excmap = _tab_with_toolbar(
            cvs,
            fullscreen_callback=lambda: self._open_fullscreen_plot('Excitability Map'),
            scroll_canvas=True,
            min_canvas_height=760,
        )
        self.cvs_excmap = cvs
        self._tab_figures['Excitability Map'] = self.fig_excmap
        self._excmap_mesh = {"spikes": None, "freq": None}
        self._excmap_cbar = {"spikes": None, "freq": None}
        return self.tab_excmap

    def _build_tab_spectro(self) -> QWidget:
        self.fig_spectro, cvs = _mpl_fig(2, 1, tight=False)
        self.ax_spectro = [self.fig_spectro.add_subplot(2, 1, k) for k in range(1, 3)]
        _set_canvas_margins(self.fig_spectro, left=0.08, right=0.94, top=0.95, bottom=0.08, hspace=0.34, wspace=0.20)
        self.cvs_spectro = cvs
        self._tab_figures['Spectrogram'] = self.fig_spectro
        self._spectro_cbar = None
        self._spectro_vm_line = None
        self._spectro_mesh = None
        self._spectro_fail_text = None
        return _tab_with_toolbar(
            cvs,
            fullscreen_callback=lambda: self._open_fullscreen_plot('Spectrogram'),
            scroll_canvas=True,
            min_canvas_height=920,
        )

    def _build_tab_impedance(self) -> QWidget:
        self.fig_impedance, cvs = _mpl_fig(2, 1, tight=False)
        self.ax_impedance = [self.fig_impedance.add_subplot(2, 1, k) for k in range(1, 3)]
        _set_canvas_margins(self.fig_impedance, left=0.08, right=0.96, top=0.95, bottom=0.08, hspace=0.36, wspace=0.20)
        self.cvs_impedance = cvs
        self._tab_figures['Impedance'] = self.fig_impedance
        self._impedance_lines: dict[str, object] = {}
        self._impedance_texts: dict[str, object] = {}
        return _tab_with_toolbar(cvs, fullscreen_callback=lambda: self._open_fullscreen_plot('Impedance'))

    def _build_tab_chaos(self) -> QWidget:
        self.fig_chaos, cvs = _mpl_fig(3, 1, figsize=(9.6, 18), tight=False)
        self.ax_chaos = [self.fig_chaos.add_subplot(3, 1, k) for k in range(1, 4)]
        _set_canvas_margins(self.fig_chaos, left=0.08, right=0.97, top=0.96, bottom=0.06, hspace=0.42, wspace=0.24)
        self.cvs_chaos = cvs
        self._tab_figures['Chaos & LLE'] = self.fig_chaos
        self._chaos_force_enabled = getattr(self, '_chaos_force_enabled', False)
        self._chaos_selected_k = 0
        self._chaos_selected_pair = 0
        self._chaos_analysis = None
        self._chaos_texts: dict[str, object] = {}
        # PERSISTENT ARTISTS: Create once, update data (avoids memory fragmentation)
        self._chaos_persistent_artists: dict[str, object] = {}
        self._create_chaos_persistent_artists()
        self._chaos_click_cid = self.cvs_chaos.mpl_connect('button_press_event', self._on_chaos_plot_clicked)

        from PySide6.QtWidgets import QWidget, QGridLayout, QLabel, QSpinBox, QDoubleSpinBox, QComboBox
        controls = QWidget()
        layout = QGridLayout(controls)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(4)

        def _add_label(text_label: str, row: int, col: int):
            lbl = QLabel(text_label)
            lbl.setStyleSheet('color:#CDD6F4; font-size:11px;')
            layout.addWidget(lbl, row, col)
            return lbl

        _add_label('Embed', 0, 0)
        self._chaos_embed_spin = QSpinBox()
        self._chaos_embed_spin.setRange(2, 8)
        self._chaos_embed_spin.setValue(3)
        self._chaos_embed_spin.valueChanged.connect(self._on_chaos_controls_changed)
        layout.addWidget(self._chaos_embed_spin, 0, 1)

        _add_label('Lag', 0, 2)
        self._chaos_lag_spin = QSpinBox()
        self._chaos_lag_spin.setRange(1, 20)
        self._chaos_lag_spin.setValue(2)
        self._chaos_lag_spin.valueChanged.connect(self._on_chaos_controls_changed)
        layout.addWidget(self._chaos_lag_spin, 0, 3)

        _add_label('Fit start', 0, 4)
        self._chaos_fit_start_spin = QDoubleSpinBox()
        self._chaos_fit_start_spin.setRange(0.0, 500.0)
        self._chaos_fit_start_spin.setDecimals(1)
        self._chaos_fit_start_spin.setValue(5.0)
        self._chaos_fit_start_spin.valueChanged.connect(self._on_chaos_controls_changed)
        layout.addWidget(self._chaos_fit_start_spin, 0, 5)

        _add_label('Fit end', 1, 0)
        self._chaos_fit_end_spin = QDoubleSpinBox()
        self._chaos_fit_end_spin.setRange(1.0, 1000.0)
        self._chaos_fit_end_spin.setDecimals(1)
        self._chaos_fit_end_spin.setValue(40.0)
        self._chaos_fit_end_spin.valueChanged.connect(self._on_chaos_controls_changed)
        layout.addWidget(self._chaos_fit_end_spin, 1, 1)

        _add_label('Min sep', 1, 2)
        self._chaos_sep_spin = QDoubleSpinBox()
        self._chaos_sep_spin.setRange(0.0, 500.0)
        self._chaos_sep_spin.setDecimals(1)
        self._chaos_sep_spin.setValue(10.0)
        self._chaos_sep_spin.valueChanged.connect(self._on_chaos_controls_changed)
        layout.addWidget(self._chaos_sep_spin, 1, 3)

        _add_label('Pair', 1, 4)
        self._chaos_pair_mode = QComboBox()
        self._chaos_pair_mode.addItems(['Representative', 'First pair', 'Pair index'])
        self._chaos_pair_mode.currentTextChanged.connect(self._on_chaos_controls_changed)
        layout.addWidget(self._chaos_pair_mode, 1, 5)

        self._chaos_pair_spin = QSpinBox()
        self._chaos_pair_spin.setRange(1, 1)
        self._chaos_pair_spin.valueChanged.connect(self._on_chaos_controls_changed)
        layout.addWidget(self._chaos_pair_spin, 1, 6)

        tip = QLabel('Click the divergence curve to inspect a specific separation horizon.')
        tip.setStyleSheet('color:#A6ADC8; font-size:11px;')
        tip.setWordWrap(True)
        layout.addWidget(tip, 0, 6, 1, 2)
        layout.setColumnStretch(7, 1)

        return _tab_with_toolbar(
            cvs,
            fullscreen_callback=lambda: self._open_fullscreen_plot('Chaos & LLE'),
            extra_widget=controls,
            scroll_canvas=True,
            min_canvas_height=1280,
        )

    def _build_tab_modulation(self) -> QWidget:
        self.fig_mod, cvs = _mpl_fig(1, 1, figsize=(8.8, 7.2), tight=False)
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
        self._mod_soft_warning = self.ax_mod.text(
            0.5, 0.5,
            '[WARNING: LOW STATISTICAL POWER (<3 CYCLES)]',
            ha='center', va='center', transform=self.ax_mod.transAxes,
            fontsize=13, color='#F38BA8', alpha=0.28, rotation=18, visible=False,
        )
        return _tab_with_toolbar(cvs, fullscreen_callback=lambda: self._open_fullscreen_plot('Phase-Locking'))

    def _create_chaos_persistent_artists(self):
        """Create persistent matplotlib artists for Chaos tab to avoid memory fragmentation.
        
        All artists are created once during tab initialization and updated via set_data()
        methods instead of being removed and recreated on each update.
        """
        if not hasattr(self, 'ax_chaos') or len(self.ax_chaos) < 3:
            return
            
        ax_div, ax_attr, ax_trace = self.ax_chaos
        
        # Divergence plot (ax_div)
        self._chaos_persistent_artists['div_line'] = ax_div.plot(
            [], [], color='#89B4FA', lw=2.0, label='Trajectory divergence'
        )[0]
        self._chaos_persistent_artists['div_scatter'] = ax_div.scatter(
            [], [], color='#F9E2AF', s=60, zorder=5, label='Selected horizon'
        )
        self._chaos_persistent_artists['fit_span'] = ax_div.axvspan(
            0, 0, color='#A6E3A1', alpha=0.12, label='Fit window', visible=False
        )
        self._chaos_persistent_artists['fit_line'] = ax_div.plot(
            [], [], '--', color='#F38BA8', lw=2.0, label='Linear trend'
        )[0]
        self._chaos_persistent_artists['summary_text'] = ax_div.text(
            0.02, 0.96, "", transform=ax_div.transAxes, ha='left', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='#1E1E2E', alpha=0.9), color='#CDD6F4'
        )
        self._chaos_persistent_artists['stoch_text'] = ax_div.text(
            0.5, 0.46, "", ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='#1E1E2E', alpha=0.9), color='#CDD6F4'
        )
        
        # Attractor plot (ax_attr)
        self._chaos_persistent_artists['emb_line'] = ax_attr.plot(
            [], [], color='#6C7086', alpha=0.4, lw=1.0, label='Embedded attractor'
        )[0]
        self._chaos_persistent_artists['emb_ref'] = ax_attr.plot(
            [], [], color='#89B4FA', lw=2.5, label='Reference segment'
        )[0]
        self._chaos_persistent_artists['emb_ref_start'] = ax_attr.scatter(
            [], [], color='#89B4FA', s=50
        )
        self._chaos_persistent_artists['emb_neighbor'] = ax_attr.plot(
            [], [], color='#F38BA8', lw=2.5, label='Neighbor segment'
        )[0]
        self._chaos_persistent_artists['emb_neighbor_start'] = ax_attr.scatter(
            [], [], color='#F38BA8', s=50
        )
        
        # Trace plot (ax_trace)
        self._chaos_persistent_artists['trace_line'] = ax_trace.plot(
            [], [], color='#89B4FA', lw=1.4, label='V_soma'
        )[0]
        self._chaos_persistent_artists['trace_ref'] = ax_trace.axvspan(
            0, 0, color='#89B4FA', alpha=0.18, label='Reference source', visible=False
        )
        self._chaos_persistent_artists['trace_neighbor'] = ax_trace.axvspan(
            0, 0, color='#F38BA8', alpha=0.18, label='Neighbor source', visible=False
        )
        
        # Initially hide all artists until data is available
        for artist in self._chaos_persistent_artists.values():
            if hasattr(artist, 'set_visible'):
                artist.set_visible(False)

    def _build_tab_fingerprint(self) -> QWidget:
        """Build the Radar Chart (Biophysical Fingerprint) tab."""
        self.fig_fingerprint, cvs = _mpl_fig(1, 1, figsize=(8.8, 8.8), tight=False)
        # Use polar projection for radar chart
        self.ax_fingerprint = self.fig_fingerprint.add_subplot(1, 1, 1, polar=True)
        self.cvs_fingerprint = cvs
        self._tab_figures['Fingerprint'] = self.fig_fingerprint
        # Initialize empty radar chart
        self._radar_labels = ['Firing Rate', 'Adaptation', 'Half-width', 'Rin', 'Energy', 'P/T Ratio']
        self._radar_angles = np.linspace(0, 2 * np.pi, len(self._radar_labels), endpoint=False).tolist()
        self._radar_angles += self._radar_angles[:1]  # Complete the circle
        self._radar_fill = None
        self._radar_line = None
        self._radar_points = None
        self._radar_text = None
        # Set up the axes
        self.ax_fingerprint.set_xticks(self._radar_angles[:-1])
        self.ax_fingerprint.set_xticklabels(self._radar_labels, size=10)
        self.ax_fingerprint.set_ylim(0, 1)
        self.ax_fingerprint.set_title('Biophysical Fingerprint', fontsize=14, fontweight='bold', pad=20)
        return _tab_with_toolbar(cvs, fullscreen_callback=lambda: self._open_fullscreen_plot('Fingerprint'))

    def _build_tab_csd(self) -> QWidget:
        """Build the CSD (Current-Source Density) spatial heatmap tab with time slider."""
        from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QSlider, QLabel, QPushButton, QFrame

        # Main container with vertical layout
        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)

        # Controls panel
        controls = QFrame()
        controls.setStyleSheet("QFrame { background: #1E1E2E; border-radius: 4px; padding: 4px; }")
        ctrl_layout = QHBoxLayout(controls)
        ctrl_layout.setContentsMargins(8, 4, 8, 4)
        ctrl_layout.setSpacing(10)

        # Time slider for spatial profile at specific time
        ctrl_layout.addWidget(QLabel("<span style='color:#CDD6F4'>Profile at t=</span>"))

        self._csd_time_slider = QSlider(Qt.Orientation.Horizontal)
        self._csd_time_slider.setRange(0, 100)
        self._csd_time_slider.setValue(50)
        self._csd_time_slider.setStyleSheet("""
            QSlider::groove:horizontal { height: 4px; background: #313244; border-radius: 2px; }
            QSlider::handle:horizontal { background: #89B4FA; width: 14px; margin: -5px 0; border-radius: 7px; }
        """)
        self._csd_time_slider.valueChanged.connect(self._on_csd_time_changed)
        ctrl_layout.addWidget(self._csd_time_slider, stretch=1)

        self._lbl_csd_time = QLabel("<span style='color:#89B4FA'>50%</span>")
        ctrl_layout.addWidget(self._lbl_csd_time)

        # Play button for animation
        self._btn_csd_play = QPushButton("▶ Play")
        self._btn_csd_play.setStyleSheet("""
            QPushButton { background: #313244; color: #A6E3A1; border: 1px solid #45475A;
                         border-radius: 4px; padding: 4px 12px; font-size: 11px; }
            QPushButton:hover { background: #45475A; }
        """)
        self._btn_csd_play.clicked.connect(self._on_csd_play_clicked)
        ctrl_layout.addWidget(self._btn_csd_play)

        # Export button
        btn_export = QPushButton("📊 Export Profile")
        btn_export.setStyleSheet("""
            QPushButton { background: #313244; color: #89B4FA; border: 1px solid #45475A;
                         border-radius: 4px; padding: 4px 12px; font-size: 11px; }
            QPushButton:hover { background: #45475A; }
        """)
        btn_export.clicked.connect(self._on_csd_export_clicked)
        ctrl_layout.addWidget(btn_export)

        main_layout.addWidget(controls)

        # Matplotlib figure
        self.fig_csd, cvs = _mpl_fig(2, 1, figsize=(10.0, 8.5), tight=False)
        self._csd_axes = [self.fig_csd.add_subplot(2, 1, k) for k in range(1, 3)]
        # Add third axis for spatial profile
        self._csd_profile_ax = self.fig_csd.add_subplot(2, 2, 4)  # Right side, bottom half
        _set_canvas_margins(self.fig_csd, left=0.08, right=0.94, top=0.95, bottom=0.10, hspace=0.38, wspace=0.30)
        self.cvs_csd = cvs
        self._tab_figures['CSD'] = self.fig_csd
        self._csd_im = None
        self._csd_cbar = None
        self._csd_vm_line = None
        self._csd_profile_line = None
        self._csd_profile_fill = None
        self._csd_current_result = None  # Store for animation

        # Store animation timer
        self._csd_play_timer = None

        main_layout.addWidget(cvs, stretch=1)

        # Add toolbar directly (container is not a canvas, can't use _tab_with_toolbar)
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
        toolbar = NavToolbar(cvs, container)
        main_layout.insertWidget(1, toolbar)  # Insert after controls, before canvas

        cvs.setMinimumHeight(920)
        cvs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Add fullscreen button to toolbar
        btn_fullscreen = QPushButton("Fullscreen")
        btn_fullscreen.setToolTip("Open plot in fullscreen window")
        btn_fullscreen.setMaximumWidth(120)
        btn_fullscreen.clicked.connect(lambda: self._open_fullscreen_plot('CSD'))
        toolbar.addWidget(btn_fullscreen)

        # Wrap in scroll area like _tab_with_toolbar does
        scroll = QScrollArea()
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        return scroll

    def _build_tab_debug(self) -> QWidget:
        """Build the Debug LTE (Local Truncation Error) Monitor tab."""
        self.fig_debug, cvs = _mpl_fig(2, 1, figsize=(10.0, 9.0), tight=False)
        self._debug_axes = [self.fig_debug.add_subplot(2, 1, k) for k in range(1, 3)]
        _set_canvas_margins(self.fig_debug, left=0.08, right=0.94, top=0.95, bottom=0.08, hspace=0.34, wspace=0.20)
        self.cvs_debug = cvs
        self._tab_figures['Debug LTE'] = self.fig_debug
        self._lte_lines = None
        self._lte_threshold_line = None
        return _tab_with_toolbar(
            cvs,
            fullscreen_callback=lambda: self._open_fullscreen_plot('Debug LTE'),
            scroll_canvas=True,
            min_canvas_height=920,
        )

    def _build_tab_metabolic(self) -> QWidget:
        """Build Metabolic Trajectory tab showing [Na+]i, [K+]o, and ATP dynamics."""
        import matplotlib.gridspec as gridspec
        self.fig_metabolic, cvs = _mpl_fig(1, 1, figsize=(10.5, 10), tight=False)
        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1],
                               figure=self.fig_metabolic, hspace=0.40)
        self.ax_metabolic = [
            self.fig_metabolic.add_subplot(gs[0, 0]),  # [Na+]i
            self.fig_metabolic.add_subplot(gs[1, 0]),  # [K+]o
            self.fig_metabolic.add_subplot(gs[2, 0]),  # ATP
        ]
        self.cvs_metabolic = cvs
        self._tab_figures['Metabolic'] = self.fig_metabolic
        self._metabolic_lines: dict[str, object] = {}
        
        # Initialize threshold lines
        ax1, ax2, ax3 = self.ax_metabolic
        ax1.axhline(y=15.0, color='#F9E2AF', linestyle='--', lw=1.5, alpha=0.7, label='Healthy max')
        ax2.axhline(y=5.0, color='#F9E2AF', linestyle='--', lw=1.5, alpha=0.7, label='Healthy max')
        ax3.axhline(y=0.5, color='#F9E2AF', linestyle='--', lw=1.5, alpha=0.7, label='Warning')
        ax3.axhline(y=0.2, color='#F38BA8', linestyle='-', lw=2, alpha=0.8, label='Critical')
        
        return _tab_with_toolbar(
            cvs,
            fullscreen_callback=lambda: self._open_fullscreen_plot('Metabolic'),
            scroll_canvas=True,
            min_canvas_height=960,
        )

    def _update_metabolic(self, result, stats):
        """Update Metabolic Trajectory tab with ion concentrations and ATP."""
        if not hasattr(self, 'fig_metabolic'):
            return  # tab not yet visited
        t = result.t
        ax_na, ax_k, ax_atp = self.ax_metabolic
        
        # [Na+]i trajectory
        na_data = None
        if getattr(result, 'na_i', None) is not None:
            na_arr = np.asarray(result.na_i, dtype=float)
            if na_arr.ndim == 2:
                na_data = na_arr[0, :] if na_arr.shape[0] > 0 else None
            else:
                na_data = na_arr.reshape(-1) if na_arr.size > 0 else None
        
        if na_data is not None and na_data.size == t.size:
            if 'na_i' not in self._metabolic_lines:
                self._metabolic_lines['na_i'] = ax_na.plot([], [], color='#F38BA8', lw=2, label='[Na⁺]ᵢ')[0]
            _set_line_data(self._metabolic_lines['na_i'], t, na_data, name='na_i')
            na_min, na_max = np.min(na_data), np.max(na_data)
            ax_na.set_ylim(max(0, na_min * 0.9), na_max * 1.1)
            status_na = f"min={na_min:.1f}, max={na_max:.1f} mM"
        else:
            if 'na_i' in self._metabolic_lines:
                _set_line_data(self._metabolic_lines['na_i'])
            status_na = "No [Na⁺]ᵢ data (enable dynamic metabolism)"
        
        ax_na.set_ylabel('[Na⁺]ᵢ (mM)')
        ax_na.set_title(f'Intracellular Sodium — {status_na}')
        ax_na.legend(fontsize=8, loc='upper left')
        ax_na.grid(alpha=0.3)
        ax_na.tick_params(labelbottom=False)
        ax_na.relim()
        ax_na.autoscale_view()
        
        # [K+]o trajectory
        k_data = None
        if getattr(result, 'k_o', None) is not None:
            k_arr = np.asarray(result.k_o, dtype=float)
            if k_arr.ndim == 2:
                k_data = k_arr[0, :] if k_arr.shape[0] > 0 else None
            else:
                k_data = k_arr.reshape(-1) if k_arr.size > 0 else None
        
        if k_data is not None and k_data.size == t.size:
            if 'k_o' not in self._metabolic_lines:
                self._metabolic_lines['k_o'] = ax_k.plot([], [], color='#89B4FA', lw=2, label='[K⁺]ₒ')[0]
            _set_line_data(self._metabolic_lines['k_o'], t, k_data, name='k_o')
            k_min, k_max = np.min(k_data), np.max(k_data)
            ax_k.set_ylim(max(0, k_min * 0.9), k_max * 1.1)
            status_k = f"min={k_min:.1f}, max={k_max:.1f} mM"
        else:
            if 'k_o' in self._metabolic_lines:
                _set_line_data(self._metabolic_lines['k_o'])
            status_k = "No [K⁺]ₒ data (enable dynamic metabolism)"
        
        ax_k.set_ylabel('[K⁺]ₒ (mM)')
        ax_k.set_title(f'Extracellular Potassium — {status_k}')
        ax_k.legend(fontsize=8, loc='upper left')
        ax_k.grid(alpha=0.3)
        ax_k.tick_params(labelbottom=False)
        ax_k.relim()
        ax_k.autoscale_view()
        
        # ATP trajectory
        atp_data = None
        if getattr(result, 'atp_level', None) is not None:
            atp_arr = np.asarray(result.atp_level, dtype=float)
            if atp_arr.ndim == 2:
                atp_data = atp_arr[0, :] if atp_arr.shape[0] > 0 else None
            else:
                atp_data = atp_arr.reshape(-1) if atp_arr.size > 0 else None
        
        if atp_data is not None and atp_data.size == t.size:
            if 'atp' not in self._metabolic_lines:
                self._metabolic_lines['atp'] = ax_atp.plot([], [], color='#A6E3A1', lw=2, label='[ATP]ᵢ')[0]
            _set_line_data(self._metabolic_lines['atp'], t, atp_data, name='atp')
            atp_min = np.min(atp_data)
            if atp_min < 0.2:
                color = '#F38BA8'
                suffix = "🔴 CRITICAL"
            elif atp_min < 0.5:
                color = '#F9E2AF'
                suffix = "🟠 WARNING"
            else:
                color = '#A6E3A1'
                suffix = "✓ Healthy"
            self._metabolic_lines['atp'].set_color(color)
            ax_atp.set_ylim(0, max(3.0, np.max(atp_data) * 1.1))
            status_atp = f"min={atp_min:.2f} mM — {suffix}"
        else:
            if 'atp' in self._metabolic_lines:
                _set_line_data(self._metabolic_lines['atp'])
            status_atp = "No ATP data (enable dynamic metabolism)"
        
        ax_atp.set_ylabel('[ATP]ᵢ (mM)')
        ax_atp.set_xlabel('Time (ms)')
        ax_atp.set_title(f'Intracellular ATP — {status_atp}')
        ax_atp.legend(fontsize=8, loc='upper right')
        ax_atp.grid(alpha=0.3)
        ax_atp.relim()
        ax_atp.autoscale_view()
        
        self.cvs_metabolic.draw_idle()

    # ─────────────────────────────────────────────────────────────────
    #  MAIN UPDATE ENTRY POINT
    # ─────────────────────────────────────────────────────────────────
    def update_analytics(self, result, stats=None):
        """Update all already-built tabs from a SimulationResult.

        Lazy tabs that haven't been visited yet are skipped — their guard
        at the top of each _update_* method returns immediately if the
        corresponding figure attribute doesn't exist yet. They will be
        populated in _on_tab_changed when the user first clicks the tab.

        v13.0: stats parameter is now optional for async analytics support.
        If stats is None, it will be computed synchronously (legacy mode).
        """
        self._last_result = result

        # v13.0: Use provided stats (async mode) or compute synchronously (legacy)
        if stats is None:
            from core.analysis import full_analysis
            stats = full_analysis(result, compute_lyapunov=False)

        self._last_stats = stats
        self._btn_compute_lle.setEnabled(True)
        self._set_passport_status(result, stats)

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
        self._update_fingerprint(result, stats)
        self._update_csd(result)
        self._update_debug(result)
        if result.morph:
            self._update_energy_balance(result)
        self._update_spike_shape(result, stats)
        self._update_poincare(result, stats)
        self._update_isi_dist(result, stats)
        self._update_metabolic(result, stats)



    def mark_analysis_pending(self, result):
        """Show a non-empty Analytics state while background analysis is running."""
        self._last_result = result
        self._last_stats = None
        self._btn_compute_lle.setEnabled(False)
        self._set_passport_status(result, None)
        self.passport_view.setPlainText(
            "Simulation complete.\n\n"
            "Analytics is running in the background.\n"
            "Passport and lazy analysis tabs will populate when statistics are ready."
        )

    def _set_passport_status(self, result, stats) -> None:
        if not hasattr(self, "_passport_status_label"):
            return
        if result is None:
            self._passport_status_label.setText("No simulation loaded")
            return
        t = getattr(result, "t", None)
        t_end = float(t[-1]) if t is not None and len(t) else 0.0
        n_comp = int(getattr(result, "n_comp", 0) or 0)
        if stats is None:
            self._passport_status_label.setText(
                f"Analysis pending | {n_comp} compartments | {t_end:.1f} ms"
            )
            return
        n_spikes = stats.get("n_spikes", stats.get("spike_count", 0))
        freq = stats.get("firing_rate_hz", stats.get("freq_hz", 0.0))
        try:
            freq_text = f"{float(freq):.2f} Hz"
        except (TypeError, ValueError):
            freq_text = "n/a"
        self._passport_status_label.setText(
            f"Analysis ready | {n_comp} compartments | {t_end:.1f} ms | spikes={n_spikes} | {freq_text}"
        )
    def _compute_lle_now(self):
        """Emit signal to trigger full simulation rerun with LLE enabled.
        
        v12.2: Changed from post-hoc analysis to full rerun honesty.
        MainWindow connects computeLleRequested to run with calc_lle=True.
        """
        if self._last_result is None:
            return
        # Emit signal - MainWindow will handle the full rerun with calc_lle=True
        self.computeLleRequested.emit()

    # ─────────────────────────────────────────────────────────────────
    #  0 — NEURON PASSPORT DASHBOARD
    # ─────────────────────────────────────────────────────────────────
    def _build_tab_passport(self) -> QWidget:
        """Build the Neuron Passport Dashboard with radar chart and progress bars."""
        from PySide6.QtWidgets import QProgressBar
        
        # Main container
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)
        
        # Left side: Radar chart (reuses fingerprint figure setup)
        self.fig_passport_radar, cvs = _mpl_fig(1, 1, figsize=(6, 6), tight=False)
        self.ax_passport_radar = self.fig_passport_radar.add_subplot(111, projection='polar')
        self.ax_passport_radar.set_facecolor('#0D1117')
        self.ax_passport_radar.tick_params(colors='#CDD6F4')
        self.ax_passport_radar.set_title('Biophysical Fingerprint', color='#CDD6F4', pad=20)
        layout.addWidget(cvs, stretch=1)
        
        # Right side: Scroll area with text and progress bars
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.setSpacing(8)

        self._passport_status_label = QLabel("No simulation loaded")
        self._passport_status_label.setStyleSheet(
            "color:#89B4FA; font-weight:bold; font-size:12px; padding:4px;"
            "background:#111827; border:1px solid #313244; border-radius:4px;"
        )
        right_layout.addWidget(self._passport_status_label)
        
        # Text view (original passport view)
        self.passport_view = QTextEdit()
        self.passport_view.setReadOnly(True)
        self.passport_view.setFont(QFont("Consolas", 10))
        self.passport_view.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.passport_view.setMinimumWidth(350)
        self.passport_view.setStyleSheet("background:#0D1117; color:#C9D1D9; border:none;")
        right_layout.addWidget(self.passport_view, stretch=1)
        
        # Progress bars section
        bars_label = QLabel("Health Metrics")
        bars_label.setStyleSheet("color:#CDD6F4; font-weight:bold; font-size:12px;")
        right_layout.addWidget(bars_label)
        
        # Metabolic Health bar (ATP level 0-5 mM)
        atp_container = QWidget()
        atp_layout = QHBoxLayout(atp_container)
        atp_layout.setContentsMargins(0, 0, 0, 0)
        atp_label = QLabel("Metabolic Health (ATP)")
        atp_label.setStyleSheet("color:#A6E3A1; font-size:10px;")
        atp_layout.addWidget(atp_label)
        self._passport_atp_bar = QProgressBar()
        self._passport_atp_bar.setRange(0, 50)  # 0-5.0 mM scaled by 10
        self._passport_atp_bar.setValue(25)  # Default 2.5 mM
        self._passport_atp_bar.setFormat("%v/50 (2.5 mM)")
        self._passport_atp_bar.setStyleSheet("""
            QProgressBar {
                background-color: #1E1E2E;
                border: 1px solid #313244;
                border-radius: 4px;
                text-align: center;
                color: #A6E3A1;
            }
            QProgressBar::chunk {
                background-color: #A6E3A1;
                border-radius: 3px;
            }
        """)
        atp_layout.addWidget(self._passport_atp_bar, stretch=1)
        right_layout.addWidget(atp_container)
        
        # Chaotic Instability bar (LLE 0-2.0)
        lle_container = QWidget()
        lle_layout = QHBoxLayout(lle_container)
        lle_layout.setContentsMargins(0, 0, 0, 0)
        lle_label = QLabel("Chaotic Instability (LLE)")
        lle_label.setStyleSheet("color:#F38BA8; font-size:10px;")
        lle_layout.addWidget(lle_label)
        self._passport_lle_bar = QProgressBar()
        self._passport_lle_bar.setRange(0, 200)  # 0-2.0 scaled by 100
        self._passport_lle_bar.setValue(0)
        self._passport_lle_bar.setFormat("%v/200 (0.00 λ)")
        self._passport_lle_bar.setStyleSheet("""
            QProgressBar {
                background-color: #1E1E2E;
                border: 1px solid #313244;
                border-radius: 4px;
                text-align: center;
                color: #F38BA8;
            }
            QProgressBar::chunk {
                background-color: #F38BA8;
                border-radius: 3px;
            }
        """)
        lle_layout.addWidget(self._passport_lle_bar, stretch=1)
        right_layout.addWidget(lle_container)
        
        scroll.setWidget(right_widget)
        layout.addWidget(scroll, stretch=1)
        
        return widget

    def _update_passport(self, result, stats: dict):
        cfg = result.config
        ch = cfg.channels
        mc = cfg.morphology

        ns = stats['n_spikes']
        V_th = stats['V_threshold']
        V_pk = stats['V_peak']
        V_ah = stats['V_ahp']
        hw = stats['halfwidth_ms']
        fi = stats['f_initial_hz']
        fs = stats['f_steady_hz']
        AI = stats['adaptation_index']
        nt = stats['neuron_type']
        nt_rule = stats.get('neuron_type_rule', nt)
        nt_ml = stats.get('neuron_type_ml', 'N/A')
        nt_ml_conf = stats.get('neuron_type_ml_confidence', np.nan)
        nt_hybrid = stats.get('neuron_type_hybrid', nt)
        nt_source = stats.get('neuron_type_hybrid_source', 'rule_only')
        nt_hybrid_conf = stats.get('neuron_type_hybrid_confidence', np.nan)
        cv = stats['conduction_vel_ms']
        tau = stats['tau_m_ms']
        Rin = stats['Rin_kohm_cm2']
        lam = stats['lambda_um']
        Q = stats['Q_per_channel']
        atp = stats['atp_nmol_cm2']

        isi_mean = stats.get('isi_mean_ms', np.nan)
        isi_std = stats.get('isi_std_ms', np.nan)
        isi_min = stats.get('isi_min_ms', np.nan)
        isi_max = stats.get('isi_max_ms', np.nan)
        cv_isi = stats.get('cv_isi', np.nan)
        lat_1st = stats.get('first_spike_latency_ms', np.nan)
        refr_per = stats.get('refractory_period_ms', np.nan)
        firing_rel = stats.get('firing_reliability', np.nan)
        lyap_class = stats.get('lyapunov_class', 'disabled')
        lyap_lle_s = stats.get('lle_per_s', np.nan)
        lyap_pairs = int(stats.get('lyapunov_valid_pairs', 0) or 0)
        modulation_valid = bool(stats.get('modulation_valid', False))
        modulation_source = stats.get('modulation_source', 'N/A')
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

        if not hasattr(result, 'currents') or not isinstance(result.currents, dict):
            logging.error("SimulationResult missing or invalid currents attribute")
            return

        for name, curr in result.currents.items():
            if curr is None or len(curr) == 0:
                continue
            curr_arr = np.asarray(curr, dtype=float)
            if curr_arr.ndim == 2:
                curr_arr = np.sum(curr_arr, axis=0)
            if curr_arr.size == 0:
                continue
            i_min = float(np.min(curr_arr))
            i_max = float(np.max(curr_arr))
            q_abs = float(np.sum(np.abs(curr_arr)) * dt_val) if dt_val > 0 else np.nan
            current_stats[name] = (i_min, i_max, q_abs)

        dominant_current = 'N/A'
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
                return 'N/A'
            return f"{v:{fmt}} {unit}".strip()

        line_major = '=' * 88
        line_minor = '-' * 88
        channels_enabled = ' '.join(
            c for c, en in [
                ('Na', True), ('K', True), ('Leak', True),
                ('Ih', ch.enable_Ih), ('ICa', ch.enable_ICa),
                ('IA', ch.enable_IA), ('SK', ch.enable_SK),
            ]
            if en
        )

        # ── METABOLIC CRISIS WARNING (v11.6) ──
        atp_min_mM = stats.get('atp_min_mM', 2.0)
        crisis_lines = []
        if atp_min_mM < 0.2:
            crisis_lines = [
                '🔴 CRITICAL: ATP < 0.2 mM — PUMPS FAILING — IMPENDING CELL DEATH',
                '🔴 КРИТИЧЕСКИЙ: АТФ < 0.2 мМ — НАСОСЫ ОСТАНАВЛИВАЮТСЯ — ГИБЕЛЬ КЛЕТКИ',
                line_major,
            ]
        elif atp_min_mM < 0.5:
            crisis_lines = [
                '🟠 WARNING: ATP < 0.5 mM — METABOLIC CRISIS — REDUCED PUMP ACTIVITY',
                '🟠 ВНИМАНИЕ: АТФ < 0.5 мМ — МЕТАБОЛИЧЕСКИЙ КРИЗИС — СНИЖЕННАЯ АКТИВНОСТЬ НАСОСОВ',
                line_major,
            ]
        
        lines = crisis_lines + [
            'NEURON PASSPORT v11.6',
            line_major,
            f'Preset class: {cfg.channels.__class__.__name__}',
            f'Temperature: {cfg.env.T_celsius:.1f} C | phi: {cfg.env.phi:.3f}',
            f'Channels: {channels_enabled}',
            line_minor,
            'PASSIVE MEMBRANE PROPERTIES',
            f'tau_m={_fmt(tau, ".3f", "ms")} | Rin={_fmt(Rin, ".3f", "kOhm*cm^2")} | lambda={_fmt(lam, ".1f", "um")}',
            line_minor,
            f'SPIKE COUNT: {ns} {"(no spikes)" if ns == 0 else ""}',
        ]

        if ns > 0:
            lines += [
                f'Threshold={_fmt(V_th, "+.1f", "mV")} | Peak={_fmt(V_pk, "+.1f", "mV")} | AHP={_fmt(V_ah, "+.1f", "mV")}',
                f'Halfwidth={_fmt(hw, ".3f", "ms")} | dV/dt={_fmt(stats["dvdt_max"], ".0f", "mV/ms")} / {_fmt(stats["dvdt_min"], ".0f", "mV/ms")}',
            ]
        if ns > 1:
            lines += [
                f'f_initial={_fmt(fi, ".1f", "Hz")} | f_steady={_fmt(fs, ".1f", "Hz")} | AI={_fmt(AI, "+.3f")}',
                f'Type(rule)={nt_rule}',
                f'Type(ML)={nt_ml} conf={_fmt(nt_ml_conf, ".2f")} source={nt_source}',
                f'Type(hybrid)={nt_hybrid} conf={_fmt(nt_hybrid_conf, ".2f")}',
            ]
        if cv > 0:
            lines.append(f'Conduction velocity={_fmt(cv, ".3f", "m/s")}')

        if ns > 1:
            lines += [
                line_minor,
                'FIRING DYNAMICS',
                f'First spike latency={_fmt(lat_1st, ".2f", "ms")} | Refractory period={_fmt(refr_per, ".3f", "ms")}',
                f'ISI mean/std={_fmt(isi_mean, ".3f", "ms")} / {_fmt(isi_std, ".3f", "ms")} | CV={_fmt(cv_isi, ".3f")}',
                f'ISI range=[{_fmt(isi_min, ".3f", "ms")}, {_fmt(isi_max, ".3f", "ms")}] | Reliability={_fmt(firing_rel, ".3f")}',
            ]

        lines += [
            line_minor,
            'DYNAMICAL STABILITY (LLE/FTLE)',
        ]
        if lyap_class == 'disabled':
            lines.append('LLE not computed. Use "Compute LLE" button.')
        else:
            lines += [
                f'Class={lyap_class} | LLE={_fmt(lyap_lle_s, "+.4f", "1/s")}',
                f'Valid trajectory pairs={lyap_pairs}',
            ]

        lines += [
            line_minor,
            'MODULATION DECOMPOSITION (NON-FFT)',
        ]
        if not modulation_valid:
            lines.append('Disabled or insufficient spikes for robust estimate.')
        else:
            lines += [
                f'Source={modulation_source} | Band={_fmt(modulation_low_hz, ".1f", "Hz")}..{_fmt(modulation_high_hz, ".1f", "Hz")}',
                f'PLV={_fmt(modulation_plv, ".3f")} | Phase={_fmt(modulation_phase_deg, ".1f", "deg")} | Nsp={modulation_spikes_used}',
                f'Depth={_fmt(modulation_depth, ".3f")} | MI={_fmt(modulation_index, ".3f")} | p={_fmt(modulation_p, ".3f")} | z={_fmt(modulation_z, ".2f")}',
            ]

        lines += [
            line_minor,
            'CHANNEL ENGAGEMENT',
            f'Dominant |Q| channel: {dominant_current}',
        ]
        top_channels = sorted(
            current_stats.items(),
            key=lambda kv: kv[1][2] if isfinite(kv[1][2]) else -1.0,
            reverse=True,
        )[:4]
        for name, (i_min, i_max, q_abs) in top_channels:
            lines.append(
                f'{name}: Imin={_fmt(i_min, ".2f", "uA/cm^2")} | Imax={_fmt(i_max, ".2f", "uA/cm^2")} | Qabs={_fmt(q_abs, ".2f", "nC/cm^2")}'
            )
        if result.n_comp > 1:
            lines.append(
                f'Delay soma->junction={_fmt(delay_junction_ms, ".2f", "ms")} | soma->terminal={_fmt(delay_terminal_ms, ".2f", "ms")}'
            )

        lines += [
            line_minor,
            'ENERGY',
        ]
        for name, q in Q.items():
            lines.append(f'Q_{name}={q:.2f} nC/cm^2')

        atp_bd = stats.get('atp_breakdown', {})
        atp_na_s = f"{atp_bd.get('Na_pump', 0.0):.3e}" if atp_bd else 'N/A'
        atp_ca_s = f"{atp_bd.get('Ca_pump', 0.0):.3e}" if atp_bd else 'N/A'
        atp_bl_s = f"{atp_bd.get('baseline', 0.0):.3e}" if atp_bd else 'N/A'
        lines += [
            f'ATP total={atp:.4e} nmol/cm^2',
            f'Na+ pump={atp_na_s} nmol/cm^2',
            f'Ca2+ pump={atp_ca_s} nmol/cm^2',
            f'Baseline={atp_bl_s} nmol/cm^2',
        ]

        # ── METABOLIC STATUS (v11.6) ──
        atp_min_mM = stats.get('atp_min_mM', 2.0)
        atp_decline = stats.get('atp_decline_rate_mM_per_s', 0.0)
        na_min = stats.get('na_i_min_mM', np.nan)
        k_max = stats.get('k_o_max_mM', np.nan)
        
        lines += [
            line_minor,
            'METABOLIC STATUS',
            f'ATP min={atp_min_mM:.3f} mM | decline={atp_decline:.4f} mM/s',
        ]
        if np.isfinite(na_min):
            lines.append(f'[Na+]i min={na_min:.2f} mM | [K+]o max={k_max:.2f} mM' if np.isfinite(k_max) else f'[Na+]i min={na_min:.2f} mM')
        
        lines.append(line_major)
        
        # ── EXPERT INSIGHTS (v11.5 bilingual) ──
        try:
            from core.expert_system import (
                generate_expert_insights, 
                get_quick_recommendations,
                format_insights_markdown,
                get_language,
                set_language
            )
            
            # Detect language from config or use default
            # Note: When full app localization is ready, this will integrate with main locale system
            current_lang = getattr(cfg, 'language', 'EN') or 'EN'
            
            # Bilingual headers
            _HEADERS = {
                'section': {
                    'EN': 'EXPERT INSIGHTS & RECOMMENDATIONS',
                    'RU': 'ЭКСПЕРТНЫЕ ИНСАЙТЫ И РЕКОМЕНДАЦИИ'
                },
                'insights': {
                    'EN': '📋 Insights:',
                    'RU': '📋 Инсайты:'
                },
                'recommendations': {
                    'EN': '💡 Recommendations:',
                    'RU': '💡 Рекомендации:'
                }
            }
            
            # Build comprehensive expert stats dictionary
            expert_stats = {
                'firing_rate_hz': (fi + fs) / 2 if ns > 0 else 0,
                'f_initial_hz': fi,
                'f_steady_hz': fs,
                'atp_min_mM': stats.get('atp_min_mM', 2.0),
                'lle_per_ms': stats.get('lle_per_ms', -1),
                'adaptation_index': AI,
                'cv_isi': cv_isi,
                'ca_i_max_nM': stats.get('ca_i_max_nM', 0),
                'threshold_rheobase_pA': stats.get('threshold_rheobase_pA', 100),
                'burst_spike_ratio': stats.get('burst_spike_ratio', 0),
                'n_spikes': ns,
                'stim_amplitude_pA': cfg.stim.Iext,
                'mean_v_mM': np.mean(result.v_soma) if len(result.v_soma) > 0 else -70,
                'temperature_celsius': cfg.env.T_celsius,
                'isi_mean_ms': isi_mean if ns > 1 else np.nan,
                'isi_std_ms': isi_std if ns > 1 else np.nan,
                'atp_decline_rate_mM_per_s': stats.get('atp_decline_rate_mM_per_s', 0),
            }
            
            insights = generate_expert_insights(expert_stats, language=current_lang)
            recommendations = get_quick_recommendations(expert_stats, language=current_lang)
            
            if insights or recommendations:
                lines += [
                    '',
                    _HEADERS['section'][current_lang],
                    line_minor,
                ]
                
                if insights:
                    lines.append(_HEADERS['insights'][current_lang])
                    lines.extend(format_insights_markdown(insights, language=current_lang).split('\n\n'))
                    lines.append('')
                
                if recommendations:
                    lines.append(_HEADERS['recommendations'][current_lang])
                    for rec in recommendations:
                        lines.append(f"  {rec}")
                    lines.append('')
                
                lines.append(line_major)
        except Exception as e:
            # Expert system is optional - fail silently in production
            # Log error in debug mode if needed
            pass

        self.passport_view.setPlainText("\n".join(lines))
        
        # v12.2: Update progress bars. Numerical diagnostics can legitimately
        # be NaN/inf for short or degenerate traces, so clamp UI values.
        atp_min_mM = self._finite_float(stats.get('atp_min_mM'), default=2.0)
        atp_scaled = max(0, min(self._passport_atp_bar.maximum(), int(atp_min_mM * 10)))
        self._passport_atp_bar.setValue(atp_scaled)
        self._passport_atp_bar.setFormat(f"{atp_scaled}/50 ({atp_min_mM:.2f} mM)")

        lyap_lle = self._finite_float(stats.get('lle_per_s'), default=0.0)
        lle_scaled = max(0, min(self._passport_lle_bar.maximum(), int(lyap_lle * 100)))
        self._passport_lle_bar.setValue(lle_scaled)
        self._passport_lle_bar.setFormat(f"{lle_scaled}/200 ({lyap_lle:.2f} lambda)")

    @staticmethod
    def _finite_float(value, *, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        if not np.isfinite(parsed):
            return default
        return parsed

    def _update_impedance(self, result):
        """Update membrane impedance magnitude/phase panels from latest run."""
        if not hasattr(self, 'fig_impedance'):
            return  # tab not yet visited
        from core.analysis import reconstruct_stimulus_trace, compute_membrane_impedance

        ax_mag, ax_phase = self.ax_impedance
        if not self._impedance_lines:
            self._impedance_lines["zmag"] = ax_mag.plot([], [], color="#2E86DE", lw=1.8, label="|Z(f)|")[0]
            self._impedance_lines["fres"] = ax_mag.axvline(0.0, color="#E67E22", ls="--", lw=1.2, visible=False)
            self._impedance_lines["zph"] = ax_phase.plot([], [], color="#8E44AD", lw=1.5, label="angle Z(f)")[0]
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
            _configure_ax_interactive(ax_mag, title="Impedance Magnitude", xlabel="Frequency (Hz)", ylabel="|Z| (kΩ·cmÂ˛)", show_legend=False)
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
            title=f"Membrane Impedance |Z(f)|  (peak={zres:.2f} kΩ·cmÂ˛ @ {fres:.2f} Hz)" if np.isfinite(fres) else "Membrane Impedance |Z(f)|",
            xlabel="Frequency (Hz)",
            ylabel="|Z| (kΩ·cmÂ˛)",
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
        """Update Lyapunov/Chaos tab with trajectory divergence plot.

        Uses native LLE (Benettin) from result.lle_convergence if available,
        otherwise falls back to post-hoc Rosenstein method from stats.
        """
        if not hasattr(self, 'fig_chaos'):
            return  # tab not yet visited

        # Check for native LLE data first (Benettin method from run_native)
        native_lle = getattr(result, 'lle_convergence', None)
        use_native = native_lle is not None and len(native_lle) > 0
        div = None  # Initialize for fallback mode (Rosenstein)

        if use_native:
            # Use native Benettin LLE convergence curve
            t = result.t  # Use simulation time vector
            lle_vals = native_lle
            duration_ms = float(t[-1] - t[0]) if len(t) > 1 else 0.0
            # Compute LLE from final value (traditional definition)
            final_lle = lle_vals[-1] if len(lle_vals) > 0 else 0.0
            lle_per_s = final_lle * 1000.0  # Convert to 1/s
            lyap_class = 'chaotic' if final_lle > 0.01 else ('neutral' if abs(final_lle) < 0.01 else 'stable')
            valid_pairs = len(lle_vals)  # Number of recorded points
            method_label = "Native (Benettin)"
        else:
            # Fallback: post-hoc Rosenstein method from analysis
            t = stats.get('ftle_time_ms', [])
            div = stats.get('ftle_log_divergence', [])
            duration_ms = float(result.t[-1] - result.t[0]) if len(result.t) > 1 else 0.0
            valid_pairs = int(stats.get('lyapunov_valid_pairs', 0) or 0)
            lyap_class = stats.get('lyapunov_class')
            lle_per_s = stats.get('lle_per_s', np.nan)
            lle_vals = None  # Will use div below
            method_label = "Rosenstein (post-hoc)"

        # Handle disabled/empty state
        has_data = use_native or (len(t) > 0 and len(div if lle_vals is None else lle_vals) > 0
                                  and lyap_class not in ('disabled', None))
        if not has_data:
            if use_native:
                msg = "Native LLE computed. Computing class..."
            elif lyap_class == 'disabled':
                msg = "LLE not computed. Click 'Compute LLE' button."
            elif duration_ms < 1000.0:
                msg = "Lyapunov analysis needs >=1000 ms of data after transients."
            elif valid_pairs < 20:
                msg = "No robust divergence fit: too few valid trajectory pairs for this trace."
            else:
                msg = "No usable divergence window was found for this simulation."
            if 'div' not in self._chaos_texts:
                self._chaos_texts['div'] = self.ax_chaos.text(
                    0.5, 0.5, msg,
                    ha='center', va='center', transform=self.ax_chaos.transAxes,
                    fontsize=11, color='#89B4FA'
                )
            else:
                self._chaos_texts['div'].set_text(msg)
            self._chaos_texts['div'].set_visible(True)
            if 'summary' in self._chaos_texts:
                self._chaos_texts['summary'].set_visible(False)
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
            if use_native:
                label = 'LLE convergence (Benettin)'
                color = '#89B4FA'
            else:
                label = 'Trajectory divergence (Rosenstein)'
                color = '#F9E2AF'  # Different color for fallback
            self._chaos_lines['div'] = self.ax_chaos.plot([], [], color=color, lw=2.5, label=label)[0]
            self._chaos_lines['trend'] = self.ax_chaos.plot([], [], color='#F38BA8', lw=1.5, ls='--', label='Linear trend')[0]
        if 'summary' not in self._chaos_texts:
            self._chaos_texts['summary'] = self.ax_chaos.text(
                0.02, 0.98, '',
                ha='left', va='top', transform=self.ax_chaos.transAxes,
                fontsize=9.5, color='#CDD6F4',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#11111B', edgecolor='#45475A', alpha=0.88),
            )

        # Plot data
        if use_native:
            # Plot LLE convergence directly (already computed per time unit)
            t_arr = np.asarray(t, dtype=float)
            lle_arr = np.asarray(lle_vals, dtype=float)
            td, yd = _downsample_xy(t_arr, lle_arr, max_points=2500)
            self._chaos_lines['div'].set_data(td, yd)
            # For native LLE, "trend" is just the mean line
            mean_lle = np.mean(lle_arr) if len(lle_arr) > 0 else 0.0
            self._chaos_lines['trend'].set_data([t_arr[0], t_arr[-1]], [mean_lle, mean_lle])
            self._chaos_lines['trend'].set_visible(True)
            ylabel = 'LLE (1/ms)'
        else:
            # Rosenstein method: plot log divergence
            div_safe = _ensure_shape_compatible(div, t, "div")
            if div_safe is not None:
                td, yd = _downsample_xy(np.asarray(t, dtype=float), np.asarray(div_safe, dtype=float), max_points=2500)
                self._chaos_lines['div'].set_data(td, yd)
            else:
                self._chaos_lines['div'].set_data([], [])
            # Linear trend line for Rosenstein
            lle_per_ms = stats.get('lle_per_ms', np.nan)
            if np.isfinite(lle_per_ms) and len(t) > 0 and div_safe is not None:
                t_trend = np.array([t[0], t[-1]])
                div_trend = div_safe[0] + lle_per_ms * (t_trend - t[0])
                self._chaos_lines['trend'].set_data(t_trend, div_trend)
                self._chaos_lines['trend'].set_visible(True)
            else:
                self._chaos_lines['trend'].set_visible(False)
            ylabel = 'Log Divergence ln(d)'

        self._chaos_texts['summary'].set_visible(True)
        self._chaos_texts['summary'].set_text(
            f"class = {lyap_class}\nLLE = {lle_per_s:+.3f} 1/s\npoints = {valid_pairs}\nmethod = {method_label}"
        )

        self.ax_chaos.relim()
        self.ax_chaos.autoscale_view()
        _configure_ax_interactive(self.ax_chaos, title='Lyapunov Exponent (Trajectory Separation)',
                                    xlabel='Time (ms)', ylabel=ylabel, show_legend=True)
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
        if hasattr(self, '_mod_soft_warning'):
            self._mod_soft_warning.set_visible(False)

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
        if hasattr(self, '_mod_soft_warning') and stats.get('modulation_low_statistical_power', False):
            self._mod_soft_warning.set_visible(True)
        self.cvs_mod.draw_idle()

    # ─────────────────────────────────────────────────────────────────
    #  2 — GATE DYNAMICS
    # ─────────────────────────────────────────────────────────────────
    def _update_gates(self, result):
        """
        Plot gate-variable time series overlaid with soma membrane potential and update interactive checkbox controls.
        
        Parameters:
            result: Simulation result object providing `t`, `v_soma`, and gate traces (as consumed by core.analysis.extract_gate_traces) used to populate and align plotted traces.
        """
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
            t_v, v_v = _downsample_xy(np.asarray(t, dtype=float), np.asarray(v_soma_safe, dtype=float), max_points=3000)
            _set_line_data(self._gates_line_v, t_v, v_v, name="gates_v_soma")
        else:
            _set_line_data(self._gates_line_v)
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
                tg, yg = _downsample_xy(np.asarray(t, dtype=float), np.asarray(trace_safe, dtype=float), max_points=3000)
                _set_line_data(line, tg, yg, name=f"gate_{name}")
            else:
                _set_line_data(line)
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
            title="Gate Dynamics",
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
            t_v, v_v = _downsample_xy(np.asarray(t, dtype=float), np.asarray(v_soma_safe, dtype=float), max_points=3000)
            _set_line_data(self._currents_line_v, t_v, v_v, name="currents_v_soma")
        else:
            _set_line_data(self._currents_line_v)
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
                tc, yc = _downsample_xy(np.asarray(t, dtype=float), np.asarray(curr_safe, dtype=float), max_points=3000)
                _set_line_data(line, tc, yc, name=f"current_{name}")
            else:
                _set_line_data(line)
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
            title="Channel Currents",
            xlabel="Time (ms)",
            ylabel="V (mV)",
            show_legend=True,
        )
        self._currents_ax2.set_ylabel("Current (µA/cmÂ˛)", fontsize=10, fontweight="bold")

        self.cvs_currents.draw_idle()

    def _on_currents_checkbox_changed(self, name: str, state: int):
        """Handle checkbox state change for current visibility."""
        self._currents_visibility[name] = (state != 0)
        if hasattr(self, '_last_result') and self._last_result is not None:
            self._update_currents(self._last_result)

    def _on_chaos_controls_changed(self, *_args):
        if getattr(self, '_last_result', None) is not None and getattr(self, '_chaos_force_enabled', False):
            self._update_chaos(self._last_result, self._last_stats or {})

    def _chaos_control_values(self) -> dict:
        return {
            'embedding_dim': int(self._chaos_embed_spin.value()) if hasattr(self, '_chaos_embed_spin') else 3,
            'lag_steps': int(self._chaos_lag_spin.value()) if hasattr(self, '_chaos_lag_spin') else 2,
            'fit_start_ms': float(self._chaos_fit_start_spin.value()) if hasattr(self, '_chaos_fit_start_spin') else 5.0,
            'fit_end_ms': float(self._chaos_fit_end_spin.value()) if hasattr(self, '_chaos_fit_end_spin') else 40.0,
            'min_separation_ms': float(self._chaos_sep_spin.value()) if hasattr(self, '_chaos_sep_spin') else 10.0,
        }

    def _on_chaos_plot_clicked(self, event):
        if not hasattr(self, 'ax_chaos') or self._chaos_analysis is None:
            return
        if event.inaxes != self.ax_chaos[0] or event.xdata is None:
            return
        t = np.asarray(self._chaos_analysis.get('ftle_time_ms', []), dtype=float)
        if len(t) == 0:
            return
        self._chaos_selected_k = int(np.argmin(np.abs(t - float(event.xdata))))
        if getattr(self, '_last_result', None) is not None:
            self._update_chaos(self._last_result, self._last_stats or {})

    def _chaos_pair_index(self, analysis: dict) -> int:
        pair_i = np.asarray(analysis.get('pair_i', []), dtype=int)
        if len(pair_i) == 0:
            return 0
        if hasattr(self, '_chaos_pair_spin'):
            self._chaos_pair_spin.blockSignals(True)
            self._chaos_pair_spin.setRange(1, max(1, len(pair_i)))
            self._chaos_pair_spin.blockSignals(False)
        mode = self._chaos_pair_mode.currentText() if hasattr(self, '_chaos_pair_mode') else 'Representative'
        if mode == 'First pair':
            return 0
        if mode == 'Pair index':
            return max(0, min(len(pair_i) - 1, int(self._chaos_pair_spin.value()) - 1))
        return len(pair_i) // 2

    def _clear_chaos_axes(self, message: str):
        """Clear chaos axes by hiding persistent artists and showing message."""
        if not hasattr(self, 'ax_chaos'):
            return
        titles = ['Trajectory divergence', 'Delay-embedded attractor', 'Source trace']
        
        # Hide all persistent artists
        for artist in getattr(self, '_chaos_persistent_artists', {}).values():
            try:
                if hasattr(artist, 'set_visible'):
                    artist.set_visible(False)
            except Exception:
                pass
        
        # Show message on each axis
        for ax, title in zip(self.ax_chaos, titles):
            _axis_message(ax, self._chaos_texts, f"msg_{title}", message, title=title)
        self.cvs_chaos.draw_idle()

    def _update_chaos(self, result, stats: dict):
        if not hasattr(self, 'fig_chaos'):
            return
        from core.analysis import estimate_ftle_lle

        if not getattr(self, '_chaos_force_enabled', False):
            self._clear_chaos_axes("LLE not computed. Click 'Compute LLE' to build the interactive analysis.")
            return

        params = self._chaos_control_values()
        analysis = estimate_ftle_lle(
            np.asarray(result.v_soma, dtype=float),
            np.asarray(result.t, dtype=float),
            is_stochastic=bool(
                getattr(result.config.stim, 'stoch_gating', False)
                or getattr(result.config.stim, 'synaptic_train_type', 'none') == 'poisson'
            ),
            **params,
        )
        self._chaos_analysis = analysis

        t_div = np.asarray(analysis.get('ftle_time_ms', []), dtype=float)
        div = np.asarray(analysis.get('ftle_log_divergence', []), dtype=float)
        pair_i = np.asarray(analysis.get('pair_i', []), dtype=int)
        pair_j = np.asarray(analysis.get('pair_j', []), dtype=int)
        emb = np.asarray(analysis.get('embedding', np.empty((0, 0))), dtype=float)
        if len(t_div) == 0 or len(div) == 0 or len(pair_i) == 0 or emb.size == 0:
            self._clear_chaos_axes('No usable divergence window was found for this simulation.')
            return

        ax_div, ax_attr, ax_trace = self.ax_chaos
        
        # Hide any error messages from previous runs
        for key in ('msg_Trajectory divergence', 'msg_Delay-embedded attractor', 'msg_Source trace'):
            _hide_axis_message(self._chaos_texts, key)

        self._chaos_selected_k = max(0, min(len(t_div) - 1, int(getattr(self, '_chaos_selected_k', 0))))
        selected_pair = self._chaos_pair_index(analysis)
        self._chaos_selected_pair = selected_pair
        k = self._chaos_selected_k
        pair_span = min(k + 1, max(1, emb.shape[0] - max(pair_i[selected_pair], pair_j[selected_pair])))

        # Update persistent divergence line
        div_line = self._chaos_persistent_artists.get('div_line')
        if div_line is not None:
            div_line.set_data(t_div, div)
            div_line.set_visible(True)
        
        # Update persistent selected point scatter
        div_scatter = self._chaos_persistent_artists.get('div_scatter')
        if div_scatter is not None:
            div_scatter.set_offsets([[t_div[k], div[k]]])
            div_scatter.set_visible(True)
        
        # Update persistent fit span
        fit_start = params['fit_start_ms']
        fit_end = params['fit_end_ms']
        fit_span = self._chaos_persistent_artists.get('fit_span')
        if fit_span is not None:
            # Use set_xy for Polygon (axvspan returns Polygon, not Rectangle)
            rect = np.array([[fit_start, 0], [fit_end, 0], [fit_end, 1], [fit_start, 1], [fit_start, 0]])
            fit_span.set_xy(rect)
            fit_span.set_visible(True)
        # Update persistent fit line
        fit_mask = (t_div >= fit_start) & (t_div <= fit_end)
        fit_line = self._chaos_persistent_artists.get('fit_line')
        if fit_line is not None:
            if np.sum(fit_mask) >= 2 and np.isfinite(analysis.get('lle_per_ms', np.nan)):
                coeffs = np.polyfit(t_div[fit_mask], div[fit_mask], 1)
                fit_line.set_data(t_div[fit_mask], np.polyval(coeffs, t_div[fit_mask]))
                fit_line.set_visible(True)
            else:
                fit_line.set_visible(False)
        
        # Update persistent summary text
        summary = (
            f"class = {analysis.get('classification', stats.get('lyapunov_class', 'n/a'))}\n"
            f"LLE = {analysis.get('lle_per_s', np.nan):+.3f} 1/s\n"
            f"pairs = {int(analysis.get('valid_pairs', 0))}\n"
            f"selected k = {k}\n"
            f"mode = {analysis.get('preprocess_mode', 'standard')}"
        )
        summary_text = self._chaos_persistent_artists.get('summary_text')
        if summary_text is not None:
            summary_text.set_text(summary)
            summary_text.set_visible(True)
        
        # Update persistent stochastic warning text
        stoch_text = self._chaos_persistent_artists.get('stoch_text')
        if stoch_text is not None:
            if analysis.get('is_stochastic', False):
                stoch_text.set_text('STOCHASTIC INPUT: LLE mainly reflects noise entropy.')
                stoch_text.set_visible(True)
            else:
                stoch_text.set_visible(False)
        _configure_ax_interactive(ax_div, title='Lyapunov Exponent (Trajectory Separation)', xlabel='Time (ms)', ylabel='Log divergence ln(d)', show_legend=True)

        # Update persistent attractor plot
        dims = emb.shape[1]
        x_idx = 0
        y_idx = 1 if dims > 1 else 0
        
        emb_line = self._chaos_persistent_artists.get('emb_line')
        if emb_line is not None:
            emb_line.set_data(emb[:, x_idx], emb[:, y_idx])
            emb_line.set_visible(True)
        
        i0 = int(pair_i[selected_pair])
        j0 = int(pair_j[selected_pair])
        seg_i = emb[i0:i0 + pair_span]
        seg_j = emb[j0:j0 + pair_span]
        
        emb_ref = self._chaos_persistent_artists.get('emb_ref')
        emb_ref_start = self._chaos_persistent_artists.get('emb_ref_start')
        if emb_ref is not None and len(seg_i) > 0:
            emb_ref.set_data(seg_i[:, x_idx], seg_i[:, y_idx])
            emb_ref.set_visible(True)
            if emb_ref_start is not None:
                emb_ref_start.set_offsets([[seg_i[0, x_idx], seg_i[0, y_idx]]])
                emb_ref_start.set_visible(True)
        elif emb_ref is not None:
            emb_ref.set_visible(False)
            if emb_ref_start is not None:
                emb_ref_start.set_visible(False)
        
        emb_neighbor = self._chaos_persistent_artists.get('emb_neighbor')
        emb_neighbor_start = self._chaos_persistent_artists.get('emb_neighbor_start')
        if emb_neighbor is not None and len(seg_j) > 0:
            emb_neighbor.set_data(seg_j[:, x_idx], seg_j[:, y_idx])
            emb_neighbor.set_visible(True)
            if emb_neighbor_start is not None:
                emb_neighbor_start.set_offsets([[seg_j[0, x_idx], seg_j[0, y_idx]]])
                emb_neighbor_start.set_visible(True)
        elif emb_neighbor is not None:
            emb_neighbor.set_visible(False)
            if emb_neighbor_start is not None:
                emb_neighbor_start.set_visible(False)
        _configure_ax_interactive(ax_attr, title='Delay-embedded attractor and selected pair', xlabel='x(t)', ylabel='x(t + lag)', show_legend=True)

        # Update persistent trace plot
        t_full = np.asarray(result.t, dtype=float)
        v_full = np.asarray(result.v_soma, dtype=float)
        
        trace_line = self._chaos_persistent_artists.get('trace_line')
        if trace_line is not None:
            trace_line.set_data(t_full, v_full)
            trace_line.set_visible(True)
        
        dt_ms = float(analysis.get('dt_ms', np.median(np.diff(t_full)) if len(t_full) > 1 else 0.1))
        span_ms = pair_span * dt_ms
        start_i_ms = t_full[min(i0, len(t_full) - 1)]
        start_j_ms = t_full[min(j0, len(t_full) - 1)]
        
        trace_ref = self._chaos_persistent_artists.get('trace_ref')
        if trace_ref is not None:
            # Use set_xy for Polygon (axvspan returns Polygon, not Rectangle)
            rect_ref = np.array([[start_i_ms, 0], [start_i_ms + span_ms, 0], [start_i_ms + span_ms, 1], [start_i_ms, 1], [start_i_ms, 0]])
            trace_ref.set_xy(rect_ref)
            trace_ref.set_visible(True)
        
        trace_neighbor = self._chaos_persistent_artists.get('trace_neighbor')
        if trace_neighbor is not None:
            # Use set_xy for Polygon
            rect_neighbor = np.array([[start_j_ms, 0], [start_j_ms + span_ms, 0], [start_j_ms + span_ms, 1], [start_j_ms, 1], [start_j_ms, 0]])
            trace_neighbor.set_xy(rect_neighbor)
            trace_neighbor.set_visible(True)
        
        _configure_ax_interactive(ax_trace, title='Time-domain trace with selected divergence segment', xlabel='Time (ms)', ylabel='V (mV)', show_legend=True)

        self.cvs_chaos.draw_idle()

    def _spike_mech_current_limit(self) -> int | None:
        if not hasattr(self, '_spike_mech_current_scope'):
            return 6
        text = self._spike_mech_current_scope.currentText()
        if text == 'All':
            return None
        return 10 if text == 'Top 10' else 6

    def _on_spike_mech_controls_changed(self, *_args):
        if getattr(self, '_spike_mech_last_result', None) is not None and getattr(self, '_spike_mech_last_stats', None) is not None:
            self._update_spike_mechanism(self._spike_mech_last_result, self._spike_mech_last_stats)

    def _on_spike_mech_clicked(self, event):
        if not hasattr(self, 'ax_spike_mech') or len(getattr(self, '_spike_mech_spike_times', [])) == 0:
            return
        if event.inaxes != self.ax_spike_mech[0] or event.xdata is None:
            return
        spike_times = np.asarray(self._spike_mech_spike_times, dtype=float)
        idx = int(np.argmin(np.abs(spike_times - float(event.xdata))))
        self._spike_mech_selected = idx
        if hasattr(self, '_spike_zoomer'):
            self._spike_zoomer.blockSignals(True)
            self._spike_zoomer.setValue(idx + 1)
            self._spike_zoomer.blockSignals(False)
        if self._spike_mech_last_result is not None:
            self._update_spike_mechanism(self._spike_mech_last_result, self._spike_mech_last_stats)

    def _reset_spike_mech_axes(self):
        """Remove dynamic spike-mechanism artists without clearing or recreating axes."""
        twin_ax = getattr(self, '_spike_mech_twin_ax', None)
        if twin_ax is not None:
            try:
                twin_ax.remove()
            except Exception:
                pass
            self._spike_mech_twin_ax = None
        for ax in getattr(self, 'ax_spike_mech', []):
            if ax.get_legend() is not None:
                ax.get_legend().remove()
            for artist in list(ax.lines) + list(ax.collections) + list(ax.texts):
                try:
                    artist.remove()
                except Exception:
                    pass
            for patch in list(ax.patches):
                if patch is ax.patch:
                    continue
                try:
                    patch.remove()
                except Exception:
                    pass
            ax.set_title("")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.grid(False)
            ax.set_axis_on()

    def _update_spike_mechanism(self, result, stats: dict):
        if not hasattr(self, 'fig_spike_mech'):
            return
        from core.analysis import detect_spikes, extract_gate_traces, spike_threshold, spike_halfwidth, after_hyperpolarization

        self._spike_mech_last_result = result
        self._spike_mech_last_stats = stats
        t = np.asarray(result.t, dtype=float)
        v = np.asarray(result.v_soma, dtype=float)
        ax1, ax2, ax3, ax4, ax5 = self.ax_spike_mech
        self._reset_spike_mech_axes()

        kwargs = _spike_detect_kwargs_from_stats(stats)
        peak_idx, spike_times, _ = detect_spikes(v, t, **kwargs)
        self._spike_mech_peak_idx = np.asarray(peak_idx, dtype=int)
        self._spike_mech_spike_times = np.asarray(spike_times, dtype=float)

        if len(peak_idx) == 0:
            for ax, title in zip(self.ax_spike_mech, ['Overview', 'Aligned spike', 'Currents', 'Gates and state', 'Interpretation']):
                ax.text(0.5, 0.5, 'No spikes detected for the current trace.', ha='center', va='center', transform=ax.transAxes, fontsize=11)
                _configure_ax_interactive(ax, title=title, xlabel='', ylabel='', show_legend=False)
            self.cvs_spike_mech.draw_idle()
            return

        self._spike_mech_selected = max(0, min(len(peak_idx) - 1, int(getattr(self, '_spike_mech_selected', 0))))
        if hasattr(self, '_spike_zoomer'):
            self._spike_zoomer.blockSignals(True)
            self._spike_zoomer.setRange(1, len(peak_idx))
            self._spike_zoomer.setValue(self._spike_mech_selected + 1)
            self._spike_zoomer.blockSignals(False)

        pk = int(peak_idx[self._spike_mech_selected])
        spike_t = float(t[pk])
        dt_ms = float(np.median(np.diff(t))) if len(t) > 1 else 0.1
        pre_ms = 4.0
        post_ms = 8.0
        lo = max(0, pk - int(pre_ms / max(dt_ms, 1e-9)))
        hi = min(len(t), pk + int(post_ms / max(dt_ms, 1e-9)))
        t_win = t[lo:hi] - spike_t
        v_win = v[lo:hi]

        ax1.plot(t, v, color='#89B4FA', lw=1.2, label='V_soma')
        ax1.scatter(t[peak_idx], v[peak_idx], color='#F38BA8', s=22, label='Spike peaks', zorder=4)
        ax1.axvspan(spike_t - pre_ms, spike_t + post_ms, color='#F9E2AF', alpha=0.18, label='Selected spike window')
        ax1.axvline(spike_t, color='#F9E2AF', ls='--', lw=1.5)
        _configure_ax_interactive(ax1, title='Spike overview', xlabel='Time (ms)', ylabel='V (mV)', show_legend=True)

        dvdt_win = np.gradient(v_win, t_win) if len(t_win) > 1 else np.zeros_like(v_win)
        ax2.plot(t_win, v_win, color='#89B4FA', lw=2.0, label='V aligned to peak')
        ax2b = ax2.twinx()
        self._spike_mech_twin_ax = ax2b
        ax2b.plot(t_win, dvdt_win, color='#F38BA8', lw=1.5, alpha=0.85, label='dV/dt')
        thr = spike_threshold(v[lo:hi], t[lo:hi]) if len(t[lo:hi]) > 3 else np.nan
        hw = spike_halfwidth(v[lo:hi], t[lo:hi], detect_algorithm='threshold_crossing') if len(t[lo:hi]) > 3 else np.nan
        ahp = after_hyperpolarization(v[lo:hi], t[lo:hi], min(pk - lo, max(0, len(v_win) - 1))) if len(v_win) > 2 else np.nan
        if np.isfinite(thr):
            thr_idx = int(np.argmin(np.abs(v_win - thr)))
            ax2.scatter([t_win[thr_idx]], [v_win[thr_idx]], color='#A6E3A1', s=45, zorder=5, label='Threshold')
        ax2.scatter([0.0], [v[pk]], color='#F9E2AF', s=55, zorder=5, label='Peak')
        if np.isfinite(ahp):
            ahp_idx = int(np.argmin(v_win))
            ax2.scatter([t_win[ahp_idx]], [v_win[ahp_idx]], color='#FAB387', s=45, zorder=5, label='AHP min')
        ax2.set_ylabel('V (mV)')
        ax2b.set_ylabel('dV/dt (mV/ms)')
        ax2.set_title(f'Aligned spike waveform | threshold={thr:.2f} mV | half-width={hw:.2f} ms')
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines2b, labels2b = ax2b.get_legend_handles_labels()
        ax2.legend(lines2 + lines2b, labels2 + labels2b, fontsize=9, loc='upper right')
        ax2.grid(alpha=0.25)

        current_limit = self._spike_mech_current_limit()
        normalize = bool(self._spike_mech_normalize.isChecked()) if hasattr(self, '_spike_mech_normalize') else True
        currents = {}
        for name, curr in getattr(result, 'currents', {}).items():
            arr = np.asarray(curr, dtype=float)
            if arr.ndim == 2:
                arr = np.sum(arr, axis=0)
            if arr.shape[0] != len(t):
                continue
            window = arr[lo:hi]
            if np.max(np.abs(window)) > 1e-9:
                currents[name] = window
        ranked = sorted(currents.items(), key=lambda kv: float(np.max(np.abs(kv[1]))), reverse=True)
        if current_limit is not None:
            ranked = ranked[:current_limit]
        for name, arr in ranked:
            y = np.asarray(arr, dtype=float)
            if normalize:
                y = y / (np.max(np.abs(y)) + 1e-12)
            ax3.plot(t_win, y, lw=1.6, color=CHAN_COLORS.get(name, '#888888'), label=name)
        _configure_ax_interactive(ax3, title='Current decomposition around selected spike', xlabel='Time from peak (ms)', ylabel='Normalized current' if normalize else 'Current (uA/cm^2)', show_legend=True)

        gates = extract_gate_traces(result)
        gate_items = sorted(gates.items(), key=lambda kv: float(np.nanmax(kv[1]) - np.nanmin(kv[1])), reverse=True)
        for name, arr in gate_items[:5]:
            arr = np.asarray(arr, dtype=float)
            if arr.shape[0] == len(t):
                ax4.plot(t_win, arr[lo:hi], lw=1.5, label=f'gate {name}', color=GATE_COLORS.get(name, '#888888'))
        if getattr(result, 'ca_i', None) is not None:
            ca_arr = np.asarray(result.ca_i, dtype=float)
            if ca_arr.ndim == 2:
                ca_arr = ca_arr[0, :]
            ca_seg = ca_arr[lo:hi]
            if np.max(np.abs(ca_seg)) > 0:
                ax4.plot(t_win, ca_seg / (np.max(np.abs(ca_seg)) + 1e-12), '--', color='#F38BA8', label='Ca_i (norm)')
        if getattr(result, 'atp_level', None) is not None:
            atp_arr = np.asarray(result.atp_level, dtype=float)
            if atp_arr.ndim == 2:
                atp_arr = atp_arr[0, :]
            atp_seg = atp_arr[lo:hi]
            if np.max(np.abs(atp_seg)) > 0:
                ax4.plot(t_win, atp_seg / (np.max(np.abs(atp_seg)) + 1e-12), ':', color='#A6E3A1', label='ATP (norm)')
        _configure_ax_interactive(ax4, title='Gate and state context', xlabel='Time from peak (ms)', ylabel='State / normalized pool', show_legend=True)

        inward = []
        outward = []
        if ranked:
            rel_idx = int(np.argmin(np.abs(t_win))) if len(t_win) else 0
            for name, arr in ranked:
                val = float(arr[rel_idx])
                if val < 0:
                    inward.append((name, abs(val)))
                elif val > 0:
                    outward.append((name, abs(val)))
        inward.sort(key=lambda item: item[1], reverse=True)
        outward.sort(key=lambda item: item[1], reverse=True)
        inward_txt = ', '.join(name for name, _ in inward[:3]) or 'none'
        outward_txt = ', '.join(name for name, _ in outward[:3]) or 'none'
        summary_lines = [
            f'Selected spike: {self._spike_mech_selected + 1} / {len(peak_idx)} at {spike_t:.2f} ms',
            f'Peak voltage: {float(v[pk]):.2f} mV',
            f'Threshold estimate: {thr:.2f} mV' if np.isfinite(thr) else 'Threshold estimate unavailable',
            f'Half-width: {hw:.2f} ms' if np.isfinite(hw) else 'Half-width unavailable',
            f'AHP minimum: {ahp:.2f} mV' if np.isfinite(ahp) else 'AHP minimum unavailable',
            f'Dominant inward currents near peak: {inward_txt}',
            f'Dominant outward currents near peak: {outward_txt}',
        ]
        if 'IM' in dict(ranked) or 'SK' in dict(ranked):
            summary_lines.append('Interpretation: slow outward currents are shaping spike termination and post-spike recovery.')
        elif 'IA' in dict(ranked):
            summary_lines.append('Interpretation: A-type potassium current is likely delaying re-excitation and sharpening onset.')
        else:
            summary_lines.append('Interpretation: classic Na/K balance dominates this spike waveform.')
        ax5.axis('off')
        ax5.text(0.02, 0.98, '\n'.join(summary_lines), ha='left', va='top', transform=ax5.transAxes, fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='#1E1E2E', alpha=0.92), color='#CDD6F4')

        self.cvs_spike_mech.draw_idle()

    def _on_spike_zoomer_changed(self, spike_num: int):
        if len(getattr(self, '_spike_mech_peak_idx', [])) == 0:
            return
        self._spike_mech_selected = max(0, min(len(self._spike_mech_peak_idx) - 1, int(spike_num) - 1))
        if self._spike_mech_last_result is not None:
            self._update_spike_mechanism(self._spike_mech_last_result, self._spike_mech_last_stats)

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
            self._equil_lines["m_inf"] = ax1.plot([], [], color=GATE_COLORS['m'], lw=2.5, label='m_inf (Na act)', alpha=0.9)[0]
            self._equil_lines["h_inf"] = ax1.plot([], [], color=GATE_COLORS['h'], lw=2.5, label='h_inf (Na inact)', alpha=0.9)[0]
            self._equil_lines["n_inf"] = ax1.plot([], [], color=GATE_COLORS['n'], lw=2.5, label='n_inf (K act)', alpha=0.9)[0]

            self._equil_lines["tau_m"] = ax2.plot([], [], color=GATE_COLORS['m'], lw=2.5, label='tau_m', alpha=0.9)[0]
            self._equil_lines["tau_h"] = ax2.plot([], [], color=GATE_COLORS['h'], lw=2.5, label='tau_h', alpha=0.9)[0]
            self._equil_lines["tau_n"] = ax2.plot([], [], color=GATE_COLORS['n'], lw=2.5, label='tau_n', alpha=0.9)[0]

            self._equil_lines["phase_m"] = ax3.plot([], [], color=GATE_COLORS['m'], lw=2, label='m (Na act)', alpha=0.9)[0]
            self._equil_lines["phase_h"] = ax3.plot([], [], color=GATE_COLORS['h'], lw=2, label='h (Na inact)', alpha=0.9)[0]

            self._equil_lines["g_na"] = ax4.plot([], [], color=GATE_COLORS['m'], lw=2.5, label='g_Na(t)', alpha=0.9)[0]
            self._equil_lines["g_k"] = ax4.plot([], [], color=GATE_COLORS['n'], lw=2.5, label='g_K(t)', alpha=0.9)[0]
            self._equil_init_done = True

        # x_inf(V) - improved layout
        self._equil_lines["m_inf"].set_data(V_rng, eq['m_inf'])
        self._equil_lines["h_inf"].set_data(V_rng, eq['h_inf'])
        self._equil_lines["n_inf"].set_data(V_rng, eq['n_inf'])
        active_opt_inf = set()
        for k in ('r_inf', 's_inf', 'u_inf', 'a_inf', 'b_inf'):
            if k in opt:
                lbl = k.replace('_inf', '_inf')
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
        _configure_ax_interactive(ax1, title='Steady-state gating (x_inf)',
                                  xlabel='V (mV)', ylabel='x_inf', show_legend=True)

        # tau(V) - main gating time constants
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
        _configure_ax_interactive(ax2, title=f'Time constants (Ď† = {phi:.2f})',
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
    def _on_phase_y_changed(self, text: str):
        if hasattr(self, '_last_result') and self._last_result is not None:
            self._update_phase(self._last_result, self._last_stats)

    def _update_phase(self, result, stats: dict):
        if not hasattr(self, 'fig_phase'):
            return  # tab not yet visited
        from core.analysis import compute_nullclines, extract_gate_traces, detect_spikes

        t = np.asarray(result.t)
        V = np.asarray(result.v_soma)
        cfg = result.config
        I_stm = cfg.stim.Iext if cfg.stim.stim_type == 'const' else 0.0

        gate_key = 'n'
        if hasattr(self, '_phase_y_combo'):
            gate_key = self._phase_y_combo.currentText().split(' ')[0]

        gates = extract_gate_traces(result)
        if gate_key not in gates:
            gate_key = 'n'
            if hasattr(self, '_phase_y_combo'):
                self._phase_y_combo.blockSignals(True)
                self._phase_y_combo.setCurrentText('n (K act)')
                self._phase_y_combo.blockSignals(False)

        gate_t = np.asarray(gates[gate_key])
        self._phase_full_data = (t, V, gate_t, cfg, I_stm, stats, gate_key)

        window_start = None
        window_end = None
        if hasattr(self, '_phase_time_start') and hasattr(self, '_phase_time_end') and len(t) > 0:
            max_time = int(t[-1])
            for slider, default in ((self._phase_time_start, 0), (self._phase_time_end, max_time)):
                old_max = int(slider.maximum())
                cur_val = int(slider.value())
                slider.blockSignals(True)
                slider.setRange(0, max_time)
                if cur_val > max_time or (old_max <= 100 and cur_val == old_max):
                    slider.setValue(default)
                slider.blockSignals(False)
            start_val = int(self._phase_time_start.value())
            end_val = int(self._phase_time_end.value())
            if end_val < start_val:
                end_val = start_val
                self._phase_time_end.blockSignals(True)
                self._phase_time_end.setValue(end_val)
                self._phase_time_end.blockSignals(False)
            if start_val <= 0 and end_val >= max_time:
                self._phase_time_label.setText("All")
                if hasattr(self, '_phase_window_source_label'):
                    self._phase_window_source_label.setText("Source: full trace")
            else:
                self._phase_time_label.setText(f"{start_val}-{end_val} ms")
                if hasattr(self, '_phase_window_source_label'):
                    self._phase_window_source_label.setText("Source: selected segment")
                window_start = start_val
                window_end = end_val

        if window_start is not None or window_end is not None:
            idx_start = 0 if window_start is None else int(np.searchsorted(t, window_start, side='left'))
            idx_end = len(t) if window_end is None else int(np.searchsorted(t, window_end, side='right'))
            idx_end = max(idx_start + 1, idx_end)
            V = V[idx_start:idx_end]
            gate_t = gate_t[idx_start:idx_end]
            t = t[idx_start:idx_end]

        ax = self.ax_phase
        if not self._phase_lines:
            self._phase_lines['traj_initial'] = ax.plot([], [], color='#2060CC', lw=1.5, zorder=3, label='Initial Spike (blue)')[0]
            self._phase_lines['traj_middle'] = ax.plot([], [], color='#888888', lw=1.5, zorder=3, label='Middle (gray)')[0]
            self._phase_lines['traj_final'] = ax.plot([], [], color='#DC5A10', lw=1.5, zorder=3, label='Final Spike (red)')[0]
            self._phase_lines['rest'] = ax.plot([], [], 'go', ms=8, zorder=5, label='Resting state')[0]
            self._phase_lines['spikes'] = ax.plot([], [], 'r*', ms=12, zorder=6, label='Spike peaks')[0]
            self._phase_lines['n_null'] = ax.plot([], [], color='#40CC40', lw=2, ls='--', label='dn/dt = 0  (n∞)')[0]
            self._phase_lines['v_null'] = ax.plot([], [], color='#CC4040', lw=2, ls='--', label='dV/dt = 0')[0]

        if len(t) > 0 and t[-1] > 100:
            t_end_initial = min(100, t[-1] / 3)
            idx_initial = np.searchsorted(t, t_end_initial)
            V_initial = V[:idx_initial]
            n_initial = gate_t[:idx_initial]

            t_start_middle = t_end_initial
            t_end_middle = t_start_middle + (t[-1] - t_start_middle) / 3
            idx_start_middle = np.searchsorted(t, t_start_middle)
            idx_end_middle = np.searchsorted(t, t_end_middle)
            V_middle = V[idx_start_middle:idx_end_middle]
            n_middle = gate_t[idx_start_middle:idx_end_middle]

            idx_final = idx_end_middle
            V_final = V[idx_final:]
            n_final = gate_t[idx_final:]
        else:
            V_initial, n_initial = V, gate_t
            self._phase_lines['traj_middle'].set_visible(False)
            self._phase_lines['traj_final'].set_visible(False)
            V_middle, n_middle = V[0:0], gate_t[0:0]
            V_final, n_final = V[0:0], gate_t[0:0]

        self._phase_lines['traj_initial'].set_data(V_initial, n_initial)
        self._phase_lines['traj_middle'].set_data(V_middle, n_middle)
        self._phase_lines['traj_final'].set_data(V_final, n_final)
        if len(V) > 0:
            self._phase_lines['rest'].set_data([V[0]], [gate_t[0]])
        else:
            self._phase_lines['rest'].set_data([], [])

        if len(t) > 0 and t[-1] > 100:
            self._phase_lines['traj_middle'].set_visible(True)
            self._phase_lines['traj_final'].set_visible(True)

        if stats['n_spikes'] > 0 and len(V) > 1 and len(t) > 1:
            pk_idx, _, _ = detect_spikes(V, t, **_spike_detect_kwargs_from_stats(stats))
            self._phase_lines['spikes'].set_data(V[pk_idx], gate_t[pk_idx])
        else:
            self._phase_lines['spikes'].set_data([], [])

        ch = cfg.channels
        complex_active = (
            ch.enable_Ih or ch.enable_ICa or ch.enable_IA or ch.enable_SK
            or ch.enable_ITCa or ch.enable_IM or ch.enable_NaP or ch.enable_NaR
        )
        nullclines_valid = (gate_key == 'n') and not complex_active

        if nullclines_valid:
            V_rng = np.linspace(-100, 60, 500)
            
            # Calculate effective reversal potentials if dynamic ATP is enabled
            ENa_eff = None
            EK_eff = None
            if cfg.metabolism.enable_dynamic_atp:
                from core.rhs import nernst_na_ion, nernst_k_ion
                
                if window_start is not None or window_end is not None:
                    idx_start = 0 if window_start is None else int(np.searchsorted(t, window_start, side='left'))
                    idx_end = len(t) if window_end is None else int(np.searchsorted(t, window_end, side='right'))
                    idx_end = max(idx_start + 1, idx_end)
                else:
                    idx_start = 0
                    idx_end = len(t)
                
                # SAFE EXTRACTION: Check for None and use 2D array slicing
                if getattr(result, 'na_i', None) is not None and getattr(result, 'k_o', None) is not None:
                    na_i_slice = result.na_i[0, idx_start:idx_end]
                    k_o_slice = result.k_o[0, idx_start:idx_end]
                    
                    na_i_mean = np.mean(na_i_slice) if len(na_i_slice) > 0 else None
                    k_o_mean = np.mean(k_o_slice) if len(k_o_slice) > 0 else None
                    
                    if na_i_mean is not None:
                        ENa_eff = nernst_na_ion(na_i_mean, cfg.env.T_celsius)
                    if k_o_mean is not None:
                        EK_eff = nernst_k_ion(k_o_mean, cfg.env.T_celsius)
            
            n_V_null, n_n_null = compute_nullclines(V_range=V_rng, config=cfg, I_stim=I_stm, ENa_eff=ENa_eff, EK_eff=EK_eff)
            self._phase_lines['n_null'].set_visible(True)
            self._phase_lines['v_null'].set_visible(True)
            self._phase_lines['n_null'].set_data(V_rng, n_n_null)
            valid_idx = ~np.isnan(n_V_null)
            self._phase_lines['v_null'].set_data(V_rng[valid_idx], n_V_null[valid_idx])
            if self._phase_warning_text is not None:
                self._phase_warning_text.set_text('')
        else:
            self._phase_lines['n_null'].set_visible(False)
            self._phase_lines['v_null'].set_visible(False)
            if self._phase_warning_text is not None:
                self._phase_warning_text.set_text('⚠ Nullclines hidden (complex channels active or non-Na/K projection)')
        if hasattr(self, '_phase_explain_label'):
            if nullclines_valid:
                self._phase_explain_label.setText(
                    f"Projection: V vs {gate_key}. Window={self._phase_time_label.text()}. "
                    "Nullclines are available here because the view is close to a classic HH-style reduced plane."
                )
            else:
                self._phase_explain_label.setText(
                    f"Projection: V vs {gate_key}. Window={self._phase_time_label.text()}. "
                    "Use this to inspect loops, threshold turns, and rebound structure; treat nullclines cautiously because extra channel families make the system higher-dimensional."
                )

        ax.set_xlabel('V (mV)', fontsize=11)
        ax.set_ylabel(f'Gate: {gate_key}', fontsize=11)

        title_suffix = f' ({self._phase_time_label.text()})' if hasattr(self, '_phase_time_label') and self._phase_time_label.text() != 'All' else ''
        ax.set_title(f'Phase Plane Trajectory{title_suffix}', fontsize=12, fontweight='bold')
        
        # Autoscale to fit data (handles non-standard projections with Ca, Ih, etc.)
        # Only apply hard limits if data stays within classic HH bounds
        ax.relim()
        ax.autoscale_view()
        
        # Apply classic HH limits as defaults, but allow expansion for non-standard gates
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        
        # V typically -100 to 60 mV for HH, but allow expansion
        if x_min >= -100 and x_max <= 60:
            ax.set_xlim(-100, 60)
        else:
            # Expand limits with 10% padding for non-standard ranges
            ax.set_xlim(x_min - 0.1*(x_max-x_min), x_max + 0.1*(x_max-x_min))
            
        # Gate typically 0 to 1 for HH gating variables, but allow expansion for Ca, etc.
        if y_min >= -0.05 and y_max <= 1.05:
            ax.set_ylim(-0.05, 1.05)
        else:
            # Expand limits with 10% padding for non-standard ranges (e.g., [Ca] in µM)
            ax.set_ylim(y_min - 0.1*(y_max-y_min), y_max + 0.1*(y_max-y_min))
        
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

        # ── Vector Field Overlay ───────────────────────────────────────
        if hasattr(self, '_cb_vector_field') and self._cb_vector_field.isChecked():
            # v11.7: Check for sufficient variance to prevent griddata singular matrix crash
            # when neuron is silent (constant V) or gate is frozen
            has_variance = len(t) > 1 and len(V) > 1 and np.std(V) > 1e-4 and np.std(gate_t) > 1e-4
            if has_variance:
                dt = float(np.median(np.diff(t)))
                dV = np.gradient(V, dt)
                dgate = np.gradient(gate_t, dt)
                # Create grid for quiver plot
                V_grid = np.linspace(-90, 40, 15)
                gate_grid = np.linspace(0, 1, 10)
                VG, GG = np.meshgrid(V_grid, gate_grid)
                # Interpolate derivatives to grid
                points = np.column_stack([V, gate_t])
                dV_grid = griddata(points, dV, (VG, GG), method='linear', fill_value=0)
                dG_grid = griddata(points, dgate, (VG, GG), method='linear', fill_value=0)
                # Normalize arrows
                mag = np.sqrt(dV_grid**2 + dG_grid**2)
                mag[mag == 0] = 1
                scale = 5.0
                if self._vector_field_quiver is None:
                    self._vector_field_quiver = ax.quiver(
                        VG, GG, dV_grid/mag*scale, dG_grid/mag*scale,
                        mag, cmap='viridis', scale=20, width=0.003, alpha=0.7
                    )
                else:
                    self._vector_field_quiver.set_offsets(np.column_stack([VG.ravel(), GG.ravel()]))
                    self._vector_field_quiver.set_UVC(
                        (dV_grid/mag*scale).ravel(),
                        (dG_grid/mag*scale).ravel(),
                        mag.ravel()
                    )
            else:
                # Remove quiver if insufficient variance (silent neuron)
                if self._vector_field_quiver is not None:
                    self._vector_field_quiver.remove()
                    self._vector_field_quiver = None
        else:
            if self._vector_field_quiver is not None:
                self._vector_field_quiver.remove()
                self._vector_field_quiver = None

        self.cvs_phase.draw_idle()

    def _on_phase_time_slider_changed(self, value: int):
        """Handle time slider change to update phase plane plot with selected time window."""
        if self._phase_full_data is None:
            return
        if hasattr(self, '_last_result') and self._last_result is not None:
            self._update_phase(self._last_result, self._last_stats)

    def _on_phase_vector_field_changed(self, state: int):
        """Handle vector field checkbox change."""
        if hasattr(self, '_last_result') and self._last_result is not None:
            self._update_phase(self._last_result, self._last_stats)

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

        # Build two axonal paths: soma -> Branch1 tip, soma -> Branch2 tip
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
        ax1.set_ylabel('Compartment (soma -> B1 tip)')
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
        ax2.set_ylabel('Compartment (soma -> B2 tip)')
        ax2.set_title('Kymograph — Path to Branch 2')

        _set_canvas_margins(self.fig_kymo, left=0.08, right=0.96, top=0.95, bottom=0.08, hspace=0.32, wspace=0.20)
        self.cvs_kymo.draw_idle()

    def _update_energy_balance(self, result):
        """Combined Energy & Balance tab with 4 rows: Balance Error, Cumulative Charge, Power, ATP Pool."""
        if not hasattr(self, 'fig_energy'):
            return  # tab not yet visited
        t   = result.t
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
            _set_line_data(self._balance_lines["abs_err"])
        else:
            _set_line_data(self._balance_lines["abs_err"], t, np.abs(I_bal) + 1e-12, name="energy_balance_error")
        ax1.set_ylabel('|Error| (µA/cmÂ˛)')
        ax1.set_title(f'Current Balance Error (log) — max|error| = {err:.5f} µA/cmÂ˛  '
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
            Q_cum = cumulative_trapezoid(np.abs(curr_arr), x=t, initial=0.0)
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
                _set_line_data(self._energy_lines[q_key], t, Q_cum_safe, name=q_key)
            else:
                _set_line_data(self._energy_lines[q_key])
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
                _set_line_data(self._energy_lines[p_key], t, P_safe, name=p_key)
                P_total += P_safe
            else:
                _set_line_data(self._energy_lines[p_key])

        for key, line in self._energy_lines.items():
            if key.startswith("Q_") and key not in active_q:
                _set_line_data(line)
            if key.startswith("P_") and key not in active_p and key != "P_total":
                _set_line_data(line)

        ax2.set_ylabel('Cumulative charge (nC/cm^2)')
        ax2.set_title('Energy - Cumulative ionic charge transfer')
        ax2.legend(fontsize=7, loc='upper left', bbox_to_anchor=(0, 1), framealpha=0.8)
        ax2.grid(alpha=0.3)
        ax2.relim()
        ax2.autoscale_view()
        ax2.tick_params(labelbottom=False)

        # Row 3: Instantaneous Power P
        if "P_total" not in self._energy_lines:
            self._energy_lines["P_total"] = ax3.plot([], [], 'k-', lw=2, label='Total', zorder=5)[0]
        P_total_safe = _ensure_shape_compatible(P_total, t, "P_total")
        if P_total_safe is not None:
            _set_line_data(self._energy_lines["P_total"], t, P_total_safe, name="P_total")
        else:
            _set_line_data(self._energy_lines["P_total"])

        ax3.set_xlabel('Time (ms)')
        ax3.set_ylabel('Power (uW/cm^2)')
        atp_mode = "dynamic ATP state" if getattr(result, 'atp_level', None) is not None else "pump-cost proxy only"
        ax3.set_title(f'Instantaneous power   ATP ~= {result.atp_estimate:.3e} nmol/cm^2   [{atp_mode}]')
        ax3.legend(fontsize=7, loc='upper left', bbox_to_anchor=(0, 1), framealpha=0.8)
        ax3.grid(alpha=0.3)
        ax3.relim()
        ax3.autoscale_view()
        ax3.tick_params(labelbottom=False)

        # ── Row 4: ATP Pool Time Series ─────────────────────────────────
        # ATP pool from pre-extracted SimulationResult state
        atp_data = None
        if getattr(result, 'atp_level', None) is not None:
            atp_arr = np.asarray(result.atp_level, dtype=float)
            if atp_arr.ndim == 2:
                atp_data = atp_arr[0, :] if atp_arr.shape[0] > 1 else atp_arr[0, :]
            else:
                atp_data = atp_arr.reshape(-1)

        # ── Row 4: ATP Pool Time Series (v11.6 Enhanced) ─────────────────
        # ATP pool from pre-extracted SimulationResult state
        atp_data = None
        if getattr(result, 'atp_level', None) is not None:
            atp_arr = np.asarray(result.atp_level, dtype=float)
            if atp_arr.ndim == 2:
                atp_data = atp_arr[0, :] if atp_arr.shape[0] > 1 else atp_arr[0, :]
            else:
                atp_data = atp_arr.reshape(-1)

        if atp_data is not None:
            _hide_axis_message(self._energy_texts, "atp_disabled")
            if self._atp_line is None:
                self._atp_line = ax4.plot([], [], color='#A6E3A1', lw=2, label='[ATP]i')[0]
            # v11.6: Dual threshold system for metabolic crisis visualization
            if self._atp_threshold_line is None:
                self._atp_threshold_line = ax4.axhline(y=0.5, color='#F9E2AF', linestyle='--', lw=1.5, label='Warning (0.5)')
                self._atp_critical_line = ax4.axhline(y=0.2, color='#F38BA8', linestyle='-', lw=2, label='Critical (0.2)')
                # Fill zones
                ax4.axhspan(0, 0.2, alpha=0.15, color='#F38BA8')  # Critical zone
                ax4.axhspan(0.2, 0.5, alpha=0.10, color='#F9E2AF')  # Warning zone

            atp_data_safe = _ensure_shape_compatible(atp_data, t, "atp_data")
            if atp_data_safe is not None:
                _set_line_data(self._atp_line, t, atp_data_safe, name="atp_level")
                # Color based on minimum ATP level
                atp_min = np.min(atp_data_safe)
                if atp_min < 0.2:
                    self._atp_line.set_color('#F38BA8')  # Red for critical
                    title_suffix = "🔴 CRITICAL ATP DEPLETION"
                elif atp_min < 0.5:
                    self._atp_line.set_color('#F9E2AF')  # Yellow for warning
                    title_suffix = "🟠 METABOLIC CRISIS"
                else:
                    self._atp_line.set_color('#A6E3A1')  # Green for healthy
                    title_suffix = "✓ Healthy"
                ax4.set_ylim(0, max(3.0, np.max(atp_data_safe) * 1.1))
            else:
                _set_line_data(self._atp_line)
                title_suffix = ""
            ax4.set_ylabel('[ATP]i (mM)')
            ax4.set_xlabel('Time (ms)')
            atp_mode = "dynamic ATP state" if getattr(result, 'atp_level', None) is not None else "pump-cost proxy only"
            ax4.set_title(f'Intracellular ATP Pool — {title_suffix} [{atp_mode}]')
            ax4.legend(fontsize=8, loc='upper right')
            ax4.grid(alpha=0.3)
            ax4.relim()
            ax4.autoscale_view()
        else:
            if self._atp_line is not None:
                _set_line_data(self._atp_line)
            _axis_message(
                ax4,
                self._energy_texts,
                "atp_disabled",
                'Enable dynamic ATP in config\nto see ATP state dynamics.\nThe row above still shows a pump-cost proxy.',
                title='Intracellular ATP Pool (Proxy Only)',
                xlabel='Time (ms)',
                ylabel='[ATP]i (mM)',
            )
        
        _set_canvas_margins(self.fig_energy, left=0.08, right=0.96, top=0.96, bottom=0.06, hspace=0.42, wspace=0.28)

        # Add crosshair and zoom to time-series axes only (not stacked bar)
        for i, ax in enumerate(self.ax_energy[:4]):  # First 4 axes (time-series plots)
            if hasattr(ax, 'crosshair'):
                continue  # Already added
            from matplotlib.widgets import Cursor
            ax.crosshair = Cursor(ax, useblit=True, color='red', linewidth=0.5, linestyle='--')
            ax.set_navigate(True)

        # ── Row 5: ATP Breakdown Horizontal Stacked Bar (top-right) ──
        atp_bd = getattr(result, 'atp_breakdown', None)
        if atp_bd is None or not isinstance(atp_bd, dict):
            atp_bd = {}
        na_pump = atp_bd.get('Na_pump', 0.0)
        ca_pump = atp_bd.get('Ca_pump', 0.0)
        baseline = atp_bd.get('baseline', 0.0)
        total = atp_bd.get('total', 0.0)

        if total > 0:
            pcts = [100.0 * na_pump / total, 100.0 * ca_pump / total, 100.0 * baseline / total]
            left = 0.0
            for patch, w in zip(self._atp_bar_patches, pcts):
                patch.set_width(w)
                patch.set_x(left)
                patch.set_visible(True)
                left += w
            # Highlight Ca2+ pump in red if > 30% (metabolic stress)
            ca_ratio = ca_pump / total
            if ca_ratio > 0.3:
                self._atp_bar_patches[1].set_facecolor('#FF0000')
            else:
                self._atp_bar_patches[1].set_facecolor(CHAN_COLORS.get('ICa', '#FA9600'))
            title_suffix = ' (STRESS)' if ca_ratio > 0.3 else ''
            ax5.set_title(f'ATP Breakdown (Tot: {total:.2e}){title_suffix}', fontsize=10)
            self._atp_bar_no_data_text.set_visible(False)
        else:
            for patch in self._atp_bar_patches:
                patch.set_visible(False)
            self._atp_bar_no_data_text.set_visible(True)
            ax5.set_title('ATP Breakdown', fontsize=10)

        _set_canvas_margins(self.fig_energy, left=0.08, right=0.96, top=0.96, bottom=0.06, hspace=0.42, wspace=0.28)
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

        for artist in getattr(self, '_spike_shape_dynamic_artists', []):
            try:
                artist.remove()
            except Exception:
                pass
        self._spike_shape_dynamic_artists = []
        self._spike_shape_empty_text.set_visible(False)

        if n_sp == 0:
            self._spike_shape_empty_text.set_text('No spikes detected')
            self._spike_shape_empty_text.set_visible(True)
            _configure_ax_interactive(
                ax,
                title='Spike Shape Overlay',
                xlabel='Time relative to peak (ms)',
                ylabel='V (mV)',
                show_legend=False,
            )
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

        # Build LineCollection for performance (much faster than looped ax.plot)
        # Each spike becomes a line segment in the collection
        from matplotlib.collections import LineCollection
        
        segments = []
        for spike_t, spike_v in spikes:
            # Create segment as array of (x, y) points
            segment = np.column_stack([spike_t, spike_v])
            segments.append(segment)
        
        if color_by_index:
            # Color by absolute spike index (shows evolution)
            colors = plt.cm.viridis(np.linspace(0, 1, n_sp))
            line_colors = [colors[spike_indices[i] - 1] for i in range(len(spikes))]
        else:
            # Single color for all selected spikes
            soma_color = PLOT_THEMES.get("Default", PLOT_THEMES["Default"]).get("soma", "#4080FF")
            line_colors = [soma_color] * len(spikes)
        
        # Create LineCollection - single artist for all spikes
        lc = LineCollection(segments, colors=line_colors, linewidths=1.5, alpha=0.7)
        ax.add_collection(lc)
        self._spike_shape_dynamic_artists.append(lc)
        
        # Autoscale view to fit all spike segments (LineCollection doesn't auto-update limits)
        ax.autoscale_view()

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
    #  17 — POINCARĂ‰ PLOT (ISI DYNAMICS)
    # ─────────────────────────────────────────────────────────────────
    def _update_poincare(self, result, stats: dict):
        """Poincare plot of ISI dynamics: ISI[n+1] vs ISI[n]."""
        if not hasattr(self, 'fig_poincare'):
            return  # tab not yet visited
        from core.analysis import detect_spikes

        t = np.asarray(result.t, dtype=float)
        v = np.asarray(result.v_soma, dtype=float)
        ax = self.ax_poincare

        kwargs = _spike_detect_kwargs_from_stats(stats)
        peak_idx, spike_times, _ = detect_spikes(v, t, **kwargs)
        n_sp = len(spike_times)
        self._poincare_lines["msg"].set_visible(False)

        if n_sp < 3:
            self._poincare_lines["scatter"].set_offsets(np.empty((0, 2)))
            self._poincare_lines["scatter"].set_array(np.array([]))
            _set_line_data(self._poincare_lines["diag"])
            self._poincare_lines["msg"].set_text('Need >=3 spikes for Poincare plot')
            self._poincare_lines["msg"].set_visible(True)
            _configure_ax_interactive(
                ax,
                title='Poincare Plot (ISI Dynamics)',
                xlabel='ISI[n] (ms)',
                ylabel='ISI[n+1] (ms)',
                show_legend=False,
            )
            self.cvs_poincare.draw_idle()
            return

        # Calculate ISIs
        isi = np.diff(spike_times)

        # Poincare plot: ISI[n+1] vs ISI[n]
        isi_n = isi[:-1]
        isi_n_plus_1 = isi[1:]

        offsets = np.column_stack([isi_n, isi_n_plus_1])
        self._poincare_lines["scatter"].set_offsets(offsets)
        self._poincare_lines["scatter"].set_array(np.arange(len(isi_n), dtype=float))

        # Diagonal line (ISI[n+1] = ISI[n])
        isi_min = min(isi_n.min(), isi_n_plus_1.min())
        isi_max = max(isi_n.max(), isi_n_plus_1.max())
        _set_line_data(self._poincare_lines["diag"], [isi_min, isi_max], [isi_min, isi_max], name="poincare_diag")

        _configure_ax_interactive(
            ax,
            title=f'Poincare Plot (ISI Dynamics, N={n_sp})',
            xlabel='ISI[n] (ms)',
            ylabel='ISI[n+1] (ms)',
            show_legend=True,
        )
        ax.grid(alpha=0.2)
        ax.set_aspect('equal', adjustable='datalim')

        self.cvs_poincare.draw_idle()

    # ─────────────────────────────────────────────────────────────────
    #  18 — ISI DISTRIBUTION (Shannon Entropy Dashboard)
    # ─────────────────────────────────────────────────────────────────
    def _update_isi_dist(self, result, stats: dict):
        """Update ISI Distribution histogram with Shannon entropy metrics."""
        if not hasattr(self, 'fig_isi_dist'):
            return

        from core.analysis import detect_spikes, shannon_entropy_isi

        ax_hist, ax_metrics = self._isi_dist_axes

        # Get spike times
        if stats and 'spike_times' in stats:
            spike_times = np.asarray(stats['spike_times'])
        else:
            t = np.asarray(result.t, dtype=float)
            v = np.asarray(result.v_soma, dtype=float)
            peak_idx, spike_times, _ = detect_spikes(v, t)

        n_spikes = len(spike_times)

        # Need at least 2 spikes for ISI
        if n_spikes < 2:
            ax_hist.clear()
            ax_hist.text(0.5, 0.5, 'Need ≥2 spikes for ISI distribution',
                        ha='center', va='center', transform=ax_hist.transAxes,
                        fontsize=12, color='#89B4FA')
            ax_metrics.clear()
            ax_metrics.set_xlim(0, 1)
            ax_metrics.set_ylim(0, 1)
            ax_metrics.axis('off')
            self.cvs_isi_dist.draw_idle()
            return

        # Calculate ISIs
        isi = np.diff(spike_times)

        # Histogram
        ax_hist.clear()
        bins = min(20, max(5, n_spikes // 2))
        counts, bin_edges, patches = ax_hist.hist(isi, bins=bins, alpha=0.7, color='#89B4FA', edgecolor='#45475A')

        # Fit exponential if enough data
        if len(isi) >= 5:
            try:
                loc, scale = expon.fit(isi)
                x_fit = np.linspace(isi.min(), isi.max(), 100)
                y_fit = expon.pdf(x_fit, loc, scale) * len(isi) * (bin_edges[1] - bin_edges[0])
                ax_hist.plot(x_fit, y_fit, 'r--', lw=2, label=f'Exp fit (τ={scale:.1f}ms)')
                ax_hist.legend(fontsize=9)
            except Exception:
                pass

        ax_hist.set_xlabel('ISI (ms)')
        ax_hist.set_ylabel('Count')
        ax_hist.set_title(f'ISI Distribution (N={n_spikes} spikes, {len(isi)} ISIs)')
        ax_hist.grid(alpha=0.2)

        # Metrics panel with Shannon entropy
        ax_metrics.clear()
        ax_metrics.set_xlim(0, 1)
        ax_metrics.set_ylim(0, 1)
        ax_metrics.axis('off')

        # Compute statistics
        mean_isi = np.mean(isi)
        std_isi = np.std(isi)
        cv_isi = std_isi / mean_isi if mean_isi > 0 else 0

        # Shannon entropy (need at least 5 ISIs for meaningful calculation)
        n_bins = max(2, min(20, len(isi) // 2)) if len(isi) >= 5 else 2
        if len(isi) >= 5:
            entropy_bits = shannon_entropy_isi(spike_times, bins=n_bins)
        else:
            entropy_bits = 0.0  # Not enough data for meaningful entropy

        # Max entropy for comparison (uniform distribution)
        max_entropy = np.log2(n_bins) if n_bins > 0 else 0
        entropy_ratio = entropy_bits / max_entropy if max_entropy > 0 else 0

        # Regularity metrics
        is_regular = cv_isi < 0.1
        is_bursting = np.any(isi[:-1] / np.maximum(isi[1:], 1e-10) > 3)

        # Format metrics text
        metrics_text = (
            f"ISI Statistics:\n"
            f"  Mean: {mean_isi:.2f} ms\n"
            f"  Std:  {std_isi:.2f} ms\n"
            f"  CV:   {cv_isi:.3f} {'(regular)' if is_regular else '(irregular)'}\n"
            f"\n"
            f"Shannon Entropy:\n"
            f"  H = {entropy_bits:.2f} bits\n"
            f"  H/Hmax = {entropy_ratio:.2%}\n"
            f"  Pattern: {'Regular/Poisson' if entropy_ratio > 0.7 else 'Bursting/Structured' if is_bursting else 'Adaptive/Mixed'}\n"
            f"\n"
            f"Classification:\n"
            f"  {'✓ Regular spiking' if is_regular else '⚡ Irregular spiking'}\n"
            f"  {'⚡ Bursting detected' if is_bursting else '✓ No bursting'}"
        )

        ax_metrics.text(0.05, 0.95, metrics_text, transform=ax_metrics.transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       color='#CDD6F4',
                       bbox=dict(boxstyle='round', facecolor='#1E1E2E', alpha=0.8, edgecolor='#45475A'))

        self.cvs_isi_dist.draw_idle()

    # ─────────────────────────────────────────────────────────────────
    #  12 — SPECTROGRAM  (STFT of soma Vm)
    # ─────────────────────────────────────────────────────────────────
    def _update_spectrogram(self, result):
        if not hasattr(self, 'fig_spectro'):
            return  # tab not yet visited
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

        # STFT — fs in Hz, dt in ms -> fs = 1000/dt
        dt_ms = float(t[1] - t[0]) if len(t) > 1 else 0.05
        fs_hz = 1000.0 / dt_ms

        # Physiological window: target 500ms (Theta/Alpha/Gamma), capped at 25% of trace
        target_window_ms = min(500.0, (t[-1] - t[0]) * 0.25)
        n_seg = min(2048, max(64, int(target_window_ms / dt_ms)))
        n_overlap = n_seg * 3 // 4

        try:
            freqs, times_stft, Zxx = stft(v, fs=fs_hz, nperseg=n_seg,
                                           noverlap=n_overlap, window='hann')
            # Convert STFT output times to simulation time axis
            t_stft = t[0] + times_stft * 1000.0  # seconds -> ms

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

        _set_canvas_margins(self.fig_spectro, left=0.08, right=0.96, top=0.94, bottom=0.08, hspace=0.28, wspace=0.20)
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
        if not bif_data:
            if hasattr(self, '_bif_summary_label'):
                self._bif_summary_label.setText(
                    f"No bifurcation samples yet for {param_name}. Run the scan to reveal rest, tonic, burst, or block transitions."
                )
            for line in getattr(self, '_bif_lines', {}).values():
                try:
                    line.set_data([], [])
                except Exception:
                    pass
            self.cvs_bif.draw_idle()
            return
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

        if hasattr(self, '_bif_summary_label'):
            active = n_sp > 0
            if not np.any(active):
                summary = f"{param_name}: rest-only in scanned range."
            else:
                onset = float(vals[np.argmax(active)])
                max_freq = float(np.nanmax(freq[active])) if np.any(np.isfinite(freq[active])) else 0.0
                max_spikes = int(np.nanmax(n_sp[active])) if np.any(active) else 0
                block_like = bool(np.any(active) and n_sp[-1] == 0 and np.any(n_sp[:-1] > 0))
                if max_spikes >= 4 and max_freq < 25.0:
                    regime = "burst-prone"
                elif max_freq >= 25.0:
                    regime = "tonic-spiking"
                else:
                    regime = "low-rate / near-threshold"
                suffix = " High-end block is suggested." if block_like else ""
                summary = (
                    f"{param_name}: onset around {onset:.2f}. Dominant regime looks {regime}; "
                    f"peak firing rate ≈ {max_freq:.1f} Hz, max spike count {max_spikes}.{suffix}"
                )
            self._bif_summary_label.setText(summary)

        _set_canvas_margins(self.fig_bif, left=0.08, right=0.96, top=0.94, bottom=0.07, hspace=0.42, wspace=0.26)
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

        if hasattr(self, '_sweep_summary_label'):
            if not param_vals:
                self._sweep_summary_label.setText(
                    f"No successful sweep traces for {param_name}. Check solver stability or sweep bounds."
                )
            else:
                active = [i for i, nsp in enumerate(n_sps) if nsp > 0]
                max_freq = float(np.nanmax(freqs)) if len(freqs) else 0.0
                max_spikes = int(np.nanmax(n_sps)) if len(n_sps) else 0
                if active:
                    onset_val = float(param_vals[active[0]])
                    onset_txt = f"onset ≈ {onset_val:.2f}"
                else:
                    onset_txt = "no spike onset in range"
                mode_txt = (
                    "Dedicated f-I readout"
                    if str(param_name).lower() in {"iext", "stim.iext"}
                    else "General sweep"
                )
                self._sweep_summary_label.setText(
                    f"{mode_txt}: {param_name} scan with {len(param_vals)} successful samples, {onset_txt}, "
                    f"peak firing rate ≈ {max_freq:.1f} Hz, max spike count {max_spikes}."
                )

        _set_canvas_margins(self.fig_sweep, left=0.08, right=0.96, top=0.94, bottom=0.07, hspace=0.42, wspace=0.26)
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
                                               label=f'Rheobase = {I_rh:.2f} µA/cmÂ˛')
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
        ax1.set_xlabel('Pulse duration (ms)');  ax1.set_ylabel('I threshold (µA/cmÂ˛)')
        ax1.set_title('Strength-Duration Curve');  ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)
        ax1.relim(); ax1.autoscale_view()

        self._sd_lines["qth"].set_data(dur, Q_th)
        ax2.set_xlabel('Pulse duration (ms)');  ax2.set_ylabel('Charge threshold (nC/cm²)')
        ax2.set_title('Minimum charge vs duration')
        ax2.legend();  ax2.grid(alpha=0.3)
        ax2.relim(); ax2.autoscale_view()

        _set_canvas_margins(self.fig_sd, left=0.08, right=0.96, top=0.94, bottom=0.08, hspace=0.36, wspace=0.24)
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
        ax1.set_xlabel('Duration (ms)');  ax1.set_ylabel('I_ext (µA/cmÂ˛)')
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
        ax2.set_xlabel('Duration (ms)');  ax2.set_ylabel('I_ext (µA/cmÂ˛)')
        ax2.set_title('Mean frequency map')

        _set_canvas_margins(self.fig_excmap, left=0.08, right=0.96, top=0.94, bottom=0.08, hspace=0.32, wspace=0.24)
        self.cvs_excmap.draw_idle()
        self.setCurrentWidget(self.tab_excmap)

    # ─────────────────────────────────────────────────────────────────
    #  12 — FINGERPRINT (Radar Chart)
    # ─────────────────────────────────────────────────────────────────
    # Reference: Healthy Pyramidal L5 neuron baseline (normalized 0-1)
    _REFERENCE_BASELINE = [0.5, 0.5, 0.5, 0.5, 0.3, 0.5]  # [rate, adapt, hw, Rin, energy, P/T]

    def _update_fingerprint(self, result, stats):
        """Update the Radar Chart (Biophysical Fingerprint) tab."""
        if not hasattr(self, 'fig_fingerprint'):
            return
        from core.analysis import compute_biophysical_metrics

        ax = self.ax_fingerprint
        metrics = compute_biophysical_metrics(result, stats)

        # Get normalized values for each axis
        values = [
            metrics['firing_rate_norm'],
            metrics['adaptation_norm'],
            metrics['halfwidth_norm'],
            metrics['resistance_norm'],
            metrics['energy_norm'],
            metrics['peak_threshold_norm'],
        ]
        values_closed = values + values[:1]

        # Reference baseline (healthy pyramidal) — for comparison
        ref_values = self._REFERENCE_BASELINE + self._REFERENCE_BASELINE[:1]

        # Update or create the radar chart elements
        if self._radar_fill is None:
            # Reference polygon (dashed gray)
            self._radar_ref_line = ax.plot(
                self._radar_angles, ref_values, '--', linewidth=1.5,
                color='#6C7086', alpha=0.7, label='Healthy L5 Ref'
            )[0]
            self._radar_fill = ax.fill(
                self._radar_angles, values_closed, alpha=0.25, color='#89B4FA'
            )[0]
            self._radar_line = ax.plot(
                self._radar_angles, values_closed, 'o-', linewidth=2,
                color='#89B4FA', label='Current', markersize=6
            )[0]
            # Add legend
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
                     fontsize=8, framealpha=0.8, facecolor='#1E1E2E')
        else:
            self._radar_fill.set_xy(np.column_stack([self._radar_angles, values_closed]))
            self._radar_line.set_data(self._radar_angles, values_closed)

        # Compute deviation from reference for summary
        deviations = [
            (values[i] - self._REFERENCE_BASELINE[i]) / max(0.01, self._REFERENCE_BASELINE[i]) * 100
            for i in range(6)
        ]
        avg_deviation = np.mean(np.abs(deviations))

        # Color-coded summary
        summary = (
            f"Firing Rate: {metrics['firing_rate_hz']:.1f} Hz "
            f"({values[0]:.2f}) {self._deviation_arrow(deviations[0])}\n"
            f"Adaptation: {metrics['adaptation_index']:.2f} "
            f"({values[1]:.2f}) {self._deviation_arrow(deviations[1])}\n"
            f"Half-width: {metrics['spike_halfwidth_ms']:.2f} ms "
            f"({values[2]:.2f}) {self._deviation_arrow(deviations[2])}\n"
            f"Input R: {metrics['input_resistance_mohm']:.1f} MΩ·cm² "
            f"({values[3]:.2f}) {self._deviation_arrow(deviations[3])}\n"
            f"Energy: {metrics['energy_cost_atp']:.2e} "
            f"({values[4]:.2f}) {self._deviation_arrow(deviations[4])}\n"
            f"P/T Ratio: {metrics['peak_threshold_ratio']:.2f} "
            f"({values[5]:.2f}) {self._deviation_arrow(deviations[5])}\n"
            f"────────────────────\n"
            f"Avg Deviation: {avg_deviation:.1f}%"
        )

        if self._radar_text is None:
            self._radar_text = ax.text(
                1.35, 0.5, summary, transform=ax.transAxes, fontsize=9,
                verticalalignment='center', color='#CDD6F4',
                bbox=dict(boxstyle='round', facecolor='#1E1E2E', alpha=0.8),
                fontfamily='monospace'
            )
        else:
            self._radar_text.set_text(summary)

        ax.set_ylim(0, 1)
        self.cvs_fingerprint.draw_idle()

    def _deviation_arrow(self, pct_dev: float) -> str:
        """Return arrow indicator for deviation from reference."""
        if abs(pct_dev) < 10:
            return "●"  # Within 10% — close to reference
        elif pct_dev > 0:
            return "↑" if pct_dev < 50 else "↑↑"  # Elevated
        else:
            return "↓" if pct_dev > -50 else "↓↓"  # Reduced

    # ─────────────────────────────────────────────────────────────────
    #  13 — CSD (Current-Source Density)
    # ─────────────────────────────────────────────────────────────────
    def _update_csd(self, result):
        """Update the CSD spatial heatmap tab."""
        if not hasattr(self, 'fig_csd'):
            return
        if not hasattr(result, 'v_all') or result.v_all is None:
            return
        if result.v_all.shape[0] < 3:
            return  # Need at least 3 compartments for CSD

        # Stop any running animation when new result arrives
        if hasattr(self, '_csd_play_timer') and self._csd_play_timer is not None:
            if self._csd_play_timer.isActive():
                self._csd_play_timer.stop()
            self._csd_play_timer = None
        if hasattr(self, '_btn_csd_play'):
            self._btn_csd_play.setText("▶ Play")

        from core.analysis import compute_csd

        # Store result for animation and export
        self._csd_current_result = result

        ax_csd, ax_vm = self._csd_axes
        ax_profile = self._csd_profile_ax

        # Compute CSD using graph Laplacian for correct branched morphology
        # CSD = -L @ V represents the divergence of axial currents, which equals
        # the net transmembrane current (C_m dV/dt + I_ion - I_stim).
        # Pump current is already included through I_ion contribution to V dynamics,
        # so no additional terms are needed here.
        csd = compute_csd(result.v_all, result.morph)

        t = result.t
        n_comp = csd.shape[0]
        x_pos = np.arange(n_comp)  # Compartment indices as spatial position

        # Create meshgrid for heatmap
        T, X = np.meshgrid(t, x_pos)

        # Plot CSD heatmap with auto-scaling (handle spike transients up to ±1000)
        csd_vmax = max(100.0, np.percentile(np.abs(csd), 99.0))
        if self._csd_im is None:
            self._csd_im = ax_csd.pcolormesh(T, X, csd, cmap='RdBu_r', shading='auto', vmin=-csd_vmax, vmax=csd_vmax)
            self._csd_cbar = self.fig_csd.colorbar(self._csd_im, ax=ax_csd, label='CSD (mV/cm²)')
        else:
            self._csd_im.set_array(csd.ravel())
            if self._csd_cbar is not None:
                self._csd_cbar.update_normal(self._csd_im)

        ax_csd.set_xlabel('Time (ms)')
        ax_csd.set_ylabel('Compartment')
        ax_csd.set_title('Current-Source Density (CSD): Sinks (red) = ion entry, Sources (blue) = ion exit')

        # Plot voltage heatmap below for comparison
        if self._csd_vm_line is None:
            self._csd_vm_line = ax_vm.pcolormesh(T, X, result.v_all, cmap='viridis', shading='auto')
            self.fig_csd.colorbar(self._csd_vm_line, ax=ax_vm, label='Vm (mV)')
        else:
            self._csd_vm_line.set_array(result.v_all.ravel())

        ax_vm.set_xlabel('Time (ms)')
        ax_vm.set_ylabel('Compartment')
        ax_vm.set_title('Membrane Potential (Vm)')

        # Add vertical line on heatmaps showing current time position
        if hasattr(self, '_csd_time_slider'):
            slider_pos = self._csd_time_slider.value() / 100.0
        else:
            slider_pos = 0.5  # Default to middle
        time_idx = int(slider_pos * (len(t) - 1))
        current_t = t[time_idx]

        # Draw/update time marker line
        if hasattr(self, '_csd_time_line_csd'):
            self._csd_time_line_csd.set_xdata([current_t, current_t])
        else:
            self._csd_time_line_csd = ax_csd.axvline(x=current_t, color='white', lw=2, alpha=0.7)
        if hasattr(self, '_csd_time_line_vm'):
            self._csd_time_line_vm.set_xdata([current_t, current_t])
        else:
            self._csd_time_line_vm = ax_vm.axvline(x=current_t, color='white', lw=2, alpha=0.7)

        # Update spatial profile plot
        self._update_csd_profile(csd, result.v_all, x_pos, time_idx, current_t)

        _set_canvas_margins(self.fig_csd, left=0.08, right=0.94, top=0.95, bottom=0.10, hspace=0.38, wspace=0.30)
        self.cvs_csd.draw_idle()

    def _update_csd_profile(self, csd, v_all, x_pos, time_idx, current_t):
        """Update the spatial profile plot at current time."""
        ax_profile = self._csd_profile_ax

        # Get CSD and Vm profiles at current time
        csd_profile = csd[:, time_idx]
        vm_profile = v_all[:, time_idx]

        # Plot CSD profile
        if self._csd_profile_line is None:
            self._csd_profile_line = ax_profile.plot(csd_profile, x_pos, 'b-', lw=2, label='CSD')[0]
            # Fill positive (sinks) in red, negative (sources) in blue
            self._csd_profile_fill_pos = ax_profile.fill_betweenx(
                x_pos, 0, csd_profile, where=(csd_profile >= 0), alpha=0.3, color='red', interpolate=True
            )
            self._csd_profile_fill_neg = ax_profile.fill_betweenx(
                x_pos, 0, csd_profile, where=(csd_profile < 0), alpha=0.3, color='blue', interpolate=True
            )
            ax_profile.set_xlabel('CSD (mV/cm²)', color='#89B4FA')
            ax_profile.set_ylabel('Compartment')
            ax_profile.set_title(f'Spatial Profile @ t={current_t:.1f} ms')
            ax_profile.axvline(x=0, color='gray', lw=0.5, linestyle='--')
            ax_profile.grid(alpha=0.3)
        else:
            self._csd_profile_line.set_data(csd_profile, x_pos)
            ax_profile.set_title(f'Spatial Profile @ t={current_t:.1f} ms')

        ax_profile.set_xlim(-max(100, np.max(np.abs(csd_profile)) * 1.1),
                          max(100, np.max(np.abs(csd_profile)) * 1.1))

    def _on_csd_time_changed(self, value):
        """Handle CSD time slider change."""
        if self._csd_current_result is None:
            return

        # Update label
        self._lbl_csd_time.setText(f"<span style='color:#89B4FA'>{value}%</span>")

        # Update CSD plot with new time position
        self._update_csd(self._csd_current_result)

    def _on_csd_play_clicked(self):
        """Animate CSD spatial profile over time."""
        from PySide6.QtCore import QTimer

        if self._csd_play_timer is not None and self._csd_play_timer.isActive():
            # Stop animation
            self._csd_play_timer.stop()
            self._btn_csd_play.setText("▶ Play")
            return

        if self._csd_current_result is None:
            return
        if not hasattr(self, '_csd_time_slider'):
            return

        # Start animation
        self._btn_csd_play.setText("⏸ Pause")

        def advance_frame():
            if not hasattr(self, '_csd_time_slider'):
                return
            current = self._csd_time_slider.value()
            if current >= 100:
                self._csd_time_slider.setValue(0)
            else:
                self._csd_time_slider.setValue(current + 2)  # +2% per frame

        self._csd_play_timer = QTimer(self)
        self._csd_play_timer.timeout.connect(advance_frame)
        self._csd_play_timer.start(100)  # 100ms = 10 fps

    def _on_csd_export_clicked(self):
        """Export CSD spatial profile to CSV."""
        import numpy as np
        from PySide6.QtWidgets import QFileDialog

        if self._csd_current_result is None:
            return
        if not hasattr(self, '_csd_time_slider'):
            return

        # Get current time index
        slider_pos = self._csd_time_slider.value() / 100.0
        t = self._csd_current_result.t
        time_idx = int(slider_pos * (len(t) - 1))
        current_t = t[time_idx]

        # Compute CSD using graph Laplacian
        from core.analysis import compute_csd
        csd = compute_csd(self._csd_current_result.v_all, self._csd_current_result.morph)

        # Prepare data
        n_comp = csd.shape[0]
        x_pos = np.arange(n_comp)
        csd_profile = csd[:, time_idx]
        vm_profile = self._csd_current_result.v_all[:, time_idx]

        # Export dialog
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export CSD Profile", f"csd_profile_t{current_t:.1f}ms.csv",
            "CSV files (*.csv);;All files (*)"
        )

        if filename:
            try:
                np.savetxt(
                    filename,
                    np.column_stack([x_pos, csd_profile, vm_profile]),
                    delimiter=',',
                    header='Compartment,CSD_mV_per_cm2,Vm_mV',
                    comments='',
                    fmt='%.6f'
                )
                print(f"✅ CSD profile exported to: {filename}")
            except Exception as e:
                print(f"❌ Export failed: {e}")

    # ─────────────────────────────────────────────────────────────────
    #  14 — DEBUG LTE (Local Truncation Error) Monitor
    # ─────────────────────────────────────────────────────────────────
    def _update_debug(self, result):
        """Update the Debug LTE (Local Truncation Error) Monitor tab."""
        if not hasattr(self, 'fig_debug'):
            return

        ax_lte, ax_dt = self._debug_axes
        t = result.t

        # Check if we have the necessary data for LTE computation
        if hasattr(result, 'solver_residual') and result.solver_residual is not None:
            lte_data = np.asarray(result.solver_residual)
        else:
            # Fallback: compute a proxy from voltage derivative discontinuity
            if len(t) > 2:
                V = result.v_soma
                dt = np.median(np.diff(t))
                # Estimate residual from second derivative
                d2V = np.gradient(np.gradient(V, dt), dt)
                # Scale by dt for a proxy of truncation error
                lte_data = np.abs(d2V) * (dt ** 2) / 2.0
            else:
                lte_data = np.zeros_like(t)

        # Plot LTE over time
        if self._lte_lines is None:
            self._lte_lines = ax_lte.plot(t, lte_data, color='#F38BA8', lw=1.5)[0]
        else:
            self._lte_lines.set_data(t, lte_data)

        ax_lte.set_xlabel('Time (ms)')
        ax_lte.set_ylabel('LTE (mV)')
        ax_lte.set_title('Local Truncation Error Monitor')
        ax_lte.grid(alpha=0.3)

        # Add threshold warning line at 1.0 mV
        if self._lte_threshold_line is None:
            self._lte_threshold_line = ax_lte.axhline(y=1.0, color='#FAB387', linestyle='--', linewidth=2, label='Warning threshold (1.0 mV)')
        ax_lte.legend(fontsize=9)

        # Check for excessive error and suggest adaptive timestep
        max_lte = np.max(lte_data) if len(lte_data) > 0 else 0.0
        mean_lte = np.mean(lte_data) if len(lte_data) > 0 else 0.0
        current_dt = np.median(np.diff(t)) if len(t) > 1 else 0.05

        # Smart adaptive timestep calculation
        # LTE ∝ dt² for second-order methods, so to achieve target LTE:
        # dt_new = dt_old * sqrt(target_LTE / max_LTE)
        target_lte = 0.5  # Target max LTE (half of warning threshold)
        if max_lte > target_lte:
            suggested_dt = current_dt * np.sqrt(target_lte / max_lte)
            # Round to reasonable precision
            suggested_dt = max(0.001, round(suggested_dt, 4))  # Min 1 µs

            warning_text = (
                f'⚠ High LTE: {max_lte:.2f} mV (mean: {mean_lte:.3f} mV)\n'
                f'   Current dt: {current_dt:.3f} ms\n'
                f'   ➤ Suggested dt: {suggested_dt:.4f} ms\n'
                f'   (for target LTE < {target_lte} mV)'
            )
            warning_color = '#F38BA8'
        else:
            # LTE is acceptable - show status
            safety_margin = target_lte / max(max_lte, 0.01)
            warning_text = (
                f'✅ LTE OK: {max_lte:.3f} mV max\n'
                f'   Safety margin: {safety_margin:.1f}×\n'
                f'   dt can be ↑ by {np.sqrt(safety_margin):.1f}×'
            )
            warning_color = '#A6E3A1'

        # Remove old warning text if exists
        for txt in ax_lte.texts:
            if hasattr(txt, '_lte_warning'):
                txt.remove()

        # Add new warning/status text
        lte_text = ax_lte.text(
            0.02, 0.98, warning_text,
            transform=ax_lte.transAxes, fontsize=10, color=warning_color,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#1E1E2E', alpha=0.9, edgecolor=warning_color)
        )
        lte_text._lte_warning = True  # Mark for future removal

        # Plot dt vs time in second panel
        dt_arr = np.diff(t)
        dt_arr = np.append(dt_arr, dt_arr[-1])  # Make same length as t
        if hasattr(self, '_dt_lines') and self._dt_lines is not None:
            self._dt_lines.set_data(t, dt_arr)
        else:
            self._dt_lines = ax_dt.plot(t, dt_arr, color='#89B4FA', lw=1.5)[0]

        ax_dt.set_xlabel('Time (ms)')
        ax_dt.set_ylabel('dt (ms)')
        ax_dt.set_title('Integration Time Step')
        ax_dt.grid(alpha=0.3)

        _set_canvas_margins(self.fig_debug, left=0.08, right=0.94, top=0.95, bottom=0.08, hspace=0.34, wspace=0.20)
        self.cvs_debug.draw_idle()

    def open_fullscreen(self):
        """Open analytics clone in a maximized window preserving current tab/data."""
        idx = int(self.currentIndex())
        win = QMainWindow(self)
        win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        win.setWindowTitle("NeuroModelPort - Analytics (Full Screen)")
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
        'ITCa': ch.E_Ca,
        'NaP':  ch.ENa,  'NaR': ch.ENa,
        'IA':   ch.EK,   'IM':  ch.EK,
        'SK':   ch.EK,   'KATP': ch.EK, 'PumpNaK': 0.0,
    }
    return mapping.get(name, 0.0)
