"""
gui/main_window.py — Main Application Window v10.0

Tabs: Parameters | Oscilloscope | Analytics | Topology | Guide
Run modes: Standard | Monte-Carlo | Sweep | S-D Curve | Excit. Map | Stochastic
"""
import csv
import logging
import os
from pathlib import Path
import numpy as np

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPushButton, QLabel, QComboBox, QStatusBar,
    QScrollArea, QMessageBox, QApplication, QFileDialog,
    QGroupBox, QToolBar, QProgressDialog, QLayout, QSizePolicy,
    QFrame, QSplitter, QSlider, QTextEdit, QDockWidget, QMenuBar,
    QLineEdit,
)
from PySide6.QtCore import Qt, Signal, QObject, QTimer
from PySide6.QtGui import QIcon, QAction
import pyqtgraph as pg

from gui.locales import T
from gui.simulation_controller import SimulationController
from gui.config_manager import ConfigManager
from core.models import FullModelConfig
from core.solver import NeuronSolver
from core.errors import SimulationParameterError
from core.presets import get_preset_names, apply_synaptic_stimulus
from core.validation import validate_simulation_config, build_preset_mode_warnings
from gui.widgets.form_generator import PydanticFormWidget
from gui.widgets.unit_toggle_widget import UnitToggleWidget
from gui.widgets.stim_form_with_units import StimFormWithUnits
from gui.plots import OscilloscopeWidget
from gui.analytics import AnalyticsWidget
from gui.topology import TopologyWidget
from gui.axon_biophysics import AxonBiophysicsWidget
from gui.dual_stimulation_widget import DualStimulationWidget
from gui.text_sanitize import repair_text, repair_widget_tree
from core.dendritic_filter import get_ac_attenuation
from gui.ui_layout import LAYOUT_PRESETS, preset_for_width

# v13.0: Module-level logger
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
#  LIVE DECK PARAMETER REGISTRY
# ─────────────────────────────────────────────────────────────────────
#  (name, lo, hi, setter, getter)
_LIVE_PARAMS = {
    'Iext':      (-50.0, 250.0,
                  lambda cfg, v: setattr(cfg.stim,      'Iext',      v),
                  lambda cfg:    cfg.stim.Iext),
    'gNa_max':   (0.0,   360.0,
                  lambda cfg, v: setattr(cfg.channels,  'gNa_max',   v),
                  lambda cfg:    cfg.channels.gNa_max),
    'gK_max':    (0.0,   200.0,
                  lambda cfg, v: setattr(cfg.channels,  'gK_max',    v),
                  lambda cfg:    cfg.channels.gK_max),
    'T_celsius': (6.3,    42.0,
                  lambda cfg, v: setattr(cfg.env,       'T_celsius', v),
                  lambda cfg:    cfg.env.T_celsius),
    'Ra':        (10.0,  500.0,
                  lambda cfg, v: setattr(cfg.morphology,'Ra',        v),
                  lambda cfg:    cfg.morphology.Ra),
    'gL':        (0.0,     2.0,
                  lambda cfg, v: setattr(cfg.channels,  'gL',        v),
                  lambda cfg:    cfg.channels.gL),
    'ENa':       (0.0,   100.0,
                  lambda cfg, v: setattr(cfg.channels,  'ENa',       v),
                  lambda cfg:    cfg.channels.ENa),
    'EK':        (-120.0, -40.0,
                  lambda cfg, v: setattr(cfg.channels,  'EK',        v),
                  lambda cfg:    cfg.channels.EK),
    'EL':        (-100.0, -20.0,
                  lambda cfg, v: setattr(cfg.channels,  'EL',        v),
                  lambda cfg:    cfg.channels.EL),
    'tau_Ca':    (10.0, 1500.0,
                  lambda cfg, v: setattr(cfg.calcium,   'tau_Ca',    v),
                  lambda cfg:    cfg.calcium.tau_Ca),
    'B_Ca':      (1e-6,  1e-3,
                  lambda cfg, v: setattr(cfg.calcium,   'B_Ca',      v),
                  lambda cfg:    cfg.calcium.B_Ca),
    'pulse_dur': (0.1,   100.0,
                  lambda cfg, v: setattr(cfg.stim,      'pulse_dur', v),
                  lambda cfg:    cfg.stim.pulse_dur),
}
_LIVE_PARAM_NAMES = list(_LIVE_PARAMS.keys())
# Default suggestions for editable combos (dot-separated paths)
_LIVE_PARAM_SUGGESTIONS = [
    "stim.Iext",
    "channels.gNa_max",
    "channels.gK_max",
    "channels.ENa",
    "channels.EK",
    "channels.EL",
    "env.T_celsius",
    "morphology.Ra",
    "channels.gL",
    "calcium.tau_Ca",
    "calcium.B_Ca",
    "stim.pulse_dur",
]
_SWEEP_PARAM_SUGGESTIONS = [
    "stim.Iext",
    "channels.gNa_max",
    "channels.gK_max",
    "channels.gL",
    "channels.gTCa_max",
    "channels.gIh_max",
    "channels.gIM_max",
    "channels.gSK_max",
    "calcium.tau_Ca",
    "calcium.B_Ca",
    "metabolism.atp_synthesis_rate",
    "metabolism.g_katp_max",
    "env.T_celsius",
    "morphology.Ra",
]
_LIVE_SLIDER_STEPS = 1000   # integer slider resolution


# ─────────────────────────────────────────────────────────────────────
#  MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):

    _STYLE = """
        QMainWindow, QWidget { background: #1E1E2E; color: #CDD6F4; }
        QGroupBox {
            border: 1px solid #45475A;
            border-radius: 6px;
            margin-top: 8px;
            font-weight: bold; color: #89B4FA;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 8px; top: 2px; }
        QTabWidget::pane  { border: 1px solid #45475A; border-radius: 4px; }
        QTabBar::tab {
            background: #313244; color: #CDD6F4;
            padding: 6px 14px; border-radius: 4px 4px 0 0;
            margin-right: 2px;
        }
        QTabBar::tab:selected { background: #89B4FA; color: #1E1E2E; font-weight: bold; }
        QTabBar::tab:hover    { background: #585B70; }
        QComboBox, QSpinBox, QDoubleSpinBox {
            background: #313244; color: #CDD6F4;
            border: 1px solid #585B70; border-radius: 4px; padding: 2px 6px;
        }
        QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover { border-color: #89B4FA; }
        QPushButton:focus, QComboBox:focus, QLineEdit:focus, QTextEdit:focus {
            border: 2px solid #F9E2AF;
        }
        QCheckBox { color: #CDD6F4; spacing: 6px; }
        QCheckBox::indicator { width: 16px; height: 16px; }
        QScrollBar:vertical {
            background: #313244; width: 10px; border-radius: 5px;
        }
        QScrollBar::handle:vertical { background: #585B70; border-radius: 5px; }
        QLabel { color: #BAC2DE; }
        QStatusBar { background: #181825; color: #A6ADC8; }
        QTextEdit { background: #0D1117; color: #C9D1D9; }
    """

    def __init__(self):
        super().__init__()
        self.resize(1100, 700)
        self.setStyleSheet(self._STYLE)
        
        # Initialize services
        self.config_manager = ConfigManager()
        self.config = self.config_manager.config  # Backward-compatible read path for older GUI helpers/tests.
        self.sim_controller = SimulationController()
        
        self._dual_stim_signal_connected = False
        self._delay_target_name = "Terminal"
        self._delay_custom_index = 1
        self._sidebar_visible = True
        # Live deck state
        self._live_combos: list[QComboBox] = []
        self._live_sliders: list[QSlider] = []
        self._live_labels: list[QLabel] = []
        self._live_custom_bounds: dict[int, tuple[float, float]] = {}  # Cache bounds for custom parameters
        self._is_live_run = False  # Flag to prevent tab switch on Live timer simulations
        self._live_timer = QTimer(self)
        self._live_timer.setSingleShot(True)
        self._live_timer.setInterval(350)
        self._live_timer.timeout.connect(self._on_live_timer_fired)

        # v13.0: Debounced analytics timer for async rendering
        self._analytics_debounce_timer = QTimer(self)
        self._analytics_debounce_timer.setSingleShot(True)
        self._analytics_debounce_timer.timeout.connect(self._on_analytics_debounce_fired)
        self._pending_result_for_analytics = None
        self._pending_morph_for_analytics = None
        # Generation counter to prevent race conditions with stale analytics
        self._analytics_generation = 0
        self._pending_analytics_generation = 0
        self._progress_dialog = None

        # v12.7: Create oscilloscope first (will be set as central widget)
        self.oscilloscope = OscilloscopeWidget()
        self.oscilloscope.delay_target_changed.connect(self._on_delay_target_changed)
        self._delay_target_name, self._delay_custom_index = (
            self.oscilloscope.get_delay_target_selection()
        )
        self.oscilloscope.sync_delay_controls_for_config(self.config_manager.config)
        # v12.7: Re-sync local values — oscilloscope clamps them to valid range
        self._delay_target_name, self._delay_custom_index = (
            self.oscilloscope.get_delay_target_selection()
        )
        
        # v12.7: Set oscilloscope as TRUE central widget for proper scaling
        self.setCentralWidget(self.oscilloscope)
        self.setMinimumSize(980, 680)

        # v12.7: Setup toolbars and docks
        self._setup_toolbars()
        self._setup_cockpit_docks()
        self._setup_view_menu()
        self._setup_status_bar()
        self._wire_service_signals()
        self.retranslate_ui()
        
        self._restore_session_or_default()
        self._restore_or_reset_dock_layout()

    def _wire_service_signals(self):
        """Connect ConfigManager and SimulationController signals to MainWindow slots."""
        # ConfigManager signals
        self.config_manager.config_changed.connect(self._on_config_changed)
        
        # SimulationController signals
        self.sim_controller.simulation_started.connect(self._on_simulation_started)
        self.sim_controller.simulation_finished.connect(self._on_simulation_done)
        self.sim_controller.progress_updated.connect(self._on_progress_updated)
        self.sim_controller.error_occurred.connect(self._on_sim_error)
    
    def _on_config_changed(self):
        """Handle config change signal from ConfigManager."""
        self._refresh_all_forms()
    
    def _on_simulation_started(self):
        """Handle simulation started signal from SimulationController."""
        self._lock_ui(True)
        self._status(T.tr('status_computing'))
        QApplication.processEvents()
    
    def _on_progress_updated(self, current: int, total: int, value: float):
        """Handle progress update signal from SimulationController."""
        pct = int(100 * current / max(1, total))
        self._status(f"Progress: {pct}% ({current}/{total}) - Value: {value:.3g}")
        if total > 0:
            max_total = max(1, total)
            if self._progress_dialog is None:
                self._progress_dialog = QProgressDialog("Simulation in progress...", "Cancel", 0, max_total, self)
                self._progress_dialog.setWindowTitle("NeuroModelPort")
                self._progress_dialog.setAutoClose(True)
                self._progress_dialog.canceled.connect(self._request_cancel)
            self._progress_dialog.setMaximum(max_total)
            self._progress_dialog.setValue(min(current, max_total))
        QApplication.processEvents()


    # ─────────────────────────────────────────────────────────────────
    #  TOOLBARS (v12.7 Cockpit Paradigm)
    # ─────────────────────────────────────────────────────────────────
    def _setup_toolbars(self):
        """v12.7: Native QToolBars for clean cockpit layout."""
        # Main Toolbar (Run, Stoch, Sweep, etc.)
        self.main_toolbar = QToolBar("Main", self)
        self.main_toolbar.setObjectName("main_toolbar")  # v12.8 FIX: Required for saveState
        self.main_toolbar.setMovable(False)
        self.main_toolbar.setFloatable(False)
        self.addToolBar(Qt.TopToolBarArea, self.main_toolbar)

        # Run button
        self.btn_run = QPushButton("RUN")
        self.btn_run.setToolTip("Run simulation")
        self.btn_run.clicked.connect(self._on_run_button_clicked)
        self.main_toolbar.addWidget(self.btn_run)

        # Cancel button
        self.btn_cancel = QPushButton("CANCEL")
        self.btn_cancel.setToolTip("Cancel running computation")
        self.btn_cancel.clicked.connect(self._request_cancel)
        self.btn_cancel.setVisible(False)
        self._cancel_requested = False
        self.main_toolbar.addWidget(self.btn_cancel)

        self.main_toolbar.addSeparator()

        # Hines toggle
        self.btn_hines = QPushButton("HINES")
        self.btn_hines.setCheckable(True)
        self.btn_hines.setChecked(False)
        self.btn_hines.setToolTip("Toggle native Hines solver")
        self.btn_hines.clicked.connect(self._on_hines_toggled)
        self.main_toolbar.addWidget(self.btn_hines)

        # Jacobian selector
        self.combo_jacobian = QComboBox()
        self.combo_jacobian.addItems(["dense_fd", "sparse_fd", "analytic_sparse"])
        self.combo_jacobian.setToolTip("Jacobian mode for SciPy BDF")
        self.combo_jacobian.setMaximumWidth(100)
        self.combo_jacobian.currentTextChanged.connect(self._on_jacobian_changed)
        self.main_toolbar.addWidget(self.combo_jacobian)

        self.main_toolbar.addSeparator()

        # Analysis buttons
        self.btn_stoch = QPushButton("STOCH")
        self.btn_stoch.setToolTip("Stochastic simulation")
        self.btn_stoch.clicked.connect(lambda: self.sim_controller.run_stochastic(
            self.config_manager.config, 1, on_success=self._on_stoch_done, on_error=self._on_sim_error))
        self.main_toolbar.addWidget(self.btn_stoch)

        self.btn_sweep = QPushButton("SWEEP")
        self.btn_sweep.setToolTip("Parameter sweep")
        self.btn_sweep.clicked.connect(self.run_sweep)
        self.main_toolbar.addWidget(self.btn_sweep)

        self.btn_fi = QPushButton("f-I")
        self.btn_fi.setToolTip("Frequency-Current curve")
        self.btn_fi.clicked.connect(self.run_fi_curve)
        self.main_toolbar.addWidget(self.btn_fi)

        self.btn_sd = QPushButton("S-D")
        self.btn_sd.setToolTip("Strength-Duration curve")
        self.btn_sd.clicked.connect(self.run_sd_curve)
        self.main_toolbar.addWidget(self.btn_sd)

        self.btn_excmap = QPushButton("EXCMAP")
        self.btn_excmap.setToolTip("Excitability map")
        self.btn_excmap.clicked.connect(self.run_excmap)
        self.main_toolbar.addWidget(self.btn_excmap)

        self.main_toolbar.addSeparator()

        # Export buttons
        self.btn_export_plot = QPushButton("Plot")
        self.btn_export_plot.setToolTip("Export plot")
        self.btn_export_plot.setEnabled(False)
        self.btn_export_plot.clicked.connect(self.export_plot)
        self.main_toolbar.addWidget(self.btn_export_plot)

        self.btn_export = QPushButton("CSV")
        self.btn_export.setToolTip("Export CSV")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_csv)
        self.main_toolbar.addWidget(self.btn_export)

        self.btn_export_nml = QPushButton("NeuroML")
        self.btn_export_nml.setToolTip("Export to NeuroML")
        self.btn_export_nml.clicked.connect(self.export_neuroml)
        self.main_toolbar.addWidget(self.btn_export_nml)

        self.main_toolbar.addSeparator()

        # Config buttons
        self.btn_save_config = QPushButton("Save")
        self.btn_save_config.setToolTip("Save config")
        self.btn_save_config.clicked.connect(self.save_config_as)
        self.main_toolbar.addWidget(self.btn_save_config)

        self.btn_load_config = QPushButton("Open")
        self.btn_load_config.setToolTip("Load config")
        self.btn_load_config.clicked.connect(self.load_config_from)
        self.main_toolbar.addWidget(self.btn_load_config)

        self.btn_more_actions = QPushButton("Actions")
        self.btn_more_actions.setToolTip("More actions")
        self._build_more_actions_menu()
        self.main_toolbar.addWidget(self.btn_more_actions)

        # Settings Toolbar (Preset, Lang, Mode)
        self.addToolBarBreak(Qt.TopToolBarArea)
        self.settings_toolbar = QToolBar("Settings", self)
        self.settings_toolbar.setObjectName("settings_toolbar")  # v12.8 FIX: Required for saveState
        self.settings_toolbar.setMovable(False)
        self.settings_toolbar.setFloatable(False)
        self.addToolBar(Qt.TopToolBarArea, self.settings_toolbar)

        # Preset selector
        self.lbl_preset = QLabel("Preset:")
        self.settings_toolbar.addWidget(self.lbl_preset)
        self.combo_presets = QComboBox()
        self.combo_presets.setMinimumWidth(160)
        self.combo_presets.addItems(["— Select preset —"] + get_preset_names())
        self.combo_presets.currentTextChanged.connect(self.load_preset)
        self.settings_toolbar.addWidget(self.combo_presets)

        self.settings_toolbar.addSeparator()

        # Language selector
        self.lbl_lang = QLabel("Lang:")
        self.settings_toolbar.addWidget(self.lbl_lang)
        self.combo_lang = QComboBox()
        self.combo_lang.addItems(["EN", "RU"])
        self.combo_lang.setCurrentText("EN")
        self.combo_lang.currentTextChanged.connect(self.change_language)
        self.settings_toolbar.addWidget(self.combo_lang)

        self.settings_toolbar.addSeparator()

        # Mode selector
        self.settings_toolbar.addWidget(QLabel("Mode:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["🔬 Microscope", "🌉 Bridge", "🧪 Research"])
        self.combo_mode.setCurrentText("🌉 Bridge")
        self.combo_mode.currentTextChanged.connect(self._toggle_ui_complexity)
        self.settings_toolbar.addWidget(self.combo_mode)

        self.settings_toolbar.addSeparator()

        # Sidebar toggle
        self.btn_toggle_sidebar = QPushButton("Sidebar")
        self.btn_toggle_sidebar.setToolTip("Toggle parameters panel")
        self.btn_toggle_sidebar.setCheckable(True)
        self.btn_toggle_sidebar.setChecked(True)
        self.btn_toggle_sidebar.clicked.connect(self._toggle_sidebar)
        self.settings_toolbar.addWidget(self.btn_toggle_sidebar)

        # Window size preset button
        self.btn_window_size = QPushButton("Size")
        self.btn_window_size.setToolTip("Window size preset (for laptops)")
        self.btn_window_size.clicked.connect(self._show_window_size_menu)
        self.settings_toolbar.addWidget(self.btn_window_size)

        # Sparkline (compact Vm preview)
        self._sparkline = pg.PlotWidget()
        self._sparkline.setFixedSize(120, 28)
        self._sparkline.hideAxis('left')
        self._sparkline.hideAxis('bottom')
        self._sparkline.setBackground('#0D1117')
        self._sparkline.setToolTip("Latest Vm trace")
        self._sparkline_curve = self._sparkline.plot([], [], pen=pg.mkPen('#89B4FA', width=1))
        self.settings_toolbar.addWidget(self._sparkline)

    # ─────────────────────────────────────────────────────────────────
    #  DOCK WIDGETS (v12.7 Cockpit Docks)
    # ─────────────────────────────────────────────────────────────────
    def _setup_cockpit_docks(self):
        """
        v12.7: Dock-based professional interface.
        
        Oscilloscope is now the TRUE central widget (set in __init__).
        Docks surround the central visualization:
        - Left: Parameters (sidebar with forms)
        - Right: Live Controls (sliders), Topology (heatmap)
        - Bottom: Analytics (matplotlib tabs)
        - Floating: Stimulation Studio, Axon Biophysics, Guide
        """
        # Note: Oscilloscope is already created and set as central widget in __init__

        # ── Dock 1: Parameters (Left) ───────────────────────────────
        self._dock_params = QDockWidget("Parameters", self)
        self._dock_params.setObjectName("dock_params")
        self._dock_params.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | 
            Qt.DockWidgetArea.RightDockWidgetArea
        )
        
        self._sidebar_frame = QFrame()
        self._sidebar_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self._sidebar_frame.setMinimumWidth(320)
        self._sidebar_frame.setMaximumWidth(900)
        self._build_sidebar_panel()
        self._dock_params.setWidget(self._sidebar_frame)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._dock_params)
        self.tab_params = self._sidebar_frame  # backward compat

        # ── Dock 2: Live Control Deck (Right) ───────────────────────
        self._dock_live = QDockWidget("🎛️ Live Controls", self)
        self._dock_live.setObjectName("dock_live")
        self._dock_live.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | 
            Qt.DockWidgetArea.RightDockWidgetArea
        )
        
        self._live_deck_frame = QFrame()
        self._live_deck_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self._build_live_deck_panel()
        self._dock_live.setWidget(self._live_deck_frame)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._dock_live)
        self.tabifyDockWidget(self._dock_params, self._dock_live)

        # ── Dock 3: Analytics (Bottom) ─────────────────────────────
        self._dock_analytics = QDockWidget("Analytics", self)
        self._dock_analytics.setObjectName("dock_analytics")
        self._dock_analytics.setAllowedAreas(
            Qt.DockWidgetArea.TopDockWidgetArea |
            Qt.DockWidgetArea.BottomDockWidgetArea |
            Qt.DockWidgetArea.LeftDockWidgetArea |
            Qt.DockWidgetArea.RightDockWidgetArea
        )
        
        analytics_container = QWidget()
        analytics_layout = QVBoxLayout(analytics_container)
        analytics_layout.setContentsMargins(0, 0, 0, 0)
        analytics_layout.setSpacing(0)
        
        self.analytics = AnalyticsWidget()
        analytics_layout.addWidget(self.analytics)
        
        # Session notes (collapsible)
        notes_group = QWidget()
        notes_layout = QVBoxLayout(notes_group)
        notes_layout.setContentsMargins(8, 0, 8, 8)
        notes_layout.setSpacing(4)

        self.btn_toggle_notes = QPushButton("Show Researcher Notes")
        self.btn_toggle_notes.setCheckable(True)
        self.btn_toggle_notes.setChecked(False)
        self.btn_toggle_notes.setStyleSheet("""
            QPushButton { background: transparent; color: #CBA6F7; text-align: left; font-weight: bold; border: none; }
            QPushButton:hover { color: #EBA0AC; }
        """)

        self.session_notes = QTextEdit()
        self.session_notes.setMaximumHeight(100)
        self.session_notes.setPlaceholderText("Enter notes about this simulation session...")
        self.session_notes.setStyleSheet("background: #1E1E2E; color: #CDD6F4; border: 1px solid #45475A;")
        self.session_notes.setVisible(False)
        self.session_notes.textChanged.connect(self._on_notes_changed)
        self.session_notes.setPlainText(self.config_manager.config.notes)

        self.btn_toggle_notes.clicked.connect(lambda checked: self.session_notes.setVisible(checked))
        self.btn_toggle_notes.clicked.connect(
            lambda checked: self.btn_toggle_notes.setText(
                "Hide Researcher Notes" if checked else "Show Researcher Notes"
            )
        )

        notes_layout.addWidget(self.btn_toggle_notes)
        notes_layout.addWidget(self.session_notes)
        analytics_layout.addWidget(notes_group)
        
        self._dock_analytics.setWidget(analytics_container)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._dock_analytics)

        # Connect oscilloscope to analytics
        self.oscilloscope.time_highlighted.connect(self.analytics.highlight_time)
        
        # v12.2: Connect LLE compute request to full simulation rerun
        self.analytics.computeLleRequested.connect(self._run_with_lle_enabled)

        # ── Dock 4: Topology (Right, tabbed with Live Controls) ───────
        self._dock_topology = QDockWidget("Topology", self)
        self._dock_topology.setObjectName("dock_topology")
        self._dock_topology.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | 
            Qt.DockWidgetArea.RightDockWidgetArea |
            Qt.DockWidgetArea.TopDockWidgetArea |
            Qt.DockWidgetArea.BottomDockWidgetArea
        )
        
        self.topology = TopologyWidget()
        self.topology.set_delay_focus(
            self._delay_target_name,
            self._delay_custom_index,
        )
        self._dock_topology.setWidget(self.topology)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._dock_topology)
        
        # Tabify Topology with Live Controls
        self.tabifyDockWidget(self._dock_live, self._dock_topology)

        # Bi-directional time sync
        self.oscilloscope.time_highlighted.connect(self.topology.highlight_time)
        self.topology.time_scrubbed.connect(self.oscilloscope.set_time_marker)
        self.topology.compartment_selected.connect(self.oscilloscope.set_delay_compartment)

        # ── Dock 5: Stimulation Studio (Floating/Left) ───────────────
        self._dock_stim = QDockWidget("Stimulation Studio", self)
        self._dock_stim.setObjectName("dock_stim")
        self._dock_stim.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        
        stim_container = QWidget()
        stim_layout = QVBoxLayout(stim_container)
        stim_layout.setContentsMargins(0, 0, 0, 0)
        stim_layout.setSpacing(0)
        
        stim_scroll = QScrollArea()
        stim_scroll.setWidgetResizable(True)
        stim_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        stim_scroll.setStyleSheet("QScrollArea { background: #1E1E2E; }")
        
        stim_content = self._build_stimulation_studio()
        stim_scroll.setWidget(stim_content)
        stim_layout.addWidget(stim_scroll)
        
        self._dock_stim.setWidget(stim_container)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._dock_stim)
        self._dock_stim.setFloating(True)
        self._dock_stim.move(self.x() + 100, self.y() + 100)
        self._dock_stim.resize(900, 700)

        # ── Dock 6: Axon Biophysics (Tabbed with Stimulation) ───────
        self._dock_axon = QDockWidget("Axon Biophysics", self)
        self._dock_axon.setObjectName("dock_axon")
        self._dock_axon.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        
        self.axon_biophysics = AxonBiophysicsWidget()
        self._dock_axon.setWidget(self.axon_biophysics)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._dock_axon)
        self.tabifyDockWidget(self._dock_stim, self._dock_axon)

        # ── Dock 7: Guide (Tabbed) ───────────────────────────────────
        self._dock_guide = QDockWidget("Guide", self)
        self._dock_guide.setObjectName("dock_guide")
        self._dock_guide.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        
        from PySide6.QtWidgets import QTextBrowser
        self.guide_browser = QTextBrowser()
        self.guide_browser.setOpenExternalLinks(True)
        self.guide_browser.setStyleSheet(
            "background:#0D1117; color:#C9D1D9; font-size:13px; border:none;"
        )
        self._dock_guide.setWidget(self.guide_browser)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._dock_guide)
        self.tabifyDockWidget(self._dock_live, self._dock_guide)

        # Raise Live Controls and Oscilloscope to front
        self._dock_live.raise_()
        self.oscilloscope.raise_()

        # Pass form widgets to ConfigManager
        self.config_manager.set_dual_stim_widget(self.dual_stim_widget)
        self.config_manager.set_form_widgets(
            self.form_stim, self.form_stim_loc, self.form_preset_modes
        )
        self._apply_layout_preset("Laptop", resize_window=False)

    def _build_more_actions_menu(self):
        from PySide6.QtWidgets import QMenu

        menu = QMenu(self)
        actions = [
            ("Stochastic", lambda: self.btn_stoch.click()),
            ("Sweep", lambda: self.btn_sweep.click()),
            ("f-I Curve", lambda: self.btn_fi.click()),
            ("S-D Curve", lambda: self.btn_sd.click()),
            ("Excitability Map", lambda: self.btn_excmap.click()),
            ("Export NeuroML", lambda: self.btn_export_nml.click()),
            ("Analytics Workspace", self._focus_analytics_workspace),
            ("Open Stimulation Studio", lambda: self._dock_stim.setVisible(True)),
        ]
        for label, callback in actions:
            action = menu.addAction(label)
            action.triggered.connect(callback)
        self.btn_more_actions.setMenu(menu)

    def _set_secondary_actions_visible(self, visible: bool) -> None:
        for widget in (
            self.btn_stoch,
            self.btn_sweep,
            self.btn_fi,
            self.btn_sd,
            self.btn_excmap,
            self.btn_export_nml,
        ):
            widget.setVisible(visible)
        self.btn_more_actions.setVisible(not visible)

    def _sync_preset_combo_text(self, text: str) -> None:
        for combo in (self.combo_presets, getattr(self, "_sidebar_preset_combo", None)):
            if combo is None:
                continue
            if combo.findText(text) < 0:
                combo.addItem(text)
            combo.blockSignals(True)
            combo.setCurrentText(text)
            combo.blockSignals(False)

    def _restore_session_or_default(self) -> None:
        if os.path.exists(".last_session.json") and self.config_manager.load_config_from(".last_session.json"):
            self.config = self.config_manager.config
            self._sync_preset_combo_text(self.config_manager.current_preset_name or "Custom Config")
            self._refresh_all_forms()
            self._sync_hines_button_state()
            self.oscilloscope.sync_delay_controls_for_config(self.config_manager.config)
            # Re-sync local values after oscilloscope clamps them
            self._delay_target_name, self._delay_custom_index = (
                self.oscilloscope.get_delay_target_selection()
            )
            self.topology.draw_neuron(self.config_manager.config)
            self._status("Restored previous session as custom configuration.")
            return
        self.load_preset("A: Squid Giant Axon (HH 1952)")

    def _restore_or_reset_dock_layout(self) -> None:
        restored = False
        if self._can_restore_ui_state():
            try:
                with open(".dock_geometry.bin", "rb") as f:
                    self.restoreGeometry(f.read())
                with open(".dock_state.bin", "rb") as f:
                    restored = bool(self.restoreState(f.read()))
            except Exception as exc:
                logger.warning("Failed to restore dock state: %s", exc)
                restored = False
        if not restored:
            self._reset_dock_layout()
        self._ensure_usable_layout()

    def _can_restore_ui_state(self) -> bool:
        if self._is_headless_qt_platform():
            return False
        required = (".dock_geometry.bin", ".dock_state.bin", ".ui_state_meta.json")
        if not all(os.path.exists(path) for path in required):
            return False
        try:
            import json
            with open(".ui_state_meta.json", "r", encoding="utf-8") as f:
                meta = json.load(f)
            return (
                meta.get("version") == 1
                and meta.get("layout_engine") == "dock-shell-v1"
                and meta.get("qt_platform") not in {"offscreen", "minimal"}
            )
        except Exception as exc:
            logger.warning("Ignoring invalid UI state metadata: %s", exc)
            return False

    def _is_headless_qt_platform(self) -> bool:
        platform = ""
        app = QApplication.instance()
        if app is not None:
            try:
                platform = QApplication.platformName()
            except Exception:
                platform = ""
        platform = platform or os.environ.get("QT_QPA_PLATFORM", "")
        return platform.lower() in {"offscreen", "minimal"}

    def _ensure_usable_layout(self) -> None:
        """v12.7: Ensure oscilloscope is central widget and docks are visible."""
        self._apply_layout_preset(preset_for_width(max(1, self.width())).name, resize_window=False)
        if hasattr(self, "_dock_params"):
            self._dock_params.setVisible(True)
            self._dock_params.raise_()
        if hasattr(self, "_dock_analytics"):
            self._dock_analytics.setVisible(True)
        # v12.7: Oscilloscope is always the central widget in cockpit paradigm
        if self.centralWidget() is not self.oscilloscope:
            self.setCentralWidget(self.oscilloscope)
        self.oscilloscope.raise_()

    def _build_sidebar_panel(self):
        """Build the collapsible left parameter panel (replaces old '1) Setup' tab)."""
        layout = QVBoxLayout(self._sidebar_frame)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # ── Quick Preset shortcut row (Step 5) ────────────────────────
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("HINES Preset:"))
        self._sidebar_preset_combo = QComboBox()
        self._sidebar_preset_combo.addItems(["— Select preset —"] + get_preset_names())
        self._sidebar_preset_combo.currentTextChanged.connect(self.load_preset)
        preset_row.addWidget(self._sidebar_preset_combo, 1)
        layout.addLayout(preset_row)

        # ── Quick-Set dropdown for parameter groups ────────────────────
        quickset_row = QHBoxLayout()
        quickset_row.addWidget(QLabel("📋 Quick-Set:"))
        self._quickset_combo = QComboBox()
        self._quickset_combo.addItems(["— Quick-Set —", "Channel Densities", "Dendritic Cable", "Metabolism"])
        self._quickset_combo.currentTextChanged.connect(self._on_quick_set_changed)
        quickset_row.addWidget(self._quickset_combo, 1)
        layout.addLayout(quickset_row)

        ux_row = QHBoxLayout()
        ux_row.addWidget(QLabel("Filter:"))
        self._form_search = QLineEdit()
        self._form_search.setPlaceholderText("Search parameters")
        self._form_search.textChanged.connect(self._apply_form_ux_filters)
        ux_row.addWidget(self._form_search, 1)
        self._form_priority = QComboBox()
        self._form_priority.addItems(["all", "critical", "basic", "advanced"])
        self._form_priority.setCurrentText("basic")
        self._form_priority.currentTextChanged.connect(self._apply_form_ux_filters)
        ux_row.addWidget(self._form_priority)
        layout.addLayout(ux_row)

        # ── Params hint ───────────────────────────────────────────────
        self.lbl_params_hint = QLabel("")
        self.lbl_params_hint.setWordWrap(True)
        self.lbl_params_hint.setStyleSheet(
            "QLabel {"
            "background:#1f2538; color:#cdd6f4; border:1px solid #45475a; "
            "border-radius:6px; padding:8px; font-size:12px;"
            "}"
        )
        layout.addWidget(self.lbl_params_hint)

        # ── Scrollable forms ──────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setSizeAdjustPolicy(QScrollArea.SizeAdjustPolicy.AdjustIgnored)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        content = QWidget()
        content.setMinimumSize(0, 0)
        c_layout = QVBoxLayout(content)
        c_layout.setSpacing(8)

        # Forms
        self.form_stim = StimFormWithUnits(
            self.config_manager.config.stim,
            self.config_manager.config.morphology.d_soma,
            on_change=self._on_stim_field_changed,
        )
        self.form_stim_loc = PydanticFormWidget(
            self.config_manager.config.stim_location,
            "Stimulus Location",
            on_change=self._on_stim_loc_field_changed,
        )
        self.form_dfilter = PydanticFormWidget(
            self.config_manager.config.dendritic_filter,
            "Dendritic Filter",
            on_change=self._on_dfilter_field_changed,
        )
        self.form_preset_modes = PydanticFormWidget(
            self.config_manager.config.preset_modes,
            "Preset Modes (K/N/O)",
            on_change=self._on_preset_mode_changed
        )
        self.form_chan = PydanticFormWidget(
            self.config_manager.config.channels,
            "Ion Channels",
            on_change=self._on_channel_field_changed,
        )
        self.form_calcium = PydanticFormWidget(self.config_manager.config.calcium, "Calcium Dynamics")

        # Metabolism form with scroll area (many fields need scrollable space)
        self.form_metabolism = PydanticFormWidget(self.config_manager.config.metabolism, "ATP / Metabolism")
        self._scroll_metabolism = QScrollArea()
        self._scroll_metabolism.setWidgetResizable(True)
        self._scroll_metabolism.setMaximumHeight(250)
        self._scroll_metabolism.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll_metabolism.setWidget(self.form_metabolism)
        self.form_morph = PydanticFormWidget(
            self.config_manager.config.morphology,
            "Morphology",
            on_change=self._on_morph_field_changed,
        )
        self.form_env = PydanticFormWidget(self.config_manager.config.env, "Environment")
        self.form_ana = PydanticFormWidget(
            self.config_manager.config.analysis,
            "Analysis / Sweep / Map",
        )

        # Group 0: Preset Modes (variations)
        grp_modes = QGroupBox("Preset Modes")
        modes_layout = QVBoxLayout(grp_modes)
        modes_layout.setSpacing(4)
        modes_layout.addWidget(self.form_preset_modes)
        c_layout.addWidget(grp_modes)

        # Group 1: Morphology (geometry)
        self.grp_morph = QGroupBox("Morphology")
        morph_layout = QVBoxLayout(self.grp_morph)
        morph_layout.setSpacing(4)
        morph_layout.addWidget(self.form_morph)
        c_layout.addWidget(self.grp_morph)

        # Group 2: Biophysics Context (Environment + Calcium + ATP)
        grp_context = QGroupBox("Biophysics Context")
        context_layout = QHBoxLayout(grp_context)
        context_layout.setSpacing(8)
        context_left = QVBoxLayout()
        context_right = QVBoxLayout()
        context_left.setSpacing(4)
        context_right.setSpacing(4)
        context_left.addWidget(self.form_env)
        context_right.addWidget(self.form_calcium)
        context_right.addWidget(self._scroll_metabolism)
        context_layout.addLayout(context_left)
        context_layout.addLayout(context_right)
        c_layout.addWidget(grp_context)

        # Group 3: Ion Channels (conductances)
        grp_chan = QGroupBox("Ion Channels")
        chan_layout = QVBoxLayout(grp_chan)
        chan_layout.setSpacing(4)
        chan_layout.addWidget(self.form_chan)
        c_layout.addWidget(grp_chan)

        # Group 4: Analysis Settings
        self.grp_ana = QGroupBox("Analysis Settings")
        ana_layout = QVBoxLayout(self.grp_ana)
        ana_layout.setSpacing(4)
        sweep_selector = QWidget()
        sweep_selector_layout = QVBoxLayout(sweep_selector)
        sweep_selector_layout.setContentsMargins(0, 0, 0, 0)
        sweep_selector_layout.setSpacing(4)
        sweep_row = QHBoxLayout()
        sweep_row.setContentsMargins(0, 0, 0, 0)
        sweep_row.setSpacing(6)
        sweep_row.addWidget(QLabel("Sweep parameter:"))
        self._sweep_param_combo = QComboBox()
        self._sweep_param_combo.setEditable(True)
        self._sweep_param_combo.addItems(_SWEEP_PARAM_SUGGESTIONS)
        self._sweep_param_combo.setCurrentText(self._canonical_sweep_param(self.config_manager.config.analysis.sweep_param))
        self._sweep_param_combo.currentTextChanged.connect(self._on_sweep_param_changed)
        sweep_row.addWidget(self._sweep_param_combo, 1)
        btn_iext = QPushButton("Use Iext")
        btn_iext.setMaximumWidth(80)
        btn_iext.clicked.connect(lambda: self._sweep_param_combo.setCurrentText("stim.Iext"))
        sweep_row.addWidget(btn_iext)
        sweep_selector_layout.addLayout(sweep_row)
        self._sweep_param_help = QLabel("")
        self._sweep_param_help.setWordWrap(True)
        self._sweep_param_help.setStyleSheet("color:#BAC2DE; font-size:11px;")
        sweep_selector_layout.addWidget(self._sweep_param_help)
        ana_layout.addWidget(sweep_selector)
        ana_layout.addWidget(self.form_ana)

        # Auto-Rheobase Search button
        btn_find_threshold = QPushButton("🔍 Find Threshold (Auto-Rheobase)")
        btn_find_threshold.setStyleSheet("""
            QPushButton {
                background: #313244;
                color: #89B4FA;
                border: 1px solid #45475A;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #45475A;
                color: #A6E3A1;
            }
        """)
        btn_find_threshold.clicked.connect(self._on_find_threshold_clicked)
        ana_layout.addWidget(btn_find_threshold)

        self._lbl_rheobase_result = QLabel("")
        self._lbl_rheobase_result.setWordWrap(True)
        self._lbl_rheobase_result.setStyleSheet("color:#A6E3A1; font-size:11px;")
        ana_layout.addWidget(self._lbl_rheobase_result)

        c_layout.addWidget(self.grp_ana)

        scroll.setWidget(content)
        layout.addWidget(scroll)

    def _build_live_deck_panel(self):
        """Build the Live Control Deck panel (now separate dock)."""
        # Safety check: _live_deck_frame must exist (created in _setup_cockpit)
        if not hasattr(self, '_live_deck_frame') or self._live_deck_frame is None:
            raise RuntimeError("_build_live_deck_panel called before _live_deck_frame initialization")
        layout = QVBoxLayout(self._live_deck_frame)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        header = QLabel("🎛️ Live Parameter Controls")
        header.setStyleSheet("font-weight: bold; color: #89B4FA; font-size: 13px;")
        layout.addWidget(header)

        self._live_combos.clear()
        self._live_sliders.clear()
        self._live_labels.clear()

        for row_i, default_param in enumerate(('stim.Iext', 'channels.gNa_max', 'env.T_celsius')):
            row_w = QWidget()
            row_h = QHBoxLayout(row_w)
            row_h.setContentsMargins(0, 0, 0, 0)
            row_h.setSpacing(4)

            combo = QComboBox()
            combo.setEditable(True)
            combo.addItems(_LIVE_PARAM_SUGGESTIONS)
            combo.setCurrentText(default_param)
            combo.setFixedWidth(120)
            combo.currentTextChanged.connect(
                lambda _text, ri=row_i: self._on_live_combo_changed(ri)
            )
            self._live_combos.append(combo)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, _LIVE_SLIDER_STEPS)
            slider.setValue(self._val_to_live_slider(default_param, row_i))
            slider.valueChanged.connect(
                lambda raw, ri=row_i: self._on_live_slider_moved(ri, raw)
            )
            self._live_sliders.append(slider)

            obj, attr = self._resolve_param(default_param)
            if obj is not None and attr is not None:
                cur_val = getattr(obj, attr)
            elif default_param in _LIVE_PARAMS:
                lo, hi, _, getter = _LIVE_PARAMS[default_param]
                cur_val = getter(self.config_manager.config)
            else:
                cur_val = self.config_manager.config.stim.Iext
            lbl = QLabel(f"{cur_val:.2f}")
            lbl.setFixedWidth(52)
            lbl.setStyleSheet("color:#CBA6F7; font-size:11px;")
            self._live_labels.append(lbl)

            row_h.addWidget(combo)
            row_h.addWidget(slider, 1)
            row_h.addWidget(lbl)
            layout.addWidget(row_w)

        layout.addStretch()

    def _build_stimulation_studio(self) -> QWidget:
        """Build Stimulation Studio content widget."""
        stim_studio_widget = QWidget()
        stim_main_layout = QVBoxLayout(stim_studio_widget)
        stim_main_layout.setSpacing(6)
        stim_main_layout.setContentsMargins(4, 4, 4, 4)

        stim_controls_widget = QWidget()
        stim_layout = QHBoxLayout(stim_controls_widget)
        stim_layout.setSpacing(10)
        stim_layout.setContentsMargins(0, 0, 0, 0)

        left_col = QWidget()
        left_layout = QVBoxLayout(left_col)
        left_layout.setSpacing(6)
        left_layout.setContentsMargins(0, 0, 0, 0)

        if not hasattr(self, 'form_stim'):
            self.form_stim = StimFormWithUnits(
                self.config_manager.config.stim,
                self.config_manager.config.morphology.d_soma,
                on_change=self._on_stim_field_changed,
            )
        if not hasattr(self, 'form_stim_loc'):
            self.form_stim_loc = PydanticFormWidget(
                self.config_manager.config.stim_location,
                "Stimulus Location",
                on_change=self._on_stim_loc_field_changed,
            )
        if not hasattr(self, 'form_dfilter'):
            self.form_dfilter = PydanticFormWidget(
                self.config_manager.config.dendritic_filter,
                "Dendritic Filter",
                on_change=self._on_dfilter_field_changed,
            )

        self.lbl_attenuation_hint = QLabel("")
        self.lbl_attenuation_hint.setStyleSheet("color: #6C7086; font-size: 11px; padding: 4px;")
        self.lbl_attenuation_hint.setWordWrap(True)

        left_layout.addWidget(self.form_stim)
        left_layout.addWidget(self.form_stim_loc)
        left_layout.addWidget(self.form_dfilter)
        left_layout.addWidget(self.lbl_attenuation_hint)
        left_layout.addStretch()

        self.dual_stim_widget = getattr(self, "dual_stim_widget", DualStimulationWidget())
        if not self._dual_stim_signal_connected:
            self.dual_stim_widget.config_changed.connect(self._on_dual_stim_config_changed)
            self._dual_stim_signal_connected = True

        stim_layout.addWidget(left_col, 1)
        stim_layout.addWidget(self.dual_stim_widget, 1)

        stim_splitter = QSplitter(Qt.Orientation.Vertical)
        stim_splitter.setChildrenCollapsible(False)
        stim_splitter.addWidget(stim_controls_widget)

        preview_container = QWidget()
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(4)

        preview_controls = QHBoxLayout()
        preview_controls.setSpacing(8)

        btn_refresh_preview = QPushButton("🔄 Refresh Preview")
        btn_refresh_preview.setToolTip("Manually refresh stimulus preview")
        btn_refresh_preview.clicked.connect(self._update_stim_preview)
        btn_refresh_preview.setMaximumWidth(140)

        self._toggle_conductance = QPushButton("I / g")
        self._toggle_conductance.setCheckable(True)
        self._toggle_conductance.setChecked(False)
        self._toggle_conductance.setToolTip("Toggle between current and conductance view")
        self._toggle_conductance.setMaximumWidth(60)
        self._toggle_conductance.clicked.connect(self._update_stim_preview)

        preview_controls.addWidget(btn_refresh_preview)
        preview_controls.addWidget(self._toggle_conductance)
        preview_controls.addStretch()

        self._stim_preview_plot = pg.PlotWidget(title="Stimulus Protocol Preview")
        self._stim_preview_plot.setLabel('left', 'Total Input Current')
        self._stim_preview_plot.setLabel('bottom', 'Time (ms)')
        self._stim_preview_plot.setBackground('#1E1E2E')
        self._stim_preview_plot.setMinimumHeight(150)

        preview_layout.addLayout(preview_controls)
        preview_layout.addWidget(self._stim_preview_plot)

        stim_splitter.addWidget(preview_container)
        stim_splitter.setSizes([250, 300])
        stim_splitter.setStretchFactor(0, 1)
        stim_splitter.setStretchFactor(1, 2)

        stim_main_layout.addWidget(stim_splitter)

        return stim_studio_widget

    def _setup_view_menu(self):
        """Create View menu with dock toggle actions."""
        menubar = self.menuBar()
        view_menu = menubar.addMenu("View")

        docks = [
            ("Parameters", getattr(self, '_dock_params', None)),
            ("Live Controls", getattr(self, '_dock_live', None)),
            ("Analytics", getattr(self, '_dock_analytics', None)),
            ("Topology", getattr(self, '_dock_topology', None)),
            ("Stimulation Studio", getattr(self, '_dock_stim', None)),
            ("Axon Biophysics", getattr(self, '_dock_axon', None)),
            ("Guide", getattr(self, '_dock_guide', None)),
        ]

        for name, dock in docks:
            if dock is not None:
                action = dock.toggleViewAction()
                action.setText(name)
                view_menu.addAction(action)

        view_menu.addSeparator()

        reset_action = QAction("Reset Layout", self)
        reset_action.triggered.connect(self._reset_dock_layout)
        view_menu.addAction(reset_action)

        analytics_action = QAction("Analytics Workspace", self)
        analytics_action.triggered.connect(self._focus_analytics_workspace)
        view_menu.addAction(analytics_action)

        view_menu.addSeparator()
        for preset_name in ("Laptop", "Desktop", "Presentation", "Debug"):
            action = QAction(f"{preset_name} Layout", self)
            action.triggered.connect(
                lambda _checked=False, name=preset_name: self._apply_layout_preset(name)
            )
            view_menu.addAction(action)

    def _focus_analytics_workspace(self) -> None:
        """Prioritize Analytics without hiding the core simulation controls."""
        if hasattr(self, "_dock_params"):
            self._dock_params.setVisible(True)
        if hasattr(self, "_dock_analytics"):
            self._dock_analytics.setVisible(True)
            self._dock_analytics.raise_()
            try:
                self.resizeDocks(
                    [self._dock_analytics],
                    [max(320, int(self.height() * 0.46))],
                    Qt.Orientation.Vertical,
                )
            except Exception as exc:
                logger.debug("Could not resize analytics dock: %s", exc)
        if hasattr(self, "analytics"):
            self.analytics.setCurrentIndex(0)
        self._status("Analytics workspace")

    def _reset_dock_layout(self):
        """Reset dock layout to default state."""
        # Main docks
        if hasattr(self, '_dock_params'):
            self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._dock_params)
            self._dock_params.setVisible(True)
        if hasattr(self, '_dock_live'):
            self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._dock_live)
            self._dock_live.setVisible(True)
        if hasattr(self, '_dock_params') and hasattr(self, '_dock_live'):
            self.tabifyDockWidget(self._dock_params, self._dock_live)
        if hasattr(self, '_dock_analytics'):
            self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._dock_analytics)
            self._dock_analytics.setVisible(True)
        # Tabify right side: Live Controls + Topology + Guide
        if hasattr(self, '_dock_topology') and hasattr(self, '_dock_live'):
            self.tabifyDockWidget(self._dock_live, self._dock_topology)
            self._dock_topology.setVisible(True)
        if hasattr(self, '_dock_guide') and hasattr(self, '_dock_live'):
            self.tabifyDockWidget(self._dock_live, self._dock_guide)
            self._dock_guide.setVisible(True)
        # Floating docks: Stimulation Studio + Axon Biophysics (tabbed together)
        if hasattr(self, '_dock_stim'):
            self._dock_stim.setFloating(True)
            self._dock_stim.move(self.x() + 100, self.y() + 100)
            self._dock_stim.resize(900, 700)
            self._dock_stim.setVisible(True)
        if hasattr(self, '_dock_axon') and hasattr(self, '_dock_stim'):
            self.tabifyDockWidget(self._dock_stim, self._dock_axon)
            self._dock_axon.setVisible(True)

    def _apply_layout_preset(self, preset_name: str, *, resize_window: bool = True) -> None:
        preset = LAYOUT_PRESETS.get(preset_name, LAYOUT_PRESETS["Laptop"])
        if resize_window:
            self.resize(preset.width, preset.height)
        if hasattr(self, "_sidebar_frame"):
            self._sidebar_frame.setMaximumWidth(preset.sidebar_max_width)
        self._set_secondary_actions_visible(preset.show_secondary_actions)
        if hasattr(self, "_dock_stim"):
            self._dock_stim.setFloating(preset.float_stimulation)
            self._dock_stim.setVisible(preset.float_stimulation)
            if preset.float_stimulation:
                self._dock_stim.resize(900, 700)
        if hasattr(self, "_dock_axon"):
            self._dock_axon.setVisible(preset.float_stimulation)
        if hasattr(self, "_dock_live"):
            self._dock_live.setVisible(True)
        if hasattr(self, "_dock_topology"):
            self._dock_topology.setVisible(preset.name != "Laptop")
        if hasattr(self, "_sb"):
            self._status(f"{preset.name} layout")

    def _toggle_ui_complexity(self, mode: str):
        """Toggle UI complexity based on experience mode.
        
        Microscope: Simplified for students - hide topology/axon docks, hide morphology/analysis groups, force single-comp
        Bridge: Full functionality
        Research: Future SWC/Network support (identical to Bridge for now)
        """
        if mode == "🔬 Microscope":
            # Hide topology and axon biophysics docks
            if hasattr(self, '_dock_topology'):
                self._dock_topology.setVisible(False)
            if hasattr(self, '_dock_axon'):
                self._dock_axon.setVisible(False)
            
            # Hide morphology and analysis groups in sidebar
            if hasattr(self, 'grp_morph'):
                self.grp_morph.setVisible(False)
            if hasattr(self, 'grp_ana'):
                self.grp_ana.setVisible(False)
            
            # Force single-compartment mode
            self.config_manager.config.morphology.single_comp = True
            self._refresh_all_forms()
            
        elif mode == "🌉 Bridge":
            # Show all docks
            for dock_name in ('_dock_params', '_dock_live', '_dock_analytics', 
                              '_dock_topology', '_dock_stim', '_dock_axon', '_dock_guide'):
                dock = getattr(self, dock_name, None)
                if dock:
                    dock.setVisible(True)
            
            # Show all sidebar groups
            if hasattr(self, 'grp_morph'):
                self.grp_morph.setVisible(True)
            if hasattr(self, 'grp_ana'):
                self.grp_ana.setVisible(True)
            
        elif mode == "🧪 Research":
            # Identical to Bridge for now (future SWC/Network support)
            for dock_name in ('_dock_params', '_dock_live', '_dock_analytics',
                              '_dock_topology', '_dock_stim', '_dock_axon', '_dock_guide'):
                dock = getattr(self, dock_name, None)
                if dock:
                    dock.setVisible(True)
            
            if hasattr(self, 'grp_morph'):
                self.grp_morph.setVisible(True)
            if hasattr(self, 'grp_ana'):
                self.grp_ana.setVisible(True)

    def _val_to_live_slider(self, param_name: str, row_i: int | None = None) -> int:
        # Try custom path resolution first
        obj, attr = self._resolve_param(param_name)
        if obj is not None and attr is not None:
            val = getattr(obj, attr)
            # Check if this is a known parameter with specific ranges
            simple_name = param_name.split('.')[-1] if '.' in param_name else param_name
            if simple_name in _LIVE_PARAMS:
                lo, hi, _, _ = _LIVE_PARAMS[simple_name]
            elif row_i is not None and row_i in self._live_custom_bounds:
                lo, hi = self._live_custom_bounds[row_i]
            else:
                # One-time fallback bounds for unresolved custom rows.
                if val == 0.0:
                    lo, hi = -1.0, 1.0
                elif val > 0:
                    lo, hi = val * 0.1, val * 5.0
                else:
                    lo, hi = val * 5.0, val * 0.1
        elif param_name in _LIVE_PARAMS:
            lo, hi, _, getter = _LIVE_PARAMS[param_name]
            val = getter(self.config_manager.config)
        else:
            # Fallback to Iext if unknown
            lo, hi = -50.0, 250.0
            val = self.config_manager.config.stim.Iext
        return int(round(max(0.0, min(1.0, (val - lo) / max(hi - lo, 1e-9))) * _LIVE_SLIDER_STEPS))

    def _resolve_param(self, path: str):
        """Safely get object and attribute name from a dot-separated path (e.g. 'channels.gNa_max')."""
        if not path or not path.strip():
            return None, None
        
        obj = self.config_manager.config
        parts = [p for p in path.split('.') if p]  # Filter out empty parts
        
        if not parts:
            return None, None
        
        for part in parts[:-1]:
            if not hasattr(obj, part):
                return None, None
            obj = getattr(obj, part)
        if not hasattr(obj, parts[-1]):
            return None, None
        return obj, parts[-1]

    def _canonical_sweep_param(self, path: str) -> str:
        text = str(path or "").strip()
        if not text:
            return "stim.Iext"
        if "." in text:
            return text
        if hasattr(self.config_manager.config, text):
            return text
        for section in (
            "stim",
            "channels",
            "env",
            "morphology",
            "calcium",
            "metabolism",
            "analysis",
            "preset_modes",
            "stim_location",
            "dendritic_filter",
        ):
            obj = getattr(self.config_manager.config, section, None)
            if obj is not None and hasattr(obj, text):
                return f"{section}.{text}"
        return text

    def _refresh_sweep_param_ui(self):
        if not hasattr(self, "_sweep_param_combo"):
            return
        ana = self.config_manager.config.analysis
        canonical = self._canonical_sweep_param(getattr(ana, "sweep_param", "stim.Iext"))
        ana.sweep_param = canonical
        self._sweep_param_combo.blockSignals(True)
        self._sweep_param_combo.setCurrentText(canonical)
        self._sweep_param_combo.blockSignals(False)
        obj, attr = self._resolve_param(canonical)
        if obj is None or attr is None:
            self._sweep_param_help.setText(
                "Enter a dotted path like stim.Iext or channels.gNa_max. Short names are accepted and normalized."
            )
            self._sweep_param_combo.setStyleSheet("color:#F38BA8;")
        else:
            current_value = getattr(obj, attr)
            self._sweep_param_help.setText(
                f"Current target: {canonical} = {float(current_value):.4g}. "
                "This field drives both the top SWEEP button and the dedicated f-I run."
            )
            self._sweep_param_combo.setStyleSheet("")

    def _apply_form_ux_filters(self, *_args):
        search = self._form_search.text() if hasattr(self, "_form_search") else ""
        priority = self._form_priority.currentText() if hasattr(self, "_form_priority") else "all"
        for form in (
            getattr(self, "form_stim", None),
            getattr(self, "form_stim_loc", None),
            getattr(self, "form_dfilter", None),
            getattr(self, "form_preset_modes", None),
            getattr(self, "form_chan", None),
            getattr(self, "form_calcium", None),
            getattr(self, "form_metabolism", None),
            getattr(self, "form_morph", None),
            getattr(self, "form_env", None),
            getattr(self, "form_ana", None),
        ):
            if form is None:
                continue
            if hasattr(form, "set_search_filter"):
                form.set_search_filter(search)
            if hasattr(form, "set_priority_filter"):
                form.set_priority_filter(priority)

    def _on_sweep_param_changed(self, text: str):
        self.config_manager.config.analysis.sweep_param = self._canonical_sweep_param(text)
        self._refresh_sweep_param_ui()

    def _on_live_combo_changed(self, row_i: int):
        """Sync slider position to new parameter's current value."""
        param = self._live_combos[row_i].currentText()
        combo = self._live_combos[row_i]

        # Try custom path resolution first
        obj, attr = self._resolve_param(param)
        if obj is not None and attr is not None:
            val = getattr(obj, attr)
            # Cache bounds based on initial value to prevent runaway drift
            if param in _LIVE_PARAMS:
                lo, hi, _, _ = _LIVE_PARAMS[param]
            else:
                if val == 0.0:
                    lo, hi = -1.0, 1.0
                elif val > 0:
                    lo, hi = val * 0.1, val * 5.0
                else:
                    lo, hi = val * 5.0, val * 0.1
                self._live_custom_bounds[row_i] = (lo, hi)
            combo.setStyleSheet("")  # Clear error style
        elif param in _LIVE_PARAMS:
            lo, hi, _, getter = _LIVE_PARAMS[param]
            val = getter(self.config_manager.config)
            combo.setStyleSheet("")  # Clear error style
        else:
            combo.setStyleSheet("color: #F38BA8;")  # Red text for invalid path
            return

        new_pos = int(round(max(0.0, min(1.0, (val - lo) / max(hi - lo, 1e-9))) * _LIVE_SLIDER_STEPS))
        self._live_sliders[row_i].blockSignals(True)
        self._live_sliders[row_i].setValue(new_pos)
        self._live_sliders[row_i].blockSignals(False)
        self._live_labels[row_i].setText(f"{val:.2f}")

    def _on_live_slider_moved(self, row_i: int, raw_val: int):
        param = self._live_combos[row_i].currentText()
        combo = self._live_combos[row_i]
        slider = self._live_sliders[row_i]

        # Try custom path resolution first
        obj, attr = self._resolve_param(param)
        if obj is not None and attr is not None:
            # Use cached bounds instead of recalculating to prevent runaway drift
            if param in _LIVE_PARAMS:
                lo, hi, _, _ = _LIVE_PARAMS[param]
            elif row_i in self._live_custom_bounds:
                lo, hi = self._live_custom_bounds[row_i]
            else:
                lo, hi = -100.0, 500.0  # Fallback
            
            # Dynamic precision: use logarithmic mapping for very small scales
            range_size = hi - lo
            if range_size < 0.001 and lo > 0 and hi > 0:
                # Logarithmic mapping for small positive values
                val = lo * (hi / lo) ** (raw_val / _LIVE_SLIDER_STEPS)
                self._live_labels[row_i].setText(f"{val:.6f}")
            elif range_size < 0.1:
                # Higher precision for small ranges
                val = lo + (raw_val / _LIVE_SLIDER_STEPS) * (hi - lo)
                self._live_labels[row_i].setText(f"{val:.6f}")
            else:
                # Standard precision
                val = lo + (raw_val / _LIVE_SLIDER_STEPS) * (hi - lo)
                self._live_labels[row_i].setText(f"{val:.2f}")
            
            setattr(obj, attr, val)
            combo.setStyleSheet("")  # Clear error style
        elif param in _LIVE_PARAMS:
            lo, hi, setter, _ = _LIVE_PARAMS[param]
            range_size = hi - lo
            if range_size < 0.001 and lo > 0 and hi > 0:
                # Logarithmic mapping for small positive values
                val = lo * (hi / lo) ** (raw_val / _LIVE_SLIDER_STEPS)
                self._live_labels[row_i].setText(f"{val:.6f}")
            elif range_size < 0.1:
                # Higher precision for small ranges
                val = lo + (raw_val / _LIVE_SLIDER_STEPS) * (hi - lo)
                self._live_labels[row_i].setText(f"{val:.6f}")
            else:
                # Standard precision
                val = lo + (raw_val / _LIVE_SLIDER_STEPS) * (hi - lo)
                self._live_labels[row_i].setText(f"{val:.2f}")
            setter(self.config_manager.config, val)
            combo.setStyleSheet("")  # Clear error style
        else:
            combo.setStyleSheet("color: #F38BA8;")  # Red text for invalid path
            # Reset slider to previous valid position (based on current actual value)
            slider.blockSignals(True)
            slider.setValue(self._val_to_live_slider(param, row_i))
            slider.blockSignals(False)
            return

        # v12.7: Debounce Fix - Stop analytics timer during slider movement
        # Prevent race between solver and heavy analytics during rapid slider changes
        if self._analytics_debounce_timer.isActive():
            self._analytics_debounce_timer.stop()
        # Debounced auto-run if HINES mode is active
        self._live_timer.start()

    def _on_quick_set_changed(self, selection: str):
        """Handle Quick-Set dropdown selection to configure live deck sliders."""
        if selection == "— Quick-Set —":
            return

        quick_set_params = {
            "Channel Densities": ["channels.gNa_max", "channels.gK_max", "channels.gCa_max"],
            "Dendritic Cable": ["morphology.Ra", "dendritic_filter.distance_um", "dendritic_filter.space_constant_um"],
            "Metabolism": ["metabolism.atp_synthesis_rate", "metabolism.g_katp_max", "calcium.tau_Ca"],
        }

        if selection not in quick_set_params:
            return

        params = quick_set_params[selection]
        # Ensure we have enough rows in the live deck
        while len(self._live_combos) < len(params):
            # Add a new row (copy from existing row creation logic)
            row_i = len(self._live_combos)
            row_w = QWidget()
            row_h = QHBoxLayout(row_w)
            row_h.setContentsMargins(0, 0, 0, 0)
            row_h.setSpacing(4)

            combo = QComboBox()
            combo.setEditable(True)
            combo.addItems(_LIVE_PARAM_SUGGESTIONS)
            combo.setCurrentText("")
            combo.setFixedWidth(120)
            combo.currentTextChanged.connect(
                lambda _text, ri=row_i: self._on_live_combo_changed(ri)
            )
            self._live_combos.append(combo)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, _LIVE_SLIDER_STEPS)
            slider.setValue(0)
            slider.valueChanged.connect(
                lambda raw, ri=row_i: self._on_live_slider_moved(ri, raw)
            )
            self._live_sliders.append(slider)

            lbl = QLabel("0.00")
            lbl.setFixedWidth(52)
            lbl.setStyleSheet("color:#CBA6F7; font-size:11px;")
            self._live_labels.append(lbl)

            row_h.addWidget(combo)
            row_h.addWidget(slider, 1)
            row_h.addWidget(lbl)
            # Get the deck layout
            deck = self.findChild(QGroupBox, "🎛️ Live Control Deck")
            if deck:
                deck.layout().addWidget(row_w)

        # Set the parameters
        for i, param in enumerate(params):
            if i < len(self._live_combos):
                self._live_combos[i].setCurrentText(param)
                self._on_live_combo_changed(i)

        self._sync_live_deck_to_config()

    def _on_live_timer_fired(self):
        """Auto-run after debounce if HINES is active."""
        if not self.btn_hines.isChecked():
            return
        # Prevent thread race: don't start new simulation if one is already running
        if not self.btn_run.isEnabled():
            return
        # Mark as live run to prevent tab switching
        self._is_live_run = True
        try:
            self.sim_controller.run_single(
                self.config_manager.config,
                on_success=self._on_simulation_done,
                on_error=self._on_sim_error,
            )
        except Exception:
            self._is_live_run = False  # Reset on error

    # ─────────────────────────────────────────────────────────────────
    #  SPARKLINE
    # ─────────────────────────────────────────────────────────────────
    def _update_sparkline(self, res) -> None:
        """Update the mini Vm trace in the top bar after each simulation."""
        try:
            t = res.t
            v = res.v_soma
            # Downsample to ≤512 points for performance
            step = max(1, len(t) // 512)
            self._sparkline_curve.setData(t[::step], v[::step])
            self._sparkline.setYRange(float(v.min()) - 2, float(v.max()) + 2, padding=0)
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────
    #  STATUS BAR
    # ─────────────────────────────────────────────────────────────────
    def _setup_status_bar(self):
        self._sb = QStatusBar()
        self.setStatusBar(self._sb)

    def _status(self, msg: str):
        self._sb.showMessage(msg)

    def _sync_stim_type_controls(self):
        """Show only stimulation parameters relevant for current stim_type."""
        # Primary stimulus type always comes from canonical config.stim.
        stype = str(getattr(self.config_manager.config.stim, "stim_type", "const"))
        stim_fields = self.form_stim.widgets_map
        labels = self.form_stim.labels_map

        alpha_like = {"alpha"}
        synaptic_like = {"AMPA", "NMDA", "GABAA", "GABAB", "Kainate", "Nicotinic"}
        pulse_like = {"pulse"}
        zap_like = {"zap"}

        show_pulse_start = stype in (alpha_like | synaptic_like | pulse_like | zap_like)
        show_pulse_dur = stype in (pulse_like | zap_like)
        show_alpha_tau = stype in (alpha_like | synaptic_like)

        show_zap = stype in zap_like

        visibility = {
            "pulse_start": show_pulse_start,
            "pulse_dur": show_pulse_dur,
            "alpha_tau": show_alpha_tau,
            "zap_f0_hz": show_zap,
            "zap_f1_hz": show_zap,
        }
        for field_name, is_visible in visibility.items():
            if hasattr(self.form_stim, "hidden_fields"):
                if is_visible:
                    self.form_stim.hidden_fields.discard(field_name)
                else:
                    self.form_stim.hidden_fields.add(field_name)
            w = stim_fields.get(field_name)
            l = labels.get(field_name)
            if w is not None:
                w.setVisible(is_visible)
            if l is not None:
                l.setVisible(is_visible)
        if hasattr(self.form_stim, "_apply_visibility_filters"):
            self.form_stim._apply_visibility_filters()

    def _on_morph_field_changed(self, field_name: str, _value):
        self.oscilloscope.sync_delay_controls_for_config(self.config_manager.config)
        # Always refresh hint so absolute current (nA) reflects updated geometry.
        self.lbl_params_hint.setText(self.config_manager.get_hint_text())
        # Update soma diameter in stim form if d_soma changed
        if field_name == 'd_soma' and hasattr(self.form_stim, 'update_soma_diameter'):
            self.form_stim.update_soma_diameter(self.config_manager.config.morphology.d_soma)
        self._refresh_topology_preview()
        self._update_stim_preview()
        # Sync live deck to reflect morphology changes (e.g., diameter affects current density)
        self._sync_live_deck_to_config()

    def _on_stim_field_changed(self, field_name: str, value):
        if field_name == "stim_type":
            stype = str(value)
            cfg = self.config_manager.config
            syn_map = {
                "AMPA": "SYN: AMPA-receptor (Fast Excitation, 1-3 ms)",
                "NMDA": "SYN: NMDA-receptor (Slow Excitation, 50-100 ms)",
                "GABAA": "SYN: GABA-A receptor (Fast Inhibition, 3-5 ms)",
                "GABAB": "SYN: GABA-B receptor (Slow Inhibition, 100-300 ms)",
                "Kainate": "SYN: Kainate-receptor (Intermediate, 10-15 ms)",
                "Nicotinic": "SYN: Nicotinic ACh (Fast Excitation, 5-10 ms)",
            }
            syn_name = syn_map.get(stype)
            if syn_name is not None:
                apply_synaptic_stimulus(cfg, syn_name)
            
            # Auto-set appropriate defaults based on stimulus type (v10.3)
            if stype == "zap":
                # ZAP/impedance stimulus: set reasonable frequency sweep defaults
                if cfg.stim.zap_f0_hz < 0.1:
                    cfg.stim.zap_f0_hz = 0.5  # Start at 0.5 Hz
                if cfg.stim.zap_f1_hz < cfg.stim.zap_f0_hz:
                    cfg.stim.zap_f1_hz = 40.0  # End at 40 Hz
                if cfg.stim.zap_rise_ms == 0:
                    cfg.stim.zap_rise_ms = 5.0  # Smooth ramp to avoid spectral leakage
                # Longer pulse for frequency sweep
                if cfg.stim.pulse_dur < 50:
                    cfg.stim.pulse_dur = 100.0  # 100ms for full sweep
            elif stype in ("const", "pulse"):
                # Classical stimulation: reset ZAP-specific parameters
                cfg.stim.zap_f0_hz = 0.5
                cfg.stim.zap_f1_hz = 40.0
                cfg.stim.zap_rise_ms = 0.0  # No Tukey window for simple pulses
            elif stype in syn_map:
                # Synaptic stimulation: set appropriate pulse duration
                if cfg.stim.pulse_dur < 1:
                    cfg.stim.pulse_dur = 50.0  # Standard synaptic input duration
            
            self.form_stim.refresh()
            self._sync_stim_type_controls()
        if field_name in {"Iext", "stim_type"}:
            self.lbl_params_hint.setText(self.config_manager.get_hint_text())
            self.form_stim.refresh()
            self._sync_stim_type_controls()
        if field_name == "t_sim":
            # t_sim change requires updating stim preview with new time range
            self._update_stim_preview()
        self.lbl_params_hint.setText(self.config_manager.get_hint_text())
        self._refresh_topology_preview()
        if field_name != "t_sim":
            self._update_stim_preview()
        self._sync_live_deck_to_config()

    def _on_stim_loc_field_changed(self, _field_name: str, _value):
        self.lbl_params_hint.setText(self.config_manager.get_hint_text())
        self._refresh_topology_preview()
        self._update_stim_preview()
        self._sync_live_deck_to_config()

    def _on_dfilter_field_changed(self, _field_name: str, _value):
        self._refresh_topology_preview()
        self._update_stim_preview()
        self._update_attenuation_hint()
        self._sync_live_deck_to_config()

    def _update_attenuation_hint(self):
        """Update dynamic attenuation hint for dendritic filter (v10.3+)."""
        if not hasattr(self, "lbl_attenuation_hint"):
            return
        dfilter = self.config_manager.config.dendritic_filter
        if not dfilter.enabled:
            self.lbl_attenuation_hint.setText("")
            return
        
        distance = dfilter.distance_um
        space_const = dfilter.space_constant_um
        tau_m = 10.0  # Default membrane time constant (ms)
        
        if dfilter.filter_mode == "Physiological (AC)":
            freq = dfilter.input_frequency
            attn = get_ac_attenuation(distance, space_const, tau_m, freq)
            mode_str = f"at {freq:.0f}Hz"
        else:
            attn = np.exp(-distance / space_const)
            mode_str = "DC"
        
        # Calculate propagation delay
        velocity = getattr(dfilter, 'conduction_velocity_um_ms', 250.0)
        delay_ms = distance / max(velocity, 1e-12)
        
        # Bilingual hint with attenuation and delay info
        if T.lang == 'RU':
            text = f"Затухание: {attn:.2f} ({mode_str}) | Задержка: {delay_ms:.2f} мс | v={velocity:.0f} µm/ms"
        else:
            text = f"Attenuation: {attn:.2f} ({mode_str}) | Delay: {delay_ms:.2f} ms | v={velocity:.0f} µm/ms"
        self.lbl_attenuation_hint.setText(text)

    def _on_channel_field_changed(self, _field_name: str, _value):
        self._refresh_topology_preview()
        self._sync_live_deck_to_config()

    def _refresh_topology_preview(self):
        if not hasattr(self, "topology"):
            return
        dual_cfg = (
            self.dual_stim_widget.config
            if hasattr(self, "dual_stim_widget") and self.dual_stim_widget.config.enabled
            else None
        )
        self.topology.draw_neuron(
            self.config_manager.config,
            dual_config=dual_cfg,
            delay_target_name=self._delay_target_name,
            delay_custom_index=self._delay_custom_index,
        )

    def _on_delay_target_changed(self, target_name: str, custom_index: int):
        self._delay_target_name = str(target_name)
        self._delay_custom_index = int(custom_index)
        if hasattr(self, "topology"):
            self.topology.set_delay_focus(
                self._delay_target_name,
                self._delay_custom_index,
            )

    def _sync_dual_stim_into_config(self) -> bool:
        """
        Sync dual-stimulation GUI config into main model config.
        Delegated to ConfigManager.

        Returns
        -------
        bool
            True if dual stimulation is enabled.
        """
        return self.config_manager.sync_dual_stim_into_config()

    def _sync_preset_mode_controls(self):
        """Show only the mode selector that applies to the active preset."""
        preset_name = self.config_manager.current_preset_name or ""
        preset_code = preset_name.partition(":")[0].strip().upper()
        active = {
            "l5_mode": preset_code == "B",
            "purkinje_mode": preset_code == "E",
            "anesthesia_mode": preset_code == "G",
            "k_mode": preset_code == "K",
            "alzheimer_mode": preset_code == "N",
            "hypoxia_mode": preset_code == "O",
            "ach_mode": preset_code == "R",
            "dravet_mode": preset_code == "S",
        }
        title_suffix = []
        any_active = False
        for field_name, widget in self.form_preset_modes.widgets_map.items():
            is_active = bool(active.get(field_name, False))
            widget.setEnabled(is_active)
            widget.setVisible(is_active)
            label = self.form_preset_modes.labels_map.get(field_name)
            if label is not None:
                label.setVisible(is_active)
            if is_active:
                any_active = True
                title_suffix.append(field_name.replace("_mode", "").upper())
                widget.setToolTip("Active for current preset.")
            else:
                widget.setToolTip("Ignored for current preset.")
        if any_active:
            self.form_preset_modes.group_box.setTitle(
                f"Preset Modes ({'/'.join(title_suffix)})"
            )
        else:
            self.form_preset_modes.group_box.setTitle("Preset Modes (not used for current preset)")
        self.lbl_params_hint.setText(self.config_manager.get_hint_text())

    # ─────────────────────────────────────────────────────────────────
    #  PRESET & LANGUAGE
    # ─────────────────────────────────────────────────────────────────
    def load_preset(self, name: str):
        """
        Apply a named preset configuration and update the UI and internal state to reflect it.
        
        Ignores placeholder names containing "—" or "Select". Synchronizes the preset selection controls without emitting signals, loads the preset into the configuration manager, performs preset-specific postprocessing (auto-selects the jacobian mode, updates the Hines toggle state), refreshes all form widgets, synchronizes oscilloscope delay controls, resets the dual-stimulation widget to its preset defaults, updates preset-mode controls, redraws the topology using the current delay focus, and updates the status bar with the applied preset name and active mode suffix.
        """
        if "—" in name or "Select" in name:
            return
        # Keep both preset combos in sync without re-firing
        for combo in (self.combo_presets, getattr(self, '_sidebar_preset_combo', None)):
            if combo is None:
                continue
            try:
                if combo.currentText() != name:
                    combo.blockSignals(True)
                    combo.setCurrentText(name)
                    combo.blockSignals(False)
            except RuntimeError:
                # Qt C++ object was deleted - skip this combo
                pass
        self.config_manager.load_preset(name)
        self.config_manager.auto_select_jacobian_for_preset()
        self._sync_hines_button_state()
        self._refresh_all_forms()
        self.oscilloscope.sync_delay_controls_for_config(self.config_manager.config)
        # Re-sync local values after oscilloscope clamps them to valid range
        self._delay_target_name, self._delay_custom_index = (
            self.oscilloscope.get_delay_target_selection()
        )
        
        # --- ФИКС: Синхронизируем виджет с пресетом, а не сбрасываем его ---
        if self.config_manager.config.dual_stimulation is not None:
            self.dual_stim_widget.config = self.config_manager.config.dual_stimulation
            self.dual_stim_widget.update_ui_from_config()
        else:
            self.dual_stim_widget.load_default_preset()
            
        self._sync_preset_mode_controls()
        self.topology.draw_neuron(
            self.config_manager.config,
            delay_target_name=self._delay_target_name,
            delay_custom_index=self._delay_custom_index,
        )
        self._status(f"Preset applied: {name}{self.config_manager.active_mode_suffix()}")

    def _sync_live_deck_to_config(self):
        """Sync all live deck sliders/labels to reflect the current config values."""
        for row_i, (combo, slider, lbl) in enumerate(
                zip(self._live_combos, self._live_sliders, self._live_labels)):
            param = combo.currentText()
            
            # Try custom path resolution first
            obj, attr = self._resolve_param(param)
            if obj is not None and attr is not None:
                val = getattr(obj, attr)
                # Use reasonable default ranges for custom paths
                lo, hi = -100.0, 500.0
                if param in _LIVE_PARAMS:
                    lo, hi, _, _ = _LIVE_PARAMS[param]
                elif row_i in self._live_custom_bounds:
                    lo, hi = self._live_custom_bounds[row_i]
                
                # DYNAMIC BOUNDS EXPANSION: if preset value is outside cached bounds,
                # expand the range to accommodate it (with smart capping)
                if val < lo or val > hi:
                    margin = abs(val) * 0.5 if val != 0 else 1.0
                    # Cap margin for physiological values to prevent huge ranges
                    if abs(val) < 1000 and margin > 1000:
                        margin = 1000
                    new_lo = max(-1e6, val - margin)  # Hard limit ±1M
                    new_hi = min(1e6, val + margin)
                    lo, hi = new_lo, new_hi
                    if param not in _LIVE_PARAMS:
                        self._live_custom_bounds[row_i] = (lo, hi)  # Update cache
            elif param in _LIVE_PARAMS:
                lo, hi, _, getter = _LIVE_PARAMS[param]
                val = getter(self.config_manager.config)
            else:
                continue
            
            pos = int(round(max(0.0, min(1.0, (val - lo) / max(hi - lo, 1e-9))) * _LIVE_SLIDER_STEPS))
            slider.blockSignals(True)
            slider.setValue(pos)
            slider.blockSignals(False)
            lbl.setText(f"{val:.2f}")
    
    def _update_stim_preview(self):
        """Quickly plot the expected stimulus without running the full HH solver."""
        if not hasattr(self, '_stim_preview_plot'):
            return
        
        try:
            cfg = self.config_manager.config.stim
            # Use synaptic train duration if available, otherwise use simulation time
            t_sim = cfg.t_sim
            if cfg.synaptic_train_type != 'none':
                t_sim = max(t_sim, cfg.synaptic_train_duration_ms + cfg.pulse_start + 50)
            n_preview = int(min(max(t_sim * 10, 1000), 50000))
            t = np.linspace(0, t_sim, n_preview)
            
            # Use existing reconstruct_stimulus_trace logic
            # Construct a dummy SimulationResult with just t and config
            class DummyResult:
                pass
            res = DummyResult()
            res.t = t
            res.config = self.config_manager.config
            res.v_dendritic_filtered = None  # Ignore filter state for pure input preview
            
            from core.analysis import reconstruct_stimulus_trace
            i_stim = reconstruct_stimulus_trace(res)
            
            # Check if conductance view is enabled
            show_conductance = self._toggle_conductance.isChecked()
            
            self._stim_preview_plot.clear()
            
            if show_conductance:
                # Show conductance (g) instead of current (I)
                # For conductance-based synapses, i_stim is already conductance
                # For current-based stimuli, we need to convert I to g by dividing by (V - Erev)
                # For preview purposes, use V = -60 mV (typical resting potential)
                v_rest = -60.0
                # For excitatory synapses (AMPA, NMDA), Erev ≈ 0 mV
                # For inhibitory synapses (GABA), Erev ≈ -75 mV
                # Simplified: assume excitatory for preview
                e_exc = 0.0
                g_stim = i_stim / (v_rest - e_exc) if (v_rest - e_exc) != 0 else i_stim
                self._stim_preview_plot.plot(t, g_stim, pen=pg.mkPen('#89B4FA', width=2))
                self._stim_preview_plot.setLabel('left', 'Conductance', units='mS/cm²')
            else:
                # Show current (I)
                self._stim_preview_plot.plot(t, i_stim, pen=pg.mkPen('#F5C2E7', width=2))
                self._stim_preview_plot.setLabel('left', 'Total Input Current', units='µA/cmÂ˛')
        except Exception as e:
            # Show error message in plot area instead of silent failure
            self._stim_preview_plot.clear()
            from pyqtgraph import TextItem
            error_text = TextItem(
                f"Preview unavailable: {str(e)[:50]}...",
                color='#F38BA8',
                anchor=(0.5, 0.5)
            )
            error_text.setPos(self._stim_preview_plot.width() / 2, 0)
            self._stim_preview_plot.addItem(error_text)

    def _on_find_threshold_clicked(self):
        """v13.0: Run Auto-Rheobase Search using non-blocking async worker.

        Previously ran on main thread, freezing UI. Now uses SimulationController
        worker with progress updates. UI remains responsive during search.
        """
        self._lbl_rheobase_result.setText("🔍 Searching for threshold...")
        self._status("Running Auto-Rheobase Search...")
        self._lock_ui(True)  # Lock UI during search

        def on_progress(message, data):
            """Update UI with progress from worker thread (via signal)."""
            self._lbl_rheobase_result.setText(f"🔍 {message}")
            if data.get('phase') == 2:
                self._status(f"Auto-Rheobase: iter {data['iteration']}, I={data['I_mid']:.2f}")

        def on_success(result):
            """Handle completed rheobase search."""
            I_rheobase = result['I_rheobase']
            uncertainty = result['uncertainty']
            search_history = result['search_history']

            # Build visualization
            viz_text = self._format_rheobase_visualization(search_history, I_rheobase, uncertainty)

            # Guard against division by zero
            if abs(I_rheobase) > 1e-9:
                uncertainty_pct = uncertainty / I_rheobase * 100
                uncertainty_str = f"{uncertainty_pct:.1f}%"
            else:
                uncertainty_str = "N/A (I_rheobase ≈ 0)"

            self._lbl_rheobase_result.setText(
                f"✅ Rheobase: {I_rheobase:.2f} ± {uncertainty:.2f} µA/cm²\n"
                f"   Uncertainty: {uncertainty_str}\n\n"
                f"{viz_text}"
            )
            self._status(f"Auto-Rheobase complete: {I_rheobase:.2f} µA/cm²")

            # Update config
            self.config_manager.config.stim.Iext = I_rheobase
            self._refresh_all_forms()
            self._lock_ui(False)

        def on_error(error_msg):
            """Handle rheobase search error."""
            self._lbl_rheobase_result.setText(f"❌ Auto-Rheobase failed: {error_msg}")
            self._status("Auto-Rheobase Search failed")
            self._lock_ui(False)

        # Start async rheobase search
        self.sim_controller.run_rheobase(
            self.config_manager.config,
            on_progress=on_progress,
            on_success=on_success,
            on_error=on_error
        )

    def _format_rheobase_visualization(self, history, final_value, uncertainty):
        """Format binary search history as ASCII visualization."""
        if not history:
            return ""

        # ASCII tree visualization
        lines = ["Search Tree (I_ext in uA/cm2):"]
        lines.append("-" * 40)

        # Find bounds for scaling
        all_values = [h[2] for h in history]
        max_val = max(all_values) if all_values else 100.0

        # Build bar chart of search
        for i, (I_low, I_high, I_mid, has_spike) in enumerate(history[-10:]):  # Last 10 iterations
            # Scale to 30 chars
            bar_len = int(30 * I_mid / max_val) if max_val > 0 else 0
            bar = "#" * bar_len + "-" * (30 - bar_len)
            marker = "+" if has_spike else "-"
            lines.append(f"  {i+1:2d}: [{bar}] {I_mid:6.2f} {marker}")

        lines.append("-" * 40)
        lines.append(f"  + = spike detected  - = no spike")
        lines.append(f"  Final: {final_value:.2f} µA/cm²")

        return "\n".join(lines)

    def _toggle_sidebar(self):
        """v12.7: Toggle sidebar visibility with checkable toolbar button."""
        # Use button state as source of truth
        show = self.btn_toggle_sidebar.isChecked()
        self._sidebar_visible = show
        if show:
            self._sidebar_frame.show()
            self._dock_params.show()
        else:
            self._sidebar_frame.hide()
            self._dock_params.hide()
    
    def _show_window_size_menu(self):
        """Show layout preset menu for laptop-first responsive shell."""
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        for preset_name in ("Laptop", "Presentation", "Desktop", "Debug"):
            action = menu.addAction(f"{preset_name} Layout")
            action.triggered.connect(
                lambda _checked=False, name=preset_name: self._apply_layout_preset(name)
            )
        menu.addSeparator()
        action_fullscreen = menu.addAction("Fullscreen")
        action_fullscreen.triggered.connect(self._toggle_fullscreen)
        menu.exec(self.btn_window_size.mapToGlobal(self.btn_window_size.rect().bottomLeft()))

    def _set_window_size(self, width: int, height: int):
        """Set window size and apply the matching responsive layout preset."""
        self.resize(width, height)
        self._apply_layout_preset(preset_for_width(width).name, resize_window=False)
        self._status(f"Window resized to {width}x{height}")

    def closeEvent(self, event):
        """Save UI state on close."""
        try:
            if not self._is_headless_qt_platform():
                self.config_manager.save_config_as(".last_session.json")
                # Save dock geometry/state only for real desktop sessions. Headless
                # GUI tests must not poison the next interactive startup.
                with open(".dock_geometry.bin", "wb") as f:
                    f.write(self.saveGeometry())
                with open(".dock_state.bin", "wb") as f:
                    f.write(self.saveState())
                import json
                with open(".ui_state_meta.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "version": 1,
                            "layout_engine": "dock-shell-v1",
                            "qt_platform": QApplication.platformName(),
                        },
                        f,
                    )
        except Exception as e:
            logger.warning(f"Failed to save UI state on close: {e}")
        super().closeEvent(event)
    
    def _toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        if self.isFullScreen():
            self.showNormal()
            self._status("Windowed mode")
        else:
            self.showFullScreen()
            self._status("Fullscreen mode")

    def _reset_layout(self):
        """Reset dock layout to default state."""
        self._reset_dock_layout()
        self._status("Layout reset to default")

    def _on_preset_mode_changed(self, _field_name: str, _value):
        """Apply only the active mode overlay when user changes a mode selector."""
        if not self.config_manager.current_preset_name:
            return
        from core.presets import (
            _apply_k_mode,
            _apply_l5_mode,
            _apply_alzheimer_mode,
            _apply_hypoxia_mode,
            _apply_ach_mode,
            _apply_purkinje_mode,
            _apply_anesthesia_mode,
            _apply_dravet_mode,
        )

        # Apply only mode-specific overlay without wiping user customizations.
        preset_code = self.config_manager.current_preset_name.split(":", 1)[0].strip().upper()
        cfg = self.config_manager.config
        
        if preset_code == "K":
            _apply_k_mode(cfg)
        elif preset_code == "B":
            _apply_l5_mode(cfg)
        elif preset_code == "R":
            _apply_ach_mode(cfg)
        elif preset_code == "E":
            _apply_purkinje_mode(cfg)
        elif preset_code == "N":
            _apply_alzheimer_mode(cfg)
        elif preset_code == "O":
            _apply_hypoxia_mode(cfg)
        elif preset_code == "G":
            _apply_anesthesia_mode(cfg)
        elif preset_code == "S":
            _apply_dravet_mode(cfg)

        self._refresh_all_forms()
        self.oscilloscope.sync_delay_controls_for_config(self.config_manager.config)
        self.topology.draw_neuron(
            self.config_manager.config,
            delay_target_name=self._delay_target_name,
            delay_custom_index=self._delay_custom_index,
        )
        self._status(
            f"Preset mode updated: {self.config_manager.current_preset_name}{self.config_manager.active_mode_suffix()}"
        )

    def change_language(self, lang: str):
        T.set_language(lang)
        self.retranslate_ui()

    def retranslate_ui(self):
        self.setWindowTitle(T.tr('app_title'))
        self.btn_run.setText(T.tr('btn_run'))
        self.btn_stoch.setText(T.tr('btn_stoch'))
        self.btn_sweep.setText(T.tr('btn_sweep'))
        self.btn_fi.setText("f-I")
        self.btn_sd.setText(T.tr('btn_sd'))
        self.btn_excmap.setText(T.tr('btn_excmap'))
        self.btn_export.setText(T.tr('btn_export'))
        self.lbl_preset.setText(T.tr('preset_label'))
        self.lbl_lang.setText(T.tr('lbl_language'))
        self._status(T.tr('status_ready'))
        # Update guide text
        self.guide_browser.setHtml(repair_text(_GUIDE_HTML))
        if hasattr(self, 'dual_stim_widget'):
            self.dual_stim_widget.retranslate_ui()
        repair_widget_tree(self)

    # ─────────────────────────────────────────────────────────────────
    #  SIMULATION HELPERS
    # ─────────────────────────────────────────────────────────────────
    def _lock_ui(self, busy: bool):
        for btn in (self.btn_run, self.btn_stoch, self.btn_sweep, self.btn_fi,
                    self.btn_sd, self.btn_excmap):
            btn.setEnabled(not busy)
        self.btn_cancel.setVisible(busy)
        if not busy:
            self._cancel_requested = False

    def _request_cancel(self):
        """User clicked Cancel - set flag for running simulations to check."""
        self._cancel_requested = True
        self._status("Cancellation requested...")
        self.btn_cancel.setEnabled(False)

    def _on_sim_error(self, msg: str):
        if self._progress_dialog is not None:
            self._progress_dialog.close()
            self._progress_dialog = None
        self._lock_ui(False)
        QMessageBox.critical(self, "Simulation Error", msg)
        self._status("Error.")

    def _sync_hines_button_state(self):
        """Sync Hines button and Jacobian combo with current jacobian_mode in config."""
        if not hasattr(self, 'btn_hines'):
            return
        jac_mode = str(getattr(self.config_manager.config.stim, "jacobian_mode", "dense_fd"))
        is_hines = jac_mode == "native_hines"
        self.btn_hines.setChecked(is_hines)

        # Sync combo (non-Hines modes)
        if hasattr(self, 'combo_jacobian'):
            self.combo_jacobian.setEnabled(not is_hines)
            if not is_hines and jac_mode in ["dense_fd", "sparse_fd", "analytic_sparse"]:
                self.combo_jacobian.setCurrentText(jac_mode)

        if is_hines:
            self.btn_hines.setText("HINES ON")
            self.btn_hines.setStyleSheet("""
                QPushButton {
                    background: #00BCD4; color: #FFFFFF; border-radius: 6px;
                    border: 2px solid #00BCD4;
                    font-weight: bold;
                }
                QPushButton:hover { background: #00A97F; }
                QPushButton:disabled { color: #555568; }
            """)
        else:
            self.btn_hines.setText("HINES")
            self.btn_hines.setStyleSheet("""
                QPushButton {
                    background: #313244; color: #89DCEB; border-radius: 6px;
                    border: 1px solid #45475A;
                    font-weight: bold;
                }
                QPushButton:hover { background: #3E3F5E; }
                QPushButton:disabled { color: #555568; }
            """)

    def _on_hines_toggled(self, checked: bool):
        """Toggle native Hines solver mode."""
        if checked:
            self.config_manager.config.stim.jacobian_mode = "native_hines"
            self.btn_hines.setStyleSheet("""
                QPushButton {
                    background: #00BCD4; color: #FFFFFF; border-radius: 6px;
                    border: 2px solid #00BCD4;
                    font-weight: bold;
                }
                QPushButton:hover { background: #00A97F; }
                QPushButton:disabled { color: #555568; }
            """)
            self.btn_hines.setText("HINES ON")
            self._sb.showMessage("Hines solver ACTIVE (experimental). Dense-FD BDF suspended.", 4000)
        else:
            self.config_manager.config.stim.jacobian_mode = "dense_fd"
            self.btn_hines.setStyleSheet("""
                QPushButton {
                    background: #313244; color: #89DCEB; border-radius: 6px;
                    border: 1px solid #45475A;
                    font-weight: bold;
                }
                QPushButton:hover { background: #3E3F5E; }
                QPushButton:disabled { color: #555568; }
            """)
            self.btn_hines.setText("HINES")
            self._sb.showMessage("Hines solver OFF. Using dense_fd BDF.", 3000)
        # Show/hide jacobian combo based on Hines state
        self.combo_jacobian.setEnabled(not checked)
        self.combo_jacobian.setStyleSheet("" if not checked else "QComboBox { color: #555568; }")

    def _on_jacobian_changed(self, mode: str):
        """Handle Jacobian mode selection (when Hines is OFF)."""
        if mode and not self.btn_hines.isChecked():
            self.config_manager.config.stim.jacobian_mode = mode
            self._sb.showMessage(f"Jacobian mode: {mode}", 3000)

    def _preflight_validate(self) -> bool:
        """
        Validate configuration before launching long simulation runs.

        Hard errors stop execution, non-fatal warnings are surfaced in status bar.
        """
        try:
            warnings = validate_simulation_config(self.config_manager.config)
            warnings.extend(
                build_preset_mode_warnings(self.config_manager.config, self.config_manager.current_preset_name)
            )
        except SimulationParameterError as exc:
            QMessageBox.critical(self, "Parameter Validation Error", str(exc))
            self._status("Validation error.")
            return False

        if warnings:
            preview = " | ".join(warnings[:2])
            if len(warnings) > 2:
                preview += " | ..."
            self._status(f"Warnings: {preview}")
        return True

    def _report_progress(self, i: int, n: int, val):
        """Update status bar with progress from long operations."""
        pct = int(100 * i / max(1, n))
        self._status(f"Progress: {pct}% ({i}/{n}) — Value: {val:.3g}")
        QApplication.processEvents()

    # ─────────────────────────────────────────────────────────────────
    #  1. STANDARD RUN
    # ─────────────────────────────────────────────────────────────────
    def run_simulation(self):
        self._lock_ui(True)
        self._status(T.tr('status_computing'))
        QApplication.processEvents()

        if not self._preflight_validate():
            self._lock_ui(False)
            return

        # Sync dual stim config from widget to main config.
        self._sync_dual_stim_into_config()

        # Capture config snapshot and flags for the simulation thread
        run_mc = self.config_manager.config.analysis.run_mc
        run_bif = self.config_manager.config.analysis.run_bifurcation
        bif_param = self.config_manager.config.analysis.bif_param
        mc_trials = self.config_manager.config.analysis.mc_trials

        if run_mc:
            self._status(f"Monte-Carlo ({mc_trials} trials)...")
        QApplication.processEvents()

        # Run simulation through SimulationController
        if run_mc:
            self.sim_controller.run_monte_carlo(
                self.config_manager.config,
                mc_trials,
                on_success=self._on_simulation_done,
                on_error=self._on_sim_error
            )
        else:
            self.sim_controller.run_single(
                self.config_manager.config,
                on_success=self._on_simulation_done,
                on_error=self._on_sim_error
            )

    def _on_compartment_selected(self, comp_idx: int):
        """Handle compartment selection from topology click."""
        cfg = self.config_manager.config
        cfg.stim.stim_comp = comp_idx
        self._status(f"Stimulation compartment set to idx {comp_idx}")
        # Update only the stim_comp field to avoid breaking focus on other widgets
        self._set_stim_form_value('stim_comp', comp_idx)
        # Sync oscilloscope delay target to clicked compartment
        try:
            self.oscilloscope._combo_delay_target.setCurrentText("Custom Compartment")
            self.oscilloscope._spin_delay_comp.setValue(comp_idx)
        except RuntimeError:
            pass  # Qt C++ object was deleted
        self._delay_target_name = "Custom Compartment"
        self._delay_custom_index = comp_idx
        # Redraw topology to move PRI marker
        dual_cfg = self.dual_stim_widget.config if self.dual_stim_widget.config.enabled else None
        self.topology.draw_neuron(
            cfg,
            dual_config=dual_cfg,
            delay_target_name=self._delay_target_name,
            delay_custom_index=self._delay_custom_index,
        )

    def _on_run_button_clicked(self):
        """Handle Run button click — reset live flag and start simulation."""
        self._is_live_run = False  # Ensure this is a manual run, not live timer
        self.sim_controller.run_single(
            self.config_manager.config,
            on_success=self._on_simulation_done,
            on_error=self._on_sim_error,
        )

    def _run_with_lle_enabled(self):
        """Run simulation with LLE computation enabled (v12.2).

        Triggered by "Compute LLE" button click. Performs full rerun
        with calc_lle=True for accurate Lyapunov exponent calculation.
        """
        self._is_live_run = False
        self._status("Running simulation with LLE computation...")
        cfg = self.config_manager.config
        # Enable LLE in config (stored in stim section, not analysis)
        cfg.stim.calc_lle = True
        subspace_text = "Voltage Only"
        if hasattr(self, "analytics") and hasattr(self.analytics, "_chaos_subspace_combo"):
            subspace_text = self.analytics._chaos_subspace_combo.currentText()
        lle_subspace_mode = {
            "Voltage Only": 0,
            "Voltage + Gates": 1,
            "Full State": 2,
        }.get(subspace_text, 0)
        self.sim_controller.run_single(
            cfg,
            on_success=self._on_simulation_done,
            on_error=self._on_sim_error,
            compute_lyapunov=True,
            lle_subspace_mode=lle_subspace_mode,
        )

    def _on_simulation_done(self, result: dict):
        """
        Handle completed simulation payloads and update the UI accordingly.
        
        Processes the provided result dictionary which may contain:
        - 'single': a single-run result object — stores it as the last result, updates plots, analytics, topology, axon biophysics, sparkline, enables export actions, and updates the status with compartment count and ATP estimate. If the result indicates divergence (`res.diverged`), updates plots/analytics, sets a warning status, and aborts the normal completion flow.
        - 'mc_results': a list of Monte Carlo trial results — updates oscilloscope Monte Carlo view and status with trial count.
        - 'bif': bifurcation data (paired with 'bif_param') — updates the analytics bifurcation view.
        
        The method always selects the oscilloscope tab, shows a critical message box on unexpected errors, and ensures the UI is unlocked at the end.
        
        Parameters:
            result (dict): Simulation outcome payload containing one or more of the keys described above.
        """
        try:
            result = self._normalize_worker_payload(result)
            if 'mc_results' in result:
                self.oscilloscope.update_plots_mc(result['mc_results'])
                self._status(f"MC done - {len(result['mc_results'])} trials.")
            elif 'single' in result:
                res = result['single']
                self._last_result = res
                # Check for divergence (v12.0)
                if getattr(res, 'diverged', False):
                    self._status("Simulation diverged: check for non-physical parameters (e.g. zero resistance or infinite conductance)")
                    # Still update plots with partial result
                    self.oscilloscope.update_plots(res)
                    self.analytics.update_analytics(res)
                    self._lock_ui(False)
                    return
                # v13.0: Immediate oscilloscope update (fast, 60FPS capable)
                self.oscilloscope.update_plots(res)

                # v13.0: Debounced analytics — restart timer if already running
                # Increment generation to invalidate stale analytics
                stats = result.get('stats')
                if stats is not None:
                    self._analytics_generation += 1
                    self._pending_result_for_analytics = None
                    self._pending_morph_for_analytics = None
                    self._analytics_debounce_timer.stop()
                    self.analytics.update_analytics(res, stats)
                else:
                    self._analytics_generation += 1
                    self._pending_analytics_generation = self._analytics_generation
                    self._pending_result_for_analytics = res
                    self._pending_morph_for_analytics = result.get('morph')  # may be None from some paths
                    self.analytics.mark_analysis_pending(res)
                    self._analytics_debounce_timer.stop()
                    # v12.7: Only start analytics debounce if user has finished moving slider
                    # (live timer is NOT active). This prevents analytics choke during rapid changes.
                    if not self._live_timer.isActive():
                        self._analytics_debounce_timer.start(300)  # 300ms debounce

                dual_cfg = self.dual_stim_widget.config if self.dual_stim_widget.config.enabled else None
                self.topology.draw_neuron(
                    self.config_manager.config,
                    dual_config=dual_cfg,
                    delay_target_name=self._delay_target_name,
                    delay_custom_index=self._delay_custom_index,
                    result=res,
                )
                self.axon_biophysics.plot_axon_data(res, self.config_manager.config)
                self._update_sparkline(res)
                self.btn_export_plot.setEnabled(True)
                self.btn_export.setEnabled(True)
                self._status(
                    f"Done — {res.n_comp} compartments, "
                    f"ATP ≈ {res.atp_estimate:.3e} nmol/cm²"
                )

            if 'bif' in result:
                self.analytics.update_bifurcation(result['bif'], result['bif_param'])

            # Only switch to oscilloscope on manual runs, not live timer updates
            if not self._is_live_run:
                self._dock_analytics.raise_()
                self.oscilloscope.raise_()
        except Exception as e:
            QMessageBox.critical(self, "Simulation Error", str(e))
            self._status("Error — check parameters.")
        finally:
            if self._progress_dialog is not None:
                self._progress_dialog.close()
                self._progress_dialog = None
            self._lock_ui(False)
            self._is_live_run = False  # Reset flag after any simulation completion

    def _on_analytics_debounce_fired(self):
        """v13.0: Start async analytics after debounce period."""
        if self._pending_result_for_analytics is None:
            return

        res = self._pending_result_for_analytics
        # Capture generation for this analytics run
        current_generation = self._pending_analytics_generation

        # Build morphology if not already available
        morph = self._pending_morph_for_analytics
        if morph is None:
            from core.morphology import MorphologyBuilder
            morph = MorphologyBuilder.build(self.config_manager.config)

        # Run analytics in background thread with generation tracking
        def on_analytics_done_wrapped(stats):
            """Wrapper to check generation before updating UI."""
            # Only update if this is still the current generation
            if current_generation == self._analytics_generation:
                self._on_analytics_done(stats)
            else:
                logger.debug(f"Stale analytics result dropped (gen {current_generation} != {self._analytics_generation})")

        self.sim_controller.run_analytics(
            res, morph, self.config_manager.config,
            on_success=on_analytics_done_wrapped,
            on_error=lambda e: logger.error(f"Analytics error: {e}")
        )

    def _on_analytics_done(self, stats):
        """v13.0: Update analytics widget when async analytics completes."""
        if self._pending_result_for_analytics is None:
            return

        res = self._pending_result_for_analytics
        # Clear pending state to prevent double-processing
        self._pending_result_for_analytics = None
        self._pending_morph_for_analytics = None
        # Note: generation is NOT reset here - it's used to detect stale results

        # Update analytics with result AND precomputed stats
        self.analytics.update_analytics(res, stats)

    def _normalize_worker_payload(self, payload):
        """Normalize legacy worker callbacks to the dict payload contract."""
        if isinstance(payload, dict):
            return payload
        if payload is None:
            return {}
        return {'single': payload}

    def _on_notes_changed(self):
        """
        Synchronize the session notes editor content into the application's configuration.
        
        Sets self.config_manager.config.notes to the current text of the session_notes QTextEdit.
        """
        self.config_manager.config.notes = self.session_notes.toPlainText()

    def _on_dual_stim_config_changed(self):
        """Handle dual stimulation configuration changes."""
        if not hasattr(self, 'config_manager'):
            return
        self._sync_dual_stim_into_config()
        if hasattr(self, "form_stim"):
            self.form_stim.refresh()
            self._sync_stim_type_controls()
            self._apply_form_ux_filters()
        self._refresh_topology_preview()
        self._update_stim_preview()
        if self.dual_stim_widget.config.enabled:
            self._status(
                f"Dual stim enabled: Secondary at {self.dual_stim_widget.config.secondary_location}"
            )
        else:
            self._status("Dual stimulation disabled")

    # ─────────────────────────────────────────────────────────────────
    #  2. STOCHASTIC
    # ─────────────────────────────────────────────────────────────────
    def run_stochastic(self):
        self._lock_ui(True)
        self._status("Stochastic simulation...")
        QApplication.processEvents()

        if not self._preflight_validate():
            self._lock_ui(False)
            return

        self._sync_dual_stim_into_config()

        # Run stochastic simulation through SimulationController
        self.sim_controller.run_stochastic(
            self.config_manager.config,
            1,
            on_success=self._on_stoch_done,
            on_error=self._on_sim_error
        )

    def _on_stoch_done(self, payload):
        self._on_simulation_done(self._normalize_worker_payload(payload))
        self._status("Stochastic run complete.")
        self.oscilloscope.raise_()

    # ─────────────────────────────────────────────────────────────────
    #  3. SWEEP
    # ─────────────────────────────────────────────────────────────────
    def run_sweep(self):
        ana = self.config_manager.config.analysis
        ana.sweep_param = self._canonical_sweep_param(getattr(ana, 'sweep_param', 'stim.Iext'))
        self._refresh_sweep_param_ui()
        if not hasattr(ana, 'sweep_param') or not ana.sweep_param:
            QMessageBox.warning(self, "Sweep",
                                "Set sweep_param in the Analysis section first.")
            return
        if not self._preflight_validate():
            return
        self._sync_dual_stim_into_config()

        import numpy as np
        param_vals = np.linspace(ana.sweep_min, ana.sweep_max, ana.sweep_steps)

        self._lock_ui(True)
        self._status(f"Sweep: {ana.sweep_param}  [{ana.sweep_min}...{ana.sweep_max}]  "
                     f"{ana.sweep_steps} steps...")
        QApplication.processEvents()

        # Run sweep through SimulationController
        self.sim_controller.run_sweep(
            self.config_manager.config,
            ana.sweep_param,
            param_vals,
            on_success=lambda res: self._on_sweep_done(res, ana.sweep_param),
            on_error=self._on_sim_error
        )

    def run_fi_curve(self):
        ana = self.config_manager.config.analysis
        ana.sweep_param = "stim.Iext"
        self._refresh_sweep_param_ui()
        if not np.isfinite(float(ana.sweep_min)) or float(ana.sweep_min) < 0.0:
            ana.sweep_min = 0.0
        if not np.isfinite(float(ana.sweep_max)) or float(ana.sweep_max) <= float(ana.sweep_min):
            ana.sweep_max = max(25.0, float(ana.sweep_min) + 20.0)
        if int(ana.sweep_steps) < 8:
            ana.sweep_steps = 16
        self._status(
            f"f-I sweep armed: Iext [{ana.sweep_min}...{ana.sweep_max}] over {ana.sweep_steps} steps."
        )
        self.run_sweep()

    def _on_sweep_done(self, results, param_name):
        self.analytics.update_sweep(results, param_name)
        self._lock_ui(False)
        self._status(f"Sweep complete — {len(results)} steps.")
        self._dock_analytics.raise_()

    # ─────────────────────────────────────────────────────────────────
    #  4. S-D CURVE
    # ─────────────────────────────────────────────────────────────────
    def run_sd_curve(self):
        if not self._preflight_validate():
            return
        dual_enabled = self._sync_dual_stim_into_config()
        cfg_for_sd = self.config_manager.config
        if dual_enabled:
            cfg_for_sd = copy.deepcopy(self.config_manager.config)
            cfg_for_sd.dual_stimulation = None

        self._lock_ui(True)
        if dual_enabled:
            self._status("Computing Strength-Duration curve (dual-stim disabled for S-D analysis)...")
        else:
            self._status("Computing Strength-Duration curve (binary search)...")
        QApplication.processEvents()
        # Run S-D curve through SimulationController
        self.sim_controller.run_sd_curve(
            cfg_for_sd,
            on_success=self._on_sd_done,
            on_error=self._on_sim_error,
            on_progress=self._report_progress
        )

    def _on_sd_done(self, sd):
        rh = sd['rheobase']
        tc = sd['chronaxie']
        self.analytics.update_sd_curve(sd)
        self._lock_ui(False)
        self._status(
            f"S-D done — Rheobase={rh:.2f} µA/cm²  "
            f"Chronaxie={'—' if tc != tc else f'{tc:.2f} ms'}"
        )
        self._dock_analytics.raise_()

    # ─────────────────────────────────────────────────────────────────
    #  5. EXCITABILITY MAP
    # ─────────────────────────────────────────────────────────────────
    def run_excmap(self):
        if not self._preflight_validate():
            return
        dual_enabled = self._sync_dual_stim_into_config()
        cfg_for_excmap = self.config_manager.config
        if dual_enabled:
            cfg_for_excmap = copy.deepcopy(self.config_manager.config)
            cfg_for_excmap.dual_stimulation = None

        ana = self.config_manager.config.analysis
        total = ana.excmap_NI * ana.excmap_ND
        self._lock_ui(True)
        if dual_enabled:
            self._status(
                f"Excitability map {ana.excmap_NI}x{ana.excmap_ND} = {total} runs "
                "(dual-stim disabled for map analysis)..."
            )
        else:
            self._status(f"Excitability map {ana.excmap_NI}x{ana.excmap_ND} = {total} runs...")
        QApplication.processEvents()
        # Run excitability map through SimulationController
        self.sim_controller.run_excmap(
            cfg_for_excmap,
            on_success=self._on_excmap_done,
            on_error=self._on_sim_error,
            on_progress=self._report_progress
        )

    def _on_excmap_done(self, exc):
        self.analytics.update_excmap(exc)
        self._lock_ui(False)
        self._status("Excitability map complete.")
        self._dock_analytics.raise_()

    # ─────────────────────────────────────────────────────────────────
    #  EXPORT CSV
    # ─────────────────────────────────────────────────────────────────
    def export_plot(self):
        if not hasattr(self, '_last_result'):
            QMessageBox.information(self, "Export Plot", "Run simulation first to export a plot.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Plot",
            "neuro_plot.png",
            "Image Files (*.png *.jpg *.jpeg *.bmp);;Vector Files (*.svg *.pdf);;All Files (*)"
        )
        if not path:
            return

        ok, err = self.oscilloscope.export_plot(path)
        if not ok:
            QMessageBox.critical(self, "Export Plot Error", err)
            return

        self._status(f"Plot exported: {path}")
        QMessageBox.information(self, "Export Plot", f"Saved to:\n{path}")

    def export_csv(self):
        """
        Export the most recent simulation result to a CSV file selected by the user.
        
        Opens a file-save dialog and, if a path is chosen, writes a CSV with per-time-row data from self._last_result. Columns produced:
        - t_ms, V_soma_mV
        - If the result has multiple compartments: V_AIS_mV, V_terminal_mV
        - Per-current columns: for single-compartment results `I_{name}_uA_cm2`; for multi-compartment results `I_{name}_Soma_uA_cm2`, `I_{name}_AIS_uA_cm2`, `I_{name}_Terminal_uA_cm2`
        - Calcium concentration column(s): `Ca_i_mM` or `Ca_i_Soma_mM`, `Ca_i_AIS_mM`, `Ca_i_Terminal_mM` if present
        - Gate traces: `gate_{name}` for each extracted gate trace
        
        If no last result is available (no attribute `_last_result`), the function returns without prompting. On successful export, updates the status via self._status and shows an information dialog. Exceptions during export are caught and reported with a critical message box.
        """
        if not hasattr(self, '_last_result'):
            return
        res = self._last_result

        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "neuro_result.csv",
            "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return

        try:
            import csv as _csv
            with open(path, 'w', newline='') as f:
                writer = _csv.writer(f)
                # Header
                header = ['t_ms', 'V_soma_mV']
                if res.n_comp > 1:
                    header += ['V_AIS_mV', 'V_terminal_mV']
                
                # Spatial current headers
                for k in res.currents:
                    if res.n_comp > 1:
                        header += [f'I_{k}_Soma_uA_cm2', f'I_{k}_AIS_uA_cm2', f'I_{k}_Terminal_uA_cm2']
                    else:
                        header.append(f'I_{k}_uA_cm2')
                
                if res.ca_i is not None:
                    if res.n_comp > 1:
                        header += ['Ca_i_Soma_mM', 'Ca_i_AIS_mM', 'Ca_i_Terminal_mM']
                    else:
                        header.append('Ca_i_mM')

                # Gate names
                from core.analysis import extract_gate_traces, extract_spatial_traces
                gates = extract_gate_traces(res)
                header += [f'gate_{k}' for k in gates]
                writer.writerow(header)

                for i, t in enumerate(res.t):
                    row = [f"{t:.4f}", f"{res.v_soma[i]:.4f}"]
                    if res.n_comp > 1:
                        _, v_ais, v_terminal = extract_spatial_traces(res.v_all, res.n_comp)
                        row.append(f"{v_ais[i]:.4f}")
                        row.append(f"{v_terminal[i]:.4f}")
                    
                    # Spatial current data (safe for 1D, 2D and single-comp traces)
                    for curr in res.currents.values():
                        c_arr = np.asarray(curr, dtype=float)
                        if res.n_comp > 1:
                            if c_arr.ndim == 2:
                                ais_idx = 1 if c_arr.shape[0] > 1 else 0
                                term_idx = c_arr.shape[0] - 1
                                row.append(f"{c_arr[0, i]:.6f}")
                                row.append(f"{c_arr[ais_idx, i]:.6f}")
                                row.append(f"{c_arr[term_idx, i]:.6f}")
                            else:
                                # Fallback for malformed 1D current trace
                                flat_val = c_arr.reshape(-1)[i]
                                row.append(f"{flat_val:.6f}")
                                row.append(f"{flat_val:.6f}")
                                row.append(f"{flat_val:.6f}")
                        else:
                            # Single-comp output
                            if c_arr.ndim == 2:
                                flat_val = c_arr[0, i]
                            else:
                                flat_val = c_arr.reshape(-1)[i]
                            row.append(f"{flat_val:.6f}")
                    
                    if res.ca_i is not None:
                        if res.n_comp > 1:
                            ca_soma, ca_ais, ca_terminal = extract_spatial_traces(res.ca_i, res.n_comp)
                            row.append(f"{ca_soma[i]:.8f}")
                            row.append(f"{ca_ais[i]:.8f}")
                            row.append(f"{ca_terminal[i]:.8f}")
                        else:
                            row.append(f"{np.asarray(res.ca_i, dtype=float).reshape(-1)[i]:.8f}")
                    
                    for gv in gates.values():
                        row.append(f"{gv[i]:.6f}")
                    writer.writerow(row)

            self._status(f"Exported: {path}")
            QMessageBox.information(self, "Export", f"Saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    # ─────────────────────────────────────────────────────────────────
    #  EXPORT NEUROML (Stage 6.4)
    # ─────────────────────────────────────────────────────────────────
    def export_neuroml(self):
        """Export current FullModelConfig to NeuroML 2.2 XML."""
        from core.neuroml_export import export_neuroml as _export
        path, _ = QFileDialog.getSaveFileName(
            self, "Export NeuroML", "neuron_model.nml",
            "NeuroML 2 Files (*.nml *.xml);;All Files (*)"
        )
        if not path:
            return
        try:
            _export(self.config_manager.config, path=path)
            self._status(f"NeuroML exported: {path}")
            QMessageBox.information(self, "NeuroML Export",
                                    f"Model exported to NeuroML 2.2:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "NeuroML Export Error", str(e))

    def save_config_as(self):
        """Save current configuration to JSON file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration", "neuro_config.json",
            "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        try:
            self.config_manager.save_config_as(path)
            self._status(f"Configuration saved to {path}")
            QMessageBox.information(self, "Save Configuration", 
                                    f"Configuration saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def load_config_from(self):
        """Load configuration from JSON file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", "",
            "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        if self.config_manager.load_config_from(path):
            self.config = self.config_manager.config
            self._sync_preset_combo_text(self.config_manager.current_preset_name or "Custom Config")
            self._refresh_all_forms()
            self._status(f"Configuration loaded from {path}")
            QMessageBox.information(self, "Load Configuration", 
                                    f"Configuration loaded from:\n{path}")

    def _refresh_all_forms(self):
        """Refresh all form widgets to match current config."""
        if hasattr(self, 'form_stim'):
            self.form_stim.refresh()
        if hasattr(self, 'form_stim_loc'):
            self.form_stim_loc.refresh()
        if hasattr(self, 'form_dfilter'):
            self.form_dfilter.refresh()
        if hasattr(self, 'form_chan'):
            self.form_chan.refresh()
        if hasattr(self, 'form_morph'):
            self.form_morph.refresh()
        if hasattr(self, 'form_calcium'):
            self.form_calcium.refresh()
        if hasattr(self, 'form_metabolism'):
            self.form_metabolism.refresh()
        if hasattr(self, 'form_env'):
            self.form_env.refresh()
        if hasattr(self, 'form_ana'):
            self.form_ana.refresh()
        if hasattr(self, 'form_preset_modes'):
            self.form_preset_modes.refresh()
        self._refresh_sweep_param_ui()
        self._update_attenuation_hint()
        self._sync_stim_type_controls()
        self._sync_preset_mode_controls()
        self._sync_live_deck_to_config()
        self._apply_form_ux_filters()

    def _set_stim_form_value(self, field_name: str, value):
        """Update a single field in the stimulation form without refreshing others.
        
        Use this for targeted updates to avoid breaking focus/grab on other widgets.
        Specifically addresses the 'topology click breaks slider' issue.
        
        Parameters:
            field_name: Name of the field to update (e.g., 'stim_comp')
            value: New value for the field
        """
        if not hasattr(self, 'form_stim') or self.form_stim is None:
            return
        
        try:
            # Check if the form has the field
            if hasattr(self.form_stim, field_name):
                # Get the widget for this field if available
                field_widget = getattr(self.form_stim, field_name, None)
                if field_widget is not None and hasattr(field_widget, 'setValue'):
                    # Use blockSignals to prevent triggering other updates
                    was_blocked = field_widget.signalsBlocked()
                    field_widget.blockSignals(True)
                    field_widget.setValue(value)
                    field_widget.blockSignals(was_blocked)
                else:
                    # Fallback: set attribute directly on the config object
                    # The form will pick it up on next full refresh
                    setattr(self.config_manager.config.stim, field_name, value)
            else:
                # Field doesn't exist in form, set on config directly
                setattr(self.config_manager.config.stim, field_name, value)
        except Exception:
            # Silently fail — this is a UI optimization, not critical functionality
            pass

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for tab navigation and actions."""
        key = event.key()
        modifiers = event.modifiers()
        
        # Dock focus shortcuts: Ctrl+1..7
        if modifiers == Qt.ControlModifier:
            if key == Qt.Key_1:
                if hasattr(self, '_dock_params'): self._dock_params.raise_()  # Parameters
            elif key == Qt.Key_2:
                self.oscilloscope.raise_()  # Oscilloscope (central)
            elif key == Qt.Key_3:
                if hasattr(self, '_dock_analytics'): self._dock_analytics.raise_()  # Analytics
            elif key == Qt.Key_4:
                if hasattr(self, '_dock_topology'): self._dock_topology.raise_()  # Topology
            elif key == Qt.Key_5:
                if hasattr(self, '_dock_stim'): self._dock_stim.raise_()  # Stimulation
            elif key == Qt.Key_6:
                if hasattr(self, '_dock_live'): self._dock_live.raise_()  # Live Controls
            elif key == Qt.Key_7:
                if hasattr(self, '_dock_guide'): self._dock_guide.raise_()  # Guide
            elif key == Qt.Key_R:
                self._on_run_button_clicked()  # Ctrl+R = Run
            elif key == Qt.Key_S:
                self.export_csv()  # Ctrl+S = Export CSV
            elif key == Qt.Key_Q:
                self.close()  # Ctrl+Q = Quit
            else:
                super().keyPressEvent(event)
        # Function keys (no modifiers)
        elif modifiers == Qt.NoModifier:
            if key == Qt.Key_F5:
                self._on_run_button_clicked()  # F5 = Run
            elif key == Qt.Key_F11:
                self._toggle_fullscreen()  # F11 = Fullscreen
            elif key == Qt.Key_F1:
                if hasattr(self, '_dock_guide'): self._dock_guide.raise_()  # F1 = Guide
            else:
                super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)
    
    def _toggle_fullscreen(self):
        """Toggle window fullscreen state."""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()


# ─────────────────────────────────────────────────────────────────────
#  GUIDE HTML
# ─────────────────────────────────────────────────────────────────────
_GUIDE_HTML = """
<html><body style="background:#0D1117; color:#C9D1D9; font-family:Segoe UI,sans-serif; padding:20px;">

<h1 style="color:#89B4FA;">Hodgkin-Huxley Neuron Simulator v11.3</h1>
<p>A research-grade biophysical simulator based on the Hodgkin-Huxley (1952) formalism,
extended with multi-compartment morphology, optional ion channels, and advanced analysis tools.</p>

<h2 style="color:#A6E3A1;">Quick Start</h2>
<ol>
  <li>Select a <b>Preset</b> from the dropdown (e.g. <i>Squid Giant Axon</i>).</li>
  <li>Adjust parameters in <b>1) Setup</b> (Run Setup / Model Biophysics / Advanced Analysis Tools).</li>
  <li>If needed, enable and configure secondary stimulation in <b>2) Dual Stim</b>.
      Dual Stim adds an optional secondary stimulus source.</li>
  <li>Click <b>RUN SIMULATION</b> - results appear in Oscilloscope and Analytics.</li>
  <li>Inspect traces in <b>3) Oscilloscope</b> and metrics in <b>4) Analytics</b>.</li>
</ol>
<p style="color:#BAC2DE;">
In <b>Oscilloscope -> View</b>, conduction delay can be measured from soma to
<i>Terminal</i>, <i>AIS</i>, <i>Trunk Junction</i>, or a custom compartment index.
</p>

<h2 style="color:#A6E3A1;">Run Modes</h2>
<table style="border-collapse:collapse; width:100%;">
<tr style="background:#1E3A5F;">
  <th style="padding:6px; text-align:left;">Button</th>
  <th style="padding:6px; text-align:left;">What it does</th>
</tr>
<tr><td style="padding:4px;"><b>▶ RUN</b></td>
    <td>Standard deterministic simulation (BDF stiff ODE solver)</td></tr>
<tr style="background:#1A1A2E;"><td style="padding:4px;"><b>Jacobian mode</b></td>
    <td>Set <i>stim.jacobian_mode</i> in Parameters -> Stimulation:
        <i>dense_fd</i>, <i>sparse_fd</i>, or <i>analytic_sparse</i>.
        Heavy presets typically run faster with sparse modes.</td></tr>
<tr style="background:#1A1A2E;"><td style="padding:4px;"><b>STOCHASTIC</b></td>
    <td>Native Hines stochastic mode with Langevin gate noise (Fox &amp; Lu 1994).
        Enable via <i>stoch_gating</i> flag or use <i>noise_sigma</i>.</td></tr>
<tr><td style="padding:4px;"><b>SWEEP</b></td>
    <td>Parametric sweep. Set <i>sweep_param</i>, <i>sweep_min/max/steps</i>
        in Analysis section. Produces f-I curves and voltage traces.</td></tr>
<tr style="background:#1A1A2E;"><td style="padding:4px;"><b>S-D</b></td>
    <td>Strength-Duration curve. Binary search for threshold at 13 durations.
        Reports Rheobase (I at infinite duration) and Chronaxie (t at 2 x I_rh).</td></tr>
<tr><td style="padding:4px;"><b>EXCIT. MAP</b></td>
    <td>2-D excitability map: spike count as function of (I_ext x pulse_dur).
        Set <i>excmap_*</i> parameters in Analysis section.</td></tr>
</table>

<h2 style="color:#A6E3A1;">Ion Channels</h2>
<ul>
  <li><b>Na / K / Leak</b> - classic Hodgkin-Huxley (1952) channels, always active.</li>
  <li><b>Ih</b> - HCN pacemaker current (Destexhe 1993). Causes rhythmic firing.</li>
  <li><b>ICa</b> - L-type calcium (Huguenard 1992). Enables plateau potentials.</li>
  <li><b>IA</b> - A-current, transient K+ (Connor-Stevens 1971). Delays first spike.</li>
  <li><b>SK</b> - Ca2+-activated K+ (NEW). Causes spike-frequency adaptation.</li>
</ul>

<h2 style="color:#A6E3A1;">Neuron Passport (Analytics -> Passport)</h2>
<p>After each simulation, the Passport tab shows:</p>
<ul>
  <li>Passive: tau_m, R_in, lambda (space constant)</li>
  <li>Spike: threshold, peak, AHP, halfwidth, dV/dt rate</li>
  <li>Firing: f_initial, f_steady, Adaptation Index, cell type classification (FS/RS/IB/LTS)</li>
  <li>Conduction velocity (multi-compartment mode)</li>
  <li>Energy: cumulative charge Q per channel, ATP estimate</li>
</ul>
<p style="color:#BAC2DE;">
Spike detector settings are configurable in <b>Parameters -> Analysis</b>:
algorithm, threshold, prominence, baseline, refractory window and repolarization window.
</p>

<h2 style="color:#A6E3A1;">Phase Plane</h2>
<p>Shows the AP trajectory in V-n space plus nullclines (dV/dt=0 and dn/dt=0).
Fixed points are where both nullclines intersect. Limit cycles = sustained firing.</p>

<h2 style="color:#A6E3A1;">Morphology</h2>
<p>Multi-compartment cable model: Soma -> AIS (high gNa density) -> Trunk -> Bifurcation -> Branch 1/2.
The Laplacian matrix couples adjacent compartments via axial conductance g_ax = d/(4 * Ra * dx).</p>

<h2 style="color:#A6E3A1;">Preset Modes (K / N / O)</h2>
<p>Use <b>Parameters -> Preset Modes</b> to switch validated stage/mode overlays:</p>
<ul>
  <li><b>K (Thalamic Relay)</b>: <i>baseline</i> (lower throughput) vs <i>activated</i> (higher relay output).</li>
  <li><b>N (Alzheimer's)</b>: <i>progressive</i> usually shows early spikes with attenuation; <i>terminal</i> is markedly less excitable.</li>
  <li><b>O (Hypoxia)</b>: <i>progressive</i> shows short early spiking then failure/attenuation; <i>terminal</i> approximates severe failure state.</li>
</ul>
<p style="color:#BAC2DE;">
Default profile uses <b>K baseline</b>; switch to <b>activated</b> for high-throughput relay behavior.
</p>
<p style="color:#BAC2DE;">
Interpret terminal modes as late-pathology behavior for analysis/education, not as healthy baseline physiology.
</p>
<p style="color:#BAC2DE;">
<b>F (Multiple Sclerosis)</b> is currently modeled as a single-stage preset (no progressive/terminal switch).
</p>
<p style="color:#BAC2DE;">
When <b>Dual Stim</b> is enabled, an additional secondary stimulus can be configured in the Dual tab.
</p>

<h2 style="color:#A6E3A1;">💾 Export</h2>
<p>After a run, click <b>💾 Export CSV</b> to save all traces (V, currents, gates, Ca) as a CSV file
compatible with Excel, MATLAB, Python/pandas, etc. Use <b>Export Plot</b> to save the current
oscilloscope view as PNG, SVG, or PDF.</p>

<hr style="border-color:#45475A;">
<p style="color:#585B70; font-size:11px;">
HH Simulator v11.3 — Python/PySide6 port of Scilab HH v9.0 |
Numba JIT kinetics | scipy.sparse Laplacian | BDF + Native Hines solvers
</p>
</body></html>
"""
