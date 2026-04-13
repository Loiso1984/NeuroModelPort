"""
gui/main_window.py — Main Application Window v10.0

Tabs: Parameters | Oscilloscope | Analytics | Topology | Guide
Run modes: Standard | Monte-Carlo | Sweep | S-D Curve | Excit. Map | Stochastic
"""
import csv
import os
from pathlib import Path
import numpy as np

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPushButton, QLabel, QComboBox, QStatusBar,
    QScrollArea, QMessageBox, QApplication, QFileDialog,
    QGroupBox, QToolBar, QProgressDialog, QLayout, QSizePolicy,
    QFrame, QSplitter, QSlider, QTextEdit,
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
        self.resize(1400, 900)
        self.setStyleSheet(self._STYLE)
        
        # Initialize services
        self.config_manager = ConfigManager()
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
        self._live_timer = QTimer(self)
        self._live_timer.setSingleShot(True)
        self._live_timer.setInterval(350)
        self._live_timer.timeout.connect(self._on_live_timer_fired)

        central = QWidget()
        self.setCentralWidget(central)
        self._main_layout = QVBoxLayout(central)
        self._main_layout.setContentsMargins(4, 4, 4, 4)
        self._main_layout.setSpacing(4)
        self._main_layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        central.setMinimumSize(0, 0)
        self.setMinimumSize(1000, 700)

        self._setup_top_bar()
        self._setup_tabs()
        self._setup_status_bar()
        self._wire_service_signals()
        self.retranslate_ui()
        
        # Try to restore last session, otherwise load default preset
        import os
        if os.path.exists(".last_session.json"):
            if self.config_manager.load_config_from(".last_session.json"):
                self._refresh_all_forms()
                self._sync_hines_button_state()
                self.oscilloscope.sync_delay_controls_for_config(self.config_manager.config)
                self.topology.draw_neuron(self.config_manager.config)
                self._status("Restored previous session.")
            else:
                self.load_preset("A: Squid Giant Axon (HH 1952)")
        else:
            self.load_preset("A: Squid Giant Axon (HH 1952)")
        
        # Load UI state if available
        if os.path.exists(".ui_state.json"):
            try:
                import json
                with open(".ui_state.json", "r") as f:
                    ui_state = json.load(f)
                # Restore splitter sizes
                if "splitter_sizes" in ui_state:
                    self._main_splitter.setSizes(ui_state["splitter_sizes"])
                # Restore live deck combo values
                if "live_deck" in ui_state and len(ui_state["live_deck"]) == len(self._live_combos):
                    for i, combo_text in enumerate(ui_state["live_deck"]):
                        self._live_combos[i].setCurrentText(combo_text)
            except Exception:
                pass  # Silently fail on UI state load errors

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
        self._status(f"Progress: {pct}% ({current}/{total}) — Value: {value:.3g}")
        QApplication.processEvents()

    # ─────────────────────────────────────────────────────────────────
    #  TOP BAR
    # ─────────────────────────────────────────────────────────────────
    def _setup_top_bar(self):
        # Create a two-row layout to prevent overlap
        top_container = QWidget()
        top_layout = QVBoxLayout(top_container)
        top_layout.setSpacing(4)
        top_layout.setContentsMargins(0, 0, 0, 0)
        
        # Row 1: Main action buttons
        row1 = QHBoxLayout()
        row1.setSpacing(6)

        # ── Run button ──────────────────────────────────────────────
        self.btn_run = QPushButton("RUN SIMULATION")
        self.btn_run.setMinimumHeight(38)
        self.btn_run.setStyleSheet("""
            QPushButton {
                font-weight: bold; font-size: 14px;
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #40A060, stop:1 #2E7D32);
                color: white; border-radius: 5px;
            }
            QPushButton:hover { background: #4CAF70; }
            QPushButton:disabled { background: #555568; color: #888; }
        """)
        self.btn_run.clicked.connect(lambda: self.sim_controller.run_single(self.config_manager.config, on_success=self._on_simulation_done, on_error=self._on_sim_error))

        # ── Stochastic button ────────────────────────────────────────
        self.btn_stoch = QPushButton("STOCHASTIC")
        self.btn_stoch.setMinimumHeight(38)
        self.btn_stoch.setStyleSheet("""
            QPushButton {
                font-weight: bold; font-size: 12px;
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #6050A0, stop:1 #40306A);
                color: white; border-radius: 5px;
            }
            QPushButton:hover { background: #7060B0; }
            QPushButton:disabled { background: #555568; color: #888; }
        """)
        self.btn_stoch.setToolTip("Run stochastic simulation (Langevin gate & membrane noise via Native Hines)")
        self.btn_stoch.clicked.connect(lambda: self.sim_controller.run_stochastic(self.config_manager.config, 1, on_success=self._on_stoch_done, on_error=self._on_sim_error))

        # ── Sweep button ─────────────────────────────────────────────
        self.btn_sweep = QPushButton("SWEEP")
        self.btn_sweep.setMinimumHeight(38)
        self.btn_sweep.setStyleSheet("""
            QPushButton {
                font-weight: bold; font-size: 12px;
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #0070A0, stop:1 #004E70);
                color: white; border-radius: 5px;
            }
            QPushButton:hover { background: #0080B0; }
            QPushButton:disabled { background: #555568; color: #888; }
        """)
        self.btn_sweep.setToolTip("Run parametric sweep (configured in Analysis tab)")
        self.btn_sweep.clicked.connect(self.run_sweep)

        self.btn_fi = QPushButton("f-I")
        self.btn_fi.setMinimumHeight(38)
        self.btn_fi.setStyleSheet("""
            QPushButton {
                font-weight: bold; font-size: 12px;
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #2A8F7A, stop:1 #1D6758);
                color: white; border-radius: 5px;
            }
            QPushButton:hover { background: #33A18A; }
            QPushButton:disabled { background: #555568; color: #888; }
        """)
        self.btn_fi.setToolTip("Run a dedicated f-I sweep over Iext with results opened in the f-I analytics tab")
        self.btn_fi.clicked.connect(self.run_fi_curve)

        # ── SD / ExcMap buttons ───────────────────────────────────────
        self.btn_sd = QPushButton("S-D")
        self.btn_sd.setMinimumHeight(38)
        self.btn_sd.setStyleSheet("""
            QPushButton {
                font-weight: bold; font-size: 12px;
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #906030, stop:1 #603010);
                color: white; border-radius: 5px;
            }
            QPushButton:hover { background: #A07040; }
            QPushButton:disabled { background: #555568; color: #888; }
        """)
        self.btn_sd.setToolTip("Compute Strength-Duration curve (binary search)")
        self.btn_sd.clicked.connect(self.run_sd_curve)

        self.btn_excmap = QPushButton("EXCIT. MAP")
        self.btn_excmap.setMinimumHeight(38)
        self.btn_excmap.setStyleSheet("""
            QPushButton {
                font-weight: bold; font-size: 12px;
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #208080, stop:1 #106060);
                color: white; border-radius: 5px;
            }
            QPushButton:hover { background: #30A0A0; }
            QPushButton:disabled { background: #555568; color: #888; }
        """)
        self.btn_excmap.setToolTip("Compute 2-D excitability map (I x duration)")
        self.btn_excmap.clicked.connect(self.run_excmap)

        # ── Cancel button (hidden until computation starts) ──────────
        self.btn_cancel = QPushButton("CANCEL")
        self.btn_cancel.setMinimumHeight(38)
        self.btn_cancel.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #CC3030, stop:1 #991818);
                color: white; border-radius: 5px;
            }
            QPushButton:hover { background: #FF4040; }
        """)
        self.btn_cancel.setToolTip("Cancel the running computation")
        self.btn_cancel.clicked.connect(self._request_cancel)
        self.btn_cancel.setVisible(False)
        self._cancel_requested = False

        # ── Hines solver toggle ──────────────────────────────────────
        self.btn_hines = QPushButton("HINES")
        self.btn_hines.setMinimumHeight(38)
        self.btn_hines.setCheckable(True)
        self.btn_hines.setChecked(False)
        self.btn_hines.setStyleSheet("""
            QPushButton {
                background: #313244; color: #89DCEB; border-radius: 5px;
                border: 1px solid #45475A;
                font-weight: bold;
            }
            QPushButton:hover { background: #3E3F5E; }
            QPushButton:disabled { color: #555568; }
        """)
        self.btn_hines.setToolTip("Toggle native Hines solver (O(N) vs SciPy BDF)")
        self.btn_hines.clicked.connect(self._on_hines_toggled)

        # ── Export buttons ─────────────────────────────────────
        self.btn_export = QPushButton("CSV")
        self.btn_export_plot = QPushButton("Plot")
        self.btn_export_plot.setMinimumHeight(38)
        self.btn_export_plot.setEnabled(False)
        self.btn_export_plot.setStyleSheet("""
            QPushButton {
                background: #313244; color: #A6E3A1; border-radius: 5px;
                border: 1px solid #45475A;
            }
            QPushButton:hover  { background: #3E3F5E; }
            QPushButton:disabled { color: #555568; }
        """)
        self.btn_export_plot.clicked.connect(self.export_plot)

        self.btn_export.setMinimumHeight(38)
        self.btn_export.setEnabled(False)
        self.btn_export.setStyleSheet("""
            QPushButton {
                background: #313244; color: #89DCEB; border-radius: 5px;
                border: 1px solid #45475A;
            }
            QPushButton:hover  { background: #3E3F5E; }
            QPushButton:disabled { color: #555568; }
        """)
        self.btn_export.clicked.connect(self.export_csv)

        self.btn_export_nml = QPushButton("NeuroML")
        self.btn_export_nml.setMinimumHeight(38)
        self.btn_export_nml.setToolTip("Export model config to NeuroML 2.2 XML (Stage 6.4)")
        self.btn_export_nml.setStyleSheet("""
            QPushButton {
                background: #313244; color: #CBA6F7; border-radius: 5px;
                font-size: 12px; font-weight: bold;
            }
            QPushButton:hover { background: #45475A; }
        """)
        self.btn_export_nml.clicked.connect(self.export_neuroml)

        # ── Save/Load Config buttons ─────────────────────────────
        self.btn_save_config = QPushButton("Save")
        self.btn_save_config.setMinimumHeight(38)
        self.btn_save_config.setStyleSheet("""
            QPushButton {
                font-weight: bold; font-size: 12px;
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #28A745, stop:1 #20893D);
                color: white; border-radius: 5px;
            }
            QPushButton:hover { background: #36A349; }
            QPushButton:disabled { background: #555568; color: #888; }
        """)
        self.btn_save_config.setToolTip("Save current configuration to JSON file")
        self.btn_save_config.clicked.connect(self.save_config_as)

        self.btn_load_config = QPushButton("Open")
        self.btn_load_config.setMinimumHeight(38)
        self.btn_load_config.setStyleSheet("""
            QPushButton {
                font-weight: bold; font-size: 12px;
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #6F42C1, stop:1 #4A1E61);
                color: white; border-radius: 5px;
            }
            QPushButton:hover { background: #8B5CF6; }
            QPushButton:disabled { background: #555568; color: #888; }
        """)
        self.btn_load_config.setToolTip("Load configuration from JSON file")
        self.btn_load_config.clicked.connect(self.load_config_from)

        # ── Sidebar toggle button ──────────────────────────────
        self.btn_toggle_sidebar = QPushButton("<")
        self.btn_toggle_sidebar.setFixedWidth(30)
        self.btn_toggle_sidebar.setFixedHeight(38)
        self.btn_toggle_sidebar.setToolTip("Show / hide parameters panel")
        self.btn_toggle_sidebar.setStyleSheet(
            "QPushButton { background:#313244; color:#89B4FA; border-radius:5px; font-size:14px; }"
            "QPushButton:hover { background:#45475A; }"
        )
        self.btn_toggle_sidebar.clicked.connect(self._toggle_sidebar)

        # ── Window size preset button ────────────────────────────
        self.btn_window_size = QPushButton("Size")
        self.btn_window_size.setFixedWidth(30)
        self.btn_window_size.setFixedHeight(38)
        self.btn_window_size.setToolTip("Window size preset (for laptops)")
        self.btn_window_size.setStyleSheet(
            "QPushButton { background:#313244; color:#CBA6F7; border-radius:5px; font-size:14px; }"
            "QPushButton:hover { background:#45475A; }"
        )
        self.btn_window_size.clicked.connect(self._show_window_size_menu)

        # Add row 1 buttons
        row1.addWidget(self.btn_run, 4)
        row1.addWidget(self.btn_cancel, 2)
        row1.addWidget(self.btn_hines, 2)
        row1.addWidget(self.btn_stoch, 2)
        row1.addWidget(self.btn_sweep, 2)
        row1.addWidget(self.btn_fi, 1)
        row1.addWidget(self.btn_sd, 2)
        row1.addWidget(self.btn_excmap, 2)
        row1.addWidget(self.btn_export_plot, 2)
        row1.addWidget(self.btn_export, 2)
        row1.addWidget(self.btn_export_nml, 2)
        row1.addWidget(self.btn_save_config, 2)
        row1.addWidget(self.btn_load_config, 2)
        row1.addWidget(self.btn_window_size)
        row1.addWidget(self.btn_toggle_sidebar)
        row1.addStretch()
        
        # Row 2: Preset, Language, and Sparkline
        row2 = QHBoxLayout()
        row2.setSpacing(8)

        # ── Preset selector ───────────────────────────────────────────
        self.lbl_preset = QLabel("Preset:")
        self.combo_presets = QComboBox()
        self.combo_presets.setMinimumWidth(200)
        self.combo_presets.addItems(["— Select preset —"] + get_preset_names())
        self.combo_presets.currentTextChanged.connect(self.load_preset)

        # ── Language selector ─────────────────────────────────────────
        self.lbl_lang = QLabel("Lang:")
        self.combo_lang = QComboBox()
        self.combo_lang.addItems(["EN", "RU"])
        self.combo_lang.setCurrentText("EN")
        self.combo_lang.currentTextChanged.connect(self.change_language)

        # ── Experience mode selector ───────────────────────────────
        self.lbl_mode = QLabel("Mode:")
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["🔬 Microscope", "🌉 Bridge", "🧪 Research"])
        self.combo_mode.setCurrentText("🌉 Bridge")
        self.combo_mode.currentTextChanged.connect(self._toggle_ui_complexity)

        # ── Sparkline (mini Vm trace) ──────────────────────────
        self._sparkline = pg.PlotWidget()
        self._sparkline.setFixedHeight(36)
        self._sparkline.setMinimumWidth(150)
        self._sparkline.hideAxis('left')
        self._sparkline.hideAxis('bottom')
        self._sparkline.setBackground('#0D1117')
        self._sparkline.setToolTip("Latest somatic Vm trace")
        self._sparkline_curve = self._sparkline.plot([], [], pen=pg.mkPen('#89B4FA', width=1))

        row2.addWidget(self.lbl_preset)
        row2.addWidget(self.combo_presets, 2)
        row2.addWidget(self.lbl_lang)
        row2.addWidget(self.combo_lang)
        row2.addWidget(self.lbl_mode)
        row2.addWidget(self.combo_mode)
        row2.addWidget(self._sparkline, 1)
        row2.addStretch()

        top_layout.addLayout(row1)
        top_layout.addLayout(row2)
        self._main_layout.addWidget(top_container)

    # ─────────────────────────────────────────────────────────────────
    #  TABS  +  COLLAPSIBLE SIDEBAR
    # ─────────────────────────────────────────────────────────────────
    def _setup_tabs(self):
        """
        Constructs and configures the application's main tabbed interface, collapsible sidebar, and the widgets contained in each tab.
        
        Creates and assigns:
        - the QTabWidget stored on `self.tabs`;
        - a sidebar frame used as the primary parameter/quick-controls panel (`self._sidebar_frame`);
        - the "Stimulation Studio" tab with primary stimulus forms (`self.form_stim`, `self.form_stim_loc`, `self.form_dfilter`), the secondary/dual stimulation widget (`self.dual_stim_widget`), and a resizable real-time stimulus preview plot with an I/g toggle (`self._stim_preview_plot`, `self._toggle_conductance`);
        - the "Oscilloscope" tab (`self.oscilloscope`) and syncs its delay controls with the current config;
        - the "Analytics" tab (`self.analytics`) with a session notes editor (`self.session_notes`);
        - the "Topology", "Axon Biophysics", and "Guide" tabs (`self.topology`, `self.axon_biophysics`, `self.guide_browser`);
        - the horizontal main splitter (`self._main_splitter`) that contains the sidebar and the tabs.
        
        Also wires relevant signals (e.g., stimulus/dual-stim change handlers, oscilloscope ↔ analytics time highlighting), sets initial widget states from the configuration, and registers form widgets with the ConfigManager.
        """
        self.tabs = QTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.tabs.setMovable(True)  # Allow users to rearrange tabs

        # ── Sidebar panel (replaces old "1) Setup" tab) ───────────────
        self._sidebar_frame = QFrame()
        self._sidebar_frame.setFrameShape(QFrame.Shape.StyledPanel)
        # Calculate sidebar width based on screen size (15-20% of screen width, min 320px)
        from PySide6.QtWidgets import QApplication
        screen_width = QApplication.primaryScreen().availableSize().width()
        calculated_width = max(320, min(int(screen_width * 0.18), 400))
        self._sidebar_frame.setMinimumWidth(calculated_width)
        self._sidebar_frame.setMaximumWidth(520)
        self._build_sidebar_panel()
        # Dummy tab_params kept for backward-compat references (not in tabs)
        self.tab_params = self._sidebar_frame

        # ── Tab 1: Stimulation Studio ───────────────────────────────
        # Step 2: Stimulation Studio (consolidated stimulation controls)
        stim_studio_widget_outer = QWidget()
        stim_outer_layout = QVBoxLayout(stim_studio_widget_outer)
        stim_outer_layout.setContentsMargins(0, 0, 0, 0)
        stim_outer_layout.setSpacing(0)
        
        # Create scroll area for stim studio content
        from PySide6.QtWidgets import QScrollArea
        stim_scroll = QScrollArea()
        stim_scroll.setWidgetResizable(True)
        stim_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        stim_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        stim_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        stim_scroll.setStyleSheet("QScrollArea { background: #1E1E2E; }")
        
        stim_studio_widget = QWidget()
        stim_main_layout = QVBoxLayout(stim_studio_widget)
        stim_main_layout.setSpacing(6)
        stim_main_layout.setContentsMargins(4, 4, 4, 4)
        
        # Top section: Primary and Secondary stimulus controls
        stim_controls_widget = QWidget()
        stim_layout = QHBoxLayout(stim_controls_widget)
        stim_layout.setSpacing(10)
        stim_layout.setContentsMargins(0, 0, 0, 0)
        
        # Left column: Primary stimulus controls
        left_col = QWidget()
        left_layout = QVBoxLayout(left_col)
        left_layout.setSpacing(6)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
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
        
        # Dynamic attenuation hint label (v10.3)
        self.lbl_attenuation_hint = QLabel("")
        self.lbl_attenuation_hint.setStyleSheet("color: #6C7086; font-size: 11px; padding: 4px;")
        self.lbl_attenuation_hint.setWordWrap(True)
        
        left_layout.addWidget(self.form_stim)
        left_layout.addWidget(self.form_stim_loc)
        left_layout.addWidget(self.form_dfilter)
        left_layout.addWidget(self.lbl_attenuation_hint)
        left_layout.addStretch()
        
        # Right column: Secondary stimulus (dual stimulation widget)
        self.dual_stim_widget = getattr(self, "dual_stim_widget", DualStimulationWidget())
        if not self._dual_stim_signal_connected:
            self.dual_stim_widget.config_changed.connect(self._on_dual_stim_config_changed)
            self._dual_stim_signal_connected = True
        
        stim_layout.addWidget(left_col, 1)
        stim_layout.addWidget(self.dual_stim_widget, 1)
        
        # Vertical splitter for resizable stim preview
        stim_splitter = QSplitter(Qt.Orientation.Vertical)
        stim_splitter.setChildrenCollapsible(False)  # Prevent complete collapse
        stim_splitter.addWidget(stim_controls_widget)
        
        # Bottom section: Real-time stimulus preview (resizable)
        preview_container = QWidget()
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(4)
        
        # Preview controls row
        preview_controls = QHBoxLayout()
        preview_controls.setSpacing(8)
        
        btn_refresh_preview = QPushButton("🔄 Refresh Preview")
        btn_refresh_preview.setToolTip("Manually refresh stimulus preview with current parameters")
        btn_refresh_preview.clicked.connect(self._update_stim_preview)
        btn_refresh_preview.setMaximumWidth(140)
        
        self._toggle_conductance = QPushButton("I / g")
        self._toggle_conductance.setCheckable(True)
        self._toggle_conductance.setChecked(False)
        self._toggle_conductance.setToolTip("Toggle between current (I) and conductance (g) view")
        self._toggle_conductance.setMaximumWidth(60)
        self._toggle_conductance.clicked.connect(self._update_stim_preview)
        
        preview_controls.addWidget(btn_refresh_preview)
        preview_controls.addWidget(self._toggle_conductance)
        preview_controls.addStretch()
        
        self._stim_preview_plot = pg.PlotWidget(title="Stimulus Protocol Preview")
        self._stim_preview_plot.setLabel('left', 'Total Input Current')
        self._stim_preview_plot.setLabel('bottom', 'Time (ms)')
        self._stim_preview_plot.setBackground('#1E1E2E')
        self._stim_preview_plot.setMinimumHeight(150)  # Larger minimum for better visibility
        self._stim_preview_plot.setMaximumHeight(500)  # Allow more space for plot
        
        preview_layout.addLayout(preview_controls)
        preview_layout.addWidget(self._stim_preview_plot)
        
        stim_splitter.addWidget(preview_container)
        stim_splitter.setSizes([250, 300])  # Give preview more space initially
        stim_splitter.setStretchFactor(0, 1)  # Controls get 1x stretch
        stim_splitter.setStretchFactor(1, 2)  # Preview gets 2x stretch
        
        stim_main_layout.addWidget(stim_splitter)
        
        stim_scroll.setWidget(stim_studio_widget)
        stim_outer_layout.addWidget(stim_scroll)
        
        self.tabs.addTab(stim_studio_widget_outer, "2) Stimulation Studio")

        # ── Tab 2: Oscilloscope ───────────────────────────────────────
        # Step 3: Oscilloscope
        self.oscilloscope = OscilloscopeWidget()
        self.oscilloscope.delay_target_changed.connect(self._on_delay_target_changed)
        self._delay_target_name, self._delay_custom_index = (
            self.oscilloscope.get_delay_target_selection()
        )
        self.oscilloscope.sync_delay_controls_for_config(self.config_manager.config)
        self.tabs.addTab(self.oscilloscope, "3) Oscilloscope")

        # ── Tab 3: Analytics ──────────────────────────────────────────
        # Step 4: Analytics
        analytics_container = QWidget()
        analytics_layout = QVBoxLayout(analytics_container)
        analytics_layout.setContentsMargins(0, 0, 0, 0)
        analytics_layout.setSpacing(0)
        
        self.analytics = AnalyticsWidget()
        analytics_layout.addWidget(self.analytics)
        
        # Collapsible Session Notes
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
        
        self.tabs.addTab(analytics_container, "4) Analytics")
        
        # Connect oscilloscope time_highlighted to analytics for linked cursors
        self.oscilloscope.time_highlighted.connect(self.analytics.highlight_time)

        # ── Tab 4: Topology ───────────────────────────────────────────
        # Step 5: Topology
        self.topology = TopologyWidget()
        self.topology.set_delay_focus(
            self._delay_target_name,
            self._delay_custom_index,
        )
        self.tabs.addTab(self.topology,     "5) Topology")

        # ── Bi-directional time sync between Oscilloscope and Topology ──
        # Oscilloscope crosshair -> Topology heatmap update
        self.oscilloscope.time_highlighted.connect(self.topology.highlight_time)
        # Topology time slider -> Oscilloscope vertical marker line
        self.topology.time_scrubbed.connect(self.oscilloscope.set_time_marker)
        # Topology compartment click -> Oscilloscope delay target
        self.topology.compartment_selected.connect(self.oscilloscope.set_delay_compartment)

        # ── Tab 5: Axon Biophysics ───────────────────────────────────
        # Step 6: Axon Biophysics
        self.axon_biophysics = AxonBiophysicsWidget()
        self.tabs.addTab(self.axon_biophysics, "6) Axon Biophysics")

        # ── Tab 6: Guide ──────────────────────────────────────────────
        # Step 7: Guide
        from PySide6.QtWidgets import QTextBrowser
        self.guide_browser = QTextBrowser()
        self.guide_browser.setOpenExternalLinks(True)
        self.guide_browser.setStyleSheet(
            "background:#0D1117; color:#C9D1D9; font-size:13px; border:none;"
        )
        self.tabs.addTab(self.guide_browser, "7) Guide")

        # ── Main splitter: sidebar | tabs ────────────────────────────
        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._main_splitter.setChildrenCollapsible(False)  # Prevent complete collapse
        self._main_splitter.setHandleWidth(6)
        self._main_splitter.addWidget(self._sidebar_frame)
        self._main_splitter.addWidget(self.tabs)
        # Use smaller initial sizes for laptop screens
        self._main_splitter.setSizes([350, 800])
        self._main_splitter.setStretchFactor(0, 0)
        self._main_splitter.setStretchFactor(1, 1)
        self._main_layout.addWidget(self._main_splitter, stretch=1)

        # Pass form widgets to ConfigManager for sync operations
        self.config_manager.set_dual_stim_widget(self.dual_stim_widget)
        self.config_manager.set_form_widgets(
            self.form_stim, self.form_stim_loc, self.form_preset_modes
        )

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

        # ── Live Control Deck (Step 2) ─────────────────────────────────
        deck = QGroupBox("🎛️ Live Control Deck")
        deck_layout = QVBoxLayout(deck)
        deck_layout.setSpacing(6)
        deck_layout.setContentsMargins(6, 10, 6, 6)
        deck.setStyleSheet("QGroupBox { font-weight:bold; color:#89B4FA; }")

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

            # Get current value for label using path resolution
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
            deck_layout.addWidget(row_w)

        layout.addWidget(deck)

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
        self.form_metabolism = PydanticFormWidget(self.config_manager.config.metabolism, "ATP / Metabolism")
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
        context_right.addWidget(self.form_metabolism)
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

    def _toggle_ui_complexity(self, mode: str):
        """Toggle UI complexity based on experience mode.
        
        Microscope: Simplified for students - hide topology/axon tabs, hide morphology/analysis groups, force single-comp
        Bridge: Full functionality
        Research: Future SWC/Network support (identical to Bridge for now)
        """
        if mode == "🔬 Microscope":
            # Hide topology and axon biophysics tabs
            for i in range(self.tabs.count()):
                tab_text = self.tabs.tabText(i)
                if "Topology" in tab_text or "Axon Biophysics" in tab_text:
                    self.tabs.setTabVisible(i, False)
            
            # Hide morphology and analysis groups in sidebar
            if hasattr(self, 'grp_morph'):
                self.grp_morph.setVisible(False)
            if hasattr(self, 'grp_ana'):
                self.grp_ana.setVisible(False)
            
            # Force single-compartment mode
            self.config_manager.config.morphology.single_comp = True
            self._refresh_all_forms()
            
        elif mode == "🌉 Bridge":
            # Show all tabs
            for i in range(self.tabs.count()):
                self.tabs.setTabVisible(i, True)
            
            # Show all sidebar groups
            if hasattr(self, 'grp_morph'):
                self.grp_morph.setVisible(True)
            if hasattr(self, 'grp_ana'):
                self.grp_ana.setVisible(True)
            
        elif mode == "🧪 Research":
            # Identical to Bridge for now (future SWC/Network tabs)
            for i in range(self.tabs.count()):
                self.tabs.setTabVisible(i, True)
            
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
        try:
            self.sim_controller.run_single(
                self.config_manager.config,
                on_success=self._on_simulation_done,
                on_error=self._on_sim_error,
            )
        except Exception:
            pass

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
            w = stim_fields.get(field_name)
            l = labels.get(field_name)
            if w is not None:
                w.setVisible(is_visible)
            if l is not None:
                l.setVisible(is_visible)

    def _set_stim_form_value(self, field_name: str, value):
        """Set stim-form widget value without emitting change callbacks."""
        w = self.form_stim.widgets_map.get(field_name)
        if w is None:
            return
        w.blockSignals(True)
        try:
            if isinstance(w, QComboBox):
                w.setCurrentText(str(value))
            else:
                w.setValue(value)
        finally:
            w.blockSignals(False)

    def _on_morph_field_changed(self, field_name: str, _value):
        self.oscilloscope.sync_delay_controls_for_config(self.config_manager.config)
        # Always refresh hint so absolute current (nA) reflects updated geometry.
        self.lbl_params_hint.setText(self.config_manager.get_hint_text())
        # Update soma diameter in stim form if d_soma changed
        if field_name == 'd_soma' and hasattr(self.form_stim, 'update_soma_diameter'):
            self.form_stim.update_soma_diameter(self.config_manager.config.morphology.d_soma)
        self._refresh_topology_preview()
        self._update_stim_preview()

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
            
            self._sync_stim_type_controls()
            self.form_stim.refresh()
        if field_name in {"Iext", "stim_type"}:
            self.lbl_params_hint.setText(self.config_manager.get_hint_text())
            self.form_stim.refresh()
        if field_name == "t_sim":
            # t_sim change requires updating stim preview with new time range
            self._update_stim_preview()
        self.lbl_params_hint.setText(self.config_manager.get_hint_text())
        self._refresh_topology_preview()
        if field_name != "t_sim":
            self._update_stim_preview()

    def _on_stim_loc_field_changed(self, _field_name: str, _value):
        self.lbl_params_hint.setText(self.config_manager.get_hint_text())
        self._refresh_topology_preview()
        self._update_stim_preview()

    def _on_dfilter_field_changed(self, _field_name: str, _value):
        self._refresh_topology_preview()
        self._update_stim_preview()
        self._update_attenuation_hint()

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
            if combo is not None and combo.currentText() != name:
                combo.blockSignals(True)
                combo.setCurrentText(name)
                combo.blockSignals(False)
        self.config_manager.load_preset(name)
        self.config_manager.auto_select_jacobian_for_preset()
        self._sync_hines_button_state()
        self._refresh_all_forms()
        self.oscilloscope.sync_delay_controls_for_config(self.config_manager.config)
        
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

    def closeEvent(self, event):
        """Save session state on application exit."""
        try:
            import os
            self.config_manager.save_config_as(".last_session.json")
        except Exception:
            pass
        super().closeEvent(event)

    def _refresh_all_forms(self):
        for form in (self.form_morph, self.form_env, self.form_chan,
                     self.form_calcium, self.form_metabolism, self.form_stim, self.form_stim_loc,
                     self.form_dfilter, self.form_ana, self.form_preset_modes):
            form.refresh()
        # Update soma diameter in stim form after refresh in case morphology changed
        if hasattr(self.form_stim, 'update_soma_diameter'):
            self.form_stim.update_soma_diameter(self.config_manager.config.morphology.d_soma)
        self.lbl_params_hint.setText(self.config_manager.get_hint_text())
        self._sync_stim_type_controls()
        self._sync_preset_mode_controls()
        self._sync_live_deck_to_config()

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
        """Run Auto-Rheobase Search using binary search algorithm with visualization.

        Runs up to 10 mini-simulations (50ms each) to find the absolute
        minimum I_ext required to trigger a single spike.
        Includes convergence visualization and search tree display.
        """
        from core.solver import SimulationController
        from core.analysis import detect_spikes
        from copy import deepcopy
        import numpy as np

        self._lbl_rheobase_result.setText("🔍 Searching for threshold...")
        self._status("Running Auto-Rheobase Search...")
        QApplication.processEvents()

        # Initialize visualization
        search_history = []  # [(I_low, I_high, I_mid, has_spike), ...]

        # Binary search bounds (µA/cm²)
        I_low = 0.0
        I_high = 100.0

        # First, find an upper bound that elicits a spike
        cfg = deepcopy(self.config_manager.config)
        cfg.stim.t_sim = 50.0  # Short simulation

        self._lbl_rheobase_result.setText("🔍 Phase 1: Finding upper bound...")
        max_attempts = 10
        for attempt in range(max_attempts):
            cfg.stim.Iext = I_high
            try:
                solver = SimulationController(cfg)
                result = solver.run_single()
                pk_idx, spike_times, _ = detect_spikes(result.v_soma, result.t, threshold=-20.0)
                if len(spike_times) > 0:
                    search_history.append((0.0, I_high, I_high, True))
                    break
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug(f"Rheobase search exception: {e}")
                pass
            I_high *= 2.0
            search_history.append((0.0, I_high, I_high, False))
            if I_high > 1000.0:
                self._lbl_rheobase_result.setText("❌ Could not find upper bound for rheobase")
                self._status("Auto-Rheobase Search failed")
                return
            self._lbl_rheobase_result.setText(
                f"🔍 Phase 1: Expanding bound... I_high = {I_high:.1f} µA/cm²"
            )
            QApplication.processEvents()

        # Binary search with visualization tracking
        self._lbl_rheobase_result.setText("🔍 Phase 2: Binary search...")
        for iteration in range(10):
            I_mid = (I_low + I_high) / 2.0
            cfg.stim.Iext = I_mid

            try:
                solver = SimulationController(cfg)
                result = solver.run_single()
                pk_idx, spike_times, _ = detect_spikes(result.v_soma, result.t, threshold=-20.0)
                has_spike = len(spike_times) > 0
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug(f"Rheobase iteration exception: {e}")
                has_spike = False

            search_history.append((I_low, I_high, I_mid, has_spike))

            if has_spike:
                I_high = I_mid
            else:
                I_low = I_mid

            self._lbl_rheobase_result.setText(
                f"🔍 Iteration {iteration+1}/10: I = {I_mid:.2f} µA/cm²\n"
                f"   Range: [{I_low:.2f}, {I_high:.2f}]"
            )
            self._status(f"Auto-Rheobase: iter {iteration+1}, I={I_mid:.2f}, spike={'YES' if has_spike else 'NO'}")
            QApplication.processEvents()

        # Final result
        I_rheobase = (I_low + I_high) / 2.0
        uncertainty = (I_high - I_low) / 2.0

        # Build visualization text
        viz_text = self._format_rheobase_visualization(search_history, I_rheobase, uncertainty)

        self._lbl_rheobase_result.setText(
            f"✅ Rheobase: {I_rheobase:.2f} ± {uncertainty:.2f} µA/cm²\n"
            f"   Uncertainty: {uncertainty/I_rheobase*100:.1f}%\n\n"
            f"{viz_text}"
        )
        self._status(f"Auto-Rheobase complete: {I_rheobase:.2f} µA/cm²")

        # Update config
        self.config_manager.config.stim.Iext = I_rheobase
        self._refresh_all_forms()

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
        """Toggle sidebar visibility."""
        if self._sidebar_visible:
            self._sidebar_frame.hide()
            self.btn_toggle_sidebar.setText(">")
        else:
            self._sidebar_frame.show()
            self.btn_toggle_sidebar.setText("<")
        self._sidebar_visible = not self._sidebar_visible
    
    def _show_window_size_menu(self):
        """Show window size preset menu for laptop users."""
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)

        action_desktop = menu.addAction("🖥 Desktop (1400x900)")
        action_desktop.triggered.connect(lambda: self._set_window_size(1400, 900))

        action_laptop_small = menu.addAction("💻 Laptop Small (1100x700)")
        action_laptop_small.triggered.connect(lambda: self._set_window_size(1100, 700))

        action_laptop_large = menu.addAction("💻 Laptop Large (1280x800)")
        action_laptop_large.triggered.connect(lambda: self._set_window_size(1280, 800))
        
        menu.addSeparator()
        
        action_fullscreen = menu.addAction("⛶ Fullscreen")
        action_fullscreen.triggered.connect(self._toggle_fullscreen)
        
        menu.exec(self.btn_window_size.mapToGlobal(self.btn_window_size.rect().bottomLeft()))
    
    def _set_window_size(self, width: int, height: int):
        """Set window to specified size and adjust splitter sizes for laptop screens."""
        self.resize(width, height)
        
        # Adjust splitter sizes based on window width
        if width < 1200:  # Small laptop
            self._main_splitter.setSizes([280, width - 320])
            self._sidebar_frame.setMaximumWidth(350)
        elif width < 1400:  # Large laptop
            self._main_splitter.setSizes([320, width - 360])
            self._sidebar_frame.setMaximumWidth(450)
        else:  # Desktop
            self._main_splitter.setSizes([350, width - 400])
            self._sidebar_frame.setMaximumWidth(520)
        
        self._status(f"Window resized to {width}x{height}")
    
    def closeEvent(self, event):
        """Save UI state on close."""
        try:
            import json
            import logging
            logger = logging.getLogger(__name__)
            
            self.config_manager.save_config_as(".last_session.json")
            # Save UI state - check if widgets are still valid before accessing
            try:
                splitter_sizes = self._main_splitter.sizes()
            except:
                splitter_sizes = []
            
            live_deck = []
            for combo in self._live_combos:
                try:
                    if hasattr(combo, 'currentText'):
                        live_deck.append(combo.currentText())
                except:
                    pass  # Widget already deleted, skip
            
            ui_state = {
                "splitter_sizes": splitter_sizes,
                "live_deck": live_deck
            }
            with open(".ui_state.json", "w") as f:
                json.dump(ui_state, f)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
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
        """Reset splitter sizes to default (350/800)."""
        self._main_splitter.setSizes([350, 800])
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
        self._lock_ui(False)
        QMessageBox.critical(self, "Simulation Error", msg)
        self._status("Error.")

    def _sync_hines_button_state(self):
        """Sync Hines button state with current jacobian_mode in config."""
        if not hasattr(self, 'btn_hines'):
            return
        is_hines = str(getattr(self.config_manager.config.stim, "jacobian_mode", "dense_fd")) == "native_hines"
        self.btn_hines.setChecked(is_hines)
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
        # Refresh stimulation form to show new value
        self._refresh_all_forms()
        # Sync oscilloscope delay target to clicked compartment
        self.oscilloscope._combo_delay_target.setCurrentText("Custom Compartment")
        self.oscilloscope._spin_delay_comp.setValue(comp_idx)
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
                self.oscilloscope.update_plots(res)
                self.analytics.update_analytics(res)
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

            self.tabs.setCurrentWidget(self.oscilloscope)
        except Exception as e:
            QMessageBox.critical(self, "Simulation Error", str(e))
            self._status("Error — check parameters.")
        finally:
            self._lock_ui(False)

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
        self.tabs.setCurrentWidget(self.oscilloscope)

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
        self.tabs.setCurrentWidget(self.analytics)

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
            f"S-D done — Rheobase={rh:.2f} µA/cmÂ˛  "
            f"Chronaxie={'—' if tc != tc else f'{tc:.2f} ms'}"
        )
        self.tabs.setCurrentWidget(self.analytics)

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
        self.tabs.setCurrentWidget(self.analytics)

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
