"""
gui/main_window.py — Main Application Window v10.0

Tabs: Parameters | Oscilloscope | Analytics | Topology | Guide
Run modes: Standard | Monte-Carlo | Sweep | S-D Curve | Excit. Map | Stochastic
"""
import csv
import copy
import os
from pathlib import Path
import numpy as np

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPushButton, QLabel, QComboBox, QStatusBar,
    QScrollArea, QMessageBox, QApplication, QFileDialog,
    QGroupBox, QToolBar, QProgressDialog
)
from PySide6.QtCore import Qt, QThread, Signal, QObject, QRunnable, QThreadPool
from PySide6.QtGui import QIcon, QAction

from gui.locales import T
from core.models import FullModelConfig
from core.solver import NeuronSolver
from core.errors import SimulationParameterError
from core.presets import get_preset_names, apply_preset, apply_synaptic_stimulus
from core.advanced_sim import (SWEEP_PARAMS, run_sweep,
                                 run_sd_curve, run_excitability_map,
                                 run_euler_maruyama)
from core.validation import validate_simulation_config, build_preset_mode_warnings
from gui.widgets.form_generator import PydanticFormWidget
from gui.plots import OscilloscopeWidget
from gui.analytics import AnalyticsWidget
from gui.topology import TopologyWidget
from gui.axon_biophysics import AxonBiophysicsWidget
from gui.dual_stimulation_widget import DualStimulationWidget


# ─────────────────────────────────────────────────────────────────────
#  WORKER (background thread for long-running analyses)
# ─────────────────────────────────────────────────────────────────────
class WorkerSignals(QObject):
    finished = Signal(object)
    error    = Signal(str)
    progress = Signal(int, int, str)


class Worker(QRunnable):
    def __init__(self, fn, *args, progress_fn=None, **kwargs):
        super().__init__()
        self.fn      = fn
        self.args    = args
        self.kwargs  = kwargs
        self.signals = WorkerSignals()
        self.progress_fn = progress_fn  # Callback to report progress
        
        # If progress callback requested, inject it into kwargs
        if progress_fn is not None:
            self.kwargs['progress_cb'] = self._progress_callback

    def _progress_callback(self, i, n, val):
        """Report progress to UI layer."""
        if self.progress_fn:
            self.progress_fn(i, n, val)

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except Exception as e:
            self.signals.error.emit(str(e))


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
        self.config = FullModelConfig()
        self._current_preset_name = ""
        self._thread_pool = QThreadPool()
        self._dual_stim_signal_connected = False
        self._delay_target_name = "Terminal"
        self._delay_custom_index = 1

        central = QWidget()
        self.setCentralWidget(central)
        self._main_layout = QVBoxLayout(central)
        self._main_layout.setContentsMargins(6, 6, 6, 6)
        self._main_layout.setSpacing(6)

        self._setup_top_bar()
        self._setup_tabs()
        self._setup_status_bar()
        self.retranslate_ui()
        self.load_preset("A: Squid Giant Axon (HH 1952)")

    # ─────────────────────────────────────────────────────────────────
    #  TOP BAR
    # ─────────────────────────────────────────────────────────────────
    def _setup_top_bar(self):
        bar = QHBoxLayout()
        bar.setSpacing(8)

        # ── Run button ──────────────────────────────────────────────
        self.btn_run = QPushButton("▶ RUN SIMULATION")
        self.btn_run.setMinimumHeight(46)
        self.btn_run.setStyleSheet("""
            QPushButton {
                font-weight: bold; font-size: 15px;
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #40A060, stop:1 #2E7D32);
                color: white; border-radius: 6px;
            }
            QPushButton:hover { background: #4CAF70; }
            QPushButton:disabled { background: #555568; color: #888; }
        """)
        self.btn_run.clicked.connect(self.run_simulation)

        # ── Stochastic button ────────────────────────────────────────
        self.btn_stoch = QPushButton("🎲 STOCHASTIC")
        self.btn_stoch.setMinimumHeight(46)
        self.btn_stoch.setStyleSheet("""
            QPushButton {
                font-weight: bold; font-size: 13px;
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #6050A0, stop:1 #40306A);
                color: white; border-radius: 6px;
            }
            QPushButton:hover { background: #7060B0; }
            QPushButton:disabled { background: #555568; color: #888; }
        """)
        self.btn_stoch.setToolTip("Run Euler-Maruyama stochastic simulation (Langevin gate noise)")
        self.btn_stoch.clicked.connect(self.run_stochastic)

        # ── Sweep button ─────────────────────────────────────────────
        self.btn_sweep = QPushButton("↔ SWEEP")
        self.btn_sweep.setMinimumHeight(46)
        self.btn_sweep.setStyleSheet("""
            QPushButton {
                font-weight: bold; font-size: 13px;
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #0070A0, stop:1 #004E70);
                color: white; border-radius: 6px;
            }
            QPushButton:hover { background: #0080B0; }
            QPushButton:disabled { background: #555568; color: #888; }
        """)
        self.btn_sweep.setToolTip("Run parametric sweep (configured in Analysis tab)")
        self.btn_sweep.clicked.connect(self.run_sweep)

        # ── SD / ExcMap buttons ───────────────────────────────────────
        self.btn_sd = QPushButton("⏱ S-D")
        self.btn_sd.setMinimumHeight(46)
        self.btn_sd.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #906030, stop:1 #603010);
                color: white; border-radius: 6px;
            }
            QPushButton:hover { background: #A07040; }
            QPushButton:disabled { background: #555568; color: #888; }
        """)
        self.btn_sd.setToolTip("Compute Strength-Duration curve (binary search)")
        self.btn_sd.clicked.connect(self.run_sd_curve)

        self.btn_excmap = QPushButton("🗺 EXCIT. MAP")
        self.btn_excmap.setMinimumHeight(46)
        self.btn_excmap.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #208080, stop:1 #106060);
                color: white; border-radius: 6px;
            }
            QPushButton:hover { background: #30A0A0; }
            QPushButton:disabled { background: #555568; color: #888; }
        """)
        self.btn_excmap.setToolTip("Compute 2-D excitability map (I × duration)")
        self.btn_excmap.clicked.connect(self.run_excmap)

        # ── Cancel button (hidden until computation starts) ──────────
        self.btn_cancel = QPushButton("⛔ CANCEL")
        self.btn_cancel.setMinimumHeight(46)
        self.btn_cancel.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #CC3030, stop:1 #991818);
                color: white; border-radius: 6px;
            }
            QPushButton:hover { background: #FF4040; }
        """)
        self.btn_cancel.setToolTip("Cancel the running computation")
        self.btn_cancel.clicked.connect(self._request_cancel)
        self.btn_cancel.setVisible(False)
        self._cancel_requested = False

        # ── Hines solver toggle ──────────────────────────────────────
        self.btn_hines = QPushButton("⚡ HINES")
        self.btn_hines.setMinimumHeight(46)
        self.btn_hines.setCheckable(True)
        self.btn_hines.setChecked(False)
        self.btn_hines.setStyleSheet("""
            QPushButton {
                background: #313244; color: #89DCEB; border-radius: 6px;
                border: 1px solid #45475A;
                font-weight: bold;
            }
            QPushButton:hover { background: #3E3F5E; }
            QPushButton:disabled { color: #555568; }
        """)
        self.btn_hines.setToolTip("Toggle native Hines solver (O(N) vs SciPy BDF)")
        self.btn_hines.clicked.connect(self._on_hines_toggled)

        # ── Export button ─────────────────────────────────────────────
        self.btn_export = QPushButton("💾 Export CSV")
        self.btn_export_plot = QPushButton("Export Plot")
        self.btn_export_plot.setMinimumHeight(46)
        self.btn_export_plot.setEnabled(False)
        self.btn_export_plot.setStyleSheet("""
            QPushButton {
                background: #313244; color: #A6E3A1; border-radius: 6px;
                border: 1px solid #45475A;
            }
            QPushButton:hover  { background: #3E3F5E; }
            QPushButton:disabled { color: #555568; }
        """)
        self.btn_export_plot.clicked.connect(self.export_plot)

        self.btn_export.setMinimumHeight(46)
        self.btn_export.setEnabled(False)
        self.btn_export.setStyleSheet("""
            QPushButton {
                background: #313244; color: #89DCEB; border-radius: 6px;
                border: 1px solid #45475A;
            }
            QPushButton:hover  { background: #3E3F5E; }
            QPushButton:disabled { color: #555568; }
        """)
        self.btn_export.clicked.connect(self.export_csv)

        # ── Preset selector ───────────────────────────────────────────
        self.lbl_preset = QLabel("Preset:")
        self.combo_presets = QComboBox()
        self.combo_presets.setMinimumWidth(260)
        self.combo_presets.addItems(["— Select preset —"] + get_preset_names())
        self.combo_presets.currentTextChanged.connect(self.load_preset)

        # ── Language selector ─────────────────────────────────────────
        self.lbl_lang = QLabel("Lang:")
        self.combo_lang = QComboBox()
        self.combo_lang.addItems(["EN", "RU"])
        self.combo_lang.setCurrentText("EN")
        self.combo_lang.currentTextChanged.connect(self.change_language)

        bar.addWidget(self.btn_run,    4)
        bar.addWidget(self.btn_cancel, 2)
        bar.addWidget(self.btn_hines, 2)
        bar.addWidget(self.btn_stoch, 2)
        bar.addWidget(self.btn_sweep, 2)
        bar.addWidget(self.btn_sd,     2)
        bar.addWidget(self.btn_excmap, 2)
        bar.addWidget(self.btn_export_plot, 2)
        bar.addWidget(self.btn_export, 2)
        bar.addWidget(self.lbl_preset)
        bar.addWidget(self.combo_presets, 3)
        bar.addWidget(self.lbl_lang)
        bar.addWidget(self.combo_lang)
        self._main_layout.addLayout(bar)

    # ─────────────────────────────────────────────────────────────────
    #  TABS
    # ─────────────────────────────────────────────────────────────────
    def _setup_tabs(self):
        self.tabs = QTabWidget()

        # ── Tab 0: Parameters ─────────────────────────────────────────
        # Step 1: Setup
        self.tab_params = QWidget()
        self._build_params_tab()
        self.tabs.addTab(self.tab_params,   "1) Setup")

        # ── Tab 1: Dual Stimulation ───────────────────────────────────
        # Step 2: Dual Stimulation
        self.dual_stim_widget = getattr(self, "dual_stim_widget", DualStimulationWidget())
        if not self._dual_stim_signal_connected:
            self.dual_stim_widget.config_changed.connect(self._on_dual_stim_config_changed)
            self._dual_stim_signal_connected = True
        self.tabs.addTab(self.dual_stim_widget, "2) Dual Stim")

        # ── Tab 2: Oscilloscope ───────────────────────────────────────
        # Step 3: Oscilloscope
        self.oscilloscope = OscilloscopeWidget()
        self.oscilloscope.delay_target_changed.connect(self._on_delay_target_changed)
        self._delay_target_name, self._delay_custom_index = (
            self.oscilloscope.get_delay_target_selection()
        )
        self.oscilloscope.sync_delay_controls_for_config(self.config)
        self.tabs.addTab(self.oscilloscope, "3) Oscilloscope")

        # ── Tab 3: Analytics ──────────────────────────────────────────
        # Step 4: Analytics
        self.analytics = AnalyticsWidget()
        self.tabs.addTab(self.analytics,    "4) Analytics")

        # ── Tab 4: Topology ───────────────────────────────────────────
        # Step 5: Topology
        self.topology = TopologyWidget()
        self.topology.set_delay_focus(
            self._delay_target_name,
            self._delay_custom_index,
        )
        self.tabs.addTab(self.topology,     "5) Topology")

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

        self._main_layout.addWidget(self.tabs, stretch=1)

    def _build_params_tab(self):
        layout = QVBoxLayout(self.tab_params)

        self.lbl_params_hint = QLabel("")
        self.lbl_params_hint.setWordWrap(True)
        self.lbl_params_hint.setStyleSheet(
            "QLabel {"
            "background:#1f2538; color:#cdd6f4; border:1px solid #45475a; "
            "border-radius:6px; padding:8px; font-size:12px;"
            "}"
        )
        layout.addWidget(self.lbl_params_hint)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        content = QWidget()
        c_layout = QVBoxLayout(content)
        c_layout.setSpacing(10)

        # Forms
        self.form_stim = PydanticFormWidget(
            self.config.stim,
            "Stimulation",
            on_change=self._on_stim_field_changed,
        )
        self.form_stim_loc = PydanticFormWidget(
            self.config.stim_location,
            "Stimulus Location",
            on_change=self._on_stim_loc_field_changed,
        )
        self.form_dfilter = PydanticFormWidget(
            self.config.dendritic_filter,
            "Dendritic Filter",
            on_change=self._on_dfilter_field_changed,
        )
        self.form_preset_modes = PydanticFormWidget(
            self.config.preset_modes,
            "Preset Modes (K/N/O)",
            on_change=self._on_preset_mode_changed
        )
        self.form_chan = PydanticFormWidget(
            self.config.channels,
            "Ion Channels",
            on_change=self._on_channel_field_changed,
        )
        self.form_calcium = PydanticFormWidget(self.config.calcium, "Calcium Dynamics")
        self.form_morph = PydanticFormWidget(
            self.config.morphology,
            "Morphology",
            on_change=self._on_morph_field_changed,
        )
        self.form_env = PydanticFormWidget(self.config.env, "Environment")
        self.form_ana = PydanticFormWidget(self.config.analysis, "Analysis / Sweep / Map")

        # Group 1: run setup (user-edited first)
        grp_setup = QGroupBox("Run Setup (Edit First)")
        setup_layout = QHBoxLayout(grp_setup)
        setup_left = QVBoxLayout()
        setup_right = QVBoxLayout()
        setup_left.addWidget(self.form_stim)
        setup_left.addWidget(self.form_stim_loc)
        setup_left.addWidget(self.form_dfilter)
        setup_right.addWidget(self.form_preset_modes)
        self.lbl_dual_priority = QLabel("")
        self.lbl_dual_priority.setWordWrap(True)
        self.lbl_dual_priority.setStyleSheet(
            "QLabel { color:#f9e2af; background:#2a1f1f; border:1px solid #5a4a3a; border-radius:6px; padding:8px; }"
        )
        setup_right.addWidget(self.lbl_dual_priority)
        setup_right.addStretch()
        setup_layout.addLayout(setup_left)
        setup_layout.addLayout(setup_right)
        c_layout.addWidget(grp_setup)

        # Group 2: model biophysics
        grp_model = QGroupBox("Model Biophysics")
        model_layout = QHBoxLayout(grp_model)
        model_left = QVBoxLayout()
        model_right = QVBoxLayout()
        model_left.addWidget(self.form_chan)
        model_left.addWidget(self.form_calcium)
        model_right.addWidget(self.form_morph)
        model_right.addWidget(self.form_env)
        model_layout.addLayout(model_left)
        model_layout.addLayout(model_right)
        c_layout.addWidget(grp_model)

        # Group 3: advanced analysis/sweep tools
        grp_ana = QGroupBox("Advanced Analysis Tools")
        ana_layout = QVBoxLayout(grp_ana)
        ana_layout.addWidget(self.form_ana)
        c_layout.addWidget(grp_ana)

        scroll.setWidget(content)
        layout.addWidget(scroll)

    # ─────────────────────────────────────────────────────────────────
    #  STATUS BAR
    # ─────────────────────────────────────────────────────────────────
    def _setup_status_bar(self):
        self._sb = QStatusBar()
        self.setStatusBar(self._sb)

    def _status(self, msg: str):
        self._sb.showMessage(msg)

    def _update_params_hint(self):
        """Refresh setup hint text so parameter priority is always explicit."""
        dual_enabled = bool(getattr(self.dual_stim_widget, "config", None) and self.dual_stim_widget.config.enabled)
        p = (self._current_preset_name or "").lower()
        pm = self.config.preset_modes

        mode_note = "Mode flags: none for this preset."
        if "thalamic" in p:
            mode_note = f"Mode flags: K mode={pm.k_mode}."
        elif "alzheimer" in p:
            mode_note = f"Mode flags: N mode={pm.alzheimer_mode}."
        elif "hypoxia" in p:
            mode_note = f"Mode flags: O mode={pm.hypoxia_mode}."
        elif "multiple sclerosis" in p:
            mode_note = "Mode flags: F is single-stage (no progressive/terminal switch)."

        if dual_enabled:
            priority_note = (
                "Priority: Dual Stim is ON, so primary stimulation values from the Dual Stim tab "
                "override main Stimulation/Stimulus Location fields."
            )
        else:
            priority_note = (
                "Priority: Dual Stim is OFF, so main Stimulation/Stimulus Location fields are active."
            )

        stim_note = ""
        if self.config.stim.stim_type == "const" and (
            "interneuron" in p or "hippocampal ca1" in p or "purkinje" in p
        ):
            stim_note = (
                " Const here represents tonic drive proxy (current-clamp style), "
                "not a single synaptic event; switch stim_type to alpha/AMPA/NMDA for event-like input."
            )

        jac = str(getattr(self.config.stim, "jacobian_mode", "dense_fd"))
        is_multi = not bool(self.config.morphology.single_comp)
        heavy_family = any(k in p for k in ("thalamic", "alzheimer", "hypoxia", "multiple sclerosis"))
        if is_multi and heavy_family:
            jac_note = (
                f" Jacobian: {jac}. Recommended for heavy presets: sparse_fd/analytic_sparse "
                "(faster than dense_fd)."
            )
        else:
            jac_note = f" Jacobian: {jac}."

        if hasattr(self, "lbl_params_hint"):
            self.lbl_params_hint.setText(f"{priority_note}  {mode_note}{stim_note}{jac_note}")
        if hasattr(self, "lbl_dual_priority"):
            self.lbl_dual_priority.setText(priority_note)

    def _auto_select_jacobian_for_preset(self):
        """
        Prefer sparse_fd for computationally heavy multi-compartment presets
        unless user explicitly selected another non-dense mode or native_hines.
        """
        p = (self._current_preset_name or "").lower()
        heavy_family = any(k in p for k in ("thalamic", "alzheimer", "hypoxia", "multiple sclerosis"))
        if not heavy_family or bool(self.config.morphology.single_comp):
            return
        current_mode = str(getattr(self.config.stim, "jacobian_mode", "dense_fd"))
        # Don't override user's Hines selection or other non-dense modes
        if current_mode == "dense_fd":
            self.config.stim.jacobian_mode = "sparse_fd"
            # Update Hines button state to reflect change
            if hasattr(self, 'btn_hines'):
                self.btn_hines.setChecked(False)
                self.btn_hines.setText("⚡ HINES")
                self.btn_hines.setStyleSheet("""
                    QPushButton {
                        background: #313244; color: #89DCEB; border-radius: 6px;
                        border: 1px solid #45475A;
                        font-weight: bold;
                    }
                    QPushButton:hover { background: #3E3F5E; }
                    QPushButton:disabled { color: #555568; }
                """)

    def _sync_stim_type_controls(self):
        """Show only stimulation parameters relevant for current stim_type."""
        dual_enabled = bool(
            hasattr(self, "dual_stim_widget")
            and bool(self.dual_stim_widget.config.enabled)
        )
        stype = (
            str(getattr(self.dual_stim_widget.config, "primary_stim_type", "const"))
            if dual_enabled
            else str(getattr(self.config.stim, "stim_type", "const"))
        )
        stim_fields = self.form_stim.widgets_map
        labels = self.form_stim.labels_map

        alpha_like = {"alpha"}
        synaptic_like = {"AMPA", "NMDA", "GABAA", "GABAB", "Kainate", "Nicotinic"}
        pulse_like = {"pulse"}

        show_pulse_start = stype in (alpha_like | synaptic_like | pulse_like)
        show_pulse_dur = stype in pulse_like
        show_alpha_tau = stype in (alpha_like | synaptic_like)

        visibility = {
            "pulse_start": show_pulse_start,
            "pulse_dur": show_pulse_dur,
            "alpha_tau": show_alpha_tau,
        }
        for field_name, is_visible in visibility.items():
            w = stim_fields.get(field_name)
            l = labels.get(field_name)
            if w is not None:
                w.setVisible(is_visible)
            if l is not None:
                l.setVisible(is_visible)

    def _recompute_absolute_iext(self):
        """Update display-only absolute current from density and current soma size."""
        d = float(self.config.morphology.d_soma)
        area = np.pi * d * d
        self.config.stim.Iext_absolute_nA = float(self.config.stim.Iext) * area * 1000.0

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

    def _sync_primary_stim_preview_from_dual(self):
        """
        Mirror active dual-primary stimulus into disabled main stim controls.
        This removes ambiguity about which parameters actually drive the solver.
        """
        if not hasattr(self, "dual_stim_widget"):
            return
        dc = self.dual_stim_widget.config
        if not bool(getattr(dc, "enabled", False)):
            return

        self._set_stim_form_value("stim_type", dc.primary_stim_type)
        self._set_stim_form_value("Iext", float(dc.primary_Iext))
        self._set_stim_form_value("pulse_start", float(dc.primary_start))
        self._set_stim_form_value("pulse_dur", float(dc.primary_duration))
        self._set_stim_form_value("alpha_tau", float(dc.primary_alpha_tau))

        d = float(self.config.morphology.d_soma)
        area = np.pi * d * d
        i_abs = float(dc.primary_Iext) * area * 1000.0
        self._set_stim_form_value("Iext_absolute_nA", i_abs)

    def _on_morph_field_changed(self, field_name: str, _value):
        self.oscilloscope.sync_delay_controls_for_config(self.config)
        if field_name == "d_soma":
            self._recompute_absolute_iext()
            if "Iext_absolute_nA" in self.form_stim.widgets_map:
                self.form_stim.refresh()
        self._refresh_topology_preview()

    def _on_stim_field_changed(self, field_name: str, value):
        if field_name == "stim_type":
            stype = str(value)
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
                apply_synaptic_stimulus(self.config, syn_name)
            self._sync_stim_type_controls()
            self.form_stim.refresh()
        if field_name in {"Iext", "stim_type"}:
            self._recompute_absolute_iext()
            self.form_stim.refresh()
        self._update_params_hint()
        self._refresh_topology_preview()

    def _on_stim_loc_field_changed(self, _field_name: str, _value):
        self._update_params_hint()
        self._refresh_topology_preview()

    def _on_dfilter_field_changed(self, _field_name: str, _value):
        self._refresh_topology_preview()

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
            self.config,
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

        Returns
        -------
        bool
            True if dual stimulation is enabled.
        """
        dual_enabled = bool(self.dual_stim_widget.config.enabled)
        if dual_enabled:
            self.config.dual_stimulation = self.dual_stim_widget.get_config()
        else:
            self.config.dual_stimulation = None
        return dual_enabled

    def _sync_stim_controls_with_dual_mode(self):
        """Disable conflicting primary-stim controls when dual stimulation is enabled."""
        dual_enabled = bool(self.dual_stim_widget.config.enabled)
        if dual_enabled:
            self._sync_primary_stim_preview_from_dual()
        else:
            # Restore canonical values from config when dual mode is off.
            self.form_stim.refresh()
        overridden_stim_fields = (
            "stim_type",
            "Iext",
            "Iext_absolute_nA",
            "pulse_start",
            "pulse_dur",
            "alpha_tau",
            "stim_comp",
        )
        for field_name in overridden_stim_fields:
            w = self.form_stim.widgets_map.get(field_name)
            if w is not None:
                w.setEnabled(not dual_enabled)
        w_loc = self.form_stim_loc.widgets_map.get("location")
        if w_loc is not None:
            w_loc.setEnabled(not dual_enabled)

        if dual_enabled:
            self.form_stim.group_box.setTitle("Stimulation (Primary Overridden by Dual Stim)")
            self.form_stim_loc.group_box.setTitle("Stimulus Location (Overridden by Dual Stim)")
        else:
            self.form_stim.group_box.setTitle("Stimulation")
            self.form_stim_loc.group_box.setTitle("Stimulus Location")
        self._sync_stim_type_controls()
        self._update_params_hint()

    def _sync_preset_mode_controls(self):
        """Show only the mode selector that applies to the active preset."""
        p = (self._current_preset_name or "").lower()
        active = {
            "k_mode": "thalamic" in p,
            "alzheimer_mode": "alzheimer" in p,
            "hypoxia_mode": "hypoxia" in p,
        }
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
                widget.setToolTip("Active for current preset.")
            else:
                widget.setToolTip("Ignored for current preset.")
        if any_active:
            self.form_preset_modes.group_box.setTitle("Preset Modes (K/N/O)")
        else:
            self.form_preset_modes.group_box.setTitle("Preset Modes (not used for current preset)")
        self._update_params_hint()

    def _active_mode_suffix(self) -> str:
        """Compact status suffix for active preset mode selector state."""
        if not self._current_preset_name:
            return ""
        name = self._current_preset_name
        pm = self.config.preset_modes
        if "Thalamic" in name:
            return f" | K mode={pm.k_mode}"
        if "Alzheimer" in name:
            return f" | N mode={pm.alzheimer_mode}"
        if "Hypoxia" in name:
            return f" | O mode={pm.hypoxia_mode}"
        return ""

    # ─────────────────────────────────────────────────────────────────
    #  PRESET & LANGUAGE
    # ─────────────────────────────────────────────────────────────────
    def load_preset(self, name: str):
        if "—" in name or "Select" in name:
            return
        self._current_preset_name = name
        apply_preset(self.config, name)
        self._auto_select_jacobian_for_preset()
        self._sync_hines_button_state()
        self._refresh_all_forms()
        self.oscilloscope.sync_delay_controls_for_config(self.config)
        # Reset dual stim when loading new preset
        self.dual_stim_widget.load_default_preset()
        self._sync_stim_controls_with_dual_mode()
        self._sync_preset_mode_controls()
        self.topology.draw_neuron(
            self.config,
            delay_target_name=self._delay_target_name,
            delay_custom_index=self._delay_custom_index,
        )
        self._status(f"Preset applied: {name}{self._active_mode_suffix()}")

    def _refresh_all_forms(self):
        for form in (self.form_morph, self.form_env, self.form_chan,
                     self.form_calcium, self.form_stim, self.form_stim_loc,
                     self.form_dfilter, self.form_ana, self.form_preset_modes):
            form.refresh()
        self._recompute_absolute_iext()
        self.form_stim.refresh()
        self._sync_stim_type_controls()
        self._sync_stim_controls_with_dual_mode()
        self._sync_preset_mode_controls()

    def _on_preset_mode_changed(self, _field_name: str, _value):
        """Reapply active preset when user changes a mode selector."""
        if not self._current_preset_name:
            return
        apply_preset(self.config, self._current_preset_name)
        self._refresh_all_forms()
        self.oscilloscope.sync_delay_controls_for_config(self.config)
        self.topology.draw_neuron(
            self.config,
            delay_target_name=self._delay_target_name,
            delay_custom_index=self._delay_custom_index,
        )
        self._status(
            f"Preset mode updated: {self._current_preset_name}{self._active_mode_suffix()}"
        )

    def change_language(self, lang: str):
        T.set_language(lang)
        self.retranslate_ui()

    def retranslate_ui(self):
        self.setWindowTitle(T.tr('app_title'))
        self.btn_run.setText(T.tr('btn_run'))
        self.lbl_preset.setText(T.tr('preset_label'))
        self.lbl_lang.setText(T.tr('lbl_language'))
        self._status(T.tr('status_ready'))
        # Update guide text
        self.guide_browser.setHtml(_GUIDE_HTML)

    # ─────────────────────────────────────────────────────────────────
    #  SIMULATION HELPERS
    # ─────────────────────────────────────────────────────────────────
    def _lock_ui(self, busy: bool):
        for btn in (self.btn_run, self.btn_stoch, self.btn_sweep,
                    self.btn_sd, self.btn_excmap):
            btn.setEnabled(not busy)
        self.btn_cancel.setVisible(busy)
        if not busy:
            self._cancel_requested = False

    def _request_cancel(self):
        """User clicked Cancel — set flag for running workers to check."""
        self._cancel_requested = True
        self._status("Cancellation requested…")
        self.btn_cancel.setEnabled(False)

    def _on_sim_error(self, msg: str):
        self._lock_ui(False)
        QMessageBox.critical(self, "Simulation Error", msg)
        self._status("Error.")

    def _sync_hines_button_state(self):
        """Sync Hines button state with current jacobian_mode in config."""
        if not hasattr(self, 'btn_hines'):
            return
        is_hines = str(getattr(self.config.stim, "jacobian_mode", "dense_fd")) == "native_hines"
        self.btn_hines.setChecked(is_hines)
        if is_hines:
            self.btn_hines.setText("⚡ HINES ON")
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
            self.btn_hines.setText("⚡ HINES")
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
            self.config.stim.jacobian_mode = "native_hines"
            self.btn_hines.setStyleSheet("""
                QPushButton {
                    background: #00BCD4; color: #FFFFFF; border-radius: 6px;
                    border: 2px solid #00BCD4;
                    font-weight: bold;
                }
                QPushButton:hover { background: #00A97F; }
                QPushButton:disabled { color: #555568; }
            """)
            self.btn_hines.setText("⚡ HINES ON")
            self._sb.showMessage("Hines solver ACTIVE (experimental). Dense-FD BDF suspended.", 4000)
        else:
            self.config.stim.jacobian_mode = "dense_fd"
            self.btn_hines.setStyleSheet("""
                QPushButton {
                    background: #313244; color: #89DCEB; border-radius: 6px;
                    border: 1px solid #45475A;
                    font-weight: bold;
                }
                QPushButton:hover { background: #3E3F5E; }
                QPushButton:disabled { color: #555568; }
            """)
            self.btn_hines.setText("⚡ HINES")
            self._sb.showMessage("Hines solver OFF. Using dense_fd BDF.", 3000)

    def _preflight_validate(self) -> bool:
        """
        Validate configuration before launching long simulation runs.

        Hard errors stop execution, non-fatal warnings are surfaced in status bar.
        """
        try:
            warnings = validate_simulation_config(self.config)
            warnings.extend(
                build_preset_mode_warnings(self.config, self._current_preset_name)
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

        # Capture config snapshot and flags for the worker thread
        run_mc = self.config.analysis.run_mc
        run_bif = self.config.analysis.run_bifurcation
        bif_param = self.config.analysis.bif_param
        mc_trials = self.config.analysis.mc_trials

        if run_mc:
            self._status(f"Monte-Carlo ({mc_trials} trials)…")
        QApplication.processEvents()

        def _compute():
            """Run solver in background thread (scipy/numba release GIL)."""
            solver = NeuronSolver(self.config)
            result = {}
            if run_mc:
                result['mc_results'] = solver.run_monte_carlo()
            else:
                result['single'] = solver.run_single()
            if run_bif:
                result['bif'] = solver.run_bifurcation()
                result['bif_param'] = bif_param
            return result

        w = Worker(_compute)
        w.signals.finished.connect(self._on_simulation_done)
        w.signals.error.connect(self._on_sim_error)
        self._thread_pool.start(w)

    def _on_simulation_done(self, result: dict):
        """Handle simulation results on the main thread (UI updates)."""
        try:
            if 'mc_results' in result:
                self.oscilloscope.update_plots_mc(result['mc_results'])
                self._status(f"MC done — {len(result['mc_results'])} trials.")
            elif 'single' in result:
                res = result['single']
                self._last_result = res
                self.oscilloscope.update_plots(res)
                self.analytics.update_analytics(res)
                dual_cfg = self.dual_stim_widget.config if self.dual_stim_widget.config.enabled else None
                self.topology.draw_neuron(
                    self.config,
                    dual_config=dual_cfg,
                    delay_target_name=self._delay_target_name,
                    delay_custom_index=self._delay_custom_index,
                )
                self.axon_biophysics.plot_axon_data(res, self.config)
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

    def _on_dual_stim_config_changed(self):
        """Handle dual stimulation config changes from widget."""
        self._sync_dual_stim_into_config()
        self._sync_stim_controls_with_dual_mode()
        if self.dual_stim_widget.config.enabled:
            self._status(
                "Dual stim enabled: "
                f"{self.dual_stim_widget.config.primary_location} + "
                f"{self.dual_stim_widget.config.secondary_location} "
                "(Dual tab parameters override primary stimulation fields)"
            )
        else:
            self._status("Dual stimulation disabled")

    # ─────────────────────────────────────────────────────────────────
    #  2. STOCHASTIC (Euler-Maruyama)
    # ─────────────────────────────────────────────────────────────────
    def run_stochastic(self):
        self._lock_ui(True)
        self._status("Stochastic simulation (Euler-Maruyama)…")
        QApplication.processEvents()

        if not self._preflight_validate():
            self._lock_ui(False)
            return

        self._sync_dual_stim_into_config()

        def _do():
            return run_euler_maruyama(self.config)

        w = Worker(_do)
        w.signals.finished.connect(self._on_stoch_done)
        w.signals.error.connect(self._on_sim_error)
        self._thread_pool.start(w)

    def _on_stoch_done(self, res):
        self._last_result = res
        self.oscilloscope.update_plots(res)
        self.analytics.update_analytics(res)
        dual_cfg = self.dual_stim_widget.config if self.dual_stim_widget.config.enabled else None
        self.topology.draw_neuron(
            self.config,
            dual_config=dual_cfg,
            delay_target_name=self._delay_target_name,
            delay_custom_index=self._delay_custom_index,
        )
        self.axon_biophysics.plot_axon_data(res, self.config)
        self.btn_export_plot.setEnabled(True)
        self.btn_export.setEnabled(True)
        self._lock_ui(False)
        self._status("Stochastic run complete.")
        self.tabs.setCurrentWidget(self.oscilloscope)

    # ─────────────────────────────────────────────────────────────────
    #  3. SWEEP
    # ─────────────────────────────────────────────────────────────────
    def run_sweep(self):
        ana = self.config.analysis
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
        self._status(f"Sweep: {ana.sweep_param}  [{ana.sweep_min}…{ana.sweep_max}]  "
                     f"{ana.sweep_steps} steps…")
        QApplication.processEvents()

        def _do():
            return run_sweep(self.config, ana.sweep_param, param_vals)

        w = Worker(_do)
        w.signals.finished.connect(
            lambda res: self._on_sweep_done(res, ana.sweep_param)
        )
        w.signals.error.connect(self._on_sim_error)
        self._thread_pool.start(w)

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
        cfg_for_sd = self.config
        if dual_enabled:
            cfg_for_sd = copy.deepcopy(self.config)
            cfg_for_sd.dual_stimulation = None

        self._lock_ui(True)
        if dual_enabled:
            self._status("Computing Strength-Duration curve (dual-stim disabled for S-D analysis)…")
        else:
            self._status("Computing Strength-Duration curve (binary search)…")
        QApplication.processEvents()
        w = Worker(run_sd_curve, cfg_for_sd, progress_fn=self._report_progress)
        w.signals.finished.connect(self._on_sd_done)
        w.signals.error.connect(self._on_sim_error)
        self._thread_pool.start(w)

    def _on_sd_done(self, sd):
        rh = sd['rheobase']
        tc = sd['chronaxie']
        self.analytics.update_sd_curve(sd)
        self._lock_ui(False)
        self._status(
            f"S-D done — Rheobase={rh:.2f} µA/cm²  "
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
        cfg_for_excmap = self.config
        if dual_enabled:
            cfg_for_excmap = copy.deepcopy(self.config)
            cfg_for_excmap.dual_stimulation = None

        ana = self.config.analysis
        total = ana.excmap_NI * ana.excmap_ND
        self._lock_ui(True)
        if dual_enabled:
            self._status(
                f"Excitability map {ana.excmap_NI}×{ana.excmap_ND} = {total} runs "
                "(dual-stim disabled for map analysis)…"
            )
        else:
            self._status(f"Excitability map {ana.excmap_NI}×{ana.excmap_ND} = {total} runs…")
        QApplication.processEvents()
        w = Worker(run_excitability_map, cfg_for_excmap, progress_fn=self._report_progress)
        w.signals.finished.connect(self._on_excmap_done)
        w.signals.error.connect(self._on_sim_error)
        self._thread_pool.start(w)

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
                header += [f'I_{k}_uA_cm2' for k in res.currents]
                if res.ca_i is not None:
                    header.append('Ca_i_mM')

                # Gate names
                from core.analysis import extract_gate_traces
                gates = extract_gate_traces(res)
                header += [f'gate_{k}' for k in gates]
                writer.writerow(header)

                for i, t in enumerate(res.t):
                    row = [f"{t:.4f}", f"{res.v_soma[i]:.4f}"]
                    if res.n_comp > 1:
                        row.append(f"{res.v_all[1, i]:.4f}")
                        row.append(f"{res.v_all[-1, i]:.4f}")
                    for curr in res.currents.values():
                        row.append(f"{curr[i]:.6f}")
                    if res.ca_i is not None:
                        row.append(f"{res.ca_i[0, i]:.8f}")
                    for gv in gates.values():
                        row.append(f"{gv[i]:.6f}")
                    writer.writerow(row)

            self._status(f"Exported: {path}")
            QMessageBox.information(self, "Export", f"Saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))


# ─────────────────────────────────────────────────────────────────────
#  GUIDE HTML
# ─────────────────────────────────────────────────────────────────────
_GUIDE_HTML = """
<html><body style="background:#0D1117; color:#C9D1D9; font-family:Segoe UI,sans-serif; padding:20px;">

<h1 style="color:#89B4FA;">🧠 Hodgkin-Huxley Neuron Simulator v10.0</h1>
<p>A research-grade biophysical simulator based on the Hodgkin-Huxley (1952) formalism,
extended with multi-compartment morphology, optional ion channels, and advanced analysis tools.</p>

<h2 style="color:#A6E3A1;">▶ Quick Start</h2>
<ol>
  <li>Select a <b>Preset</b> from the dropdown (e.g. <i>Squid Giant Axon</i>).</li>
  <li>Adjust parameters in <b>1) Setup</b> (Run Setup / Model Biophysics / Advanced Analysis Tools).</li>
  <li>If needed, enable and configure secondary stimulation in <b>2) Dual Stim</b>.
      When Dual Stim is ON, dual primary fields override main stimulation fields.</li>
  <li>Click <b>▶ RUN SIMULATION</b> — results appear in Oscilloscope and Analytics.</li>
  <li>Inspect traces in <b>3) Oscilloscope</b> and metrics in <b>4) Analytics</b>.</li>
</ol>
<p style="color:#BAC2DE;">
In <b>Oscilloscope → View</b>, conduction delay can be measured from soma to
<i>Terminal</i>, <i>AIS</i>, <i>Trunk Junction</i>, or a custom compartment index.
</p>

<h2 style="color:#A6E3A1;">🔬 Run Modes</h2>
<table style="border-collapse:collapse; width:100%;">
<tr style="background:#1E3A5F;">
  <th style="padding:6px; text-align:left;">Button</th>
  <th style="padding:6px; text-align:left;">What it does</th>
</tr>
<tr><td style="padding:4px;"><b>▶ RUN</b></td>
    <td>Standard deterministic simulation (BDF stiff ODE solver)</td></tr>
<tr style="background:#1A1A2E;"><td style="padding:4px;"><b>Jacobian mode</b></td>
    <td>Set <i>stim.jacobian_mode</i> in Parameters → Stimulation:
        <i>dense_fd</i>, <i>sparse_fd</i>, or <i>analytic_sparse</i>.
        Heavy presets typically run faster with sparse modes.</td></tr>
<tr style="background:#1A1A2E;"><td style="padding:4px;"><b>🎲 STOCHASTIC</b></td>
    <td>Euler-Maruyama integrator with Langevin gate noise (Fox &amp; Lu 1994).
        Enable via <i>stoch_gating</i> flag or use <i>noise_sigma</i>.</td></tr>
<tr><td style="padding:4px;"><b>↔ SWEEP</b></td>
    <td>Parametric sweep. Set <i>sweep_param</i>, <i>sweep_min/max/steps</i>
        in Analysis section. Produces f-I curves and voltage traces.</td></tr>
<tr style="background:#1A1A2E;"><td style="padding:4px;"><b>⏱ S-D</b></td>
    <td>Strength-Duration curve. Binary search for threshold at 13 durations.
        Reports Rheobase (I at infinite duration) and Chronaxie (t at 2×I_rh).</td></tr>
<tr><td style="padding:4px;"><b>🗺 EXCIT. MAP</b></td>
    <td>2-D excitability map: spike count as function of (I_ext × pulse_dur).
        Set <i>excmap_*</i> parameters in Analysis section.</td></tr>
</table>

<h2 style="color:#A6E3A1;">⚙ Ion Channels</h2>
<ul>
  <li><b>Na / K / Leak</b> — classic Hodgkin-Huxley (1952) channels, always active.</li>
  <li><b>Ih</b> — HCN pacemaker current (Destexhe 1993). Causes rhythmic firing.</li>
  <li><b>ICa</b> — L-type calcium (Huguenard 1992). Enables plateau potentials.</li>
  <li><b>IA</b> — A-current, transient K⁺ (Connor-Stevens 1971). Delays first spike.</li>
  <li><b>SK</b> — Ca²⁺-activated K⁺ (NEW). Causes spike-frequency adaptation.</li>
</ul>

<h2 style="color:#A6E3A1;">🧬 Neuron Passport (Analytics → Passport)</h2>
<p>After each simulation, the Passport tab shows:</p>
<ul>
  <li>Passive: τ_m, R_in, λ (space constant)</li>
  <li>Spike: threshold, peak, AHP, halfwidth, dV/dt rate</li>
  <li>Firing: f_initial, f_steady, Adaptation Index, cell type classification (FS/RS/IB/LTS)</li>
  <li>Conduction velocity (multi-compartment mode)</li>
  <li>Energy: cumulative charge Q per channel, ATP estimate</li>
</ul>
<p style="color:#BAC2DE;">
Spike detector settings are configurable in <b>Parameters → Analysis</b>:
algorithm, threshold, prominence, baseline, refractory window and repolarization window.
</p>

<h2 style="color:#A6E3A1;">🔄 Phase Plane</h2>
<p>Shows the AP trajectory in V–n space plus nullclines (dV/dt=0 and dn/dt=0).
Fixed points are where both nullclines intersect. Limit cycles = sustained firing.</p>

<h2 style="color:#A6E3A1;">📐 Morphology</h2>
<p>Multi-compartment cable model: Soma → AIS (high gNa density) → Trunk → Bifurcation → Branch 1/2.
The Laplacian matrix couples adjacent compartments via axial conductance g_ax = d/(4·Ra·dx).</p>

<h2 style="color:#A6E3A1;">🧪 Preset Modes (K / N / O)</h2>
<p>Use <b>Parameters → Preset Modes</b> to switch validated stage/mode overlays:</p>
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
When <b>Dual Stim</b> is enabled, Dual tab primary settings override standard stimulation fields.
</p>

<h2 style="color:#A6E3A1;">💾 Export</h2>
<p>After a run, click <b>💾 Export CSV</b> to save all traces (V, currents, gates, Ca) as a CSV file
compatible with Excel, MATLAB, Python/pandas, etc. Use <b>Export Plot</b> to save the current
oscilloscope view as PNG, SVG, or PDF.</p>

<hr style="border-color:#45475A;">
<p style="color:#585B70; font-size:11px;">
HH Simulator v10.0 — Python/PySide6 port of Scilab HH v9.0 |
Numba JIT kinetics | scipy.sparse Laplacian | BDF + Euler-Maruyama solvers
</p>
</body></html>
"""
