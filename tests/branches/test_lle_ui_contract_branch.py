from __future__ import annotations

from types import SimpleNamespace
import inspect


def test_chaos_tab_exposes_only_native_benettin_controls():
    from PySide6.QtWidgets import QApplication, QComboBox, QSpinBox, QDoubleSpinBox
    from gui.analytics import AnalyticsWidget

    _ = QApplication.instance() or QApplication([])
    analytics = AnalyticsWidget()
    try:
        tab = analytics._build_tab_chaos()
        assert analytics.ax_chaos is not None
        assert hasattr(analytics, "_chaos_subspace_combo")
        assert analytics._chaos_subspace_combo.findText("Voltage Only") >= 0
        assert analytics._chaos_subspace_combo.findText("Voltage + Gates") >= 0
        assert analytics._chaos_subspace_combo.findText("Full State") >= 0
        assert not hasattr(analytics, "_chaos_embed_spin")
        assert not hasattr(analytics, "_chaos_pair_mode")
        assert len(tab.findChildren(QComboBox)) == 1
        assert not tab.findChildren(QSpinBox)
        assert not tab.findChildren(QDoubleSpinBox)
    finally:
        analytics.close()


def test_run_with_lle_enabled_passes_subspace_mode_to_controller():
    from gui.main_window import MainWindow

    class Combo:
        def currentText(self):
            return "Voltage + Gates"

    class Controller:
        def __init__(self):
            self.kwargs = None

        def run_single(self, _cfg, **kwargs):
            self.kwargs = kwargs

    cfg = SimpleNamespace(stim=SimpleNamespace(calc_lle=False))
    win = MainWindow.__new__(MainWindow)
    win.config_manager = SimpleNamespace(config=cfg)
    win.analytics = SimpleNamespace(_chaos_subspace_combo=Combo())
    win.sim_controller = Controller()
    win._status = lambda _msg: None

    MainWindow._run_with_lle_enabled(win)

    assert cfg.stim.calc_lle is True
    assert win.sim_controller.kwargs["compute_lyapunov"] is True
    assert win.sim_controller.kwargs["lle_subspace_mode"] == 1


def test_simulation_controller_routes_native_lle_to_run_native():
    from gui.simulation_controller import SimulationController

    sig = inspect.signature(SimulationController.run_single)
    assert "lle_subspace_mode" in sig.parameters

    source = inspect.getsource(SimulationController.run_single)
    assert "solver.run_native(config, calc_lle=True, lle_subspace_mode=int(lle_subspace_mode))" in source
