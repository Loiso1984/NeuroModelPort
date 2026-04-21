from __future__ import annotations

import inspect


def test_chaos_tab_is_plot_only_without_lle_launch_controls():
    from PySide6.QtWidgets import QApplication, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox
    from gui.analytics import AnalyticsWidget

    _ = QApplication.instance() or QApplication([])
    analytics = AnalyticsWidget()
    try:
        tab = analytics._build_tab_chaos()
        assert analytics.ax_chaos is not None
        assert not hasattr(analytics, "_btn_compute_lle")
        assert not hasattr(analytics, "_btn_chaos_compute_lle")
        assert not hasattr(analytics, "_chaos_subspace_combo")
        assert not hasattr(analytics, "_chaos_embed_spin")
        assert not hasattr(analytics, "_chaos_pair_mode")
        assert not tab.findChildren(QComboBox)
        assert not tab.findChildren(QSpinBox)
        assert not tab.findChildren(QDoubleSpinBox)
        assert not [btn for btn in tab.findChildren(QPushButton) if "LLE" in btn.text()]
    finally:
        analytics.close()


def test_debug_lte_tab_is_removed():
    from PySide6.QtWidgets import QApplication
    from gui.analytics import AnalyticsWidget

    _ = QApplication.instance() or QApplication([])
    analytics = AnalyticsWidget()
    try:
        titles = [spec["title"] for spec in analytics._all_tab_specs.values()]
        assert "Debug LTE" not in titles
        assert not hasattr(analytics, "_build_tab_debug")
        assert not hasattr(analytics, "_update_debug")
    finally:
        analytics.close()


def test_lle_results_tab_is_visible_and_focusable_from_filters():
    from PySide6.QtWidgets import QApplication
    from gui.analytics import AnalyticsWidget

    _ = QApplication.instance() or QApplication([])
    analytics = AnalyticsWidget()
    try:
        titles = [spec["title"] for spec in analytics._all_tab_specs.values()]
        assert "Lyapunov (LLE)" in titles

        analytics._combo_category.setCurrentText("Physics")
        assert "Lyapunov (LLE)" not in [analytics.tabText(i) for i in range(analytics.count())]

        assert analytics.show_lle_tab() is True
        assert analytics.currentWidget() is not None
        assert analytics.tabText(analytics.currentIndex()) == "Lyapunov (LLE)"
        assert hasattr(analytics, "ax_chaos")
    finally:
        analytics.close()


def test_experiment_studio_lle_passes_pydantic_subspace_to_controller(monkeypatch):
    from PySide6.QtWidgets import QApplication
    from gui.main_window import MainWindow

    _ = QApplication.instance() or QApplication([])
    win = MainWindow()

    class Controller:
        def __init__(self):
            self.kwargs = None

        def run_single(self, _cfg, **kwargs):
            self.kwargs = kwargs

    try:
        win.sim_controller = Controller()
        monkeypatch.setattr(win, "_preflight_validate", lambda: True)
        monkeypatch.setattr(win, "_sync_dual_stim_into_config", lambda: False)
        monkeypatch.setattr(win, "_sync_hines_button_state", lambda: None)
        monkeypatch.setattr(win, "_lock_ui", lambda _busy: None)
        monkeypatch.setattr(win, "_status", lambda _msg: None)

        win.exp_lle_subspace_combo.setCurrentText("Custom")
        win.exp_lle_delta_spin.setValue(1e-5)
        win.exp_lle_evolve_spin.setValue(2.0)
        win.exp_lle_custom_gates_edit.setText("m, h, s")
        win.exp_lle_custom_ca_check.setChecked(True)
        win._sync_experiment_studio_to_config()
        win._run_lle_experiment()

        assert win.config_manager.config.analysis.lle_subspace == "Custom"
        assert win.config_manager.config.analysis.lle_custom_gates == "m, h, s"
        assert win.config_manager.config.stim.jacobian_mode == "native_hines"
        assert win.sim_controller.kwargs["compute_lyapunov"] is True
        assert win.sim_controller.kwargs["lle_subspace_mode"] == 3
    finally:
        win.close()


def test_simulation_controller_routes_lle_to_native_with_pydantic_params():
    from gui.simulation_controller import SimulationController

    sig = inspect.signature(SimulationController.run_single)
    assert "lle_subspace_mode" in sig.parameters

    source = inspect.getsource(SimulationController.run_single)
    assert "solver.run_native(" in source
    assert "calc_lle=True" in source
    assert "lle_delta=float(getattr(config.analysis" in source
    assert "lle_t_evolve=float(getattr(config.analysis" in source
