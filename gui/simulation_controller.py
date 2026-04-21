"""
gui/simulation_controller.py - Thread Management for Simulations

Separates PyQt thread management from MainWindow UI logic.
Keeps QRunnable, QThreadPool, and Worker classes in the GUI package.
"""

from PySide6.QtCore import QObject, Signal, QThreadPool, QRunnable
from typing import Callable, Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


class WorkerSignals(QObject):
    """Signals for worker thread communication."""
    finished = Signal(object)  # SimulationResult
    error = Signal(str)        # Error message
    progress = Signal(int, int, float)  # (current, total, value) for progress updates
    # v13.0: Rich progress signal for rheobase (message, data_dict)
    progress_rich = Signal(str, dict)  # (message, data) for detailed progress updates


class Worker(QRunnable):
    """Runnable worker for simulation execution."""

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        """Execute the simulation function."""
        try:
            result = self.func(*self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except Exception as e:
            logger.error(f"Worker execution failed: {e}")
            self.signals.error.emit(str(e))


class SimulationController(QObject):
    """Manages simulation thread execution and signal routing.

    Separates thread management from UI logic in MainWindow.

    Signals:
        simulation_started: Emitted when a simulation starts
        simulation_finished: Emitted when simulation completes with result
        analytics_started: Emitted when analytics processing starts
        analytics_finished: Emitted when analytics completes with stats dict
        progress_updated: Emitted with (current, total, value) for progress
        error_occurred: Emitted when an error occurs
    """

    # Public signals for MainWindow to connect to
    simulation_started = Signal()
    simulation_finished = Signal(object)  # result dict
    analytics_started = Signal()  # v13.0: separate analytics signal
    analytics_finished = Signal(object)  # v13.0: stats dict
    progress_updated = Signal(int, int, float)  # (current, total, value)
    # v13.0: Rich progress for rheobase search (thread-safe)
    rheobase_progress = Signal(str, dict)  # (message, data) for rheobase updates
    error_occurred = Signal(str)  # error message

    def __init__(self, parent=None):
        super().__init__(parent)
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(4)  # Limit concurrent simulations

    @staticmethod
    def _snapshot_config(config):
        """Create an immutable simulation snapshot for worker threads."""
        if hasattr(config, "model_copy"):
            return config.model_copy(deep=True)
        from copy import deepcopy
        return deepcopy(config)

    def run_configured_simulation(
        self,
        config,
        on_success: Callable[[Any], None] = None,
        on_error: Callable[[str], None] = None,
    ) -> dict:
        """Dispatch single-run vs Monte-Carlo according to config.analysis.run_mc."""
        analysis = getattr(config, "analysis", None)
        run_mc = bool(getattr(analysis, "run_mc", False))
        if run_mc:
            mc_trials = int(getattr(analysis, "mc_trials", 1))
            self.run_monte_carlo(
                config,
                mc_trials,
                on_success=on_success,
                on_error=on_error,
            )
            return {"mode": "mc", "mc_trials": mc_trials}

        self.run_single(
            config,
            on_success=on_success,
            on_error=on_error,
        )
        return {"mode": "single", "mc_trials": 0}

    def run_stochastic_from_config(
        self,
        config,
        on_success: Callable[[Any], None] = None,
        on_error: Callable[[str], None] = None,
    ) -> int:
        """Run stochastic simulation using trial count from config.analysis.mc_trials."""
        analysis = getattr(config, "analysis", None)
        n_trials = int(getattr(analysis, "mc_trials", 1))
        self.run_stochastic(
            config,
            n_trials=n_trials,
            on_success=on_success,
            on_error=on_error,
        )
        return n_trials

    def run_sweep_from_config(
        self,
        config,
        on_success: Callable[[Any], None] = None,
        on_error: Callable[[str], None] = None,
    ) -> tuple[str, np.ndarray]:
        """Run sweep using analysis.sweep_* fields from config."""
        analysis = config.analysis
        param_name = str(getattr(analysis, "sweep_param", "stim.Iext"))
        param_vals = np.linspace(
            float(analysis.sweep_min),
            float(analysis.sweep_max),
            int(analysis.sweep_steps),
        )
        self.run_sweep(
            config,
            param_name,
            param_vals,
            on_success=on_success,
            on_error=on_error,
        )
        return param_name, param_vals

    def run_sd_curve_from_config(
        self,
        config,
        on_success: Callable[[Any], None] = None,
        on_error: Callable[[str], None] = None,
        on_progress: Callable[[str], None] = None,
    ) -> None:
        """Run S-D curve for the provided config."""
        self.run_sd_curve(
            config,
            on_success=on_success,
            on_error=on_error,
            on_progress=on_progress,
        )

    def run_excmap_from_config(
        self,
        config,
        on_success: Callable[[Any], None] = None,
        on_error: Callable[[str], None] = None,
        on_progress: Callable[[str], None] = None,
    ) -> None:
        """Run excitability map for the provided config."""
        self.run_excmap(
            config,
            on_success=on_success,
            on_error=on_error,
            on_progress=on_progress,
        )
        
    def run_single(self, config, on_success: Callable[[Any], None] = None,
                   on_error: Callable[[str], None] = None, on_progress: Callable[[str], None] = None,
                   compute_lyapunov: bool = False, lle_subspace_mode: int = 0):
        """Run single simulation in background thread with async analysis.
        
        The worker now performs post-processing and full_analysis in the background
        thread to eliminate GUI micro-freezes (Async Analytical Pipeline v11.8).
        """
        from core.solver import NeuronSolver
        from core.morphology import MorphologyBuilder
        from core.analysis import full_analysis
        config_snapshot = self._snapshot_config(config)
        
        self.simulation_started.emit()
        
        def run_simulation():
            solver = NeuronSolver(config_snapshot)
            if compute_lyapunov:
                result = solver.run_native(
                    config_snapshot,
                    calc_lle=True,
                    lle_delta=float(getattr(config_snapshot.analysis, "lle_delta", 1e-6)),
                    lle_t_evolve=float(getattr(config_snapshot.analysis, "lle_t_evolve_ms", 1.0)),
                    lle_subspace_mode=int(lle_subspace_mode),
                )
            else:
                result = solver.run_single()
            
            # ── Async Analytical Pipeline (v11.8) ──
            # Post-processing and analysis run in background thread
            # to keep GUI thread responsive (< 16 ms per frame)
            
            # Build morphology for post-processing
            morph = MorphologyBuilder.build(config_snapshot)
            
            # Post-process physics (current reconstruction, ATP estimates)
            solver._post_process_physics(result, morph)
            
            # Native Benettin LLE is already attached to result.lle_convergence.
            stats = full_analysis(result, compute_lyapunov=False)
            
            return {
                'single': result,
                'stats': stats,
                'morph': morph,
            }
            
        worker = Worker(run_simulation)
        
        # Connect to controller's public signals
        worker.signals.finished.connect(self.simulation_finished)
        worker.signals.error.connect(self.error_occurred)
        worker.signals.progress.connect(self.progress_updated)
        
        # Legacy callback support (for backward compatibility during transition)
        if on_success:
            worker.signals.finished.connect(on_success)
        if on_error:
            worker.signals.error.connect(on_error)
        if on_progress:
            worker.signals.progress.connect(lambda i, n, v: on_progress(f"Progress: {i}/{n}"))
            
        self.thread_pool.start(worker)

    def run_analytics(self, result, morph, config, on_success: Callable[[Any], None] = None,
                      on_error: Callable[[str], None] = None, compute_lyapunov: bool = False):
        """v13.0: Run analytics (full_analysis) in separate background thread.

        This allows immediate oscilloscope update while analytics processes.
        Matplotlib rendering happens only after analytics completes.
        """
        self.analytics_started.emit()
        config_snapshot = self._snapshot_config(config)

        from core.solver import NeuronSolver
        from core.analysis import full_analysis

        def run_analysis():
            # Post-process physics (current reconstruction, ATP estimates)
            solver = NeuronSolver(config_snapshot)
            solver._post_process_physics(result, morph)

            # Full analysis (spike detection, statistics, LLE if requested)
            stats = full_analysis(result, compute_lyapunov=compute_lyapunov)
            return stats

        worker = Worker(run_analysis)

        # Connect signals
        worker.signals.finished.connect(self.analytics_finished)
        if on_success:
            worker.signals.finished.connect(on_success)
        if on_error:
            worker.signals.error.connect(on_error)
            worker.signals.error.connect(self.error_occurred)

        self.thread_pool.start(worker)

    def run_stochastic(self, config, n_trials: int, on_success: Callable[[Any], None] = None,
                      on_error: Callable[[str], None] = None, on_progress: Callable[[str], None] = None,
                      compute_lyapunov: bool = False):
        """Run stochastic simulation through the primary solver path with async analysis."""
        self.simulation_started.emit()
        config_snapshot = self._snapshot_config(config)
        
        def run_simulation():
            from core.solver import NeuronSolver
            from core.morphology import MorphologyBuilder
            from core.analysis import full_analysis
            
            cfg = config_snapshot
            cfg.stim.stoch_gating = True
            solver = NeuronSolver(cfg)
            result = solver.run_single()
            
            # Async Analytical Pipeline (v11.8)
            morph = MorphologyBuilder.build(cfg)
            solver._post_process_physics(result, morph)
            stats = full_analysis(result, compute_lyapunov=compute_lyapunov)
            
            return {
                'single': result,
                'stats': stats,
                'morph': morph,
            }
            
        worker = Worker(run_simulation)
        
        # Connect to controller's public signals
        worker.signals.finished.connect(self.simulation_finished)
        worker.signals.error.connect(self.error_occurred)
        worker.signals.progress.connect(self.progress_updated)
        
        # Legacy callback support
        if on_success:
            worker.signals.finished.connect(on_success)
        if on_error:
            worker.signals.error.connect(on_error)
        if on_progress:
            worker.signals.progress.connect(lambda i, n, v: on_progress(f"Progress: {i}/{n}"))
            
        self.thread_pool.start(worker)
        
    def run_monte_carlo(self, config, n_trials: int, on_success: Callable[[Any], None] = None,
                        on_error: Callable[[str], None] = None, on_progress: Callable[[str], None] = None):
        """Run Monte-Carlo simulation."""
        self.simulation_started.emit()
        config_snapshot = self._snapshot_config(config)
        
        from core.solver import NeuronSolver
        
        def run_simulation():
            solver = NeuronSolver(config_snapshot)
            return {'mc_results': solver.run_mc(n_trials)}
            
        worker = Worker(run_simulation)
        
        # Connect to controller's public signals
        worker.signals.finished.connect(self.simulation_finished)
        worker.signals.error.connect(self.error_occurred)
        worker.signals.progress.connect(self.progress_updated)
        
        # Legacy callback support
        if on_success:
            worker.signals.finished.connect(on_success)
        if on_error:
            worker.signals.error.connect(on_error)
        if on_progress:
            worker.signals.progress.connect(lambda i, n, v: on_progress(f"Progress: {i}/{n}"))
            
        self.thread_pool.start(worker)
        
    def run_sweep(self, config, param_name: str, param_range, on_success: Callable[[Any], None] = None,
                  on_error: Callable[[str], None] = None, on_progress: Callable[[str], None] = None):
        """Run parametric sweep."""
        self.simulation_started.emit()
        config_snapshot = self._snapshot_config(config)
        if isinstance(param_range, np.ndarray):
            param_range_snapshot = np.array(param_range, copy=True)
        else:
            try:
                param_range_snapshot = list(param_range)
            except TypeError:
                param_range_snapshot = param_range

        def run_simulation():
            from core.advanced_sim import run_sweep
            return run_sweep(config_snapshot, param_name, param_range_snapshot)
            
        worker = Worker(run_simulation)
        
        # Connect to controller's public signals
        worker.signals.finished.connect(self.simulation_finished)
        worker.signals.error.connect(self.error_occurred)
        worker.signals.progress.connect(self.progress_updated)
        
        # Legacy callback support
        if on_success:
            worker.signals.finished.connect(on_success)
        if on_error:
            worker.signals.error.connect(on_error)
        if on_progress:
            worker.signals.progress.connect(lambda i, n, v: on_progress(f"Progress: {i}/{n}"))
            
        self.thread_pool.start(worker)
        
    def run_sd_curve(self, config, on_success: Callable[[Any], None] = None,
                     on_error: Callable[[str], None] = None, on_progress: Callable[[str], None] = None):
        """Run Strength-Duration curve."""
        self.simulation_started.emit()
        config_snapshot = self._snapshot_config(config)

        def run_simulation():
            from core.advanced_sim import run_sd_curve
            return run_sd_curve(config_snapshot)
            
        worker = Worker(run_simulation)
        
        # Connect to controller's public signals
        worker.signals.finished.connect(self.simulation_finished)
        worker.signals.error.connect(self.error_occurred)
        worker.signals.progress.connect(self.progress_updated)
        
        # Legacy callback support
        if on_success:
            worker.signals.finished.connect(on_success)
        if on_error:
            worker.signals.error.connect(on_error)
        if on_progress:
            worker.signals.progress.connect(lambda i, n, v: on_progress(f"Progress: {i}/{n}"))
            
        self.thread_pool.start(worker)
        
    def run_excmap(self, config, on_success: Callable[[Any], None] = None,
                   on_error: Callable[[str], None] = None, on_progress: Callable[[str], None] = None):
        """Run excitability map."""
        self.simulation_started.emit()
        config_snapshot = self._snapshot_config(config)

        def run_simulation():
            from core.advanced_sim import run_excitability_map
            return run_excitability_map(config_snapshot)
            
        worker = Worker(run_simulation)
        
        # Connect to controller's public signals
        worker.signals.finished.connect(self.simulation_finished)
        worker.signals.error.connect(self.error_occurred)
        worker.signals.progress.connect(self.progress_updated)
        
        # Legacy callback support
        if on_success:
            worker.signals.finished.connect(on_success)
        if on_error:
            worker.signals.error.connect(on_error)
        if on_progress:
            worker.signals.progress.connect(lambda i, n, v: on_progress(f"Progress: {i}/{n}"))
            
        self.thread_pool.start(worker)
        
    def run_rheobase(self, config, on_progress: Callable[[str, dict], None] = None,
                     on_success: Callable[[Any], None] = None,
                     on_error: Callable[[str], None] = None):
        """v13.0: Run non-blocking Auto-Rheobase search in worker thread.

        Performs binary search to find minimum I_ext required to trigger a spike.
        Emits progress updates through signals (thread-safe).
        """
        self.simulation_started.emit()
        config_snapshot = self._snapshot_config(config)

        from core.solver import NeuronSolver
        from core.analysis import detect_spikes

        def run_search(progress_signal):
            """Worker function that emits progress through signal."""
            search_history = []
            I_low, I_high = 0.0, 100.0

            cfg = config_snapshot
            cfg.stim.t_sim = 50.0  # Short simulation
            solver = NeuronSolver(cfg)

            # Phase 1: Find upper bound
            for attempt in range(10):
                cfg.stim.Iext = I_high
                try:
                    result = solver.run_single(cfg)
                    pk_idx, spike_times, _ = detect_spikes(result.v_soma, result.t, threshold=-20.0)
                    if len(spike_times) > 0:
                        search_history.append((0.0, I_high, I_high, True))
                        break
                except Exception:
                    pass
                I_high *= 2.0
                search_history.append((0.0, I_high, I_high, False))
                if I_high > 1000.0:
                    raise RuntimeError("Could not find upper bound for rheobase")
                # Thread-safe: emit progress through signal
                progress_signal.emit(
                    f"Phase 1: Expanding bound... I_high = {I_high:.1f}",
                    {'phase': 1, 'I_high': I_high, 'attempt': attempt}
                )

            # Phase 2: Binary search
            for iteration in range(10):
                I_mid = (I_low + I_high) / 2.0
                cfg.stim.Iext = I_mid

                try:
                    result = solver.run_single(cfg)
                    pk_idx, spike_times, _ = detect_spikes(result.v_soma, result.t, threshold=-20.0)
                    has_spike = len(spike_times) > 0
                except Exception:
                    has_spike = False

                search_history.append((I_low, I_high, I_mid, has_spike))

                if has_spike:
                    I_high = I_mid
                else:
                    I_low = I_mid

                # Thread-safe: emit progress through signal
                progress_signal.emit(
                    f"Iteration {iteration+1}/10: I = {I_mid:.2f}, spike={'YES' if has_spike else 'NO'}",
                    {'phase': 2, 'iteration': iteration+1, 'I_mid': I_mid,
                     'I_low': I_low, 'I_high': I_high, 'has_spike': has_spike}
                )

            I_rheobase = (I_low + I_high) / 2.0
            uncertainty = (I_high - I_low) / 2.0

            return {
                'I_rheobase': I_rheobase,
                'uncertainty': uncertainty,
                'search_history': search_history,
            }

        # Custom worker class to pass signal to run function
        class RheobaseWorker(QRunnable):
            def __init__(self, func, signals):
                super().__init__()
                self.func = func
                self.signals = signals

            def run(self):
                try:
                    result = self.func(self.signals.progress_rich)
                    self.signals.finished.emit(result)
                except Exception as e:
                    logger.error(f"Rheobase worker failed: {e}")
                    self.signals.error.emit(str(e))

        signals = WorkerSignals()
        worker = RheobaseWorker(run_search, signals)

        # Connect signals
        # Note: rheobase results are NOT emitted to simulation_finished
        # because they have a different payload structure (I_rheobase, uncertainty, etc.)
        # Only use on_success callback for rheobase results
        if on_success:
            worker.signals.finished.connect(on_success)
        worker.signals.error.connect(self.error_occurred)
        if on_error:
            worker.signals.error.connect(on_error)
        # v13.0: Thread-safe progress routing through controller signal
        worker.signals.progress_rich.connect(self.rheobase_progress)
        if on_progress:
            worker.signals.progress_rich.connect(on_progress)

        self.thread_pool.start(worker)

    def shutdown(self):
        """Wait for all threads to finish."""
        self.thread_pool.waitForDone(5000)  # Wait up to 5 seconds
