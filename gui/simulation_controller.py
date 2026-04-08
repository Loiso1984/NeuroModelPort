"""
gui/simulation_controller.py - Thread Management for Simulations

Separates PyQt thread management from MainWindow UI logic.
Keeps QRunnable, QThreadPool, and Worker classes in the GUI package.
"""

from PySide6.QtCore import QObject, Signal, QThreadPool, QRunnable
from typing import Callable, Any, Optional
import logging

logger = logging.getLogger(__name__)


class WorkerSignals(QObject):
    """Signals for worker thread communication."""
    finished = Signal(object)  # SimulationResult
    error = Signal(str)        # Error message
    progress = Signal(int, int, float)  # (current, total, value) for progress updates


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
        progress_updated: Emitted with (current, total, value) for progress
        error_occurred: Emitted when an error occurs
    """
    
    # Public signals for MainWindow to connect to
    simulation_started = Signal()
    simulation_finished = Signal(object)  # result dict
    progress_updated = Signal(int, int, float)  # (current, total, value)
    error_occurred = Signal(str)  # error message
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(4)  # Limit concurrent simulations
        
    def run_single(self, config, on_success: Callable[[Any], None] = None,
                   on_error: Callable[[str], None] = None, on_progress: Callable[[str], None] = None):
        """Run single simulation in background thread."""
        from core.solver import NeuronSolver
        
        self.simulation_started.emit()
        
        def run_simulation():
            solver = NeuronSolver(config)
            return {'single': solver.run_single()}
            
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
        
    def run_stochastic(self, config, n_trials: int, on_success: Callable[[Any], None] = None,
                      on_error: Callable[[str], None] = None, on_progress: Callable[[str], None] = None):
        """Run stochastic Euler-Maruyama simulation."""
        self.simulation_started.emit()
        
        def run_simulation():
            from core.advanced_sim import run_euler_maruyama
            return {'single': run_euler_maruyama(config)}
            
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
        
        from core.solver import NeuronSolver
        
        def run_simulation():
            solver = NeuronSolver(config)
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
        
        from core.solver import NeuronSolver
        
        def run_simulation():
            from core.advanced_sim import run_sweep
            return run_sweep(config, param_name, param_range)
            
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
        
        from core.solver import NeuronSolver
        
        def run_simulation():
            from core.advanced_sim import run_sd_curve
            return run_sd_curve(config)
            
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
        
        from core.solver import NeuronSolver
        
        def run_simulation():
            from core.advanced_sim import run_excitability_map
            return run_excitability_map(config)
            
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
        
    def shutdown(self):
        """Wait for all threads to finish."""
        self.thread_pool.waitForDone(5000)  # Wait up to 5 seconds
