"""
gui/simulation_controller.py - Thread Management for Simulations

Separates PyQt thread management from MainWindow UI logic.
Keeps QRunnable, QThreadPool, and Worker classes in the GUI package.
"""

from PySide6.QtCore import QObject, Signal, QThreadPool, QRunnable
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)


class WorkerSignals(QObject):
    """Signals for worker thread communication."""
    finished = Signal(object)  # SimulationResult
    error = Signal(str)        # Error message
    progress = Signal(str)     # Status update


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
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(4)  # Limit concurrent simulations
        
    def run_single(self, config, on_success: Callable[[Any], None], 
                   on_error: Callable[[str], None], on_progress: Callable[[str], None] = None):
        """Run single simulation in background thread."""
        from core.solver import NeuronSolver
        
        def run_simulation():
            solver = NeuronSolver(config)
            return solver.run_single()
            
        worker = Worker(run_simulation)
        
        # Connect signals
        worker.signals.finished.connect(on_success)
        worker.signals.error.connect(on_error)
        if on_progress:
            worker.signals.progress.connect(on_progress)
            
        self.thread_pool.start(worker)
        
    def run_stochastic(self, config, n_trials: int, on_success: Callable[[Any], None],
                      on_error: Callable[[str], None], on_progress: Callable[[str], None] = None):
        """Run stochastic Euler-Maruyama simulation."""
        def run_simulation():
            from core.advanced_sim import run_euler_maruyama
            return run_euler_maruyama(config)
            
        worker = Worker(run_simulation)
        
        # Connect signals
        worker.signals.finished.connect(on_success)
        worker.signals.error.connect(on_error)
        if on_progress:
            worker.signals.progress.connect(on_progress)
            
        self.thread_pool.start(worker)
        
    def run_monte_carlo(self, config, n_trials: int, on_success: Callable[[Any], None],
                        on_error: Callable[[str], None], on_progress: Callable[[str], None] = None):
        """Run Monte-Carlo simulation."""
        from core.solver import NeuronSolver
        
        def run_simulation():
            solver = NeuronSolver(config)
            return solver.run_mc(n_trials)
            
        worker = Worker(run_simulation)
        
        # Connect signals
        worker.signals.finished.connect(on_success)
        worker.signals.error.connect(on_error)
        if on_progress:
            worker.signals.progress.connect(on_progress)
            
        self.thread_pool.start(worker)
        
    def run_sweep(self, config, param_name: str, param_range, on_success: Callable[[Any], None],
                  on_error: Callable[[str], None], on_progress: Callable[[str], None] = None):
        """Run parametric sweep."""
        from core.solver import NeuronSolver
        
        def run_simulation():
            from core.advanced_sim import run_sweep
            return run_sweep(config, param_name, param_range)
            
        worker = Worker(run_simulation)
        
        # Connect signals
        worker.signals.finished.connect(on_success)
        worker.signals.error.connect(on_error)
        if on_progress:
            worker.signals.progress.connect(on_progress)
            
        self.thread_pool.start(worker)
        
    def run_sd_curve(self, config, on_success: Callable[[Any], None],
                     on_error: Callable[[str], None], on_progress: Callable[[str], None] = None):
        """Run Strength-Duration curve."""
        from core.solver import NeuronSolver
        
        def run_simulation():
            from core.advanced_sim import run_sd_curve
            return run_sd_curve(config)
            
        worker = Worker(run_simulation)
        
        # Connect signals
        worker.signals.finished.connect(on_success)
        worker.signals.error.connect(on_error)
        if on_progress:
            worker.signals.progress.connect(on_progress)
            
        self.thread_pool.start(worker)
        
    def run_excmap(self, config, on_success: Callable[[Any], None],
                   on_error: Callable[[str], None], on_progress: Callable[[str], None] = None):
        """Run excitability map."""
        from core.solver import NeuronSolver
        
        def run_simulation():
            from core.advanced_sim import run_excitability_map
            return run_excitability_map(config)
            
        worker = Worker(run_simulation)
        
        # Connect signals
        worker.signals.finished.connect(on_success)
        worker.signals.error.connect(on_error)
        if on_progress:
            worker.signals.progress.connect(on_progress)
            
        self.thread_pool.start(worker)
        
    def shutdown(self):
        """Wait for all threads to finish."""
        self.thread_pool.waitForDone(5000)  # Wait up to 5 seconds
