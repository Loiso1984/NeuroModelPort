"""
Stochastic RNG Manager for NeuroModelPort v10.0

Centralized random number generation for reproducible stochastic simulations.
Integrates with PhysicsParams for consistent stochastic behavior across all components.
"""

import numpy as np
from typing import Optional, Union
from dataclasses import dataclass


@dataclass
class StochasticRNG:
    """Centralized random number generator for stochastic simulations."""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize RNG with optional seed for reproducibility.
        
        Args:
            seed: Random seed for reproducible simulations
        """
        self.seed = seed
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()
    
    def reset(self, seed: Optional[int] = None):
        """Reset RNG with new seed."""
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
        else:
            self.seed = None
            self.rng = np.random.default_rng()
    
    def get_state(self) -> dict:
        """Get current RNG state for saving/loading."""
        return {
            'seed': self.seed,
            'state': self.rng.bit_generator.state
        }
    
    def set_state(self, state: dict):
        """Restore RNG from saved state."""
        self.seed = state['seed']
        self.rng = np.random.default_rng(self.seed)
        try:
            self.rng.bit_generator.state = state['state']
        except (AttributeError, ValueError, TypeError) as e:
            # Fallback: recreate RNG with seed if state restoration fails
            # This ensures compatibility across different NumPy versions
            import warnings
            warnings.warn(f"RNG state restoration failed: {e}. Using seed-only restoration.", UserWarning)
            self.rng = np.random.default_rng(self.seed)
    
    # Convenience methods for common distributions
    def normal(self, loc: float = 0.0, scale: float = 1.0, size: Optional[Union[int, tuple]] = None) -> np.ndarray:
        """Normal distribution N(loc, scale²)."""
        return self.rng.normal(loc, scale, size)
    
    def randn(self, *shape: int) -> np.ndarray:
        """Standard normal N(0, 1)."""
        return self.rng.standard_normal(shape)
    
    def uniform(self, low: float = 0.0, high: float = 1.0, size: Optional[Union[int, tuple]] = None) -> np.ndarray:
        """Uniform distribution U(low, high)."""
        return self.rng.uniform(low, high, size)
    
    def choice(self, a: np.ndarray, size: Optional[int] = None, replace: bool = True) -> np.ndarray:
        """Random choice from array."""
        return self.rng.choice(a, size=size, replace=replace)
    
    def exponential(self, scale: float = 1.0, size: Optional[Union[int, tuple]] = None) -> np.ndarray:
        """Exponential distribution."""
        return self.rng.exponential(scale, size)


# Global RNG instance (can be overridden by PhysicsParams)
_GLOBAL_RNG: Optional[StochasticRNG] = None


def get_rng() -> StochasticRNG:
    """Get global stochastic RNG instance."""
    global _GLOBAL_RNG
    if _GLOBAL_RNG is None:
        _GLOBAL_RNG = StochasticRNG()  # Default: no seed
    return _GLOBAL_RNG


def set_rng(rng: StochasticRNG):
    """Set global stochastic RNG instance."""
    global _GLOBAL_RNG
    _GLOBAL_RNG = rng


def reset_rng(seed: Optional[int] = None):
    """Reset global RNG with new seed."""
    global _GLOBAL_RNG
    _GLOBAL_RNG = StochasticRNG(seed)


def seed_all(seed: int):
    """Convenience function to seed all stochastic components."""
    reset_rng(seed)
    # Also seed numpy's legacy random for compatibility
    np.random.seed(seed)
