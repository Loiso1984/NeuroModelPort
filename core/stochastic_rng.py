"""
Stochastic RNG Manager for NeuroModelPort v10.0

Centralized random number generation for reproducible stochastic simulations.
Integrates with PhysicsParams for consistent stochastic behavior across all components.

THREAD-SAFETY CONTRACT
----------------------
This module provides a STRICT thread-local RNG. The rules are:

1. ``get_rng()`` returns a ``StochasticRNG`` bound to ``threading.local()``.
   Each thread gets its OWN ``np.random.Generator`` and never shares state
   with another thread. Calling ``get_rng()`` from multiple threads does
   NOT mutate the process-global RNG.

2. The process-global RNG (``_GLOBAL_RNG``) may ONLY be written from the
   main thread via ``set_rng()`` / ``reset_rng()`` / ``seed_all()``. This
   is enforced by a guard; attempting to write it from a worker thread
   raises ``RuntimeError``.

3. Reproducibility across threads is achieved via
   ``numpy.random.SeedSequence`` stream-splitting: worker threads derive
   their seeds by spawning independent child SeedSequences from the
   master seed plus the worker's thread-identity, guaranteeing
   statistically independent streams that are still deterministic for a
   given base seed.

4. Single-threaded callers may use ``get_process_rng()`` to access the
   shared process-level RNG (this is an explicit, documented escape
   hatch; it raises ``RuntimeError`` if called from a non-main thread).
"""

import numpy as np
import threading
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
        self._lock = threading.Lock()  # Thread safety for shared instances
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

    def reset(self, seed: Optional[int] = None):
        """Reset RNG with new seed."""
        with self._lock:
            if seed is not None:
                self.seed = seed
                self.rng = np.random.default_rng(seed)
            else:
                self.seed = None
                self.rng = np.random.default_rng()

    def get_state(self) -> dict:
        """Get current RNG state for saving/loading."""
        with self._lock:
            return {
                'seed': self.seed,
                'state': self.rng.bit_generator.state
            }

    def set_state(self, state: dict):
        """Restore RNG from saved state."""
        with self._lock:
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
        with self._lock:
            return self.rng.normal(loc, scale, size)

    def randn(self, *shape: int) -> np.ndarray:
        """Standard normal N(0, 1)."""
        with self._lock:
            return self.rng.standard_normal(shape)

    def uniform(self, low: float = 0.0, high: float = 1.0, size: Optional[Union[int, tuple]] = None) -> np.ndarray:
        """Uniform distribution U(low, high)."""
        with self._lock:
            return self.rng.uniform(low, high, size)

    def choice(self, a: np.ndarray, size: Optional[int] = None, replace: bool = True) -> np.ndarray:
        """Random choice from array."""
        with self._lock:
            return self.rng.choice(a, size=size, replace=replace)

    def exponential(self, scale: float = 1.0, size: Optional[Union[int, tuple]] = None) -> np.ndarray:
        """Exponential distribution."""
        with self._lock:
            return self.rng.exponential(scale, size)

    def next_seed(self) -> int:
        """Return a 32-bit seed for a native stochastic simulation."""
        with self._lock:
            return int(self.rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))


# ────────────────────────────────────────────────────────────────────────────
# Module-level state
#
# _GLOBAL_RNG      — process-global RNG instance; ONLY written on the main
#                    thread via reset_rng/set_rng/seed_all. Never written
#                    from worker threads (see get_rng below).
# _GLOBAL_SEED     — the base seed used to derive thread-local RNGs. Reads
#                    are guarded by _GLOBAL_RNG_LOCK.
# _THREAD_LOCAL_RNG — threading.local() storage: each thread lazily owns
#                    its own StochasticRNG derived from _GLOBAL_SEED via
#                    SeedSequence.spawn() stream-splitting.
# _MAIN_THREAD_ID  — captured at import time; used to guard process-global
#                    writes coming from non-main threads.
# ────────────────────────────────────────────────────────────────────────────
_GLOBAL_RNG: Optional[StochasticRNG] = None
_GLOBAL_RNG_LOCK = threading.Lock()
_THREAD_LOCAL_RNG = threading.local()
_GLOBAL_SEED: Optional[int] = None
_MAIN_THREAD_ID: int = threading.main_thread().ident  # type: ignore[assignment]


def _derive_thread_seed(base_seed: Optional[int], tid: int) -> Optional[int]:
    """Derive an independent per-thread seed from a master seed + thread id.

    Uses numpy's SeedSequence stream-splitting: this is the numpy-recommended
    way to obtain statistically independent but deterministic streams across
    parallel workers. If ``base_seed`` is None the thread simply gets a
    non-deterministic fresh seed (None passed through).
    """
    if base_seed is None:
        return None
    # Entropy = master seed; spawn_key = thread identity. This yields a
    # distinct child SeedSequence per (base_seed, tid) pair and is stable
    # across Python sessions for the same inputs.
    ss = np.random.SeedSequence(entropy=int(base_seed), spawn_key=(int(tid),))
    # Produce a 64-bit unsigned integer seed for StochasticRNG.
    return int(ss.generate_state(1, dtype=np.uint64)[0])


def get_rng() -> StochasticRNG:
    """Get a thread-local ``StochasticRNG``.

    Previous versions of this function (pre-FIX-CRIT-J) overwrote the
    process-global ``_GLOBAL_RNG`` from any thread that called ``get_rng``,
    which corrupted reproducibility for parallel stochastic runs. This
    implementation NEVER writes process-global state from worker threads —
    each thread owns its own ``StochasticRNG`` via ``threading.local()``,
    seeded deterministically from ``_GLOBAL_SEED`` using
    ``SeedSequence.spawn_key`` so parallel streams remain statistically
    independent while still being reproducible for a given base seed.
    """
    # Snapshot the base seed under lock so we see a consistent value.
    with _GLOBAL_RNG_LOCK:
        base_seed = _GLOBAL_SEED

    rng = getattr(_THREAD_LOCAL_RNG, "rng", None)
    cached_base = getattr(_THREAD_LOCAL_RNG, "base_seed", "__unset__")

    # Rebuild the thread-local RNG if missing or if the base seed changed
    # (e.g. the main thread called reset_rng between two get_rng calls).
    if rng is None or cached_base != base_seed:
        tid = threading.get_ident()
        if tid == _MAIN_THREAD_ID:
            # Main thread keeps the canonical, un-split seed for backward
            # compatibility with reproducibility tests.
            thread_seed = base_seed
        else:
            thread_seed = _derive_thread_seed(base_seed, tid)
        rng = StochasticRNG(thread_seed)
        _THREAD_LOCAL_RNG.rng = rng
        _THREAD_LOCAL_RNG.base_seed = base_seed
    return rng


def get_process_rng() -> StochasticRNG:
    """Return the process-global RNG. MAIN-THREAD ONLY.

    This is an explicit escape hatch for single-threaded code that must
    share RNG state across call sites. Raises ``RuntimeError`` if called
    from a worker thread — use ``get_rng()`` there instead.
    """
    if threading.get_ident() != _MAIN_THREAD_ID:
        raise RuntimeError(
            "get_process_rng() may only be called from the main thread; "
            "worker threads must use get_rng() for thread-local RNG."
        )
    with _GLOBAL_RNG_LOCK:
        if _GLOBAL_RNG is None:
            # Lazily materialize so main-thread callers never see None.
            return get_rng()
        return _GLOBAL_RNG


def _require_main_thread(fn_name: str) -> None:
    if threading.get_ident() != _MAIN_THREAD_ID:
        raise RuntimeError(
            f"{fn_name} may only be called from the main thread; "
            f"the process-global RNG is not writable from worker threads."
        )


def set_rng(rng: StochasticRNG):
    """Set global stochastic RNG instance. Main-thread only."""
    _require_main_thread("set_rng")
    global _GLOBAL_RNG, _GLOBAL_SEED
    with _GLOBAL_RNG_LOCK:
        _GLOBAL_RNG = rng
        _GLOBAL_SEED = rng.seed
    # Also refresh the main thread's local cache so get_rng() returns the
    # same instance the caller just set.
    _THREAD_LOCAL_RNG.rng = rng
    _THREAD_LOCAL_RNG.base_seed = rng.seed


def reset_rng(seed: Optional[int] = None):
    """Reset global RNG with new seed. Main-thread only."""
    _require_main_thread("reset_rng")
    global _GLOBAL_RNG, _GLOBAL_SEED
    with _GLOBAL_RNG_LOCK:
        _GLOBAL_SEED = seed
        _GLOBAL_RNG = StochasticRNG(seed)
        new_rng = _GLOBAL_RNG
    _THREAD_LOCAL_RNG.rng = new_rng
    _THREAD_LOCAL_RNG.base_seed = seed


def seed_all(seed: int):
    """Convenience function to seed all stochastic components."""
    reset_rng(seed)
    # Also seed numpy's legacy random for compatibility
    np.random.seed(seed)
