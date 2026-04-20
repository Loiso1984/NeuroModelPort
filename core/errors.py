"""Custom errors for simulation configuration/runtime handling."""


class SimulationParameterError(ValueError):
    """Raised when simulation parameters are invalid for execution."""


class PhysicsIntegrityError(RuntimeError):
    """Raised when a physics parameter violates a known physiological constraint.

    Use when a value is structurally valid (correct type/range) but biophysically
    implausible — e.g., a conductance that would make the cell permanently inactive.
    """


class SolverDivergenceError(RuntimeError):
    """Raised when the numerical solver detects divergence or NaN in the state vector.

    Signals that the current parameter set or timestep is causing the integration
    to become unstable, not that the parameters themselves are wrong.
    """


class ChannelConfigError(ValueError):
    """Raised when a channel is enabled but its required parameters are inconsistent.

    More specific than SimulationParameterError — reserved for channel-level conflicts
    (e.g., enabling a channel whose gate kinetics are undefined for the current model).
    """


class MorphologyError(ValueError):
    """Raised when compartment layout or cable parameters are geometrically invalid."""


class PresetLoadError(IOError):
    """Raised when a preset file cannot be loaded or fails schema validation."""
