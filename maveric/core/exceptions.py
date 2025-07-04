"""Custom exceptions for MAVERIC."""


class MAVERICError(Exception):
    """Base exception for all MAVERIC errors."""
    pass


class ConfigurationError(MAVERICError):
    """Raised when configuration is invalid or missing."""
    pass


class DatasetError(MAVERICError):
    """Raised when dataset operations fail."""
    pass


class ModelError(MAVERICError):
    """Raised when model operations fail."""
    pass


class CacheError(MAVERICError):
    """Raised when cache operations fail."""
    pass


class MetricError(MAVERICError):
    """Raised when metric computation fails."""
    pass


class VisualizationError(MAVERICError):
    """Raised when visualization operations fail."""
    pass