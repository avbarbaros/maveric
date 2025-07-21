"""Custom exceptions for MAVERIC."""

class MAVERICError(Exception):
    """
    Base exception class that inherits from Python's built-in Exception.
    Base exception for all MAVERIC-specific errors.
    """
    pass  # No additional implementation needed, inherits all functionality from Exception


class ConfigurationError(MAVERICError):
    """
    Inherits from MAVERICError (creating an exception hierarchy).
    Raised when configuration is invalid or missing.
    
    This includes issues with YAML files, environment variables, or other config sources.
    Provides a clear error message for configuration-related failures.
    Usage:
    raise ConfigurationError("Invalid configuration for the model.")    
    """
    pass  # Inherits error handling behavior from parent classes


class DatasetError(MAVERICError):
    """
    Exception class for dataset-related failures.
    Raised when dataset loading, processing, or access operations fail.
    """
    pass  # Inherits error handling behavior from parent classes


class ModelError(MAVERICError):
    """
    Exception class for model-related operations.
    Raised when model loading, inference, or training operations fail.
    """
    pass  # Inherits standard exception behavior


class CacheError(MAVERICError):
    """
    Exception class for cache system failures.
    Raised when cache read/write/management operations fail.
    """
    pass  # Standard exception inheritance


class MetricError(MAVERICError):
    """
    Exception class for quality metric computation failures.
    Raised when metric calculations fail.
    """
    pass  # No custom behavior needed


class VisualizationError(MAVERICError):
    """
    Exception class for visualization and plotting failures.
    Raised when plot generation and display operations fail.
    """
    pass  # Inherits exception functionality


# Exception Hierarchy:
# Exception (Python built-in)
# └── MAVERICError (base for all MAVERIC exceptions)
#     ├── ConfigurationError
#     ├── DatasetError
#     ├── ModelError
#     ├── CacheError
#     ├── MetricError
#     └── VisualizationError
#
# This allows developers to:
# - Catch all MAVERIC errors with: except MAVERICError
# - Catch specific error types with: except ConfigurationError
# - Create meaningful error messages for different failure scenarios