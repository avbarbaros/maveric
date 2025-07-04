"""Core components and base classes for MAVERIC."""

from .base import BaseComponent, BaseDataset, BaseMetric
from .interfaces import (
    RetrievalResult,
    QualityResult,
    CustomizationResult,
    ProgressCallback
)
from .exceptions import (
    MAVERICError,
    ConfigurationError,
    DatasetError,
    ModelError,
    CacheError
)

__all__ = [
    "BaseComponent",
    "BaseDataset",
    "BaseMetric",
    "RetrievalResult",
    "QualityResult",
    "CustomizationResult",
    "ProgressCallback",
    "MAVERICError",
    "ConfigurationError",
    "DatasetError",
    "ModelError",
    "CacheError"
]