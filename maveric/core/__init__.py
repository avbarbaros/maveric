"""Core components and base classes for MAVERIC.

This module serves as the package initialization and public API gateway for the core module. It:

1. Exposes core interfaces: Imports and exports base classes, result types, and callback interfaces
2. Centralizes exception handling: Makes all MAVERIC-specific exceptions available from one import
3. Defines public API: The __all__ list explicitly controls which components are exported when someone imports from maveric.core
4. Simplifies imports: Allows other parts of the codebase to do `from maveric.core import BaseComponent` instead of `from maveric.core.base import BaseComponent`
"""

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
from .progress import RealTimeStats

__all__ = [
    "BaseComponent",
    "BaseDataset",
    "BaseMetric",
    "RetrievalResult",
    "QualityResult",
    "CustomizationResult",
    "ProgressCallback",
    "RealTimeStats",
    "MAVERICError",
    "ConfigurationError",
    "DatasetError",
    "ModelError",
    "CacheError"
]