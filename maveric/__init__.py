"""
MAVERIC: Multi-modal Adaptive Visual Embedding Retrieval with Integrated Consistency

A sophisticated quality control system for multi-modal dataset curation.
"""

from .config import MAVERICConfig, TrainingConfig
from .core.exceptions import MAVERICError
from .main import MAVERIC

__version__ = "0.1.0"
__all__ = ["MAVERIC", "MAVERICConfig", "TrainingConfig", "MAVERICError"]
__author__ = "Ali V. Barbaros"