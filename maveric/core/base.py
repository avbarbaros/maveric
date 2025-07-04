"""Base classes for MAVERIC components."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import logging
from PIL import Image
import numpy as np
import pandas as pd


class BaseComponent(ABC):
    """
    Base class for all MAVERIC components.
    Provides common functionality like logging and configuration.
    """
    
    def __init__(self, name: str = None):
        """
        Initialize base component with logging capabilities.
        
        Args:
            name: Component name for logging (defaults to class name)
        """
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"maveric.{self.name}")
        
    def log_info(self, message: str):
        """Log an info message."""
        self.logger.info(message)
        
    def log_warning(self, message: str):
        """Log a warning message."""
        self.logger.warning(message)
        
    def log_error(self, message: str):
        """Log an error message."""
        self.logger.error(message)
        
    def log_debug(self, message: str):
        """Log a debug message."""
        self.logger.debug(message)


class BaseDataset(BaseComponent):
    """
    Abstract base class for dataset handlers.
    All dataset implementations should inherit from this class.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the dataset name."""
        pass
    
    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        """Return list of class names in the dataset."""
        pass
    
    @property
    def num_classes(self) -> int:
        """Return number of classes in the dataset."""
        return len(self.class_names)
    
    @abstractmethod
    def get_reference_samples(self, n_per_class: int) -> Dict[str, List[Image.Image]]:
        """
        Get reference samples for each class.
        
        Args:
            n_per_class: Number of reference samples per class
            
        Returns:
            Dictionary mapping class names to lists of PIL images
        """
        pass
    
    @abstractmethod
    def get_text_templates(self) -> List[str]:
        """
        Get text templates for creating prompts.
        
        Returns:
            List of template strings with {} placeholder for class name
        """
        pass
    
    def create_prompts(self, class_name: str) -> List[str]:
        """
        Create prompts for a specific class using templates.
        
        Args:
            class_name: Name of the class
            
        Returns:
            List of prompts for the class
        """
        templates = self.get_text_templates()
        return [template.format(class_name) for template in templates]


class BaseMetric(BaseComponent):
    """
    Abstract base class for quality metrics.
    All quality metric implementations should inherit from this class.
    """
    
    def __init__(self, name: str = None, weight: float = 1.0):
        """
        Initialize metric with name and weight.
        
        Args:
            name: Metric name (defaults to class name)
            weight: Weight for this metric in composite scores
        """
        super().__init__(name)
        self.weight = weight
        self._cache = {}
        
    @property
    @abstractmethod
    def metric_name(self) -> str:
        """Return unique name for this metric."""
        pass
    
    @property
    def requires_reference(self) -> bool:
        """Whether this metric requires reference samples."""
        return False
    
    @abstractmethod
    def compute(self, image: Image.Image, metadata: Dict[str, Any]) -> float:
        """
        Compute quality score for an image.
        
        Args:
            image: PIL Image to evaluate
            metadata: Additional metadata (url, caption, etc.)
            
        Returns:
            Quality score (typically normalized to 0-1 range)
        """
        pass
    
    def batch_compute(self, 
                     images: List[Image.Image], 
                     metadata_list: List[Dict[str, Any]]) -> List[float]:
        """
        Compute scores for multiple images.
        Can be overridden for more efficient batch processing.
        
        Args:
            images: List of PIL images
            metadata_list: List of metadata dictionaries
            
        Returns:
            List of quality scores
        """
        scores = []
        for img, meta in zip(images, metadata_list):
            try:
                score = self.compute(img, meta)
                scores.append(score)
            except Exception as e:
                self.log_warning(f"Error computing {self.metric_name} for image: {e}")
                scores.append(0.0)
        return scores
    
    def normalize_score(self, score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """
        Normalize score to standard range.
        
        Args:
            score: Raw score value
            min_val: Minimum expected value
            max_val: Maximum expected value
            
        Returns:
            Normalized score in 0-1 range
        """
        if max_val == min_val:
            return 0.5
        normalized = (score - min_val) / (max_val - min_val)
        return np.clip(normalized, 0.0, 1.0)
