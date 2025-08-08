from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
# Python's standard logging framework
import logging
# Image processing library for handling images
from PIL import Image
# Data processing libraries
import numpy as np
import pandas as pd

"""
Base classes for MAVERIC components.

Import necessary modules for abstract base classes and type hints
ABC (Abstract Base Class) is used to create abstract base classes that:
1. Provide the foundation for creating abstract classes that cannot be instantiated directly
2. Interface Definition: Forces subclasses to implement required methods marked with @abstractmethod
3. Contract Enforcement: Ensures all dataset handlers and metrics follow the same interface
4. Runtime Validation: Python will raise TypeError if you try to instantiate a class that inherits from ABC but doesn't implement all abstract methods
"""

class BaseComponent(ABC):
    """
    Abstract base class that all MAVERIC components inherit from.
    Provides common functionality like logging and configuration.
    """
    
    def __init__(self, name: str = None):
        """
        Initialize base component with logging capabilities.
        
        Args:
            name: Component name for logging (defaults to class name)
        """
        # Use provided name or default to class name
        self.name = name or self.__class__.__name__
        # Create a logger with namespace maveric.{component_name}
        self.logger = logging.getLogger(f"maveric.{self.name}")
        
    def log_info(self, message: str):
        """Log info-level messages for normal operations."""
        self.logger.info(message)
        
    def log_warning(self, message: str):
        """Log warning messages for non-fatal issues."""
        self.logger.warning(message)
        
    def log_error(self, message: str):
        """Log error messages for failures."""
        self.logger.error(message)
        
    def log_debug(self, message: str):
        """Log debug messages for troubleshooting."""
        self.logger.debug(message)


class BaseDataset(BaseComponent):
    """
    Inherits ABC through BaseComponent
    Abstract base class for all dataset handlers (CIFAR, Elevater, etc.).
    All dataset implementations should inherit from this class.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Abstract property - each dataset must define its name."""
        pass # No implementation - subclasses must provide this
    
    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        """Abstract property - list of all classes in dataset."""
        pass
    
    @property
    def num_classes(self) -> int:
        """Computed property - returns length of class_names list."""
        return len(self.class_names)
    
    @abstractmethod
    def get_reference_samples(self, n_per_class: int) -> Dict[str, List[Image.Image]]:
        """
        Get reference samples for each class - used for CLIP embedding generation.
        
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
            Example: "a photo of a {}"
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
        # Substitute class name into each template
        return [template.format(class_name) for template in templates]


class BaseMetric(BaseComponent):
    """
    Inherits ABC through BaseComponent
    Abstract base class for all quality metrics (sharpness, resolution, etc.).
    All quality metric implementations should inherit from this class.
    """
    
    def __init__(self, name: str = None, weight: float = 1.0):
        """
        Initialize metric with name and weight.
        
        Args:
            name: Metric name (defaults to class name)
            weight: Weight for this metric in composite scores
        """
        # Call parent constructor for logging
        super().__init__(name)
        self.weight = weight
        # Create empty cache dictionary for performance
        self._cache = {}
        
    @property
    @abstractmethod
    def metric_name(self) -> str:
        """Abstract property - unique identifier for this metric."""
        pass
    
    @property
    def requires_reference(self) -> bool:
        """Property indicating whether metric needs reference samples (defaults to False)."""
        return False
    
    @abstractmethod
    def compute(self, image: Image.Image, metadata: Dict[str, Any]) -> float:
        """
        Abstract method - core computation each metric must implement.
        
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
        # Loop through images and call compute() for each
        for img, meta in zip(images, metadata_list):
            try:
                score = self.compute(img, meta)
                scores.append(score)
            # Error handling - log warnings and return 0.0 for failed computations
            except Exception as e:
                self.log_warning(f"Error computing {self.metric_name} for image: {e}")
                scores.append(0.0)
        return scores
    
    def normalize_score(self, score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """
        Utility method for score normalization.
        
        Args:
            score: Raw score value
            min_val: Minimum expected value
            max_val: Maximum expected value
            
        Returns:
            Normalized score in 0-1 range
        """
        # Handle edge case where min equals max
        if max_val == min_val:
            return 0.5
        # Linear normalization formula: (score - min) / (max - min)
        normalized = (score - min_val) / (max_val - min_val)
        # Clip result to 0-1 range using numpy
        return np.clip(normalized, 0.0, 1.0)
