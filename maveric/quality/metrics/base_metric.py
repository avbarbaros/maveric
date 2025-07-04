"""Base classes for quality metrics."""

from abc import abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image

from ...core.base import BaseMetric


class BaseQualityMetric(BaseMetric):
    """
    Extended base class for quality metrics with additional functionality.
    
    This class provides common utilities for all quality metrics, including
    caching, normalization, and batch processing optimizations.
    """
    
    def __init__(self, name: str = None, weight: float = 1.0, 
                 cache_enabled: bool = True):
        """
        Initialize quality metric.
        
        Args:
            name: Metric name
            weight: Weight for composite scoring
            cache_enabled: Whether to cache computed scores
        """
        super().__init__(name, weight)
        self.cache_enabled = cache_enabled
        self._cache = {} if cache_enabled else None
    
    def compute_cached(self, image: Image.Image, metadata: Dict[str, Any],
                      cache_key: Optional[str] = None) -> float:
        """
        Compute score with optional caching.
        
        Args:
            image: PIL Image
            metadata: Image metadata
            cache_key: Optional cache key (defaults to URL if available)
            
        Returns:
            Quality score
        """
        # Use URL as cache key if not provided
        if cache_key is None and 'url' in metadata:
            cache_key = metadata['url']
        
        # Check cache
        if self.cache_enabled and cache_key and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Compute score
        score = self.compute(image, metadata)
        
        # Cache result
        if self.cache_enabled and cache_key:
            self._cache[cache_key] = score
        
        return score
    
    def clear_cache(self):
        """Clear the score cache."""
        if self._cache is not None:
            self._cache.clear()
    
    def get_statistics(self, scores: List[float]) -> Dict[str, float]:
        """
        Calculate statistics for a list of scores.
        
        Args:
            scores: List of quality scores
            
        Returns:
            Dictionary with statistics (mean, std, min, max, median)
        """
        if not scores:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0
            }
        
        scores_array = np.array(scores)
        return {
            'mean': float(np.mean(scores_array)),
            'std': float(np.std(scores_array)),
            'min': float(np.min(scores_array)),
            'max': float(np.max(scores_array)),
            'median': float(np.median(scores_array))
        }


class CompositeMetric(BaseQualityMetric):
    """
    Composite metric that combines multiple metrics.
    
    This allows creating complex quality assessments by combining
    multiple individual metrics with configurable weights.
    """
    
    def __init__(self, name: str, metrics: List[BaseQualityMetric],
                 weights: Optional[List[float]] = None):
        """
        Initialize composite metric.
        
        Args:
            name: Name for this composite metric
            metrics: List of metrics to combine
            weights: Optional weights for each metric (defaults to equal)
        """
        super().__init__(name)
        self.metrics = metrics
        
        if weights is None:
            weights = [1.0] * len(metrics)
        elif len(weights) != len(metrics):
            raise ValueError("Number of weights must match number of metrics")
        
        # Normalize weights
        total_weight = sum(weights)
        self.weights = [w / total_weight for w in weights]
    
    @property
    def metric_name(self) -> str:
        """Return metric name."""
        return f"composite_{self.name}"
    
    def compute(self, image: Image.Image, metadata: Dict[str, Any]) -> float:
        """
        Compute composite score.
        
        Args:
            image: PIL Image
            metadata: Image metadata
            
        Returns:
            Weighted average of component metric scores
        """
        scores = []
        for metric, weight in zip(self.metrics, self.weights):
            try:
                score = metric.compute(image, metadata)
                scores.append(score * weight)
            except Exception as e:
                self.log_warning(f"Error in {metric.metric_name}: {e}")
                scores.append(0.0)
        
        return sum(scores)
    
    def get_component_scores(self, image: Image.Image, 
                           metadata: Dict[str, Any]) -> Dict[str, float]:
        """
        Get individual scores from each component metric.
        
        Args:
            image: PIL Image
            metadata: Image metadata
            
        Returns:
            Dictionary mapping metric names to scores
        """
        component_scores = {}
        for metric in self.metrics:
            try:
                score = metric.compute(image, metadata)
                component_scores[metric.metric_name] = score
            except Exception as e:
                self.log_warning(f"Error in {metric.metric_name}: {e}")
                component_scores[metric.metric_name] = 0.0
        
        return component_scores
