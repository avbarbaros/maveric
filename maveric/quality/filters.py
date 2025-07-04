"""Filtering strategies for quality control."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from collections import defaultdict
import random

from ..core.base import BaseComponent


class QualityFilter(BaseComponent, ABC):
    """
    Abstract base class for quality filters.
    
    Filters are applied sequentially to refine the dataset based on
    various quality criteria.
    """
    
    @abstractmethod
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the filter to the data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Filtered DataFrame
        """
        pass


class ThresholdFilter(QualityFilter):
    """
    Filter based on quality thresholds.
    
    This filter removes samples that don't meet minimum quality standards
    across various metrics.
    """
    
    def __init__(self, thresholds: Dict[str, float]):
        """
        Initialize threshold filter.
        
        Args:
            thresholds: Dictionary mapping metric names to threshold values
        """
        super().__init__("ThresholdFilter")
        self.thresholds = thresholds
    
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply threshold filtering.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Filtered DataFrame
        """
        filtered_data = data.copy()
        initial_count = len(filtered_data)
        
        # Apply each threshold
        for metric, threshold in self.thresholds.items():
            if metric in filtered_data.columns:
                before_count = len(filtered_data)
                filtered_data = filtered_data[filtered_data[metric] >= threshold]
                removed = before_count - len(filtered_data)
                
                if removed > 0:
                    self.log_debug(
                        f"Threshold {metric} >= {threshold} removed {removed} samples"
                    )
        
        total_removed = initial_count - len(filtered_data)
        retention_rate = len(filtered_data) / initial_count * 100 if initial_count > 0 else 0
        
        self.log_info(
            f"Threshold filtering: {initial_count} → {len(filtered_data)} samples "
            f"({retention_rate:.1f}% retained)"
        )
        
        return filtered_data


class BalancedFilter(QualityFilter):
    """
    Filter to balance class distribution.
    
    This filter ensures balanced representation across classes using various
    strategies like undersampling, oversampling, or hybrid approaches.
    """
    
    def __init__(self,
                 strategy: str = 'median',
                 min_threshold: int = 15,
                 max_target: int = 200,
                 enable_oversampling: bool = False,
                 sort_by: str = 'consistency'):
        """
        Initialize balanced filter.
        
        Args:
            strategy: Balancing strategy ('median', 'mean', 'min', 'max')
            min_threshold: Minimum samples required per class
            max_target: Maximum samples per class
            enable_oversampling: Whether to oversample small classes
            sort_by: Metric to use for selecting best samples
        """
        super().__init__("BalancedFilter")
        self.strategy = strategy
        self.min_threshold = min_threshold
        self.max_target = max_target
        self.enable_oversampling = enable_oversampling
        self.sort_by = sort_by
    
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply class balancing.
        
        Args:
            data: Input DataFrame with 'label' column
            
        Returns:
            Balanced DataFrame
        """
        if 'label' not in data.columns:
            self.log_warning("No 'label' column found, returning original data")
            return data
        
        # Group by class
        class_groups = defaultdict(list)
        for idx, row in data.iterrows():
            class_groups[row['label']].append(idx)
        
        # Calculate statistics
        class_sizes = {cls: len(indices) for cls, indices in class_groups.items()}
        total_original = len(data)
        
        self.log_info(
            f"Original distribution: {total_original} samples across "
            f"{len(class_groups)} classes"
        )
        
        # Filter out classes below minimum threshold
        sufficient_classes = {
            cls: indices for cls, indices in class_groups.items()
            if len(indices) >= self.min_threshold
        }
        
        removed_classes = set(class_groups.keys()) - set(sufficient_classes.keys())
        if removed_classes:
            self.log_info(
                f"Removing {len(removed_classes)} classes with "
                f"< {self.min_threshold} samples"
            )
        
        if not sufficient_classes:
            self.log_warning("No classes meet minimum threshold")
            return pd.DataFrame()
        
        # Calculate target samples per class
        remaining_sizes = [len(indices) for indices in sufficient_classes.values()]
        
        if self.strategy == 'median':
            target_samples = int(np.median(remaining_sizes))
        elif self.strategy == 'mean':
            target_samples = int(np.mean(remaining_sizes))
        elif self.strategy == 'min':
            target_samples = min(remaining_sizes)
        elif self.strategy == 'max':
            target_samples = max(remaining_sizes)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Apply max target constraint
        target_samples = min(target_samples, self.max_target)
        target_samples = max(target_samples, self.min_threshold)
        
        self.log_info(f"Target samples per class: {target_samples}")
        
        # Balance each class
        balanced_indices = []
        
        for class_name, indices in sufficient_classes.items():
            class_data = data.loc[indices]
            
            # Sort by quality metric
            if self.sort_by in class_data.columns:
                class_data = class_data.sort_values(self.sort_by, ascending=False)
            
            current_size = len(class_data)
            
            if current_size > target_samples:
                # Undersample: take top samples
                selected_indices = class_data.index[:target_samples].tolist()
            elif current_size == target_samples:
                # Perfect size
                selected_indices = class_data.index.tolist()
            else:
                # Smaller than target
                if self.enable_oversampling:
                    # Oversample by duplicating high-quality samples
                    selected_indices = class_data.index.tolist()
                    needed = target_samples - current_size
                    
                    # Duplicate samples cyclically
                    for i in range(needed):
                        duplicate_idx = class_data.index[i % current_size]
                        selected_indices.append(duplicate_idx)
                else:
                    # Keep original size
                    selected_indices = class_data.index.tolist()
            
            balanced_indices.extend(selected_indices)
        
        # Create balanced dataset
        balanced_data = data.loc[balanced_indices].copy()
        
        # Shuffle
        balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Log results
        final_distribution = balanced_data['label'].value_counts()
        self.log_info(
            f"Balanced dataset: {len(balanced_data)} samples, "
            f"{len(final_distribution)} classes"
        )
        
        return balanced_data


class PercentileFilter(QualityFilter):
    """
    Filter based on percentile thresholds.
    
    This filter removes samples below a certain percentile for each metric,
    which is more adaptive than fixed thresholds.
    """
    
    def __init__(self, percentiles: Dict[str, float]):
        """
        Initialize percentile filter.
        
        Args:
            percentiles: Dictionary mapping metric names to percentile values (0-100)
        """
        super().__init__("PercentileFilter")
        self.percentiles = percentiles
    
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply percentile filtering.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Filtered DataFrame
        """
        filtered_data = data.copy()
        
        for metric, percentile in self.percentiles.items():
            if metric in filtered_data.columns:
                threshold = filtered_data[metric].quantile(percentile / 100)
                filtered_data = filtered_data[filtered_data[metric] >= threshold]
                
                self.log_debug(
                    f"Percentile filter {metric} >= {percentile}th percentile "
                    f"(threshold: {threshold:.3f})"
                )
        
        return filtered_data


class CompositeFilter(QualityFilter):
    """
    Composite filter that applies multiple filters in sequence.
    
    This allows creating complex filtering pipelines by combining
    simpler filters.
    """
    
    def __init__(self, filters: List[QualityFilter]):
        """
        Initialize composite filter.
        
        Args:
            filters: List of filters to apply in sequence
        """
        super().__init__("CompositeFilter")
        self.filters = filters
    
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all filters in sequence.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Filtered DataFrame
        """
        result = data
        
        for filter_instance in self.filters:
            before_count = len(result)
            result = filter_instance.apply(result)
            after_count = len(result)
            
            self.log_debug(
                f"{filter_instance.__class__.__name__}: "
                f"{before_count} → {after_count} samples"
            )
        
        return result
