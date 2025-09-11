"""Main quality control system for MAVERIC."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from collections import defaultdict
import json
from pathlib import Path

from ..core.base import BaseComponent
from ..core.interfaces import QualityResult
from ..core.exceptions import DatasetError
from .metrics.base_metric import BaseQualityMetric
from .filters import ThresholdFilter, BalancedFilter


class QualityController(BaseComponent):
    """
    Main controller for quality assessment and filtering.
    
    This class orchestrates the quality control process, applying various
    metrics and filters to create high-quality datasets for model training.
    """
    
    def __init__(self, 
                 data: Union[pd.DataFrame, List[Dict], str] = None,
                 metrics: Optional[List[BaseQualityMetric]] = None):
        """
        Initialize quality controller.
        
        Args:
            data: Input data (DataFrame, list of dicts, or file path)
            metrics: List of quality metrics to use
        """
        super().__init__("QualityController")
        
        # Load data
        self.data = None
        self.filtered_data = None
        if data is not None:
            self.load_data(data)
        
        # Initialize metrics
        self.metrics = metrics or []
        self.metric_registry = {m.metric_name: m for m in self.metrics}
        
        # Default thresholds and weights
        self.thresholds = {
            'weighted_class_score': 0.493,
            'consistency': 0.796,
            'resolution_score': 0.370,
            'sharpness_score': 0.880,
            'color_score': 0.768
        }
        
        self.class_weights = {
            'img2img': 0.40,
            'txt2txt': 0.20,
            'img2txt': 0.20,
            'txt2img': 0.20
        }
        
        # Class selection weights: balance between similarity and quality
        self.class_selection_weights = {
            'similarity_weight': 0.7,  # Weight for similarity-based scoring
            'quality_weight': 0.3      # Weight for semantic quality scoring
        }
        
        # Filters
        self.filters = []
        
    def load_data(self, data: Union[pd.DataFrame, List[Dict], str]):
        """
        Load data from various sources.
        
        Args:
            data: Data source (DataFrame, list, or file path)
        """
        if isinstance(data, pd.DataFrame):
            self.data = data
        elif isinstance(data, list):
            self.data = pd.DataFrame(data)
        elif isinstance(data, str):
            # Load from file
            path = Path(data)
            if path.suffix == '.json':
                with open(path, 'r') as f:
                    loaded_data = json.load(f)
                self.data = pd.DataFrame(loaded_data)
            elif path.suffix == '.csv':
                self.data = pd.read_csv(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        self.log_info(f"Loaded {len(self.data)} samples")
        
        # Calculate best class if not already present
        if 'label' not in self.data.columns:
            self._calculate_best_class()
        
        # Initialize filtered data
        self.filtered_data = self.data.copy()
    
    def _calculate_best_class(self):
        """
        Calculate the best class for each item using both similarity and quality scores.
        
        This method identifies the most likely class for each sample by combining:
        1. Similarity-based scores (img2img, txt2txt, img2txt, txt2img)
        2. Target class quality score (target_class_quality) for universal quality assessment
        
        The final score is a weighted combination of both factors for better class selection.
        """
        if self.data is None:
            return
        
        # Extract class names from column names
        class_columns = [col for col in self.data.columns if col.startswith('Class_')]
        if not class_columns:
            self.log_warning("No class score columns found")
            return
        
        # Get unique class names
        class_names = set()
        for col in class_columns:
            parts = col.split('_')
            if len(parts) >= 3:
                class_name = parts[1]
                class_names.add(class_name)
        
        class_names = sorted(list(class_names))
        
        # Calculate weighted scores for each sample
        best_classes = []
        weighted_scores = []
        consistency_scores = []
        
        for _, row in self.data.iterrows():
            class_scores = {}
            
            for class_name in class_names:
                similarity_score = 0.0
                valid_weights_sum = 0.0
                
                # Calculate weighted similarity score
                for metric, weight in self.class_weights.items():
                    col_name = f"Class_{class_name}_{metric}"
                    
                    if col_name in row and not pd.isna(row[col_name]):
                        similarity_score += row[col_name] * weight
                        valid_weights_sum += weight
                
                # Get class-specific target class quality score
                target_class_quality_col = f"Class_{class_name}_target_class_quality"
                target_class_quality = row.get(target_class_quality_col, 0.0)
                if pd.isna(target_class_quality):
                    target_class_quality = 0.0
                
                # Normalize similarity score
                if valid_weights_sum > 0:
                    similarity_score /= valid_weights_sum
                    
                    # Combine similarity score with class-specific quality score
                    combined_score = (
                        self.class_selection_weights['similarity_weight'] * similarity_score +
                        self.class_selection_weights['quality_weight'] * target_class_quality
                    )
                    
                    class_scores[class_name] = combined_score
            
            # Find best class
            if class_scores:
                best_class = max(class_scores.items(), key=lambda x: x[1])
                best_classes.append(best_class[0])
                weighted_scores.append(best_class[1])
                
                # Get consistency score
                consistency_col = f"Class_{best_class[0]}_consistency"
                if consistency_col in row:
                    consistency_scores.append(row[consistency_col])
                else:
                    consistency_scores.append(0.0)
            else:
                best_classes.append(None)
                weighted_scores.append(0.0)
                consistency_scores.append(0.0)
        
        # Add calculated columns
        self.data['label'] = best_classes
        self.data['weighted_class_score'] = weighted_scores
        self.data['consistency'] = consistency_scores
        
        self.log_info("Calculated best class for all samples")
    
    def set_threshold(self, metric: str, value: float):
        """
        Set quality threshold for a metric.
        
        Args:
            metric: Metric name
            value: Threshold value
        """
        self.thresholds[metric] = value
        self.log_debug(f"Set {metric} threshold to {value}")
    
    def set_class_weight(self, metric: str, value: float):
        """
        Set weight for a class similarity metric.
        
        Args:
            metric: Metric name (img2img, txt2txt, etc.)
            value: Weight value
        """
        self.class_weights[metric] = value
        # Recalculate best class with new weights
        self._calculate_best_class()
        self.log_debug(f"Set {metric} weight to {value}")
    
    def set_class_selection_weight(self, weight_type: str, value: float):
        """
        Set class selection weight for balancing similarity vs quality.
        
        Args:
            weight_type: Either 'similarity_weight' or 'quality_weight'
            value: Weight value (should sum to 1.0 with the other weight)
        """
        if weight_type not in self.class_selection_weights:
            raise ValueError(f"Invalid weight type: {weight_type}")
        
        self.class_selection_weights[weight_type] = value
        
        # Normalize weights to sum to 1.0
        total = sum(self.class_selection_weights.values())
        if total > 0:
            for key in self.class_selection_weights:
                self.class_selection_weights[key] /= total
        
        # Recalculate best class with new weights
        self._calculate_best_class()
        self.log_info(f"Set {weight_type} to {value:.3f} (normalized to {self.class_selection_weights[weight_type]:.3f})")
    
    def get_class_selection_weights(self) -> Dict[str, float]:
        """
        Get current class selection weights.
        
        Returns:
            Dictionary of class selection weights
        """
        return self.class_selection_weights.copy()
    
    def add_filter(self, filter_instance):
        """
        Add a filter to the quality control pipeline.
        
        Args:
            filter_instance: Filter instance
        """
        self.filters.append(filter_instance)
    
    def apply_thresholds(self) -> int:
        """
        Apply current thresholds to filter data.
        
        Returns:
            Number of samples after filtering
        """
        if self.data is None:
            raise DatasetError("No data loaded")
        
        # Start with all data
        self.filtered_data = self.data.copy()
        
        # Create threshold filter
        threshold_filter = ThresholdFilter(self.thresholds)
        self.filtered_data = threshold_filter.apply(self.filtered_data)
        
        # Apply any additional filters
        for filter_instance in self.filters:
            self.filtered_data = filter_instance.apply(self.filtered_data)
        
        filtered_count = len(self.filtered_data)
        retention_rate = filtered_count / len(self.data) * 100
        
        self.log_info(
            f"Filtered data: {filtered_count} samples "
            f"({retention_rate:.1f}% retention)"
        )
        
        return filtered_count
    
    def balance_dataset(self, 
                       strategy: str = 'median',
                       min_samples: int = 15,
                       enable_oversampling: bool = False) -> pd.DataFrame:
        """
        Balance the filtered dataset across classes.
        
        Args:
            strategy: Balancing strategy ('median', 'mean', 'min', 'max')
            min_samples: Minimum samples per class
            enable_oversampling: Whether to oversample small classes
            
        Returns:
            Balanced DataFrame
        """
        if self.filtered_data is None:
            raise DatasetError("No filtered data available")
        
        balancer = BalancedFilter(
            strategy=strategy,
            min_threshold=min_samples,
            enable_oversampling=enable_oversampling
        )
        
        balanced_data = balancer.apply(self.filtered_data)
        
        # Update filtered data
        self.filtered_data = balanced_data
        
        return balanced_data
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the data.
        
        Returns:
            Dictionary with various statistics
        """
        if self.data is None:
            return {}
        
        stats = {
            'total_samples': len(self.data),
            'filtered_samples': len(self.filtered_data) if self.filtered_data is not None else 0,
            'retention_rate': 0.0,
            'class_distribution': {},
            'metric_statistics': {}
        }
        
        # Calculate retention rate
        if self.filtered_data is not None and len(self.data) > 0:
            stats['retention_rate'] = len(self.filtered_data) / len(self.data)
        
        # Class distribution
        if 'label' in self.data.columns:
            original_dist = self.data['label'].value_counts().to_dict()
            stats['class_distribution']['original'] = original_dist
            
            if self.filtered_data is not None and 'label' in self.filtered_data.columns:
                filtered_dist = self.filtered_data['label'].value_counts().to_dict()
                stats['class_distribution']['filtered'] = filtered_dist
        
        # Metric statistics
        metric_columns = [col for col in self.data.columns 
                         if 'score' in col or 'consistency' in col]
        
        for col in metric_columns:
            if col in self.data.columns:
                values = self.data[col].dropna()
                if len(values) > 0:
                    stats['metric_statistics'][col] = {
                        'mean': round(float(values.mean()), 5),
                        'std': round(float(values.std()), 5),
                        'min': round(float(values.min()), 5),
                        'max': round(float(values.max()), 5),
                        'median': round(float(values.median()), 5)
                    }
        
        return stats
    
    def create_quality_result(self, 
                            thresholds: Optional[Dict[str, float]] = None,
                            balance_strategy: str = "none") -> QualityResult:
        """
        Create a QualityResult object from current state.
        
        Args:
            thresholds: Thresholds used (defaults to current)
            balance_strategy: Balance strategy used
            
        Returns:
            QualityResult instance
        """
        if self.data is None:
            raise DatasetError("No data loaded")
        
        # Use current thresholds if not provided
        if thresholds is None:
            thresholds = self.thresholds.copy()
        
        # Create result
        result = QualityResult(
            filtered_samples=self.filtered_data.to_dict('records') if self.filtered_data is not None else [],
            original_samples=self.data.to_dict('records'),
            thresholds=thresholds,
            weights=self.class_weights.copy(),
            balance_strategy=balance_strategy
        )
        
        return result
    
    def save_filtered_data(self, output_path: str, format: str = 'json'):
        """
        Save filtered data to file.
        
        Args:
            output_path: Output file path
            format: Output format ('json' or 'csv')
        """
        if self.filtered_data is None:
            raise DatasetError("No filtered data to save")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            # Convert to simplified format for JSON
            simplified_data = []
            
            for _, row in self.filtered_data.iterrows():
                item = {
                    'id': int(row.get('id', 0)),
                    'url': row.get('url', ''),
                    'label': row.get('label', ''),
                    'text': row.get('text', ''),
                    'weighted_class_score': round(float(row.get('weighted_class_score', 0)), 5),
                    'consistency': round(float(row.get('consistency', 0)), 5)
                }
                
                # Add quality scores
                for metric in ['resolution_score', 'sharpness_score', 'color_score', 'target_class_quality']:
                    if metric in row:
                        item[metric] = round(float(row[metric]), 5)
                
                simplified_data.append(item)
            
            with open(output_path, 'w') as f:
                json.dump(simplified_data, f, indent=2)
                
        elif format == 'csv':
            self.filtered_data.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.log_info(f"Saved {len(self.filtered_data)} samples to {output_path}")
