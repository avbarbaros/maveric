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
                 metrics: Optional[List[BaseQualityMetric]] = None,
                 config: Optional[Any] = None):
        """
        Initialize quality controller.

        Args:
            data: Input data (DataFrame, list of dicts, or file path)
            metrics: List of quality metrics to use
            config: MAVERICConfig instance for weights and thresholds
        """
        super().__init__("QualityController")

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

        # Use config values if provided, otherwise use defaults
        if config and hasattr(config, 'metric_weights'):
            self.class_weights = config.metric_weights.copy()
        else:
            self.class_weights = {
                'img2img': 0.40,
                'txt2txt': 0.20,
                'img2txt': 0.20,
                'txt2img': 0.20
            }


        # Use config default thresholds if provided
        if config and hasattr(config, 'default_thresholds'):
            # Update default thresholds with config values
            self.thresholds.update(config.default_thresholds)

        # Filters
        self.filters = []

        # Load data
        self.data = None
        self.original_data = None  # Store original unfiltered data
        self.filtered_data = None
        if data is not None:
            self.load_data(data)
        
    def load_data(self, data: Union[pd.DataFrame, List[Dict], str]):
        """
        Load data from various sources.

        Args:
            data: Data source (DataFrame, list, or file path)
        """
        if isinstance(data, pd.DataFrame):
            self.data = data.copy()
            self.original_data = data.copy()  # Store original copy
        elif isinstance(data, list):
            self.data = pd.DataFrame(data)
            self.original_data = self.data.copy()  # Store original copy
        elif isinstance(data, str):
            # Load from file
            path = Path(data)
            if path.suffix == '.json':
                with open(path, 'r') as f:
                    loaded_data = json.load(f)
                self.data = pd.DataFrame(loaded_data)
                self.original_data = self.data.copy()  # Store original copy
            elif path.suffix == '.csv':
                self.data = pd.read_csv(path)
                self.original_data = self.data.copy()  # Store original copy
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
        Calculate the best class for each item using similarity scores.

        This method identifies the most likely class for each sample using
        weighted similarity-based scores (img2img, txt2txt, img2txt, txt2img).
        The weights are defined in metric_weights configuration.
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
                
                # Normalize similarity score
                if valid_weights_sum > 0:
                    similarity_score /= valid_weights_sum
                    class_scores[class_name] = similarity_score
            
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

    def reset_to_original_data(self):
        """
        Reset all data to original unfiltered state.

        This method:
        1. Restores the original data before any filtering
        2. Clears any filtered data
        3. Resets filters
        """
        if self.original_data is None:
            self.log_warning("No original data available to reset to")
            return False

        # Restore original data
        self.data = self.original_data.copy()

        # Clear filtered data
        self.filtered_data = None

        # Clear filters
        self.filters = []

        self.log_info(f"Reset to original data: {len(self.data)} samples")
        return True

    def reset_to_defaults(self):
        """
        Reset thresholds and weights to default values.

        This method resets all thresholds and metric weights to their
        default values as defined in the configuration.
        """
        # Reset thresholds to defaults
        default_thresholds = {
            'weighted_class_score': 0.493,
            'consistency': 0.796,
            'resolution_score': 0.370,
            'sharpness_score': 0.880,
            'color_score': 0.768,
            'text_quality_score': 0.600,
            'caption_length_score': 0.700,
            'target_class_quality': 0.493
        }

        # Reset weights to defaults
        default_weights = {
            'img2img': 0.40,
            'txt2txt': 0.20,
            'img2txt': 0.20,
            'txt2img': 0.20
        }

        self.thresholds.update(default_thresholds)
        self.class_weights.update(default_weights)

        self.log_info("Reset thresholds and weights to default values")

    def full_reset(self):
        """
        Perform a complete reset: restore original data AND reset to defaults.

        This is the most comprehensive reset option that:
        1. Restores original unfiltered data
        2. Resets all thresholds to defaults
        3. Resets all metric weights to defaults
        4. Clears all filters
        """
        # Reset data
        self.reset_to_original_data()

        # Reset thresholds and weights
        self.reset_to_defaults()

        self.log_info("Performed full reset: data, thresholds, and weights")

    def get_threshold_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for each individual threshold.

        Returns:
            Dictionary with threshold statistics for each metric
        """
        if self.data is None:
            return {}

        threshold_stats = {}
        total_samples = len(self.data)

        for metric, threshold in self.thresholds.items():
            if metric in self.data.columns:
                values = self.data[metric].dropna()
                if len(values) > 0:
                    # Count samples that pass this threshold
                    passing_samples = (values >= threshold).sum()
                    failing_samples = len(values) - passing_samples
                    pass_rate = (passing_samples / len(values)) * 100

                    threshold_stats[metric] = {
                        'threshold': threshold,
                        'total_samples': len(values),
                        'passing_samples': passing_samples,
                        'failing_samples': failing_samples,
                        'pass_rate': round(pass_rate, 1),
                        'samples_filtered_out': failing_samples
                    }

        return threshold_stats
    
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
    
    def save_filtered_data(self, output_path: str, format: str = 'json', copy_images: bool = True, cache_manager=None):
        """
        Save filtered data to file and optionally copy images to dataset-specific folder.

        Args:
            output_path: Output file path
            format: Output format ('json' or 'csv')
            copy_images: Whether to copy images to dataset-specific images folder
            cache_manager: CacheManager instance for accessing cached images
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

        # Copy images to dataset-specific folder
        if copy_images and cache_manager is not None:
            self._copy_training_images(output_path, cache_manager)

    def _copy_training_images(self, output_path: Path, cache_manager):
        """
        Copy training images to dataset-specific images folder.
        If image is not in cache, download it.

        Args:
            output_path: Path to saved training data JSON
            cache_manager: CacheManager instance for accessing cached images
        """
        import hashlib
        import shutil
        import requests
        from tqdm import tqdm
        from PIL import Image
        from io import BytesIO

        output_path = Path(output_path)
        images_dir = output_path.parent / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)

        total_images = len(self.filtered_data)
        self.log_info(f"Processing {total_images} training images to {images_dir}...")

        # Check how many already exist
        existing_count = sum(1 for _, row in self.filtered_data.iterrows()
                           if row.get('url') and (images_dir / f"img_{hashlib.md5(row.get('url').encode()).hexdigest()}.jpg").exists())

        if existing_count > 0:
            self.log_info(f"Found {existing_count} images already in destination, will process {total_images - existing_count} remaining images")

        copied_count = 0
        downloaded_count = 0
        failed_count = 0
        failed_downloads = []  # Store failed downloads to report at the end

        # Use tqdm with leave=True to keep the bar after completion
        pbar = tqdm(self.filtered_data.iterrows(), total=total_images, desc="Processing images", leave=True)

        for _, row in pbar:
            url = row.get('url')
            if not url:
                failed_count += 1
                continue

            # Calculate image hash
            url_hash = hashlib.md5(url.encode()).hexdigest()
            src_filename = f"img_{url_hash}.jpg"

            # Check hierarchical structure first (new format: image_cache/ae/img_aeb88f14....jpg)
            subdir = url_hash[:2]
            src_path_hierarchical = cache_manager.image_cache_dir / subdir / src_filename

            # Check flat structure for backward compatibility
            src_path_flat = cache_manager.image_cache_dir / src_filename

            # Destination: dataset-specific images folder
            dst_path = images_dir / src_filename

            # Skip if already exists in destination
            if dst_path.exists():
                continue

            # Try to copy from cache (hierarchical first, then flat)
            src_found = False
            for src_path in [src_path_hierarchical, src_path_flat]:
                if src_path.exists():
                    try:
                        shutil.copy2(src_path, dst_path)
                        copied_count += 1
                        src_found = True
                        break
                    except Exception as e:
                        continue

            # If not found in cache, download it
            if not src_found:
                try:
                    response = requests.get(url, timeout=(10, 30))  # Increased timeout: 10s connect, 30s read
                    response.raise_for_status()

                    # Load and validate image
                    image = Image.open(BytesIO(response.content)).convert('RGB')

                    # Save to destination
                    image.save(str(dst_path), 'JPEG', quality=95)
                    downloaded_count += 1

                    # Also save to hierarchical cache for future use
                    try:
                        cache_subdir = cache_manager.image_cache_dir / subdir
                        cache_subdir.mkdir(parents=True, exist_ok=True)
                        image.save(str(src_path_hierarchical), 'JPEG', quality=95)
                    except Exception:
                        pass  # Cache save failed, but we have the training image

                except Exception as e:
                    failed_count += 1
                    failed_downloads.append({
                        'filename': src_filename,
                        'url': url,
                        'error': str(e)
                    })

        pbar.close()

        # Calculate successful total
        success_count = existing_count + copied_count + downloaded_count
        self.log_info(f"Successfully processed {success_count}/{total_images} images: {copied_count} copied from cache, {downloaded_count} downloaded, {existing_count} already existed")

        if failed_count > 0:
            self.log_warning(f"{failed_count} images failed to process:")
            for failed in failed_downloads:
                self.log_warning(f"  ❌ {failed['filename']}")
                self.log_warning(f"     URL: {failed['url']}")
                self.log_warning(f"     Error: {failed['error']}")
