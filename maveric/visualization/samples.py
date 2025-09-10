"""Sample visualization for quality inspection."""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from typing import List, Dict, Union, Optional, Tuple
import pandas as pd
import random

from ..core.base import BaseComponent
from ..core.exceptions import VisualizationError


class SampleVisualizer(BaseComponent):
    """
    Visualizes actual image samples with their quality metrics.
    
    This helps researchers understand what high/low quality samples look like
    and verify that quality metrics correspond to actual visual quality.
    """
    
    def __init__(self, figsize_per_image: Tuple[int, int] = (4, 5)):
        """
        Initialize sample visualizer.
        
        Args:
            figsize_per_image: Size for each image in the grid
        """
        super().__init__("SampleVisualizer")
        self.figsize_per_image = figsize_per_image
    
    def visualize_samples(self,
                         data: Union[pd.DataFrame, List[Dict]],
                         n_samples: int = 5,
                         sample_type: str = 'random',
                         metrics_to_show: Optional[List[str]] = None,
                         seed: Optional[int] = None,
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Display sample images with their quality metrics.
        
        Args:
            data: Dataset (DataFrame or list of dicts)
            n_samples: Number of samples to show
            sample_type: 'random', 'best', 'worst', 'diverse', or 'specific'
            metrics_to_show: List of metrics to display (None for default)
            seed: Random seed for reproducibility
            save_path: Path to save figure
            
        Returns:
            Figure showing sample images with metrics
        """
        # Convert to DataFrame if needed
        if isinstance(data, list):
            data = pd.DataFrame(data)
        
        # Select samples
        samples = self._select_samples(data, n_samples, sample_type, seed)
        
        if len(samples) == 0:
            raise VisualizationError("No samples selected")
        
        # Create figure
        figsize = (self.figsize_per_image[0] * len(samples), 
                  self.figsize_per_image[1])
        fig, axes = plt.subplots(1, len(samples), figsize=figsize)
        
        if len(samples) == 1:
            axes = [axes]
        
        # Default metrics to show
        if metrics_to_show is None:
            metrics_to_show = [
                'label', 'weighted_class_score', 'consistency',
                'resolution_score', 'sharpness_score', 'color_score', 'composite_quality'
            ]
        
        # Display each sample
        for idx, (_, sample) in enumerate(samples.iterrows()):
            ax = axes[idx]
            
            # Load and display image
            try:
                image = self._load_image(sample.get('url', ''))
                if image:
                    ax.imshow(image)
                    
                    # Format metrics
                    metrics_text = self._format_metrics(sample, metrics_to_show)
                    ax.set_title(metrics_text, fontsize=10, pad=10)
                else:
                    ax.text(0.5, 0.5, "Failed to load image",
                           ha='center', va='center', transform=ax.transAxes)
                    
            except Exception as e:
                ax.text(0.5, 0.5, f"Error: {str(e)[:50]}...",
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=9, wrap=True)
            
            ax.axis('off')
        
        # Overall title
        title_map = {
            'random': 'Random Samples',
            'best': 'Best Quality Samples',
            'worst': 'Worst Quality Samples',
            'diverse': 'Diverse Quality Samples',
            'specific': 'Selected Samples'
        }
        fig.suptitle(f'{title_map.get(sample_type, sample_type)} from Dataset',
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            self.log_info(f"Saved figure to {save_path}")
        
        return fig
    
    def _select_samples(self, data: pd.DataFrame, n_samples: int,
                       sample_type: str, seed: Optional[int]) -> pd.DataFrame:
        """Select samples based on sampling strategy."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Ensure we have required columns for some strategies
        if sample_type in ['best', 'worst', 'diverse']:
            if 'weighted_class_score' not in data.columns:
                self.log_warning("No 'weighted_class_score' found, using random sampling")
                sample_type = 'random'
        
        if sample_type == 'random':
            # Random sampling
            n_samples = min(n_samples, len(data))
            return data.sample(n=n_samples, random_state=seed)
        
        elif sample_type == 'best':
            # Top scoring samples
            return data.nlargest(n_samples, 'weighted_class_score')
        
        elif sample_type == 'worst':
            # Bottom scoring samples
            return data.nsmallest(n_samples, 'weighted_class_score')
        
        elif sample_type == 'diverse':
            # Samples across quality spectrum
            sorted_data = data.sort_values('weighted_class_score')
            indices = np.linspace(0, len(sorted_data) - 1, n_samples, dtype=int)
            return sorted_data.iloc[indices]
        
        else:
            # Default to random
            return data.sample(n=min(n_samples, len(data)), random_state=seed)
    
    def _load_image(self, url: str, timeout: int = 5) -> Optional[Image.Image]:
        """Load image from URL."""
        if not url:
            return None
        
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            return image
        except Exception as e:
            self.log_warning(f"Failed to load image from {url}: {e}")
            return None
    
    def _format_metrics(self, sample: pd.Series, metrics_to_show: List[str]) -> str:
        """Format metrics for display."""
        lines = []
        
        # Add ID if available
        if 'id' in sample:
            lines.append(f"ID: {sample['id']}")
        
        # Add each metric
        for metric in metrics_to_show:
            if metric in sample and not pd.isna(sample[metric]):
                value = sample[metric]
                
                # Format based on type
                if isinstance(value, (int, np.integer)):
                    lines.append(f"{metric}: {value}")
                elif isinstance(value, (float, np.floating)):
                    lines.append(f"{metric}: {value:.3f}")
                else:
                    lines.append(f"{metric}: {value}")
        
        return '\n'.join(lines)
    
    def create_quality_grid(self,
                           data: pd.DataFrame,
                           metric: str = 'weighted_class_score',
                           grid_size: Tuple[int, int] = (3, 3),
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a grid showing samples across quality range.
        
        Args:
            data: Dataset
            metric: Metric to use for quality sorting
            grid_size: (rows, cols) for the grid
            save_path: Path to save figure
            
        Returns:
            Figure with quality grid
        """
        n_samples = grid_size[0] * grid_size[1]
        
        # Sort by metric
        sorted_data = data.sort_values(metric)
        
        # Select samples evenly across range
        indices = np.linspace(0, len(sorted_data) - 1, n_samples, dtype=int)
        samples = sorted_data.iloc[indices]
        
        # Create figure
        fig, axes = plt.subplots(grid_size[0], grid_size[1],
                                figsize=(grid_size[1] * 3, grid_size[0] * 3))
        axes = axes.flatten()
        
        # Display samples
        for idx, (_, sample) in enumerate(samples.iterrows()):
            ax = axes[idx]
            
            # Load image
            image = self._load_image(sample.get('url', ''))
            if image:
                ax.imshow(image)
                score = sample.get(metric, 0)
                ax.set_title(f"{metric}: {score:.3f}", fontsize=10)
            else:
                ax.text(0.5, 0.5, "No image", ha='center', va='center')
            
            ax.axis('off')
        
        fig.suptitle(f'Quality Grid Sorted by {metric}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def compare_classes(self,
                       data: pd.DataFrame,
                       classes: List[str],
                       samples_per_class: int = 3,
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare samples from different classes.
        
        Args:
            data: Dataset with 'label' column
            classes: List of class names to compare
            samples_per_class: Number of samples per class
            save_path: Path to save figure
            
        Returns:
            Figure comparing classes
        """
        # Filter classes that exist in data
        if 'label' not in data.columns:
            raise VisualizationError("No 'label' column in data")
        
        available_classes = data['label'].unique()
        valid_classes = [c for c in classes if c in available_classes]
        
        if not valid_classes:
            raise VisualizationError("No valid classes found")
        
        # Create figure
        fig, axes = plt.subplots(len(valid_classes), samples_per_class,
                                figsize=(samples_per_class * 3, len(valid_classes) * 3))
        
        if len(valid_classes) == 1:
            axes = axes.reshape(1, -1)
        
        # Display samples for each class
        for class_idx, class_name in enumerate(valid_classes):
            class_data = data[data['label'] == class_name]
            
            # Sample from class
            n_available = len(class_data)
            n_to_sample = min(samples_per_class, n_available)
            
            if n_to_sample > 0:
                class_samples = class_data.sample(n=n_to_sample)
                
                for sample_idx, (_, sample) in enumerate(class_samples.iterrows()):
                    ax = axes[class_idx, sample_idx]
                    
                    # Load image
                    image = self._load_image(sample.get('url', ''))
                    if image:
                        ax.imshow(image)
                        score = sample.get('weighted_class_score', 0)
                        ax.set_title(f"Score: {score:.3f}", fontsize=9)
                    else:
                        ax.text(0.5, 0.5, "No image", ha='center', va='center')
                    
                    ax.axis('off')
                    
                    # Add class label to first column
                    if sample_idx == 0:
                        ax.set_ylabel(class_name, fontsize=12, rotation=0,
                                     ha='right', va='center', labelpad=50)
            
            # Fill remaining slots
            for sample_idx in range(n_to_sample, samples_per_class):
                ax = axes[class_idx, sample_idx]
                ax.axis('off')
        
        fig.suptitle('Class Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
