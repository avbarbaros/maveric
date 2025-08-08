"""Distribution visualization for quality metrics."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from ..core.base import BaseComponent
from ..core.exceptions import VisualizationError


class MetricsVisualizer(BaseComponent):
    """
    Visualizes distribution of quality metrics with statistical information.
    
    This helps researchers understand the characteristics of their data
    and make informed decisions about threshold selection.
    """
    
    def __init__(self, 
                 style: str = 'default',
                 figsize: Tuple[int, int] = (10, 6),
                 dpi: int = 100):
        """
        Initialize metrics visualizer.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
            dpi: Figure resolution
        """
        super().__init__("MetricsVisualizer")
        self.style = style
        self.figsize = figsize
        self.dpi = dpi
        
        # Set style
        try:
            plt.style.use(style)
        except OSError:
            # Fallback to default if style is not available
            plt.style.use('default')
        
        try:
            sns.set_palette("husl")
        except Exception:
            # Continue without seaborn if not available
            pass
    
    def plot_metric_distribution(self,
                               data: pd.DataFrame,
                               metric_name: str,
                               threshold: Optional[float] = None,
                               bins: int = 50,
                               figsize: Optional[Tuple[int, int]] = None,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot distribution of a single metric with statistical overlays.
        
        This visualization shows:
        - Histogram of metric values
        - Mean line (green)
        - Standard deviation boundaries (blue dotted)
        - Current threshold (red dashed)
        - Pass/fail regions (shaded)
        
        Args:
            data: DataFrame containing metric values
            metric_name: Name of metric column to plot
            threshold: Current threshold value (optional)
            bins: Number of histogram bins
            figsize: Figure size (uses default if None)
            save_path: Path to save figure (optional)
            
        Returns:
            Matplotlib figure object
        """
        if metric_name not in data.columns:
            raise VisualizationError(f"Metric '{metric_name}' not found in data")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize or self.figsize, dpi=self.dpi)
        
        # Extract values
        values = data[metric_name].dropna()
        
        if len(values) == 0:
            ax.text(0.5, 0.5, f"No data for {metric_name}",
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=14)
            return fig
        
        # Calculate statistics
        mean_val = values.mean()
        std_val = values.std()
        median_val = values.median()
        
        # Create histogram
        counts, bins_edges, patches = ax.hist(
            values, bins=bins, alpha=0.7,
            color='skyblue', edgecolor='black',
            density=True
        )
        
        # Add KDE curve
        values_sorted = np.sort(values)
        from scipy import stats
        kde = stats.gaussian_kde(values)
        x_range = np.linspace(values.min(), values.max(), 200)
        ax.plot(x_range, kde(x_range), 'b-', linewidth=2, label='KDE')
        
        # Add statistical lines
        ax.axvline(mean_val, color='green', linestyle='-', linewidth=2,
                  label=f'Mean: {mean_val:.4f}')
        ax.axvline(median_val, color='orange', linestyle='-', linewidth=2,
                  label=f'Median: {median_val:.4f}')
        ax.axvline(mean_val - std_val, color='blue', linestyle=':', linewidth=2,
                  label=f'Mean-σ: {(mean_val-std_val):.4f}')
        ax.axvline(mean_val + std_val, color='blue', linestyle=':', linewidth=2,
                  label=f'Mean+σ: {(mean_val+std_val):.4f}')
        
        # Add threshold and shading
        if threshold is not None:
            ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
                      label=f'Threshold: {threshold:.4f}')
            
            # Calculate pass rate
            pass_rate = (values >= threshold).sum() / len(values) * 100
            
            # Shade regions
            ax.axvspan(values.min(), threshold, alpha=0.1, color='red')
            ax.axvspan(threshold, values.max(), alpha=0.1, color='green')
            
            # Add pass rate text
            ax.text(0.02, 0.95, f'Pass rate: {pass_rate:.1f}%',
                   transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Formatting
        ax.set_xlabel(f'{metric_name}', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Distribution of {metric_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        # Add statistics box
        stats_text = f'N: {len(values):,}\nMin: {values.min():.4f}\nMax: {values.max():.4f}'
        ax.text(0.98, 0.95, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.log_info(f"Saved figure to {save_path}")
        
        return fig
    
    def plot_multi_metric_distributions(self,
                                      data: pd.DataFrame,
                                      metrics: List[str],
                                      thresholds: Optional[Dict[str, float]] = None,
                                      ncols: int = 2,
                                      figsize_per_plot: Tuple[int, int] = (6, 4),
                                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a grid of distribution plots for multiple metrics.
        
        Args:
            data: DataFrame containing metric values
            metrics: List of metric names to plot
            thresholds: Dictionary of thresholds for each metric
            ncols: Number of columns in the grid
            figsize_per_plot: Size of each subplot
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        # Filter valid metrics
        valid_metrics = [m for m in metrics if m in data.columns]
        
        if not valid_metrics:
            raise VisualizationError("No valid metrics found in data")
        
        # Calculate grid dimensions
        n_metrics = len(valid_metrics)
        nrows = (n_metrics + ncols - 1) // ncols
        
        # Create figure
        figsize = (figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=self.dpi)
        
        # Flatten axes array
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Plot each metric
        for idx, metric in enumerate(valid_metrics):
            ax = axes[idx]
            
            # Get threshold
            threshold = thresholds.get(metric) if thresholds else None
            
            # Plot distribution
            self._plot_single_distribution(
                ax, data, metric, threshold
            )
        
        # Hide empty subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        # Overall title
        fig.suptitle('Quality Metric Distributions', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.log_info(f"Saved figure to {save_path}")
        
        return fig
    
    def _plot_single_distribution(self, ax, data, metric, threshold=None):
        """Plot distribution on a single axis."""
        values = data[metric].dropna()
        
        if len(values) == 0:
            ax.text(0.5, 0.5, f"No data", ha='center', va='center')
            ax.set_title(metric)
            return
        
        # Statistics
        mean_val = values.mean()
        std_val = values.std()
        
        # Histogram
        ax.hist(values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Statistical lines
        ax.axvline(mean_val, color='green', linestyle='-', linewidth=2)
        ax.axvline(mean_val - std_val, color='blue', linestyle=':', linewidth=1)
        ax.axvline(mean_val + std_val, color='blue', linestyle=':', linewidth=1)
        
        # Threshold
        if threshold is not None:
            ax.axvline(threshold, color='red', linestyle='--', linewidth=2)
            pass_rate = (values >= threshold).sum() / len(values) * 100
            ax.text(0.02, 0.95, f'{pass_rate:.0f}% pass',
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Labels
        ax.set_xlabel(metric)
        ax.set_ylabel('Count')
        ax.set_title(f'{metric}', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def plot_metric_comparison(self,
                             original_data: pd.DataFrame,
                             filtered_data: pd.DataFrame,
                             metrics: List[str],
                             figsize: Optional[Tuple[int, int]] = None,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare metric distributions before and after filtering.
        
        Args:
            original_data: Original dataset
            filtered_data: Filtered dataset
            metrics: Metrics to compare
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        valid_metrics = [m for m in metrics if m in original_data.columns]
        
        if not valid_metrics:
            raise VisualizationError("No valid metrics found")
        
        # Create figure
        fig, axes = plt.subplots(1, len(valid_metrics), 
                                figsize=figsize or (5 * len(valid_metrics), 5),
                                dpi=self.dpi)
        
        if len(valid_metrics) == 1:
            axes = [axes]
        
        # Plot each metric
        for idx, metric in enumerate(valid_metrics):
            ax = axes[idx]
            
            # Original data
            original_values = original_data[metric].dropna()
            ax.hist(original_values, bins=30, alpha=0.5, label='Original',
                   color='blue', density=True)
            
            # Filtered data
            if len(filtered_data) > 0 and metric in filtered_data.columns:
                filtered_values = filtered_data[metric].dropna()
                ax.hist(filtered_values, bins=30, alpha=0.5, label='Filtered',
                       color='green', density=True)
            
            # Statistics
            ax.axvline(original_values.mean(), color='blue', linestyle='--',
                      label=f'Orig mean: {original_values.mean():.3f}')
            
            if len(filtered_data) > 0 and metric in filtered_data.columns:
                ax.axvline(filtered_values.mean(), color='green', linestyle='--',
                          label=f'Filt mean: {filtered_values.mean():.3f}')
            
            ax.set_xlabel(metric)
            ax.set_ylabel('Density')
            ax.set_title(f'{metric} Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        fig.suptitle('Original vs Filtered Distributions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_threshold_impact(self,
                            data: pd.DataFrame,
                            metric: str,
                            threshold_range: Tuple[float, float],
                            n_points: int = 50,
                            figsize: Optional[Tuple[int, int]] = None,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize impact of different threshold values.
        
        Args:
            data: DataFrame with metric values
            metric: Metric to analyze
            threshold_range: (min, max) threshold values to test
            n_points: Number of threshold points to test
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        values = data[metric].dropna()
        
        # Generate threshold values
        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_points)
        
        # Calculate metrics for each threshold
        pass_rates = []
        sample_counts = []
        
        for thresh in thresholds:
            passed = values >= thresh
            pass_rates.append(passed.sum() / len(values) * 100)
            sample_counts.append(passed.sum())
        
        # Create figure with two y-axes
        fig, ax1 = plt.subplots(figsize=figsize or self.figsize, dpi=self.dpi)
        
        # Plot pass rate
        color = 'tab:blue'
        ax1.set_xlabel(f'{metric} Threshold', fontsize=12)
        ax1.set_ylabel('Pass Rate (%)', color=color, fontsize=12)
        line1 = ax1.plot(thresholds, pass_rates, color=color, linewidth=2,
                        label='Pass Rate')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)
        
        # Create second y-axis for sample count
        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Sample Count', color=color, fontsize=12)
        line2 = ax2.plot(thresholds, sample_counts, color=color, linewidth=2,
                        label='Sample Count')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Add current distribution stats
        ax1.axvline(values.mean(), color='green', linestyle='--', alpha=0.7,
                   label=f'Mean: {values.mean():.3f}')
        ax1.axvline(values.median(), color='red', linestyle='--', alpha=0.7,
                   label=f'Median: {values.median():.3f}')
        
        # Legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines + ax1.lines[1:], labels + ['Mean', 'Median'],
                  loc='best', frameon=True, fancybox=True)
        
        # Title
        ax1.set_title(f'Threshold Impact Analysis for {metric}',
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
