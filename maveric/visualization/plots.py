"""Additional plotting utilities for MAVERIC."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


def plot_class_distribution(data: pd.DataFrame,
                          column: str = 'label',
                          top_n: int = 20,
                          figsize: Tuple[int, int] = (12, 6),
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot class distribution as a bar chart.
    
    Args:
        data: DataFrame with class labels
        column: Column name containing class labels
        top_n: Number of top classes to show
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")
    
    # Count classes
    class_counts = data[column].value_counts()
    
    # Select top classes
    if len(class_counts) > top_n:
        top_classes = class_counts.head(top_n)
        other_count = class_counts[top_n:].sum()
        if other_count > 0:
            top_classes['Other'] = other_count
    else:
        top_classes = class_counts
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar plot
    bars = ax.bar(range(len(top_classes)), top_classes.values)
    
    # Color bars
    colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Labels
    ax.set_xticks(range(len(top_classes)))
    ax.set_xticklabels(top_classes.index, rotation=45, ha='right')
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Class Distribution (Top {min(top_n, len(class_counts))} Classes)',
                fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, top_classes.values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(top_classes.values),
               f'{count:,}', ha='center', va='bottom', fontsize=9)
    
    # Add grid
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_correlation_matrix(data: pd.DataFrame,
                          metrics: Optional[List[str]] = None,
                          figsize: Tuple[int, int] = (10, 8),
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot correlation matrix of quality metrics.
    
    Args:
        data: DataFrame with metric columns
        metrics: List of metrics to include (None for all numeric)
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Select metrics
    if metrics is None:
        # Use all numeric columns that look like metrics
        metrics = [col for col in data.select_dtypes(include=[np.number]).columns
                  if 'score' in col or 'consistency' in col]
    
    # Filter valid metrics
    valid_metrics = [m for m in metrics if m in data.columns]
    
    if len(valid_metrics) < 2:
        raise ValueError("Need at least 2 metrics for correlation matrix")
    
    # Calculate correlation
    corr_matrix = data[valid_metrics].corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
               cmap='coolwarm', center=0, square=True,
               linewidths=1, cbar_kws={"shrink": 0.8},
               ax=ax)
    
    # Labels
    ax.set_title('Quality Metrics Correlation Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_quality_comparison(original_stats: Dict[str, Any],
                          filtered_stats: Dict[str, Any],
                          figsize: Tuple[int, int] = (12, 8),
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Create comprehensive comparison plot of original vs filtered data.
    
    Args:
        original_stats: Statistics from original data
        filtered_stats: Statistics from filtered data
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Sample counts
    ax1 = fig.add_subplot(gs[0, 0])
    counts = [original_stats['total_samples'], filtered_stats['filtered_samples']]
    bars = ax1.bar(['Original', 'Filtered'], counts, color=['blue', 'green'])
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Dataset Size Comparison')
    
    # Add labels
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(counts),
                f'{count:,}', ha='center', va='bottom')
    
    # 2. Retention rate pie chart
    ax2 = fig.add_subplot(gs[0, 1])
    retention = filtered_stats.get('retention_rate', 0) * 100
    sizes = [retention, 100 - retention]
    colors = ['green', 'red']
    labels = [f'Retained\n{retention:.1f}%', f'Filtered out\n{100-retention:.1f}%']
    ax2.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90)
    ax2.set_title('Data Retention')
    
    # 3. Metric comparison
    ax3 = fig.add_subplot(gs[1, :])
    
    # Get common metrics
    orig_metrics = original_stats.get('metric_statistics', {})
    filt_metrics = filtered_stats.get('metric_statistics', {})
    common_metrics = set(orig_metrics.keys()) & set(filt_metrics.keys())
    
    if common_metrics:
        metric_names = sorted(list(common_metrics))
        x = np.arange(len(metric_names))
        width = 0.35
        
        # Extract means
        orig_means = [orig_metrics[m]['mean'] for m in metric_names]
        filt_means = [filt_metrics[m]['mean'] for m in metric_names]
        
        # Create bars
        bars1 = ax3.bar(x - width/2, orig_means, width, label='Original', color='blue', alpha=0.7)
        bars2 = ax3.bar(x + width/2, filt_means, width, label='Filtered', color='green', alpha=0.7)
        
        # Labels
        ax3.set_xlabel('Metrics')
        ax3.set_ylabel('Mean Value')
        ax3.set_title('Mean Metric Values Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels([m.replace('_', ' ').title() for m in metric_names], rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, axis='y', alpha=0.3)
    
    plt.suptitle('Quality Control Results Summary', fontsize=16, fontweight='bold')
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_summary_report(quality_result: 'QualityResult',
                        output_dir: str,
                        include_samples: bool = True) -> Dict[str, str]:
    """
    Create a comprehensive summary report with multiple visualizations.
    
    Args:
        quality_result: QualityResult object
        output_dir: Directory to save report files
        include_samples: Whether to include sample visualizations
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    # Convert to DataFrames
    original_df = pd.DataFrame(quality_result.original_samples)
    filtered_df = pd.DataFrame(quality_result.filtered_samples)
    
    # 1. Class distribution
    if 'label' in filtered_df.columns:
        fig = plot_class_distribution(filtered_df)
        path = output_dir / 'class_distribution.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_files['class_distribution'] = str(path)
    
    # 2. Quality metrics correlation
    metrics = [col for col in filtered_df.columns if 'score' in col or 'consistency' in col]
    if len(metrics) >= 2:
        fig = plot_correlation_matrix(filtered_df, metrics)
        path = output_dir / 'correlation_matrix.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_files['correlation_matrix'] = str(path)
    
    # 3. Summary statistics
    fig = plot_quality_comparison(
        {'total_samples': len(original_df), 'metric_statistics': {}},
        {'filtered_samples': len(filtered_df), 'retention_rate': quality_result.retention_rate,
         'metric_statistics': {}}
    )
    path = output_dir / 'summary_stats.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_files['summary_stats'] = str(path)
    
    # 4. Sample visualizations
    if include_samples and len(filtered_df) > 0:
        from .samples import SampleVisualizer
        viz = SampleVisualizer()
        
        # Best samples
        fig = viz.visualize_samples(filtered_df, n_samples=5, sample_type='best')
        path = output_dir / 'best_samples.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_files['best_samples'] = str(path)
        
        # Diverse samples
        fig = viz.visualize_samples(filtered_df, n_samples=5, sample_type='diverse')
        path = output_dir / 'diverse_samples.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_files['diverse_samples'] = str(path)
    
    return saved_files
