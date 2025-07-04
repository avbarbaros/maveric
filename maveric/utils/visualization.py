"""Visualization utility functions."""

import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


def create_figure_grid(n_items: int, 
                      n_cols: int = 3,
                      figsize_per_item: Tuple[float, float] = (4, 4)) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a figure with grid of subplots.
    
    Args:
        n_items: Number of items to plot
        n_cols: Number of columns
        figsize_per_item: Size per subplot
        
    Returns:
        Tuple of (figure, axes_array)
    """
    n_rows = (n_items + n_cols - 1) // n_cols
    
    figsize = (figsize_per_item[0] * n_cols, figsize_per_item[1] * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes
    if n_items == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    
    # Hide extra axes
    for i in range(n_items, len(axes)):
        axes[i].set_visible(False)
    
    return fig, axes


def save_figure(fig: plt.Figure, 
                path: str,
                dpi: int = 150,
                bbox_inches: str = 'tight',
                **kwargs):
    """
    Save matplotlib figure.
    
    Args:
        fig: Figure to save
        path: Output path
        dpi: Resolution
        bbox_inches: Bounding box setting
        **kwargs: Additional arguments for savefig
    """
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
    plt.close(fig)


def plot_history(history: Dict[str, List[float]],
                 metrics: Optional[List[str]] = None,
                 figsize: Tuple[float, float] = (12, 4)) -> plt.Figure:
    """
    Plot training history.
    
    Args:
        history: Dictionary of metric histories
        metrics: Metrics to plot (plots all if None)
        figsize: Figure size
        
    Returns:
        Figure object
    """
    if metrics is None:
        metrics = list(history.keys())
    
    # Separate into loss and accuracy metrics
    loss_metrics = [m for m in metrics if 'loss' in m]
    acc_metrics = [m for m in metrics if 'acc' in m]
    
    n_plots = (1 if loss_metrics else 0) + (1 if acc_metrics else 0)
    
    if n_plots == 0:
        return None
    
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Plot loss
    if loss_metrics:
        ax = axes[plot_idx]
        for metric in loss_metrics:
            if metric in history:
                ax.plot(history[metric], label=metric)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot accuracy
    if acc_metrics:
        ax = axes[plot_idx]
        for metric in acc_metrics:
            if metric in history:
                ax.plot(history[metric], label=metric)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Training Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
