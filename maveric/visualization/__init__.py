"""Visualization tools for MAVERIC."""

from .distributions import MetricsVisualizer
from .samples import SampleVisualizer
from .plots import (
    plot_class_distribution,
    plot_correlation_matrix,
    plot_quality_comparison,
    create_summary_report
)

# Interactive GUI (requires ipywidgets)
try:
    from .interactive import InteractiveDataCuration, create_interactive_gui
    INTERACTIVE_AVAILABLE = True
except ImportError:
    INTERACTIVE_AVAILABLE = False
    InteractiveDataCuration = None
    create_interactive_gui = None

__all__ = [
    "MetricsVisualizer",
    "SampleVisualizer",
    "plot_class_distribution",
    "plot_correlation_matrix",
    "plot_quality_comparison",
    "create_summary_report",
    "InteractiveDataCuration",
    "create_interactive_gui",
    "INTERACTIVE_AVAILABLE"
]