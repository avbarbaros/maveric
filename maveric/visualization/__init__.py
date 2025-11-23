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
    from .interactive import (
        MAVERICInteractiveQualityControl,
        create_quality_control,
        start_interactive_gui
    )
    INTERACTIVE_AVAILABLE = True
    
    def check_interactive_requirements():
        """Check if interactive GUI requirements are met"""
        try:
            import ipywidgets
            from IPython.display import display
            return True
        except ImportError as e:
            print(f"❌ Missing requirement for interactive GUI: {e}")
            print("📦 Please install: pip install ipywidgets")
            print("🔧 For Colab: !pip install ipywidgets")
            return False

except ImportError as e:
    INTERACTIVE_AVAILABLE = False
    MAVERICInteractiveQualityControl = None
    
    def create_quality_control(*args, **kwargs):
        """Fallback function when interactive GUI is not available"""
        print(f"❌ Interactive GUI not available: {e}")
        print("📦 Please install required packages:")
        print("   !pip install ipywidgets")
        print("   !jupyter nbextension enable --py widgetsnbextension")
        print("   # Then restart kernel")
        return None
    
    def start_interactive_gui(*args, **kwargs):
        """Fallback function when interactive GUI is not available"""
        return create_quality_control(*args, **kwargs)

    def check_interactive_requirements():
        return False

__all__ = [
    "MetricsVisualizer",
    "SampleVisualizer",
    "plot_class_distribution",
    "plot_correlation_matrix",
    "plot_quality_comparison",
    "create_summary_report",
    "MAVERICInteractiveQualityControl",
    "create_quality_control",
    "start_interactive_gui",
    "check_interactive_requirements",
    "INTERACTIVE_AVAILABLE"
]