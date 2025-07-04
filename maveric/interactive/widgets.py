"""Widget creation utilities for interactive components."""

import ipywidgets as widgets
from typing import Dict, List, Optional, Tuple, Callable


def create_threshold_widget(metric_name: str,
                          default_value: float,
                          min_value: float = 0.0,
                          max_value: float = 1.0,
                          step: float = 0.001,
                          description: Optional[str] = None) -> widgets.FloatSlider:
    """
    Create a threshold adjustment widget.
    
    Args:
        metric_name: Name of the metric
        default_value: Default threshold value
        min_value: Minimum value
        max_value: Maximum value
        step: Step size
        description: Display description (defaults to formatted metric name)
        
    Returns:
        Configured FloatSlider widget
    """
    if description is None:
        description = metric_name.replace('_', ' ').title()
    
    slider = widgets.FloatSlider(
        value=default_value,
        min=min_value,
        max=max_value,
        step=step,
        description=description,
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.3f',
        layout=widgets.Layout(width='500px'),
        style={'description_width': '150px'}
    )
    
    slider.tooltip = f"Threshold for {metric_name}"
    
    return slider


def create_weight_widget(metric_name: str,
                        default_value: float,
                        description: Optional[str] = None) -> widgets.FloatSlider:
    """
    Create a weight adjustment widget.
    
    Args:
        metric_name: Name of the metric
        default_value: Default weight value
        description: Display description
        
    Returns:
        Configured FloatSlider widget
    """
    if description is None:
        description = metric_name.upper()
    
    slider = widgets.FloatSlider(
        value=default_value,
        min=0.0,
        max=1.0,
        step=0.01,
        description=description,
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
        layout=widgets.Layout(width='400px'),
        style={'description_width': '100px'}
    )
    
    slider.tooltip = f"Weight for {metric_name}"
    
    return slider


def create_metric_selector(available_metrics: List[str],
                         default_selection: Optional[List[str]] = None,
                         max_height: str = '200px') -> widgets.SelectMultiple:
    """
    Create a metric selection widget.
    
    Args:
        available_metrics: List of available metric names
        default_selection: Metrics to select by default
        max_height: Maximum height of the widget
        
    Returns:
        Configured SelectMultiple widget
    """
    if default_selection is None:
        # Select first 3 metrics by default
        default_selection = available_metrics[:3]
    
    selector = widgets.SelectMultiple(
        options=available_metrics,
        value=default_selection,
        description='Metrics:',
        rows=min(10, len(available_metrics)),
        layout=widgets.Layout(width='300px', max_height=max_height)
    )
    
    return selector


def create_progress_bar(description: str = "Processing") -> Tuple[widgets.IntProgress, widgets.Label]:
    """
    Create a progress bar with label.
    
    Args:
        description: Description text
        
    Returns:
        Tuple of (progress_bar, label)
    """
    progress = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        description=description,
        bar_style='info',
        style={'bar_color': '#00ff00'},
        orientation='horizontal',
        layout=widgets.Layout(width='500px')
    )
    
    label = widgets.Label(value='0%')
    
    return progress, label


def create_status_widget(initial_text: str = "Ready") -> widgets.HTML:
    """
    Create a status display widget.
    
    Args:
        initial_text: Initial status text
        
    Returns:
        HTML widget for status display
    """
    return widgets.HTML(
        value=f'<div style="background-color: #f0f0f0; padding: 10px; '
              f'border-radius: 5px; font-weight: bold;">{initial_text}</div>'
    )


def create_file_upload_widget(accept: str = '.json,.csv',
                            description: str = 'Upload file:') -> widgets.FileUpload:
    """
    Create a file upload widget.
    
    Args:
        accept: File types to accept
        description: Widget description
        
    Returns:
        FileUpload widget
    """
    return widgets.FileUpload(
        accept=accept,
        multiple=False,
        description=description,
        layout=widgets.Layout(width='300px')
    )


def create_output_area(height: str = '400px') -> widgets.Output:
    """
    Create an output area widget.
    
    Args:
        height: Height of the output area
        
    Returns:
        Output widget
    """
    return widgets.Output(
        layout=widgets.Layout(
            height=height,
            overflow_y='auto',
            border='1px solid #ddd',
            padding='10px'
        )
    )