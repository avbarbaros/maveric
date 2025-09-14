"""Interactive threshold selection GUI for Jupyter notebooks."""

import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Any
import matplotlib.pyplot as plt

from ..core.base import BaseComponent
from ..quality.quality_controller import QualityController
from ..visualization.distributions import MetricsVisualizer
from ..visualization.samples import SampleVisualizer


class InteractiveThresholdSelector(BaseComponent):
    """
    Interactive GUI for selecting quality thresholds with real-time feedback.
    
    This component helps researchers find optimal thresholds by seeing
    immediate effects on data filtering through an intuitive interface.
    """
    
    def __init__(self, quality_controller: QualityController):
        """
        Initialize interactive threshold selector.
        
        Args:
            quality_controller: QualityController instance with loaded data
        """
        super().__init__("InteractiveThresholdSelector")
        self.qc = quality_controller
        self.widgets = {
            'thresholds': {},
            'weights': {}
        }
        self.output = widgets.Output()
        self.status_html = None
        self.visualizers = {
            'metrics': MetricsVisualizer(),
            'samples': SampleVisualizer()
        }
        
        # Callbacks
        self.on_filter_change = []
    
    def create_gui(self) -> widgets.VBox:
        """
        Create the complete interactive GUI.
        
        Returns:
            IPython widgets VBox containing the full interface
        """
        # Create main components
        threshold_widgets = self._create_threshold_widgets()
        weight_widgets = self._create_weight_widgets()
        control_widgets = self._create_control_widgets()
        
        # Store widgets for later access
        self.widgets['thresholds'] = threshold_widgets
        self.widgets['weights'] = weight_widgets
        
        # Create tabs
        tab = widgets.Tab()
        tab.children = [
            self._create_threshold_tab(threshold_widgets),
            self._create_weight_tab(weight_widgets),
            self._create_visualization_tab()
        ]
        tab.set_title(0, 'Quality Thresholds')
        tab.set_title(1, 'Class Weights')
        tab.set_title(2, 'Live Preview')
        
        # Status display
        self.status_html = widgets.HTML(
            value=self._get_status_html(len(self.qc.data), len(self.qc.data))
        )
        
        # Header
        header = widgets.HTML(
            "<h3>MAVERIC Interactive Quality Control</h3>"
            "<p>Adjust thresholds and weights to filter your dataset. "
            "The visualization updates in real-time to show the effects.</p>"
        )
        
        # Combine all widgets
        gui = widgets.VBox([
            header,
            self.status_html,
            tab,
            control_widgets,
            self.output
        ])
        
        # Show initial visualization
        self._update_visualizations()
        
        return gui
    
    def _create_threshold_widgets(self) -> Dict[str, widgets.FloatSlider]:
        """Create sliders for each quality threshold."""
        threshold_widgets = {}
        
        for metric, default_value in self.qc.thresholds.items():
            # Determine range based on metric
            if 'resolution' in metric:
                max_value = 5.0
                step = 0.01
            else:
                max_value = 1.0
                step = 0.001
            
            # Create slider with custom styling
            slider = widgets.FloatSlider(
                value=default_value,
                min=0.0,
                max=max_value,
                step=step,
                description=self._format_metric_name(metric),
                disabled=False,
                continuous_update=False,
                orientation='horizontal',
                readout=True,
                readout_format='.3f',
                layout=widgets.Layout(width='600px'),
                style={'description_width': '200px'}
            )
            
            # Add hover text
            slider.tooltip = f"Threshold for {metric}"
            
            # Add observer for real-time preview
            slider.observe(self._on_threshold_change, names='value')
            
            threshold_widgets[metric] = slider
        
        return threshold_widgets
    
    def _create_class_selection_widgets(self) -> widgets.VBox:
        """Create widgets for class selection weight configuration."""
        class_selection_weights = self.qc.get_class_selection_weights()
        
        # Similarity vs Quality weight sliders
        similarity_slider = widgets.FloatSlider(
            value=class_selection_weights['similarity_weight'],
            min=0.0,
            max=1.0,
            step=0.05,
            description='Similarity Weight',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.2f',
            layout=widgets.Layout(width='500px'),
            style={'description_width': '150px'}
        )
        
        quality_slider = widgets.FloatSlider(
            value=class_selection_weights['quality_weight'], 
            min=0.0,
            max=1.0,
            step=0.05,
            description='Quality Weight',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.2f',
            layout=widgets.Layout(width='500px'),
            style={'description_width': '150px'}
        )
        
        # Store references for later use
        self.similarity_weight_slider = similarity_slider
        self.quality_weight_slider = quality_slider
        
        # Add observers for automatic normalization
        def on_similarity_change(change):
            # Auto-adjust quality weight
            quality_weight = 1.0 - change['new']
            self.quality_weight_slider.value = quality_weight
            self.qc.set_class_selection_weight('similarity_weight', change['new'])
            
        def on_quality_change(change):
            # Auto-adjust similarity weight
            similarity_weight = 1.0 - change['new']
            self.similarity_weight_slider.value = similarity_weight
            self.qc.set_class_selection_weight('quality_weight', change['new'])
        
        similarity_slider.observe(on_similarity_change, names='value')
        quality_slider.observe(on_quality_change, names='value')
        
        # Explanation
        explanation = widgets.HTML(
            "<div style='background-color: #e8f4f8; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>"
            "<b>Class Selection Strategy:</b><br>"
            "• <b>Similarity Weight</b>: Emphasizes similarity to reference images/text<br>"
            "• <b>Quality Weight</b>: Emphasizes universal semantic quality<br>"
            "• Weights automatically sum to 1.0<br>"
            "• Higher similarity weight = more class-specific matching<br>"
            "• Higher quality weight = better overall sample quality"
            "</div>"
        )
        
        return widgets.VBox([
            explanation,
            widgets.HTML("<h4>Class Selection Weights</h4>"),
            similarity_slider,
            quality_slider
        ])
    
    def _create_weight_widgets(self) -> Dict[str, widgets.FloatSlider]:
        """Create sliders for class similarity weights."""
        weight_widgets = {}
        
        for metric, default_value in self.qc.class_weights.items():
            slider = widgets.FloatSlider(
                value=default_value,
                min=0.0,
                max=1.0,
                step=0.01,
                description=metric.upper(),
                disabled=False,
                continuous_update=False,
                orientation='horizontal',
                readout=True,
                readout_format='.2f',
                layout=widgets.Layout(width='500px'),
                style={'description_width': '150px'}
            )
            
            slider.tooltip = f"Weight for {metric} similarity"
            slider.observe(self._on_weight_change, names='value')
            
            weight_widgets[metric] = slider
        
        return weight_widgets
    
    def _create_control_widgets(self) -> widgets.HBox:
        """Create control buttons."""
        # Apply button
        apply_button = widgets.Button(
            description='Apply Filters',
            button_style='primary',
            icon='check',
            layout=widgets.Layout(width='150px')
        )
        apply_button.on_click(self._on_apply_clicked)
        
        # Reset button
        reset_button = widgets.Button(
            description='Reset to Defaults',
            button_style='warning',
            icon='refresh',
            layout=widgets.Layout(width='150px')
        )
        reset_button.on_click(self._on_reset_clicked)
        
        # Export button
        export_button = widgets.Button(
            description='Export Config',
            button_style='info',
            icon='download',
            layout=widgets.Layout(width='150px')
        )
        export_button.on_click(self._on_export_clicked)
        
        # Visualize button
        visualize_button = widgets.Button(
            description='Show Samples',
            button_style='success',
            icon='image',
            layout=widgets.Layout(width='150px')
        )
        visualize_button.on_click(self._on_visualize_clicked)
        
        return widgets.HBox([
            apply_button,
            reset_button,
            export_button,
            visualize_button
        ], layout=widgets.Layout(justify_content='center'))
    
    def _create_threshold_tab(self, threshold_widgets: Dict) -> widgets.VBox:
        """Create the threshold adjustment tab."""
        # Group related thresholds
        score_widgets = widgets.VBox([
            widgets.HTML("<h4>Score Thresholds</h4>"),
            threshold_widgets.get('weighted_class_score', widgets.Label('N/A')),
            threshold_widgets.get('consistency', widgets.Label('N/A')),
            threshold_widgets.get('imagenet_probability', widgets.Label('N/A'))
        ])

        quality_widgets = widgets.VBox([
            widgets.HTML("<h4>Quality Thresholds</h4>"),
            *[w for k, w in threshold_widgets.items()
              if k not in ['weighted_class_score', 'consistency', 'imagenet_probability']]
        ])
        
        # Add explanatory text
        explanation = widgets.HTML(
            "<div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'>"
            "<b>Tips:</b><br>"
            "• Higher thresholds = stricter filtering (fewer samples)<br>"
            "• Start with default values and adjust based on preview<br>"
            "• Watch the retention rate as you adjust thresholds"
            "</div>"
        )
        
        return widgets.VBox([
            explanation,
            widgets.HBox([score_widgets, quality_widgets],
                        layout=widgets.Layout(justify_content='space-around'))
        ])
    
    def _create_weight_tab(self, weight_widgets: Dict) -> widgets.VBox:
        """Create the weight adjustment tab."""
        # Class selection weights (similarity vs quality)
        class_selection_widgets = self._create_class_selection_widgets()
        
        # Normalize button for similarity weights
        normalize_button = widgets.Button(
            description='Normalize Similarity Weights',
            button_style='info',
            icon='balance-scale'
        )
        normalize_button.on_click(self._normalize_weights)
        
        # Weight display
        weight_sum_label = widgets.Label(
            value=f"Total similarity weight: {sum(self.qc.class_weights.values()):.2f}"
        )
        self.weight_sum_label = weight_sum_label
        
        # Explanation
        explanation = widgets.HTML(
            "<div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'>"
            "<b>Class Similarity Weights:</b><br>"
            "• img2img: How similar the image is to reference images<br>"
            "• txt2txt: How similar the caption is to reference captions<br>"
            "• img2txt: How well the image matches reference text<br>"
            "• txt2img: How well the caption matches reference images<br>"
            "<br>Weights should sum to 1.0 for proper scoring."
            "</div>"
        )
        
        # Layout - combine class selection and similarity weights
        similarity_weight_controls = widgets.VBox([
            widgets.HTML("<h4>Similarity Component Weights</h4>"),
            *weight_widgets.values(),
            widgets.HBox([normalize_button, weight_sum_label])
        ])
        
        return widgets.VBox([
            class_selection_widgets,  # Class selection strategy at the top
            widgets.HTML("<hr>"),     # Visual separator
            explanation,              # Original explanation
            similarity_weight_controls # Similarity weights at the bottom
        ])
    
    def _create_visualization_tab(self) -> widgets.VBox:
        """Create the live preview tab."""
        # Options
        preview_options = widgets.HBox([
            widgets.IntSlider(
                value=5,
                min=1,
                max=20,
                description='Preview samples:',
                style={'description_width': '120px'}
            ),
            widgets.Dropdown(
                options=['random', 'best', 'worst', 'diverse'],
                value='diverse',
                description='Sample type:',
                style={'description_width': '100px'}
            )
        ])
        
        # Preview output
        preview_output = widgets.Output()
        
        # Update button
        update_preview = widgets.Button(
            description='Update Preview',
            button_style='primary'
        )
        
        def update_preview_callback(b):
            with preview_output:
                clear_output(wait=True)
                n_samples = preview_options.children[0].value
                sample_type = preview_options.children[1].value
                
                if self.qc.filtered_data is not None and len(self.qc.filtered_data) > 0:
                    fig = self.visualizers['samples'].visualize_samples(
                        self.qc.filtered_data,
                        n_samples=n_samples,
                        sample_type=sample_type
                    )
                    plt.show()
                else:
                    print("No filtered data available for preview")
        
        update_preview.on_click(update_preview_callback)
        
        return widgets.VBox([
            widgets.HTML("<h4>Live Preview of Filtered Samples</h4>"),
            preview_options,
            update_preview,
            preview_output
        ])
    
    def _format_metric_name(self, metric: str) -> str:
        """Format metric name for display."""
        # Replace underscores with spaces and capitalize
        formatted = metric.replace('_', ' ').title()
        
        # Special cases
        replacements = {
            'Weighted Class Score': 'Class Score',
            'Resolution Score': 'Resolution',
            'Sharpness Score': 'Sharpness',
            'Color Score': 'Color Diversity',
            'Composite Quality': 'Semantic Quality',
            'Imagenet Probability': 'ImageNet Quality'
        }
        
        return replacements.get(formatted, formatted)
    
    def _get_status_html(self, original_count: int, filtered_count: int) -> str:
        """Generate status HTML."""
        retention_rate = (filtered_count / original_count * 100) if original_count > 0 else 0
        
        # Color based on retention rate
        if retention_rate > 80:
            color = 'green'
        elif retention_rate > 50:
            color = 'orange'
        else:
            color = 'red'
        
        return (
            f"<div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px;'>"
            f"<b>Dataset Status:</b> "
            f"{original_count:,} → {filtered_count:,} samples "
            f"(<span style='color: {color};'>{retention_rate:.1f}% retained</span>)"
            f"</div>"
        )
    
    def _on_threshold_change(self, change):
        """Handle threshold slider change (preview only)."""
        # Update preview count in status
        if self.qc.data is not None:
            # Estimate filtered count without full processing
            preview_count = self._estimate_filtered_count()
            self.status_html.value = self._get_status_html(
                len(self.qc.data), 
                preview_count
            ) + " <i>(preview)</i>"
    
    def _on_weight_change(self, change):
        """Handle weight slider change."""
        # Update weight sum display
        total = sum(w.value for w in self.widgets['weights'].values())
        self.weight_sum_label.value = f"Total weight: {total:.2f}"
        
        # Color code based on sum
        if abs(total - 1.0) < 0.01:
            self.weight_sum_label.style = {'text_color': 'green'}
        else:
            self.weight_sum_label.style = {'text_color': 'red'}
    
    def _estimate_filtered_count(self) -> int:
        """Quickly estimate filtered count without full processing."""
        if self.qc.data is None:
            return 0
        
        # Simple estimation based on current widget values
        count = len(self.qc.data)
        
        # Apply each threshold
        for metric, widget in self.widgets['thresholds'].items():
            if metric in self.qc.data.columns:
                passing = (self.qc.data[metric] >= widget.value).sum()
                count = min(count, passing)
        
        return count
    
    def _on_apply_clicked(self, button):
        """Handle Apply button click."""
        with self.output:
            clear_output(wait=True)
            
            # Update thresholds
            for metric, widget in self.widgets['thresholds'].items():
                self.qc.set_threshold(metric, widget.value)
            
            # Update weights
            for metric, widget in self.widgets['weights'].items():
                self.qc.set_class_weight(metric, widget.value)
            
            # Apply filtering
            filtered_count = self.qc.apply_thresholds()
            
            # Update status
            self.status_html.value = self._get_status_html(
                len(self.qc.data),
                filtered_count
            )
            
            # Update visualizations
            self._update_visualizations()
            
            # Trigger callbacks
            for callback in self.on_filter_change:
                callback(self.qc)
    
    def _on_reset_clicked(self, button):
        """Handle Reset button click."""
        # Reset to default values
        default_thresholds = {
            'weighted_class_score': 0.493,
            'consistency': 0.796,
            'imagenet_probability': 0.5,
            'resolution_score': 0.370,
            'sharpness_score': 0.880,
            'color_score': 0.768
        }
        
        default_weights = {
            'img2img': 0.40,
            'txt2txt': 0.20,
            'img2txt': 0.20,
            'txt2img': 0.20
        }
        
        # Update widgets
        for metric, widget in self.widgets['thresholds'].items():
            if metric in default_thresholds:
                widget.value = default_thresholds[metric]
        
        for metric, widget in self.widgets['weights'].items():
            if metric in default_weights:
                widget.value = default_weights[metric]
        
        # Reset class selection weights to defaults
        if hasattr(self, 'similarity_weight_slider'):
            self.similarity_weight_slider.value = 0.7
        if hasattr(self, 'quality_weight_slider'):
            self.quality_weight_slider.value = 0.3
        
        # Apply
        self._on_apply_clicked(button)
    
    def _on_export_clicked(self, button):
        """Handle Export button click."""
        config = {
            'thresholds': {
                metric: widget.value 
                for metric, widget in self.widgets['thresholds'].items()
            },
            'weights': {
                metric: widget.value 
                for metric, widget in self.widgets['weights'].items()
            },
            'class_selection_weights': self.qc.get_class_selection_weights()
        }
        
        with self.output:
            clear_output(wait=True)
            print("# Current Configuration")
            print("```python")
            print("config = {")
            print("    'thresholds': {")
            for k, v in config['thresholds'].items():
                print(f"        '{k}': {v:.3f},")
            print("    },")
            print("    'weights': {")
            for k, v in config['weights'].items():
                print(f"        '{k}': {v:.2f},")
            print("    },")
            print("    'class_selection_weights': {")
            for k, v in config['class_selection_weights'].items():
                print(f"        '{k}': {v:.2f},")
            print("    }")
            print("}")
            print("```")
    
    def _on_visualize_clicked(self, button):
        """Handle Visualize button click."""
        with self.output:
            clear_output(wait=True)
            
            if self.qc.filtered_data is not None and len(self.qc.filtered_data) > 0:
                # Show sample images
                fig = self.visualizers['samples'].visualize_samples(
                    self.qc.filtered_data,
                    n_samples=8,
                    sample_type='diverse'
                )
                plt.show()
                
                # Show class distribution
                if 'label' in self.qc.filtered_data.columns:
                    from ..visualization.plots import plot_class_distribution
                    fig = plot_class_distribution(self.qc.filtered_data, top_n=15)
                    plt.show()
            else:
                print("No filtered data available")
    
    def _normalize_weights(self, button):
        """Normalize weights to sum to 1.0."""
        total = sum(w.value for w in self.widgets['weights'].values())
        
        if total > 0:
            for widget in self.widgets['weights'].values():
                widget.value = widget.value / total
    
    def _update_visualizations(self):
        """Update metric distribution visualizations."""
        with self.output:
            clear_output(wait=True)
            
            if self.qc.data is None:
                return
            
            # Select key metrics to visualize
            metrics_to_show = []
            for metric in ['weighted_class_score', 'consistency', 'imagenet_probability', 'sharpness_score']:
                if metric in self.qc.data.columns:
                    metrics_to_show.append(metric)
            
            if metrics_to_show:
                # Create multi-metric plot
                fig = self.visualizers['metrics'].plot_multi_metric_distributions(
                    self.qc.data,
                    metrics_to_show,
                    self.qc.thresholds,
                    ncols=len(metrics_to_show)
                )
                plt.show()
                
                # Show statistics
                stats = self.qc.get_statistics()
                print("\n📊 Dataset Statistics:")
                print(f"Total samples: {stats['total_samples']:,}")
                print(f"Filtered samples: {stats['filtered_samples']:,}")
                print(f"Retention rate: {stats['retention_rate']:.1%}")

                # Show individual threshold statistics
                threshold_stats = self.qc.get_threshold_statistics()
                if threshold_stats:
                    print("\n🎚️ Individual Threshold Statistics:")
                    for metric, stat in threshold_stats.items():
                        metric_display = self._format_metric_name(metric)
                        print(f"{metric_display}:")
                        print(f"  Threshold: {stat['threshold']:.3f}")
                        print(f"  Pass rate: {stat['pass_rate']:.1f}% ({stat['passing_samples']:,}/{stat['total_samples']:,})")
                        print(f"  Samples filtered out: {stat['samples_filtered_out']:,}")
                        print()
    
    def add_callback(self, callback: Callable):
        """
        Add callback for filter changes.
        
        Args:
            callback: Function to call when filters are applied
        """
        self.on_filter_change.append(callback)
