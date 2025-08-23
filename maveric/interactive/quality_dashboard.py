"""Comprehensive quality control dashboard."""

import ipywidgets as widgets
from IPython.display import display
import pandas as pd
from typing import Union, Optional
from pathlib import Path

from ..core.base import BaseComponent
from ..core.interfaces import RetrievalResult, QualityResult
from ..quality.quality_controller import QualityController
from .threshold_selector import InteractiveThresholdSelector
from ..visualization import MetricsVisualizer, SampleVisualizer


class QualityDashboard(BaseComponent):
    """
    Comprehensive dashboard combining all visualization and interaction features.
    
    This provides a complete interface for quality control analysis with
    multiple tabs for different aspects of the data.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize quality dashboard.
        
        Args:
            cache_dir: Directory for saving/loading data
        """
        super().__init__("QualityDashboard")
        self.cache_dir = Path(cache_dir) if cache_dir else Path.cwd()
        self.qc = None
        self.selector = None
        self.visualizers = {
            'metrics': MetricsVisualizer(),
            'samples': SampleVisualizer()
        }
    
    def launch(self, 
               data: Union[str, pd.DataFrame, RetrievalResult, QualityResult],
               auto_balance: bool = True) -> widgets.Widget:
        """
        Launch the full interactive dashboard.
        
        Args:
            data: Data source (file path, DataFrame, or Result object)
            auto_balance: Whether to show balance options
            
        Returns:
            Interactive dashboard widget
        """
        # Load data
        data_df = self._load_data(data)
        
        # Create quality controller
        self.qc = QualityController(data_df)
        
        # Create interactive selector
        self.selector = InteractiveThresholdSelector(self.qc)
        
        # Create dashboard layout
        dashboard = self._create_dashboard_layout(auto_balance)
        
        return dashboard
    
    def _load_data(self, data: Union[str, pd.DataFrame, RetrievalResult, QualityResult]) -> pd.DataFrame:
        """Load data from various sources."""
        if isinstance(data, str):
            # Load from file
            path = Path(data)
            if path.suffix == '.json':
                import json
                with open(path, 'r') as f:
                    loaded = json.load(f)
                return pd.DataFrame(loaded)
            elif path.suffix == '.csv':
                return pd.read_csv(path)
            else:
                raise ValueError(f"Unsupported file type: {path.suffix}")
        
        elif isinstance(data, pd.DataFrame):
            return data
        
        elif isinstance(data, RetrievalResult):
            return data.to_dataframe()
        
        elif isinstance(data, QualityResult):
            return pd.DataFrame(data.filtered_samples)
        
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _create_dashboard_layout(self, auto_balance: bool) -> widgets.Tab:
        """Create the complete dashboard layout."""
        # Tab 1: Threshold Selection
        threshold_tab = self.selector.create_gui()
        
        # Tab 2: Data Explorer
        explorer_tab = self._create_explorer_tab()
        
        # Tab 3: Sample Gallery
        gallery_tab = self._create_gallery_tab()
        
        # Tab 4: Balance & Export
        export_tab = self._create_export_tab(auto_balance)
        
        # Tab 5: Statistics
        stats_tab = self._create_statistics_tab()
        
        # Combine into main tab widget
        main_tabs = widgets.Tab()
        main_tabs.children = [
            threshold_tab,
            explorer_tab,
            gallery_tab,
            export_tab,
            stats_tab
        ]
        
        main_tabs.set_title(0, '🎚️ Threshold Selection')
        main_tabs.set_title(1, '📊 Data Explorer')
        main_tabs.set_title(2, '🖼️ Sample Gallery')
        main_tabs.set_title(3, '⚖️ Balance & Export')
        main_tabs.set_title(4, '📈 Statistics')
        
        return main_tabs
    
    def _create_explorer_tab(self) -> widgets.VBox:
        """Create data exploration tab."""
        # Metric selector
        available_metrics = [col for col in self.qc.data.columns 
                           if 'score' in col or 'consistency' in col]
        
        metric_selector = widgets.SelectMultiple(
            options=available_metrics,
            value=available_metrics[:3] if len(available_metrics) >= 3 else available_metrics,
            description='Metrics:',
            rows=min(10, len(available_metrics)),
            layout=widgets.Layout(width='300px')
        )
        
        # Plot type selector
        plot_type = widgets.RadioButtons(
            options=['Distribution', 'Correlation', 'Comparison'],
            value='Distribution',
            description='Plot Type:'
        )
        
        # Output area
        explorer_output = widgets.Output()
        
        # Update button
        def update_exploration(b):
            with explorer_output:
                clear_output(wait=True)
                
                selected_metrics = list(metric_selector.value)
                
                if not selected_metrics:
                    print("Please select at least one metric")
                    return
                
                if plot_type.value == 'Distribution':
                    fig = self.visualizers['metrics'].plot_multi_metric_distributions(
                        self.qc.data,
                        selected_metrics,
                        self.qc.thresholds
                    )
                    plt.show()
                
                elif plot_type.value == 'Correlation':
                    if len(selected_metrics) >= 2:
                        from ..visualization.plots import plot_correlation_matrix
                        fig = plot_correlation_matrix(self.qc.data, selected_metrics)
                        plt.show()
                    else:
                        print("Need at least 2 metrics for correlation matrix")
                
                elif plot_type.value == 'Comparison' and self.qc.filtered_data is not None:
                    fig = self.visualizers['metrics'].plot_metric_comparison(
                        self.qc.data,
                        self.qc.filtered_data,
                        selected_metrics
                    )
                    plt.show()
        
        update_button = widgets.Button(
            description='Update Plot',
            button_style='primary',
            icon='refresh'
        )
        update_button.on_click(update_exploration)
        
        # Layout
        controls = widgets.HBox([
            widgets.VBox([
                widgets.HTML("<b>Select Metrics:</b>"),
                metric_selector
            ]),
            widgets.VBox([
                widgets.HTML("<b>Plot Type:</b>"),
                plot_type,
                update_button
            ])
        ])
        
        return widgets.VBox([
            widgets.HTML("<h4>Data Exploration</h4>"),
            controls,
            explorer_output
        ])
    
    def _create_gallery_tab(self) -> widgets.VBox:
        """Create sample gallery tab."""
        # Gallery controls
        n_samples = widgets.IntSlider(
            value=10,
            min=1,
            max=50,
            description='Samples:',
            style={'description_width': '80px'}
        )
        
        sample_type = widgets.Dropdown(
            options=['random', 'best', 'worst', 'diverse'],
            value='diverse',
            description='Type:',
            style={'description_width': '80px'}
        )
        
        n_cols = widgets.IntSlider(
            value=5,
            min=1,
            max=10,
            description='Columns:',
            style={'description_width': '80px'}
        )
        
        # Class filter
        if 'label' in self.qc.data.columns:
            unique_classes = sorted(self.qc.data['label'].unique())
            class_filter = widgets.SelectMultiple(
                options=['All'] + unique_classes,
                value=['All'],
                description='Classes:',
                rows=min(10, len(unique_classes) + 1)
            )
        else:
            class_filter = None
        
        # Output
        gallery_output = widgets.Output()
        
        # Update function
        def update_gallery(b):
            with gallery_output:
                clear_output(wait=True)
                
                # Filter by class if needed
                data_to_show = self.qc.filtered_data if self.qc.filtered_data is not None else self.qc.data
                
                if class_filter and 'All' not in class_filter.value:
                    data_to_show = data_to_show[data_to_show['label'].isin(class_filter.value)]
                
                if len(data_to_show) == 0:
                    print("No data matches the filters")
                    return
                
                # Create gallery
                actual_samples = min(n_samples.value, len(data_to_show))
                n_rows = (actual_samples + n_cols.value - 1) // n_cols.value
                
                # Use grid functionality
                fig = self.visualizers['samples'].create_quality_grid(
                    data_to_show,
                    grid_size=(n_rows, n_cols.value)
                )
                plt.show()
        
        update_gallery_button = widgets.Button(
            description='Update Gallery',
            button_style='primary',
            icon='refresh'
        )
        update_gallery_button.on_click(update_gallery)
        
        # Layout
        if class_filter:
            controls = widgets.HBox([
                widgets.VBox([n_samples, sample_type, n_cols, update_gallery_button]),
                widgets.VBox([widgets.HTML("<b>Filter by Class:</b>"), class_filter])
            ])
        else:
            controls = widgets.VBox([n_samples, sample_type, n_cols, update_gallery_button])
        
        return widgets.VBox([
            widgets.HTML("<h4>Sample Gallery</h4>"),
            controls,
            gallery_output
        ])
    
    def _create_export_tab(self, auto_balance: bool) -> widgets.VBox:
        """Create balance and export tab."""
        # Balance controls
        balance_section = widgets.VBox([
            widgets.HTML("<h4>Dataset Balancing</h4>")
        ])
        
        if auto_balance:
            strategy = widgets.Dropdown(
                options=['none', 'median', 'mean', 'min', 'max'],
                value='median',
                description='Strategy:',
                style={'description_width': '100px'}
            )
            
            min_samples = widgets.IntSlider(
                value=15,
                min=1,
                max=100,
                description='Min samples:',
                style={'description_width': '100px'}
            )
            
            
            enable_oversampling = widgets.Checkbox(
                value=False,
                description='Enable oversampling'
            )
            
            balance_output = widgets.Output()
            
            def apply_balance(b):
                with balance_output:
                    clear_output(wait=True)
                    
                    if strategy.value != 'none':
                        balanced = self.qc.balance_dataset(
                            strategy=strategy.value,
                            min_samples=min_samples.value,
                            enable_oversampling=enable_oversampling.value
                        )
                        
                        print(f"Balanced dataset: {len(balanced)} samples")
                        
                        # Show distribution
                        if 'label' in balanced.columns:
                            from ..visualization.plots import plot_class_distribution
                            fig = plot_class_distribution(balanced, top_n=20)
                            plt.show()
            
            balance_button = widgets.Button(
                description='Apply Balance',
                button_style='warning',
                icon='balance-scale'
            )
            balance_button.on_click(apply_balance)
            
            balance_section.children = balance_section.children + (
                strategy,
                min_samples,
                enable_oversampling,
                balance_button,
                balance_output
            )
        
        # Export controls
        export_section = widgets.VBox([
            widgets.HTML("<h4>Export Data</h4>")
        ])
        
        export_format = widgets.RadioButtons(
            options=['JSON', 'CSV'],
            value='JSON',
            description='Format:'
        )
        
        export_path = widgets.Text(
            value=str(self.cache_dir / 'filtered_data.json'),
            description='Path:',
            layout=widgets.Layout(width='500px'),
            style={'description_width': '50px'}
        )
        
        export_output = widgets.Output()
        
        def export_data(b):
            with export_output:
                clear_output()
                
                if self.qc.filtered_data is None:
                    print("No filtered data to export")
                    return
                
                try:
                    format_type = export_format.value.lower()
                    self.qc.save_filtered_data(export_path.value, format=format_type)
                    print(f"✅ Exported {len(self.qc.filtered_data)} samples to {export_path.value}")
                except Exception as e:
                    print(f"❌ Export failed: {e}")
        
        export_button = widgets.Button(
            description='Export Data',
            button_style='success',
            icon='download'
        )
        export_button.on_click(export_data)
        
        export_section.children = export_section.children + (
            export_format,
            export_path,
            export_button,
            export_output
        )
        
        return widgets.VBox([balance_section, export_section])
    
    def _create_statistics_tab(self) -> widgets.VBox:
        """Create statistics tab."""
        stats_output = widgets.Output()
        
        def update_stats(b):
            with stats_output:
                clear_output()
                
                stats = self.qc.get_statistics()
                
                # Overall statistics
                print("📊 **Overall Statistics**")
                print(f"Total samples: {stats['total_samples']:,}")
                print(f"Filtered samples: {stats['filtered_samples']:,}")
                print(f"Retention rate: {stats['retention_rate']:.1%}")
                print()
                
                # Class distribution
                if 'class_distribution' in stats and stats['class_distribution']:
                    print("📈 **Class Distribution**")
                    
                    if 'original' in stats['class_distribution']:
                        print("\nOriginal:")
                        for cls, count in sorted(stats['class_distribution']['original'].items()):
                            print(f"  {cls}: {count:,}")
                    
                    if 'filtered' in stats['class_distribution']:
                        print("\nFiltered:")
                        for cls, count in sorted(stats['class_distribution']['filtered'].items()):
                            print(f"  {cls}: {count:,}")
                
                # Metric statistics
                if 'metric_statistics' in stats and stats['metric_statistics']:
                    print("\n📏 **Metric Statistics**")
                    
                    for metric, values in stats['metric_statistics'].items():
                        print(f"\n{metric}:")
                        print(f"  Mean: {values['mean']:.4f}")
                        print(f"  Std: {values['std']:.4f}")
                        print(f"  Min: {values['min']:.4f}")
                        print(f"  Max: {values['max']:.4f}")
                        print(f"  Median: {values['median']:.4f}")
        
        update_button = widgets.Button(
            description='Update Statistics',
            button_style='primary',
            icon='refresh'
        )
        update_button.on_click(update_stats)
        
        # Initial update
        update_stats(None)
        
        return widgets.VBox([
            widgets.HTML("<h4>Dataset Statistics</h4>"),
            update_button,
            stats_output
        ])
