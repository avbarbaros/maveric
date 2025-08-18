#!/usr/bin/env python3
"""
MAVERIC Data Curation GUI - Interactive Jupyter Notebook Interface

Interactive Jupyter notebook interface for analyzing raw retrieval data and determining 
optimal quality control thresholds using ipywidgets. Designed for Google Colab environment.

Features:
- Interactive sliders for threshold and weight adjustment
- Real-time visualization updates
- Sample image gallery with quality metrics
- Live filtering results
- Configuration saving to YAML file

Usage in Jupyter/Colab:
    python 02_data_curation_gui.py -d <dataset_name> -c <config_file>
    
Examples:
    python 02_data_curation_gui.py -d cifar10 -c maveric_config.yaml
    python 02_data_curation_gui.py -d cifar100 -c experiments/my_config.yaml

The script creates an interactive interface with:
1. Threshold sliders for quality metrics
2. Weight sliders for metric importance
3. Real-time filtering and visualization
4. Sample image display
5. Configuration save functionality
"""

import os
import sys
import yaml
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import requests
from io import BytesIO
from PIL import Image
warnings.filterwarnings('ignore')

# Jupyter widgets
import ipywidgets as widgets
from IPython.display import display, clear_output

# Add MAVERIC to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import MAVERIC components
from maveric import MAVERIC
from maveric.config import MAVERICConfig
from maveric.core.interfaces import RetrievalResult

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="MAVERIC Interactive Data Curation GUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 02_data_curation_gui.py -d cifar10 -c maveric_config.yaml
  python 02_data_curation_gui.py -d cifar100 -c experiments/my_config.yaml
        """
    )
    
    parser.add_argument(
        '-d', '--dataset',
        type=str,
        required=True,
        help='Target dataset name (e.g., cifar10, cifar100, food101)'
    )
    
    parser.add_argument(
        '-c', '--config',
        type=str,
        required=True,
        help='Path to MAVERIC configuration YAML file'
    )
    
    return parser.parse_args()

class MAVERICInteractiveGUI:
    """Interactive GUI for MAVERIC data curation using ipywidgets"""
    
    def __init__(self, target_dataset: str, config_file: str):
        """
        Initialize the interactive GUI
        
        Args:
            target_dataset: Target dataset name (e.g., 'cifar10')
            config_file: Path to MAVERIC configuration file
        """
        self.target_dataset = target_dataset.lower()
        self.config_file = config_file
        self.config = None
        self.maveric = None
        self.retrieval_result = None
        self.raw_data_dir = None
        self.data_df = None
        self.filtered_data = None
        
        # Default thresholds and weights
        self.thresholds = {
            'weighted_class_score': 0.350,
            'consistency': 0.750,
            'resolution_score': 1.000,
            'sharpness_score': 0.900,
            'color_diversity': 0.800
        }
        
        self.weights = {
            'weighted_class_score': 1.0,
            'consistency': 1.0,
            'resolution_score': 1.0,
            'sharpness_score': 1.0,
            'color_diversity': 1.0
        }
        
        # Widgets storage
        self.threshold_widgets = {}
        self.weight_widgets = {}
        self.output_widget = widgets.Output()
        self.filtered_count_widget = widgets.HTML()
        
        print(f"🎯 Initializing MAVERIC GUI for {self.target_dataset.upper()}")
        print(f"📋 Configuration: {self.config_file}")
        
        # Initialize system
        self.load_configuration()
        self.initialize_maveric()
        self.load_data()
        
        if self.data_df is not None:
            self.calculate_best_class()
            self.create_gui()
        else:
            print("❌ Failed to load data. Please check the raw data directory.")
    
    def load_configuration(self):
        """Load MAVERIC configuration from YAML file"""
        try:
            with open(self.config_file, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Determine raw data directory
            results_dir = self.config.get('results_dir', './results')
            self.raw_data_dir = f"{results_dir}/{self.target_dataset}/raw"
            
            print(f"✅ Configuration loaded")
            print(f"📁 Raw data directory: {self.raw_data_dir}")
            
            # Update thresholds and weights from config
            config_thresholds = self.config.get('quality_thresholds', {})
            config_weights = self.config.get('metric_weights', {})
            
            self.thresholds.update(config_thresholds)
            self.weights.update(config_weights)
            
        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
            self.config = {}
            self.raw_data_dir = f"./results/{self.target_dataset}/raw"
    
    def initialize_maveric(self):
        """Initialize MAVERIC system with configuration"""
        try:
            print("🔧 Initializing MAVERIC...")
            
            # Create MAVERICConfig from loaded config
            maveric_config = MAVERICConfig(
                cache_base_dir=self.config.get('cache_base_dir', './cache'),
                clip_model=self.config.get('clip_model', 'ViT-B/32'),
                batch_size=self.config.get('batch_size', 32),
                device=self.config.get('device', 'auto'),
                enable_image_cache=self.config.get('caching', {}).get('enable_image_cache', True),
                default_thresholds=self.config.get('quality_thresholds', {}),
                balance_min_samples=self.config.get('elevater', {}).get('quality_control', {}).get('min_samples_per_class', 15),
                retrieval_rotation_size=self.config.get('retrieval_rotation_size', 1000),
                enable_real_time_stats=False,  # Disable for GUI
                metric_weights=self.config.get('metric_weights', {}),
                num_workers=self.config.get('performance', {}).get('num_workers', 4),
                log_level='WARNING',  # Reduce logging noise
                viz_save_figures=False
            )
            
            # Initialize MAVERIC
            self.maveric = MAVERIC(maveric_config)
            print("✅ MAVERIC initialized")
            
        except Exception as e:
            print(f"❌ Error initializing MAVERIC: {e}")
            self.maveric = None
    
    def load_data(self):
        """Load retrieval data using MAVERIC's built-in function"""
        if not os.path.exists(self.raw_data_dir):
            print(f"❌ Raw data directory not found: {self.raw_data_dir}")
            return
        
        try:
            print("🔍 Loading retrieval data...")
            
            # Use MAVERIC's built-in function to load rotation files
            self.retrieval_result = RetrievalResult.from_rotation_files(
                dataset_name=self.target_dataset,
                input_dir=self.raw_data_dir,
                source_dataset="react-vl/react-retrieval-datasets"
            )
            
            # Convert to DataFrame
            self.data_df = self.retrieval_result.to_dataframe()
            
            print(f"✅ Loaded {len(self.data_df):,} samples")
            
        except Exception as e:
            print(f"❌ Error loading retrieval data: {e}")
            self.retrieval_result = None
    
    def calculate_best_class(self):
        """Calculate best class and weighted scores for each sample"""
        if self.data_df is None:
            return
        
        print("🧮 Calculating best class scores...")
        
        # Get available class columns
        class_columns = [col for col in self.data_df.columns if col.startswith('Class_') and '_hybrid_score' in col]
        
        if not class_columns:
            print("⚠️ No class hybrid scores found")
            return
        
        # Calculate best class for each sample
        best_classes = []
        weighted_scores = []
        consistency_scores = []
        
        for idx, row in self.data_df.iterrows():
            class_scores = {}
            
            # Find best class based on hybrid scores
            for col in class_columns:
                if not pd.isna(row[col]):
                    class_name = col.replace('Class_', '').replace('_hybrid_score', '')
                    class_scores[class_name] = row[col]
            
            if class_scores:
                best_class = max(class_scores.items(), key=lambda x: x[1])
                best_classes.append(best_class[0])
                weighted_scores.append(best_class[1])
                
                # Get consistency for best class
                consistency_col = f"Class_{best_class[0]}_consistency"
                consistency_scores.append(row.get(consistency_col, 0.8))
            else:
                best_classes.append('unknown')
                weighted_scores.append(0.0)
                consistency_scores.append(0.0)
        
        # Add columns to dataframe
        self.data_df['label'] = best_classes
        self.data_df['weighted_class_score'] = weighted_scores
        self.data_df['consistency'] = consistency_scores
        
        # Initialize filtered data
        self.filtered_data = self.data_df.copy()
        
        print(f"✅ Calculated best class for all samples")
        print(f"📊 Found {len(set(best_classes))} unique classes")
    
    def apply_thresholds(self):
        """Apply current thresholds to filter the data"""
        if self.data_df is None:
            return 0
        
        # Start with all data
        self.filtered_data = self.data_df.copy()
        
        # Apply each threshold
        for metric, threshold in self.thresholds.items():
            if metric in self.filtered_data.columns:
                self.filtered_data = self.filtered_data[
                    self.filtered_data[metric] >= threshold
                ]
        
        return len(self.filtered_data)
    
    def create_gui(self):
        """Create the interactive GUI using ipywidgets"""
        print("🎨 Creating interactive GUI...")
        
        # Create threshold sliders
        self.create_threshold_widgets()
        
        # Create weight sliders
        self.create_weight_widgets()
        
        # Create buttons
        apply_button = widgets.Button(
            description='Apply Settings',
            button_style='primary',
            icon='check'
        )
        
        visualize_button = widgets.Button(
            description='Show Sample Images',
            button_style='info',
            icon='image'
        )
        
        save_button = widgets.Button(
            description='Save Configuration',
            button_style='success',
            icon='save'
        )
        
        # Connect button callbacks
        apply_button.on_click(self.on_apply_clicked)
        visualize_button.on_click(self.on_visualize_clicked)
        save_button.on_click(self.on_save_clicked)
        
        # Create tabs
        tab = widgets.Tab()
        tab.children = [
            widgets.VBox(list(self.threshold_widgets.values())),
            widgets.VBox(list(self.weight_widgets.values()))
        ]
        tab.set_title(0, 'Quality Thresholds')
        tab.set_title(1, 'Metric Weights')
        
        # Create layout
        button_box = widgets.HBox([apply_button, visualize_button, save_button])
        
        # Initial filter count
        count = self.apply_thresholds()
        self.filtered_count_widget.value = f"<h4>Filtered Data: {count:,} samples ({count/len(self.data_df)*100:.1f}%)</h4>"
        
        # Main layout
        main_layout = widgets.VBox([
            widgets.HTML("<h2>🎛️ MAVERIC Interactive Data Curation</h2>"),
            widgets.HTML(f"<p><b>Dataset:</b> {self.target_dataset.upper()} | <b>Total Samples:</b> {len(self.data_df):,}</p>"),
            tab,
            self.filtered_count_widget,
            button_box,
            self.output_widget
        ])
        
        # Display the GUI
        display(main_layout)
        
        # Show initial visualizations
        with self.output_widget:
            self.visualize_metrics_distribution()
    
    def create_threshold_widgets(self):
        """Create threshold slider widgets"""
        for metric, default_value in self.thresholds.items():
            # Determine max value based on metric
            if metric in self.data_df.columns:
                max_val = float(self.data_df[metric].max())
                min_val = float(self.data_df[metric].min())
                step = 0.001
                
                # Adjust step for resolution score
                if metric == 'resolution_score':
                    step = 0.01
            else:
                max_val = 1.0
                min_val = 0.0
                step = 0.001
            
            # Create slider
            self.threshold_widgets[metric] = widgets.FloatSlider(
                value=default_value,
                min=min_val,
                max=max_val,
                step=step,
                description=f'{metric.replace("_", " ").title()}:',
                disabled=False,
                continuous_update=False,
                orientation='horizontal',
                readout=True,
                readout_format='.3f',
                layout=widgets.Layout(width='500px'),
                style={'description_width': 'initial'}
            )
    
    def create_weight_widgets(self):
        """Create weight slider widgets"""
        for metric in self.thresholds.keys():
            default_weight = self.weights.get(metric, 1.0)
            
            self.weight_widgets[metric] = widgets.FloatSlider(
                value=default_weight,
                min=0.1,
                max=3.0,
                step=0.1,
                description=f'{metric.replace("_", " ").title()}:',
                disabled=False,
                continuous_update=False,
                orientation='horizontal',
                readout=True,
                readout_format='.1f',
                layout=widgets.Layout(width='500px'),
                style={'description_width': 'initial'}
            )
    
    def on_apply_clicked(self, button):
        """Handle apply button click"""
        # Update thresholds and weights from widgets
        for metric, widget in self.threshold_widgets.items():
            self.thresholds[metric] = widget.value
        
        for metric, widget in self.weight_widgets.items():
            self.weights[metric] = widget.value
        
        # Apply thresholds
        count = self.apply_thresholds()
        self.filtered_count_widget.value = f"<h4>Filtered Data: {count:,} samples ({count/len(self.data_df)*100:.1f}%)</h4>"
        
        # Update visualizations
        with self.output_widget:
            clear_output()
            print(f"✅ Applied settings: {count:,} samples retained")
            self.visualize_metrics_distribution()
    
    def on_visualize_clicked(self, button):
        """Handle visualize button click"""
        with self.output_widget:
            clear_output()
            self.visualize_sample_images()
    
    def on_save_clicked(self, button):
        """Handle save button click"""
        with self.output_widget:
            clear_output()
            success = self.save_configuration()
            if success:
                print("✅ Configuration saved successfully!")
                print(f"📁 Updated: {self.config_file}")
                print(f"💾 Backup: {self.config_file}.backup")
                print(f"\n🚀 Next Steps:")
                print(f"Run: python 02_data_curation.py -d {self.target_dataset} -c {self.config_file}")
            else:
                print("❌ Failed to save configuration")
    
    def visualize_metrics_distribution(self):
        """Visualize distribution of quality metrics"""
        if self.filtered_data is None:
            print("No data to visualize")
            return
        
        metrics = list(self.thresholds.keys())
        valid_metrics = [m for m in metrics if m in self.filtered_data.columns]
        
        if not valid_metrics:
            print("No valid metrics found")
            return
        
        # Create subplots
        n_metrics = len(valid_metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4*n_metrics))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(valid_metrics):
            ax = axes[i]
            
            # Get data
            original_data = self.data_df[metric].dropna()
            filtered_data = self.filtered_data[metric].dropna()
            
            if len(original_data) == 0:
                continue
            
            # Plot histograms
            ax.hist(original_data, bins=50, alpha=0.5, label='Original', color='blue', density=True)
            if len(filtered_data) > 0:
                ax.hist(filtered_data, bins=30, alpha=0.7, label='Filtered', color='green', density=True)
            
            # Add threshold line
            threshold = self.thresholds[metric]
            ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.3f}')
            
            # Add statistics
            mean_val = original_data.mean()
            ax.axvline(mean_val, color='orange', linestyle='-', alpha=0.7, label=f'Mean: {mean_val:.3f}')
            
            # Formatting
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_ylabel('Density')
            ax.set_title(f'Distribution of {metric.replace("_", " ").title()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Show class distribution
        if 'label' in self.filtered_data.columns:
            plt.figure(figsize=(12, 6))
            class_counts = self.filtered_data['label'].value_counts().head(20)
            
            plt.bar(range(len(class_counts)), class_counts.values)
            plt.xticks(range(len(class_counts)), class_counts.index, rotation=45, ha='right')
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.title('Class Distribution in Filtered Data')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    
    def visualize_sample_images(self, n_samples=6):
        """Visualize sample images from filtered data"""
        if self.filtered_data is None or len(self.filtered_data) == 0:
            print("No filtered data available for visualization")
            return
        
        # Select samples
        if len(self.filtered_data) <= n_samples:
            samples = self.filtered_data
        else:
            samples = self.filtered_data.sample(n=n_samples, random_state=42)
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (idx, row) in enumerate(samples.iterrows()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            try:
                # Load image
                if 'url' in row and pd.notna(row['url']):
                    response = requests.get(row['url'], timeout=10)
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                    ax.imshow(image)
                    
                    # Create metrics text
                    metrics_text = []
                    metrics_text.append(f"ID: {row.get('id', i)}")
                    metrics_text.append(f"Class: {row.get('label', 'unknown')}")
                    
                    for metric in ['weighted_class_score', 'consistency', 'resolution_score', 'sharpness_score']:
                        if metric in row:
                            metrics_text.append(f"{metric.replace('_', ' ').title()}: {row[metric]:.3f}")
                    
                    ax.set_title('\n'.join(metrics_text), fontsize=9)
                else:
                    ax.text(0.5, 0.5, "No image URL", ha='center', va='center')
                    
            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading image:\n{str(e)[:30]}...", 
                       ha='center', va='center', transform=ax.transAxes)
            
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(len(samples), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def save_configuration(self):
        """Save current thresholds and weights to configuration file"""
        try:
            # Read current configuration
            with open(self.config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Update thresholds and weights
            config_data['quality_thresholds'] = self.thresholds
            config_data['metric_weights'] = self.weights
            
            # Create backup
            backup_file = f"{self.config_file}.backup"
            with open(backup_file, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            
            # Save updated configuration
            with open(self.config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
            
            return True
            
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False

def main():
    """Main function"""
    print("🚀 MAVERIC Interactive Data Curation GUI")
    print("=" * 50)
    
    # Parse arguments
    args = parse_arguments()
    
    # Validate config file
    if not os.path.exists(args.config):
        print(f"❌ Configuration file not found: {args.config}")
        return
    
    # Create GUI
    try:
        gui = MAVERICInteractiveGUI(
            target_dataset=args.dataset,
            config_file=args.config
        )
        print("✅ Interactive GUI created successfully!")
        print("Use the sliders above to adjust thresholds and weights.")
        
    except Exception as e:
        print(f"❌ Error creating GUI: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set matplotlib backend for Jupyter
    try:
        import matplotlib
        matplotlib.use('inline')
    except:
        pass
    
    main()