"""Interactive GUI components for MAVERIC data curation."""

import os
import json
import glob
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Jupyter widgets
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    print("⚠️ ipywidgets not available. Install with: pip install ipywidgets")


class MAVERICInteractiveQualityControl:
    """Interactive Quality Control system for MAVERIC data curation"""
    
    def __init__(self, dataset_name='cifar10', data_path=None, config_file=None):
        """
        Initialize the MAVERIC Interactive Quality Control system
        
        Args:
            dataset_name: Target dataset name ('cifar10', 'cifar100', etc.)
            data_path: Directory containing MAVERIC JSON files (auto-detected if None)
            config_file: MAVERIC configuration file (optional)
        """
        if not WIDGETS_AVAILABLE:
            raise ImportError("ipywidgets is required for interactive GUI. Install with: pip install ipywidgets")
        
        self.dataset_name = dataset_name.lower()
        self.data_path = data_path
        self.config_file = config_file
        self.data = None
        self.filtered_data = None
        
        # Class names for different datasets
        self.cifar10_class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                                   'dog', 'frog', 'horse', 'ship', 'truck']
        
        self.cifar100_class_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
                                    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
                                    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
                                    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
                                    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
                                    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
                                    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
                                    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
                                    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
                                    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
                                    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
                                    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
                                    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
                                    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
        
        # Set class names based on dataset
        if self.dataset_name == "cifar10":
            self.class_names = self.cifar10_class_names
        elif self.dataset_name == "cifar100":
            self.class_names = self.cifar100_class_names
        else:
            # For other datasets, will auto-detect from data
            self.class_names = []
        
        # Weights for calculating best class score
        self.class_weights = {
            'img2img': 0.40,
            'txt2txt': 0.20,
            'img2txt': 0.20,
            'txt2img': 0.20
        }
        
        # Default quality thresholds
        self.thresholds = {
            'weighted_class_score': 0.400,
            'consistency': 0.780,
            'resolution_score': 1.000,
            'sharpness_score': 0.850,
            'color_score': 0.750
        }
        
        # Auto-detect data path if not provided
        if self.data_path is None:
            self.data_path = self._detect_data_path()
        
        # Load config file to get proper results directory if available
        if self.config_file and os.path.exists(self.config_file):
            self._load_config_and_update_paths()
        
        # Set matplotlib backend
        try:
            import matplotlib
            matplotlib.use('inline')
        except:
            pass
        
        print(f"🎯 MAVERIC Interactive Quality Control")
        print(f"📊 Dataset: {self.dataset_name.upper()}")
        print(f"📁 Data path: {self.data_path}")
        print("-" * 50)
        
        # Load and initialize data
        self._load_data()
        
        if self.data is not None:
            self._calculate_best_class()
            print(f"✅ Loaded {len(self.data):,} samples")
            print(f"📋 Classes: {len(set(self.data['label']))}")
        else:
            print("❌ Failed to load data. Please check the data path.")
    
    def _detect_data_path(self):
        """Auto-detect data path based on common locations"""
        possible_paths = [
            f'/content/drive/MyDrive/MAVERIC/maveric_experiments/{self.dataset_name}/raw',
            f'./results/{self.dataset_name}/raw',
            f'/content/drive/MyDrive/MAVERIC_Cache',
            f'/content/{self.dataset_name}_cache',
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                # Check if it contains JSON files for this dataset
                pattern = os.path.join(path, f"{self.dataset_name}*maveric*.json")
                if glob.glob(pattern):
                    return path
        
        # Default fallback
        return '/content/drive/MyDrive/MAVERIC/maveric_experiments'
    
    def _load_config_and_update_paths(self):
        """Load config file and update data path based on results_dir"""
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Get results_dir from config
            results_dir = config.get('results_dir', '/content/drive/MyDrive/MAVERIC/maveric_experiments')
            
            # Construct the correct data path: results_dir/dataset_name/raw
            config_data_path = os.path.join(results_dir, self.dataset_name, 'raw')
            
            # Check if the config-based path exists and has data
            if os.path.exists(config_data_path):
                pattern = os.path.join(config_data_path, f"{self.dataset_name}*maveric*.json")
                if glob.glob(pattern):
                    self.data_path = config_data_path
                    print(f"📁 Using data path from config: {config_data_path}")
                    return
            
            # Fallback to original detected path
            print(f"⚠️ Config data path not found: {config_data_path}")
            print(f"📁 Using detected path: {self.data_path}")
            
        except Exception as e:
            print(f"⚠️ Could not load config for path detection: {e}")
            print(f"📁 Using detected path: {self.data_path}")
    
    def _load_data(self):
        """Load all MAVERIC JSON files from the data directory"""
        print(f"🔍 Loading data from {self.data_path}")
        
        # Find JSON files matching the pattern
        patterns = [
            f"{self.dataset_name}*raw*maveric*.json",
            f"{self.dataset_name}*maveric*.json",
            f"*{self.dataset_name}*.json"
        ]
        
        json_files = []
        for pattern in patterns:
            file_pattern = os.path.join(self.data_path, pattern)
            files = glob.glob(file_pattern)
            if files:
                json_files = files
                break
        
        if not json_files:
            print(f"❌ No files found in {self.data_path}")
            print(f"💡 Expected patterns: {patterns}")
            return
        
        print(f"📁 Found {len(json_files)} dataset files")
        
        # Load and combine all data
        all_data = []
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    file_data = json.load(f)
                    if isinstance(file_data, list):
                        all_data.extend(file_data)
                    else:
                        all_data.append(file_data)
                print(f"✅ Loaded {os.path.basename(file_path)}")
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
        
        if all_data:
            self.data = pd.DataFrame(all_data)
            self.filtered_data = self.data.copy()
        else:
            print("❌ No data loaded")
    
    def _calculate_best_class(self):
        """Calculate the best class for each item using weighted scores"""
        if self.data is None:
            return
        
        print("🧮 Calculating best class scores...")
        
        # Auto-detect class names if not predefined
        if not self.class_names:
            class_columns = [col for col in self.data.columns if col.startswith('Class_')]
            self.class_names = list(set([col.split('_')[1] for col in class_columns]))
            print(f"📋 Auto-detected classes: {len(self.class_names)}")
        
        best_classes = []
        weighted_scores = []
        consistency_scores = []
        
        for _, row in self.data.iterrows():
            class_scores = {}
            
            for class_name in self.class_names:
                weighted_score = 0.0
                valid_weights_sum = 0.0
                
                # Calculate weighted score across sub-metrics
                for metric, weight in self.class_weights.items():
                    col_name = f"Class_{class_name}_{metric}"
                    
                    if col_name in row and not pd.isna(row[col_name]):
                        weighted_score += row[col_name] * weight
                        valid_weights_sum += weight
                
                # Normalize score
                if valid_weights_sum > 0:
                    weighted_score /= valid_weights_sum
                
                class_scores[class_name] = weighted_score
            
            # Find best class
            if class_scores:
                best_class = max(class_scores.items(), key=lambda x: x[1])
                best_classes.append(best_class[0])
                weighted_scores.append(best_class[1])
                
                # Get consistency score
                consistency_col = f"Class_{best_class[0]}_consistency"
                consistency_scores.append(row.get(consistency_col, 0.8))
            else:
                best_classes.append('unknown')
                weighted_scores.append(0.0)
                consistency_scores.append(0.0)
        
        # Add calculated columns
        self.data['label'] = best_classes
        self.data['weighted_class_score'] = weighted_scores
        self.data['consistency'] = consistency_scores
        
        print(f"✅ Calculated best class scores")
    
    def set_threshold(self, metric, value):
        """Set a threshold for a quality metric"""
        if metric in self.thresholds:
            self.thresholds[metric] = value
            print(f"🎛️ Set {metric} threshold to {value:.3f}")
        else:
            print(f"❌ Unknown metric: {metric}")
            print(f"Available: {list(self.thresholds.keys())}")
    
    def set_class_weight(self, metric, value):
        """Set a weight for a class metric"""
        if metric in self.class_weights:
            self.class_weights[metric] = value
            print(f"⚖️ Set {metric} weight to {value:.2f}")
            # Recalculate best class scores
            self._calculate_best_class()
        else:
            print(f"❌ Unknown metric: {metric}")
            print(f"Available: {list(self.class_weights.keys())}")
    
    def apply_thresholds(self):
        """Apply current thresholds to filter the data"""
        if self.data is None:
            print("❌ No data loaded")
            return 0
        
        # Start with all data
        self.filtered_data = self.data.copy()
        
        # Apply each threshold
        for metric, threshold in self.thresholds.items():
            if metric in self.filtered_data.columns:
                self.filtered_data = self.filtered_data[
                    self.filtered_data[metric] >= threshold
                ]
        
        count = len(self.filtered_data)
        retention = count / len(self.data) * 100
        print(f"📊 Filtered: {count:,} samples ({retention:.1f}% retention)")
        return count
    
    def visualize_distributions(self, metrics=None):
        """Visualize metric distributions with thresholds"""
        if self.data is None:
            print("❌ No data loaded")
            return
        
        if metrics is None:
            metrics = ['weighted_class_score', 'consistency', 'resolution_score', 'sharpness_score', 'color_score']
        
        # Filter valid metrics
        metrics = [m for m in metrics if m in self.data.columns]
        
        if not metrics:
            print("❌ No valid metrics found")
            return
        
        # Create subplots
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4*n_metrics))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            data = self.data[metric].dropna()
            
            if len(data) == 0:
                continue
            
            # Statistics
            mean_val = data.mean()
            std_val = data.std()
            threshold = self.thresholds.get(metric, 0)
            
            # Histogram
            ax.hist(data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            
            # Statistical lines
            ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.3f}')
            ax.axvline(mean_val, color='green', linestyle='-', linewidth=2, label=f'Mean: {mean_val:.3f}')
            ax.axvline(mean_val - std_val, color='blue', linestyle=':', linewidth=1, label=f'Mean±Std')
            ax.axvline(mean_val + std_val, color='blue', linestyle=':', linewidth=1)
            
            # Formatting
            ax.set_title(f'Distribution of {metric.replace("_", " ").title()}', fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Count')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_sample_images(self, n_samples=6, random_seed=42):
        """Visualize sample images from filtered data"""
        if self.filtered_data is None or len(self.filtered_data) == 0:
            print("❌ No filtered data available")
            return
        
        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Select samples
        if len(self.filtered_data) <= n_samples:
            samples = self.filtered_data
        else:
            sample_indices = np.random.choice(len(self.filtered_data), n_samples, replace=False)
            samples = self.filtered_data.iloc[sample_indices]
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (_, row) in enumerate(samples.iterrows()):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            try:
                # Load image
                url = row.get('url', '')
                if url:
                    response = requests.get(url, timeout=5)
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                    ax.imshow(image)
                    
                    # Display metrics
                    label = row.get('label', 'unknown')
                    metrics_text = f"ID: {row.get('id', i)}\n"
                    metrics_text += f"Label: {label}\n"
                    metrics_text += f"Score: {row.get('weighted_class_score', 0):.3f}\n"
                    metrics_text += f"Consistency: {row.get('consistency', 0):.3f}\n"
                    metrics_text += f"Resolution: {row.get('resolution_score', 0):.2f}\n"
                    metrics_text += f"Sharpness: {row.get('sharpness_score', 0):.3f}"
                    
                    ax.set_title(metrics_text, fontsize=9)
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
    
    def save_filtered_data(self, output_file=None):
        """Save filtered data to JSON file"""
        if self.filtered_data is None:
            print("❌ No filtered data available")
            return None
        
        if output_file is None:
            # Save to parent directory of raw data (remove /raw from path)
            if self.data_path.endswith('/raw'):
                base_dir = os.path.dirname(self.data_path)
            else:
                base_dir = self.data_path
            output_file = os.path.join(base_dir, f"{self.dataset_name}_filtered_dataset.json")
        
        try:
            # Create simplified output format
            simplified_data = []
            
            for i, (_, row) in enumerate(self.filtered_data.iterrows()):
                item = {
                    'id': int(row.get('id', i)),
                    'url': row.get('url', ''),
                    'label': row.get('label', ''),
                    'text': row.get('text', ''),
                    'weighted_class_score': float(row.get('weighted_class_score', 0)),
                    'consistency': float(row.get('consistency', 0))
                }
                simplified_data.append(item)
            
            # Save to JSON
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(simplified_data, f, indent=2)
            
            print(f"💾 Saved {len(simplified_data)} filtered items to {output_file}")
            return output_file
        
        except Exception as e:
            print(f"❌ Error saving data: {e}")
            return None
    
    def save_configuration(self, config_file=None):
        """Save current thresholds and weights to MAVERIC configuration file"""
        if config_file is None:
            config_file = self.config_file
        
        if config_file is None:
            # Try to find config file in common locations
            possible_configs = [
                'maveric_config.yaml',
                '/content/drive/MyDrive/MAVERIC/maveric_experiments/maveric_config.yaml',
                '/content/drive/MyDrive/MAVERIC/repo/maveric/experiments/maveric_config.yaml',
                './experiments/maveric_config.yaml',
                '../maveric_config.yaml'
            ]
            
            for config_path in possible_configs:
                if os.path.exists(config_path):
                    config_file = config_path
                    break
        
        if config_file is None or not os.path.exists(config_file):
            print("❌ Configuration file not found. Please provide config file path.")
            return False
        
        try:
            # Load existing configuration
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Update quality thresholds
            if 'quality_metrics' not in config:
                config['quality_metrics'] = {}
            
            config['quality_metrics']['thresholds'] = self.thresholds.copy()
            
            # Update class weights  
            if 'class_scoring' not in config:
                config['class_scoring'] = {}
            
            config['class_scoring']['weights'] = self.class_weights.copy()
            
            # Save updated configuration
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            print(f"💾 Configuration saved to: {config_file}")
            print("📊 Saved thresholds:")
            for metric, threshold in self.thresholds.items():
                print(f"   • {metric}: {threshold:.3f}")
            print("⚖️ Saved class weights:")
            for metric, weight in self.class_weights.items():
                print(f"   • {metric}: {weight:.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving configuration: {e}")
            return False
    
    def create_interactive_gui(self):
        """Create interactive GUI with sliders and controls"""
        if not WIDGETS_AVAILABLE:
            print("❌ ipywidgets not available. Install with: pip install ipywidgets")
            return
        
        # Create threshold sliders
        threshold_widgets = {}
        for metric, default_value in self.thresholds.items():
            if metric in self.data.columns:
                data_range = self.data[metric].dropna()
                max_value = float(data_range.max())
                min_value = float(data_range.min())
                
                threshold_widgets[metric] = widgets.FloatSlider(
                    value=max(min_value, min(default_value, max_value)),
                    min=min_value,
                    max=max_value,
                    step=0.001 if metric != 'resolution_score' else 0.01,
                    description=f'{metric.replace("_", " ").title()}:',
                    continuous_update=False,
                    readout_format='.3f',
                    layout=widgets.Layout(width='500px'),
                    style={'description_width': '180px'}
                )
        
        # Create weight sliders
        weight_widgets = {}
        for metric, default_value in self.class_weights.items():
            weight_widgets[metric] = widgets.FloatSlider(
                value=default_value,
                min=0.0,
                max=1.0,
                step=0.01,
                description=f'{metric.replace("_", " ").title()}:',
                continuous_update=False,
                readout_format='.2f',
                layout=widgets.Layout(width='500px'),
                style={'description_width': '180px'}
            )
        
        # Create tabs
        tab = widgets.Tab()
        tab.children = [
            widgets.VBox(list(threshold_widgets.values())),
            widgets.VBox(list(weight_widgets.values()))
        ]
        tab.set_title(0, 'Quality Thresholds')
        tab.set_title(1, 'Class Weights')
        
        # Create buttons
        apply_button = widgets.Button(description='Apply Settings', button_style='primary', icon='check')
        visualize_button = widgets.Button(description='Show Samples', button_style='info', icon='image')
        save_data_button = widgets.Button(description='Save Data', button_style='success', icon='save')
        save_config_button = widgets.Button(description='Save Config', button_style='warning', icon='cog')
        
        # Output widget for results
        output = widgets.Output()
        filtered_count = widgets.HTML(value=f"<h4>Filtered data: {len(self.filtered_data):,} items</h4>")
        
        # Define callbacks
        def on_apply_clicked(b):
            with output:
                clear_output()
                
                # Update thresholds
                for metric, widget in threshold_widgets.items():
                    self.set_threshold(metric, widget.value)
                
                # Update weights
                for metric, widget in weight_widgets.items():
                    self.set_class_weight(metric, widget.value)
                
                # Apply filters
                count = self.apply_thresholds()
                filtered_count.value = f"<h4>Filtered data: {count:,} items</h4>"
                
                # Show visualizations
                self.visualize_distributions()
        
        def on_visualize_clicked(b):
            with output:
                clear_output()
                self.visualize_sample_images()
        
        def on_save_data_clicked(b):
            with output:
                clear_output()
                save_path = self.save_filtered_data()
                if save_path:
                    print(f"✅ Data saved to: {save_path}")
        
        def on_save_config_clicked(b):
            with output:
                clear_output()
                if self.config_file is None:
                    print("❌ No configuration file specified")
                    print("💡 Please provide config_file parameter when creating the GUI")
                    print("   Example: start_interactive_gui('cifar10', '/path/to/maveric_config.yaml')")
                    return
                
                success = self.save_configuration()
                if success:
                    print(f"✅ Configuration saved successfully!")
                else:
                    print(f"❌ Failed to save configuration")
        
        # Connect callbacks
        apply_button.on_click(on_apply_clicked)
        visualize_button.on_click(on_visualize_clicked)
        save_data_button.on_click(on_save_data_clicked)
        save_config_button.on_click(on_save_config_clicked)
        
        # Layout
        button_box = widgets.HBox([apply_button, visualize_button, save_data_button, save_config_button])
        
        # Display GUI
        display(widgets.VBox([
            widgets.HTML(f"<h2>🎛️ MAVERIC Quality Control - {self.dataset_name.upper()}</h2>"),
            widgets.HTML(f"<p><b>Total samples:</b> {len(self.data):,} | <b>Classes:</b> {len(set(self.data['label']))}</p>"),
            tab,
            filtered_count,
            button_box,
            output
        ]))
        
        # Show initial distribution
        with output:
            self.visualize_distributions()


# Convenience functions for easy use
def create_quality_control(dataset_name='cifar10', data_path=None, config_file=None):
    """
    Create MAVERIC Quality Control interface
    
    Args:
        dataset_name: Target dataset ('cifar10', 'cifar100', etc.)
        data_path: Path to data directory (auto-detected if None)
        config_file: MAVERIC configuration file path (required for saving config)
    
    Returns:
        MAVERICInteractiveQualityControl instance
    """
    if config_file is None:
        print("⚠️ Warning: No config file provided. Configuration saving will be disabled.")
        print("💡 Usage: create_quality_control('cifar10', config_file='/path/to/maveric_config.yaml')")
    
    return MAVERICInteractiveQualityControl(dataset_name, data_path, config_file)

def start_interactive_gui(dataset_name, config_file, data_path=None):
    """
    Start interactive GUI for quality control
    
    Args:
        dataset_name: Target dataset ('cifar10', 'cifar100', etc.)  
        config_file: MAVERIC configuration file path (required)
        data_path: Path to data directory (auto-detected if None)
    
    Returns:
        MAVERICInteractiveQualityControl instance with GUI displayed
    
    Example:
        gui = start_interactive_gui('cifar10', '/content/drive/MyDrive/MAVERIC/maveric_config.yaml')
    """
    if config_file is None:
        raise ValueError("config_file parameter is required. Please provide path to maveric_config.yaml")
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    qc = MAVERICInteractiveQualityControl(dataset_name, data_path, config_file)
    if qc.data is not None:
        qc.create_interactive_gui()
    else:
        print("❌ Failed to load data. Check your data path and dataset name.")
    return qc