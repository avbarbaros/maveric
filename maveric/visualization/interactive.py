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
    # Define dummy clear_output for when widgets are not available
    def clear_output(wait=False):
        pass


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
        self.cache_base_dir = None  # Will be loaded from config
        
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

        
        # Default quality thresholds (ordered by application sequence)
        self.thresholds = {
            'resolution_score': 1.000,
            'sharpness_score': 0.850,
            'color_score': 0.750,
            'weighted_class_score': 0.400,
            'consistency': 0.780
        }
        
        # Default balance settings
        self.balance_settings = {
            'balance_strategy': 'median',
            'balance_min_samples': 15,
            'balance_enable_oversampling': False
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

            # Get cache_base_dir from config
            self.cache_base_dir = config.get('cache_base_dir', '/content/drive/MyDrive/MAVERIC/maveric_cache')

            # Get results_dir from config
            results_dir = config.get('results_dir', '/content/drive/MyDrive/MAVERIC/maveric_experiments')
            
            # Load balance settings from config
            elevater_config = config.get('elevater', {})
            quality_control_config = elevater_config.get('quality_control', {})
            
            if quality_control_config:
                self.balance_settings.update({
                    'balance_strategy': quality_control_config.get('balance_strategy', 'median'),
                    'balance_min_samples': quality_control_config.get('min_samples_per_class', 15),
                    'balance_enable_oversampling': quality_control_config.get('enable_oversampling', False)
                })
                print(f"⚖️ Loaded balance settings: {self.balance_settings}")
            
            # Also check top-level config for balance settings
            if 'balance_strategy' in config:
                self.balance_settings['balance_strategy'] = config['balance_strategy']
            if 'balance_min_samples' in config:
                self.balance_settings['balance_min_samples'] = config['balance_min_samples']
            if 'balance_enable_oversampling' in config:
                self.balance_settings['balance_enable_oversampling'] = config['balance_enable_oversampling']

            # Load metric weights from config
            metric_weights = config.get('metric_weights', {})
            if metric_weights:
                self.class_weights.update(metric_weights)
                print(f"⚖️ Loaded metric weights: {self.class_weights}")

            # Load quality thresholds from config (excluding imagenet_probability)
            quality_thresholds = config.get('quality_thresholds', {})
            if quality_thresholds:
                # Only update thresholds that we want to support (excluding imagenet_probability)
                for metric, threshold in quality_thresholds.items():
                    if metric in self.thresholds:  # Only update thresholds we have defined
                        self.thresholds[metric] = threshold
                print(f"🎚️ Loaded quality thresholds: {self.thresholds}")
            
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
                similarity_score = 0.0
                valid_weights_sum = 0.0

                # Calculate weighted similarity score
                for metric, weight in self.class_weights.items():
                    col_name = f"Class_{class_name}_{metric}"

                    if col_name in row and not pd.isna(row[col_name]):
                        similarity_score += row[col_name] * weight
                        valid_weights_sum += weight

                # Normalize similarity score
                if valid_weights_sum > 0:
                    similarity_score /= valid_weights_sum
                    class_scores[class_name] = similarity_score
            
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
    
    def set_class_weight(self, metric, value, recalculate=True):
        """Set a weight for a class metric"""
        if metric in self.class_weights:
            self.class_weights[metric] = value
            print(f"⚖️ Set {metric} weight to {value:.2f}")
            # Only recalculate if explicitly requested (default: True for backward compatibility)
            if recalculate:
                self._calculate_best_class()
        else:
            print(f"❌ Unknown metric: {metric}")
            print(f"Available: {list(self.class_weights.keys())}")
    
    def set_class_weights(self, weights_dict):
        """Set multiple class weights at once and recalculate scores only once"""
        updated_weights = []
        for metric, value in weights_dict.items():
            if metric in self.class_weights:
                self.class_weights[metric] = value
                updated_weights.append(f"{metric}: {value:.2f}")
            else:
                print(f"❌ Unknown metric: {metric}")
        
        if updated_weights:
            print(f"⚖️ Updated weights: {', '.join(updated_weights)}")
            # Recalculate best class scores only once
            self._calculate_best_class()
        else:
            print("❌ No valid weights were updated")
    
    def apply_thresholds(self):
        """Apply current thresholds to filter the data"""
        if self.data is None:
            print("❌ No data loaded")
            return 0
        
        # Start with all data
        self.filtered_data = self.data.copy()
        initial_count = len(self.filtered_data)
        
        print(f"\n🎯 Applying Quality Thresholds:")
        print("=" * 60)
        print(f"{'Metric':<20} {'Threshold':<12} {'Before':<10} {'After':<10} {'Removed':<10}")
        print("-" * 60)
        
        # Apply each threshold and show impact
        for metric, threshold in self.thresholds.items():
            if metric in self.filtered_data.columns:
                before_count = len(self.filtered_data)
                self.filtered_data = self.filtered_data[
                    self.filtered_data[metric] >= threshold
                ]
                after_count = len(self.filtered_data)
                removed_count = before_count - after_count
                
                print(f"{metric:<20} {threshold:<12.3f} {before_count:<10,} {after_count:<10,} {removed_count:<10,}")
            else:
                print(f"{metric:<20} {'N/A':<12} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
        
        print("=" * 60)
        
        final_count = len(self.filtered_data)
        total_removed = initial_count - final_count
        retention = (final_count / initial_count) * 100 if initial_count > 0 else 0
        
        print(f"📊 SUMMARY:")
        print(f"  • Initial dataset: {initial_count:,} samples")
        print(f"  • Final dataset: {final_count:,} samples")
        print(f"  • Total removed: {total_removed:,} samples")
        print(f"  • Retention rate: {retention:.1f}%")
        
        # Show class distribution
        self._show_class_distribution()
        
        return final_count
    
    def apply_balance(self):
        """Apply balancing strategy to the filtered data"""
        if self.filtered_data is None or len(self.filtered_data) == 0:
            print("❌ No filtered data available for balancing")
            return 0
        
        if 'label' not in self.filtered_data.columns:
            print("❌ No 'label' column found, cannot balance")
            return len(self.filtered_data)
        
        strategy = self.balance_settings['balance_strategy']
        min_samples = self.balance_settings['balance_min_samples']
        enable_oversampling = self.balance_settings['balance_enable_oversampling']
        
        if strategy == 'none':
            print("ℹ️  No balancing applied (strategy='none')")
            return len(self.filtered_data)
        
        print(f"\n⚖️  Applying Balance Strategy: {strategy}")
        print("=" * 60)
        
        # Get class distribution before balancing
        class_counts = self.filtered_data['label'].value_counts()
        print("Before balancing:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count:,} samples")
        
        # Filter out classes below minimum threshold
        sufficient_classes = class_counts[class_counts >= min_samples]
        removed_classes = set(class_counts.index) - set(sufficient_classes.index)
        
        if removed_classes:
            print(f"\nRemoving {len(removed_classes)} classes with < {min_samples} samples:")
            for cls in removed_classes:
                print(f"  {cls}: {class_counts[cls]} samples")
            
            # Keep only sufficient classes
            self.filtered_data = self.filtered_data[
                self.filtered_data['label'].isin(sufficient_classes.index)
            ]
        
        if len(sufficient_classes) == 0:
            print("❌ No classes meet minimum threshold")
            self.filtered_data = pd.DataFrame()
            return 0
        
        # Calculate target samples per class
        remaining_sizes = sufficient_classes.values
        
        if strategy == 'median':
            target_samples = int(np.median(remaining_sizes))
        elif strategy == 'mean':
            target_samples = int(np.mean(remaining_sizes))
        elif strategy == 'min':
            target_samples = min(remaining_sizes)
        elif strategy == 'max':
            target_samples = max(remaining_sizes)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Apply minimum threshold constraint
        target_samples = max(target_samples, min_samples)
        print(f"\nTarget samples per class: {target_samples:,}")
        
        # Balance each class
        balanced_data = []
        
        for class_name in sufficient_classes.index:
            class_data = self.filtered_data[self.filtered_data['label'] == class_name].copy()
            
            # Sort by consistency score for best sample selection
            if 'consistency' in class_data.columns:
                class_data = class_data.sort_values('consistency', ascending=False)
            
            current_size = len(class_data)
            
            if current_size > target_samples:
                # Undersample: take top samples
                selected_data = class_data.head(target_samples)
            elif current_size == target_samples:
                # Perfect size
                selected_data = class_data
            else:
                # Smaller than target
                if enable_oversampling:
                    # Oversample by duplicating high-quality samples
                    selected_data = class_data.copy()
                    needed = target_samples - current_size
                    
                    # Duplicate samples cyclically
                    for i in range(needed):
                        duplicate_idx = i % current_size
                        selected_data = pd.concat([selected_data, class_data.iloc[duplicate_idx:duplicate_idx+1]], ignore_index=True)
                else:
                    # Keep original size
                    selected_data = class_data
            
            balanced_data.append(selected_data)
        
        # Combine balanced data
        self.filtered_data = pd.concat(balanced_data, ignore_index=True)
        
        # Shuffle the data
        self.filtered_data = self.filtered_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Show final distribution
        final_counts = self.filtered_data['label'].value_counts()
        print("\nAfter balancing:")
        for class_name, count in final_counts.items():
            print(f"  {class_name}: {count:,} samples")
        
        print("=" * 60)
        print(f"📊 Balanced dataset: {len(self.filtered_data):,} samples, {len(final_counts)} classes")
        
        return len(self.filtered_data)
    
    def _show_class_distribution(self):
        """Display class distribution of filtered data"""
        if self.filtered_data is None or len(self.filtered_data) == 0:
            print("📋 No filtered data available for class distribution")
            return
        
        if 'label' not in self.filtered_data.columns:
            print("📋 No class labels available for distribution")
            return
        
        # Calculate class distribution
        class_counts = self.filtered_data['label'].value_counts().sort_index()
        total_samples = len(self.filtered_data)
        
        print(f"\n📋 Class Distribution ({total_samples:,} samples):")
        print("=" * 50)
        
        for class_name, count in class_counts.items():
            percentage = (count / total_samples) * 100
            print(f"  • {class_name:<15}: {count:>6,} samples ({percentage:>5.1f}%)")
        
        # Show summary statistics
        min_count = class_counts.min()
        max_count = class_counts.max()
        mean_count = class_counts.mean()
        
        print("-" * 50)
        print(f"  📊 Balance Summary:")
        print(f"    Min: {min_count:,} | Max: {max_count:,} | Mean: {mean_count:.1f}")
        
        if max_count > 0:
            balance_ratio = min_count / max_count
            print(f"    Balance ratio: {balance_ratio:.3f} (1.0 = perfect balance)")
        
        print("=" * 50)
    
    def show_class_comparison(self):
        """Show comparison between original and filtered class distributions"""
        if self.data is None:
            print("❌ No data loaded")
            return
        
        if 'label' not in self.data.columns:
            print("📋 No class labels available")
            return
        
        # Original distribution
        original_counts = self.data['label'].value_counts().sort_index()
        
        # Filtered distribution
        if self.filtered_data is not None and len(self.filtered_data) > 0:
            filtered_counts = self.filtered_data['label'].value_counts().sort_index()
        else:
            filtered_counts = pd.Series(dtype=int)
        
        print(f"\n📊 Class Distribution Comparison:")
        print("=" * 70)
        print(f"{'Class':<15} {'Original':<12} {'Filtered':<12} {'Retention':<12}")
        print("-" * 70)
        
        for class_name in original_counts.index:
            original_count = original_counts.get(class_name, 0)
            filtered_count = filtered_counts.get(class_name, 0)
            retention = (filtered_count / original_count * 100) if original_count > 0 else 0
            
            print(f"{class_name:<15} {original_count:>8,} {filtered_count:>8,} {retention:>8.1f}%")
        
        print("=" * 70)
        total_original = len(self.data)
        total_filtered = len(self.filtered_data) if self.filtered_data is not None else 0
        overall_retention = (total_filtered / total_original * 100) if total_original > 0 else 0
        
        print(f"{'TOTAL':<15} {total_original:>8,} {total_filtered:>8,} {overall_retention:>8.1f}%")
        print("=" * 70)
    
    def visualize_distributions(self, metrics=None):
        """Visualize metric distributions with thresholds"""
        if self.data is None:
            print("❌ No data loaded")
            return
        
        if metrics is None:
            metrics = ['resolution_score', 'sharpness_score', 'color_score', 'weighted_class_score', 'consistency']
        
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
            median_val = data.median()
            threshold = self.thresholds.get(metric, 0)
            
            # Histogram
            ax.hist(data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            
            # Statistical lines with comprehensive legend
            ax.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                      label=f'Threshold: {threshold:.3f}')
            ax.axvline(mean_val, color='green', linestyle='-', linewidth=2, 
                      label=f'Mean: {mean_val:.3f}')
            ax.axvline(mean_val - std_val, color='blue', linestyle=':', linewidth=1, 
                      label=f'Mean-Std: {mean_val - std_val:.3f}')
            ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1, 
                      label=f'Mean+Std: {mean_val + std_val:.3f}')
            ax.axvline(median_val, color='purple', linestyle='-.', linewidth=2, 
                      label=f'Median: {median_val:.3f}')
            
            # Formatting
            ax.set_title(f'Distribution of {metric.replace("_", " ").title()}', fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Count')
            
            # Enhanced legend with better positioning
            ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=9, framealpha=0.9)
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
                    metrics_text += f"Sharpness: {row.get('sharpness_score', 0):.3f}\n"
                    metrics_text += f"Color: {row.get('color_score', 0):.3f}"
                    
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
    
    def save_filtered_data(self, output_file=None, rotation_size=None, copy_images=True):
        """
        Save filtered data in the same format as 02_data_curation.py script.

        Args:
            output_file: Output file path (optional)
            rotation_size: Number of samples per file (optional)
            copy_images: Whether to copy images to dataset-specific folder (default: True)
        """
        if self.filtered_data is None:
            print("❌ No filtered data available")
            return None
        
        # Determine output directory (same logic as 02_data_curation.py)
        if output_file is None:
            # Save to parent directory of raw data (remove /raw from path)
            if self.data_path.endswith('/raw'):
                base_dir = os.path.dirname(self.data_path)
            else:
                base_dir = self.data_path
            
            # Ensure output directory exists
            os.makedirs(base_dir, exist_ok=True)
        
        # Get rotation size from config file or use default
        if rotation_size is None:
            if self.config_file and os.path.exists(self.config_file):
                try:
                    import yaml
                    with open(self.config_file, 'r') as f:
                        config = yaml.safe_load(f)
                    rotation_size = config.get('retrieval_rotation_size', 1000)
                except:
                    rotation_size = 1000
            else:
                rotation_size = 1000
        
        try:
            # Format samples according to 02_data_curation.py specification
            formatted_samples = []
            
            for i, (_, row) in enumerate(self.filtered_data.iterrows()):
                formatted_sample = {
                    'id': int(row.get('id', row.get('sample_id', i + 1))),
                    'url': str(row.get('url', '')),
                    'label': str(row.get('label', row.get('class', ''))),
                    'text': str(row.get('text', '')),
                    'weighted_class_score': round(float(row.get('weighted_class_score', 
                                                row.get('hybrid_score', 0.0))), 5),
                    'consistency': round(float(row.get('consistency', 0.0)), 5)
                }
                formatted_samples.append(formatted_sample)
            
            # Export files based on rotation size (same logic as 02_data_curation.py)
            total_samples = len(formatted_samples)
            
            if output_file is not None:
                # User specified exact output file
                output_path = output_file
                with open(output_path, 'w') as f:
                    json.dump(formatted_samples, f, indent=2)
                print(f"💾 Saved {total_samples} filtered items to {output_path}")
                return output_path
            
            elif total_samples <= rotation_size:
                # Single file export
                filename = f"{self.dataset_name}_training_maveric_1.json"
                if self.data_path.endswith('/raw'):
                    base_dir = os.path.dirname(self.data_path)
                else:
                    base_dir = self.data_path
                output_path = os.path.join(base_dir, filename)
                
                with open(output_path, 'w') as f:
                    json.dump(formatted_samples, f, indent=2)
                    
                print(f"💾 Saved {total_samples} filtered items to {output_path}")
                return output_path
            else:
                # Multiple file export with rotation (same as 02_data_curation.py)
                if self.data_path.endswith('/raw'):
                    base_dir = os.path.dirname(self.data_path)
                else:
                    base_dir = self.data_path
                    
                output_paths = []
                sequence_number = 1
                num_files = (total_samples + rotation_size - 1) // rotation_size  # Ceiling division
                
                print(f"💾 Exporting training dataset JSON ({total_samples} samples → {num_files} files, {rotation_size} samples per file)...")
                
                for i in range(0, total_samples, rotation_size):
                    batch = formatted_samples[i:i + rotation_size]
                    filename = f"{self.dataset_name}_training_maveric_{sequence_number}.json"
                    output_path = os.path.join(base_dir, filename)
                    
                    with open(output_path, 'w') as f:
                        json.dump(batch, f, indent=2)
                    
                    output_paths.append(output_path)
                    sequence_number += 1
                
                print(f"📁 Exported {len(output_paths)} training dataset files to: {base_dir}")
                print(f"   First file: {os.path.basename(output_paths[0])}")

                # Copy images to dataset-specific folder
                if copy_images and self.cache_base_dir:
                    self._copy_training_images(base_dir)

                # Return the first file path for backward compatibility
                return output_paths[0]
        
        except Exception as e:
            print(f"❌ Error saving data: {e}")
            return None

    def _copy_training_images(self, output_dir):
        """
        Copy training images from global cache to dataset-specific images folder.
        If image is not in cache, download it.

        Args:
            output_dir: Directory where training JSON files are saved
        """
        import hashlib
        import shutil
        import requests
        from pathlib import Path
        from PIL import Image
        from io import BytesIO

        try:
            # Create images directory
            images_dir = Path(output_dir) / 'images'
            images_dir.mkdir(parents=True, exist_ok=True)

            # Source: global cache
            global_cache_dir = Path(self.cache_base_dir) / 'image_cache'
            global_cache_dir.mkdir(parents=True, exist_ok=True)

            total_images = len(self.filtered_data)
            print(f"📦 Processing {total_images} training images to {images_dir}...")

            # Check how many already exist
            existing_count = sum(1 for _, row in self.filtered_data.iterrows()
                               if row.get('url') and (images_dir / f"img_{hashlib.md5(row.get('url').encode()).hexdigest()}.jpg").exists())

            if existing_count > 0:
                print(f"ℹ️  Found {existing_count} images already in destination, will process {total_images - existing_count} remaining images")

            copied_count = 0
            downloaded_count = 0
            failed_count = 0

            for idx, (_, row) in enumerate(self.filtered_data.iterrows(), 1):
                url = row.get('url')
                if not url:
                    failed_count += 1
                    continue

                # Calculate image hash
                url_hash = hashlib.md5(url.encode()).hexdigest()
                src_filename = f"img_{url_hash}.jpg"

                # Check hierarchical structure first (new format: image_cache/ae/img_aeb88f14....jpg)
                subdir = url_hash[:2]
                src_path_hierarchical = global_cache_dir / subdir / src_filename

                # Check flat structure for backward compatibility
                src_path_flat = global_cache_dir / src_filename

                # Destination path
                dst_path = images_dir / src_filename

                # Skip if already exists
                if dst_path.exists():
                    continue

                # Try to copy from cache (hierarchical first, then flat)
                src_found = False
                for src_path in [src_path_hierarchical, src_path_flat]:
                    if src_path.exists():
                        try:
                            shutil.copy2(src_path, dst_path)
                            copied_count += 1
                            src_found = True
                            break
                        except Exception as e:
                            continue

                # If not found in cache, download it
                if not src_found:
                    try:
                        response = requests.get(url, timeout=(10, 30))  # Increased timeout: 10s connect, 30s read
                        response.raise_for_status()

                        # Load and validate image
                        image = Image.open(BytesIO(response.content)).convert('RGB')

                        # Save to destination
                        image.save(dst_path, 'JPEG', quality=95)
                        downloaded_count += 1

                        # Also save to hierarchical cache for future use
                        try:
                            cache_subdir = global_cache_dir / subdir
                            cache_subdir.mkdir(parents=True, exist_ok=True)
                            image.save(src_path_hierarchical, 'JPEG', quality=95)
                        except Exception:
                            pass  # Cache save failed, but we have the training image

                    except Exception as e:
                        failed_count += 1
                        # Print failed download immediately
                        print(f"❌ Failed to download: {src_filename}")
                        print(f"   URL: {url}")
                        print(f"   Error: {str(e)}")

            # Calculate successful total
            success_count = existing_count + copied_count + downloaded_count
            print(f"\n✅ Successfully processed {success_count}/{total_images} images: {copied_count} copied from cache, {downloaded_count} downloaded, {existing_count} already existed")
            if failed_count > 0:
                print(f"⚠️  {failed_count} images failed to process (errors shown above)")

        except Exception as e:
            print(f"❌ Error processing images: {e}")

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
            
            # Update quality thresholds (save to quality_thresholds section)
            if 'quality_thresholds' not in config:
                config['quality_thresholds'] = {}
            
            # Update the thresholds that match our GUI settings
            for metric, threshold in self.thresholds.items():
                if metric == 'weighted_class_score':
                    # Skip this as it's calculated, not a direct quality threshold
                    continue
                elif metric == 'consistency':
                    config['quality_thresholds']['consistency'] = threshold
                elif metric == 'resolution_score':
                    config['quality_thresholds']['resolution_score'] = threshold
                elif metric == 'sharpness_score':
                    config['quality_thresholds']['sharpness_score'] = threshold
                elif metric == 'color_score':
                    config['quality_thresholds']['color_score'] = threshold
            
            # Update class weights (save to metric_weights section)
            if 'metric_weights' not in config:
                config['metric_weights'] = {}
            
            # Update metric weights that match our GUI settings
            for metric, weight in self.class_weights.items():
                config['metric_weights'][metric] = weight
            
            # Save updated configuration
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            print(f"💾 Configuration saved to: {config_file}")
            print("📊 Saved quality thresholds to 'quality_thresholds' section:")
            for metric, threshold in self.thresholds.items():
                if metric != 'weighted_class_score':  # Skip calculated metric
                    config_key = metric
                    print(f"   • {config_key}: {threshold:.3f}")
            print("⚖️ Saved class weights to 'metric_weights' section:")
            for metric, weight in self.class_weights.items():
                print(f"   • {metric}: {weight:.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving configuration: {e}")
            return False
    
    def _create_efficientnet_tab(self):
        """Create EfficientNet prediction analysis tab"""
        if not WIDGETS_AVAILABLE:
            return widgets.VBox([widgets.HTML("❌ ipywidgets not available")])

        # Analysis output widget
        analysis_output = widgets.Output()

        # Refresh button to update analysis
        refresh_button = widgets.Button(
            description='Refresh Analysis',
            button_style='info',
            icon='refresh',
            layout=widgets.Layout(width='150px')
        )

        # Apply filter button
        apply_filter_button = widgets.Button(
            description='Apply Prediction Filter',
            button_style='warning',
            icon='filter',
            layout=widgets.Layout(width='200px')
        )

        # Status display
        status_display = widgets.HTML(value="<p>Click 'Refresh Analysis' to analyze predictions</p>")

        def on_refresh_clicked(b):
            with analysis_output:
                clear_output(wait=True)
                try:
                    self._analyze_class_predictions()
                    status_display.value = f"<p style='color:green'>✅ Analysis complete for {len(self.filtered_data):,} samples</p>"
                except Exception as e:
                    status_display.value = f"<p style='color:red'>❌ Error: {str(e)}</p>"

        def on_apply_filter_clicked(b):
            with analysis_output:
                clear_output(wait=True)
                try:
                    original_count = len(self.filtered_data)
                    self._apply_prediction_filter()
                    new_count = len(self.filtered_data)
                    filtered_out = original_count - new_count

                    print(f"🔍 Prediction Filter Applied:")
                    print(f"   Original samples: {original_count:,}")
                    print(f"   Filtered samples: {new_count:,}")
                    print(f"   Removed samples: {filtered_out:,} ({filtered_out/original_count*100:.1f}%)")
                    print(f"   Retention rate: {new_count/original_count*100:.1f}%")

                    status_display.value = f"<p style='color:green'>✅ Filter applied - {new_count:,} samples remaining</p>"
                except Exception as e:
                    status_display.value = f"<p style='color:red'>❌ Filter error: {str(e)}</p>"

        # Attach callbacks
        refresh_button.on_click(on_refresh_clicked)
        apply_filter_button.on_click(on_apply_filter_clicked)

        # Explanation
        explanation = widgets.HTML(
            "<div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>"
            "<b>EfficientNet Prediction Analysis</b><br>"
            "• <b>Weighted Score Class</b>: Class selected by similarity-based metrics<br>"
            "• <b>EfficientNet Class</b>: Class with highest EfficientNet probability<br>"
            "• <b>Match Rate</b>: Percentage where both predictions agree<br>"
            "• <b>Apply Filter</b>: Remove samples where predictions don't match"
            "</div>"
        )

        # Create tab content
        tab_content = widgets.VBox([
            explanation,
            widgets.HBox([refresh_button, apply_filter_button]),
            status_display,
            analysis_output
        ], layout=widgets.Layout(
            background_color='#f8f8f8',
            padding='10px',
            border='1px solid #ddd'
        ))

        return tab_content

    def _analyze_class_predictions(self):
        """Analyze class predictions from weighted scores vs EfficientNet"""
        if self.filtered_data is None or len(self.filtered_data) == 0:
            print("❌ No filtered data available. Apply quality thresholds first.")
            return

        print("🔍 Analyzing Class Predictions...")
        print("=" * 60)

        # Get weighted score based class (from 'label' column)
        weighted_classes = self.filtered_data['label'].values

        # Find EfficientNet-based predictions (highest probability class)
        efficientnet_classes = []

        # Find all EfficientNet score columns
        efficientnet_columns = [col for col in self.filtered_data.columns
                               if 'efficientNet_score' in col and col.startswith('Class_')]

        if not efficientnet_columns:
            print("❌ No EfficientNet score columns found. Make sure target_class_quality metric was computed.")
            return

        # For each sample, find the class with highest EfficientNet score
        for _, row in self.filtered_data.iterrows():
            max_score = -1
            best_class = 'unknown'

            for col in efficientnet_columns:
                score = row.get(col, 0)
                if pd.notna(score) and score > max_score:
                    max_score = score
                    # Extract class name from column name: Class_airplane_efficientNet_score -> airplane
                    best_class = col.split('_')[1]

            efficientnet_classes.append(best_class)

        # Convert to pandas Series with the same index as filtered_data
        weighted_series = pd.Series(weighted_classes, index=self.filtered_data.index)
        efficientnet_series = pd.Series(efficientnet_classes, index=self.filtered_data.index)

        # Calculate match statistics
        matches = (weighted_series == efficientnet_series)
        match_count = matches.sum()
        total_count = len(matches)
        match_rate = (match_count / total_count) * 100

        print(f"📊 Prediction Analysis Results:")
        print(f"   Total samples: {total_count:,}")
        print(f"   Matching predictions: {match_count:,}")
        print(f"   Non-matching predictions: {total_count - match_count:,}")
        print(f"   Match rate: {match_rate:.1f}%")
        print()

        # Show class distribution comparison
        print("📋 Class Distribution Comparison:")
        print("-" * 40)

        weighted_dist = weighted_series.value_counts().sort_index()
        efficientnet_dist = efficientnet_series.value_counts().sort_index()

        all_classes = sorted(set(weighted_dist.index) | set(efficientnet_dist.index))

        print(f"{'Class':<15} {'Weighted':<10} {'EfficientNet':<12} {'Difference':<10}")
        print("-" * 50)

        for class_name in all_classes:
            weighted_count = weighted_dist.get(class_name, 0)
            efficientnet_count = efficientnet_dist.get(class_name, 0)
            difference = weighted_count - efficientnet_count

            print(f"{class_name:<15} {weighted_count:<10} {efficientnet_count:<12} {difference:+<10}")

        print()

        # Show mismatched samples by class
        print("🔍 Mismatch Analysis by Class:")
        print("-" * 40)

        if match_count < total_count:
            # Get indices where predictions don't match
            mismatch_indices = matches[~matches].index

            # Create mismatch analysis
            mismatch_weighted = weighted_series[mismatch_indices]
            mismatch_efficientnet = efficientnet_series[mismatch_indices]

            # Create summary of mismatches
            mismatch_pairs = list(zip(mismatch_weighted.values, mismatch_efficientnet.values))
            mismatch_counts = pd.Series(mismatch_pairs).value_counts()

            print("Top mismatches (Weighted → EfficientNet):")
            for (weighted_class, efficient_class), count in mismatch_counts.head(10).items():
                percentage = (count / total_count) * 100
                print(f"   {weighted_class} → {efficient_class}: {count:,} samples ({percentage:.1f}%)")
        else:
            print("   No mismatches found - perfect agreement!")

        # Store analysis results for filtering
        self._last_analysis = {
            'matches': matches,
            'weighted_classes': weighted_series,
            'efficientnet_classes': efficientnet_series,
            'match_rate': match_rate
        }

    def _apply_prediction_filter(self):
        """Filter out samples where weighted score and EfficientNet predictions don't match"""
        if not hasattr(self, '_last_analysis') or self._last_analysis is None:
            print("❌ No analysis available. Run 'Refresh Analysis' first.")
            return

        if self.filtered_data is None or len(self.filtered_data) == 0:
            print("❌ No filtered data available.")
            return

        matches = self._last_analysis['matches']

        # Get original class distribution before filtering
        original_distribution = self.filtered_data['label'].value_counts().sort_index()
        original_count = len(self.filtered_data)

        # Filter using the boolean mask - ensure indices align
        matching_indices = matches[matches].index
        self.filtered_data = self.filtered_data.loc[matching_indices].reset_index(drop=True)
        new_count = len(self.filtered_data)

        # Get new class distribution after filtering
        new_distribution = self.filtered_data['label'].value_counts().sort_index()

        print(f"✅ Prediction filter applied successfully!")
        print(f"   Kept samples with matching predictions: {new_count:,}")
        print(f"   Removed mismatched samples: {original_count - new_count:,}")
        print(f"   Retention rate: {new_count/original_count*100:.1f}%")
        print()

        # Display class distribution comparison
        print("📊 Class Distribution After Prediction Filtering:")
        print("-" * 50)

        all_classes = sorted(set(original_distribution.index) | set(new_distribution.index))

        print(f"{'Class':<15} {'Before':<10} {'After':<10} {'Change':<10} {'%Retained':<10}")
        print("-" * 60)

        for class_name in all_classes:
            before_count = original_distribution.get(class_name, 0)
            after_count = new_distribution.get(class_name, 0)
            change = after_count - before_count
            retention_pct = (after_count / before_count * 100) if before_count > 0 else 0

            print(f"{class_name:<15} {before_count:<10} {after_count:<10} {change:+<10} {retention_pct:<9.1f}%")

        print()
        print(f"📈 Overall retention rate: {new_count/original_count*100:.1f}%")

    def create_interactive_gui(self):
        """Create interactive GUI with sliders and controls"""
        if not WIDGETS_AVAILABLE:
            print("❌ ipywidgets not available. Install with: pip install ipywidgets")
            return
        
        # Create threshold sliders with statistical option comboboxes
        threshold_widgets = {}
        threshold_combos = {}
        threshold_containers = {}

        for metric, default_value in self.thresholds.items():
            if metric in self.data.columns:
                data_range = self.data[metric].dropna()
                max_value = float(data_range.max())
                min_value = float(data_range.min())
                mean_val = float(data_range.mean())
                median_val = float(data_range.median())
                std_val = float(data_range.std())

                # Create slider
                slider = widgets.FloatSlider(
                    value=max(min_value, min(default_value, max_value)),
                    min=min_value,
                    max=max_value,
                    step=0.01 if metric == 'resolution_score' else 0.001,
                    description=f'{metric.replace("_", " ").title()}:',
                    continuous_update=False,
                    readout_format='.3f',
                    layout=widgets.Layout(width='400px'),
                    style={'description_width': '180px'}
                )

                # Create combobox with statistical options
                stats_combo = widgets.Dropdown(
                    options=[
                        ('Custom', 'custom'),
                        ('Mean', 'mean'),
                        ('Median', 'median'),
                        ('Mean - Std', 'mean_minus_std'),
                        ('Mean + Std', 'mean_plus_std')
                    ],
                    value='custom',
                    description='',
                    layout=widgets.Layout(width='120px'),
                    style={'description_width': '0px'}
                )

                # Store statistical values for this metric
                stats_values = {
                    'mean': mean_val,
                    'median': median_val,
                    'mean_minus_std': max(min_value, mean_val - std_val),
                    'mean_plus_std': min(max_value, mean_val + std_val)
                }

                # Flag to track if slider change is programmatic
                slider._programmatic_change = False

                # Create callback for stats combo
                def make_stats_callback(slider_widget, combo_widget, stats_dict, metric_name):
                    def on_stats_change(change):
                        if change['new'] != 'custom':
                            new_value = stats_dict[change['new']]
                            # Set flag to indicate programmatic change
                            slider_widget._programmatic_change = True
                            slider_widget.value = new_value
                            slider_widget._programmatic_change = False
                            # Update threshold in the object
                            self.set_threshold(metric_name, new_value)
                    return on_stats_change

                # Create callback for slider to reset combo to custom when manually changed
                def make_slider_callback(combo_widget, stats_dict):
                    def on_slider_change(change):
                        # Only reset to custom if this is not a programmatic change
                        if not getattr(change['owner'], '_programmatic_change', False):
                            # Check if the current slider value matches any of the statistical values
                            current_value = change['new']
                            tolerance = 0.001  # Small tolerance for floating point comparison

                            # Check if current value matches any statistical option
                            for stat_name, stat_value in stats_dict.items():
                                if abs(current_value - stat_value) < tolerance:
                                    combo_widget.value = stat_name
                                    return

                            # If no match found, set to custom
                            combo_widget.value = 'custom'
                    return on_slider_change

                # Attach callbacks
                stats_combo.observe(make_stats_callback(slider, stats_combo, stats_values, metric), names='value')
                slider.observe(make_slider_callback(stats_combo, stats_values), names='value')

                # Create horizontal container for slider and combo
                container = widgets.HBox([
                    slider,
                    stats_combo
                ], layout=widgets.Layout(align_items='center', margin='2px 0'))

                threshold_widgets[metric] = slider
                threshold_combos[metric] = stats_combo
                threshold_containers[metric] = container
        
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
        
        # Create balance settings widgets
        balance_strategy_widget = widgets.Dropdown(
            options=['none', 'median', 'mean', 'min', 'max'],
            value=self.balance_settings['balance_strategy'],
            description='Strategy:',
            style={'description_width': '180px'}
        )
        
        balance_min_samples_widget = widgets.IntSlider(
            value=self.balance_settings['balance_min_samples'],
            min=1,
            max=100,
            step=1,
            description='Min Samples:',
            continuous_update=False,
            layout=widgets.Layout(width='500px'),
            style={'description_width': '180px'}
        )
        
        balance_oversampling_widget = widgets.Checkbox(
            value=self.balance_settings['balance_enable_oversampling'],
            description='Enable Oversampling',
            style={'description_width': '180px'}
        )
        
        balance_button = widgets.Button(
            description='Apply Balance',
            button_style='warning',
            icon='balance-scale',
            layout=widgets.Layout(width='200px')
        )
        
        # Create balance tab content
        balance_tab_content = widgets.VBox([
            balance_strategy_widget,
            balance_min_samples_widget,
            balance_oversampling_widget,
            balance_button
        ])

        # Create EfficientNet Prediction tab content
        efficientnet_tab_content = self._create_efficientnet_tab()
        
        # Create tabs
        tab = widgets.Tab()
        tab.children = [
            widgets.VBox(list(weight_widgets.values())),
            widgets.VBox(list(threshold_containers.values())),
            efficientnet_tab_content,
            balance_tab_content
        ]
        tab.set_title(0, 'Metric Weights')
        tab.set_title(1, 'Quality Thresholds')
        tab.set_title(2, 'EfficientNet Prediction')
        tab.set_title(3, 'Balance Settings')
        
        # Create buttons
        apply_button = widgets.Button(description='Apply Settings', button_style='primary', icon='check')
        reset_button = widgets.Button(description='Reset All', button_style='warning', icon='refresh')
        visualize_button = widgets.Button(description='Show Samples', button_style='info', icon='image')
        compare_button = widgets.Button(description='Class Comparison', button_style='', icon='bar-chart')
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
                
                # Update weights (batch update to recalculate scores only once)
                weight_updates = {metric: widget.value for metric, widget in weight_widgets.items()}
                self.set_class_weights(weight_updates)
                
                # Update balance settings
                self.balance_settings.update({
                    'balance_strategy': balance_strategy_widget.value,
                    'balance_min_samples': balance_min_samples_widget.value,
                    'balance_enable_oversampling': balance_oversampling_widget.value
                })
                
                # Apply filters
                count = self.apply_thresholds()
                filtered_count.value = f"<h4>Filtered data: {count:,} items</h4>"
                
                # Show visualizations
                self.visualize_distributions()
        
        def on_visualize_clicked(b):
            with output:
                clear_output()
                # Use current time as seed to ensure different images each time
                import time
                random_seed = int(time.time() * 1000) % 1000000  # Use milliseconds for more randomness
                self.visualize_sample_images(random_seed=random_seed)
        
        def on_compare_clicked(b):
            with output:
                clear_output()
                self.show_class_comparison()
        
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
        
        def on_reset_clicked(b):
            with output:
                clear_output()
                print("🔄 Resetting to original data and default settings...")

                # Reset filtered data to original data
                self.filtered_data = self.data.copy()

                # Reset thresholds to defaults
                default_thresholds = {
                    'resolution_score': 1.000,
                    'sharpness_score': 0.850,
                    'color_score': 0.750,
                    'weighted_class_score': 0.400,
                    'consistency': 0.780
                }

                # Reset weights to defaults
                default_weights = {
                    'img2img': 0.40,
                    'txt2txt': 0.20,
                    'img2txt': 0.20,
                    'txt2img': 0.20
                }

                # Reset balance settings to defaults
                default_balance = {
                    'balance_strategy': 'median',
                    'balance_min_samples': 15,
                    'balance_enable_oversampling': False
                }

                # Update internal settings
                self.thresholds.update(default_thresholds)
                self.class_weights.update(default_weights)
                self.balance_settings.update(default_balance)

                # Update widgets to match defaults
                for metric, widget in threshold_widgets.items():
                    if metric in default_thresholds:
                        widget.value = default_thresholds[metric]
                        # Reset combo boxes to custom
                        if metric in threshold_combos:
                            threshold_combos[metric].value = 'custom'

                for metric, widget in weight_widgets.items():
                    if metric in default_weights:
                        widget.value = default_weights[metric]

                # Update balance widgets
                balance_strategy_widget.value = default_balance['balance_strategy']
                balance_min_samples_widget.value = default_balance['balance_min_samples']
                balance_oversampling_widget.value = default_balance['balance_enable_oversampling']

                # Recalculate best class with default weights
                self._calculate_best_class()

                # Update filtered count display
                original_count = len(self.data)
                filtered_count.value = f"<h4>Filtered data: {original_count:,} items</h4>"

                print(f"✅ Reset complete! Restored to {original_count:,} original samples")
                print("📋 All thresholds, weights, and balance settings restored to defaults")
                print("🎯 Ready to start fresh data curation process")

                # Show original data distribution
                self.visualize_distributions()

        def on_balance_clicked(b):
            with output:
                clear_output()

                # Update balance settings
                self.balance_settings.update({
                    'balance_strategy': balance_strategy_widget.value,
                    'balance_min_samples': balance_min_samples_widget.value,
                    'balance_enable_oversampling': balance_oversampling_widget.value
                })

                # Apply balance
                count = self.apply_balance()
                filtered_count.value = f"<h4>Balanced data: {count:,} items</h4>"
        
        # Connect callbacks
        apply_button.on_click(on_apply_clicked)
        reset_button.on_click(on_reset_clicked)
        visualize_button.on_click(on_visualize_clicked)
        compare_button.on_click(on_compare_clicked)
        save_data_button.on_click(on_save_data_clicked)
        save_config_button.on_click(on_save_config_clicked)
        balance_button.on_click(on_balance_clicked)

        # Layout
        button_box = widgets.HBox([apply_button, reset_button, visualize_button, compare_button, save_data_button, save_config_button])
        
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