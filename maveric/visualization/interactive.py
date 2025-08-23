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
    
    def save_filtered_data(self, output_file=None, rotation_size=None):
        """Save filtered data in the same format as 02_data_curation.py script"""
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
                
                # Return the first file path for backward compatibility
                return output_paths[0]
        
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
                    step=0.01 if metric == 'resolution_score' else 0.001,
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
        
        # Create tabs
        tab = widgets.Tab()
        tab.children = [
            widgets.VBox(list(threshold_widgets.values())),
            widgets.VBox(list(weight_widgets.values())),
            balance_tab_content
        ]
        tab.set_title(0, 'Quality Thresholds')
        tab.set_title(1, 'Class Weights')
        tab.set_title(2, 'Balance Settings')
        
        # Create buttons
        apply_button = widgets.Button(description='Apply Settings', button_style='primary', icon='check')
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
                self.visualize_sample_images()
        
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
        visualize_button.on_click(on_visualize_clicked)
        compare_button.on_click(on_compare_clicked)
        save_data_button.on_click(on_save_data_clicked)
        save_config_button.on_click(on_save_config_clicked)
        balance_button.on_click(on_balance_clicked)
        
        # Layout
        button_box = widgets.HBox([apply_button, visualize_button, compare_button, save_data_button, save_config_button])
        
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