#!/usr/bin/env python3
"""
MAVERIC Data Curation GUI - Interactive Threshold Selection

Interactive script for analyzing raw retrieval data, determining optimal quality control thresholds,
and saving configuration for data curation. Uses MAVERIC built-in functions throughout.

Features:
- Uses MAVERIC's RetrievalResult.from_rotation_files() to load data
- Uses MAVERIC's quality_control() function for filtering analysis
- Uses MAVERIC's built-in visualization functions from maveric.visualization
- Shows sample images with quality scores and distribution plots
- Tests multiple threshold strategies (conservative, balanced, aggressive)
- Interactive threshold and weight selection with live testing
- Saves chosen parameters to maveric_config.yaml automatically
- Creates configuration backup before saving

Workflow:
1. Analyzes raw retrieval data and shows visualizations
2. Tests different threshold strategies and shows results
3. Offers interactive threshold and weight selection
4. Tests user selections in real-time
5. Saves final configuration to YAML file
6. Provides next steps for running 02_data_curation.py

Usage:
    python 02_data_curation_gui.py -d <dataset_name> -c <config_file>
    
Examples:
    python 02_data_curation_gui.py -d cifar10 -c maveric_config.yaml
    python 02_data_curation_gui.py -d imagenet -c experiments/my_config.yaml

After saving configuration, run:
    python 02_data_curation.py -d <dataset_name> -c <config_file>
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
warnings.filterwarnings('ignore')

# Add MAVERIC to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import MAVERIC components
from maveric import MAVERIC
from maveric.config import MAVERICConfig
from maveric.core.interfaces import RetrievalResult

# Import MAVERIC visualization components
from maveric.visualization import (
    MetricsVisualizer,
    SampleVisualizer,
    plot_class_distribution,
    plot_correlation_matrix,
    plot_quality_comparison
)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="MAVERIC Data Curation Analysis with Built-in Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 02_data_curation_gui.py -d cifar10 -c maveric_config.yaml
  python 02_data_curation_gui.py -d imagenet -c experiments/my_config.yaml
  python 02_data_curation_gui.py -d food101 -c /path/to/config.yaml
        """
    )
    
    parser.add_argument(
        '-d', '--dataset',
        type=str,
        required=True,
        help='Target dataset name (e.g., cifar10, imagenet, food101)'
    )
    
    parser.add_argument(
        '-c', '--config',
        type=str,
        required=True,
        help='Path to MAVERIC configuration YAML file'
    )
    
    return parser.parse_args()

class MAVERICDataCurationAnalyzer:
    """
    Analyzer for MAVERIC data curation using built-in MAVERIC functions and visualizations
    """
    
    def __init__(self, target_dataset: str, config_file: str):
        """
        Initialize the analyzer with MAVERIC components
        
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
        self.strategy_results = {}
        
        # Initialize MAVERIC visualization components
        self.metrics_visualizer = MetricsVisualizer(style='default', figsize=(12, 6))
        self.sample_visualizer = SampleVisualizer(figsize_per_image=(4, 5))
        
        print(f"🎯 Target dataset: {self.target_dataset.upper()}")
        print(f"📋 Configuration file: {self.config_file}")
        
        # Load configuration and initialize MAVERIC
        self.load_configuration()
        self.initialize_maveric()
        self.load_retrieval_data()
        
        if self.retrieval_result is not None:
            self.analyze_data()
    
    def load_configuration(self):
        """Load MAVERIC configuration from YAML file"""
        try:
            with open(self.config_file, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Determine raw data directory
            results_dir = self.config.get('results_dir', './results')
            self.raw_data_dir = f"{results_dir}/{self.target_dataset}/raw"
            
            print(f"✅ Configuration loaded successfully")
            print(f"📁 Raw data directory: {self.raw_data_dir}")
            
        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
            print("Using default configuration...")
            self.config = {}
            self.raw_data_dir = f"./results/{self.target_dataset}/raw"
    
    def initialize_maveric(self):
        """Initialize MAVERIC system with configuration"""
        try:
            print("🔧 Initializing MAVERIC system...")
            
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
                enable_real_time_stats=self.config.get('enable_real_time_stats', True),
                metric_weights=self.config.get('metric_weights', {}),
                num_workers=self.config.get('performance', {}).get('num_workers', 4),
                log_level=self.config.get('logging', {}).get('level', 'INFO'),
                viz_save_figures=False  # No file output for GUI version
            )
            
            # Initialize MAVERIC
            self.maveric = MAVERIC(maveric_config)
            print("✅ MAVERIC initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing MAVERIC: {e}")
            self.maveric = None
    
    def load_retrieval_data(self):
        """Load retrieval data using MAVERIC's built-in function"""
        if not os.path.exists(self.raw_data_dir):
            print(f"❌ Raw data directory not found: {self.raw_data_dir}")
            print(f"\\n🔧 Expected directory structure:")
            print(f"  {self.raw_data_dir}/")
            print(f"    ├── {self.target_dataset}_raw_maveric_1.json")
            print(f"    ├── {self.target_dataset}_raw_maveric_2.json")
            print(f"    └── ...")
            print(f"\\n💡 Make sure to run 01_data_retrieval.py first!")
            return
        
        try:
            print("🔍 Loading retrieval data using MAVERIC...")
            
            # Use MAVERIC's built-in function to load rotation files
            self.retrieval_result = RetrievalResult.from_rotation_files(
                dataset_name=self.target_dataset,
                input_dir=self.raw_data_dir,
                source_dataset="react-vl/react-retrieval-datasets"
            )
            
            print(f"✅ Loaded {self.retrieval_result.total_samples:,} samples")
            print(f"📊 Source dataset: {self.retrieval_result.source_dataset}")
            print(f"🎯 Target dataset: {self.retrieval_result.target_dataset}")
            
            # Show class distribution
            if self.retrieval_result.class_distribution:
                print(f"📋 Classes detected: {len(self.retrieval_result.class_distribution)}")
                
        except Exception as e:
            print(f"❌ Error loading retrieval data: {e}")
            self.retrieval_result = None
    
    def analyze_data(self):
        """Perform comprehensive data analysis using MAVERIC functions"""
        if self.retrieval_result is None or self.maveric is None:
            return
        
        print("\\n" + "="*80)
        print("📈 DATA ANALYSIS REPORT")
        print("="*80)
        
        # Basic statistics from RetrievalResult
        print(f"\\n📊 Dataset Overview:")
        print(f"  • Total samples: {self.retrieval_result.total_samples:,}")
        print(f"  • Available metrics: {len(self.retrieval_result.available_metrics)}")
        
        # Class distribution
        if self.retrieval_result.class_distribution:
            class_dist = pd.Series(self.retrieval_result.class_distribution)
            print(f"  • Unique classes: {len(class_dist)}")
            print(f"  • Samples per class (avg): {class_dist.mean():.1f}")
            print(f"  • Class balance (std): {class_dist.std():.1f}")
        
        # Score statistics
        if self.retrieval_result.score_statistics:
            print(f"\\n📋 Quality Metrics Analysis:")
            
            for metric, stats in self.retrieval_result.score_statistics.items():
                print(f"\\n  {metric.replace('_', ' ').title()}:")
                print(f"    Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
                print(f"    Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
                print(f"    Median: {stats['median']:.3f}")
        
        # Test different threshold strategies
        self.test_threshold_strategies()
        
        # Create visualizations
        self.create_visualizations()
    
    def test_threshold_strategies(self):
        """Test different threshold strategies using MAVERIC's quality_control"""
        print(f"\\n🎯 THRESHOLD STRATEGY ANALYSIS:")
        print("-" * 50)
        
        # Get current thresholds from config
        current_thresholds = self.config.get('quality_thresholds', {})
        
        # Define threshold strategies based on score statistics
        threshold_sets = {
            'conservative': {},
            'balanced': {},
            'aggressive': {},
            'current': current_thresholds
        }
        
        # Build threshold sets from statistics
        for metric, stats in self.retrieval_result.score_statistics.items():
            if metric in ['weighted_class_score', 'consistency', 'resolution_score', 'sharpness_score', 'color_score']:
                threshold_sets['conservative'][metric] = max(0, stats['mean'] - 0.5 * stats['std'])
                threshold_sets['balanced'][metric] = stats['median']
                threshold_sets['aggressive'][metric] = stats['mean']
        
        # Test each strategy using MAVERIC's quality control
        for strategy_name, thresholds in threshold_sets.items():
            if not thresholds:
                continue
                
            try:
                print(f"\\n🧪 Testing {strategy_name} strategy...")
                
                # Use MAVERIC's quality control function
                quality_result = self.maveric.quality_control(
                    data=(self.target_dataset, self.raw_data_dir),
                    thresholds=thresholds,
                    weights=self.config.get('metric_weights', {}),
                    balance_strategy='none'
                )
                
                if quality_result:
                    retention_rate = quality_result.retention_rate * 100
                    self.strategy_results[strategy_name] = {
                        'result': quality_result,
                        'retention_rate': retention_rate,
                        'thresholds': thresholds
                    }
                    
                    print(f"  ✅ {strategy_name.title()}: {quality_result.filtered_count:,} samples ({retention_rate:.1f}% retention)")
                else:
                    print(f"  ❌ {strategy_name} strategy failed")
                    
            except Exception as e:
                print(f"  ❌ Error testing {strategy_name}: {e}")
        
        # Recommend best strategy
        self.recommend_strategy()
    
    def recommend_strategy(self):
        """Recommend the best threshold strategy"""
        if not self.strategy_results:
            return
        
        print(f"\\n🎛️  STRATEGY RECOMMENDATIONS:")
        print("-" * 40)
        
        # Sort by retention rate
        sorted_strategies = sorted(
            self.strategy_results.items(),
            key=lambda x: x[1]['retention_rate'],
            reverse=True
        )
        
        for strategy_name, result in sorted_strategies:
            retention = result['retention_rate']
            count = result['result'].filtered_count
            
            if retention > 80:
                quality_level = "High Retention"
            elif retention > 50:
                quality_level = "Balanced"
            elif retention > 20:
                quality_level = "High Quality"
            else:
                quality_level = "Very Selective"
            
            print(f"  • {strategy_name.title()}: {count:,} samples ({retention:.1f}%) - {quality_level}")
        
        # Recommend balanced approach
        if 'balanced' in self.strategy_results:
            balanced = self.strategy_results['balanced']
            print(f"\\n🌟 RECOMMENDED: Balanced strategy")
            print(f"   → {balanced['result'].filtered_count:,} samples ({balanced['retention_rate']:.1f}% retention)")
            print(f"   → Good balance between quality and quantity")
    
    def create_visualizations(self):
        """Create comprehensive visualizations using MAVERIC built-in functions"""
        print(f"\\n🎨 Creating visualizations using MAVERIC visualization functions...")
        
        # Get data as DataFrame for visualization
        df = self.retrieval_result.to_dataframe()
        
        # 1. Metric distributions using MAVERIC's MetricsVisualizer
        self.plot_metric_distributions_maveric(df)
        
        # 2. Class distribution using MAVERIC's plot_class_distribution
        self.plot_class_distribution_maveric(df)
        
        # 3. Correlation matrix using MAVERIC's plot_correlation_matrix
        self.plot_correlation_matrix_maveric(df)
        
        # 4. Sample images using MAVERIC's SampleVisualizer
        self.show_sample_images_maveric(df)
        
        # 5. Strategy comparison using MAVERIC's plot_quality_comparison
        self.plot_strategy_comparison_maveric()
    
    def plot_metric_distributions_maveric(self, df):
        """Plot metric distributions using MAVERIC's MetricsVisualizer"""
        print("📊 Plotting metric distributions...")
        
        # Get available metrics
        metrics = [col for col in df.columns if any(metric in col for metric in 
                  ['weighted_class_score', 'consistency', 'resolution_score', 'sharpness_score', 'color_score'])]
        
        if not metrics:
            print("⚠️  No quality metrics found for distribution plots")
            return
        
        # Get current thresholds
        current_thresholds = self.config.get('quality_thresholds', {})
        
        try:
            # Use MAVERIC's MetricsVisualizer for multiple metrics
            self.metrics_visualizer.plot_multi_metric_distributions(
                data=df,
                metrics=metrics,
                thresholds=current_thresholds,
                ncols=2,
                figsize_per_plot=(6, 4)
            )
            plt.show()
        except Exception as e:
            print(f"❌ Error plotting metric distributions: {e}")
    
    def plot_class_distribution_maveric(self, df):
        """Plot class distribution using MAVERIC's plot_class_distribution"""
        print("📊 Plotting class distribution...")
        
        if 'label' not in df.columns:
            print("⚠️  No 'label' column found for class distribution")
            return
        
        try:
            # Use MAVERIC's built-in plot_class_distribution
            plot_class_distribution(
                data=df,
                column='label',
                top_n=20,
                figsize=(14, 6)
            )
            plt.show()
        except Exception as e:
            print(f"❌ Error plotting class distribution: {e}")
    
    def plot_correlation_matrix_maveric(self, df):
        """Plot correlation matrix using MAVERIC's plot_correlation_matrix"""
        print("📊 Plotting correlation matrix...")
        
        # Get quality metrics
        metrics = [col for col in df.columns if 'score' in col or 'consistency' in col]
        
        if len(metrics) < 2:
            print("⚠️  Need at least 2 metrics for correlation matrix")
            return
        
        try:
            # Use MAVERIC's built-in plot_correlation_matrix
            plot_correlation_matrix(
                data=df,
                metrics=metrics,
                figsize=(10, 8)
            )
            plt.show()
        except Exception as e:
            print(f"❌ Error plotting correlation matrix: {e}")
    
    def show_sample_images_maveric(self, df):
        """Show sample images using MAVERIC's SampleVisualizer"""
        print("🖼️  Showing sample images...")
        
        if 'url' not in df.columns:
            print("⚠️  No 'url' column found for sample images")
            return
        
        try:
            # Show diverse samples
            self.sample_visualizer.visualize_samples(
                data=df,
                n_samples=6,
                sample_type='diverse',
                seed=42
            )
            plt.show()
            
            # Show best samples if we have quality scores
            if 'weighted_class_score' in df.columns:
                self.sample_visualizer.visualize_samples(
                    data=df,
                    n_samples=5,
                    sample_type='best',
                    seed=42
                )
                plt.show()
                
        except Exception as e:
            print(f"❌ Error showing sample images: {e}")
    
    def plot_strategy_comparison_maveric(self):
        """Plot strategy comparison using MAVERIC's plot_quality_comparison"""
        if not self.strategy_results:
            return
        
        print("📊 Plotting strategy comparison...")
        
        try:
            # Prepare statistics for comparison
            original_stats = {
                'total_samples': self.retrieval_result.total_samples,
                'metric_statistics': self.retrieval_result.score_statistics
            }
            
            # Use balanced strategy for comparison
            if 'balanced' in self.strategy_results:
                balanced_result = self.strategy_results['balanced']['result']
                filtered_stats = {
                    'filtered_samples': balanced_result.filtered_count,
                    'retention_rate': balanced_result.retention_rate,
                    'metric_statistics': {}  # Would need to calculate from filtered data
                }
                
                # Use MAVERIC's built-in plot_quality_comparison
                plot_quality_comparison(
                    original_stats=original_stats,
                    filtered_stats=filtered_stats,
                    figsize=(14, 8)
                )
                plt.show()
                
        except Exception as e:
            print(f"❌ Error plotting strategy comparison: {e}")
    
    def print_summary(self):
        """Print final summary and recommendations"""
        if not self.strategy_results:
            return
        
        print("\\n" + "="*80)
        print("🎯 SUMMARY & RECOMMENDATIONS")
        print("="*80)
        
        print(f"\\n📊 Dataset: {self.target_dataset.upper()}")
        print(f"  • Total samples analyzed: {self.retrieval_result.total_samples:,}")
        print(f"  • Classes detected: {len(self.retrieval_result.class_distribution)}")
        
        # Show strategy results
        print(f"\\n🎛️  Strategy Results:")
        for strategy_name, result in sorted(self.strategy_results.items(), 
                                          key=lambda x: x[1]['retention_rate'], reverse=True):
            retention = result['retention_rate']
            count = result['result'].filtered_count
            print(f"  • {strategy_name.title()}: {count:,} samples ({retention:.1f}% retention)")
        
        print(f"\\n⚙️  Next Steps:")
        print(f"  1. Choose your preferred strategy based on quality vs quantity needs")
        print(f"  2. Update maveric_config.yaml with the chosen thresholds")
        print(f"  3. Run 02_data_curation.py to apply filtering and generate training dataset")
        
        # Show recommended thresholds for balanced strategy
        if 'balanced' in self.strategy_results:
            print(f"\\n📋 Recommended Thresholds (Balanced Strategy):")
            thresholds = self.strategy_results['balanced']['thresholds']
            for metric, value in thresholds.items():
                print(f"  {metric}: {value:.3f}")
        
        print(f"\\n" + "="*80)
    
    def interactive_threshold_selection(self):
        """Interactive threshold and weight selection with save functionality"""
        if not self.strategy_results:
            print("❌ No strategy results available for interactive selection")
            return
        
        print(f"\\n" + "="*80)
        print("🎛️  INTERACTIVE THRESHOLD & WEIGHT SELECTION")
        print("="*80)
        
        # Show available strategies first
        print(f"\\n📊 Available Strategies:")
        for i, (strategy_name, result) in enumerate(self.strategy_results.items(), 1):
            retention = result['retention_rate']
            count = result['result'].filtered_count
            print(f"  {i}. {strategy_name.title()}: {count:,} samples ({retention:.1f}% retention)")
        
        # Let user choose starting strategy
        while True:
            try:
                choice = input(f"\\n🔧 Choose starting strategy (1-{len(self.strategy_results)}) or 'custom' for manual: ").strip().lower()
                
                if choice == 'custom':
                    selected_thresholds = {}
                    selected_weights = self.config.get('metric_weights', {})
                    break
                else:
                    strategy_idx = int(choice) - 1
                    strategy_names = list(self.strategy_results.keys())
                    if 0 <= strategy_idx < len(strategy_names):
                        strategy_name = strategy_names[strategy_idx]
                        selected_thresholds = self.strategy_results[strategy_name]['thresholds'].copy()
                        selected_weights = self.config.get('metric_weights', {})
                        print(f"✅ Starting with {strategy_name} strategy")
                        break
                    else:
                        print(f"❌ Please enter a number between 1 and {len(self.strategy_results)}")
            except ValueError:
                print("❌ Please enter a valid number or 'custom'")
        
        # Interactive threshold adjustment
        print(f"\\n🎯 THRESHOLD ADJUSTMENT")
        print("-" * 40)
        
        available_metrics = list(self.retrieval_result.score_statistics.keys())
        for metric in available_metrics:
            if metric in ['weighted_class_score', 'consistency', 'resolution_score', 'sharpness_score', 'color_score']:
                stats = self.retrieval_result.score_statistics[metric]
                current_value = selected_thresholds.get(metric, stats['median'])
                
                print(f"\\n📈 {metric.replace('_', ' ').title()}:")
                print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
                print(f"  Mean: {stats['mean']:.3f}, Median: {stats['median']:.3f}")
                print(f"  Current: {current_value:.3f}")
                
                while True:
                    try:
                        new_value = input(f"  Enter new threshold (or press Enter to keep {current_value:.3f}): ").strip()
                        if new_value == "":
                            break
                        new_value = float(new_value)
                        if stats['min'] <= new_value <= stats['max']:
                            selected_thresholds[metric] = new_value
                            print(f"  ✅ Updated {metric} threshold to {new_value:.3f}")
                            break
                        else:
                            print(f"  ❌ Value must be between {stats['min']:.3f} and {stats['max']:.3f}")
                    except ValueError:
                        print("  ❌ Please enter a valid number")
        
        # Interactive weight adjustment
        print(f"\\n⚖️  METRIC WEIGHT ADJUSTMENT")
        print("-" * 40)
        print("Weights determine relative importance of each metric (default: equal weight)")
        
        for metric in available_metrics:
            if metric in selected_thresholds:
                current_weight = selected_weights.get(metric, 1.0)
                print(f"\\n📊 {metric.replace('_', ' ').title()}:")
                print(f"  Current weight: {current_weight:.2f}")
                
                while True:
                    try:
                        new_weight = input(f"  Enter new weight (0.1-5.0, or press Enter to keep {current_weight:.2f}): ").strip()
                        if new_weight == "":
                            break
                        new_weight = float(new_weight)
                        if 0.1 <= new_weight <= 5.0:
                            selected_weights[metric] = new_weight
                            print(f"  ✅ Updated {metric} weight to {new_weight:.2f}")
                            break
                        else:
                            print("  ❌ Weight must be between 0.1 and 5.0")
                    except ValueError:
                        print("  ❌ Please enter a valid number")
        
        # Show final selection
        print(f"\\n📋 FINAL SELECTION SUMMARY")
        print("-" * 40)
        print("Thresholds:")
        for metric, value in selected_thresholds.items():
            print(f"  {metric}: {value:.3f}")
        
        print("\\nWeights:")
        for metric, value in selected_weights.items():
            if metric in selected_thresholds:
                print(f"  {metric}: {value:.2f}")
        
        # Test the selection
        print(f"\\n🧪 Testing your selection...")
        try:
            test_result = self.maveric.quality_control(
                data=(self.target_dataset, self.raw_data_dir),
                thresholds=selected_thresholds,
                weights=selected_weights,
                balance_strategy='none'
            )
            
            if test_result:
                retention_rate = test_result.retention_rate * 100
                print(f"✅ Test successful: {test_result.filtered_count:,} samples ({retention_rate:.1f}% retention)")
            else:
                print("❌ Test failed - please adjust thresholds")
                return
                
        except Exception as e:
            print(f"❌ Test error: {e}")
            return
        
        # Save option
        while True:
            save_choice = input(f"\\n💾 Save these settings to configuration file? (y/n): ").strip().lower()
            if save_choice in ['y', 'yes']:
                self.save_configuration(selected_thresholds, selected_weights)
                break
            elif save_choice in ['n', 'no']:
                print("Settings not saved. You can run this script again to make changes.")
                break
            else:
                print("Please enter 'y' for yes or 'n' for no")
    
    def save_configuration(self, thresholds: dict, weights: dict):
        """Save thresholds and weights to configuration file"""
        try:
            # Read current configuration
            with open(self.config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Update thresholds and weights
            config_data['quality_thresholds'] = thresholds
            config_data['metric_weights'] = weights
            
            # Create backup
            backup_file = f"{self.config_file}.backup"
            with open(backup_file, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            
            # Save updated configuration
            with open(self.config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
            
            print(f"\\n✅ Configuration saved successfully!")
            print(f"📁 Updated file: {self.config_file}")
            print(f"💾 Backup created: {backup_file}")
            print(f"\\n🚀 Next Steps:")
            print(f"1. Run: python 02_data_curation.py -d {self.target_dataset} -c {self.config_file}")
            print(f"2. This will apply your chosen thresholds and create the filtered dataset")
            
        except Exception as e:
            print(f"❌ Error saving configuration: {e}")
            print("Please check file permissions and try again")

def main():
    """Main function"""
    print("🚀 MAVERIC Data Curation Analysis")
    print("Using MAVERIC Built-in Functions")
    print("="*50)
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Validate config file exists
    if not os.path.exists(args.config):
        print(f"❌ Configuration file not found: {args.config}")
        print("\\n🔧 Please provide a valid path to your MAVERIC configuration file")
        return
    
    # Initialize analyzer
    analyzer = MAVERICDataCurationAnalyzer(
        target_dataset=args.dataset,
        config_file=args.config
    )
    
    # Print final summary and offer interactive selection
    if analyzer.retrieval_result is not None:
        analyzer.print_summary()
        
        # Ask user if they want to interactively select thresholds
        while True:
            interactive_choice = input(f"\\n🎛️  Would you like to interactively select thresholds and weights? (y/n): ").strip().lower()
            if interactive_choice in ['y', 'yes']:
                analyzer.interactive_threshold_selection()
                break
            elif interactive_choice in ['n', 'no']:
                print("You can run this script again anytime to adjust thresholds and weights.")
                break
            else:
                print("Please enter 'y' for yes or 'n' for no")
    else:
        print("\\n❌ Analysis failed - no data could be loaded")
        print("\\n🔧 Troubleshooting:")
        print("1. Ensure raw data files exist in the expected directory")
        print("2. Check that 01_data_retrieval.py was run successfully")
        print("3. Verify the configuration file path and content")
        print("4. Make sure the dataset name matches the directory structure")

if __name__ == "__main__":
    # Set matplotlib backend for better Colab compatibility
    try:
        if 'google.colab' in sys.modules:
            plt.switch_backend('Agg')
        else:
            # For Jupyter and local environments
            plt.ion()  # Interactive mode
    except:
        pass
    
    main()