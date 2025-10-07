#!/usr/bin/env python3
"""
MAVERIC Hyperparameter Search Experiment
Systematic hyperparameter tuning to optimize model performance.
"""

import json
import os
import sys
import yaml
import argparse
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np

# Add maveric to path if running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from maveric import MAVERIC
from maveric.config import MAVERICConfig, TrainingConfig
from maveric.core.interfaces import QualityResult


class HyperparameterSearch:
    """Manages hyperparameter search experiments."""

    def __init__(self,
                 base_config_path: str,
                 input_path: str,
                 output_dir: str):
        """
        Initialize hyperparameter search.

        Args:
            base_config_path: Path to base MAVERIC configuration
            input_path: Path to training dataset
            output_dir: Directory for search results
        """
        self.base_config_path = base_config_path
        self.input_path = input_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load base configuration
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)

        # Results tracking
        self.results = []
        self.best_result = None

    def define_search_space(self, search_type: str = "focused") -> Dict[str, List[Any]]:
        """
        Define hyperparameter search space.

        Args:
            search_type: Type of search ("focused", "broad", "regularization")

        Returns:
            Dictionary mapping parameter names to value lists
        """
        if search_type == "focused":
            # Focused search around optimal regularization_weight=0.5
            return {
                'regularization_weight': [0.40, 0.45, 0.50, 0.55, 0.60],
                'learning_rate': [5e-7, 1e-6, 1.5e-6, 2e-6],
                'weight_decay': [0.005, 0.01, 0.015],
                'epochs': [10, 15]
            }

        elif search_type == "broad":
            # Broader search across multiple dimensions
            return {
                'regularization_weight': [0.1, 0.25, 0.5, 0.75, 1.0],
                'learning_rate': [1e-7, 5e-7, 1e-6, 5e-6, 1e-5],
                'weight_decay': [0.001, 0.005, 0.01, 0.05],
                'epochs': [5, 10, 15, 20],
                'warmup_steps': [0, 50, 100],
                'optimizer': ['adamw', 'adam'],
                'scheduler': ['cosine', 'linear', 'constant']
            }

        elif search_type == "regularization":
            # Fine-grained search focused only on regularization
            return {
                'regularization_weight': [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65],
                'learning_rate': [1e-6],  # Keep fixed
                'weight_decay': [0.01]     # Keep fixed
            }

        elif search_type == "learning_rate":
            # Focused on learning rate optimization
            return {
                'regularization_weight': [0.5],  # Keep optimal
                'learning_rate': [1e-7, 5e-7, 8e-7, 1e-6, 1.2e-6, 1.5e-6, 2e-6, 3e-6],
                'weight_decay': [0.005, 0.01, 0.015]
            }

        else:
            raise ValueError(f"Unknown search type: {search_type}")

    def grid_search(self, search_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations for grid search.

        Args:
            search_space: Dictionary mapping parameters to value lists

        Returns:
            List of parameter configuration dictionaries
        """
        keys = list(search_space.keys())
        values = list(search_space.values())

        configurations = []
        for combination in itertools.product(*values):
            config = dict(zip(keys, combination))
            configurations.append(config)

        print(f"Generated {len(configurations)} configurations for grid search")
        return configurations

    def random_search(self,
                     search_space: Dict[str, List[Any]],
                     n_samples: int = 20,
                     seed: int = 42) -> List[Dict[str, Any]]:
        """
        Generate random parameter combinations.

        Args:
            search_space: Dictionary mapping parameters to value lists
            n_samples: Number of random configurations to generate
            seed: Random seed for reproducibility

        Returns:
            List of parameter configuration dictionaries
        """
        np.random.seed(seed)

        configurations = []
        for _ in range(n_samples):
            config = {}
            for param, values in search_space.items():
                config[param] = np.random.choice(values)
            configurations.append(config)

        print(f"Generated {len(configurations)} random configurations")
        return configurations

    def run_experiment(self, params: Dict[str, Any], exp_id: int) -> Optional[Dict[str, Any]]:
        """
        Run a single experiment with given hyperparameters.

        Args:
            params: Hyperparameter configuration
            exp_id: Experiment ID for tracking

        Returns:
            Results dictionary or None if failed
        """
        print(f"\n{'='*80}")
        print(f"Experiment {exp_id}")
        print(f"{'='*80}")
        print("Parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        print(f"{'='*80}\n")

        try:
            # Create modified config
            config = self.base_config.copy()

            # Update training parameters
            if 'training' not in config:
                config['training'] = {}

            for param, value in params.items():
                config['training'][param] = value

            # Extract dataset info from input path
            target_dataset, dataset_id = self._extract_dataset_info(self.input_path)

            # Create experiment-specific output directory
            exp_output_dir = self.output_dir / f"exp_{exp_id:03d}"
            exp_output_dir.mkdir(parents=True, exist_ok=True)

            # Save experiment config
            config_path = exp_output_dir / "config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            # Load training data
            training_data = self._load_training_dataset(self.input_path)
            if not training_data:
                print(f"❌ Failed to load training data for experiment {exp_id}")
                return None

            # Setup MAVERIC
            maveric_config = MAVERICConfig(
                cache_base_dir=config['cache_base_dir'],
                clip_model=config.get('clip_model', 'ViT-B/32'),
                batch_size=config.get('batch_size', 32),
                device=config.get('device', 'auto')
            )

            maveric = MAVERIC(maveric_config)

            # Create training config
            training_config = TrainingConfig(**config['training'])
            training_config.checkpoint_dir = str(exp_output_dir / 'models')

            # Convert to QualityResult
            quality_result = QualityResult(
                filtered_samples=training_data,
                original_samples=training_data,
                thresholds={},
                balance_strategy="applied"
            )

            # Get class names
            class_names = self._get_class_names(training_data)

            # Run customization
            print(f"🚀 Starting model customization for experiment {exp_id}...")
            customization_result = maveric.customize_model(
                quality_result=quality_result,
                model_name=config.get('clip_model', 'ViT-B/32'),
                training_config=training_config,
                target_dataset=target_dataset
            )

            if customization_result is None:
                print(f"❌ Experiment {exp_id} failed")
                return None

            # Compile results
            result = {
                'experiment_id': exp_id,
                'timestamp': datetime.now().isoformat(),
                'parameters': params,
                'metrics': {
                    'test_accuracy': customization_result.test_accuracy,
                    'zero_shot_baseline': customization_result.zero_shot_baseline,
                    'improvement': customization_result.improvement,
                    'training_samples': customization_result.training_samples
                },
                'class_accuracies': customization_result.class_accuracies,
                'checkpoint_path': customization_result.checkpoint_path
            }

            # Save experiment result
            result_path = exp_output_dir / "result.json"
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)

            print(f"\n✅ Experiment {exp_id} completed")
            print(f"   Test Accuracy: {customization_result.test_accuracy:.2f}%")
            print(f"   Improvement: {customization_result.improvement:+.2f}%")
            print(f"   Results saved to: {result_path}")

            return result

        except Exception as e:
            print(f"❌ Experiment {exp_id} failed with error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_search(self,
                   search_type: str = "focused",
                   method: str = "grid",
                   n_random_samples: int = 20,
                   resume_from: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Run hyperparameter search.

        Args:
            search_type: Type of search space to use
            method: Search method ("grid" or "random")
            n_random_samples: Number of samples for random search
            resume_from: Resume from experiment ID (for interrupted searches)

        Returns:
            List of all experiment results
        """
        # Define search space
        search_space = self.define_search_space(search_type)

        print(f"\n🔍 Starting {method} search with '{search_type}' search space")
        print(f"Search space:")
        for param, values in search_space.items():
            print(f"  {param}: {values}")

        # Generate configurations
        if method == "grid":
            configurations = self.grid_search(search_space)
        elif method == "random":
            configurations = self.random_search(search_space, n_random_samples)
        else:
            raise ValueError(f"Unknown search method: {method}")

        # Run experiments
        start_idx = resume_from if resume_from is not None else 0

        for idx, params in enumerate(configurations[start_idx:], start=start_idx):
            result = self.run_experiment(params, idx + 1)

            if result:
                self.results.append(result)

                # Track best result
                if self.best_result is None or result['metrics']['test_accuracy'] > self.best_result['metrics']['test_accuracy']:
                    self.best_result = result
                    print(f"\n🏆 New best result! Accuracy: {result['metrics']['test_accuracy']:.2f}%")

            # Save intermediate results after each experiment
            self._save_summary()

        return self.results

    def _save_summary(self):
        """Save summary of all results."""
        summary_path = self.output_dir / "search_summary.json"

        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(self.results),
            'best_result': self.best_result,
            'all_results': self.results
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n📊 Search summary saved to: {summary_path}")

    def analyze_results(self):
        """Analyze and visualize search results."""
        if not self.results:
            print("No results to analyze")
            return

        print(f"\n{'='*80}")
        print("HYPERPARAMETER SEARCH RESULTS")
        print(f"{'='*80}\n")

        # Sort by accuracy
        sorted_results = sorted(self.results, key=lambda x: x['metrics']['test_accuracy'], reverse=True)

        # Top 5 configurations
        print("🏆 Top 5 Configurations:\n")
        for i, result in enumerate(sorted_results[:5], 1):
            print(f"{i}. Experiment {result['experiment_id']}: {result['metrics']['test_accuracy']:.2f}%")
            print(f"   Parameters:")
            for param, value in result['parameters'].items():
                print(f"     {param}: {value}")
            print()

        # Best configuration details
        if self.best_result:
            print(f"\n{'='*80}")
            print("BEST CONFIGURATION")
            print(f"{'='*80}\n")
            print(f"Experiment ID: {self.best_result['experiment_id']}")
            print(f"Test Accuracy: {self.best_result['metrics']['test_accuracy']:.2f}%")
            print(f"Improvement: {self.best_result['metrics']['improvement']:+.2f}%")
            print(f"\nOptimal Parameters:")
            for param, value in self.best_result['parameters'].items():
                print(f"  {param}: {value}")
            print(f"\nCheckpoint: {self.best_result.get('checkpoint_path', 'N/A')}")

        # Parameter impact analysis
        print(f"\n{'='*80}")
        print("PARAMETER IMPACT ANALYSIS")
        print(f"{'='*80}\n")
        self._analyze_parameter_impact()

    def _analyze_parameter_impact(self):
        """Analyze impact of each parameter on performance."""
        if not self.results:
            return

        # Collect parameter values and corresponding accuracies
        param_impacts = {}

        for result in self.results:
            for param, value in result['parameters'].items():
                if param not in param_impacts:
                    param_impacts[param] = {}

                if value not in param_impacts[param]:
                    param_impacts[param][value] = []

                param_impacts[param][value].append(result['metrics']['test_accuracy'])

        # Calculate average accuracy for each parameter value
        for param, values in param_impacts.items():
            print(f"\n{param}:")
            sorted_values = sorted(values.items(), key=lambda x: np.mean(x[1]), reverse=True)
            for value, accuracies in sorted_values:
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                print(f"  {value}: {mean_acc:.2f}% ± {std_acc:.2f}% (n={len(accuracies)})")

    def _extract_dataset_info(self, input_path: str) -> tuple:
        """Extract dataset name from input path."""
        if os.path.isdir(input_path):
            directory = Path(input_path)
            json_files = list(directory.glob("*training*maveric*.json"))
            if json_files:
                basename = json_files[0].stem
                if '_training_maveric_' in basename:
                    parts = basename.split('_training_maveric_')
                    return parts[0], "multiple"
            return directory.name, "multiple"
        else:
            basename = Path(input_path).stem
            if '_training_maveric_dataset' in basename:
                parts = basename.split('_training_maveric_dataset')
                dataset_name = parts[0]
                dataset_id = int(parts[1]) if parts[1].isdigit() else 1
                return dataset_name, dataset_id
            elif '_training_maveric_' in basename:
                parts = basename.split('_training_maveric_')
                return parts[0], 1
            return "unknown", 1

    def _load_training_dataset(self, input_path: str) -> Optional[List[Dict]]:
        """Load training dataset."""
        try:
            if os.path.isdir(input_path):
                directory = Path(input_path)
                json_files = list(directory.glob("*training*maveric*.json"))
                all_data = []
                for json_file in sorted(json_files):
                    with open(json_file, 'r') as f:
                        file_data = json.load(f)
                        if isinstance(file_data, list):
                            all_data.extend(file_data)
                        else:
                            all_data.append(file_data)
                return all_data
            else:
                with open(input_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading training dataset: {e}")
            return None

    def _get_class_names(self, data: List[Dict]) -> List[str]:
        """Extract class names from training data."""
        labels = set()
        for sample in data:
            if 'label' in sample:
                labels.add(sample['label'])
        return sorted(list(labels))


def main():
    """Main hyperparameter search function."""
    parser = argparse.ArgumentParser(
        description="MAVERIC Hyperparameter Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Focused search around optimal regularization_weight
  python 05_hyperparameter_search.py -i data/training/ -c maveric_config.yaml -o results/hp_search/ --search-type focused

  # Fine-grained regularization search
  python 05_hyperparameter_search.py -i data/training/ -c maveric_config.yaml -o results/hp_search/ --search-type regularization

  # Learning rate optimization
  python 05_hyperparameter_search.py -i data/training/ -c maveric_config.yaml -o results/hp_search/ --search-type learning_rate

  # Random search (20 samples)
  python 05_hyperparameter_search.py -i data/training/ -c maveric_config.yaml -o results/hp_search/ --method random -n 20

  # Resume interrupted search
  python 05_hyperparameter_search.py -i data/training/ -c maveric_config.yaml -o results/hp_search/ --resume-from 5
        """
    )

    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to training dataset or directory')
    parser.add_argument('--config', '-c', type=str, required=True,
                       help='Path to base MAVERIC configuration YAML')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output directory for search results')
    parser.add_argument('--search-type', type=str, default='focused',
                       choices=['focused', 'broad', 'regularization', 'learning_rate'],
                       help='Type of search space to use')
    parser.add_argument('--method', type=str, default='grid',
                       choices=['grid', 'random'],
                       help='Search method (grid or random)')
    parser.add_argument('--n-random', '-n', type=int, default=20,
                       help='Number of random samples (for random search)')
    parser.add_argument('--resume-from', type=int, default=None,
                       help='Resume from experiment ID')

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.input):
        print(f"❌ Input path not found: {args.input}")
        return False

    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        return False

    # Initialize search
    print("🔍 Initializing hyperparameter search...")
    search = HyperparameterSearch(
        base_config_path=args.config,
        input_path=args.input,
        output_dir=args.output
    )

    # Run search
    results = search.run_search(
        search_type=args.search_type,
        method=args.method,
        n_random_samples=args.n_random,
        resume_from=args.resume_from
    )

    # Analyze results
    search.analyze_results()

    print(f"\n✅ Hyperparameter search complete!")
    print(f"📊 Results saved to: {args.output}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
