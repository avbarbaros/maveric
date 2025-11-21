#!/usr/bin/env python3
"""
MAVERIC Model Customization Experiment
Processes training dataset JSON and outputs the best performing CLIP model.
"""

import json
import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Add maveric to path if running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from maveric import MAVERIC
from maveric.config import MAVERICConfig, TrainingConfig
from maveric.core.interfaces import QualityResult


def load_config_file(config_path: str) -> Dict:
    """Load MAVERIC configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Configuration loaded from: {config_path}")
        return config
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return None


def load_training_dataset(input_path: str) -> Optional[List[Dict]]:
    """Load training dataset from directory or single JSON file."""
    try:
        if os.path.isdir(input_path):
            # Load from directory containing multiple JSON files
            return load_training_dataset_from_directory(input_path)
        else:
            # Load single JSON file (backward compatibility)
            with open(input_path, 'r') as f:
                data = json.load(f)
            print(f"✅ Training dataset loaded from: {input_path}")
            print(f"📊 Total training samples: {len(data)}")
            return data
    except Exception as e:
        print(f"❌ Error loading training dataset: {e}")
        return None


def load_training_dataset_from_directory(directory_path: str) -> Optional[List[Dict]]:
    """Load and combine multiple training dataset JSON files from directory."""
    try:
        directory = Path(directory_path)
        json_files = list(directory.glob("*training*maveric*.json"))
        
        if not json_files:
            print(f"❌ No training JSON files found in directory: {directory_path}")
            return None
        
        print(f"🔍 Found {len(json_files)} training JSON files in directory: {directory_path}")
        
        all_data = []
        for json_file in sorted(json_files):
            try:
                with open(json_file, 'r') as f:
                    file_data = json.load(f)
                    if isinstance(file_data, list):
                        all_data.extend(file_data)
                    else:
                        all_data.append(file_data)
                print(f"   ✅ Loaded {json_file.name}: {len(file_data) if isinstance(file_data, list) else 1} samples")
            except Exception as e:
                print(f"   ❌ Error loading {json_file.name}: {e}")
        
        if not all_data:
            print("❌ No data loaded from any files")
            return None
            
        print(f"📊 Total training samples from all files: {len(all_data)}")
        return all_data
        
    except Exception as e:
        print(f"❌ Error loading from directory: {e}")
        return None


def convert_to_quality_result(data: List[Dict], input_path: str = None) -> QualityResult:
    """Convert training dataset JSON to QualityResult object."""
    # For model customization, we treat the training data as both original and filtered
    # since it's already been through quality control
    return QualityResult(
        filtered_samples=data,
        original_samples=data,  # Same as filtered since already curated
        thresholds={},  # Already applied in previous step
        balance_strategy="applied",
        source_path=input_path  # Store input path for dataset-specific image cache
    )


def extract_dataset_info_from_path(input_path: str) -> tuple:
    """Extract dataset name from input path (directory or filename)."""
    if os.path.isdir(input_path):
        # Extract from directory name or first file found
        directory = Path(input_path)
        json_files = list(directory.glob("*training*maveric*.json"))
        if json_files:
            # Extract from first file
            basename = json_files[0].stem
            if '_training_maveric_' in basename:
                parts = basename.split('_training_maveric_')
                dataset_name = parts[0]
                return dataset_name, "multiple"
            return json_files[0].stem.split('_')[0], "multiple"
        return directory.name, "multiple"
    else:
        # Extract from filename like "cifar10_training_maveric_dataset1.json"
        basename = Path(input_path).stem  # Remove extension
        if '_training_maveric_dataset' in basename:
            parts = basename.split('_training_maveric_dataset')
            dataset_name = parts[0]
            dataset_id = int(parts[1]) if parts[1].isdigit() else 1
            return dataset_name, dataset_id
        elif '_training_maveric_' in basename:
            parts = basename.split('_training_maveric_')
            dataset_name = parts[0]
            return dataset_name, 1
        return "unknown", 1


def get_class_names_from_data(data: List[Dict]) -> List[str]:
    """Extract unique class names from the training data."""
    labels = set()
    for sample in data:
        if 'label' in sample:
            labels.add(sample['label'])
    return sorted(list(labels))


def get_training_class_sizes(data: List[Dict]) -> Dict[str, int]:
    """Calculate class distribution (number of samples per class) in training data."""
    class_counts = {}
    for sample in data:
        if 'label' in sample:
            label = sample['label']
            class_counts[label] = class_counts.get(label, 0) + 1
    return class_counts


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MAVERIC Model Customization with Mandatory Test Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IMPORTANT: Test data evaluation is mandatory for reliable model selection.
The input files must follow the naming convention: <dataset_name>_training_maveric_<id>.json
where dataset_name is a valid ELEVATER dataset (e.g., cifar10, cifar100, food101).

Examples:
  # Single file input (uses results_dir from config)
  python 03_model_customization.py --input cifar10_training_maveric_dataset1.json --config maveric_config.yaml
  
  # Directory input (uses results_dir from config)
  python 03_model_customization.py --input ./results/cifar10/ --config maveric_config.yaml --epochs 10
  
  # Custom output directory
  python 03_model_customization.py -i /path/to/training_data/ -c config.yaml --output-dir /custom/path

Note: Models will be saved to <results_dir>/<dataset>/models/ unless --output-dir is specified.
      Results will be saved to <results_dir>/<dataset>/
      Dataset cache will be stored in <cache_base_dir>/<dataset>/datasets/

Note: The model will be evaluated on the actual test set of the target dataset at each epoch.
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to training dataset JSON file or directory containing multiple JSON files'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to MAVERIC configuration YAML file'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory for model checkpoints (default: use results_dir from config)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    
    return parser.parse_args()


def setup_maveric(config: Dict, args) -> MAVERIC:
    """Setup MAVERIC instance with configuration."""
    print("🔧 Setting up MAVERIC...")
    
    try:
        # Override batch_size if provided via CLI
        batch_size = args.batch_size if args.batch_size is not None else config.get('batch_size', 32)
        
        # Create MAVERICConfig object from loaded config
        maveric_config = MAVERICConfig(
            cache_base_dir=config['cache_base_dir'],
            clip_model=config.get('clip_model', 'ViT-B/32'),
            batch_size=batch_size,
            device=config.get('device', 'auto'),
            enable_image_cache=config.get('caching', {}).get('enable_image_cache', True)
        )
        
        # Initialize MAVERIC
        maveric = MAVERIC(maveric_config)
        print("✅ MAVERIC initialized successfully")
        return maveric
        
    except Exception as e:
        print(f"❌ Error setting up MAVERIC: {e}")
        return None


def create_training_config(config: Dict, args) -> TrainingConfig:
    """Create training configuration from config file and CLI arguments."""
    # Get training config from main config
    training_cfg = config.get('training', {})
    
    # Override with command line arguments if provided
    epochs = args.epochs if args.epochs is not None else training_cfg.get('epochs', 10)
    learning_rate = args.learning_rate if args.learning_rate is not None else training_cfg.get('learning_rate', 1e-6)
    
    return TrainingConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=training_cfg.get('weight_decay', 0.01),
        warmup_steps=training_cfg.get('warmup_steps', 0),
        use_regularization=training_cfg.get('use_regularization', True),
        regularization_weight=training_cfg.get('regularization_weight', 0.5),
        use_augmentation=training_cfg.get('use_augmentation', True),
        augmentation_strength=training_cfg.get('augmentation_strength', 2),
        augmentation_magnitude=training_cfg.get('augmentation_magnitude', 9),
        optimizer=training_cfg.get('optimizer', 'adamw'),
        scheduler=training_cfg.get('scheduler', 'cosine'),
        gradient_clip_value=training_cfg.get('gradient_clip_value', 1.0),
        eval_frequency=training_cfg.get('eval_frequency', 1),
        save_best_model=training_cfg.get('save_best_model', True),
        use_validation=training_cfg.get('use_validation', True),
        validation_method=training_cfg.get('validation_method', 'stratified_kfold'),
        validation_k_folds=training_cfg.get('validation_k_folds', 5),
        validation_split=training_cfg.get('validation_split', 0.2),
        checkpoint_dir=args.output_dir,
        save_frequency=training_cfg.get('save_frequency', 1),
        keep_last_n_checkpoints=training_cfg.get('keep_last_n_checkpoints', 3),
        mixed_precision=training_cfg.get('mixed_precision', False),
        gradient_accumulation_steps=training_cfg.get('gradient_accumulation_steps', 1)
    )


def main():
    """Main model customization function."""
    args = parse_arguments()
    
    print("🚀 Starting MAVERIC Model Customization...")
    print(f"📁 Input path: {args.input}")
    print(f"📋 Configuration file: {args.config}")
    
    # Validate input path exists
    if not os.path.exists(args.input):
        print(f"❌ Input path not found: {args.input}")
        return False
    
    if not os.path.exists(args.config):
        print(f"❌ Configuration file not found: {args.config}")
        return False
    
    try:
        # Load configuration
        config = load_config_file(args.config)
        if not config:
            print("❌ Failed to load configuration")
            return False
        
        # Load training dataset
        training_data = load_training_dataset(args.input)
        if not training_data:
            print("❌ Failed to load training dataset")
            return False
        
        # Extract dataset information from input path
        target_dataset, dataset_id = extract_dataset_info_from_path(args.input)
        print(f"🎯 Target dataset: {target_dataset}")
        print(f"🆔 Dataset ID: {dataset_id}")
        
        # Validate target dataset for test evaluation
        if target_dataset == "unknown":
            print("⚠️  Warning: Could not determine target dataset from input path.")
            print("   Test data evaluation requires a valid ELEVATER dataset name.")
            print("   Please ensure your input file follows the naming convention:")
            print("   <dataset_name>_training_maveric_<id>.json")
            return False
        
        # Get correct Title Case class names from the target dataset
        # (training data may have lowercase/normalized labels, but evaluation needs proper capitalization)
        from maveric.datasets import get_dataset
        try:
            target_dataset_handler = get_dataset(target_dataset, train=False, root=None)
            if hasattr(target_dataset_handler, 'class_names'):
                class_names = target_dataset_handler.class_names
                print(f"📊 Number of classes: {len(class_names)} (using Title Case from dataset)")
                print(f"📋 Classes: {', '.join(class_names[:10])}" + ("..." if len(class_names) > 10 else ""))
            else:
                # Fallback: extract from training data
                class_names = get_class_names_from_data(training_data)
                print(f"📊 Number of classes: {len(class_names)} (extracted from training data)")
                print(f"📋 Classes: {', '.join(class_names[:10])}" + ("..." if len(class_names) > 10 else ""))
        except Exception as e:
            print(f"⚠️  Could not load dataset class names, using training data labels: {e}")
            class_names = get_class_names_from_data(training_data)
            print(f"📊 Number of classes: {len(class_names)}")
            print(f"📋 Classes: {', '.join(class_names[:10])}" + ("..." if len(class_names) > 10 else ""))
        
        # Calculate training dataset class sizes
        training_class_sizes = get_training_class_sizes(training_data)
        print(f"📈 Training class distribution:")
        for class_name, count in sorted(training_class_sizes.items()):
            print(f"   {class_name}: {count} samples")
        
        # Setup output directory from config with dataset name
        if args.output_dir is None:
            # Use results_dir from config + dataset + models subdirectory
            results_dir = config.get('results_dir', './results')
            args.output_dir = str(Path(results_dir) / target_dataset / 'models')
        
        print(f"📁 Output directory: {args.output_dir}")
        
        # Setup MAVERIC
        maveric = setup_maveric(config, args)
        if not maveric:
            print("❌ Failed to initialize MAVERIC")
            return False
        
        # Create output directory
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Convert to QualityResult
        print("🔄 Converting to QualityResult...")
        quality_result = convert_to_quality_result(training_data, args.input)
        
        # Create training configuration
        training_config = create_training_config(config, args)
        print(f"⚙️  Training configuration:")
        print(f"   Epochs: {training_config.epochs}")
        print(f"   Learning rate: {training_config.learning_rate}")
        print(f"   Weight decay: {training_config.weight_decay}")
        print(f"   Optimizer: {training_config.optimizer}")
        print(f"   Scheduler: {training_config.scheduler}")
        print(f"   Validation: {training_config.use_validation}")
        if training_config.use_validation:
            print(f"   Validation method: {training_config.validation_method}")
            if training_config.validation_method == "stratified_kfold":
                print(f"   K-folds: {training_config.validation_k_folds}")
            else:
                print(f"   Validation split: {training_config.validation_split}")
        
        # Perform model customization
        print("🤖 Starting model customization...")
        customization_result = maveric.customize_model(
            quality_result=quality_result,
            model_name=config.get('clip_model', 'ViT-B/32'),
            training_config=training_config,
            target_dataset=target_dataset
        )
        
        if customization_result is None:
            print("❌ Model customization failed")
            return False
        
        print("✅ Model customization completed successfully!")
        
        # Display results
        print("\n📊 Customization Results:")
        print(f"   Base model: {customization_result.base_model_name}")
        print(f"   Training samples: {customization_result.training_samples:,}")
        print(f"   Zero-shot baseline: {customization_result.zero_shot_baseline:.2f}%")
        print(f"   Test accuracy: {customization_result.test_accuracy:.2f}%")
        print(f"   Improvement: {customization_result.improvement:+.2f}%")
        
        if customization_result.checkpoint_path:
            print(f"   Model saved to: {customization_result.checkpoint_path}")
        
        # Display top-performing classes
        if customization_result.class_accuracies:
            print(f"\n🏆 Top 5 performing classes (customized model):")
            sorted_classes = sorted(customization_result.class_accuracies.items(), 
                                  key=lambda x: x[1], reverse=True)[:5]
            for i, (class_name, accuracy) in enumerate(sorted_classes, 1):
                print(f"   {i}. {class_name}: {accuracy:.1f}%")
        
        # Display baseline per-class performance
        if customization_result.zero_shot_class_accuracies:
            print(f"\n📊 Top 5 baseline classes (zero-shot model):")
            sorted_baseline = sorted(customization_result.zero_shot_class_accuracies.items(), 
                                   key=lambda x: x[1], reverse=True)[:5]
            for i, (class_name, accuracy) in enumerate(sorted_baseline, 1):
                print(f"   {i}. {class_name}: {accuracy:.1f}%")
        
        # Save detailed results in dataset-specific results directory
        results_dir = config.get('results_dir', './results')
        dataset_results_dir = Path(results_dir) / target_dataset
        results_file = dataset_results_dir / f"{target_dataset}_customization_results_{dataset_id}.json"
        dataset_results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump({
                'model_name': customization_result.model_name,
                'base_model_name': customization_result.base_model_name,
                'target_dataset': target_dataset,
                'training_samples': customization_result.training_samples,
                'test_accuracy': customization_result.test_accuracy,
                'zero_shot_baseline': customization_result.zero_shot_baseline,
                'improvement': customization_result.improvement,
                'class_accuracies': customization_result.class_accuracies,
                'zero_shot_class_accuracies': customization_result.zero_shot_class_accuracies,
                'training_class_sizes': training_class_sizes,
                'training_config': customization_result.training_config,
                'checkpoint_path': customization_result.checkpoint_path
            }, f, indent=2)
        
        print(f"📁 Results saved to: {results_file}")
        
        print("\n🎉 Model customization completed successfully!")
        print(f"🤖 Best performing model available at: {customization_result.checkpoint_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during model customization: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)