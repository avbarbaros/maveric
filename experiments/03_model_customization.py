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

MODES:
  1. Per-Dataset Training (default):
     Input must follow naming convention: <dataset_name>_training_maveric_<id>.json
     where dataset_name is a valid ELEVATER dataset (e.g., cifar10, cifar100, food101).

  2. Unified Training (--unified-training):
     Input must be a directory containing dataset subdirectories, each with JSON files.
     Structure: unified_data/cifar10/*.json, unified_data/cifar100/*.json, etc.

Examples:
  # Per-dataset training (single file)
  python 03_model_customization.py --input cifar10_training_maveric_dataset1.json --config maveric_config.yaml

  # Per-dataset training (directory)
  python 03_model_customization.py --input ./results/cifar10/ --config maveric_config.yaml --epochs 10

  # Unified training (REACT-style, all datasets)
  python 03_model_customization.py --unified-training --input ./unified_training_data --config maveric_config.yaml

  # Unified training with sample balancing
  python 03_model_customization.py --unified-training --input ./unified_training_data --max-samples-per-dataset 1000

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
        help='Path to training dataset JSON file/directory (per-dataset) or unified directory (with --unified-training)'
    )

    parser.add_argument(
        '--unified-training',
        action='store_true',
        help='Enable REACT-style unified training across multiple datasets'
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

    parser.add_argument(
        '--save-augmented-grids',
        action='store_true',
        help='Save 10x10 grid visualizations of augmented/domain-adapted training samples for inspection'
    )

    parser.add_argument(
        '--max-samples-per-dataset',
        type=int,
        default=None,
        help='(Unified training only) Maximum samples per dataset for balancing (default: no limit)'
    )

    parser.add_argument(
        '--save-individual-results',
        action='store_true',
        default=True,
        help='(Unified training only) Save per-dataset results separately (default: True)'
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


def run_unified_training(config: Dict, args) -> bool:
    """
    Run unified training mode (REACT-style multi-dataset training).

    Args:
        config: Configuration dictionary
        args: Command-line arguments

    Returns:
        True if successful, False otherwise
    """
    from maveric.customization.unified_training import (
        load_datasets_from_directory,
        load_unified_dataset,
        unify_class_names,
        UnifiedELEVATERDataset,
        evaluate_unified_model_per_dataset,
        save_unified_results
    )
    from torch.utils.data import DataLoader
    from transformers import CLIPProcessor
    import torch

    try:
        print(f"📂 Loading datasets from: {args.input}")

        # Step 1: Load datasets from directory structure
        dataset_files = load_datasets_from_directory(args.input)

        # Step 2: Load and combine samples
        unified_data = load_unified_dataset(
            dataset_files=dataset_files,
            max_samples_per_dataset=args.max_samples_per_dataset,
            seed=config.get('seed', 42)
        )

        # Step 3: Create unified class space
        class_info = unify_class_names(unified_data['dataset_metadata'])

        # Step 4: Setup MAVERIC
        maveric = setup_maveric(config, args)
        if not maveric:
            print("❌ Failed to initialize MAVERIC")
            return False

        # Step 5: Load CLIP processor
        clip_model = config.get('clip_model', 'ViT-B/32')
        print(f"\n🔧 Loading CLIP processor for {clip_model}...")

        # Map OpenAI CLIP model names to Hugging Face model identifiers
        model_mapping = {
            "ViT-B/32": "openai/clip-vit-base-patch32",
            "ViT-B/16": "openai/clip-vit-base-patch16",
            "ViT-L/14": "openai/clip-vit-large-patch14",
            "ViT-L/14@336px": "openai/clip-vit-large-patch14-336",
            "RN50": "openai/clip-resnet-50",
            "RN101": "openai/clip-resnet-101",
            "RN50x4": "openai/clip-resnet-50x4",
            "RN50x16": "openai/clip-resnet-50x16",
            "RN50x64": "openai/clip-resnet-50x64"
        }
        hf_model_name = model_mapping.get(clip_model, clip_model)
        processor = CLIPProcessor.from_pretrained(hf_model_name)

        # Step 6: Create unified dataset with dataset-specific domain adaptation
        print("\n📦 Creating unified training dataset...")
        training_cfg = config.get('training', {})

        # Prepare dataset-specific domain adaptation settings
        dataset_domain_adaptation = training_cfg.get('dataset_domain_adaptation', {})

        # Prepare global domain adaptation config (used as fallback)
        global_domain_config = {
            'use_domain_adaptation': training_cfg.get('use_domain_adaptation', False),
            'domain_blur_probability': training_cfg.get('domain_blur_probability', 0.5),
            'domain_blur_sigma_range': training_cfg.get('domain_blur_sigma_range', [0.5, 2.0]),
            'domain_jpeg_probability': training_cfg.get('domain_jpeg_probability', 0.4),
            'domain_jpeg_quality_range': training_cfg.get('domain_jpeg_quality_range', [50, 90]),
            'domain_downsample_probability': training_cfg.get('domain_downsample_probability', 0.7),
            'domain_target_size': training_cfg.get('domain_target_size', None),
            'domain_downsample_scale_range': training_cfg.get('domain_downsample_scale_range', [0.5, 0.9])
        }

        training_dataset = UnifiedELEVATERDataset(
            unified_data=unified_data,
            class_info=class_info,
            processor=processor,
            use_augmentation=training_cfg.get('use_augmentation', True),
            augmentation_config={
                'augmentation_strength': training_cfg.get('augmentation_strength', 2),
                'augmentation_magnitude': training_cfg.get('augmentation_magnitude', 9)
            },
            dataset_domain_adaptation=dataset_domain_adaptation,
            global_domain_config=global_domain_config,
            cache_dir=config.get('cache_base_dir', './maveric_cache'),
            training_data_dir=Path(args.input).parent.resolve()  # Parent dir contains dataset folders with images/ subdirs
        )

        # Step 7: Create data loader with custom collate function for PIL images
        from maveric.customization.model_customizer import custom_collate_fn

        batch_size = config.get('training', {}).get('batch_size', 32)
        train_loader = DataLoader(
            training_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
            collate_fn=custom_collate_fn
        )

        print(f"   Training samples: {len(training_dataset):,}")
        print(f"   Total classes: {class_info['num_total_classes']}")
        print(f"   Datasets included: {len(unified_data['dataset_metadata'])}")
        print(f"   Batch size: {batch_size}")

        # Step 8: Setup output directory
        if args.output_dir is None:
            results_dir = config.get('results_dir', './results')
            args.output_dir = str(Path(results_dir) / 'unified_training' / 'models')

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory: {args.output_dir}")

        # Step 9: Create training configuration
        training_config = create_training_config(config, args)
        print(f"\n⚙️  Training configuration:")
        print(f"   Epochs: {training_config.epochs}")
        print(f"   Learning rate: {training_config.learning_rate}")
        print(f"   Weight decay: {training_config.weight_decay}")
        print(f"   Optimizer: {training_config.optimizer}")
        print(f"   Scheduler: {training_config.scheduler}")
        print(f"   Skip epoch evaluation: {training_config.skip_epoch_evaluation}")
        if training_config.skip_epoch_evaluation:
            print(f"   ⚠️  Per-epoch evaluation DISABLED (unified training mode)")
            print(f"   ✓  Checkpoints will be saved every {training_config.save_frequency} epoch(s)")

        # Step 10: Train unified model
        print("\n🤖 Training unified model...")
        from maveric.customization.training import Trainer
        from maveric.customization.model_customizer import ModelCustomizer, CustomizedCLIP

        # Create model customizer
        customizer = ModelCustomizer(
            base_model_name=clip_model,
            device=config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
            checkpoint_dir=args.output_dir,
            cache_base_dir=config.get('cache_base_dir', './maveric_cache')
        )

        # Wrap model in CustomizedCLIP (provides clip_model attribute and locked text encoder)
        customized_model = CustomizedCLIP(
            customizer.model,
            customizer.processor,
            regularize=training_config.use_regularization
        ).to(customizer.device)

        # Attach processor to model (required by Trainer for text encoding)
        customized_model.processor = processor

        # Train model
        trainer = Trainer(
            model=customized_model,
            device=customizer.device,
            checkpoint_dir=Path(args.output_dir)
        )

        # For unified training, use training data for monitoring (no separate test set available)
        # This is just for tracking training progress, not for true evaluation
        training_results = trainer.train(
            train_loader=train_loader,
            val_loader=None,
            test_loader=train_loader,  # Use train data for monitoring (no true test data available)
            training_config=training_config,
            class_names=class_info['global_class_names']
        )

        print("\n✅ Training completed!")
        print(f"   Final loss: {training_results.get('final_loss', 'N/A')}")

        # Step 11: Evaluate on each dataset separately
        if args.save_individual_results:
            print("\n" + "=" * 80)
            print("📊 EVALUATING UNIFIED MODEL ON INDIVIDUAL DATASETS")
            print("=" * 80)

            per_dataset_results = evaluate_unified_model_per_dataset(
                model=customizer.model,
                dataset_metadata=unified_data['dataset_metadata'],
                class_offsets=class_info['dataset_class_offsets'],
                processor=processor,
                device=customizer.device,
                batch_size=config.get('training', {}).get('batch_size', 32),
                use_templates=True,
                cache_base_dir=config.get('cache_base_dir', './maveric_cache')
            )

            # Step 12: Save results
            results_dir = Path(args.output_dir).parent  # Go up from models/ to results/unified_training/
            save_unified_results(
                results=per_dataset_results,
                output_dir=str(results_dir),
                filename="unified_training_results.json"
            )

        # Step 13: Save unified model checkpoint
        checkpoint_path = Path(args.output_dir) / "unified_model_best.pth"
        torch.save({
            'model_state_dict': customizer.model.state_dict(),
            'class_info': class_info,
            'dataset_metadata': unified_data['dataset_metadata'],
            'training_config': training_config.__dict__,
            'clip_model': clip_model,
            'num_total_classes': class_info['num_total_classes']
        }, checkpoint_path)

        print(f"\n💾 Unified model saved to: {checkpoint_path}")
        print("\n🎉 Unified training completed successfully!")

        return True

    except Exception as e:
        print(f"❌ Error during unified training: {e}")
        import traceback
        traceback.print_exc()
        return False


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
        use_domain_adaptation=training_cfg.get('use_domain_adaptation', False),
        domain_blur_probability=training_cfg.get('domain_blur_probability', 0.3),
        domain_blur_sigma_range=tuple(training_cfg.get('domain_blur_sigma_range', [0.1, 2.0])),
        domain_jpeg_probability=training_cfg.get('domain_jpeg_probability', 0.3),
        domain_jpeg_quality_range=tuple(training_cfg.get('domain_jpeg_quality_range', [30, 95])),
        domain_downsample_probability=training_cfg.get('domain_downsample_probability', 0.3),
        domain_target_size=training_cfg.get('domain_target_size', None),
        domain_downsample_scale_range=tuple(training_cfg.get('domain_downsample_scale_range', [0.5, 0.9])),
        optimizer=training_cfg.get('optimizer', 'adamw'),
        scheduler=training_cfg.get('scheduler', 'cosine'),
        gradient_clip_value=training_cfg.get('gradient_clip_value', 1.0),
        eval_frequency=training_cfg.get('eval_frequency', 1),
        save_best_model=training_cfg.get('save_best_model', True),
        skip_epoch_evaluation=training_cfg.get('skip_epoch_evaluation', False),
        use_validation=training_cfg.get('use_validation', True),
        validation_method=training_cfg.get('validation_method', 'stratified_kfold'),
        validation_k_folds=training_cfg.get('validation_k_folds', 5),
        validation_split=training_cfg.get('validation_split', 0.2),
        checkpoint_dir=args.output_dir,
        save_frequency=training_cfg.get('save_frequency', 1),
        keep_last_n_checkpoints=training_cfg.get('keep_last_n_checkpoints', 3),
        mixed_precision=training_cfg.get('mixed_precision', False),
        gradient_accumulation_steps=training_cfg.get('gradient_accumulation_steps', 1),
        checkpoint_selection_metric=training_cfg.get('checkpoint_selection_metric', 'val_acc'),
        evaluate_test_each_epoch=training_cfg.get('evaluate_test_each_epoch', False),
        text_source=training_cfg.get('text_source', 'labels')
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

        # Check if unified training mode
        if args.unified_training:
            print("\n🌐 UNIFIED TRAINING MODE (REACT-style)")
            print("=" * 80)
            return run_unified_training(config, args)
        
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
        
        # Get correct class names from ELEVATER dataset definition
        # CRITICAL: Must use exact class names from elevater_datasets.py (as REACT uses them)
        # DO NOT load from dataset handler as torchvision may override with its own class names
        from maveric.datasets.elevater_datasets import ELEVATERDataset

        if target_dataset in ELEVATERDataset.ELEVATER_DATASETS:
            # Load class names directly from ELEVATER_DATASETS dictionary
            # This ensures we use the EXACT REACT class names with proper capitalization
            class_names = ELEVATERDataset.ELEVATER_DATASETS[target_dataset]['class_names']
            print(f"📊 Number of classes: {len(class_names)} (from ELEVATER dataset definition)")

            # Handle FER2013-style list class names for display
            display_names = [name[0] if isinstance(name, list) else name for name in class_names[:10]]
            print(f"📋 First 10 classes: {', '.join(display_names)}" + ("..." if len(class_names) > 10 else ""))

            # Display first class name (handle list format)
            first_class = class_names[0][0] if isinstance(class_names[0], list) else class_names[0]
            print(f"📋 Example prompts will use: '{first_class}' (note: proper capitalization)")
        else:
            # Fallback: try to load from dataset handler (for non-ELEVATER datasets)
            print(f"⚠️  Warning: {target_dataset} not found in ELEVATER_DATASETS")
            try:
                from maveric.datasets import get_dataset
                target_dataset_handler = get_dataset(target_dataset, train=False, root=None)
                if hasattr(target_dataset_handler, 'class_names'):
                    class_names = target_dataset_handler.class_names
                    print(f"📊 Number of classes: {len(class_names)} (from dataset handler)")
                    print(f"📋 First 10 classes: {', '.join(class_names[:10])}" + ("..." if len(class_names) > 10 else ""))
                else:
                    # Last resort: extract from training data
                    class_names = get_class_names_from_data(training_data)
                    print(f"⚠️  Number of classes: {len(class_names)} (extracted from training data - may have wrong capitalization)")
                    print(f"📋 First 10 classes: {', '.join(class_names[:10])}" + ("..." if len(class_names) > 10 else ""))
            except Exception as e:
                print(f"⚠️  Could not load dataset class names: {e}")
                print(f"⚠️  Falling back to training data labels (may have wrong capitalization)")
                class_names = get_class_names_from_data(training_data)
                print(f"📊 Number of classes: {len(class_names)}")
                print(f"📋 First 10 classes: {', '.join(class_names[:10])}" + ("..." if len(class_names) > 10 else ""))
        
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
            target_dataset=target_dataset,
            class_names=class_names,  # Pass ELEVATER_DATASETS class names for accurate evaluation
            save_augmented_grids=args.save_augmented_grids  # Save grid visualizations if requested
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
                "validation_samples": customization_result.validation_samples,
                "test_samples": customization_result.test_samples,
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