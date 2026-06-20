"""Main model customization module."""

from matplotlib import image
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import hashlib
import os
from transformers import CLIPModel, CLIPProcessor
from PIL import Image, ImageFilter
import io
import random

from ..core.base import BaseComponent
from ..core.interfaces import CustomizationResult, QualityResult
from ..core.exceptions import ModelError
from ..config import TrainingConfig
from .training import Trainer
from .evaluation import Evaluator


class ModelCustomizer(BaseComponent):
    """
    Handles model customization using filtered high-quality data.
    
    This component manages the entire customization pipeline, from
    loading pre-trained models to fine-tuning with curated data.
    """
    
    def __init__(self, 
                 base_model_name: str = "openai/clip-vit-base-patch32",
                 device: str = "cuda",
                 checkpoint_dir: Optional[str] = None,
                 cache_base_dir: Optional[str] = None):
        """
        Initialize model customizer.
        
        Args:
            base_model_name: Pre-trained model to customize
            device: Device for computation
            checkpoint_dir: Directory for saving checkpoints
            cache_base_dir: Base directory for caching datasets
        """
        super().__init__("ModelCustomizer")
        
        self.base_model_name = base_model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.cache_base_dir = cache_base_dir
        
        # Components
        self.model = None
        self.processor = None
        self.trainer = None
        self.evaluator = None
        
        # Load base model
        self._load_base_model()
    
    def _load_base_model(self):
        """Load the base pre-trained model."""
        try:
            self.log_info(f"Loading base model: {self.base_model_name}")
            
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
            
            # Get the correct Hugging Face model name
            hf_model_name = model_mapping.get(self.base_model_name, self.base_model_name)
            
            if hf_model_name != self.base_model_name:
                self.log_info(f"Mapping {self.base_model_name} to Hugging Face model: {hf_model_name}")
            
            self.model = CLIPModel.from_pretrained(hf_model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(hf_model_name)
            
        except Exception as e:
            raise ModelError(f"Failed to load base model: {e}")
    
    def customize(self,
                  quality_result: QualityResult,
                  training_config: TrainingConfig,
                  target_dataset_name: str,
                  class_names: List[str],
                  validation_split: float = 0.2,
                  save_augmented_grids: bool = False) -> CustomizationResult:
        """
        Customize model using filtered data.

        Args:
            quality_result: Filtered data from quality control
            training_config: Training configuration
            target_dataset_name: Name of target dataset
            class_names: List of class names
            validation_split: Fraction of data for validation
            save_augmented_grids: If True, save 10x10 grid visualizations of augmented/domain-adapted samples

        Returns:
            CustomizationResult with trained model and metrics

        Note:
            Test data evaluation is mandatory for reliable model selection.
        """
        self.log_info("Starting model customization")
        
        # Create customized model
        customized_model = CustomizedCLIP(
            self.model,
            self.processor,
            regularize=training_config.use_regularization
        ).to(self.device)
        
        # Prepare data (test data is mandatory)
        train_loader, val_loader, test_loader = self._prepare_data(
            quality_result,
            class_names,
            training_config,
            target_dataset_name
        )

        # Save augmented/domain-adapted sample grids for inspection (if requested)
        if save_augmented_grids and train_loader is not None:
            self._save_augmented_grids(train_loader.dataset, target_dataset_name, training_config)

        # Create trainer
        self.trainer = Trainer(
            model=customized_model,
            device=self.device,
            checkpoint_dir=self.checkpoint_dir
        )
        
        # Create evaluator
        self.evaluator = Evaluator(device=self.device)
        
        # Get dataset templates for evaluation
        templates = self._get_dataset_templates(target_dataset_name)

        # Get baseline performance (always use test data)
        self.log_info("Evaluating baseline model")
        if not test_loader:
            raise ValueError(f"Test data is required for evaluation but could not load test set for {target_dataset_name}")
        baseline_accuracy, baseline_class_accuracies = self._evaluate_baseline(test_loader, class_names, templates=templates)
        
        # Train model (test data is mandatory)
        self.log_info("Training customized model")
        training_history = self.trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            training_config=training_config,
            class_names=class_names,
            templates=templates,
            evaluator=self.evaluator
        )
        
        # Load and evaluate the best model from training
        best_checkpoint = None
        if training_config.save_best_model and self.checkpoint_dir:
            # The best model is already saved during training as "best_model.pth"
            best_checkpoint = self.checkpoint_dir / "best_model.pth"
            if best_checkpoint.exists():
                self.log_info("Loading best model checkpoint for final evaluation")
                best_model = self.load_checkpoint(str(best_checkpoint))
                # Final evaluation using the best model
                self.log_info("Evaluating best customized model on test set")
                final_accuracy, class_accuracies = self.evaluator.evaluate_detailed(
                    best_model,
                    test_loader,
                    class_names,
                    templates=templates
                )
            else:
                # Fallback: evaluate current model and save it
                self.log_info("No best checkpoint found, evaluating current model")
                final_accuracy, class_accuracies = self.evaluator.evaluate_detailed(
                    customized_model,
                    test_loader,
                    class_names,
                    templates=templates
                )
                best_checkpoint = self.trainer.save_checkpoint(
                    customized_model,
                    f"best_model_{target_dataset_name}",
                    {
                        'accuracy': final_accuracy,
                        'baseline': baseline_accuracy,
                        'config': training_config.to_dict()
                    }
                )
        else:
            # No checkpointing enabled, evaluate current model
            self.log_info("Evaluating current customized model on test set")
            final_accuracy, class_accuracies = self.evaluator.evaluate_detailed(
                customized_model,
                test_loader,
                class_names,
                templates=templates
            )
        



        # Create result
        result = CustomizationResult(
            model_name=f"customized_{self.base_model_name}",
            base_model_name=self.base_model_name,
            training_config=training_config.to_dict(),
            training_samples=len(train_loader.dataset),
            validation_samples = len(val_loader.dataset) if val_loader is not None else 0,
            test_samples = len(test_loader.dataset),
            test_accuracy=final_accuracy,
            zero_shot_baseline=baseline_accuracy,
            class_accuracies=class_accuracies,
            zero_shot_class_accuracies=baseline_class_accuracies,
            training_history=training_history,
            checkpoint_path=str(best_checkpoint) if best_checkpoint else None
        )
        
        self.log_info(f"Customization complete. Improvement: {result.improvement:+.2f}%")
        
        return result
    
    def _prepare_data(self,
                      quality_result: QualityResult,
                      class_names: List[str],
                      training_config: TrainingConfig,
                      target_dataset_name: str) -> Tuple[Any, Any, Any]:
        """Prepare data loaders for training and evaluation.
        
        Test data loader is mandatory for proper model evaluation.
        Uses stratified k-fold cross-validation by default.
        """
        from torch.utils.data import DataLoader
        
        # Create full dataset
        dataset = LAIONCustomDataset(
            quality_result.filtered_samples,
            class_names,
            self.processor,
            use_augmentation=training_config.use_augmentation,
            augmentation_config={
                'num_ops': training_config.augmentation_strength,
                'magnitude': training_config.augmentation_magnitude
            },
            use_domain_adaptation=training_config.use_domain_adaptation,
            domain_adaptation_config={
                'blur_prob': training_config.domain_blur_probability,
                'blur_sigma': training_config.domain_blur_sigma_range,
                'jpeg_prob': training_config.domain_jpeg_probability,
                'jpeg_quality': training_config.domain_jpeg_quality_range,
                'downsample_prob': training_config.domain_downsample_probability,
                'target_size': training_config.domain_target_size,
                'downsample_scale': training_config.domain_downsample_scale_range
            },
            cache_dir=self.cache_base_dir,
            training_data_path=quality_result.source_path  # Use dataset-specific images folder
        )

        # Log augmentation and domain adaptation settings
        self.log_info("📦 Creating training dataset...")
        if training_config.use_augmentation:
            self.log_info(f"   Augmentation: RandAugment (num_ops={training_config.augmentation_strength}, magnitude={training_config.augmentation_magnitude})")
        else:
            self.log_info("   Augmentation: Disabled")

        if training_config.use_domain_adaptation:
            self.log_info("   Domain Adaptation: Enabled")
            self.log_info(f"      - Blur probability: {training_config.domain_blur_probability*100:.1f}%")
            self.log_info(f"      - JPEG probability: {training_config.domain_jpeg_probability*100:.1f}%")
            self.log_info(f"      - Downsample probability: {training_config.domain_downsample_probability*100:.1f}%")
            if training_config.domain_target_size:
                self.log_info(f"      - Target size: {training_config.domain_target_size}x{training_config.domain_target_size} (CIFAR-10/100 mode)")
            else:
                scale_min, scale_max = training_config.domain_downsample_scale_range
                self.log_info(f"      - Scale range: {scale_min:.2f}-{scale_max:.2f}")
        else:
            self.log_info("   Domain Adaptation: Disabled")

        # Prepare validation based on configuration
        if training_config.use_validation:
            if training_config.validation_method == "stratified_kfold":
                train_loader, val_loader = self._prepare_stratified_kfold_data(
                    dataset, training_config.validation_k_folds
                )
            else:  # simple_split
                train_loader, val_loader = self._prepare_simple_split_data(
                    dataset, training_config.validation_split
                )
        else:
            # No validation - use all data for training
            train_loader = DataLoader(
                dataset,
                batch_size=32,
                shuffle=True,
                num_workers=0,
                collate_fn=custom_collate_fn
            )
            val_loader = None
        
        # Create test loader (mandatory)
        test_loader = self._create_test_loader(target_dataset_name, class_names)
        
        if not test_loader:
            # Calculate expected path for helpful error message
            dataset_cache_dir = Path('./data')
            if hasattr(self, 'cache_base_dir') and self.cache_base_dir:
                dataset_cache_dir = Path(self.cache_base_dir) / target_dataset_name / 'datasets'
            expected_path = dataset_cache_dir / 'elevater' / target_dataset_name / 'test'

            raise ValueError(
                f"Failed to load test data for {target_dataset_name}.\n\n"
                f"Test data evaluation is mandatory for reliable model selection.\n\n"
                f"For file-based datasets (FER2013, PCAM, RESISC45, etc.), you need to:\n"
                f"  1. Download test data from official source\n"
                f"  2. Place in: {expected_path}/\n"
                f"  3. Organize as: {expected_path}/class_name/*.jpg\n\n"
                f"See FILE_BASED_DATASETS_GUIDE.md for download links and detailed instructions.\n\n"
                f"Check the log above for specific error details."
            )
        
        # Log data loader info
        if val_loader is not None:
            train_size = len(train_loader.dataset)
            val_size = len(val_loader.dataset)
            self.log_info(f"Created data loaders: {train_size} train, {val_size} validation, test from {target_dataset_name}")
        else:
            train_size = len(train_loader.dataset)
            self.log_info(f"Created data loaders: {train_size} train (no validation), test from {target_dataset_name}")
        
        return train_loader, val_loader, test_loader
    
    def _normalize_class_name(self, class_name) -> str:
        """
        Normalize class name for flexible matching.
        Converts to lowercase and replaces spaces with underscores.

        Handles both string and list-based class names (e.g., FER2013's ['happy', 'smiling']).

        Examples:
            'American Bulldog' -> 'american_bulldog'
            'american bulldog' -> 'american_bulldog'
            'Abyssinian' -> 'abyssinian'
            ['happy', 'smiling'] -> 'happy'  # Uses first element as canonical name
        """
        # Handle list-based class names (e.g., FER2013)
        if isinstance(class_name, list):
            class_name = class_name[0]
        return class_name.lower().replace(' ', '_')

    def _create_test_loader(self, target_dataset_name: str, class_names: List[str]) -> Optional[Any]:
        """Create test data loader from target dataset."""
        try:
            from ..datasets import get_dataset
            from torch.utils.data import DataLoader
            from pathlib import Path

            # Use configured cache directory for dataset downloads with dataset name
            dataset_cache_dir = Path('./data')  # Default fallback
            if hasattr(self, 'cache_base_dir') and self.cache_base_dir:
                dataset_cache_dir = Path(self.cache_base_dir) / target_dataset_name / 'datasets'

            # Load test split of the target dataset
            self.log_info(f"Loading test data from {target_dataset_name}")
            self.log_info(f"Test dataset root path: {dataset_cache_dir}")

            # Check if path exists for file-based datasets
            expected_test_path = dataset_cache_dir / 'elevater' / target_dataset_name / 'test'
            if expected_test_path.exists():
                self.log_info(f"✓ Test directory found: {expected_test_path}")
                # List class directories
                try:
                    class_dirs = [d.name for d in expected_test_path.iterdir() if d.is_dir()]
                    self.log_info(f"✓ Found {len(class_dirs)} class directories: {class_dirs[:5]}" + ("..." if len(class_dirs) > 5 else ""))
                except Exception as list_err:
                    self.log_warning(f"Could not list class directories: {list_err}")
            else:
                self.log_warning(f"✗ Test directory not found: {expected_test_path}")

            test_dataset_handler = get_dataset(target_dataset_name, train=False, root=str(dataset_cache_dir))  # Get test split

            # Create custom dataset for test data
            test_samples = []

            # Convert dataset to samples format
            if hasattr(test_dataset_handler, '_dataset') and test_dataset_handler._dataset:
                dataset = test_dataset_handler._dataset
            elif expected_test_path.exists():
                # Fallback for file-based datasets: load directly from filesystem
                self.log_info(f"File-based dataset detected. Loading from: {expected_test_path}")
                from torchvision.datasets.folder import ImageFolder

                class SimpleImageFolder(ImageFolder):
                    """Simple image folder without transforms for loading."""
                    pass

                dataset = SimpleImageFolder(root=str(expected_test_path), transform=None)
                self.log_info(f"Loaded {len(dataset)} samples from {len(dataset.classes)} classes")
            else:
                dataset = None

            if dataset:

                # CRITICAL: Use class_names from ELEVATER_DATASETS (passed as parameter)
                # NOT from test_dataset_handler which may have torchvision's dynamically generated names
                # This ensures evaluation uses EXACT REACT class names
                full_dataset_class_names = class_names  # Use parameter, not dataset handler

                # Extract canonical names from class_names (handles FER2013's list format)
                # For FER2013: ['angry'] -> 'angry', ['happy', 'smiling'] -> 'happy'
                canonical_class_names = [
                    name[0] if isinstance(name, list) else name
                    for name in class_names
                ]

                # Create mapping from training class names to indices
                # Use canonical names (not lists) as dictionary keys
                class_to_idx = {name: idx for idx, name in enumerate(canonical_class_names)}
                training_class_set = set(canonical_class_names)

                # Also create a normalized mapping for flexible matching
                # _normalize_class_name() now handles both strings and lists
                normalized_training_map = {self._normalize_class_name(name): name for name in canonical_class_names}

                self.log_info(f"Processing {len(dataset)} test samples from {target_dataset_name}")
                self.log_info(f"Training classes: {len(class_names)}, Full dataset classes: {len(full_dataset_class_names)}")

                # Use all test samples for complete evaluation
                from tqdm import tqdm

                # Build lookup from folder name -> ELEVATER canonical class name
                # This handles mixed-case folder names (e.g., 'Faces' vs 'faces')
                # where ImageFolder's alphabetical sort order differs from ELEVATER's order
                folder_to_canonical = {}
                for canonical in canonical_class_names:
                    normalized = self._normalize_class_name(canonical)
                    folder_to_canonical[canonical] = canonical          # exact match
                    folder_to_canonical[normalized] = canonical         # lowercase match
                    folder_to_canonical[canonical.lower()] = canonical  # lowercase match

                for idx in tqdm(range(len(dataset)), desc=f"Loading {target_dataset_name} test data"):
                    try:
                        image, label = dataset[idx]
                        if isinstance(label, int) and label < len(dataset.classes):

                            if isinstance(label, int) and 0 <= label < len(class_names):
                                test_class_name = class_names[label]          # robust to torchvision folder names
                            else:
                                # Get folder name from ImageFolder's own class list
                                folder_name = dataset.classes[label]# file-based ImageFolder fallback
                                
                                # Look up the ELEVATER canonical class name for this folder
                                # This handles case mismatches (e.g., folder 'Faces' matches ELEVATER 'Faces')
                                test_class_name = folder_to_canonical.get(folder_name) or \
                                                  folder_to_canonical.get(folder_name.lower())
                                if test_class_name is None:
                                    continue # Skip if folder name doesn't match any known class

                            # Check if this class exists in training data using normalized matching
                            normalized_test = self._normalize_class_name(test_class_name)
                            
                            if normalized_test not in normalized_training_map:
                                continue
                            
                            # Use the canonical test dataset class name for the label
                            # Note: Text field is not used during evaluation (templates are used in text classifier)
                            test_samples.append({'image': image, 
                                                 'label': test_class_name, 
                                                 'text': ''})

                    except Exception as e:
                        if idx < 10:  # Only log first few errors to avoid spam
                            self.log_warning(f"Error processing test sample {idx}: {e}")
                        continue
            
            if not test_samples:
                self.log_warning(f"No test samples loaded from {target_dataset_name}")
                return None

            # Log which classes are missing from training data
            # Calculate missing classes using normalized matching
            # Use canonical_class_names to avoid unhashable list types
            normalized_full_classes = {self._normalize_class_name(name): name for name in canonical_class_names}
            missing_normalized = set(normalized_full_classes.keys()) - set(normalized_training_map.keys())
            missing_classes = [normalized_full_classes[norm] for norm in missing_normalized]

            if missing_classes:
                self.log_info(f"Excluding {len(missing_classes)} classes not in training data: {sorted(missing_classes)[:10]}" +
                             ("..." if len(missing_classes) > 10 else ""))

            # Create test dataset using test dataset class names (from ELEVATER_DATASETS)
            # This ensures evaluation uses the correct class names for CLIP embeddings
            test_dataset = TestDataset(test_samples, full_dataset_class_names, self.processor)
            
            # Create test loader
            test_loader = DataLoader(
                test_dataset,
                batch_size=32,
                shuffle=False,
                num_workers=0,
                collate_fn=custom_collate_fn
            )
            
            self.log_info(f"Created test loader with {len(test_samples)} samples (complete test set)")
            return test_loader
            
        except Exception as e:
            self.log_warning(f"Failed to create test loader for {target_dataset_name}: {e}")

            # Provide helpful instructions for manual dataset setup
            expected_path = dataset_cache_dir / 'elevater' / target_dataset_name / 'test'
            self.log_info(f"\n📁 Test data setup instructions for {target_dataset_name}:")
            self.log_info(f"   Expected test data location: {expected_path}")
            self.log_info(f"   Required structure:")
            self.log_info(f"      {expected_path}/")
            self.log_info(f"      ├── class_1/")
            self.log_info(f"      │   ├── image_001.jpg")
            self.log_info(f"      │   └── ...")
            self.log_info(f"      ├── class_2/")
            self.log_info(f"      │   └── ...")
            self.log_info(f"      └── ...")
            self.log_info(f"\n   See FILE_BASED_DATASETS_GUIDE.md for download links and setup instructions")

            return None
    
    def _prepare_stratified_kfold_data(self, dataset, k_folds=5):
        """Prepare data using stratified k-fold cross-validation."""
        from sklearn.model_selection import StratifiedKFold
        from torch.utils.data import DataLoader, Subset
        import numpy as np
        
        # Labels for stratification WITHOUT loading/augmenting images.
        labels = np.array([
            dataset.normalized_to_idx.get(dataset._normalize_label(s['label']), 0)
            for s in dataset.valid_samples
        ])
        indices = np.arange(len(dataset))

        # Use stratified k-fold
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        # Get the first fold (we'll use just one fold for validation)
        train_idx, val_idx = next(iter(skf.split(indices, labels)))
        
        # Create datasets
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        
    
        # Separate val dataset: augmentation OFF, same image folder, no re-validation.
        from pathlib import Path as _Path
        _val_samples = [dataset.valid_samples[i] for i in val_idx]
        _val_img_root = str(_Path(dataset.image_cache_dir).parent)  # -> <dir>/images
        val_dataset = LAIONCustomDataset(
            _val_samples, dataset.class_names, dataset.processor,
            use_augmentation=False, training_data_path=_val_img_root, skip_validation=True,
        )
        val_dataset.valid_samples = _val_samples  # already validated upstream


        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,
            collate_fn=custom_collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            collate_fn=custom_collate_fn
        )
        
        self.log_info(f"Using stratified {k_folds}-fold validation: {len(train_idx)} train, {len(val_idx)} validation samples")
        return train_loader, val_loader
    
    def _prepare_simple_split_data(self, dataset, validation_split=0.2):
        """Prepare data using simple random split."""
        from torch.utils.data import DataLoader, random_split
        import torch
        
        # Split dataset
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size

        from torch.utils.data import Subset
        from pathlib import Path as _Path
        import numpy as _np
        perm = _np.random.default_rng(42).permutation(len(dataset))
        val_idx, train_idx = perm[:val_size], perm[val_size:]
        train_dataset = Subset(dataset, train_idx)
        _val_samples = [dataset.valid_samples[i] for i in val_idx]
        _val_img_root = str(_Path(dataset.image_cache_dir).parent)
        val_dataset = LAIONCustomDataset(
            _val_samples, dataset.class_names, dataset.processor,
            use_augmentation=False, training_data_path=_val_img_root, skip_validation=True,
        )
        val_dataset.valid_samples = _val_samples

        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,
            collate_fn=custom_collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            collate_fn=custom_collate_fn
        )
        
        self.log_info(f"Using simple split validation: {train_size} train, {val_size} validation samples")
        return train_loader, val_loader
    
    def _get_dataset_templates(self, target_dataset_name: str) -> List[str]:
        """
        Get evaluation templates for a dataset.

        Args:
            target_dataset_name: Name of the target dataset

        Returns:
            List of prompt templates for evaluation
        """
        try:
            from ..datasets import get_dataset
            # Get dataset instance to access templates
            cache_root = str(self.cache_base_dir) if self.cache_base_dir else None
            dataset = get_dataset(target_dataset_name, train=False, root=cache_root)
            if hasattr(dataset, 'get_text_templates'):
                templates = dataset.get_text_templates()
                self.log_info(f"Using {len(templates)} templates for {target_dataset_name} evaluation")
                return templates
        except Exception as e:
            self.log_warning(f"Could not load dataset templates: {e}")

        # Fallback to default template
        return None

    def _evaluate_baseline(self, val_loader: Any, class_names: List[str], templates: List[str] = None) -> Tuple[float, Dict[str, float]]:
        """Evaluate baseline model performance."""
        baseline_model = CustomizedCLIP(
            self.model,
            self.processor,
            regularize=False
        ).to(self.device)

        accuracy, class_accuracies = self.evaluator.evaluate_detailed(
            baseline_model,
            val_loader,
            class_names,
            templates=templates
        )

        self.log_info(f"Baseline model accuracy: {accuracy:.2f}%")
        return accuracy, class_accuracies
    
    def load_checkpoint(self, checkpoint_path: str) -> 'CustomizedCLIP':
        """
        Load a saved model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Loaded model
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create model
        model = CustomizedCLIP(
            self.model,
            self.processor,
            regularize=False
        ).to(self.device)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        self.log_info(f"Loaded checkpoint from {checkpoint_path}")

        return model

    def _save_augmented_grids(self, dataset, dataset_name, training_config):
        """Save 10x10 grid visualizations of augmented/domain-adapted training samples."""
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        self.log_info("📸 Saving augmented sample grids for visual inspection...")

        # Create output directory
        output_dir = Path(self.checkpoint_dir).parent / 'augmented_grids'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Sample 100 random indices
        num_samples = min(100, len(dataset))
        indices = random.sample(range(len(dataset)), num_samples)

        # Create figure with 10x10 grid
        fig = plt.figure(figsize=(30, 30))
        gs = gridspec.GridSpec(10, 10, figure=fig, hspace=0.3, wspace=0.3)

        for idx, sample_idx in enumerate(indices):
            try:
                # Get augmented/domain-adapted image from dataset
                image, text, label = dataset[sample_idx]

                # Convert CLIP processor output to displayable format
                if hasattr(image, 'numpy'):
                    # If it's a tensor, convert to numpy
                    img_array = image.permute(1, 2, 0).numpy()
                    # Denormalize if needed (CLIP uses mean=[0.48145466, 0.45782750, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
                    img_array = img_array * np.array([0.26862954, 0.26130258, 0.27577711]) + np.array([0.48145466, 0.45782750, 0.40821073])
                    img_array = np.clip(img_array, 0, 1)
                else:
                    img_array = np.array(image) / 255.0 if np.array(image).max() > 1 else np.array(image)

                # Create subplot
                ax = fig.add_subplot(gs[idx // 10, idx % 10])
                ax.imshow(img_array)
                ax.axis('off')

                # Add label
                if hasattr(dataset, 'class_names'):
                    class_name = dataset.class_names[label]
                    # Handle FER2013-style list class names (use first element for display)
                    if isinstance(class_name, list):
                        class_name = class_name[0]
                else:
                    class_name = f'Class {label}'
                ax.set_title(f'{class_name}\n#{sample_idx}', fontsize=8)

            except Exception as e:
                self.log_warning(f"Failed to process sample {sample_idx}: {e}")
                continue

        # Save figure
        output_file = output_dir / f'{dataset_name}_augmented_grid.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)

        self.log_info(f"✅ Saved augmented samples grid to: {output_file}")

        # Print summary of what transforms are shown
        if training_config.use_augmentation or training_config.use_domain_adaptation:
            self.log_info("   Grid shows effects of:")
            if training_config.use_augmentation:
                self.log_info(f"   - RandAugment (ops={training_config.augmentation_strength}, mag={training_config.augmentation_magnitude})")
            if training_config.use_domain_adaptation:
                self.log_info("   - Domain Adaptation (blur/JPEG/downsample)")


class CustomizedCLIP(nn.Module):
    """
    Customized CLIP model with locked text encoder.
    
    This model implements the locked-text tuning approach where only
    the vision encoder is fine-tuned while keeping the text encoder frozen.
    """
    
    def __init__(self, 
                 base_model: CLIPModel,
                 processor: CLIPProcessor,
                 regularize: bool = True):
        """
        Initialize customized CLIP model.
        
        Args:
            base_model: Base CLIP model
            processor: CLIP processor
            regularize: Whether to use regularization
        """
        super().__init__()
        
        self.clip_model = base_model
        self.processor = processor
        self.device = next(base_model.parameters()).device
        self.regularize = regularize
        
        # Lock text encoder
        for param in self.clip_model.text_model.parameters():
            param.requires_grad = False
        
        # Store original vision weights for regularization
        if regularize:
            self.original_vision_state = {}
            with torch.no_grad():
                for name, param in self.clip_model.vision_model.named_parameters():
                    self.original_vision_state[name] = param.detach().clone()
    
    def forward(self, images: List, text_features: Optional[torch.Tensor] = None):
        """
        Forward pass through the model.

        Args:
            images: List of PIL images or preprocessed tensors
            text_features: Pre-computed text features (optional)

        Returns:
            Logits or image features
        """

        inputs = self._safe_process_images(self.processor, images, self.device)
        image_embeds_output = self.clip_model.get_image_features(**inputs)

        # Handle both tensor and BaseModelOutputWithPooling formats
        if isinstance(image_embeds_output, torch.Tensor):
            image_embeds = image_embeds_output
        else:
            # Extract pooler_output from BaseModelOutputWithPooling
            image_embeds = image_embeds_output.pooler_output if hasattr(image_embeds_output, 'pooler_output') else image_embeds_output[0]

        # Normalize
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        # Debug: Print embedding stats
        if not hasattr(self, '_forward_debug_printed'):
            print(f"\nDEBUG Forward pass:")
            print(f"  Image embeddings shape: {image_embeds.shape}")
            print(f"  Image embeddings mean: {image_embeds.mean().item():.6f}")
            print(f"  Image embeddings std: {image_embeds.std().item():.6f}")
            print(f"  Image embeddings min/max: {image_embeds.min().item():.6f} / {image_embeds.max().item():.6f}")
            if text_features is not None:
                print(f"  Text features shape: {text_features.shape}")
                print(f"  Text features mean: {text_features.mean().item():.6f}")
                print(f"  Text features std: {text_features.std().item():.6f}")
            self._forward_debug_printed = True

        if text_features is None:
            return image_embeds

        # Compute logits with fixed scale (standard CLIP evaluation protocol)
        # Using fixed scale of 100.0 matches the CLIP paper and REACT benchmark
        # This ensures reproducible results and matches zero-shot evaluation practices
        logits = 100.0 * (image_embeds @ text_features.T)

        # Debug: Print logit stats
        if not hasattr(self, '_logit_debug_printed'):
            print(f"  Logits shape: {logits.shape}")
            print(f"  Logits mean: {logits.mean().item():.6f}")
            print(f"  Logits std: {logits.std().item():.6f}")
            print(f"  Logits min/max: {logits.min().item():.6f} / {logits.max().item():.6f}")
            print(f"  Top-1 predictions (first 5): {logits.argmax(dim=1)[:5].tolist()}")
            print(f"  Top-1 confidence (first 5): {logits.max(dim=1)[0][:5].tolist()}\n")
            self._logit_debug_printed = True

        return logits
    
    def _safe_process_images(self, processor, images, device):
        """
        Safely process images for CLIP model, handling different formats and ensuring channel compatibility.
        This is identical to the safe_process_images method from the original working code.
        
        Args:
            processor: The CLIP processor
            images: List of PIL images or tensor
            device: The device to put the processed images on

        Returns:
            Dictionary with properly processed image tensors
        """
        # Step 1: Ensure all images are RGB mode PIL images
        preprocessed_images = []
        valid_images = True

        for i, img in enumerate(images):
            try:
                if isinstance(img, Image.Image):
                    # Ensure image is RGB
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    # Test image integrity
                    img.load()
                    # Replace degenerate images (too small for processor) with placeholder
                    if img.width < 10 or img.height < 10:
                        img = Image.new('RGB', (224, 224), color=(128, 128, 128))
                    preprocessed_images.append(img)
                else:
                    # If it's already a tensor or array, we'll handle it differently
                    preprocessed_images.append(img)
            except Exception as e:
                print(f"Image at index {i} is corrupted, replacing with placeholder: {str(e)}")
                # Replace corrupted image with placeholder instead of failing
                placeholder = Image.new('RGB', (224, 224), color=(128, 128, 128))
                preprocessed_images.append(placeholder)
                valid_images = False

        # Step 2: Try to process with appropriate parameters
        try:
            # For PIL images, use the processor with DEFAULT parameters
            # This ensures correct CLIP preprocessing (resize shortest edge, then center crop)
            if isinstance(preprocessed_images[0], Image.Image):
                # Debug: Check input image sizes
                if hasattr(self, '_debug_counter'):
                    self._debug_counter += 1
                else:
                    self._debug_counter = 1

                if self._debug_counter == 1:  # Only print for first batch
                    print(f"DEBUG: Processing batch of {len(preprocessed_images)} PIL images")
                    print(f"DEBUG: First image size: {preprocessed_images[0].size}")
                    print(f"DEBUG: First image mode: {preprocessed_images[0].mode}")

                inputs = processor(
                    images=preprocessed_images,
                    return_tensors="pt"
                )

                if self._debug_counter == 1:  # Only print for first batch
                    print(f"DEBUG: Output tensor shape: {inputs['pixel_values'].shape}")
                    print(f"DEBUG: Using DEFAULT processor parameters (no explicit size/crop)")

                inputs = {k: v.to(device) for k, v in inputs.items()}
                return inputs
            else:
                # For tensors, just move to device
                return {"pixel_values": torch.stack(preprocessed_images).to(device)}
        except Exception as e:
            print(f"Error in standard processing: {str(e)}")

            # Step 3: Manual fallback processing if the processor fails
            try:
                print("Attempting manual image processing...")

                # Convert PIL images to normalized tensors manually
                tensor_images = []
                for img in preprocessed_images:
                    if isinstance(img, Image.Image):
                        # Resize if needed
                        if img.size != (224, 224):
                            img = img.resize((224, 224), Image.LANCZOS)

                        # Convert to numpy array
                        img_array = np.array(img).astype(np.float32) / 255.0

                        # Normalize with CLIP's normalization values
                        mean = [0.48145466, 0.4578275, 0.40821073]
                        std = [0.26862954, 0.26130258, 0.27577711]

                        # Apply normalization (adjust for RGB channels)
                        img_array = (img_array - np.array(mean)) / np.array(std)

                        # Convert to tensor and add batch dimension
                        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # HWC -> CHW
                        tensor_images.append(img_tensor)
                    else:
                        # If already a tensor
                        tensor_images.append(img)

                # Stack into a batch
                batch_tensor = torch.stack(tensor_images).to(device)
                return {"pixel_values": batch_tensor}

            except Exception as e2:
                # Log the detailed error and raise
                print(f"Manual processing also failed: {str(e2)}")
                raise ValueError(f"Failed to process images: original error '{str(e)}', fallback error '{str(e2)}'")
    
    def get_regularization_loss(self) -> torch.Tensor:
        """Calculate regularization loss to prevent catastrophic forgetting."""
        if not self.regularize:
            return torch.tensor(0.0, device=self.device)
        
        reg_loss = 0.0
        for name, param in self.clip_model.vision_model.named_parameters():
            if name in self.original_vision_state:
                orig_param = self.original_vision_state[name].to(param.device)
                reg_loss += nn.functional.mse_loss(param, orig_param)
        
        return reg_loss
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text to features."""
        tokens = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            text_features_output = self.clip_model.get_text_features(**tokens)

            # Handle both tensor and BaseModelOutputWithPooling formats
            if isinstance(text_features_output, torch.Tensor):
                text_embeds = text_features_output
            else:
                # Extract pooler_output from BaseModelOutputWithPooling
                text_embeds = text_features_output.pooler_output if hasattr(text_features_output, 'pooler_output') else text_features_output[0]

            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        return text_embeds


class TestDataset(torch.utils.data.Dataset):
    """Dataset for test data evaluation."""

    def __init__(self, test_samples: List[Dict], class_names: List[str], processor):
        """
        Initialize test dataset.

        Args:
            test_samples: List of test sample dictionaries
            class_names: List of class names (REACT's exact mixed-case format from ELEVATER_DATASETS)
            processor: CLIP processor
        """
        self.test_samples = test_samples
        self.class_names = class_names
        # Direct mapping for REACT class names (test data already has correct names from ELEVATER_DATASETS)
        # Handle FER2013-style list class names (use first element as canonical name)
        self.class_to_idx = {(name[0] if isinstance(name, list) else name): i for i, name in enumerate(class_names)}
        self.processor = processor

    def __len__(self):
        return len(self.test_samples)

    def __getitem__(self, idx):
        sample = self.test_samples[idx]

        # Get image (already PIL Image from dataset)
        image = sample['image']

        # Get label - test samples already have REACT class names from ELEVATER_DATASETS
        # (assigned in _create_test_loader using class_names parameter from ELEVATER_DATASETS)
        label = self.class_to_idx.get(sample['label'], 0)

        # Get text
        text = sample.get('text', f"a photo of a {sample['label']}.")

        return image, text, label


class LAIONCustomDataset(torch.utils.data.Dataset):
    """Dataset for training with LAION-style image-text pairs."""
    
    def __init__(self,
                 samples: List[Dict],
                 class_names: List[str],
                 processor: CLIPProcessor,
                 use_augmentation: bool = True,
                 augmentation_config: Optional[Dict] = None,
                 use_domain_adaptation: bool = False,
                 domain_adaptation_config: Optional[Dict] = None,
                 cache_dir: Optional[str] = None,
                 training_data_path: Optional[str] = None,
                 skip_validation: bool = False):
        """
        Initialize dataset.

        Args:
            samples: List of sample dictionaries
            class_names: List of class names
            processor: CLIP processor
            use_augmentation: Whether to use data augmentation
            augmentation_config: Augmentation configuration
            use_domain_adaptation: Whether to use domain adaptation
            domain_adaptation_config: Domain adaptation configuration
            cache_dir: Directory for caching images (global cache)
            training_data_path: Path to training data JSON (for dataset-specific cache)
            skip_validation: Skip automatic validation (caller will validate manually)
        """
        self.samples = samples
        self.class_names = class_names
        # Create case-insensitive mapping: normalized_name -> index
        # This handles training JSON having lowercase/normalized labels while evaluation uses REACT's mixed-case format
        # Handle FER2013-style list class names (use first element as canonical name)
        self.class_to_idx = {(name[0] if isinstance(name, list) else name): i for i, name in enumerate(class_names)}
        self.normalized_to_idx = {self._normalize_label(name): i for i, name in enumerate(class_names)}
        self.processor = processor

        # Setup augmentation configuration
        self.use_augmentation = use_augmentation
        self.augmentation_config = augmentation_config or {}

        # Setup domain adaptation configuration
        self.use_domain_adaptation = use_domain_adaptation
        self.domain_adaptation_config = domain_adaptation_config or {}

        # Setup image caching - prioritize dataset-specific images folder
        if training_data_path:
            # Use dataset-specific images folder (e.g., .../cifar10/images/)
            from pathlib import Path
            dataset_dir = Path(training_data_path).parent if os.path.isfile(training_data_path) else Path(training_data_path)
            self.image_cache_dir = str(dataset_dir / 'images')
            print(f"Using dataset-specific image cache: {self.image_cache_dir}")
        elif cache_dir:
            # Fallback to global cache
            self.cache_dir = cache_dir
            self.image_cache_dir = os.path.join(self.cache_dir, 'image_cache')
            print(f"Using global image cache: {self.image_cache_dir}")
        else:
            # Default fallback
            self.cache_dir = './cache'
            self.image_cache_dir = os.path.join(self.cache_dir, 'image_cache')
            print(f"Using default image cache: {self.image_cache_dir}")

        os.makedirs(self.image_cache_dir, exist_ok=True)

        # Pre-filter samples to only include those with cached images or that can be downloaded
        # This prevents training on placeholder images which can hurt performance
        self.valid_samples = []
        if not skip_validation:
            self._filter_valid_samples()
    
    def __len__(self):
        return len(self.valid_samples)

    def _normalize_label(self, label) -> str:
        """
        Normalize label for case-insensitive matching.
        Converts to lowercase and replaces spaces/hyphens with underscores.

        Handles both string labels and list-based labels (FER2013 style).

        Examples:
            'American Bulldog' -> 'american_bulldog'
            'american bulldog' -> 'american_bulldog'
            'Abyssinian' -> 'abyssinian'
            ['happy', 'smiling'] -> 'happy'  (uses first element)
        """
        # Handle FER2013-style list class names
        if isinstance(label, list):
            label = label[0]
        return label.lower().replace(' ', '_').replace('-', '_')

    def _filter_valid_samples(self):
        """Filter samples to only include those with cached images or that can be downloaded.
        This prevents training on placeholder images which can hurt performance."""
        from PIL import Image
        import requests
        from io import BytesIO
        from tqdm import tqdm
        
        print(f"Filtering {len(self.samples)} samples to find valid images...")

        for sample in tqdm(self.samples, desc="Validating samples"):
            url = sample.get('url')
            if not url:
                continue

            # Create cache path using hierarchical structure
            url_hash = hashlib.md5(url.encode()).hexdigest()

            # Check hierarchical structure first (new format: image_cache/ae/img_aeb88f14....jpg)
            subdir = url_hash[:2]
            cache_subdir = os.path.join(self.image_cache_dir, subdir)
            cache_path_hierarchical = os.path.join(cache_subdir, f"img_{url_hash}.jpg")

            # Check flat structure for backward compatibility
            cache_path_flat = os.path.join(self.image_cache_dir, f"img_{url_hash}.jpg")

            # Check if image is already cached (try hierarchical first, then flat)
            if os.path.exists(cache_path_hierarchical):
                self.valid_samples.append(sample)
                continue
            elif os.path.exists(cache_path_flat):
                self.valid_samples.append(sample)
                continue

            # Try to download and cache the image
            try:
                response = requests.get(url, timeout=(2,5))
                response.raise_for_status()

                # Test if image can be loaded
                image = Image.open(BytesIO(response.content)).convert('RGB')

                # Save to cache using hierarchical structure
                try:
                    os.makedirs(cache_subdir, exist_ok=True)
                    image.save(cache_path_hierarchical, 'JPEG', quality=95)
                except Exception:
                    pass  # Cache save failed, but image is valid

                # Image is valid, add to valid samples
                self.valid_samples.append(sample)

            except Exception:
                # Skip invalid samples
                continue
        
        print(f"Filtered dataset: {len(self.valid_samples)}/{len(self.samples)} valid samples ({len(self.valid_samples)/len(self.samples)*100:.1f}%)")
    
    def _create_placeholder_image(self):
        """Create a placeholder image when download fails"""
        from PIL import Image
        return Image.new('RGB', (224, 224), color=(128, 128, 128))

    def _safe_get_image(self, url):
        """Load image from cache (simplified since we pre-filtered valid images)"""
        from PIL import Image

        if not url:
            return self._create_placeholder_image()

        # Create a hash of the URL for caching
        url_hash = hashlib.md5(url.encode()).hexdigest()

        # Try hierarchical structure first (new format)
        subdir = url_hash[:2]
        cache_path_hierarchical = os.path.join(self.image_cache_dir, subdir, f"img_{url_hash}.jpg")

        # Try flat structure for backward compatibility
        cache_path_flat = os.path.join(self.image_cache_dir, f"img_{url_hash}.jpg")

        # Check hierarchical first, then flat
        for cache_path in [cache_path_hierarchical, cache_path_flat]:
            if os.path.exists(cache_path):
                try:
                    image = Image.open(cache_path)
                    image.load()
                    return image.convert('RGB')
                except Exception:
                    continue

        # Fallback - this should rarely happen since we pre-filtered
        return self._create_placeholder_image()
    
    def _apply_domain_adaptation(self, image):
        """
        Apply domain adaptation transforms to simulate test data characteristics.
        Applied AFTER RandAugment to ensure domain match.

        Transforms include:
        - Gaussian blur (simulates low quality/pixelation)
        - JPEG compression (adds compression artifacts)
        - Downsample/upsample (simulates resolution degradation)
        """
        config = self.domain_adaptation_config

        # 1. Gaussian Blur (simulates low quality/pixelation)
        blur_prob = config.get('blur_prob', 0.3)
        if random.random() < blur_prob:
            blur_sigma_range = config.get('blur_sigma', [0.1, 2.0])
            sigma = random.uniform(blur_sigma_range[0], blur_sigma_range[1])
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))

        # 2. JPEG Compression (adds compression artifacts)
        jpeg_prob = config.get('jpeg_prob', 0.3)
        if random.random() < jpeg_prob:
            jpeg_quality_range = config.get('jpeg_quality', [30, 95])
            quality = random.randint(jpeg_quality_range[0], jpeg_quality_range[1])
            # Simulate JPEG compression by saving and loading from buffer
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            image = Image.open(buffer).convert('RGB')

        # 3. Downsample/Upsample (simulates resolution degradation)
        downsample_prob = config.get('downsample_prob', 0.3)
        if random.random() < downsample_prob:
            original_size = image.size

            # Check if we have a target size (e.g., 32 for CIFAR-10/100, 28 for MNIST)
            target_size = config.get('target_size', None)

            if target_size is not None:
                # Downscale to target size then upscale back to original
                image = image.resize((target_size, target_size), Image.BILINEAR)
                image = image.resize(original_size, Image.BILINEAR)
            else:
                # Use scale range for continuous downsampling
                scale_range = config.get('downsample_scale', [0.5, 0.9])
                scale = random.uniform(scale_range[0], scale_range[1])
                small_size = (int(original_size[0] * scale), int(original_size[1] * scale))
                # Downsample then upsample back
                image = image.resize(small_size, Image.BILINEAR)
                image = image.resize(original_size, Image.BILINEAR)

        return image

    def _apply_transforms(self, image):
        """Apply appropriate transforms based on augmentation and domain adaptation settings"""
        if self.use_augmentation:
            try:
                from torchvision import transforms

                # Convert PIL to tensor for RandAugment, then back to PIL
                tensor_image = transforms.PILToTensor()(image)
                # Apply RandAugment (note: RandAugment expects tensor input)
                augmented_tensor = transforms.RandAugment(
                    num_ops=self.augmentation_config.get('num_ops', 2),
                    magnitude=self.augmentation_config.get('magnitude', 9),
                    interpolation=transforms.InterpolationMode.BILINEAR
                )(tensor_image)
                # Convert back to PIL Image
                augmented_image = transforms.ToPILImage()(augmented_tensor)

                # Apply domain adaptation AFTER RandAugment (if enabled)
                if self.use_domain_adaptation:
                    try:
                        augmented_image = self._apply_domain_adaptation(augmented_image)
                    except Exception as e:
                        self.log_warning(f"Domain adaptation failed: {str(e)}, using non-adapted image")

                return augmented_image
            except Exception as e:
                print(f"Error applying augmentation, using original image: {str(e)}")
                # Fallback to basic resize if augmentation fails
                return image.resize((224, 224)) if image.size != (224, 224) else image
        else:
            # Apply domain adaptation even without RandAugment (if enabled)
            if self.use_domain_adaptation:
                try:
                    image = self._apply_domain_adaptation(image)
                except Exception as e:
                    self.log_warning(f"Domain adaptation failed: {str(e)}, using non-adapted image")
            # Resize to 224x224
            return image.resize((224, 224)) if image.size != (224, 224) else image
    
    def __getitem__(self, idx):
        sample = self.valid_samples[idx]  # Use pre-filtered valid samples

        # Get image using cached approach - since we pre-filtered, this should always work
        url = sample.get('url')
        image = self._safe_get_image(url)

        # Apply transforms (augmentation or just resize)
        image = self._apply_transforms(image)

        # Get label with case-insensitive matching
        sample_label = sample['label']
        normalized_label = self._normalize_label(sample_label)
        label = self.normalized_to_idx.get(normalized_label, 0)

        return image, sample.get('text', ''), label


def custom_collate_fn(batch):
    """Custom collate function for handling PIL images."""
    images = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    labels = torch.tensor([item[2] for item in batch])
    
    return images, texts, labels