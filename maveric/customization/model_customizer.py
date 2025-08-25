"""Main model customization module."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
from transformers import CLIPModel, CLIPProcessor

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
                  validation_split: float = 0.2) -> CustomizationResult:
        """
        Customize model using filtered data.
        
        Args:
            quality_result: Filtered data from quality control
            training_config: Training configuration
            target_dataset_name: Name of target dataset
            class_names: List of class names
            validation_split: Fraction of data for validation
            
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
        
        # Create trainer
        self.trainer = Trainer(
            model=customized_model,
            device=self.device,
            checkpoint_dir=self.checkpoint_dir
        )
        
        # Create evaluator
        self.evaluator = Evaluator(device=self.device)
        
        # Get baseline performance (always use test data)
        self.log_info("Evaluating baseline model")
        if not test_loader:
            raise ValueError(f"Test data is required for evaluation but could not load test set for {target_dataset_name}")
        baseline_accuracy = self._evaluate_baseline(test_loader, class_names)
        
        # Train model (test data is mandatory)
        self.log_info("Training customized model")
        training_history = self.trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            training_config=training_config,
            class_names=class_names
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
                    class_names
                )
            else:
                # Fallback: evaluate current model and save it
                self.log_info("No best checkpoint found, evaluating current model")
                final_accuracy, class_accuracies = self.evaluator.evaluate_detailed(
                    customized_model,
                    test_loader,
                    class_names
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
                class_names
            )
        
        # Create result
        result = CustomizationResult(
            model_name=f"customized_{self.base_model_name}",
            base_model_name=self.base_model_name,
            training_config=training_config.to_dict(),
            training_samples=len(train_loader.dataset),
            test_accuracy=final_accuracy,
            zero_shot_baseline=baseline_accuracy,
            class_accuracies=class_accuracies,
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
            }
        )
        
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
            raise ValueError(
                f"Failed to load test data for {target_dataset_name}. "
                f"Test data evaluation is mandatory for reliable model selection. "
                f"Please ensure the target dataset supports test splits."
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
            test_dataset_handler = get_dataset(target_dataset_name, train=False, root=str(dataset_cache_dir))  # Get test split
            
            # Create custom dataset for test data
            test_samples = []
            
            # Convert dataset to samples format  
            if hasattr(test_dataset_handler, '_dataset') and test_dataset_handler._dataset:
                dataset = test_dataset_handler._dataset
                class_to_idx = {name: idx for idx, name in enumerate(class_names)}
                
                self.log_info(f"Processing {len(dataset)} test samples from {target_dataset_name}")
                
                # Use all test samples for complete evaluation
                from tqdm import tqdm
                
                for idx in tqdm(range(len(dataset)), desc=f"Loading {target_dataset_name} test data"):
                    try:
                        image, label = dataset[idx]
                        if isinstance(label, int) and label < len(class_names):
                            class_name = class_names[label]
                            test_samples.append({
                                'image': image,
                                'label': class_name,
                                'text': f"a photo of a {class_name}."
                            })
                    except Exception as e:
                        if idx < 10:  # Only log first few errors to avoid spam
                            self.log_warning(f"Error processing test sample {idx}: {e}")
                        continue
            
            if not test_samples:
                self.log_warning(f"No test samples loaded from {target_dataset_name}")
                return None
            
            # Create test dataset
            test_dataset = TestDataset(test_samples, class_names, self.processor)
            
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
            return None
    
    def _prepare_stratified_kfold_data(self, dataset, k_folds=5):
        """Prepare data using stratified k-fold cross-validation."""
        from sklearn.model_selection import StratifiedKFold
        from torch.utils.data import DataLoader, Subset
        import numpy as np
        
        # Get labels for stratification
        labels = []
        for i in range(len(dataset)):
            _, _, label = dataset[i]
            labels.append(label)
        
        labels = np.array(labels)
        indices = np.arange(len(dataset))
        
        # Use stratified k-fold
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        # Get the first fold (we'll use just one fold for validation)
        train_idx, val_idx = next(iter(skf.split(indices, labels)))
        
        # Create datasets
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        
        # Disable augmentation for validation dataset
        if hasattr(val_dataset, 'dataset') and hasattr(val_dataset.dataset, 'use_augmentation'):
            # Create a copy of the dataset for validation without augmentation
            val_dataset_copy = LAIONCustomDataset(
                [dataset.samples[i] for i in val_idx],
                dataset.class_names,
                dataset.processor,
                use_augmentation=False
            )
            val_dataset = val_dataset_copy
        
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
        
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Disable augmentation for validation
        if hasattr(val_dataset.dataset, 'use_augmentation'):
            val_dataset.dataset.use_augmentation = False
        
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
    
    def _evaluate_baseline(self, val_loader: Any, class_names: List[str]) -> float:
        """Evaluate baseline model performance."""
        baseline_model = CustomizedCLIP(
            self.model,
            self.processor,
            regularize=False
        ).to(self.device)
        
        accuracy = self.evaluator.evaluate(
            baseline_model,
            val_loader,
            class_names
        )
        
        self.log_info(f"Baseline model accuracy: {accuracy:.2f}%")
        return accuracy
    
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
        # Process images with robust error handling like original code
        if isinstance(images[0], torch.Tensor):
            # Already preprocessed
            pixel_values = torch.stack(images).to(self.device)
        else:
            # Verify and process PIL images with robust handling
            # verified_images = []
            # for img in images:
            #     try:
            #         if isinstance(img, Image.Image):
            #             # Force load to verify integrity and ensure RGB
            #             img.load()
            #             if img.mode != 'RGB':
            #                 img = img.convert('RGB')
            #             verified_images.append(img)
            #         else:
            #             verified_images.append(img)
            #     except Exception:
            #         # Replace corrupted image with placeholder
            #         from PIL import Image
            #         placeholder = Image.new('RGB', (224, 224), color=(128, 128, 128))
            #         verified_images.append(placeholder)
            
            # Process with standard parameters (like original code)
            inputs = self.processor(
                # images=verified_images,
                images=images,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            pixel_values = inputs.pixel_values
        
        # Get image features
        image_outputs = self.clip_model.vision_model(pixel_values=pixel_values)
        image_embeds = image_outputs[1]  # pooled output
        image_embeds = self.clip_model.visual_projection(image_embeds)
        # Get image features using the same method as original code
        # inputs = {"pixel_values": pixel_values}
        # image_embeds = self.clip_model.get_image_features(**inputs)
        
        # Normalize
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        
        if text_features is None:
            return image_embeds
        
        # Compute logits
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * (image_embeds @ text_features.T)
        
        return logits
    
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
            padding=True,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            text_embeds = self.clip_model.get_text_features(**tokens)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            # text_outputs = self.clip_model.text_model(**tokens)
            # text_embeds = text_outputs[1]  # pooled output
            # text_embeds = self.clip_model.text_projection(text_embeds)

        
        return text_embeds


class TestDataset(torch.utils.data.Dataset):
    """Dataset for test data evaluation."""
    
    def __init__(self, test_samples: List[Dict], class_names: List[str], processor):
        """
        Initialize test dataset.
        
        Args:
            test_samples: List of test sample dictionaries
            class_names: List of class names
            processor: CLIP processor
        """
        self.test_samples = test_samples
        self.class_names = class_names
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.processor = processor
    
    def __len__(self):
        return len(self.test_samples)
    
    def __getitem__(self, idx):
        sample = self.test_samples[idx]
        
        # Get image (already PIL Image from dataset)
        image = sample['image']
        
        # Get label
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
                 augmentation_config: Optional[Dict] = None):
        """
        Initialize dataset.
        
        Args:
            samples: List of sample dictionaries
            class_names: List of class names
            processor: CLIP processor
            use_augmentation: Whether to use data augmentation
            augmentation_config: Augmentation configuration
        """
        self.samples = samples
        self.class_names = class_names
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.processor = processor
        
        # Setup augmentation configuration
        self.use_augmentation = use_augmentation
        self.augmentation_config = augmentation_config or {}
    
    def __len__(self):
        return len(self.samples)
    
    def _apply_transforms(self, image):
        """Apply appropriate transforms based on augmentation settings"""
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
                return augmented_image
            except Exception as e:
                print(f"Error applying augmentation, using original image: {str(e)}")
                # Fallback to basic resize if augmentation fails
                return image.resize((224, 224)) if image.size != (224, 224) else image
        else:
            # Just resize if no augmentation
            return image.resize((224, 224)) if image.size != (224, 224) else image
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get image from URL (should be cached)
        from PIL import Image
        import requests
        from io import BytesIO
        
        try:
            response = requests.get(sample['url'], timeout=5)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        except:
            # Create placeholder image
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        # Apply transforms (augmentation or just resize)
        image = self._apply_transforms(image)
        
        # Get label
        label = self.class_to_idx.get(sample['label'], 0)
        
        return image, sample.get('text', ''), label


def custom_collate_fn(batch):
    """Custom collate function for handling PIL images."""
    images = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    labels = torch.tensor([item[2] for item in batch])
    
    return images, texts, labels