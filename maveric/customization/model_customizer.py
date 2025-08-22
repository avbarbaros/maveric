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
                 checkpoint_dir: Optional[str] = None):
        """
        Initialize model customizer.
        
        Args:
            base_model_name: Pre-trained model to customize
            device: Device for computation
            checkpoint_dir: Directory for saving checkpoints
        """
        super().__init__("ModelCustomizer")
        
        self.base_model_name = base_model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        
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
        """
        self.log_info("Starting model customization")
        
        # Create customized model
        customized_model = CustomizedCLIP(
            self.model,
            self.processor,
            regularize=training_config.use_regularization
        ).to(self.device)
        
        # Prepare data
        train_loader, val_loader = self._prepare_data(
            quality_result,
            class_names,
            training_config,
            validation_split
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=customized_model,
            device=self.device,
            checkpoint_dir=self.checkpoint_dir
        )
        
        # Create evaluator
        self.evaluator = Evaluator(device=self.device)
        
        # Get baseline performance
        self.log_info("Evaluating baseline model")
        baseline_accuracy = self._evaluate_baseline(val_loader, class_names)
        
        # Train model
        self.log_info("Training customized model")
        training_history = self.trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            training_config=training_config,
            class_names=class_names
        )
        
        # Final evaluation
        self.log_info("Evaluating customized model")
        final_accuracy, class_accuracies = self.evaluator.evaluate_detailed(
            customized_model,
            val_loader,
            class_names
        )
        
        # Get the best checkpoint path from training (already saved during training)
        best_checkpoint = None
        if training_config.save_best_model and self.checkpoint_dir:
            # The best model is already saved during training as "best_model.pth"
            best_checkpoint = self.checkpoint_dir / "best_model.pth"
            if not best_checkpoint.exists():
                # Fallback: save current model if no best checkpoint exists
                best_checkpoint = self.trainer.save_checkpoint(
                    customized_model,
                    f"best_model_{target_dataset_name}",
                    {
                        'accuracy': final_accuracy,
                        'baseline': baseline_accuracy,
                        'config': training_config.to_dict()
                    }
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
                      validation_split: float) -> Tuple[Any, Any]:
        """Prepare data loaders for training."""
        from torch.utils.data import DataLoader, random_split
        
        # Create dataset
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
        
        # Split dataset
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Disable augmentation for validation
        val_dataset.dataset.use_augmentation = False
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,  # Fixed batch size for CLIP
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
        
        self.log_info(f"Created data loaders: {train_size} train, {val_size} validation")
        
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
        # Process images
        if isinstance(images[0], torch.Tensor):
            # Already preprocessed
            pixel_values = torch.stack(images).to(self.device)
        else:
            # Process PIL images
            inputs = self.processor(
                images=images,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            pixel_values = inputs.pixel_values
        
        # Get image features
        image_outputs = self.clip_model.vision_model(pixel_values=pixel_values)
        image_embeds = image_outputs[1]  # pooled output
        image_embeds = self.clip_model.visual_projection(image_embeds)
        
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
            text_outputs = self.clip_model.text_model(**tokens)
            text_embeds = text_outputs[1]  # pooled output
            text_embeds = self.clip_model.text_projection(text_embeds)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        return text_embeds


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
        self.use_augmentation = use_augmentation
        
        # Setup augmentation
        if use_augmentation:
            from torchvision import transforms
            
            aug_config = augmentation_config or {}
            self.augmentation = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(
                    num_ops=aug_config.get('num_ops', 2),
                    magnitude=aug_config.get('magnitude', 9)
                )
            ])
        else:
            self.augmentation = None
    
    def __len__(self):
        return len(self.samples)
    
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
        
        # Apply augmentation
        if self.augmentation and self.use_augmentation:
            image = self.augmentation(image)
        
        # Get label
        label = self.class_to_idx.get(sample['label'], 0)
        
        return image, sample.get('text', ''), label


def custom_collate_fn(batch):
    """Custom collate function for handling PIL images."""
    images = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    labels = torch.tensor([item[2] for item in batch])
    
    return images, texts, labels