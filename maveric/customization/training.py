"""Training utilities for model customization."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ConstantLR
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import time
from tqdm import tqdm

from ..core.base import BaseComponent
from ..config import TrainingConfig


class Trainer(BaseComponent):
    """Handles model training with various optimization strategies."""
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = "cuda",
                 checkpoint_dir: Optional[Path] = None):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            device: Device for training
            checkpoint_dir: Directory for checkpoints
        """
        super().__init__("Trainer")
        self.model = model
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train(self,
              train_loader: Any,
              val_loader: Any,
              test_loader: Any,
              training_config: TrainingConfig,
              class_names: List[str],
              templates: Optional[List[str]] = None,
              evaluator: Optional[Any] = None,
              dataset_name: Optional[str] = None,
              evaluation_metric: str = "accuracy") -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader (mandatory for proper evaluation)
            training_config: Training configuration
            class_names: List of class names
            templates: Optional list of text templates for REACT-style evaluation
            evaluator: Optional evaluator instance for template ensembling
            dataset_name: Optional dataset name (for logging)
            evaluation_metric: Evaluation metric type (accuracy, mean_per_class, roc_auc, voc11_map)

        Returns:
            Training history dictionary

        Note:
            Test data evaluation is mandatory at each epoch for reliable model selection,
            UNLESS training_config.skip_epoch_evaluation is True (useful for unified training).
            When skip_epoch_evaluation=True, only training metrics are logged and periodic
            checkpoints are saved without evaluation overhead.
            If templates and evaluator are provided, uses REACT-style template ensembling
            for consistent evaluation between training and final evaluation.

            The evaluation_metric parameter is for logging purposes. The training loop uses
            standard accuracy for monitoring, while final evaluation uses the dataset-specific
            metric specified in the config.
        """
        # Validate test loader is provided
        if test_loader is None:
            raise ValueError("Test data loader is required for training. Test evaluation is mandatory for reliable model selection.")

        # Check text source mode
        use_caption_mode = training_config.text_source == "captions"

        # Always pre-compute class text features for validation/test evaluation
        # (regardless of training mode)
        if templates is not None and evaluator is not None:
            # REACT-style evaluation with template ensembling (consistent with final evaluation)
            self.log_info(f"Using REACT-style template ensembling with {len(templates)} templates for evaluation")
            with torch.no_grad():
                class_text_features = evaluator._create_text_classifier_with_templates(
                    self.model, class_names, templates
                )
        else:
            # Fast single-template evaluation (legacy behavior)
            self.log_info("Using single-template evaluation (fast but less accurate)")
            class_prompts = [f"a photo of a {name}." for name in class_names]
            text_inputs = self.model.processor(text=class_prompts, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                text_features_output = self.model.clip_model.get_text_features(**text_inputs)

                # Handle both tensor and BaseModelOutputWithPooling formats
                if isinstance(text_features_output, torch.Tensor):
                    class_text_features = text_features_output
                else:
                    # Extract pooler_output from BaseModelOutputWithPooling
                    class_text_features = text_features_output.pooler_output if hasattr(text_features_output, 'pooler_output') else text_features_output[0]

                class_text_features = class_text_features / class_text_features.norm(dim=-1, keepdim=True)

        if use_caption_mode:
            # Caption-based mode: Text features computed per-batch from captions during training
            # But validation/test still use pre-computed class_text_features above
            self.log_info("🎯 Using caption-based training (per-sample text from 'text' field)")
            self.log_info("   Training: InfoNCE loss with per-batch caption encoding")
            self.log_info("   Validation/Test: Label-based evaluation with pre-computed class features")

        # Setup optimizer
        optimizer = self._create_optimizer(training_config)

        # Setup loss based on mode
        if use_caption_mode:
            from .losses import InfoNCELoss
            criterion = InfoNCELoss(temperature=0.07)
            self.log_info("   Loss function: InfoNCE (contrastive learning)")
        else:
            criterion = nn.CrossEntropyLoss()
            self.log_info("   Loss function: CrossEntropy (classification)")

        # Store mode for training loop
        self._use_caption_mode = use_caption_mode
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': [],
            'test_acc': []
        }
        
        # Best model tracking
        best_val_acc = 0.0
        best_epoch = 0
        best_checkpoint_path = None
        
        # Training loop
        for epoch in range(training_config.epochs):
            self.log_info(f"Epoch {epoch + 1}/{training_config.epochs}")
            
            # Train
            train_loss, train_acc = self._train_epoch(
                train_loader,
                class_text_features,
                optimizer,
                criterion,
                training_config
            )
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            # Validate and Test (skip if disabled for unified training)
            if not training_config.skip_epoch_evaluation and epoch % training_config.eval_frequency == 0:
                # Validation (optional)
                if val_loader is not None:
                    val_loss, val_acc = self._validate_epoch(
                        val_loader,
                        class_text_features,
                        criterion
                    )
                    history['val_loss'].append(val_loss)
                    history['val_acc'].append(val_acc)
                else:
                    # No validation - use dummy values
                    val_loss, val_acc = 0.0, 0.0
                    history['val_loss'].append(val_loss)
                    history['val_acc'].append(val_acc)

                # Test is evaluated ONCE at the end (in ModelCustomizer.customize).
                # Per-epoch test eval is monitoring only and never drives selection.
                if training_config.evaluate_test_each_epoch:
                    test_loss, test_acc = self._validate_epoch(
                        test_loader, class_text_features, criterion, desc="Testing"
                    )
                    history['test_loss'].append(test_loss)
                    history['test_acc'].append(test_acc)
                else:
                    test_loss, test_acc = 0.0, 0.0  

                # Log results (always includes test metrics)
                if val_loader is not None:
                    log_msg = (f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
                else:
                    log_msg = (f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% (no validation)")
                self.log_info(log_msg)

                # Select the best checkpoint on the VALIDATION fold (never test).
                if training_config.checkpoint_selection_metric == "test_acc":
                    eval_acc = test_acc  # legacy / ablation only
                else:# default to validation accuracy for model selection
                    if val_loader is None:
                        raise ValueError(
                            "Validation-based checkpoint selection requires a validation set. "
                            "Set use_validation=True (validation_method='stratified_kfold')."
                        )
                    eval_acc = val_acc
                if eval_acc > best_val_acc:
                    best_val_acc = eval_acc
                    best_epoch = epoch

                    # Save best model (remove previous best if exists)
                    if training_config.save_best_model and self.checkpoint_dir:
                        # Remove previous best checkpoint to save disk space
                        if best_checkpoint_path and best_checkpoint_path.exists():
                            best_checkpoint_path.unlink()
                            self.log_info(f"Removed previous best checkpoint: {best_checkpoint_path}")

                        checkpoint_metadata = {
                            'epoch': epoch,
                            'test_acc': eval_acc,
                            'is_best': True
                        }
                        if val_loader is not None:
                            checkpoint_metadata['val_acc'] = val_acc
                        else:
                            checkpoint_metadata['val_acc'] = 0.0  # No validation

                        best_checkpoint_path = self.save_checkpoint(
                            self.model,
                            f"best_model",
                            checkpoint_metadata
                        )
            elif training_config.skip_epoch_evaluation:
                # Skip evaluation but save periodic checkpoints for unified training
                self.log_info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% (evaluation skipped)")

                # Save checkpoint every save_frequency epochs
                if (epoch + 1) % training_config.save_frequency == 0:
                    checkpoint_metadata = {
                        'epoch': epoch,
                        'train_acc': train_acc,
                        'train_loss': train_loss,
                        'is_best': False
                    }
                    checkpoint_path = self.save_checkpoint(
                        self.model,
                        f"checkpoint_epoch_{epoch+1}",
                        checkpoint_metadata
                    )

            # Skip periodic checkpoints to save disk space - only keep best model
            # Periodic checkpoints disabled for disk efficiency
        
        self.log_info(f"Training complete. Best epoch: {best_epoch} with {best_val_acc:.2f}% accuracy")
        
        return history
    
    def _train_epoch(self,
                     train_loader: Any,
                     class_text_features: Optional[torch.Tensor],
                     optimizer: optim.Optimizer,
                     criterion: nn.Module,
                     config: TrainingConfig) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            class_text_features: Pre-computed text features for all classes
                                (always provided, used in label mode or for validation)
        """
        self.model.train()

        # Use instance variable to determine mode (set in train() method)
        use_caption_mode = getattr(self, '_use_caption_mode', False)

        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc="Training")

        for batch_idx, (images, texts, labels) in enumerate(progress_bar):
            labels = labels.to(self.device)

            # Forward pass - MODE DEPENDENT
            if use_caption_mode:
                # Caption-based: Encode per-sample captions and compute contrastive loss
                image_embeds, text_embeds = self._forward_caption_mode(images, texts)
                loss = criterion(image_embeds, text_embeds)

                # For accuracy tracking, compute similarity and predict
                with torch.no_grad():
                    logits = 100.0 * torch.matmul(image_embeds, text_embeds.T)
                    predictions = logits.argmax(dim=1)
            else:
                # Label-based: Use pre-computed class text features
                logits = self.model(images, class_text_features)
                loss = criterion(logits, labels)
                predictions = logits.argmax(dim=1)

            reg_loss = self.model.get_regularization_loss()
            total_loss_value = loss + config.regularization_weight * reg_loss

            # Backward pass
            optimizer.zero_grad()
            total_loss_value.backward()

            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100 * correct / total:.2f}%"
            })

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    def _forward_caption_mode(self,
                             images: List,
                             texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for caption-based training mode.

        Args:
            images: List of PIL images
            texts: List of caption strings (one per image)

        Returns:
            Tuple of (image_embeds, text_embeds) - both normalized
        """
        # Get image embeddings
        inputs = self.model._safe_process_images(self.model.processor, images, self.device)
        image_embeds_output = self.model.clip_model.get_image_features(**inputs)

        # Handle BaseModelOutputWithPooling format
        if isinstance(image_embeds_output, torch.Tensor):
            image_embeds = image_embeds_output
        else:
            image_embeds = image_embeds_output.pooler_output if hasattr(image_embeds_output, 'pooler_output') else image_embeds_output[0]

        # Normalize image embeddings
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        # Get text embeddings from captions (with frozen text encoder)
        text_inputs = self.model.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)

        with torch.no_grad():
            text_embeds_output = self.model.clip_model.get_text_features(**text_inputs)

            # Handle BaseModelOutputWithPooling format
            if isinstance(text_embeds_output, torch.Tensor):
                text_embeds = text_embeds_output
            else:
                text_embeds = text_embeds_output.pooler_output if hasattr(text_embeds_output, 'pooler_output') else text_embeds_output[0]

            # Normalize text embeddings
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        return image_embeds, text_embeds

    def _validate_epoch(self,
                       val_loader: Any,
                       class_text_features: torch.Tensor,
                       criterion: nn.Module,
                       desc: str = "Validating") -> Tuple[float, float]:
        """Validate for one epoch.

        Note: Validation ALWAYS uses label-based evaluation with CrossEntropyLoss,
        even when training uses caption-based InfoNCE loss. This ensures consistent
        evaluation across training modes.
        """
        self.model.eval()

        # Always use CrossEntropyLoss for validation (label-based evaluation)
        ce_criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, texts, labels in tqdm(val_loader, desc=desc):
                labels = labels.to(self.device)

                # Forward pass (label-based classification)
                logits = self.model(images, class_text_features)
                loss = ce_criterion(logits, labels)

                # Track metrics
                total_loss += loss.item()
                predictions = logits.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy
    
    def _create_optimizer(self, config: TrainingConfig) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        if config.optimizer == "adamw":
            return optim.AdamW(
                params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == "adam":
            return optim.Adam(
                params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer == "sgd":
            return optim.SGD(
                params,
                lr=config.learning_rate,
                momentum=0.9,
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    def _create_scheduler(self, 
                         optimizer: optim.Optimizer,
                         config: TrainingConfig,
                         steps_per_epoch: int) -> Any:
        """Create learning rate scheduler."""
        total_steps = config.epochs * steps_per_epoch
        
        if config.scheduler == "cosine":
            return CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=1e-7
            )
        elif config.scheduler == "linear":
            return LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.01,
                total_iters=total_steps
            )
        elif config.scheduler == "constant":
            return ConstantLR(
                optimizer,
                factor=1.0,
                total_iters=total_steps
            )
        else:
            return None
    
    def save_checkpoint(self,
                       model: nn.Module,
                       name: str,
                       metadata: Optional[Dict] = None) -> Path:
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{name}.pth"
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'metadata': metadata or {},
            'timestamp': time.time()
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.log_info(f"Saved checkpoint: {checkpoint_path}")
        
        return checkpoint_path


class TrainingMonitor:
    """Monitor training progress with callbacks."""
    
    def __init__(self):
        self.callbacks = []
        self.metrics = {}
    
    def add_callback(self, callback):
        """Add a monitoring callback."""
        self.callbacks.append(callback)
    
    def update(self, epoch: int, metrics: Dict[str, float]):
        """Update metrics and trigger callbacks."""
        self.metrics[epoch] = metrics
        
        for callback in self.callbacks:
            callback(epoch, metrics)
    
    def get_best_epoch(self, metric: str = 'val_acc', mode: str = 'max') -> int:
        """Get epoch with best metric value."""
        if not self.metrics:
            return 0
        
        values = [(epoch, m.get(metric, 0)) for epoch, m in self.metrics.items()]
        
        if mode == 'max':
            return max(values, key=lambda x: x[1])[0]
        else:
            return min(values, key=lambda x: x[1])[0]
