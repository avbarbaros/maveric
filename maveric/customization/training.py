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
              training_config: TrainingConfig,
              class_names: List[str]) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            training_config: Training configuration
            class_names: List of class names
            
        Returns:
            Training history dictionary
        """
        # Create text features for all classes
        class_prompts = [f"a photo of a {name}." for name in class_names]
        with torch.no_grad():
            class_text_features = self.model.encode_text(class_prompts)
        
        # Setup optimizer
        optimizer = self._create_optimizer(training_config)
        
        # Setup scheduler
        scheduler = self._create_scheduler(optimizer, training_config, len(train_loader))
        
        # Setup loss
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Best model tracking
        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        
        # Training loop
        for epoch in range(training_config.epochs):
            self.log_info(f"Epoch {epoch + 1}/{training_config.epochs}")
            
            # Train
            train_loss, train_acc = self._train_epoch(
                train_loader,
                class_text_features,
                optimizer,
                criterion,
                scheduler,
                training_config
            )
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validate
            if epoch % training_config.eval_frequency == 0:
                val_loss, val_acc = self._validate_epoch(
                    val_loader,
                    class_text_features,
                    criterion
                )
                
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                self.log_info(
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
                )
                
                # Check for improvement
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    patience_counter = 0
                    
                    # Save best model
                    if training_config.save_best_model and self.checkpoint_dir:
                        self.save_checkpoint(
                            self.model,
                            f"best_epoch_{epoch}",
                            {'epoch': epoch, 'val_acc': val_acc}
                        )
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= training_config.early_stopping_patience:
                    self.log_info(f"Early stopping at epoch {epoch}")
                    break
            
            # Save periodic checkpoint
            if (epoch + 1) % training_config.save_frequency == 0 and self.checkpoint_dir:
                self.save_checkpoint(
                    self.model,
                    f"checkpoint_epoch_{epoch}",
                    {'epoch': epoch}
                )
        
        self.log_info(f"Training complete. Best epoch: {best_epoch} with {best_val_acc:.2f}% accuracy")
        
        return history
    
    def _train_epoch(self,
                     train_loader: Any,
                     class_text_features: torch.Tensor,
                     optimizer: optim.Optimizer,
                     criterion: nn.Module,
                     scheduler: Any,
                     config: TrainingConfig) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (images, texts, labels) in enumerate(progress_bar):
            labels = labels.to(self.device)
            
            # Forward pass
            logits = self.model(images, class_text_features)
            loss = criterion(logits, labels)
            
            # Add regularization
            if hasattr(self.model, 'get_regularization_loss'):
                reg_loss = self.model.get_regularization_loss()
                total_loss_value = loss + config.regularization_weight * reg_loss
            else:
                total_loss_value = loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss_value.backward()
            
            # Gradient clipping
            if config.gradient_clip_value > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    config.gradient_clip_value
                )
            
            optimizer.step()
            
            # Update scheduler
            if scheduler and hasattr(scheduler, 'step'):
                scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            predictions = logits.argmax(dim=1)
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
    
    def _validate_epoch(self,
                       val_loader: Any,
                       class_text_features: torch.Tensor,
                       criterion: nn.Module) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, texts, labels in tqdm(val_loader, desc="Validating"):
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(images, class_text_features)
                loss = criterion(logits, labels)
                
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
