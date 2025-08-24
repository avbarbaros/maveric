"""Model evaluation utilities."""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from ..core.base import BaseComponent


class Evaluator(BaseComponent):
    """Handles model evaluation and metrics calculation."""
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize evaluator.
        
        Args:
            device: Device for evaluation
        """
        super().__init__("Evaluator")
        self.device = device
    
    def evaluate(self,
                 model: nn.Module,
                 data_loader: Any,
                 class_names: List[str]) -> float:
        """
        Evaluate model accuracy.
        
        Args:
            model: Model to evaluate
            data_loader: Data loader
            class_names: List of class names
            
        Returns:
            Overall accuracy percentage
        """
        model.eval()
        
        # Create text features
        class_prompts = [f"a photo of a {name}." for name in class_names]
        text_inputs = model.processor(text=class_prompts, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            class_text_features = model.clip_model.get_text_features(**text_inputs)
            class_text_features = class_text_features / class_text_features.norm(dim=-1, keepdim=True)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, texts, labels in tqdm(data_loader, desc="Evaluating"):
                labels = labels.to(self.device)
                
                # Get predictions
                logits = model(images, class_text_features)
                predictions = logits.argmax(dim=1)
                
                # Count correct
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        accuracy = 100 * correct / total
        return accuracy
    
    def evaluate_detailed(self,
                         model: nn.Module,
                         data_loader: Any,
                         class_names: List[str]) -> Tuple[float, Dict[str, float]]:
        """
        Detailed evaluation with per-class metrics.
        
        Args:
            model: Model to evaluate
            data_loader: Data loader
            class_names: List of class names
            
        Returns:
            Tuple of (overall_accuracy, per_class_accuracies)
        """
        model.eval()
        
        # Create text features
        class_prompts = [f"a photo of a {name}." for name in class_names]
        text_inputs = model.processor(text=class_prompts, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            class_text_features = model.clip_model.get_text_features(**text_inputs)
            class_text_features = class_text_features / class_text_features.norm(dim=-1, keepdim=True)
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, texts, labels in tqdm(data_loader, desc="Evaluating"):
                labels = labels.to(self.device)
                
                # Get predictions
                logits = model(images, class_text_features)
                predictions = logits.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Overall accuracy
        overall_accuracy = 100 * (all_predictions == all_labels).mean()
        
        # Per-class accuracy
        per_class_accuracies = {}
        for i, class_name in enumerate(class_names):
            mask = all_labels == i
            if mask.any():
                class_acc = 100 * (all_predictions[mask] == all_labels[mask]).mean()
                per_class_accuracies[class_name] = class_acc
        
        return overall_accuracy, per_class_accuracies
    
    def evaluate_with_metrics(self,
                             model: nn.Module,
                             data_loader: Any,
                             class_names: List[str]) -> Dict[str, Any]:
        """
        Comprehensive evaluation with multiple metrics.
        
        Args:
            model: Model to evaluate
            data_loader: Data loader
            class_names: List of class names
            
        Returns:
            Dictionary with various evaluation metrics
        """
        model.eval()
        
        # Create text features
        class_prompts = [f"a photo of a {name}." for name in class_names]
        text_inputs = model.processor(text=class_prompts, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            class_text_features = model.clip_model.get_text_features(**text_inputs)
            class_text_features = class_text_features / class_text_features.norm(dim=-1, keepdim=True)
        
        
        all_predictions = []
        all_labels = []
        all_logits = []
        
        with torch.no_grad():
            for images, texts, labels in tqdm(data_loader, desc="Evaluating"):
                labels = labels.to(self.device)
                
                # Get predictions
                logits = model(images, class_text_features)
                predictions = logits.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_logits.append(logits.cpu())
        
        # Convert to numpy
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_logits = torch.cat(all_logits, dim=0)
        
        # Calculate metrics
        results = {
            'accuracy': 100 * (all_predictions == all_labels).mean(),
            'confusion_matrix': confusion_matrix(all_labels, all_predictions),
            'classification_report': classification_report(
                all_labels, all_predictions,
                target_names=class_names,
                output_dict=True
            )
        }
        
        # Top-5 accuracy
        top5_preds = all_logits.topk(5, dim=1)[1].numpy()
        top5_correct = sum(label in preds for label, preds in zip(all_labels, top5_preds))
        results['top5_accuracy'] = 100 * top5_correct / len(all_labels)
        
        # Average confidence
        probs = torch.softmax(all_logits, dim=1)
        max_probs = probs.max(dim=1)[0]
        results['avg_confidence'] = max_probs.mean().item()
        
        return results
    
    def compare_models(self,
                      models: Dict[str, nn.Module],
                      data_loader: Any,
                      class_names: List[str]) -> pd.DataFrame:
        """
        Compare multiple models on the same dataset.
        
        Args:
            models: Dictionary mapping model names to models
            data_loader: Data loader
            class_names: List of class names
            
        Returns:
            DataFrame with comparison results
        """
        import pandas as pd
        
        results = []
        
        for model_name, model in models.items():
            self.log_info(f"Evaluating {model_name}")
            
            # Get detailed metrics
            metrics = self.evaluate_with_metrics(model, data_loader, class_names)
            
            # Extract key metrics
            result = {
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Top-5 Accuracy': metrics['top5_accuracy'],
                'Avg Confidence': metrics['avg_confidence']
            }
            
            # Add per-class F1 scores
            for class_name in class_names[:5]:  # Top 5 classes
                if class_name in metrics['classification_report']:
                    f1 = metrics['classification_report'][class_name]['f1-score']
                    result[f'{class_name} F1'] = f1
            
            results.append(result)
        
        return pd.DataFrame(results).round(2)