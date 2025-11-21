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

    def _create_text_classifier_with_templates(self,
                                               model: nn.Module,
                                               class_names: List[str],
                                               templates: List[str]) -> torch.Tensor:
        """
        Create text classifier with template ensembling (REACT-style).

        This method follows REACT's evaluation approach:
        1. Generate prompts for each class using all templates
        2. Normalize each template embedding
        3. Average across templates
        4. Re-normalize after averaging

        Args:
            model: CLIP model
            class_names: List of class names
            templates: List of prompt templates (e.g., ["a photo of a {}, a type of pet."])

        Returns:
            Text classifier tensor (embedding_dim x num_classes)
        """
        zeroshot_weights = []

        # Debug: Print template info
        if not hasattr(self, '_template_debug_printed'):
            print(f"\nDEBUG Text Classifier Creation:")
            print(f"  Number of classes: {len(class_names)}")
            print(f"  Number of templates: {len(templates)}")
            print(f"  Templates: {templates}")
            print(f"  Example prompts for '{class_names[0]}':")
            for tmpl in templates:
                print(f"    - {tmpl.format(class_names[0])}")
            self._template_debug_printed = True

        for class_name in class_names:
            # Generate prompts for this class using all templates
            class_prompts = [template.format(class_name) for template in templates]

            # Tokenize and encode
            text_inputs = model.processor(text=class_prompts, return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                # Get text embeddings for all templates
                class_embeddings = model.clip_model.get_text_features(**text_inputs)

                # Normalize each template embedding
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

                # Average across templates
                class_embedding = class_embeddings.mean(dim=0)

                # Re-normalize after averaging (important!)
                class_embedding = class_embedding / class_embedding.norm()

                zeroshot_weights.append(class_embedding)

        # Stack to create classifier matrix: (num_classes, embedding_dim)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=0)

        return zeroshot_weights

    def evaluate(self,
                 model: nn.Module,
                 data_loader: Any,
                 class_names: List[str],
                 templates: List[str] = None,
                 use_ensemble: bool = True) -> float:
        """
        Evaluate model accuracy with optional template ensembling.

        Args:
            model: Model to evaluate
            data_loader: Data loader
            class_names: List of class names
            templates: List of prompt templates (if None, uses single default template)
            use_ensemble: Whether to use template ensembling (default: True for REACT-style evaluation)

        Returns:
            Overall accuracy percentage
        """
        model.eval()

        # Create text features with template ensembling if enabled
        if use_ensemble and templates is not None:
            # REACT-style evaluation with template ensembling
            class_text_features = self._create_text_classifier_with_templates(model, class_names, templates)
        else:
            # Simple single-template evaluation (backward compatibility)
            class_prompts = [f"a photo of a {name}." for name in class_names]
            text_inputs = model.processor(text=class_prompts, return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                class_text_features = model.clip_model.get_text_features(**text_inputs)
                class_text_features = class_text_features / class_text_features.norm(dim=-1, keepdim=True)
                # Shape: (num_classes, embedding_dim) - matches ensemble format

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
                         class_names: List[str],
                         templates: List[str] = None,
                         use_ensemble: bool = True) -> Tuple[float, Dict[str, float]]:
        """
        Detailed evaluation with per-class metrics.

        Args:
            model: Model to evaluate
            data_loader: Data loader
            class_names: List of class names
            templates: List of prompt templates (if None, uses single default template)
            use_ensemble: Whether to use template ensembling (default: True)

        Returns:
            Tuple of (overall_accuracy, per_class_accuracies)
        """
        model.eval()

        # Create text features with template ensembling if enabled
        if use_ensemble and templates is not None:
            # REACT-style evaluation with template ensembling
            class_text_features = self._create_text_classifier_with_templates(model, class_names, templates)
        else:
            # Simple single-template evaluation (backward compatibility)
            class_prompts = [f"a photo of a {name}." for name in class_names]
            text_inputs = model.processor(text=class_prompts, return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                class_text_features = model.clip_model.get_text_features(**text_inputs)
                class_text_features = class_text_features / class_text_features.norm(dim=-1, keepdim=True)
                # Shape: (num_classes, embedding_dim) - matches ensemble format
        
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
        
        # Create text features (using same method as original code)
        class_prompts = [f"a photo of a {name}." for name in class_names]
        text_inputs = model.processor(text=class_prompts, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            class_text_features = model.encode_text(class_prompts)
        
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