"""Model evaluation utilities."""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, roc_auc_score
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

    @staticmethod
    def _get_canonical_name(class_name):
        """
        Extract canonical name from class_name.
        Handles both strings and list-based class names (e.g., FER2013's ['happy', 'smiling']).

        Args:
            class_name: Either a string or list of strings

        Returns:
            String representing the canonical class name
        """
        if isinstance(class_name, list):
            return class_name[0]
        return class_name

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
            # Extract canonical name for debug printing
            first_canonical = self._get_canonical_name(class_names[0])
            print(f"\nDEBUG Text Classifier Creation:")
            print(f"  Number of classes: {len(class_names)}")
            print(f"  Number of templates: {len(templates)}")
            print(f"  Templates: {templates}")
            print(f"  Example prompts for '{first_canonical}':")
            for tmpl in templates:
                print(f"    - {tmpl.format(first_canonical)}")
            self._template_debug_printed = True

        for class_name in class_names:
            # Extract canonical name (handles FER2013's list format)
            canonical_name = self._get_canonical_name(class_name)
            # Generate prompts for this class using all templates
            class_prompts = [template.format(canonical_name) for template in templates]

            # Tokenize and encode
            text_inputs = model.processor(text=class_prompts, return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                # Get text embeddings for all templates
                text_features_output = model.clip_model.get_text_features(**text_inputs)

                # Handle both tensor and BaseModelOutputWithPooling formats
                # HuggingFace updated to return BaseModelOutputWithPooling instead of tensor
                if isinstance(text_features_output, torch.Tensor):
                    class_embeddings = text_features_output
                else:
                    # Extract pooler_output from BaseModelOutputWithPooling
                    # pooler_output has shape (batch_size, hidden_size) which is what we need
                    # [0] would give last_hidden_state with shape (batch_size, seq_len, hidden_size) - wrong!
                    class_embeddings = text_features_output.pooler_output if hasattr(text_features_output, 'pooler_output') else text_features_output[0]

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
            # Extract canonical names for string formatting
            canonical_names = [self._get_canonical_name(name) for name in class_names]
            class_prompts = [f"a photo of a {name}." for name in canonical_names]
            text_inputs = model.processor(text=class_prompts, return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                text_features_output = model.clip_model.get_text_features(**text_inputs)

                # Handle both tensor and BaseModelOutputWithPooling formats
                if isinstance(text_features_output, torch.Tensor):
                    class_text_features = text_features_output
                else:
                    # Extract pooler_output from BaseModelOutputWithPooling
                    class_text_features = text_features_output.pooler_output if hasattr(text_features_output, 'pooler_output') else text_features_output[0]

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
            # Extract canonical names for string formatting
            canonical_names = [self._get_canonical_name(name) for name in class_names]
            class_prompts = [f"a photo of a {name}." for name in canonical_names]
            text_inputs = model.processor(text=class_prompts, return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                text_features_output = model.clip_model.get_text_features(**text_inputs)

                # Handle both tensor and BaseModelOutputWithPooling formats
                if isinstance(text_features_output, torch.Tensor):
                    class_text_features = text_features_output
                else:
                    # Extract pooler_output from BaseModelOutputWithPooling
                    class_text_features = text_features_output.pooler_output if hasattr(text_features_output, 'pooler_output') else text_features_output[0]

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
                # Use canonical name as dictionary key (handles FER2013's list format)
                canonical_name = self._get_canonical_name(class_name)
                per_class_accuracies[canonical_name] = class_acc
        
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
        # Extract canonical names for string formatting
        canonical_names = [self._get_canonical_name(name) for name in class_names]
        class_prompts = [f"a photo of a {name}." for name in canonical_names]
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
        # Extract canonical names for sklearn (handles FER2013's list format)
        canonical_names = [self._get_canonical_name(name) for name in class_names]
        results = {
            'accuracy': 100 * (all_predictions == all_labels).mean(),
            'confusion_matrix': confusion_matrix(all_labels, all_predictions),
            'classification_report': classification_report(
                all_labels, all_predictions,
                target_names=canonical_names,
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
                # Extract canonical name (handles FER2013's list format)
                canonical_name = self._get_canonical_name(class_name)
                if canonical_name in metrics['classification_report']:
                    f1 = metrics['classification_report'][canonical_name]['f1-score']
                    result[f'{canonical_name} F1'] = f1
            
            results.append(result)

        return pd.DataFrame(results).round(2)

    def _compute_voc11_map(self,
                           scores: np.ndarray,
                           labels: np.ndarray,
                           num_classes: int) -> float:
        """
        Compute VOC 11-point interpolated mean Average Precision.

        Follows ELEVATER approach: uses softmax probabilities and treats
        difficult examples (flag=0) as negatives.

        Args:
            scores: Prediction scores (N, num_classes) - softmax probabilities
            labels: Ground truth multi-label binary matrix (N, num_classes)
            num_classes: Number of classes

        Returns:
            Mean Average Precision across all classes
        """
        aps = []

        for class_idx in range(num_classes):
            # Get scores and labels for this class
            class_scores = scores[:, class_idx]
            class_labels = labels[:, class_idx]

            # Count total positive samples for this class
            total_positives = np.sum(class_labels)

            # Skip if no positive samples
            if total_positives == 0:
                continue

            # Sort indices by score descending
            sorted_indices = np.argsort(-class_scores)
            sorted_labels = class_labels[sorted_indices]

            # TP: predicted as positive AND actually positive
            # FP: predicted as positive BUT actually negative
            tp = np.cumsum(sorted_labels == 1)
            fp = np.cumsum(sorted_labels == 0)

            # Calculate recall and precision curve
            recall = tp / total_positives
            precision = tp / (tp + fp)

            # 11-point interpolated AP calculation
            ap = 0.0
            for t in np.linspace(0, 1, 11):
                # Find the maximum precision where recall is greater than or equal to t
                higher_recall_mask = recall >= t
                if np.any(higher_recall_mask):
                    p = np.max(precision[higher_recall_mask])
                else:
                    p = 0.0
                ap += p / 11.0

            aps.append(ap)

        # Return mean AP across classes
        return float(np.mean(aps)) if aps else 0.0

    def evaluate_with_dataset_metric(self,
                                     model: nn.Module,
                                     data_loader: Any,
                                     class_names: List[str],
                                     dataset_name: str,
                                     metric_type: str = "accuracy",
                                     templates: List[str] = None,
                                     use_ensemble: bool = True) -> Dict[str, float]:
        """
        Evaluate model using dataset-specific metric (ELEVATER Table 5).

        Args:
            model: Model to evaluate
            data_loader: Data loader
            class_names: List of class names
            dataset_name: Name of dataset (for logging)
            metric_type: Type of metric to use:
                - "accuracy": Standard top-1 accuracy (default)
                - "mean_per_class": Balanced accuracy / mean-per-class accuracy
                - "roc_auc": ROC AUC score (binary classification)
                - "voc11_map": VOC 11-point interpolated mAP (multi-label)
            templates: List of prompt templates (if None, uses single default template)
            use_ensemble: Whether to use template ensembling (default: True)

        Returns:
            Dictionary with metric results
        """
        model.eval()

        # Create text features with template ensembling if enabled
        if use_ensemble and templates is not None:
            class_text_features = self._create_text_classifier_with_templates(model, class_names, templates)
        else:
            canonical_names = [self._get_canonical_name(name) for name in class_names]
            class_prompts = [f"a photo of a {name}." for name in canonical_names]
            text_inputs = model.processor(text=class_prompts, return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                text_features_output = model.clip_model.get_text_features(**text_inputs)

                if isinstance(text_features_output, torch.Tensor):
                    class_text_features = text_features_output
                else:
                    class_text_features = text_features_output.pooler_output if hasattr(text_features_output, 'pooler_output') else text_features_output[0]

                class_text_features = class_text_features / class_text_features.norm(dim=-1, keepdim=True)

        all_predictions = []
        all_labels = []
        all_scores = []  # Continuous scores for ROC AUC and VOC mAP

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating ({metric_type})"):
                images, texts, labels = batch
                labels = labels.to(self.device)

                # Get logits/scores
                logits = model(images, class_text_features)
                predictions = logits.argmax(dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_scores.append(logits.cpu())

        # Convert to numpy
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_scores = torch.cat(all_scores, dim=0).numpy()

        # Compute metric based on type
        results = {'dataset': dataset_name, 'metric_type': metric_type}

        if metric_type == "mean_per_class":
            # Balanced accuracy (mean-per-class accuracy)
            score = balanced_accuracy_score(all_labels, all_predictions)
            results['mean_per_class_accuracy'] = score * 100
            results['accuracy'] = score * 100  # Primary metric

            print(f"📊 {dataset_name} - Mean-per-class Accuracy: {score * 100:.2f}%")

        elif metric_type == "roc_auc":
            # ROC AUC for binary classification
            # Assumes positive class is index 1 (hateful)
            if len(class_names) == 2:
                # Use probability/score of positive class
                positive_scores = torch.softmax(torch.from_numpy(all_scores), dim=1)[:, 1].numpy()
                score = roc_auc_score(all_labels, positive_scores)
                results['roc_auc'] = score * 100
                results['accuracy'] = score * 100  # Primary metric

                print(f"📊 {dataset_name} - ROC AUC: {score * 100:.2f}%")
            else:
                raise ValueError(f"ROC AUC requires binary classification, but got {len(class_names)} classes")

        elif metric_type == "voc11_map":
            # VOC 11-point interpolated mAP (multi-label)
            # all_labels should be multi-hot vectors from VOC2007MultiLabelDataset
            num_samples = len(all_labels)
            num_classes = len(class_names)

            # Verify multi-label format
            if all_labels.ndim != 2 or all_labels.shape[1] != num_classes:
                raise ValueError(
                    f"VOC 11-point mAP requires multi-label format. "
                    f"Expected shape ({num_samples}, {num_classes}), got {all_labels.shape}. "
                    f"Ensure VOC2007MultiLabelDataset is being used for data loading."
                )

            print(f"✅ Using true multi-label annotations ({num_samples} images, {num_classes} classes)")

            # ELEVATER approach: Apply softmax to scores before mAP computation
            # logits = (100. * image_features @ text_features).softmax(dim=-1)
            all_scores_tensor = torch.from_numpy(all_scores)
            all_scores_softmax = torch.softmax(all_scores_tensor * 100.0, dim=-1).numpy()

            print(f"✅ Using ELEVATER approach: softmax(scores * 100)")

            # Compute mAP with softmax probabilities (no difficult masks)
            map_score = self._compute_voc11_map(all_scores_softmax, all_labels, num_classes)
            results['voc11_map'] = map_score * 100
            results['accuracy'] = map_score * 100  # Primary metric

            print(f"📊 {dataset_name} - VOC 11-point mAP: {map_score * 100:.2f}%")

        else:  # "accuracy" (default)
            # Standard top-1 accuracy
            score = (all_predictions == all_labels).mean()
            results['top1_accuracy'] = score * 100
            results['accuracy'] = score * 100  # Primary metric

            print(f"📊 {dataset_name} - Top-1 Accuracy: {score * 100:.2f}%")

        # Always compute standard accuracy for comparison (except for multi-label metrics)
        if metric_type not in ["accuracy", "voc11_map"]:
            standard_acc = (all_predictions == all_labels).mean()
            results['top1_accuracy'] = standard_acc * 100

        return results