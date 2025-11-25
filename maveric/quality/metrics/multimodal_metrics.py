"""Multimodal quality metrics combining vision and language."""

import numpy as np
import torch
import clip
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity

from .base_metric import BaseQualityMetric
from ...core.exceptions import MetricError


class MultimodalConsistencyMetric(BaseQualityMetric):
    """
    Multimodal consistency metric using CLIP embeddings.
    
    This metric measures how well different modalities (image and text) align
    with reference samples and templates. High consistency indicates that the
    image-text pair is coherent and represents the intended concept well.
    """
    
    def __init__(self, 
                 clip_model: str = "ViT-B/32",
                 device: str = "cuda",
                 reference_embeddings: Optional[Dict[str, np.ndarray]] = None,
                 text_embeddings: Optional[Dict[str, np.ndarray]] = None):
        """
        Initialize multimodal consistency metric.
        
        Args:
            clip_model: CLIP model to use
            device: Computation device
            reference_embeddings: Pre-computed reference image embeddings
            text_embeddings: Pre-computed text template embeddings
        """
        super().__init__("multimodal_consistency")
        
        # Set device
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Load CLIP model
        self.model, self.preprocess = clip.load(clip_model, device=self.device)
        self.model.eval()
        
        # Store reference embeddings
        self.reference_embeddings = reference_embeddings or {}
        self.text_embeddings = text_embeddings or {}
    
    @property
    def metric_name(self) -> str:
        return "consistency"
    
    @property
    def requires_reference(self) -> bool:
        return True
    
    def set_reference_embeddings(self, 
                                reference_embeddings: Dict[str, np.ndarray],
                                text_embeddings: Dict[str, np.ndarray]):
        """
        Set reference embeddings for consistency calculation.
        
        Args:
            reference_embeddings: Image embeddings for each class
            text_embeddings: Text template embeddings for each class
        """
        self.reference_embeddings = reference_embeddings
        self.text_embeddings = text_embeddings
    
    def compute(self, image: Image.Image, metadata: Dict[str, Any]) -> float:
        """
        Compute multimodal consistency score.
        
        This calculates consistency by measuring how well the image-text pair
        aligns with reference samples across multiple similarity metrics.
        
        Args:
            image: PIL Image
            metadata: Must contain 'text' and 'label'
            
        Returns:
            Consistency score (0-1)
        """
        if not self.reference_embeddings or not self.text_embeddings:
            raise MetricError("Reference embeddings not set")
        
        # Get label
        label = metadata.get('label')
        if not label or label not in self.reference_embeddings:
            return 0.0
        
        # Get text
        text = metadata.get('text', metadata.get('caption', ''))
        if not text:
            return 0.0
        
        try:
            # Compute embeddings
            image_embedding = self._encode_image(image)
            text_embedding = self._encode_text(text)
            
            # Get reference embeddings for this class
            ref_images = self.reference_embeddings[label]
            ref_texts = self.text_embeddings[label]
            
            # Compute similarities
            img2img = cosine_similarity(image_embedding, ref_images).max()
            txt2txt = cosine_similarity(text_embedding, ref_texts).max()
            img2txt = cosine_similarity(image_embedding, ref_texts).max()
            txt2img = cosine_similarity(text_embedding, ref_images).max()
            
            # Calculate consistency as inverse of standard deviation
            similarities = [img2img, txt2txt, img2txt, txt2img]
            consistency = 1.0 - np.std(similarities)
            
            return round(float(consistency), 5)
            
        except Exception as e:
            self.log_warning(f"Error computing consistency: {e}")
            return 0.0
    
    def _encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode image to CLIP embedding."""
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()
    
    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text to CLIP embedding."""
        tokens = clip.tokenize([text], truncate=True).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()


class CrossModalAlignmentMetric(BaseQualityMetric):
    """
    Cross-modal alignment metric.
    
    This metric specifically measures how well an image and its caption
    align with each other, independent of class labels. It's useful for
    filtering out misaligned image-text pairs.
    """
    
    def __init__(self, clip_model: str = "ViT-B/32", device: str = "cuda"):
        """
        Initialize cross-modal alignment metric.
        
        Args:
            clip_model: CLIP model to use
            device: Computation device
        """
        super().__init__("cross_modal_alignment")
        
        # Set device
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Load CLIP model
        self.model, self.preprocess = clip.load(clip_model, device=self.device)
        self.model.eval()
    
    @property
    def metric_name(self) -> str:
        return "alignment_score"
    
    def compute(self, image: Image.Image, metadata: Dict[str, Any]) -> float:
        """
        Compute cross-modal alignment score.
        
        This measures the direct similarity between the image and its caption
        using CLIP's joint embedding space.
        
        Args:
            image: PIL Image
            metadata: Must contain 'text' or 'caption'
            
        Returns:
            Alignment score (0-1)
        """
        text = metadata.get('text', metadata.get('caption', ''))
        if not text:
            return 0.0
        
        try:
            # Encode image
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Encode text
            text_tokens = clip.tokenize([text], truncate=True).to(self.device)
            
            with torch.no_grad():
                # Get embeddings
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(text_tokens)
                
                # Normalize
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calculate cosine similarity
                similarity = (image_features @ text_features.T).item()
                
                # Convert to 0-1 range (CLIP similarities are typically in [-1, 1])
                score = (similarity + 1.0) / 2.0
            
            return round(float(score), 5)
            
        except Exception as e:
            self.log_warning(f"Error computing alignment: {e}")
            return 0.0


class TargetClassQualityMetric(BaseQualityMetric):
    """
    Target class quality metric using CLIP-based class mapping and EfficientNet.
    
    This metric pre-computes mappings between target dataset classes and ImageNet
    classes using CLIP semantic similarity, then evaluates image quality by
    measuring EfficientNet confidence on the most relevant ImageNet classes.
    
    Simple and efficient: "How confident is EfficientNet that this image contains
    objects from the target class?"
    """
    
    def __init__(self, target_dataset: str = None):
        """Initialize target class quality metric.
        
        Args:
            target_dataset: Target dataset name (e.g., 'cifar10', 'cifar100') for class-specific mapping
        """
        super().__init__("target_class_quality")
        
        # Force CPU usage for data retrieval performance
        self.device = "cpu"
        self.target_dataset = target_dataset
        
        # Load EfficientNet-B0 for image classification (CPU)
        try:
            from torchvision.models import efficientnet_b0
            import torchvision.transforms as transforms
            self.efficientnet = efficientnet_b0(pretrained=True)
            self.efficientnet.eval()
        except ImportError:
            raise ImportError("torchvision is required for EfficientNet model")
        
        # Load CLIP for semantic similarity (more robust than sentence transformers)
        try:
            import clip
            self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
            self.clip_model.eval()
        except ImportError:
            raise ImportError("openai-clip is required for semantic similarity")
        
        # Load ImageNet class names
        try:
            from .imagenet_classes import IMAGENET_CLASSES
            self.imagenet_classes = IMAGENET_CLASSES
        except ImportError:
            raise ImportError("ImageNet classes not found")
        
        # No need to pre-compute all ImageNet embeddings with the simplified approach
        
        # Image preprocessing (EfficientNet standard)
        import torchvision.transforms as transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @property
    def metric_name(self) -> str:
        return "target_class_quality"
    
    @property
    def requires_reference(self) -> bool:
        return False
    
    
    
    def compute(self, image: Image.Image, metadata: Dict[str, Any]) -> float:
        """
        Compute target class quality using CORRECTED algorithm.
        
        CORRECT ALGORITHM:
        1. Get target class from metadata
        2. Get single highest-probability ImageNet class from EfficientNet  
        3. Calculate CLIP similarity between target class and predicted ImageNet class
        4. Return: CLIP_similarity × imagenet_probability
        
        Args:
            image: PIL Image
            metadata: Must contain 'label' or 'target_class'
            
        Returns:
            EfficientNet score: CLIP_similarity(target_class, imagenet_predicted_class) × imagenet_probability
        """
        try:
            # Get target class
            target_class = metadata.get('label', metadata.get('target_class', ''))
            
            if not target_class or len(target_class.strip()) == 0:
                self.log_debug("No target class provided, returning 0.0")
                return 0.0
            
            # Get EfficientNet probabilities
            probabilities = self.compute_image_probabilities_only(image)
            
            # Use the corrected batch method for single target class
            results = self.compute_all_mappings_from_probabilities(probabilities, [target_class])
            
            # Extract the score for the target class
            predicted_imagenet_class, clip_similarity, efficientNet_score = results.get(target_class, ("", 0.0, 0.0))
            
            return efficientNet_score
            
        except Exception as e:
            self.log_debug(f"Error computing target class quality: {e}")
            if not isinstance(e, (IndexError, ValueError)):
                self.log_warning(f"Unexpected error in target class quality: {type(e).__name__}: {e}")
            return 0.0
    
    def compute_with_best_mapping(self, image: Image.Image, metadata: Dict[str, Any]) -> Tuple[float, str, float]:
        """
        Compute target class quality using CORRECTED algorithm and return mapping details.
        
        CORRECT ALGORITHM:
        1. Get single highest-probability ImageNet class from EfficientNet
        2. Calculate CLIP similarity between target class and predicted ImageNet class  
        3. Return: (CLIP_similarity × imagenet_probability, predicted_imagenet_class, imagenet_probability)
        
        Args:
            image: PIL Image
            metadata: Must contain 'label' or 'target_class'
            
        Returns:
            Tuple of (efficientNet_score, predicted_imagenet_class_name, imagenet_probability)
        """
        try:
            # Get target class
            target_class = metadata.get('label', metadata.get('target_class', ''))
            
            if not target_class or len(target_class.strip()) == 0:
                self.log_debug("No target class provided, returning 0.0 and empty string")
                return 0.0, "", 0.0
            
            # Get EfficientNet probabilities and predicted class
            probabilities = self.compute_image_probabilities_only(image)
            predicted_imagenet_class, imagenet_probability = self.compute_single_imagenet_prediction(image)
            
            # Use the corrected batch method for single target class  
            results = self.compute_all_mappings_from_probabilities(probabilities, [target_class])
            
            # Extract the score for the target class
            _, clip_similarity, efficientNet_score = results.get(target_class, ("", 0.0, 0.0))
            
            return efficientNet_score, predicted_imagenet_class, imagenet_probability
            
        except Exception as e:
            self.log_debug(f"Error computing target class quality with best mapping: {e}")
            if not isinstance(e, (IndexError, ValueError)):
                self.log_warning(f"Unexpected error in target class quality with best mapping: {type(e).__name__}: {e}")
            return 0.0, "", 0.0
    
    def compute_all_mappings_from_probabilities(self, probabilities: torch.Tensor, target_classes: List[str]) -> Dict[str, Tuple[str, float, float]]:
        """
        Compute EfficientNet scores for all target classes using CORRECTED algorithm.
        
        CORRECT ALGORITHM:
        1. Get the single highest-probability ImageNet class from EfficientNet
        2. For each target class: Calculate CLIP similarity between target class and predicted ImageNet class
        3. Multiply: CLIP_similarity(target_class, imagenet_predicted_class) × imagenet_probability
        
        Args:
            probabilities: Pre-computed EfficientNet probabilities tensor (1000 ImageNet classes)
            target_classes: List of target class names to compute scores for
            
        Returns:
            Dictionary mapping target_class_name -> (predicted_imagenet_class, clip_similarity, efficientNet_score)
        """
        try:
            results = {}
            
            # Step 1: Get single highest-probability ImageNet class
            best_imagenet_idx = torch.argmax(probabilities).item()
            predicted_imagenet_class = self.imagenet_classes[best_imagenet_idx]
            imagenet_probability = probabilities[best_imagenet_idx].item()
            
            # Step 2 & 3: For each target class, calculate CLIP similarity and final score
            # Pre-encode the predicted ImageNet class once
            imagenet_prompt = f"a photo of a {predicted_imagenet_class}"
            imagenet_tokens = clip.tokenize([imagenet_prompt], truncate=True).to(self.device)
            
            with torch.no_grad():
                imagenet_embedding = self.clip_model.encode_text(imagenet_tokens)
                imagenet_embedding = imagenet_embedding / imagenet_embedding.norm(dim=-1, keepdim=True)
            
            imagenet_embedding = imagenet_embedding.cpu().numpy()
            
            # Calculate score for each target class
            for target_class in target_classes:
                if not target_class or len(target_class.strip()) == 0:
                    results[target_class] = (predicted_imagenet_class, 0.0, 0.0)
                    continue
                
                # Encode target class
                target_prompt = f"a photo of a {target_class}"
                target_tokens = clip.tokenize([target_prompt], truncate=True).to(self.device)
                
                with torch.no_grad():
                    target_embedding = self.clip_model.encode_text(target_tokens)
                    target_embedding = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
                
                target_embedding = target_embedding.cpu().numpy()
                
                # Calculate CLIP similarity between target class and predicted ImageNet class
                clip_similarity = cosine_similarity(target_embedding, imagenet_embedding)[0][0]
                
                # Final score: Raw CLIP_similarity × imagenet_probability (no normalization)
                # Keep raw similarity [-1,1] for data curation flexibility
                efficientNet_score = clip_similarity * imagenet_probability
                
                results[target_class] = (predicted_imagenet_class, round(clip_similarity, 5), round(efficientNet_score, 5))
            
            return results
            
        except Exception as e:
            self.log_debug(f"Error computing all mappings from probabilities: {e}")
            if not isinstance(e, (IndexError, ValueError)):
                self.log_warning(f"Unexpected error in batch mapping computation: {type(e).__name__}: {e}")
            return {target_class: ("", 0.0, 0.0) for target_class in target_classes}
    
    def compute_single_imagenet_prediction(self, image: Image.Image) -> Tuple[str, float]:
        """
        Get the single highest-probability ImageNet class prediction from EfficientNet.
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple of (predicted_imagenet_class, imagenet_probability)
        """
        try:
            probabilities = self.compute_image_probabilities_only(image)
            
            # Get single highest-probability ImageNet class
            best_imagenet_idx = torch.argmax(probabilities).item()
            predicted_class = self.imagenet_classes[best_imagenet_idx]
            imagenet_probability = probabilities[best_imagenet_idx].item()
            
            return predicted_class, round(imagenet_probability, 5)
            
        except Exception as e:
            self.log_debug(f"Error computing single ImageNet prediction: {e}")
            return "", 0.0
    
    def compute_clip_similarity_for_class(self, target_class: str, imagenet_class: str) -> float:
        """
        Compute CLIP similarity between a target class and ImageNet class.

        This method is used for cached samples where we have the ImageNet prediction
        but need to compute similarity for a new target dataset's classes.

        Args:
            target_class: Target dataset class name (e.g., "airplane")
            imagenet_class: Predicted ImageNet class name (e.g., "jet")

        Returns:
            CLIP similarity score (cosine similarity in range [-1, 1])
        """
        try:
            if not target_class or not imagenet_class:
                return 0.0

            # Encode target class
            target_prompt = f"a photo of a {target_class}"
            target_tokens = clip.tokenize([target_prompt], truncate=True).to(self.device)

            with torch.no_grad():
                target_embedding = self.clip_model.encode_text(target_tokens)
                target_embedding = target_embedding / target_embedding.norm(dim=-1, keepdim=True)

            target_embedding = target_embedding.cpu().numpy()

            # Encode ImageNet class
            imagenet_prompt = f"a photo of a {imagenet_class}"
            imagenet_tokens = clip.tokenize([imagenet_prompt], truncate=True).to(self.device)

            with torch.no_grad():
                imagenet_embedding = self.clip_model.encode_text(imagenet_tokens)
                imagenet_embedding = imagenet_embedding / imagenet_embedding.norm(dim=-1, keepdim=True)

            imagenet_embedding = imagenet_embedding.cpu().numpy()

            # Calculate CLIP similarity
            clip_similarity = cosine_similarity(target_embedding, imagenet_embedding)[0][0]

            return round(float(clip_similarity), 5)

        except Exception as e:
            self.log_debug(f"Error computing CLIP similarity: {e}")
            return 0.0

    def compute_image_probabilities_only(self, image: Image.Image) -> torch.Tensor:
        """
        Compute only the EfficientNet probabilities for an image, without any mapping logic.

        This method runs EfficientNet inference once and returns the full probability tensor
        that can be reused for multiple target class mappings.
        
        Args:
            image: PIL Image
            
        Returns:
            EfficientNet probabilities tensor for all 1000 ImageNet classes
        """
        try:
            # Process image with EfficientNet
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            
            with torch.no_grad():
                logits = self.efficientnet(image_tensor)
                probabilities = torch.softmax(logits, dim=1).squeeze()
            
            return probabilities
            
        except Exception as e:
            self.log_debug(f"Error computing image probabilities: {e}")
            if not isinstance(e, (IndexError, ValueError)):
                self.log_warning(f"Unexpected error in image probability computation: {type(e).__name__}: {e}")
            # Return zero tensor with 1000 classes if error
            return torch.zeros(1000)