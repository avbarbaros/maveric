"""Visual quality metrics for images."""

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from typing import Any, Dict, Optional

from .base_metric import BaseQualityMetric


class ResolutionMetric(BaseQualityMetric):
    """
    Resolution quality metric.
    
    This metric evaluates image resolution relative to the expected input size
    for vision models. Higher resolution images generally contain more detail
    and provide better training signals.
    """
    
    def __init__(self):
        """
        Initialize resolution metric.
        
        Uses fixed 224px target size to match original MAVERIC implementation.
        """
        super().__init__("resolution")
    
    @property
    def metric_name(self) -> str:
        return "resolution_score"
    
    def compute(self, image: Image.Image, metadata: Dict[str, Any]) -> float:
        """
        Compute resolution score.
        
        Simple resolution score matching the original MAVERIC implementation:
        min(width, height) / 224.0
        
        Args:
            image: PIL Image
            metadata: Image metadata
            
        Returns:
            Resolution score (can exceed 1 for high-res images)
        """
        # Resolution quality: 224x224 is the input size of CLIP
        width, height = image.size
        resolution_score = min(width, height) / 224.0
        
        return round(float(resolution_score), 5)


class SharpnessMetric(BaseQualityMetric):
    """
    Sharpness metric using Laplacian variance.
    
    This metric measures image sharpness by analyzing high-frequency content.
    Sharp images have clear edges and details, which are important for
    model training. Blurry images can confuse models and reduce performance.
    """
    
    def __init__(self, sigmoid_scale: float = 0.01):
        """
        Initialize sharpness metric.
        
        Args:
            sigmoid_scale: Scale factor for sigmoid normalization
        """
        super().__init__("sharpness")
        self.sigmoid_scale = sigmoid_scale
    
    @property
    def metric_name(self) -> str:
        return "sharpness_score"
    
    def compute(self, image: Image.Image, metadata: Dict[str, Any]) -> float:
        """
        Compute sharpness score using Laplacian variance.
        
        The Laplacian operator highlights regions of rapid intensity change,
        which correspond to edges. The variance of the Laplacian indicates
        how much edge information is present in the image.
        
        Args:
            image: PIL Image
            metadata: Image metadata
            
        Returns:
            Sharpness score (0-1)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # Compute Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Calculate variance
        variance = laplacian.var()
        
        # Normalize using sigmoid function
        # This maps the variance to a 0-1 range with most values between 0.2-0.9
        score = 1.0 / (1.0 + np.exp(-self.sigmoid_scale * variance))
        
        return round(float(score), 5)


class ColorDiversityMetric(BaseQualityMetric):
    """
    Color diversity metric.
    
    This metric measures the diversity of colors in an image. Images with
    good color diversity typically contain more visual information and are
    less likely to be corrupted or poorly exposed.
    """
    
    def __init__(self, normalize_factor: float = 50.0):
        """
        Initialize color diversity metric.
        
        Args:
            normalize_factor: Factor for normalizing color standard deviation
        """
        super().__init__("color_diversity")
        self.normalize_factor = normalize_factor
    
    @property
    def metric_name(self) -> str:
        return "color_score"
    
    def compute(self, image: Image.Image, metadata: Dict[str, Any]) -> float:
        """
        Compute color diversity score.
        
        This metric calculates the standard deviation of pixel values across
        color channels. Higher diversity indicates richer color information.
        
        Args:
            image: PIL Image
            metadata: Image metadata
            
        Returns:
            Color diversity score (0-1)
        """
        # Convert to numpy array
        pixels = np.array(image)
        
        # Handle grayscale images
        if len(pixels.shape) == 2:
            pixels = np.stack([pixels] * 3, axis=-1)
        
        # Reshape to list of pixels
        pixels = pixels.reshape(-1, 3)
        
        # Calculate standard deviation for each channel
        std_per_channel = np.std(pixels, axis=0)
        
        # Average across channels
        avg_std = np.mean(std_per_channel)
        
        # Normalize to 0-1 range
        score = min(avg_std / self.normalize_factor, 1.0)
        
        return round(float(score), 5)


class SemanticCaptionGuidedQualityMetric(BaseQualityMetric):
    """
    Semantic caption-guided quality metric using EfficientNet-B0.
    
    Uses semantic similarity between image captions and ImageNet class names
    to identify relevant classes, then focuses EfficientNet quality assessment
    on those classes only. Works universally across all datasets with captions.
    """
    
    def __init__(self):
        """Initialize semantic caption-guided quality metric."""
        super().__init__("semantic_caption_guided_quality")
        
        # Force CPU usage for data retrieval performance
        self.device = "cpu"
        
        # Load EfficientNet-B0 for image classification (CPU)
        try:
            from torchvision.models import efficientnet_b0
            self.efficientnet = efficientnet_b0(pretrained=True)
            self.efficientnet.eval()
        except ImportError:
            raise ImportError("torchvision is required for EfficientNet model")
        
        # Load sentence transformer for semantic similarity (lightweight)
        try:
            from sentence_transformers import SentenceTransformer
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')  # 22MB, fast
        except ImportError:
            raise ImportError("sentence-transformers is required for semantic similarity")
        
        # Load ImageNet class names
        try:
            from .imagenet_classes import IMAGENET_CLASSES
            self.imagenet_classes = IMAGENET_CLASSES
        except ImportError:
            raise ImportError("ImageNet classes not found")
        
        # Pre-compute embeddings for all ImageNet classes (one-time cost at init)
        self.log_info("Computing ImageNet class embeddings for semantic similarity...")
        self.class_embeddings = self.sentence_model.encode(self.imagenet_classes, show_progress_bar=False)
        
        # Image preprocessing (EfficientNet standard)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @property
    def metric_name(self) -> str:
        return "composite_quality"
    
    @property
    def requires_reference(self) -> bool:
        return False
    
    def _find_relevant_imagenet_classes(self, caption: str, top_k: int = 15, similarity_threshold: float = 0.25):
        """
        Find ImageNet classes most semantically similar to the caption.
        
        Args:
            caption: Image caption text
            top_k: Maximum number of classes to consider
            similarity_threshold: Minimum similarity score to include class
            
        Returns:
            tuple: (class_indices, similarity_scores)
        """
        try:
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            raise ImportError("scikit-learn is required for cosine similarity")
        
        # Encode caption
        caption_embedding = self.sentence_model.encode([caption], show_progress_bar=False)
        
        # Compute cosine similarities with all ImageNet class names
        similarities = cosine_similarity(caption_embedding, self.class_embeddings)[0]
        
        # Get top-K most similar classes above threshold
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Filter by threshold
        relevant_indices = []
        relevant_similarities = []
        for idx in top_indices:
            if similarities[idx] >= similarity_threshold:
                relevant_indices.append(idx)
                relevant_similarities.append(similarities[idx])
        
        return relevant_indices, relevant_similarities
    
    def compute(self, image: Image.Image, metadata: Dict[str, Any]) -> float:
        """
        Compute caption-guided image quality using semantic similarity.
        
        Args:
            image: PIL Image
            metadata: Must contain 'text' or 'caption' field for optimal results
            
        Returns:
            Composite quality score (0-1)
        """
        try:
            # Get caption
            caption = metadata.get('text', metadata.get('caption', ''))
            
            # Process image with EfficientNet
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            
            with torch.no_grad():
                logits = self.efficientnet(image_tensor)
                probabilities = torch.softmax(logits, dim=1).squeeze()
            
            # Standard quality metrics
            max_confidence = probabilities.max().item()
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8))
            clarity_score = 1.0 - (entropy / torch.log(torch.tensor(1000.0))).item()
            
            # Confidence gap (distinctiveness)
            sorted_probs = torch.sort(probabilities, descending=True)[0]
            confidence_gap = (sorted_probs[0] - sorted_probs[1]).item()
            
            if caption and len(caption.strip()) >= 5:  # Valid caption exists
                # Find semantically relevant ImageNet classes
                relevant_classes, similarities = self._find_relevant_imagenet_classes(caption)
                
                if relevant_classes:
                    # Filter relevant classes to only include valid EfficientNet indices (0-999)
                    valid_classes = []
                    valid_similarities = []
                    filtered_count = 0
                    
                    for i, class_idx in enumerate(relevant_classes):
                        if 0 <= class_idx < len(probabilities):  # EfficientNet has 1000 classes (0-999)
                            valid_classes.append(class_idx)
                            valid_similarities.append(similarities[i])
                        else:
                            filtered_count += 1
                    
                    # Log debug info if classes were filtered out
                    if filtered_count > 0:
                        self.log_debug(f"Filtered out {filtered_count} classes beyond EfficientNet bounds (≥1000)")  
                    
                    if valid_classes:
                        # Focus on caption-relevant classes within EfficientNet bounds
                        relevant_probs = probabilities[valid_classes]
                        similarity_weights = torch.tensor(valid_similarities, dtype=torch.float32)
                        
                        # Semantic-weighted confidence
                        semantic_weighted_confidence = (relevant_probs * similarity_weights).sum() / similarity_weights.sum()
                        max_relevant_confidence = relevant_probs.max().item()
                        
                        # Caption-image alignment score (best semantic match from valid classes)
                        alignment_score = valid_similarities[0] if valid_similarities else 0.0
                        
                        # Composite quality emphasizing caption alignment
                        composite_quality = (
                            semantic_weighted_confidence.item() * 0.4 +  # Semantic-weighted EfficientNet confidence
                            max_relevant_confidence * 0.3 +              # Best relevant class confidence
                            clarity_score * 0.2 +                       # General image clarity
                            alignment_score * 0.1                       # Caption-ImageNet semantic alignment
                        )
                        
                        return round(composite_quality, 5)
            
            # Fallback for missing/poor captions - general quality assessment
            composite_quality = max_confidence * 0.5 + clarity_score * 0.3 + confidence_gap * 0.2
            
            return round(composite_quality, 5)
            
        except Exception as e:
            # Log more specific error information for debugging
            self.log_debug(f"Error computing semantic caption-guided quality: {e}")
            # Only log as warning if it's an unexpected error type
            if not isinstance(e, (IndexError, ValueError)):
                self.log_warning(f"Unexpected error in semantic caption-guided quality: {type(e).__name__}: {e}")
            return 0.0


