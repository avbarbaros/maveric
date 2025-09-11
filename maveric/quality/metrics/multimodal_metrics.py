"""Multimodal quality metrics combining vision and language."""

import numpy as np
import torch
import clip
from PIL import Image
from typing import Any, Dict, List, Optional
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


class SemanticCaptionGuidedQualityMetric(BaseQualityMetric):
    """
    Semantic caption-guided quality metric using EfficientNet-B0.
    
    This multimodal metric combines semantic text understanding with visual quality assessment.
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
            import torchvision.transforms as transforms
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
        import torchvision.transforms as transforms
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