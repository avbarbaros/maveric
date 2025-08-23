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