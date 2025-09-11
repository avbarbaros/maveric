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
        
        # Pre-compute CLIP embeddings for all ImageNet classes (one-time cost at init)
        self.log_info("Computing ImageNet class embeddings using CLIP...")
        self.class_embeddings = self._compute_imagenet_embeddings()
        
        # Cache for target class mappings (computed on first use)
        self.class_mappings_cache = {}
        
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
    
    def _compute_imagenet_embeddings(self):
        """Compute CLIP text embeddings for all ImageNet classes."""
        import clip
        
        # Prepare text prompts for ImageNet classes
        text_prompts = [f"a photo of a {class_name}" for class_name in self.imagenet_classes]
        
        # Tokenize and encode
        text_tokens = clip.tokenize(text_prompts, truncate=True).to(self.device)
        
        with torch.no_grad():
            text_embeddings = self.clip_model.encode_text(text_tokens)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        
        return text_embeddings.cpu().numpy()
    
    def _find_relevant_imagenet_classes_for_target_class(self, target_class: str, top_k: int = 20, similarity_threshold: float = 0.3):
        """
        Find ImageNet classes relevant to a target dataset class using CLIP.
        
        Args:
            target_class: Target class name (e.g., 'dog', 'automobile', 'bird')
            top_k: Maximum number of relevant classes to return
            similarity_threshold: Minimum similarity score to include class
            
        Returns:
            List of ImageNet class indices relevant to the target class
        """
        # Check cache first
        cache_key = f"{target_class}_{top_k}_{similarity_threshold}"
        if cache_key in self.class_mappings_cache:
            return self.class_mappings_cache[cache_key]
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            import clip
        except ImportError:
            raise ImportError("scikit-learn and openai-clip are required")
        
        # Encode target class using CLIP
        target_prompt = f"a photo of a {target_class}"
        target_tokens = clip.tokenize([target_prompt], truncate=True).to(self.device)
        
        with torch.no_grad():
            target_embedding = self.clip_model.encode_text(target_tokens)
            target_embedding = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
        
        target_embedding = target_embedding.cpu().numpy()
        
        # Compute similarities with all ImageNet classes
        similarities = cosine_similarity(target_embedding, self.class_embeddings)[0]
        
        # Get top-K most similar classes above threshold
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Filter by threshold
        relevant_indices = []
        for idx in top_indices:
            if similarities[idx] >= similarity_threshold:
                relevant_indices.append(idx)
        
        # Cache result for future use
        self.class_mappings_cache[cache_key] = relevant_indices
        
        # Log the discovered mapping for transparency
        if relevant_indices:
            class_names = [self.imagenet_classes[i] for i in relevant_indices[:5]]  # Show top 5
            self.log_info(f"Target class '{target_class}' → ImageNet classes: {class_names}... (CLIP)")
        
        return relevant_indices
    
    
    def compute(self, image: Image.Image, metadata: Dict[str, Any]) -> float:
        """
        Compute target class quality using pre-computed CLIP mappings.
        
        Algorithm:
        1. Get target class from metadata 
        2. Find relevant ImageNet classes using pre-computed CLIP mapping
        3. Run EfficientNet on image
        4. Return max probability among relevant ImageNet classes
        
        Args:
            image: PIL Image
            metadata: Must contain 'label' or 'target_class'
            
        Returns:
            Quality score (0-1): Max EfficientNet probability of relevant classes
        """
        try:
            # Get target class
            target_class = metadata.get('label', metadata.get('target_class', ''))
            
            if not target_class or len(target_class.strip()) == 0:
                self.log_debug("No target class provided, returning 0.0")
                return 0.0
            
            # Process image with EfficientNet
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            
            with torch.no_grad():
                logits = self.efficientnet(image_tensor)
                probabilities = torch.softmax(logits, dim=1).squeeze()
            
            # Find ImageNet classes relevant to target class using CLIP
            relevant_classes = self._find_relevant_imagenet_classes_for_target_class(target_class)
            
            if not relevant_classes:
                self.log_debug(f"No relevant ImageNet classes found for '{target_class}'")
                return 0.0
            
            # Get max probability among relevant classes
            relevant_probs = probabilities[relevant_classes]
            quality_score = relevant_probs.max().item()
            
            # Log for transparency
            best_idx = relevant_classes[np.argmax(relevant_probs.cpu().numpy())]
            best_class_name = self.imagenet_classes[best_idx]
            self.log_debug(f"Target '{target_class}' → Best: '{best_class_name}' (prob: {quality_score:.3f})")
            
            return round(quality_score, 5)
            
        except Exception as e:
            self.log_debug(f"Error computing target class quality: {e}")
            if not isinstance(e, (IndexError, ValueError)):
                self.log_warning(f"Unexpected error in target class quality: {type(e).__name__}: {e}")
            return 0.0