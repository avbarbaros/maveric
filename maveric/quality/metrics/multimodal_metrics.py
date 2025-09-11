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
    Simplified semantic caption-guided quality metric using EfficientNet-B0.
    
    This multimodal metric uses semantic similarity to find ImageNet classes
    relevant to the caption, then returns the maximum EfficientNet probability
    among those relevant classes. Simple and interpretable: "How confident is
    the model that this image contains what the caption describes?"
    
    Works universally across all datasets with captions.
    """
    
    def __init__(self, target_dataset: str = None):
        """Initialize semantic caption-guided quality metric.
        
        Args:
            target_dataset: Target dataset name (e.g., 'cifar10', 'cifar100') for class-specific mapping
        """
        super().__init__("semantic_caption_guided_quality")
        
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
        
        # Cache for dynamic class mappings (computed on first use)
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
        return "composite_quality"
    
    @property
    def requires_reference(self) -> bool:
        return False
    
    def _find_relevant_imagenet_classes_for_target_class(self, target_class: str, top_k: int = 20, similarity_threshold: float = 0.3):
        """
        Dynamically find ImageNet classes relevant to a target dataset class.
        
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
        except ImportError:
            raise ImportError("scikit-learn is required for cosine similarity")
        
        # Encode target class name
        target_embedding = self.sentence_model.encode([target_class], show_progress_bar=False)
        
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
            self.log_info(f"Target class '{target_class}' mapped to ImageNet classes: {class_names}...")
        
        return relevant_indices
    
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
        Compute class-centric caption-guided image quality.
        
        Algorithm:
        1. If target_class is provided in metadata, use class-centric approach:
           - Find ImageNet classes relevant to target class
           - Calculate caption similarity with relevant classes
           - Return max probability among most similar classes
        2. Otherwise, fall back to caption-centric approach
        
        Args:
            image: PIL Image
            metadata: Must contain 'text'/'caption' and optionally 'label'/'target_class'
            
        Returns:
            Quality score (0-1): Max probability of relevant classes
        """
        try:
            # Get caption and target class
            caption = metadata.get('text', metadata.get('caption', ''))
            target_class = metadata.get('label', metadata.get('target_class', ''))
            
            # Process image with EfficientNet
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            
            with torch.no_grad():
                logits = self.efficientnet(image_tensor)
                probabilities = torch.softmax(logits, dim=1).squeeze()
            
            # Get max confidence for fallback case
            max_confidence = probabilities.max().item()
            
            # CLASS-CENTRIC APPROACH (Your improved idea!)
            if target_class and len(target_class.strip()) > 0:
                # Step 1: Find ImageNet classes relevant to target class
                target_relevant_classes = self._find_relevant_imagenet_classes_for_target_class(target_class)
                
                if target_relevant_classes and caption and len(caption.strip()) >= 5:
                    # Step 2: Calculate similarity between caption and target-relevant ImageNet classes
                    target_class_names = [self.imagenet_classes[i] for i in target_relevant_classes]
                    target_embeddings = self.sentence_model.encode(target_class_names, show_progress_bar=False)
                    caption_embedding = self.sentence_model.encode([caption], show_progress_bar=False)
                    
                    # Calculate similarities
                    from sklearn.metrics.pairwise import cosine_similarity
                    similarities = cosine_similarity(caption_embedding, target_embeddings)[0]
                    
                    # Step 3: Select most similar classes (top 5)
                    top_similarity_indices = np.argsort(similarities)[::-1][:5]
                    most_relevant_classes = [target_relevant_classes[i] for i in top_similarity_indices]
                    
                    # Step 4: Return max probability among most relevant classes
                    if most_relevant_classes:
                        relevant_probs = probabilities[most_relevant_classes]
                        quality_score = relevant_probs.max().item()
                        
                        # Log for transparency
                        best_class_idx = most_relevant_classes[np.argmax(relevant_probs.cpu().numpy())]
                        best_class_name = self.imagenet_classes[best_class_idx]
                        self.log_debug(f"Target '{target_class}' → Best match: '{best_class_name}' (prob: {quality_score:.3f})")
                        
                        return round(quality_score, 5)
                
                # Fallback: use all target-relevant classes without caption filtering
                elif target_relevant_classes:
                    relevant_probs = probabilities[target_relevant_classes]
                    quality_score = relevant_probs.max().item()
                    return round(quality_score, 5)
            
            # CAPTION-CENTRIC FALLBACK (original approach)
            elif caption and len(caption.strip()) >= 5:
                relevant_classes, _ = self._find_relevant_imagenet_classes(caption)
                
                if relevant_classes:
                    # Filter valid classes
                    valid_classes = [idx for idx in relevant_classes if 0 <= idx < len(probabilities)]
                    
                    if valid_classes:
                        relevant_probs = probabilities[valid_classes]
                        quality_score = relevant_probs.max().item()
                        return round(quality_score, 5)
            
            # Final fallback - general image quality
            return round(max_confidence, 5)
            
        except Exception as e:
            self.log_debug(f"Error computing semantic caption-guided quality: {e}")
            if not isinstance(e, (IndexError, ValueError)):
                self.log_warning(f"Unexpected error in semantic caption-guided quality: {type(e).__name__}: {e}")
            return 0.0