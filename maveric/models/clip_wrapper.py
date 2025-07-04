"""CLIP model wrapper for MAVERIC."""

import torch
import clip
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image

from ..core.base import BaseComponent
from ..core.exceptions import ModelError


class CLIPWrapper(BaseComponent):
    """
    Wrapper for CLIP models providing a unified interface.
    
    This wrapper handles different CLIP implementations and provides
    consistent methods for encoding images and text.
    """
    
    def __init__(self, 
                 model_name: str = "ViT-B/32",
                 device: str = "cuda",
                 jit: bool = False):
        """
        Initialize CLIP wrapper.
        
        Args:
            model_name: CLIP model variant
            device: Device for computation
            jit: Whether to use JIT-compiled model
        """
        super().__init__("CLIPWrapper")
        
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.jit = jit
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load CLIP model and preprocessing."""
        try:
            self.log_info(f"Loading CLIP model: {self.model_name}")
            self.model, self.preprocess = clip.load(
                self.model_name,
                device=self.device,
                jit=self.jit
            )
            self.model.eval()
            
            # Get model properties
            self.embedding_dim = self.model.visual.output_dim
            self.image_resolution = self.model.visual.input_resolution
            
        except Exception as e:
            raise ModelError(f"Failed to load CLIP model: {e}")
    
    def encode_image(self, 
                     images: Union[Image.Image, List[Image.Image], torch.Tensor],
                     normalize: bool = True) -> np.ndarray:
        """
        Encode images to embeddings.
        
        Args:
            images: Single image, list of images, or tensor
            normalize: Whether to normalize embeddings
            
        Returns:
            Numpy array of embeddings
        """
        # Handle single image
        if isinstance(images, Image.Image):
            images = [images]
        
        # Preprocess if needed
        if isinstance(images, list):
            # Process PIL images
            image_tensors = []
            for img in images:
                img_tensor = self.preprocess(img).unsqueeze(0)
                image_tensors.append(img_tensor)
            image_input = torch.cat(image_tensors, dim=0).to(self.device)
        else:
            # Already a tensor
            image_input = images.to(self.device)
        
        # Encode
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            
            if normalize:
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()
    
    def encode_text(self,
                    texts: Union[str, List[str]],
                    normalize: bool = True) -> np.ndarray:
        """
        Encode text to embeddings.
        
        Args:
            texts: Single text or list of texts
            normalize: Whether to normalize embeddings
            
        Returns:
            Numpy array of embeddings
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        tokens = clip.tokenize(texts, truncate=True).to(self.device)
        
        # Encode
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            
            if normalize:
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()
    
    def compute_similarity(self,
                          images: Union[Image.Image, List[Image.Image]],
                          texts: Union[str, List[str]],
                          return_probs: bool = False) -> np.ndarray:
        """
        Compute similarity between images and texts.
        
        Args:
            images: Images to compare
            texts: Texts to compare
            return_probs: Whether to return probabilities instead of logits
            
        Returns:
            Similarity matrix (images x texts)
        """
        # Encode both modalities
        image_features = self.encode_image(images, normalize=True)
        text_features = self.encode_text(texts, normalize=True)
        
        # Convert back to tensors for computation
        image_features = torch.from_numpy(image_features).to(self.device)
        text_features = torch.from_numpy(text_features).to(self.device)
        
        # Compute similarity
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * (image_features @ text_features.T)
        
        if return_probs:
            probs = logits.softmax(dim=-1)
            return probs.cpu().numpy()
        else:
            return logits.cpu().numpy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'embedding_dim': self.embedding_dim,
            'image_resolution': self.image_resolution,
            'jit': self.jit,
            'available_models': clip.available_models()
        }
    
    def create_text_classifier(self, 
                              class_names: List[str],
                              templates: Optional[List[str]] = None) -> 'TextClassifier':
        """
        Create a zero-shot text classifier.
        
        Args:
            class_names: Names of classes
            templates: Text templates (uses defaults if None)
            
        Returns:
            TextClassifier instance
        """
        return TextClassifier(self, class_names, templates)


class TextClassifier:
    """Zero-shot text classifier using CLIP."""
    
    def __init__(self,
                 clip_model: CLIPWrapper,
                 class_names: List[str],
                 templates: Optional[List[str]] = None):
        """
        Initialize text classifier.
        
        Args:
            clip_model: CLIP model wrapper
            class_names: Class names
            templates: Text templates
        """
        self.clip_model = clip_model
        self.class_names = class_names
        
        # Default templates
        if templates is None:
            templates = [
                "a photo of a {}",
                "a bad photo of a {}",
                "a origami {}",
                "a photo of the large {}",
                "a {} in a video game",
                "art of a {}",
                "a photo of the small {}"
            ]
        self.templates = templates
        
        # Create text embeddings
        self._create_text_embeddings()
    
    def _create_text_embeddings(self):
        """Create text embeddings for all classes."""
        all_embeddings = []
        
        for class_name in self.class_names:
            # Create prompts
            prompts = [template.format(class_name) for template in self.templates]
            
            # Encode
            embeddings = self.clip_model.encode_text(prompts, normalize=True)
            
            # Average across templates
            mean_embedding = embeddings.mean(axis=0)
            mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
            
            all_embeddings.append(mean_embedding)
        
        self.text_embeddings = np.stack(all_embeddings)
    
    def classify(self, 
                 images: Union[Image.Image, List[Image.Image]],
                 top_k: int = 1) -> Union[str, List[str], Dict[str, float]]:
        """
        Classify images.
        
        Args:
            images: Images to classify
            top_k: Number of top predictions to return
            
        Returns:
            Predictions (format depends on top_k)
        """
        # Encode images
        image_embeddings = self.clip_model.encode_image(images, normalize=True)
        
        # Compute similarities
        similarities = image_embeddings @ self.text_embeddings.T
        
        # Get predictions
        if top_k == 1:
            # Return single prediction
            indices = similarities.argmax(axis=1)
            if len(indices) == 1:
                return self.class_names[indices[0]]
            else:
                return [self.class_names[idx] for idx in indices]
        else:
            # Return top-k predictions with scores
            results = []
            for sim in similarities:
                top_indices = sim.argsort()[-top_k:][::-1]
                top_scores = sim[top_indices]
                
                pred_dict = {
                    self.class_names[idx]: float(score)
                    for idx, score in zip(top_indices, top_scores)
                }
                results.append(pred_dict)
            
            return results[0] if len(results) == 1 else results
