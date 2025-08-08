"""Visual quality metrics for images."""

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from typing import Any, Dict, Optional

from .base_metric import BaseQualityMetric


class ResolutionMetric(BaseQualityMetric):
    """
    Resolution quality metric.
    
    This metric evaluates image resolution relative to the expected input size
    for vision models. Higher resolution images generally contain more detail
    and provide better training signals.
    """
    
    def __init__(self, target_size: int = 224, min_size: int = 32):
        """
        Initialize resolution metric.
        
        Args:
            target_size: Target resolution for vision models
            min_size: Minimum acceptable resolution
        """
        super().__init__("resolution")
        self.target_size = target_size
        self.min_size = min_size
    
    @property
    def metric_name(self) -> str:
        return "resolution_score"
    
    def compute(self, image: Image.Image, metadata: Dict[str, Any]) -> float:
        """
        Compute resolution score.
        
        The score is based on the smallest dimension relative to the target size,
        ensuring that both width and height meet quality standards.
        
        Args:
            image: PIL Image
            metadata: Image metadata
            
        Returns:
            Resolution score (0-1, can exceed 1 for high-res images)
        """
        width, height = image.size
        min_dimension = min(width, height)
        
        # Penalize very small images
        if min_dimension < self.min_size:
            return 0.0
        
        # Score based on ratio to target size
        score = min_dimension / self.target_size
        
        # Apply soft cap to prevent extremely high scores
        if score > 2.0:
            score = 2.0 + np.log(score - 1.0)
        
        return float(score)


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
        
        return float(score)


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
        
        return float(score)


class FeatureRichnessMetric(BaseQualityMetric):
    """
    Feature richness metric using pre-trained neural network.
    
    This metric uses a pre-trained ResNet to extract features and measures
    their diversity. Images that activate diverse features are considered
    to have richer content and be more useful for training.
    """
    
    def __init__(self, model_name: str = "resnet50", device: str = "cuda"):
        """
        Initialize feature richness metric.
        
        Args:
            model_name: Pre-trained model to use
            device: Device for computation
        """
        super().__init__("feature_richness")
        
        # Set device
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Load pre-trained model
        self.model = resnet50(pretrained=True).to(self.device)
        self.model.eval()
        
        # Remove final classification layer to get features
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),              # Resize to 256x256
            transforms.CenterCrop(224),          # Crop center 224x224 (ResNet input size)
            transforms.ToTensor(),               # Convert PIL to tensor
            transforms.Normalize(                # Normalize with ImageNet stats
                mean=[0.485, 0.456, 0.406],      # ImageNet mean values
                std=[0.229, 0.224, 0.225]        # ImageNet standard deviation
            )
        ])
    
    @property
    def metric_name(self) -> str:
        return "feature_score"
    
    @property
    def requires_reference(self) -> bool:
        return False
    
    def compute(self, image: Image.Image, metadata: Dict[str, Any]) -> float:
        """
        Compute feature richness score.
        
        This extracts features using a pre-trained network and measures their
        standard deviation and mean activation. Higher values indicate more
        distinctive and diverse features.
        
        Args:
            image: PIL Image
            metadata: Image metadata
            
        Returns:
            Feature richness score (0-1)
        """
        try:
            # Ensure RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Preprocess image
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(img_tensor)
            
            # Calculate statistics
            feature_std = features.std().item()
            feature_mean = features.abs().mean().item()
            
            # Combine statistics
            # We use both std (diversity) and mean (activation strength)
            combined_score = (feature_std * 0.5 + feature_mean * 0.5)
            
            # Normalize to 0-1 range (based on empirical observations)
            normalized_score = min(combined_score / 2.0, 1.0)
            
            return float(normalized_score)
            
        except Exception as e:
            self.log_warning(f"Error computing feature richness: {e}")
            return 0.0