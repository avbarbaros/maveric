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




