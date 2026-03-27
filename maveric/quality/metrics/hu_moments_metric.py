"""Hu invariant moments similarity metric for shape-based image scoring."""

import cv2
import numpy as np
from PIL import Image
from typing import Any, Dict, Optional

from .base_metric import BaseQualityMetric


class HuMomentsSimilarityMetric(BaseQualityMetric):
    """
    Hu invariant moments similarity metric.

    Computes the first 2 Hu invariant moments (h1, h2) for an image and
    calculates similarity to reference images using Euclidean distance on
    log-transformed moment vectors.

    Uses only h1 and h2 for better stability and discriminability:
    - h1: Area-related feature
    - h2: Aspect ratio feature
    Higher moments (h3-h7) are too noisy for natural images.

    Hu moments have translation, rotation, and scale invariance,
    making them suitable for shape-based image matching.

    Reference:
        Wu et al., "Application of image retrieval based on CNN and
        Hu invariant moment algorithm," Computer Communications, 2020.
    """

    def __init__(self):
        super().__init__("hu_moments_similarity")
        self._reference_hu_vectors = {}  # {class_name: np.array shape (N, 2)}

    @property
    def metric_name(self) -> str:
        return "hu_similarity"

    @property
    def requires_reference(self) -> bool:
        return True

    @staticmethod
    def compute_hu_vector(image: Image.Image) -> Optional[np.ndarray]:
        """
        Compute log-transformed Hu invariant moments for a PIL Image.

        Uses only the first 2 Hu moments (h1, h2) for better stability:
        - h1: Area-related (captures overall size/complexity)
        - h2: Aspect ratio (captures shape elongation)

        Higher moments (h3-h7) are excluded as they are too noisy and
        numerically unstable for natural images.

        Steps:
        1. Convert to grayscale
        2. Compute cv2.moments()
        3. Compute cv2.HuMoments() (all 7)
        4. Apply log transform: -sign(h) * log10(|h| + epsilon)
        5. Return only first 2 moments

        Args:
            image: PIL Image (RGB or grayscale)

        Returns:
            np.ndarray of shape (2,) with log-transformed first 2 Hu moments,
            or None if computation fails
        """
        try:
            # Convert PIL to numpy
            img_array = np.array(image)

            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            # Compute moments
            moments = cv2.moments(gray)

            # Compute 7 Hu invariant moments
            hu_moments = cv2.HuMoments(moments).flatten()  # shape (7,)

            # Log transform to compress dynamic range
            # Use -sign(h) * log10(|h| + epsilon)
            # Increased epsilon for better numerical stability
            epsilon = 1e-10
            log_hu = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + epsilon)

            # Use only first 2 moments (most stable and discriminative)
            # h1: area-related, h2: aspect ratio
            # Higher moments (h3-h7) are too noisy for natural images
            return log_hu[:2]  # shape (2,) instead of (7,)

        except Exception:
            return None

    def set_reference_vectors(self, reference_vectors: Dict[str, np.ndarray]):
        """
        Set pre-computed reference Hu vectors for each class.

        Args:
            reference_vectors: {class_name: np.array of shape (N, 2)}
                              N = number of reference images (typically 10)
                              2 = first 2 Hu moments (h1, h2)
        """
        self._reference_hu_vectors = reference_vectors

    def compute_similarity(self, hu_vector: np.ndarray, class_name: str) -> float:
        """
        Compute similarity between a Hu vector and reference vectors of a class.

        Uses Euclidean distance on log-transformed vectors in 2D space (h1, h2),
        converted to similarity via 1 / (1 + distance). Takes best match across
        all reference images (minimum distance).

        Args:
            hu_vector: Log-transformed Hu vector of shape (2,) containing h1, h2
            class_name: Target class name

        Returns:
            Similarity score in (0, 1], where 1.0 = identical, ~0.5 = similar, <0.1 = dissimilar
        """
        if class_name not in self._reference_hu_vectors:
            return 0.0

        ref_vectors = self._reference_hu_vectors[class_name]  # (N, 7)

        # Euclidean distances to all reference vectors
        distances = np.linalg.norm(ref_vectors - hu_vector, axis=1)

        # Convert min distance to similarity: 1 / (1 + d)
        min_distance = np.min(distances)
        similarity = 1.0 / (1.0 + min_distance)

        return round(float(similarity), 5)

    def compute(self, image: Image.Image, metadata: Dict[str, Any]) -> float:
        """
        Compute Hu similarity for the best-matching class.

        Args:
            image: PIL Image
            metadata: Must contain 'label' or 'target_class'

        Returns:
            Hu similarity score
        """
        target_class = metadata.get('label', metadata.get('target_class', ''))
        if not target_class:
            return 0.0

        hu_vector = self.compute_hu_vector(image)
        if hu_vector is None:
            return 0.0

        return self.compute_similarity(hu_vector, target_class)

    def compute_all_class_similarities(self, hu_vector: np.ndarray) -> Dict[str, float]:
        """
        Compute similarity to ALL reference classes at once.

        Args:
            hu_vector: Log-transformed Hu vector of shape (2,) containing h1, h2

        Returns:
            {class_name: similarity_score}
        """
        results = {}
        for class_name in self._reference_hu_vectors:
            results[class_name] = self.compute_similarity(hu_vector, class_name)
        return results
