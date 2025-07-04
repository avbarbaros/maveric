"""Semantic quality metrics for text and captions."""

import re
from typing import Any, Dict, List
from langdetect import detect, LangDetectException

from .base_metric import BaseQualityMetric


class TextQualityMetric(BaseQualityMetric):
    """
    Text quality metric for captions.
    
    This metric evaluates the quality of text captions based on various
    linguistic features such as length, vocabulary, and language detection.
    """
    
    def __init__(self, 
                 min_words: int = 3,
                 max_words: int = 100,
                 target_language: str = "en"):
        """
        Initialize text quality metric.
        
        Args:
            min_words: Minimum acceptable word count
            max_words: Maximum acceptable word count
            target_language: Expected language code
        """
        super().__init__("text_quality")
        self.min_words = min_words
        self.max_words = max_words
        self.target_language = target_language
    
    @property
    def metric_name(self) -> str:
        return "text_quality_score"
    
    def compute(self, image: Image.Image, metadata: Dict[str, Any]) -> float:
        """
        Compute text quality score.
        
        This evaluates caption quality based on length, vocabulary diversity,
        and language detection.
        
        Args:
            image: PIL Image (not used for text metrics)
            metadata: Must contain 'text' or 'caption' field
            
        Returns:
            Text quality score (0-1)
        """
        # Get caption text
        caption = metadata.get('text', metadata.get('caption', ''))
        
        if not caption:
            return 0.0
        
        # Clean text
        caption = self._clean_text(caption)
        
        # Calculate components
        length_score = self._calculate_length_score(caption)
        vocabulary_score = self._calculate_vocabulary_score(caption)
        language_score = self._calculate_language_score(caption)
        
        # Combine scores
        combined_score = (
            length_score * 0.4 +
            vocabulary_score * 0.3 +
            language_score * 0.3
        )
        
        return float(combined_score)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()
    
    def _calculate_length_score(self, text: str) -> float:
        """Calculate score based on text length."""
        words = text.split()
        word_count = len(words)
        
        if word_count < self.min_words:
            return word_count / self.min_words
        elif word_count > self.max_words:
            return max(0.0, 1.0 - (word_count - self.max_words) / self.max_words)
        else:
            # Optimal length range
            return 1.0
    
    def _calculate_vocabulary_score(self, text: str) -> float:
        """Calculate vocabulary diversity score."""
        words = text.lower().split()
        if not words:
            return 0.0
        
        # Type-token ratio
        unique_words = set(words)
        diversity_ratio = len(unique_words) / len(words)
        
        # Adjust for text length (shorter texts naturally have higher ratios)
        length_factor = min(1.0, len(words) / 20.0)
        adjusted_score = diversity_ratio * length_factor
        
        return min(adjusted_score, 1.0)
    
    def _calculate_language_score(self, text: str) -> float:
        """Calculate language detection score."""
        try:
            detected_lang = detect(text)
            return 1.0 if detected_lang == self.target_language else 0.5
        except LangDetectException:
            # If language detection fails, give partial credit
            return 0.7


class CaptionLengthMetric(BaseQualityMetric):
    """
    Simple caption length metric.
    
    This is a simplified metric that only considers caption length,
    useful for quick filtering of too-short or too-long captions.
    """
    
    def __init__(self, min_length: int = 10, optimal_length: int = 50):
        """
        Initialize caption length metric.
        
        Args:
            min_length: Minimum character length
            optimal_length: Optimal character length
        """
        super().__init__("caption_length")
        self.min_length = min_length
        self.optimal_length = optimal_length
    
    @property
    def metric_name(self) -> str:
        return "caption_length_score"
    
    def compute(self, image: Image.Image, metadata: Dict[str, Any]) -> float:
        """
        Compute caption length score.
        
        Args:
            image: PIL Image (not used)
            metadata: Must contain 'text' or 'caption'
            
        Returns:
            Length score (0-1)
        """
        caption = metadata.get('text', metadata.get('caption', ''))
        length = len(caption.strip())
        
        if length < self.min_length:
            return length / self.min_length
        elif length <= self.optimal_length * 2:
            return 1.0
        else:
            # Penalize very long captions
            return max(0.0, 1.0 - (length - self.optimal_length * 2) / (self.optimal_length * 2))