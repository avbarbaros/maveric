"""Tests for quality metrics."""

import pytest
import numpy as np
from PIL import Image, ImageFilter

from maveric.quality.metrics import (
    ResolutionMetric,
    SharpnessMetric,
    ColorDiversityMetric,
    TextQualityMetric
)


class TestVisualMetrics:
    """Test visual quality metrics."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        # Create a 224x224 RGB image with some patterns
        img_array = np.zeros((224, 224, 3), dtype=np.uint8)
        # Add some patterns for testing
        img_array[50:150, 50:150] = [255, 0, 0]  # Red square
        img_array[100:200, 100:200] = [0, 255, 0]  # Green square
        return Image.fromarray(img_array)
    
    @pytest.fixture
    def metadata(self):
        """Sample metadata."""
        return {
            'url': 'http://example.com/image.jpg',
            'text': 'A test image with colored squares'
        }
    
    def test_resolution_metric(self, sample_image, metadata):
        """Test resolution metric."""
        metric = ResolutionMetric()
        score = metric.compute(sample_image, metadata)
        
        assert score >= 0
        assert score == 1.0  # 224/224 = 1.0
        
        # Test with smaller image
        small_image = sample_image.resize((112, 112))
        small_score = metric.compute(small_image, metadata)
        assert small_score == 0.5  # 112/224 = 0.5
    
    def test_sharpness_metric(self, sample_image, metadata):
        """Test sharpness metric."""
        metric = SharpnessMetric()
        score = metric.compute(sample_image, metadata)
        
        assert 0 <= score <= 1.0
        
        # Blurred image should have lower score
        blurred = sample_image.filter(ImageFilter.BLUR)
        blurred_score = metric.compute(blurred, metadata)
        assert blurred_score < score
    
    def test_color_diversity_metric(self, sample_image, metadata):
        """Test color diversity metric."""
        metric = ColorDiversityMetric()
        score = metric.compute(sample_image, metadata)
        
        assert 0 <= score <= 1.0
        
        # Uniform color image should have low score
        uniform_image = Image.new('RGB', (224, 224), color='red')
        uniform_score = metric.compute(uniform_image, metadata)
        assert uniform_score < score


class TestSemanticMetrics:
    """Test semantic quality metrics."""
    
    def test_text_quality_metric(self):
        """Test text quality metric."""
        metric = TextQualityMetric(min_words=3, max_words=50)
        
        # Good caption
        good_metadata = {'text': 'A beautiful sunset over the ocean with vibrant colors'}
        good_score = metric.compute(None, good_metadata)
        assert good_score > 0.8
        
        # Too short caption
        short_metadata = {'text': 'Cat'}
        short_score = metric.compute(None, short_metadata)
        assert short_score < 0.5
        
        # Empty caption
        empty_metadata = {'text': ''}
        empty_score = metric.compute(None, empty_metadata)
        assert empty_score == 0.0
