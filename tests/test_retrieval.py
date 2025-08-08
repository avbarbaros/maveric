"""Tests for retrieval module."""

import pytest
from unittest.mock import Mock, patch
import numpy as np
import torch
from PIL import Image

from maveric.retrieval import CacheManager, Retriever
from maveric.retrieval.dataset_handlers import DatasetHandler


class TestCacheManager:
    """Test CacheManager class."""
    
    @pytest.fixture
    def cache_manager(self, tmp_path):
        """Create cache manager with temp directory."""
        return CacheManager(
            base_dir=str(tmp_path),
            enable_image_cache=True
        )
    
    def test_cache_directories(self, cache_manager):
        """Test cache directory creation."""
        assert cache_manager.image_cache_dir.exists()
        assert cache_manager.result_cache_dir.exists()
        assert cache_manager.embedding_cache_dir.exists()
    
    def test_image_caching(self, cache_manager):
        """Test image caching functionality."""
        # Create test image
        test_image = Image.new('RGB', (100, 100), color='red')
        test_url = "http://example.com/test.jpg"
        
        # Cache image
        cache_path = cache_manager.cache_image(test_url, test_image)
        assert cache_path
        
        # Retrieve cached image
        cached_image = cache_manager.get_cached_image(test_url)
        assert cached_image is not None
        assert cached_image.size == test_image.size
        
        # Check cache stats
        stats = cache_manager.get_cache_stats()
        assert stats['cache_hits'] == 1
        assert stats['images_cached'] == 1
    
    def test_results_saving(self, cache_manager):
        """Test saving and loading results."""
        test_results = [
            {'id': 1, 'score': 0.9},
            {'id': 2, 'score': 0.8}
        ]
        
        # Save results
        save_path = cache_manager.save_results(
            test_results,
            dataset_name='test_dataset',
            file_id=1
        )
        
        assert save_path
        
        # Load results
        loaded_results = cache_manager.load_results('test_dataset')
        assert len(loaded_results) == len(test_results)
        assert loaded_results[0]['id'] == 1


class MockDatasetHandler(DatasetHandler):
    """Mock dataset handler for testing."""
    
    def __init__(self, samples):
        super().__init__()
        self.samples = samples
        
    def __iter__(self):
        for sample in self.samples:
            yield sample
            
    def __len__(self):
        return len(self.samples)


class TestRetriever:
    """Test Retriever class."""
    
    @pytest.fixture
    def retriever(self):
        """Create retriever with mocked CLIP model."""
        with patch('maveric.retrieval.retriever.clip.load') as mock_clip:
            # Mock CLIP model
            mock_model = Mock()
            mock_preprocess = Mock()
            mock_clip.return_value = (mock_model, mock_preprocess)
            
            retriever = Retriever(
                clip_model="ViT-B/32",
                device="cpu",
                n_reference_images=2
            )
            
            # Mock encode methods
            retriever.model.encode_image = Mock(
                return_value=torch.tensor([[1.0, 0.0, 0.0]])
            )
            retriever.model.encode_text = Mock(
                return_value=torch.tensor([[0.0, 1.0, 0.0]])
            )
            
            return retriever
    
    @patch('maveric.retrieval.retriever.get_dataset')
    def test_prepare_reference_embeddings(self, mock_get_dataset, retriever):
        """Test preparing reference embeddings."""
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.class_names = ['cat', 'dog']
        mock_dataset.get_reference_samples.return_value = {
            'cat': [Image.new('RGB', (32, 32))],
            'dog': [Image.new('RGB', (32, 32))]
        }
        mock_dataset.get_text_templates.return_value = ["a photo of a {}"]
        mock_get_dataset.return_value = mock_dataset
        
        # Prepare embeddings
        ref_embeddings, text_embeddings = retriever.prepare_reference_embeddings(
            'test_dataset'
        )
        
        assert 'cat' in ref_embeddings
        assert 'dog' in ref_embeddings
        assert 'cat' in text_embeddings
        assert 'dog' in text_embeddings
