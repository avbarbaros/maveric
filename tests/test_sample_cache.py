"""Tests for sample metadata caching functionality."""

import pytest
import numpy as np
import json
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import shutil

from maveric.retrieval.sample_cache_manager import SampleCacheManager


class TestSampleCacheManager:
    """Test suite for SampleCacheManager class."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def cache_manager(self, temp_cache_dir):
        """Create SampleCacheManager instance."""
        return SampleCacheManager(
            base_dir=temp_cache_dir,
            cache_version=2,
            enabled=True
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample data for caching."""
        return {
            'url': 'https://example.com/image.jpg',
            'text': 'A photo of a cat',
            'visual_metrics': {
                'resolution_score': 0.895,
                'sharpness_score': 0.923,
                'color_score': 0.812
            },
            'semantic_metrics': {
                'text_quality_score': 0.850,
                'caption_length_score': 0.920
            },
            'image_embedding': np.random.randn(512),
            'text_embedding': np.random.randn(512),
            'efficientnet_data': {
                'imagenet_predicted_class': 'tabby cat',
                'imagenet_probability': 0.892
            }
        }

    def test_initialization(self, temp_cache_dir):
        """Test cache manager initialization."""
        cache_manager = SampleCacheManager(
            base_dir=temp_cache_dir,
            cache_version=2,
            enabled=True
        )

        assert cache_manager.enabled is True
        assert cache_manager.cache_version == 2
        assert (Path(temp_cache_dir) / 'sample_metadata_cache').exists()
        assert cache_manager.stats['hits'] == 0
        assert cache_manager.stats['misses'] == 0

    def test_disabled_cache(self, temp_cache_dir):
        """Test cache manager with caching disabled."""
        cache_manager = SampleCacheManager(
            base_dir=temp_cache_dir,
            cache_version=2,
            enabled=False
        )

        # Cache operations should return None/False when disabled
        assert cache_manager.get_cached_sample('http://example.com/test.jpg') is None

        result = cache_manager.cache_sample(
            url='http://example.com/test.jpg',
            text='test',
            visual_metrics={},
            semantic_metrics={},
            image_embedding=np.zeros(512),
            text_embedding=np.zeros(512)
        )
        assert result is False

    def test_cache_miss(self, cache_manager):
        """Test cache miss scenario."""
        url = 'https://example.com/nonexistent.jpg'
        cached = cache_manager.get_cached_sample(url)

        assert cached is None
        assert cache_manager.stats['misses'] == 1
        assert cache_manager.stats['hits'] == 0

    def test_cache_save_and_retrieve(self, cache_manager, sample_data):
        """Test saving and retrieving cached data."""
        # Save to cache
        success = cache_manager.cache_sample(
            url=sample_data['url'],
            text=sample_data['text'],
            visual_metrics=sample_data['visual_metrics'],
            semantic_metrics=sample_data['semantic_metrics'],
            image_embedding=sample_data['image_embedding'],
            text_embedding=sample_data['text_embedding'],
            efficientnet_data=sample_data['efficientnet_data']
        )

        assert success is True
        assert cache_manager.stats['saves'] == 1

        # Retrieve from cache
        cached = cache_manager.get_cached_sample(sample_data['url'])

        assert cached is not None
        assert cached['url'] == sample_data['url']
        assert cached['text'] == sample_data['text']
        assert cached['visual_metrics'] == sample_data['visual_metrics']
        assert cached['semantic_metrics'] == sample_data['semantic_metrics']
        assert 'cache_version' in cached
        assert cached['cache_version'] == 2

        # Check embeddings
        assert 'clip_embeddings' in cached
        cached_img_emb = np.array(cached['clip_embeddings']['image_embedding'])
        cached_txt_emb = np.array(cached['clip_embeddings']['text_embedding'])

        np.testing.assert_array_almost_equal(cached_img_emb, sample_data['image_embedding'])
        np.testing.assert_array_almost_equal(cached_txt_emb, sample_data['text_embedding'])

        # Check EfficientNet data
        assert cached['efficientnet_predictions'] == sample_data['efficientnet_data']

        assert cache_manager.stats['hits'] == 1

    def test_cache_version_invalidation(self, temp_cache_dir, sample_data):
        """Test that old cache versions are invalidated."""
        # Create cache with version 1
        cache_v1 = SampleCacheManager(
            base_dir=temp_cache_dir,
            cache_version=1,
            enabled=True
        )

        # Save data
        cache_v1.cache_sample(
            url=sample_data['url'],
            text=sample_data['text'],
            visual_metrics=sample_data['visual_metrics'],
            semantic_metrics=sample_data['semantic_metrics'],
            image_embedding=sample_data['image_embedding'],
            text_embedding=sample_data['text_embedding']
        )

        # Try to retrieve with version 2 cache manager
        cache_v2 = SampleCacheManager(
            base_dir=temp_cache_dir,
            cache_version=2,
            enabled=True
        )

        cached = cache_v2.get_cached_sample(sample_data['url'])

        # Should be cache miss due to version mismatch
        assert cached is None
        assert cache_v2.stats['misses'] == 1

    def test_caption_mismatch_detection(self, cache_manager, sample_data):
        """Test that caption mismatches are detected."""
        # Save with original caption
        cache_manager.cache_sample(
            url=sample_data['url'],
            text=sample_data['text'],
            visual_metrics=sample_data['visual_metrics'],
            semantic_metrics=sample_data['semantic_metrics'],
            image_embedding=sample_data['image_embedding'],
            text_embedding=sample_data['text_embedding']
        )

        # Retrieve with same caption
        cached = cache_manager.get_cached_sample(sample_data['url'])
        assert cached is not None

        # Verify caption match
        assert cache_manager.verify_caption_match(sample_data['url'], sample_data['text']) is True
        assert cache_manager.verify_caption_match(sample_data['url'], 'Different caption') is False

    def test_hierarchical_cache_structure(self, cache_manager, sample_data):
        """Test that cache uses hierarchical directory structure."""
        # Save sample
        cache_manager.cache_sample(
            url=sample_data['url'],
            text=sample_data['text'],
            visual_metrics=sample_data['visual_metrics'],
            semantic_metrics=sample_data['semantic_metrics'],
            image_embedding=sample_data['image_embedding'],
            text_embedding=sample_data['text_embedding']
        )

        # Check that file is in subdirectory
        cache_path = cache_manager._get_cache_path(sample_data['url'])
        assert cache_path.exists()

        # Verify hierarchical structure (first 2 chars of hash)
        url_hash = cache_manager._get_url_hash(sample_data['url'])
        expected_subdir = cache_manager.cache_dir / url_hash[:2]
        assert cache_path.parent == expected_subdir
        assert cache_path.name == f"sample_{url_hash}_v2.json"

    def test_corrupted_cache_handling(self, cache_manager, sample_data):
        """Test handling of corrupted cache files."""
        # Create corrupted cache file
        cache_path = cache_manager._get_cache_path(sample_data['url'])
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Write invalid JSON
        with open(cache_path, 'w') as f:
            f.write("{ invalid json }")

        # Try to retrieve
        cached = cache_manager.get_cached_sample(sample_data['url'])

        assert cached is None
        assert cache_manager.stats['errors'] == 1

        # Corrupted file should be removed
        assert not cache_path.exists()

    def test_incomplete_cache_entry(self, cache_manager, sample_data):
        """Test handling of incomplete cache entries."""
        # Create cache entry missing required fields
        cache_path = cache_manager._get_cache_path(sample_data['url'])
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        incomplete_data = {
            'cache_version': 2,
            'url': sample_data['url'],
            'visual_metrics': sample_data['visual_metrics']
            # Missing semantic_metrics and clip_embeddings
        }

        with open(cache_path, 'w') as f:
            json.dump(incomplete_data, f)

        # Try to retrieve
        cached = cache_manager.get_cached_sample(sample_data['url'])

        assert cached is None
        assert cache_manager.stats['errors'] == 1

    def test_cache_statistics(self, cache_manager, sample_data):
        """Test cache statistics tracking."""
        # Initial stats
        stats = cache_manager.get_cache_stats()
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['saves'] == 0
        assert stats['hit_rate'] == 0.0
        assert stats['total_samples'] == 0

        # Cache some samples
        for i in range(3):
            cache_manager.cache_sample(
                url=f'https://example.com/image{i}.jpg',
                text=f'Image {i}',
                visual_metrics=sample_data['visual_metrics'],
                semantic_metrics=sample_data['semantic_metrics'],
                image_embedding=np.random.randn(512),
                text_embedding=np.random.randn(512)
            )

        # Retrieve 2 cached samples
        cache_manager.get_cached_sample('https://example.com/image0.jpg')
        cache_manager.get_cached_sample('https://example.com/image1.jpg')

        # Try to retrieve 1 non-existent sample
        cache_manager.get_cached_sample('https://example.com/nonexistent.jpg')

        # Check stats
        stats = cache_manager.get_cache_stats()
        assert stats['hits'] == 2
        assert stats['misses'] == 1
        assert stats['saves'] == 3
        assert stats['hit_rate'] == 2/3  # 2 hits out of 3 total requests
        assert stats['total_samples'] == 3
        assert stats['size_mb'] > 0

    def test_cache_clear_specific_url(self, cache_manager, sample_data):
        """Test clearing cache for specific URL."""
        # Cache sample
        cache_manager.cache_sample(
            url=sample_data['url'],
            text=sample_data['text'],
            visual_metrics=sample_data['visual_metrics'],
            semantic_metrics=sample_data['semantic_metrics'],
            image_embedding=sample_data['image_embedding'],
            text_embedding=sample_data['text_embedding']
        )

        # Verify cached
        assert cache_manager.get_cached_sample(sample_data['url']) is not None

        # Clear specific URL
        deleted = cache_manager.clear_cache(url=sample_data['url'])
        assert deleted == 1

        # Verify removed
        cache_manager.reset_stats()  # Reset stats to not count previous hit
        assert cache_manager.get_cached_sample(sample_data['url']) is None

    def test_cache_clear_all(self, cache_manager, sample_data):
        """Test clearing entire cache."""
        # Cache multiple samples
        for i in range(5):
            cache_manager.cache_sample(
                url=f'https://example.com/image{i}.jpg',
                text=f'Image {i}',
                visual_metrics=sample_data['visual_metrics'],
                semantic_metrics=sample_data['semantic_metrics'],
                image_embedding=np.random.randn(512),
                text_embedding=np.random.randn(512)
            )

        # Clear all
        deleted = cache_manager.clear_cache()
        assert deleted == 5

        # Verify all removed
        stats = cache_manager.get_cache_stats()
        assert stats['total_samples'] == 0

    def test_cache_without_efficientnet(self, cache_manager, sample_data):
        """Test caching without EfficientNet data."""
        # Save without EfficientNet data
        success = cache_manager.cache_sample(
            url=sample_data['url'],
            text=sample_data['text'],
            visual_metrics=sample_data['visual_metrics'],
            semantic_metrics=sample_data['semantic_metrics'],
            image_embedding=sample_data['image_embedding'],
            text_embedding=sample_data['text_embedding'],
            efficientnet_data=None
        )

        assert success is True

        # Retrieve and check
        cached = cache_manager.get_cached_sample(sample_data['url'])
        assert cached is not None
        assert 'efficientnet_predictions' not in cached

    def test_reset_stats(self, cache_manager, sample_data):
        """Test resetting cache statistics."""
        # Generate some cache activity
        cache_manager.cache_sample(
            url=sample_data['url'],
            text=sample_data['text'],
            visual_metrics=sample_data['visual_metrics'],
            semantic_metrics=sample_data['semantic_metrics'],
            image_embedding=sample_data['image_embedding'],
            text_embedding=sample_data['text_embedding']
        )
        cache_manager.get_cached_sample(sample_data['url'])

        assert cache_manager.stats['hits'] == 1
        assert cache_manager.stats['saves'] == 1

        # Reset
        cache_manager.reset_stats()

        assert cache_manager.stats['hits'] == 0
        assert cache_manager.stats['misses'] == 0
        assert cache_manager.stats['errors'] == 0
        assert cache_manager.stats['saves'] == 0

    def test_url_hash_consistency(self, cache_manager):
        """Test that URL hashing is consistent."""
        url = 'https://example.com/test.jpg'

        hash1 = cache_manager._get_url_hash(url)
        hash2 = cache_manager._get_url_hash(url)

        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hash length

    def test_large_embeddings(self, cache_manager):
        """Test caching with large embeddings."""
        # Create large embeddings (e.g., for larger CLIP models)
        large_embedding = np.random.randn(1024)

        success = cache_manager.cache_sample(
            url='https://example.com/large.jpg',
            text='Large embedding test',
            visual_metrics={'resolution_score': 0.9},
            semantic_metrics={'text_quality_score': 0.8},
            image_embedding=large_embedding,
            text_embedding=large_embedding
        )

        assert success is True

        # Retrieve and verify
        cached = cache_manager.get_cached_sample('https://example.com/large.jpg')
        assert cached is not None

        cached_img_emb = np.array(cached['clip_embeddings']['image_embedding'])
        assert len(cached_img_emb) == 1024
        np.testing.assert_array_almost_equal(cached_img_emb, large_embedding)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
