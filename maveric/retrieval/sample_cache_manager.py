"""Sample metadata caching for cross-dataset retrieval optimization."""

import json
import hashlib
import base64
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime

from ..core.base import BaseComponent
from ..core.exceptions import CacheError
from ..utils.io_utils import save_json_atomic


class SampleCacheManager(BaseComponent):
    """
    Manages caching of sample metadata and computed metrics.

    This cache stores reusable data that doesn't depend on the target dataset,
    including visual metrics, semantic metrics, CLIP embeddings, and EfficientNet
    predictions. This allows significant speedup when retrieving for multiple
    target datasets from the same source dataset.

    Cache Structure:
        cache_base_dir/
        └── sample_metadata_cache/
            └── {hash[:2]}/
                └── sample_{hash}_v{version}.json

    Cached Data (per sample):
        - Visual metrics (resolution, sharpness, color)
        - Semantic metrics (text quality, caption length)
        - CLIP embeddings (image and text) - base64 encoded numpy arrays
        - EfficientNet predictions (if enabled)

    Performance Impact:
        - First dataset: Same speed (builds cache)
        - Subsequent datasets: 80-95% faster (cache hits, no CLIP inference needed)

    Storage Impact:
        - Per sample: ~17KB (500 bytes metrics + ~16KB embeddings)
        - 270K samples: ~4.5GB (increased from ~135MB without embeddings)
    """

    def __init__(self, base_dir: str, cache_version: int = 3, enabled: bool = True):
        """
        Initialize sample cache manager.

        Args:
            base_dir: Base directory for all caches
            cache_version: Cache format version (increment to invalidate old cache)
            enabled: Whether caching is enabled
        """
        super().__init__()
        self.enabled = enabled
        self.cache_version = cache_version
        self.cache_dir = Path(base_dir) / 'sample_metadata_cache'

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.log_info(f"Sample cache initialized: {self.cache_dir}")

        # Statistics tracking
        self.stats = {
            'hits': 0,
            'misses': 0,
            'errors': 0,
            'saves': 0
        }

    def _get_url_hash(self, url: str) -> str:
        """Generate MD5 hash for URL (same as image cache)."""
        return hashlib.md5(url.encode()).hexdigest()

    def _encode_embedding(self, embedding: np.ndarray) -> str:
        """
        Encode numpy array to base64 string for JSON storage.

        Args:
            embedding: Numpy array to encode

        Returns:
            Base64 encoded string
        """
        # Convert to bytes and encode to base64
        embedding_bytes = embedding.tobytes()
        return base64.b64encode(embedding_bytes).decode('utf-8')

    def _decode_embedding(self, encoded: str, shape: tuple, dtype: str = 'float32') -> np.ndarray:
        """
        Decode base64 string back to numpy array.

        Args:
            encoded: Base64 encoded string
            shape: Original shape of the array
            dtype: Data type of the array

        Returns:
            Decoded numpy array
        """
        # Decode base64 to bytes
        embedding_bytes = base64.b64decode(encoded.encode('utf-8'))
        # Convert bytes to numpy array with original shape
        return np.frombuffer(embedding_bytes, dtype=dtype).reshape(shape)

    def _get_cache_path(self, url: str) -> Path:
        """
        Get cache file path for URL with hierarchical structure.

        Uses same hashing strategy as image cache:
        - MD5(URL) generates hash
        - First 2 hex chars used for subdirectory (256 subdirs)
        - Prevents too many files in single directory

        Args:
            url: Image URL

        Returns:
            Path to cache file
        """
        url_hash = self._get_url_hash(url)
        subdir = self.cache_dir / url_hash[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"sample_{url_hash}_v{self.cache_version}.json"

    def get_cached_sample(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached sample metadata.

        Args:
            url: Image URL

        Returns:
            Cached data dictionary or None if not cached/invalid

        Cached data structure:
            {
                'cache_version': int,
                'url': str,
                'url_hash': str,
                'text': str,
                'last_updated': str (ISO format),
                'visual_metrics': {...},
                'semantic_metrics': {...},
                'clip_embeddings': {
                    'image_embedding': base64 string,
                    'text_embedding': base64 string,
                    'image_shape': tuple,
                    'text_shape': tuple,
                    'dtype': str
                },
                'efficientnet_predictions': {...} (optional)
            }
        """
        if not self.enabled:
            return None

        cache_path = self._get_cache_path(url)

        if not cache_path.exists():
            self.stats['misses'] += 1
            return None

        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)

            # Validate cache version
            if data.get('cache_version') != self.cache_version:
                self.log_debug(f"Cache version mismatch for {url[:50]}, invalidating")
                self.stats['misses'] += 1
                return None

            # Validate required fields
            required_fields = ['visual_metrics', 'semantic_metrics', 'clip_embeddings']
            if not all(field in data for field in required_fields):
                self.log_warning(f"Incomplete cache entry for {url[:50]}, invalidating")
                self.stats['errors'] += 1
                return None

            # Decode CLIP embeddings from base64
            clip_data = data.get('clip_embeddings', {})
            if clip_data:
                try:
                    data['clip_embeddings']['image_embedding'] = self._decode_embedding(
                        clip_data['image_embedding'],
                        tuple(clip_data['image_shape']),
                        clip_data.get('dtype', 'float32')
                    )
                    data['clip_embeddings']['text_embedding'] = self._decode_embedding(
                        clip_data['text_embedding'],
                        tuple(clip_data['text_shape']),
                        clip_data.get('dtype', 'float32')
                    )
                except Exception as e:
                    self.log_warning(f"Failed to decode embeddings for {url[:50]}: {e}")
                    self.stats['errors'] += 1
                    return None

            self.stats['hits'] += 1
            return data

        except json.JSONDecodeError as e:
            self.log_warning(f"Corrupted cache for {url[:50]}: {e}")
            self.stats['errors'] += 1
            # Try to remove corrupted cache
            try:
                cache_path.unlink()
            except:
                pass
            return None
        except Exception as e:
            self.log_warning(f"Failed to load cache for {url[:50]}: {e}")
            self.stats['errors'] += 1
            return None

    def cache_sample(self,
                     url: str,
                     text: str,
                     visual_metrics: Dict[str, float],
                     semantic_metrics: Dict[str, float],
                     image_embedding: Optional[np.ndarray] = None,
                     text_embedding: Optional[np.ndarray] = None,
                     efficientnet_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Cache sample metadata and computed metrics including CLIP embeddings.

        Args:
            url: Image URL (used as cache key)
            text: Caption text
            visual_metrics: Visual quality metrics (resolution, sharpness, color)
            semantic_metrics: Semantic quality metrics (text_quality, caption_length)
            image_embedding: CLIP image embedding (numpy array)
            text_embedding: CLIP text embedding (numpy array)
            efficientnet_data: Optional EfficientNet predictions and metadata

        Returns:
            True if caching succeeded, False otherwise
        """
        if not self.enabled:
            return False

        try:
            cache_path = self._get_cache_path(url)
            url_hash = self._get_url_hash(url)

            # Build cache data structure
            data = {
                'cache_version': self.cache_version,
                'url': url,
                'url_hash': url_hash,
                'text': text,
                'last_updated': datetime.now().isoformat(),
                'visual_metrics': visual_metrics,
                'semantic_metrics': semantic_metrics
            }

            # Add CLIP embeddings if provided (base64 encoded for JSON compatibility)
            if image_embedding is not None and text_embedding is not None:
                data['clip_embeddings'] = {
                    'image_embedding': self._encode_embedding(image_embedding),
                    'text_embedding': self._encode_embedding(text_embedding),
                    'image_shape': list(image_embedding.shape),
                    'text_shape': list(text_embedding.shape),
                    'dtype': str(image_embedding.dtype)
                }

            # Add EfficientNet data if provided
            if efficientnet_data:
                data['efficientnet_predictions'] = efficientnet_data

            # Use atomic write to prevent corruption
            save_json_atomic(data, cache_path, indent=None)  # No indent for smaller files

            self.stats['saves'] += 1
            return True

        except Exception as e:
            self.log_error(f"Failed to cache sample {url[:50]}: {e}")
            self.stats['errors'] += 1
            return False

    def verify_caption_match(self, url: str, text: str) -> bool:
        """
        Verify that cached caption matches current caption.

        This handles cases where the caption might have changed for the same URL.

        Args:
            url: Image URL
            text: Current caption text

        Returns:
            True if cached caption matches, False if mismatch or not cached
        """
        cached = self.get_cached_sample(url)
        if cached is None:
            return False

        if cached.get('text') != text:
            self.log_debug(f"Caption mismatch for {url[:50]}, cache invalid")
            return False

        return True

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.

        Returns:
            Dictionary with cache statistics:
            - hits: Number of cache hits
            - misses: Number of cache misses
            - errors: Number of cache errors
            - saves: Number of successful cache writes
            - hit_rate: Cache hit rate (0.0-1.0)
            - total_samples: Total cached sample files
            - size_mb: Total cache size in MB
        """
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0.0

        # Calculate cache size if enabled
        total_samples = 0
        size_bytes = 0

        if self.enabled and self.cache_dir.exists():
            for cache_file in self.cache_dir.rglob(f"sample_*_v{self.cache_version}.json"):
                total_samples += 1
                try:
                    size_bytes += cache_file.stat().st_size
                except:
                    pass

        return {
            'enabled': self.enabled,
            'cache_version': self.cache_version,
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'errors': self.stats['errors'],
            'saves': self.stats['saves'],
            'hit_rate': hit_rate,
            'total_samples': total_samples,
            'size_mb': size_bytes / (1024 * 1024)
        }

    def clear_cache(self, url: Optional[str] = None) -> int:
        """
        Clear cached data.

        Args:
            url: If provided, clear cache for specific URL only.
                 If None, clear entire cache directory.

        Returns:
            Number of cache files deleted
        """
        if not self.enabled:
            return 0

        deleted_count = 0

        try:
            if url:
                # Clear specific URL
                cache_path = self._get_cache_path(url)
                if cache_path.exists():
                    cache_path.unlink()
                    deleted_count = 1
                    self.log_info(f"Cleared cache for {url[:50]}")
            else:
                # Clear entire cache
                if self.cache_dir.exists():
                    for cache_file in self.cache_dir.rglob(f"sample_*_v{self.cache_version}.json"):
                        try:
                            cache_file.unlink()
                            deleted_count += 1
                        except Exception as e:
                            self.log_warning(f"Failed to delete {cache_file}: {e}")
                    self.log_info(f"Cleared {deleted_count} cache files")

        except Exception as e:
            self.log_error(f"Failed to clear cache: {e}")

        return deleted_count

    def reset_stats(self):
        """Reset cache statistics."""
        self.stats = {
            'hits': 0,
            'misses': 0,
            'errors': 0,
            'saves': 0
        }
        self.log_debug("Cache statistics reset")
