"""Cache management for MAVERIC."""

import os
import json
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any
from PIL import Image
import requests
from io import BytesIO
import time

from ..core.base import BaseComponent
from ..core.exceptions import CacheError


class CacheManager(BaseComponent):
    """
    Manages caching of images and results.
    
    This component handles efficient caching of downloaded images and
    computed results to avoid redundant processing and network requests.
    """
    
    def __init__(self, 
                 base_dir: str,
                 enable_image_cache: bool = True,
                 cache_format: str = "jpg",
                 cache_quality: int = 95):
        """
        Initialize cache manager.
        
        Args:
            base_dir: Base directory for all cache files
            enable_image_cache: Whether to cache downloaded images
            cache_format: Image format for caching (jpg, png)
            cache_quality: JPEG quality (1-100)
        """
        super().__init__("CacheManager")
        
        self.base_dir = Path(base_dir)
        self.enable_image_cache = enable_image_cache
        self.cache_format = cache_format.lower()
        self.cache_quality = cache_quality
        
        # Create cache directories
        self.image_cache_dir = self.base_dir / 'image_cache'
        self.result_cache_dir = self.base_dir / 'results'
        self.embedding_cache_dir = self.base_dir / 'embeddings'
        
        self._create_directories()
        
        # Cache statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'images_cached': 0,
            'bytes_saved': 0
        }
    
    def _create_directories(self):
        """Create cache directory structure."""
        directories = [
            self.base_dir,
            self.image_cache_dir,
            self.result_cache_dir,
            self.embedding_cache_dir
        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise CacheError(f"Failed to create cache directory {directory}: {e}")
    
    def _get_url_hash(self, url: str) -> str:
        """Generate hash for URL."""
        return hashlib.md5(url.encode()).hexdigest()
    
    def cache_image(self, url: str, image: Image.Image) -> str:
        """
        Cache an image and return cache path.
        
        Args:
            url: Original image URL
            image: PIL Image to cache
            
        Returns:
            Path to cached image
        """
        if not self.enable_image_cache:
            return ""
        
        url_hash = self._get_url_hash(url)
        cache_filename = f"img_{url_hash}.{self.cache_format}"
        cache_path = self.image_cache_dir / cache_filename
        
        try:
            # Save image
            if self.cache_format == 'jpg':
                # Convert RGBA to RGB for JPEG
                if image.mode == 'RGBA':
                    rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                    rgb_image.paste(image, mask=image.split()[3])
                    image = rgb_image
                image.save(cache_path, 'JPEG', quality=self.cache_quality)
            else:
                image.save(cache_path, self.cache_format.upper())
            
            self.stats['images_cached'] += 1
            self.log_debug(f"Cached image: {cache_filename}")
            
            return str(cache_path)
            
        except Exception as e:
            self.log_warning(f"Failed to cache image: {e}")
            return ""
    
    def get_cached_image(self, url: str) -> Optional[Image.Image]:
        """
        Retrieve cached image if exists.
        
        Args:
            url: Original image URL
            
        Returns:
            PIL Image if cached, None otherwise
        """
        if not self.enable_image_cache:
            return None
        
        url_hash = self._get_url_hash(url)
        
        # Try different formats
        for ext in [self.cache_format, 'jpg', 'png']:
            cache_filename = f"img_{url_hash}.{ext}"
            cache_path = self.image_cache_dir / cache_filename
            
            if cache_path.exists():
                try:
                    image = Image.open(cache_path)
                    # Force load to verify integrity
                    image.load()
                    self.stats['cache_hits'] += 1
                    return image.convert('RGB')
                except Exception as e:
                    self.log_warning(f"Corrupted cache file {cache_filename}: {e}")
                    # Remove corrupted file
                    try:
                        cache_path.unlink()
                    except:
                        pass
        
        self.stats['cache_misses'] += 1
        return None
    
    def download_and_cache_image(self, url: str, max_retries: int = 3, 
                               timeout: int = 5) -> Optional[Image.Image]:
        """
        Download image with caching support.
        
        Args:
            url: Image URL
            max_retries: Maximum number of download attempts
            timeout: Request timeout in seconds
            
        Returns:
            PIL Image if successful, None otherwise
        """
        # Check cache first
        cached_image = self.get_cached_image(url)
        if cached_image is not None:
            return cached_image
        
        # Download image
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()
                
                # Load image
                image = Image.open(BytesIO(response.content)).convert('RGB')
                
                # Cache it
                self.cache_image(url, image)
                
                # Track bandwidth saved
                self.stats['bytes_saved'] += len(response.content)
                
                return image
                
            except Exception as e:
                self.log_warning(f"Download attempt {attempt + 1} failed for {url}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Brief delay before retry
        
        return None
    
    def save_results(self, 
                    results: List[Dict],
                    dataset_name: str,
                    file_id: int,
                    prefix: str = "") -> str:
        """
        Save retrieval results to JSON.
        
        Args:
            results: List of result dictionaries
            dataset_name: Name of the dataset
            file_id: File ID for rotation
            prefix: Optional prefix for filename
            
        Returns:
            Path to saved file
        """
        # Create filename
        if prefix:
            filename = f"{prefix}_{dataset_name}_results_{file_id}.json"
        else:
            filename = f"{dataset_name}_results_{file_id}.json"
        
        filepath = self.result_cache_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.log_info(f"Saved {len(results)} results to {filename}")
            return str(filepath)
            
        except Exception as e:
            raise CacheError(f"Failed to save results: {e}")
    
    def load_results(self, 
                    dataset_name: str,
                    file_pattern: Optional[str] = None) -> List[Dict]:
        """
        Load previously saved results.
        
        Args:
            dataset_name: Dataset name to load
            file_pattern: Optional file pattern to match
            
        Returns:
            Combined list of all matching results
        """
        all_results = []
        
        # Determine pattern
        if file_pattern:
            pattern = file_pattern
        else:
            pattern = f"*{dataset_name}*.json"
        
        # Find matching files
        result_files = sorted(self.result_cache_dir.glob(pattern))
        
        if not result_files:
            self.log_warning(f"No result files found matching {pattern}")
            return all_results
        
        # Load each file
        for filepath in result_files:
            try:
                with open(filepath, 'r') as f:
                    results = json.load(f)
                    if isinstance(results, list):
                        all_results.extend(results)
                    else:
                        all_results.append(results)
                    
                self.log_debug(f"Loaded {filepath.name}")
                
            except Exception as e:
                self.log_warning(f"Failed to load {filepath}: {e}")
        
        self.log_info(f"Loaded {len(all_results)} total results from {len(result_files)} files")
        return all_results
    
    def save_embeddings(self, 
                       embeddings: Dict[str, Any],
                       name: str) -> str:
        """
        Save embeddings to cache.
        
        Args:
            embeddings: Dictionary of embeddings
            name: Name for the embedding file
            
        Returns:
            Path to saved file
        """
        import numpy as np
        
        filepath = self.embedding_cache_dir / f"{name}_embeddings.npz"
        
        try:
            # Convert to numpy arrays if needed
            np_embeddings = {}
            for key, value in embeddings.items():
                if isinstance(value, list):
                    np_embeddings[key] = np.array(value)
                else:
                    np_embeddings[key] = value
            
            # Save as compressed numpy file
            np.savez_compressed(filepath, **np_embeddings)
            
            self.log_info(f"Saved embeddings to {filepath.name}")
            return str(filepath)
            
        except Exception as e:
            raise CacheError(f"Failed to save embeddings: {e}")
    
    def load_embeddings(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Load embeddings from cache.
        
        Args:
            name: Name of the embedding file
            
        Returns:
            Dictionary of embeddings if found, None otherwise
        """
        import numpy as np
        
        filepath = self.embedding_cache_dir / f"{name}_embeddings.npz"
        
        if not filepath.exists():
            return None
        
        try:
            # Load numpy file
            with np.load(filepath, allow_pickle=True) as data:
                embeddings = dict(data)
            
            self.log_info(f"Loaded embeddings from {filepath.name}")
            return embeddings
            
        except Exception as e:
            self.log_warning(f"Failed to load embeddings: {e}")
            return None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        # Calculate cache sizes
        image_size = sum(f.stat().st_size for f in self.image_cache_dir.glob('*'))
        result_size = sum(f.stat().st_size for f in self.result_cache_dir.glob('*'))
        embedding_size = sum(f.stat().st_size for f in self.embedding_cache_dir.glob('*'))
        
        return {
            **self.stats,
            'cache_size': {
                'images_mb': image_size / (1024 * 1024),
                'results_mb': result_size / (1024 * 1024),
                'embeddings_mb': embedding_size / (1024 * 1024),
                'total_mb': (image_size + result_size + embedding_size) / (1024 * 1024)
            },
            'cache_counts': {
                'images': len(list(self.image_cache_dir.glob('*'))),
                'results': len(list(self.result_cache_dir.glob('*'))),
                'embeddings': len(list(self.embedding_cache_dir.glob('*')))
            }
        }
    
    def clear_cache(self, cache_type: str = 'all'):
        """
        Clear cache files.
        
        Args:
            cache_type: Type of cache to clear ('images', 'results', 'embeddings', 'all')
        """
        if cache_type in ['images', 'all']:
            for f in self.image_cache_dir.glob('*'):
                f.unlink()
            self.log_info("Cleared image cache")
        
        if cache_type in ['results', 'all']:
            for f in self.result_cache_dir.glob('*'):
                f.unlink()
            self.log_info("Cleared results cache")
        
        if cache_type in ['embeddings', 'all']:
            for f in self.embedding_cache_dir.glob('*'):
                f.unlink()
            self.log_info("Cleared embeddings cache")
