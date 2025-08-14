"""Main retrieval module for MAVERIC."""

import numpy as np
import torch
import clip
from PIL import Image
from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm
from datetime import datetime
import time

from ..core.base import BaseComponent
from ..core.interfaces import RetrievalResult, ProgressCallback
from ..core.exceptions import DatasetError, ModelError
from ..datasets import get_dataset
from ..quality.metrics import (
    ResolutionMetric,
    SharpnessMetric,
    ColorDiversityMetric,
    FeatureRichnessMetric
)
from .cache_manager import CacheManager
from .dataset_handlers import DatasetHandler


class Retriever(BaseComponent):
    """
    Main retrieval engine for MAVERIC.
    
    This component handles the retrieval and scoring of samples from
    large-scale datasets, computing quality metrics and managing the
    retrieval process efficiently.
    """
    
    def __init__(self,
                 clip_model: str = "ViT-B/32",
                 device: str = "cuda",
                 cache_manager: Optional[CacheManager] = None,
                 n_reference_images: int = 10):
        """
        Initialize retriever.
        
        Args:
            clip_model: CLIP model to use
            device: Computation device
            cache_manager: Cache manager instance
            n_reference_images: Number of reference images per class
        """
        super().__init__("Retriever")
        
        self.clip_model_name = clip_model
        self.device = device if torch.cuda.is_available() else "cpu"
        self.cache_manager = cache_manager
        self.n_reference_images = n_reference_images
        
        # Initialize CLIP model
        self._init_clip_model()
        
        # Initialize quality metrics
        self._init_quality_metrics()
        
        # Storage for embeddings
        self.reference_embeddings = {}
        self.text_embeddings = {}
    
    def _init_clip_model(self):
        """Initialize CLIP model."""
        try:
            self.log_info(f"Loading CLIP model: {self.clip_model_name}")
            self.model, self.preprocess = clip.load(
                self.clip_model_name,
                device=self.device
            )
            self.model.eval()
        except Exception as e:
            raise ModelError(f"Failed to load CLIP model: {e}")
    
    def _init_quality_metrics(self):
        """Initialize quality metric calculators."""
        self.quality_metrics = {
            'resolution': ResolutionMetric(),
            'sharpness': SharpnessMetric(),
            'color_diversity': ColorDiversityMetric(),
            'feature_richness': FeatureRichnessMetric(device=self.device)
        }
    
    def prepare_reference_embeddings(self, 
                                   target_dataset: str,
                                   save_cache: bool = True) -> Tuple[Dict, Dict]:
        """
        Prepare reference embeddings for the target dataset.
        
        Args:
            target_dataset: Name of target dataset
            save_cache: Whether to save embeddings to cache
            
        Returns:
            Tuple of (reference_embeddings, text_embeddings)
        """
        # Try to load from cache first
        if self.cache_manager:
            cache_name = f"{target_dataset}_reference"
            cached = self.cache_manager.load_embeddings(cache_name)
            if cached:
                self.reference_embeddings = cached.get('reference', {})
                self.text_embeddings = cached.get('text', {})
                self.log_info("Loaded reference embeddings from cache")
                return self.reference_embeddings, self.text_embeddings
        
        # Load target dataset
        dataset = get_dataset(target_dataset)
        
        # Get reference samples
        self.log_info(f"Creating reference embeddings for {target_dataset}")
        reference_samples = dataset.get_reference_samples(self.n_reference_images)
        
        # Create image embeddings
        self.reference_embeddings = {}
        for class_name, images in tqdm(reference_samples.items(), 
                                      desc="Encoding reference images"):
            embeddings = []
            for img in images:
                with torch.no_grad():
                    img_input = self.preprocess(img).unsqueeze(0).to(self.device)
                    img_features = self.model.encode_image(img_input)
                    img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                    embeddings.append(img_features.cpu().numpy())
            
            if embeddings:
                self.reference_embeddings[class_name] = np.vstack(embeddings)
        
        # Create text embeddings
        self.text_embeddings = {}
        templates = dataset.get_text_templates()
        
        for class_name in tqdm(dataset.class_names, desc="Encoding text templates"):
            prompts = [template.format(class_name) for template in templates]
            
            with torch.no_grad():
                tokens = clip.tokenize(prompts).to(self.device)
                text_features = self.model.encode_text(tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                self.text_embeddings[class_name] = text_features.cpu().numpy()
        
        # Save to cache
        if save_cache and self.cache_manager:
            embeddings_data = {
                'reference': self.reference_embeddings,
                'text': self.text_embeddings
            }
            self.cache_manager.save_embeddings(embeddings_data, f"{target_dataset}_reference")
        
        return self.reference_embeddings, self.text_embeddings
    
    def _export_rotation_file(self,
                             batch: List[Dict],
                             target_dataset: str,
                             file_id: int,
                             export_dir: str):
        """
        Export rotation file with proper naming convention.
        
        Args:
            batch: List of sample dictionaries to export
            target_dataset: Target dataset name
            file_id: File identifier
            export_dir: Directory to export the file to
        """
        import json
        from pathlib import Path
        
        try:
            # Create export directory if it doesn't exist
            Path(export_dir).mkdir(parents=True, exist_ok=True)
            
            # Create filename: {datasetName}_raw_maveric_{file_id}.json
            filename = f"{target_dataset.lower()}_raw_maveric_{file_id}.json"
            filepath = Path(export_dir) / filename
            
            # Save the batch
            with open(filepath, 'w') as f:
                json.dump(batch, f, indent=2)
            
            print(f"📁 Exported rotation file: {filename} ({len(batch)} samples)")
            self.log_info(f"Exported rotation file: {filename} ({len(batch)} samples)")
            
        except Exception as e:
            self.log_warning(f"Failed to export rotation file: {e}")
    
    def compute_sample_scores(self, 
                            image_url: str,
                            text: str,
                            download_image: bool = True) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
        """
        Compute all scores for a single sample.
        
        Args:
            image_url: URL of the image
            text: Text caption
            download_image: Whether to download the image
            
        Returns:
            Tuple of (class_scores, quality_scores)
        """
        try:
            # Get image
            if download_image and self.cache_manager:
                image = self.cache_manager.download_and_cache_image(image_url)
            else:
                # Direct download without caching
                import requests
                from io import BytesIO
                response = requests.get(image_url, timeout=5)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            
            if image is None:
                return {}, {}
            
            # Compute embeddings
            with torch.no_grad():
                # Image embedding
                img_input = self.preprocess(image).unsqueeze(0).to(self.device)
                img_embedding = self.model.encode_image(img_input)
                img_embedding = img_embedding / img_embedding.norm(dim=-1, keepdim=True)
                img_embedding = img_embedding.cpu().numpy()
                
                # Text embedding
                text_tokens = clip.tokenize([text], truncate=True).to(self.device)
                text_embedding = self.model.encode_text(text_tokens)
                text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
                text_embedding = text_embedding.cpu().numpy()
            
            # Compute class scores
            class_scores = {}
            
            if not self.reference_embeddings:
                self.log_warning("No reference embeddings available for class score computation")
                return {}, {}
            
            for class_name in self.reference_embeddings.keys():
                # Similarity computations
                from sklearn.metrics.pairwise import cosine_similarity
                
                img2img = cosine_similarity(
                    img_embedding,
                    self.reference_embeddings[class_name]
                ).max()
                
                txt2txt = cosine_similarity(
                    text_embedding,
                    self.text_embeddings[class_name]
                ).max()
                
                img2txt = cosine_similarity(
                    img_embedding,
                    self.text_embeddings[class_name]
                ).max()
                
                txt2img = cosine_similarity(
                    text_embedding,
                    self.reference_embeddings[class_name]
                ).max()
                
                # Calculate hybrid score
                hybrid_score = 0.25 * (img2img + txt2txt + img2txt + txt2img)
                
                # Calculate consistency
                similarities = [img2img, txt2txt, img2txt, txt2img]
                consistency = 1.0 - np.std(similarities)
                
                class_scores[class_name] = {
                    'hybrid_score': float(hybrid_score),
                    'img2img': float(img2img),
                    'txt2txt': float(txt2txt),
                    'img2txt': float(img2txt),
                    'txt2img': float(txt2img),
                    'consistency': float(consistency)
                }
            
            # Compute quality scores
            quality_scores = {}
            metadata = {'url': image_url, 'text': text}
            
            for metric_name, metric in self.quality_metrics.items():
                try:
                    score = metric.compute(image, metadata)
                    quality_scores[f"{metric_name}_score"] = float(score)
                except Exception as e:
                    self.log_warning(f"Error computing {metric_name}: {e}")
                    quality_scores[f"{metric_name}_score"] = 0.0
            
            return class_scores, quality_scores
            
        except Exception as e:
            self.log_warning(f"Error processing sample {image_url[:50]}...: {str(e)}")
            import traceback
            self.log_debug(f"Full traceback: {traceback.format_exc()}")
            return {}, {}
    
    def retrieve(self,
                dataset_handler: DatasetHandler,
                target_dataset: str,
                rotation_size: int,
                num_samples: Optional[int] = None,
                start_index: int = 0,
                progress_callback: Optional[ProgressCallback] = None,
                export_rotation_files: bool = True,
                rotation_export_dir: Optional[str] = None) -> RetrievalResult:
        """
        Retrieve and score samples from dataset.
        
        Args:
            dataset_handler: Dataset handler instance
            target_dataset: Target dataset name
            rotation_size: Samples per file rotation (from config.retrieval_rotation_size)
            num_samples: Number of samples to retrieve (None for all)
            start_index: Starting index
            progress_callback: Progress tracking callback
            export_rotation_files: Whether to export rotation files during retrieval
            rotation_export_dir: Directory to export rotation files (None = no export)
            
        Returns:
            RetrievalResult with all retrieved samples
        """
        # Prepare reference embeddings
        self.log_info(f"Preparing reference embeddings for {target_dataset}...")
        ref_embeddings, text_embeddings = self.prepare_reference_embeddings(target_dataset)
        self.log_info(f"Reference embeddings prepared: {len(ref_embeddings)} classes, {sum(len(v) for v in ref_embeddings.values())} total embeddings")
        
        # Initialize storage
        all_samples = []
        processed_count = 0
        file_id = 1
        current_batch = []
        
        # Determine total samples
        total_samples = num_samples or len(dataset_handler)
        if total_samples == -1:  # Streaming dataset
            total_samples = num_samples or float('inf')
        
        self.log_info(f"Starting retrieval for {target_dataset}")
        start_time = time.time()
        
        # Process samples
        for idx, item in enumerate(dataset_handler):
            # Skip if before start index
            if idx < start_index:
                continue
            
            # Check if we've processed enough
            if num_samples and processed_count >= num_samples:
                break
            
            # Process sample
            url = item.get('URL', '')
            text = item.get('TEXT', '')
            
            if not url or not text:
                continue
            
            # Compute scores
            class_scores, quality_scores = self.compute_sample_scores(url, text)
            
            if not class_scores or not quality_scores:
                self.log_warning(f"Failed to compute scores for sample {processed_count + 1}: url={url[:50]}...")
                continue
            
            # Create sample record
            sample = {
                'id': start_index + processed_count + 1,
                'url': url,
                'text': text
            }
            
            # Add flattened class scores
            for class_name, scores in class_scores.items():
                for metric_name, value in scores.items():
                    sample[f"Class_{class_name}_{metric_name}"] = value
            
            # Add quality scores
            sample.update(quality_scores)
            
            # Add to batch
            current_batch.append(sample)
            all_samples.append(sample)
            processed_count += 1
            
            # Log batch progress
            if processed_count % 10 == 0 or len(current_batch) % 5 == 0:
                self.log_info(f"Batch progress: {len(current_batch)}/{rotation_size} samples, total processed: {processed_count}")
            
            # Update progress
            if progress_callback:
                progress_callback.update(
                    processed_count,
                    min(total_samples, num_samples) if num_samples else total_samples,
                    f"Processing sample {processed_count}"
                )
            
            # Save batch if needed
            if len(current_batch) >= rotation_size:
                print(f"🔄 Rotation size ({rotation_size}) reached! Exporting batch #{file_id}...")
                
                if self.cache_manager:
                    self.cache_manager.save_results(
                        current_batch,
                        target_dataset,
                        file_id,
                        prefix="raw_maveric"
                    )
                
                # Export rotation file if requested
                if export_rotation_files and rotation_export_dir:
                    self._export_rotation_file(
                        current_batch,
                        target_dataset,
                        file_id,
                        rotation_export_dir
                    )
                
                current_batch = []
                file_id += 1
            
            # Log progress periodically
            if processed_count % 100 == 0:
                elapsed = time.time() - start_time
                rate = processed_count / elapsed
                self.log_info(f"Processed {processed_count} samples ({rate:.1f} samples/sec)")
        
        # Save remaining batch
        if current_batch and self.cache_manager:
            self.cache_manager.save_results(
                current_batch,
                target_dataset,
                file_id,
                prefix="raw_maveric"
            )
        
        # Export remaining batch if requested
        if current_batch and export_rotation_files and rotation_export_dir:
            print(f"📝 Exporting final batch #{file_id} with {len(current_batch)} remaining samples...")
            self._export_rotation_file(
                current_batch,
                target_dataset,
                file_id,
                rotation_export_dir
            )
        
        # Complete progress
        if progress_callback:
            progress_callback.complete()
        
        # Create result
        elapsed_time = time.time() - start_time
        self.log_info(
            f"Retrieval complete: {processed_count} samples in {elapsed_time:.1f}s "
            f"({processed_count/elapsed_time:.1f} samples/sec)"
        )
        
        return RetrievalResult(
            samples=all_samples,
            source_dataset=dataset_handler.dataset_name if hasattr(dataset_handler, 'dataset_name') else 'unknown',
            target_dataset=target_dataset,
            config={
                'clip_model': self.clip_model_name,
                'n_reference_images': self.n_reference_images,
                'start_index': start_index,
                'rotation_size': rotation_size
            }
        )
