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
    TargetClassQualityMetric
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
                 n_reference_images: int = 10,
                 real_time_stats=None,
                 seed: int = 42,
                 enable_target_class_quality: bool = True):
        """
        Initialize retriever.

        Args:
            clip_model: CLIP model to use
            device: Computation device
            cache_manager: Cache manager instance
            n_reference_images: Number of reference images per class
            real_time_stats: Real-time stats object for progress tracking
            seed: Random seed for reproducible sampling
            enable_target_class_quality: Enable EfficientNet-based TargetClassQualityMetric (default: True)
        """
        super().__init__("Retriever")

        self.clip_model_name = clip_model
        self.device = device if torch.cuda.is_available() else "cpu"
        self.cache_manager = cache_manager
        self.n_reference_images = n_reference_images
        self.real_time_stats = real_time_stats
        self.seed = seed
        self.enable_target_class_quality = enable_target_class_quality

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
        }

        # Conditionally add TargetClassQualityMetric (EfficientNet-based, time-consuming)
        if self.enable_target_class_quality:
            self.quality_metrics['target_class_quality'] = TargetClassQualityMetric()
            self.log_info("TargetClassQualityMetric enabled (EfficientNet-based scoring)")
        else:
            self.log_info("TargetClassQualityMetric disabled (skipping EfficientNet calculations)")
    
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
                ref_cache = cached.get('reference', {})
                text_cache = cached.get('text', {})
                
                # More lenient validation for reproducibility
                if ref_cache and text_cache:
                    try:
                        # Try to use cached data
                        self.reference_embeddings = ref_cache if isinstance(ref_cache, dict) else {}
                        self.text_embeddings = text_cache if isinstance(text_cache, dict) else {}
                        
                        # Only reject if completely empty
                        if len(self.reference_embeddings) > 0 and len(self.text_embeddings) > 0:
                            self.log_info("Loaded reference embeddings from cache")
                            return self.reference_embeddings, self.text_embeddings
                    except Exception as e:
                        self.log_warning(f"Error loading cached embeddings: {e}")
                
                self.log_info(f"Cache invalid or empty for {target_dataset}, regenerating...")
                # Continue to regenerate embeddings
        
        # Load target dataset with proper cache directory
        if self.cache_manager:
            # Use cache directory for dataset storage
            dataset_cache_dir = self.cache_manager.base_dir / 'datasets'
            dataset = get_dataset(target_dataset, root=str(dataset_cache_dir))
        else:
            # Fallback to default location
            dataset = get_dataset(target_dataset)
        
        # Get reference samples with seed for reproducibility
        self.log_info(f"Creating reference embeddings for {target_dataset}")
        reference_samples = dataset.get_reference_samples(self.n_reference_images, seed=self.seed)
        
        # Get text templates for later use
        text_templates = dataset.get_text_templates()
        
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
        
        # Log target class to ImageNet class mappings at the beginning
        self._log_class_mappings(target_dataset, dataset.class_names)
        
        for class_name in tqdm(dataset.class_names, desc="Encoding text templates"):
            prompts = [template.format(class_name) for template in text_templates]
            
            with torch.no_grad():
                tokens = clip.tokenize(prompts).to(self.device)
                text_features = self.model.encode_text(tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                self.text_embeddings[class_name] = text_features.cpu().numpy()
        
        # Save to cache
        if save_cache and self.cache_manager:
            # Save embeddings
            embeddings_data = {
                'reference': self.reference_embeddings,
                'text': self.text_embeddings
            }
            self.cache_manager.save_embeddings(embeddings_data, f"{target_dataset}_reference")
            
            # Save reference images for verification
            self.cache_manager.save_reference_images(reference_samples, target_dataset)
            self.log_info(f"Saved reference images for {target_dataset}")
            
            # Save reference texts for verification
            self.cache_manager.save_reference_texts(text_templates, dataset.class_names, target_dataset)
            self.log_info(f"Saved reference texts for {target_dataset}")
            
            print(f"📸 Reference images saved: {sum(len(images) for images in reference_samples.values())} images across {len(reference_samples)} classes")
            print(f"📝 Reference texts saved: {len(text_templates)} templates for {len(dataset.class_names)} classes")
        
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
            
            # Create filename: {datasetName}_raw_maveric_dataset{file_id}.json
            filename = f"{target_dataset.lower()}_raw_maveric_dataset{file_id}.json"
            filepath = Path(export_dir) / filename
            
            # Save the batch
            with open(filepath, 'w') as f:
                json.dump(batch, f, indent=2)
            
            print(f"📁 Exported rotation file: {filename} ({len(batch)} samples)")
            self.log_info(f"Exported rotation file: {filename} ({len(batch)} samples)")
            
        except Exception as e:
            print(f"❌ Failed to export rotation file: {e}")
            self.log_warning(f"Failed to export rotation file: {e}")
    
    def _compute_all_imagenet_mappings(self, image, text: str, target_classes: List[str]) -> Dict[str, Tuple[str, float]]:
        """
        Efficiently compute ImageNet class mappings for all target classes at once.
        
        This method runs EfficientNet only once per image, then reuses the probabilities
        to compute mappings for all target classes, significantly improving performance.
        
        Args:
            image: PIL Image object
            text: Caption text
            target_classes: List of target class names to compute mappings for
            
        Returns:
            Dictionary mapping target_class_name -> (predicted_imagenet_class_name, efficientNet_score)
        """
        try:
            # Get the target class quality metric
            if 'target_class_quality' not in self.quality_metrics:
                return {class_name: ("", 0.0) for class_name in target_classes}
                
            metric = self.quality_metrics['target_class_quality']
            
            # Run EfficientNet inference only ONCE for this image
            probabilities = metric.compute_image_probabilities_only(image)
            
            # Compute mappings for all target classes using the same probabilities
            results = metric.compute_all_mappings_from_probabilities(probabilities, target_classes)
            
            return results
            
        except Exception as e:
            self.log_warning(f"Error computing ImageNet mappings: {e}")
            return {class_name: ("", 0.0) for class_name in target_classes}
    
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
                self.log_debug("No reference embeddings available for class score computation")
                return {}, {}
            
            # OPTIMIZATION: Compute ImageNet mappings for ALL target classes at once (if enabled)
            target_classes = list(self.reference_embeddings.keys())

            # Only compute EfficientNet-based scores if enabled
            if self.enable_target_class_quality:
                imagenet_mappings = self._compute_all_imagenet_mappings(image, text, target_classes)

                # Get global ImageNet prediction for this image
                global_imagenet_pred, global_imagenet_prob = ("", 0.0)
                if 'target_class_quality' in self.quality_metrics:
                    metric = self.quality_metrics['target_class_quality']
                    try:
                        global_imagenet_pred, global_imagenet_prob = metric.compute_single_imagenet_prediction(image)
                    except Exception as e:
                        self.log_debug(f"Error getting global ImageNet prediction: {e}")
                        global_imagenet_pred, global_imagenet_prob = ("", 0.0)
            else:
                # Skip EfficientNet calculations entirely
                imagenet_mappings = {class_name: ("", 0.0, 0.0) for class_name in target_classes}
                global_imagenet_pred, global_imagenet_prob = ("", 0.0)
            
            for class_name in target_classes:
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
                
                # Get pre-computed EfficientNet score and CLIP similarity (no additional EfficientNet calls!)
                predicted_imagenet_class, clip_similarity, efficientNet_score = imagenet_mappings.get(class_name, ("", 0.0, 0.0))

                # Build class scores dictionary
                class_scores[class_name] = {
                    'hybrid_score': round(float(hybrid_score), 5),
                    'img2img': round(float(img2img), 5),
                    'txt2txt': round(float(txt2txt), 5),
                    'img2txt': round(float(img2txt), 5),
                    'txt2img': round(float(txt2img), 5),
                    'consistency': round(float(consistency), 5),
                }

                # Only include EfficientNet-based scores if enabled
                if self.enable_target_class_quality:
                    class_scores[class_name]['clip_similarity_to_imagenet'] = round(float(clip_similarity), 5)
                    class_scores[class_name]['efficientNet_score'] = round(float(efficientNet_score), 5)
            
            # Compute quality scores (exclude target_class_quality as it's now per-class)
            quality_scores = {}
            metadata = {'url': image_url, 'text': text}
            
            for metric_name, metric in self.quality_metrics.items():
                # Skip target_class_quality as it's now calculated per-class as efficientNet_score
                if metric_name == 'target_class_quality':
                    continue
                    
                try:
                    score = metric.compute(image, metadata)
                    quality_scores[metric.metric_name] = round(float(score), 5)
                except Exception as e:
                    self.log_warning(f"Error computing {metric_name}: {e}")
                    quality_scores[metric.metric_name] = 0.0
            
            # Add global ImageNet prediction fields (only if EfficientNet is enabled)
            if self.enable_target_class_quality:
                quality_scores['imagenet_predicted_class'] = global_imagenet_pred
                quality_scores['imagenet_probability'] = round(float(global_imagenet_prob), 5)

            return class_scores, quality_scores
            
        except Exception as e:
            # Log to debug level instead of warning to reduce console output
            self.log_debug(f"Error processing sample {image_url[:50]}...: {str(e)}")
            import traceback
            self.log_debug(f"Full traceback: {traceback.format_exc()}")
            return {}, {}
    
    def retrieve(self,
                dataset_handler: DatasetHandler,
                target_dataset: str,
                rotation_size: int,
                num_samples: Optional[int] = None,
                start_index: int = 0,
                start_file_id: int = 1,
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
            start_file_id: Starting file sequence number for rotation files (default: 1)
            progress_callback: Progress tracking callback
            export_rotation_files: Whether to export rotation files during retrieval
            rotation_export_dir: Directory to export rotation files (None = no export)
            
        Returns:
            RetrievalResult with all retrieved samples
        """
        # Prepare reference embeddings
        self.log_info(f"Preparing reference embeddings for {target_dataset}...")
        ref_embeddings, text_embeddings = self.prepare_reference_embeddings(target_dataset)
        
        # Validate embeddings
        if not isinstance(ref_embeddings, dict) or not isinstance(text_embeddings, dict):
            raise ValueError(f"Invalid embeddings returned for {target_dataset}: ref_embeddings={type(ref_embeddings)}, text_embeddings={type(text_embeddings)}")
        
        if len(ref_embeddings) == 0:
            raise ValueError(f"No reference embeddings found for {target_dataset}")
        
        total_embeddings = sum(len(v) if hasattr(v, '__len__') else 0 for v in ref_embeddings.values())
        self.log_info(f"Reference embeddings prepared: {len(ref_embeddings)} classes, {total_embeddings} total embeddings")
        print(f"📊 Reference embeddings ready: {len(ref_embeddings)} classes")
        
        # Initialize storage
        all_samples = []
        processed_count = 0
        file_id = start_file_id
        current_batch = []
        
        # Determine total samples
        total_samples = num_samples or len(dataset_handler)
        if total_samples == -1:  # Streaming dataset
            total_samples = num_samples or float('inf')
        
        # Get actual dataset size for index tracking
        dataset_size = len(dataset_handler) if hasattr(dataset_handler, '__len__') else None
        
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
            
            # Update current index for every sample (successful or failed)
            if self.real_time_stats:
                index_stats = {
                    'current_index': idx + 1,  # +1 for 1-based indexing (idx is already absolute position)
                    'batch_size': rotation_size,
                }
                if dataset_size is not None:
                    index_stats['total_samples'] = dataset_size
                self.real_time_stats.update_stats(index_stats)
            
            # Process sample
            url = item.get('URL', '')
            text = item.get('TEXT', '')
            
            if not url or not text:
                continue
            
            # Compute scores
            class_scores, quality_scores = self.compute_sample_scores(url, text)
            
            if not class_scores or not quality_scores:
                # Log to debug level instead of warning to reduce console output
                self.log_debug(f"Failed to compute scores for sample {processed_count + 1}: url={url[:50]}...")
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
            
            # Update batch position after successful processing
            if self.real_time_stats:
                batch_stats = {
                    'current_batch_position': len(current_batch),
                }
                self.real_time_stats.update_stats(batch_stats)
            
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
                
                # Export rotation file to user-specified directory only
                if export_rotation_files and rotation_export_dir:
                    self._export_rotation_file(
                        current_batch,
                        target_dataset,
                        file_id,
                        rotation_export_dir
                    )
                
                current_batch = []
                file_id += 1
                
                # Reset batch position in stats
                if self.real_time_stats:
                    reset_stats = {'current_batch_position': 0}
                    self.real_time_stats.update_stats(reset_stats)
            
            # Log progress periodically
            if processed_count % 100 == 0:
                elapsed = time.time() - start_time
                rate = processed_count / elapsed
                self.log_info(f"Processed {processed_count} samples ({rate:.1f} samples/sec)")
        
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
    
    def _log_class_mappings(self, target_dataset: str, class_names: list):
        """
        Log target dataset class information at the start of retrieval.
        
        With the new simplified algorithm, we don't pre-compute class mappings since
        each image gets a single ImageNet prediction that's compared with all target classes.
        
        Args:
            target_dataset: Name of target dataset
            class_names: List of class names from target dataset
        """
        self.log_info(f"\n🎯 TARGET DATASET CLASSES FOR {target_dataset.upper()}")
        self.log_info("=" * 80)
        self.log_info("Using CORRECTED algorithm:")
        self.log_info("1. EfficientNet predicts single ImageNet class per image")
        self.log_info("2. CLIP calculates similarity between each target class and predicted ImageNet class")
        self.log_info("3. Final score = CLIP_similarity × imagenet_probability")
        self.log_info("-" * 80)
        
        # Simply log the target classes
        for i, class_name in enumerate(class_names, 1):
            self.log_info(f"{i:2d}. {class_name}")
        
        self.log_info("=" * 80)
        self.log_info(f"📊 Total target classes: {len(class_names)}")
        self.log_info("🔄 Starting retrieval process...\n")
