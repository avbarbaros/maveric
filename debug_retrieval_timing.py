#!/usr/bin/env python3
"""
Debug script to instrument retrieval timing and identify bottlenecks.
This patches the retriever to add detailed timing logs.
"""

import time
import sys
from pathlib import Path

# Add maveric to path
sys.path.insert(0, str(Path(__file__).parent))

# Monkey-patch the compute_sample_scores method with timing instrumentation
def create_instrumented_compute_sample_scores(original_method):
    """Wrap compute_sample_scores with detailed timing."""

    def instrumented_compute_sample_scores(self, image_url, text, download_image=True):
        print(f"\n{'='*80}")
        print(f"🔍 TIMING ANALYSIS FOR SAMPLE: {image_url[:60]}...")
        print(f"{'='*80}")

        total_start = time.time()
        timings = {}

        try:
            # Time: Cache check
            step_start = time.time()
            cached = self.sample_cache.get_cached_sample(image_url) if self.sample_cache else None
            timings['1_cache_check'] = time.time() - step_start
            print(f"⏱️  Cache check: {timings['1_cache_check']:.4f}s {'(HIT)' if cached else '(MISS)'}")

            # Time: Cache processing or download
            if cached and cached.get('text') == text:
                step_start = time.time()
                # Extract cached data
                visual_metrics = cached['visual_metrics']
                semantic_metrics = cached['semantic_metrics']
                efficientnet_data = cached.get('efficientnet_predictions', {})
                clip_data = cached.get('clip_embeddings', {})

                if clip_data and 'image_embedding' in clip_data and 'text_embedding' in clip_data:
                    img_embedding = clip_data['image_embedding']
                    text_embedding = clip_data['text_embedding']
                    image = None
                else:
                    # Need to load image and compute embeddings
                    if self.cache_manager:
                        image = self.cache_manager.get_cached_image(image_url)
                    else:
                        import requests
                        from io import BytesIO
                        response = requests.get(image_url, timeout=self.request_timeout)
                        image = Image.open(BytesIO(response.content)).convert('RGB')

                    if image is None:
                        print(f"❌ Failed to load cached image")
                        return {}, {}

                    # Compute embeddings
                    import torch
                    with torch.no_grad():
                        img_input = self.preprocess(image).unsqueeze(0).to(self.device)
                        img_embedding = self.model.encode_image(img_input)
                        img_embedding = img_embedding / img_embedding.norm(dim=-1, keepdim=True)
                        img_embedding = img_embedding.cpu().numpy()

                        import clip
                        text_tokens = clip.tokenize([text], truncate=True).to(self.device)
                        text_embedding = self.model.encode_text(text_tokens)
                        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
                        text_embedding = text_embedding.cpu().numpy()

                timings['2_cache_data_extraction'] = time.time() - step_start
                print(f"⏱️  Cache data extraction: {timings['2_cache_data_extraction']:.4f}s")

            else:
                # Cache miss - full computation
                step_start = time.time()
                if download_image and self.cache_manager:
                    image = self.cache_manager.download_and_cache_image(
                        image_url,
                        max_retries=self.max_retries,
                        timeout=self.request_timeout
                    )
                else:
                    import requests
                    from io import BytesIO
                    response = requests.get(image_url, timeout=self.request_timeout)
                    from PIL import Image
                    image = Image.open(BytesIO(response.content)).convert('RGB')

                timings['2_download'] = time.time() - step_start
                print(f"⏱️  Image download: {timings['2_download']:.4f}s")

                if image is None:
                    print(f"❌ Failed to download image")
                    return {}, {}

                # Compute CLIP embeddings
                step_start = time.time()
                import torch
                with torch.no_grad():
                    img_input = self.preprocess(image).unsqueeze(0).to(self.device)
                    img_embedding = self.model.encode_image(img_input)
                    img_embedding = img_embedding / img_embedding.norm(dim=-1, keepdim=True)
                    img_embedding = img_embedding.cpu().numpy()

                    import clip
                    text_tokens = clip.tokenize([text], truncate=True).to(self.device)
                    text_embedding = self.model.encode_text(text_tokens)
                    text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
                    text_embedding = text_embedding.cpu().numpy()

                timings['3_clip_embeddings'] = time.time() - step_start
                print(f"⏱️  CLIP embeddings: {timings['3_clip_embeddings']:.4f}s")

                # Compute visual metrics
                step_start = time.time()
                visual_metrics = {}
                metadata = {'url': image_url, 'text': text}
                for metric_name in ['resolution', 'sharpness', 'color_diversity']:
                    if metric_name in self.quality_metrics:
                        try:
                            metric = self.quality_metrics[metric_name]
                            score = metric.compute(image, metadata)
                            visual_metrics[metric.metric_name] = round(float(score), 5)
                        except Exception as e:
                            visual_metrics[metric.metric_name] = 0.0
                timings['4_visual_metrics'] = time.time() - step_start
                print(f"⏱️  Visual metrics: {timings['4_visual_metrics']:.4f}s")

                # Compute semantic metrics
                step_start = time.time()
                semantic_metrics = {}
                for metric_name in ['text_quality', 'caption_length']:
                    if metric_name in self.quality_metrics:
                        try:
                            metric = self.quality_metrics[metric_name]
                            score = metric.compute(image, metadata)
                            semantic_metrics[metric.metric_name] = round(float(score), 5)
                        except Exception as e:
                            semantic_metrics[metric.metric_name] = 0.0
                timings['5_semantic_metrics'] = time.time() - step_start
                print(f"⏱️  Semantic metrics: {timings['5_semantic_metrics']:.4f}s")

                # Compute EfficientNet if enabled
                step_start = time.time()
                efficientnet_data = {}
                if self.enable_target_class_quality and 'target_class_quality' in self.quality_metrics:
                    try:
                        metric = self.quality_metrics['target_class_quality']
                        pred_class, pred_prob = metric.compute_single_imagenet_prediction(image)
                        efficientnet_data = {
                            'imagenet_predicted_class': pred_class,
                            'imagenet_probability': round(float(pred_prob), 5)
                        }
                    except Exception as e:
                        efficientnet_data = {
                            'imagenet_predicted_class': "",
                            'imagenet_probability': 0.0
                        }
                timings['6_efficientnet'] = time.time() - step_start
                print(f"⏱️  EfficientNet: {timings['6_efficientnet']:.4f}s")

                # Cache the sample
                step_start = time.time()
                if self.sample_cache:
                    self.sample_cache.cache_sample(
                        url=image_url,
                        text=text,
                        visual_metrics=visual_metrics,
                        semantic_metrics=semantic_metrics,
                        image_embedding=img_embedding,
                        text_embedding=text_embedding,
                        efficientnet_data=efficientnet_data if efficientnet_data else None
                    )
                timings['7_cache_save'] = time.time() - step_start
                print(f"⏱️  Cache save: {timings['7_cache_save']:.4f}s")

            # Compute per-class similarity scores
            step_start = time.time()
            if not self.reference_embeddings:
                print(f"❌ No reference embeddings")
                return {}, {}

            target_classes = list(self.reference_embeddings.keys())
            print(f"📊 Computing scores for {len(target_classes)} classes...")

            # Per-class loop timing
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np

            class_loop_start = time.time()
            class_scores = {}

            for class_name in target_classes:
                img2img = cosine_similarity(img_embedding, self.reference_embeddings[class_name]).max()
                txt2txt = cosine_similarity(text_embedding, self.text_embeddings[class_name]).max()
                img2txt = cosine_similarity(img_embedding, self.text_embeddings[class_name]).max()
                txt2img = cosine_similarity(text_embedding, self.reference_embeddings[class_name]).max()

                hybrid_score = 0.25 * (img2img + txt2txt + img2txt + txt2img)
                similarities = [img2img, txt2txt, img2txt, txt2img]
                consistency = 1.0 - np.std(similarities)

                class_scores[class_name] = {
                    'img2img': round(float(img2img), 5),
                    'txt2txt': round(float(txt2txt), 5),
                    'img2txt': round(float(img2txt), 5),
                    'txt2img': round(float(txt2img), 5),
                    'hybrid_score': round(float(hybrid_score), 5),
                    'consistency': round(float(consistency), 5),
                }

            timings['8_per_class_scores'] = time.time() - class_loop_start
            print(f"⏱️  Per-class similarity ({len(target_classes)} classes): {timings['8_per_class_scores']:.4f}s")
            print(f"   └─ Per-class average: {timings['8_per_class_scores']/len(target_classes):.4f}s")

            # Build quality scores
            quality_scores = {}
            quality_scores.update(visual_metrics)
            quality_scores.update(semantic_metrics)
            if efficientnet_data:
                quality_scores.update(efficientnet_data)

            total_time = time.time() - total_start
            timings['TOTAL'] = total_time

            print(f"\n{'='*80}")
            print(f"📊 TOTAL TIME: {total_time:.4f}s")
            print(f"{'='*80}")
            print(f"\n⚠️  BREAKDOWN:")
            for step, duration in sorted(timings.items()):
                percentage = (duration / total_time) * 100 if total_time > 0 else 0
                print(f"   {step:30s}: {duration:8.4f}s ({percentage:5.1f}%)")
            print(f"{'='*80}\n")

            return class_scores, quality_scores

        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return {}, {}

    return instrumented_compute_sample_scores


# Apply the patch
print("🔧 Applying timing instrumentation to Retriever...")
from maveric.retrieval import Retriever
Retriever.compute_sample_scores = create_instrumented_compute_sample_scores(Retriever.compute_sample_scores)
print("✅ Instrumentation applied!")
print("\nNow run your 01_data_retrieval.py script - you'll see detailed timing for each sample.\n")
