#!/usr/bin/env python3
"""
Quick test to check if cache I/O is the bottleneck.
This bypasses MAVERIC and directly tests cache read/write speed.
"""

import time
import json
import numpy as np
import base64
from pathlib import Path

def encode_embedding(embedding):
    """Encode numpy array to base64 (same as in cache)."""
    return base64.b64encode(embedding.tobytes()).decode('utf-8')

def decode_embedding(encoded, shape, dtype):
    """Decode base64 to numpy array."""
    decoded = base64.b64decode(encoded.encode('utf-8'))
    return np.frombuffer(decoded, dtype=dtype).reshape(shape)

def test_json_cache_performance(cache_dir, num_tests=10):
    """Test JSON cache read/write performance."""

    cache_dir = Path(cache_dir)
    test_dir = cache_dir / "performance_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create test data (similar to real cache)
    test_embedding = np.random.randn(1, 512).astype(np.float32)
    test_data = {
        'cache_version': 3,
        'url': 'https://test.com/image.jpg',
        'url_hash': 'abc123def456',
        'text': 'A test image caption',
        'last_updated': '2025-11-20T00:00:00',
        'visual_metrics': {
            'resolution_score': 0.895,
            'sharpness_score': 0.923,
            'color_score': 0.812
        },
        'semantic_metrics': {
            'text_quality_score': 0.850,
            'caption_length_score': 0.920
        },
        'clip_embeddings': {
            'image_embedding': encode_embedding(test_embedding),
            'text_embedding': encode_embedding(test_embedding),
            'image_shape': [1, 512],
            'text_shape': [1, 512],
            'dtype': 'float32'
        },
        'efficientnet_predictions': {
            'imagenet_predicted_class': 'tabby cat',
            'imagenet_probability': 0.892
        }
    }

    print("=" * 80)
    print("CACHE PERFORMANCE TEST")
    print("=" * 80)
    print(f"Cache directory: {cache_dir}")
    print(f"Test samples: {num_tests}")
    print(f"Data size: {len(json.dumps(test_data))} bytes (~{len(json.dumps(test_data))/1024:.1f} KB)")
    print()

    # Test 1: Write performance
    print("📝 Testing WRITE performance...")
    write_times = []
    for i in range(num_tests):
        test_file = test_dir / f"test_sample_{i}.json"

        start = time.time()
        with open(test_file, 'w') as f:
            json.dump(test_data, f, indent=None)
        write_time = time.time() - start
        write_times.append(write_time)

        if i < 3:  # Show first 3
            print(f"   Sample {i+1}: {write_time:.4f}s")

    avg_write = np.mean(write_times)
    print(f"   Average write time: {avg_write:.4f}s")
    print()

    # Test 2: Read performance
    print("📖 Testing READ performance...")
    read_times = []
    for i in range(num_tests):
        test_file = test_dir / f"test_sample_{i}.json"

        start = time.time()
        with open(test_file, 'r') as f:
            data = json.load(f)
        read_time = time.time() - start
        read_times.append(read_time)

        if i < 3:  # Show first 3
            print(f"   Sample {i+1}: {read_time:.4f}s")

    avg_read = np.mean(read_times)
    print(f"   Average read time: {avg_read:.4f}s")
    print()

    # Test 3: Base64 decode performance
    print("🔓 Testing BASE64 DECODE performance...")
    decode_times = []
    for i in range(num_tests):
        test_file = test_dir / f"test_sample_{i}.json"

        with open(test_file, 'r') as f:
            data = json.load(f)

        start = time.time()
        clip_data = data['clip_embeddings']
        img_emb = decode_embedding(
            clip_data['image_embedding'],
            tuple(clip_data['image_shape']),
            np.float32
        )
        txt_emb = decode_embedding(
            clip_data['text_embedding'],
            tuple(clip_data['text_shape']),
            np.float32
        )
        decode_time = time.time() - start
        decode_times.append(decode_time)

        if i < 3:  # Show first 3
            print(f"   Sample {i+1}: {decode_time:.4f}s")

    avg_decode = np.mean(decode_times)
    print(f"   Average decode time: {avg_decode:.4f}s")
    print()

    # Summary
    total_per_sample = avg_read + avg_decode

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Average WRITE time:  {avg_write:.4f}s")
    print(f"Average READ time:   {avg_read:.4f}s")
    print(f"Average DECODE time: {avg_decode:.4f}s")
    print(f"TOTAL per sample:    {total_per_sample:.4f}s (read + decode)")
    print()

    if total_per_sample > 1.0:
        print("🚨 WARNING: Cache I/O is VERY SLOW (>1s per sample)!")
        print("   This is likely due to Google Drive NFS being slow.")
        print("   Recommendation: Use local disk for cache instead.")
    elif total_per_sample > 0.1:
        print("⚠️  WARNING: Cache I/O is moderately slow (>0.1s per sample)")
        print("   This could contribute to slowdown, especially with 10+ classes.")
    else:
        print("✅ Cache I/O performance is good (<0.1s per sample)")
    print()

    # Cleanup
    print("🧹 Cleaning up test files...")
    for i in range(num_tests):
        test_file = test_dir / f"test_sample_{i}.json"
        test_file.unlink()
    test_dir.rmdir()
    print("✅ Done!")
    print("=" * 80)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cache_dir = sys.argv[1]
    else:
        # Default to your Google Drive cache
        cache_dir = "/content/drive/MyDrive/MAVERIC/maveric_cache"

    print(f"\n🔬 Testing cache performance at: {cache_dir}\n")

    test_json_cache_performance(cache_dir, num_tests=20)
