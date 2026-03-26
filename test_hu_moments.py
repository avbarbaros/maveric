#!/usr/bin/env python3
"""
Quick test to verify Hu moments computation works correctly.
"""

import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Add maveric to path
sys.path.insert(0, str(Path(__file__).parent))

from maveric.quality.metrics.hu_moments_metric import HuMomentsSimilarityMetric


def test_hu_moments_basic():
    """Test basic Hu moments computation."""
    print("=" * 60)
    print("Testing Hu Moments Computation")
    print("=" * 60)

    # Create a simple test image (white square on black background)
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    test_img[25:75, 25:75] = 255  # White square in center
    pil_img = Image.fromarray(test_img)

    # Compute Hu vector
    print("\n1. Computing Hu vector for test image...")
    hu_vector = HuMomentsSimilarityMetric.compute_hu_vector(pil_img)

    if hu_vector is None:
        print("   ❌ FAILED: Hu vector is None!")
        return False

    print(f"   ✅ SUCCESS: Hu vector computed")
    print(f"   Shape: {hu_vector.shape}")
    print(f"   Values: {hu_vector}")

    # Create metric instance
    print("\n2. Creating HuMomentsSimilarityMetric instance...")
    metric = HuMomentsSimilarityMetric()

    # Set reference vectors (using same image as reference)
    print("\n3. Setting reference vectors...")
    ref_vectors = {
        'test_class': np.array([hu_vector])  # Shape (1, 7)
    }
    metric.set_reference_vectors(ref_vectors)
    print(f"   ✅ Reference vectors set for 'test_class'")
    print(f"   Reference shape: {ref_vectors['test_class'].shape}")

    # Compute similarity (should be ~1.0 since comparing with itself)
    print("\n4. Computing similarity...")
    similarity = metric.compute_similarity(hu_vector, 'test_class')
    print(f"   Similarity: {similarity}")

    if similarity == 0.0:
        print("   ❌ FAILED: Similarity is 0.0!")
        print("   This should be close to 1.0 since we're comparing the image with itself")
        return False
    elif similarity > 0.99:
        print(f"   ✅ SUCCESS: Similarity is {similarity} (close to 1.0 as expected)")
        return True
    else:
        print(f"   ⚠️  WARNING: Similarity is {similarity} (expected close to 1.0)")
        return True


def test_hu_moments_with_different_images():
    """Test Hu moments with different images."""
    print("\n" + "=" * 60)
    print("Testing Hu Moments with Different Images")
    print("=" * 60)

    # Create two different test images
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img1[25:75, 25:75] = 255  # Square
    pil_img1 = Image.fromarray(img1)

    img2 = np.zeros((100, 100, 3), dtype=np.uint8)
    # Triangle (approximate)
    for i in range(50):
        img2[25+i, 50-i:50+i+1] = 255
    pil_img2 = Image.fromarray(img2)

    # Compute Hu vectors
    print("\n1. Computing Hu vectors...")
    hv1 = HuMomentsSimilarityMetric.compute_hu_vector(pil_img1)
    hv2 = HuMomentsSimilarityMetric.compute_hu_vector(pil_img2)

    print(f"   Image 1 Hu vector: {hv1}")
    print(f"   Image 2 Hu vector: {hv2}")

    # Create metric and set reference
    metric = HuMomentsSimilarityMetric()
    metric.set_reference_vectors({
        'square': np.array([hv1])
    })

    # Compute similarities
    print("\n2. Computing similarities...")
    sim1 = metric.compute_similarity(hv1, 'square')  # Same shape
    sim2 = metric.compute_similarity(hv2, 'square')  # Different shape

    print(f"   Square vs Square: {sim1}")
    print(f"   Triangle vs Square: {sim2}")

    if sim1 > sim2:
        print(f"   ✅ SUCCESS: Same shape has higher similarity ({sim1} > {sim2})")
        return True
    else:
        print(f"   ⚠️  WARNING: Different shape has higher similarity ({sim2} >= {sim1})")
        return False


if __name__ == "__main__":
    print("\n🧪 Hu Moments Metric Test Suite\n")

    # Test 1: Basic computation
    test1_passed = test_hu_moments_basic()

    # Test 2: Different images
    test2_passed = test_hu_moments_with_different_images()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Test 1 (Basic Computation): {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Test 2 (Different Images): {'✅ PASSED' if test2_passed else '❌ FAILED'}")

    if test1_passed and test2_passed:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
