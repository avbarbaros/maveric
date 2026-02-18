#!/usr/bin/env python3
"""
Test script to verify the optimized ImageNet mapping implementation.
Tests that the batch processing produces the same results as individual processing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from PIL import Image
from maveric.quality.metrics.multimodal_metrics import TargetClassQualityMetric

def create_test_image():
    """Create a simple test image."""
    # Create a 224x224 RGB image with some patterns
    image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(image_array)

def test_optimization():
    """Test that the optimized batch processing produces identical results."""
    print("🧪 Testing ImageNet mapping optimization...")
    
    # Initialize the metric
    metric = TargetClassQualityMetric()
    
    # Create test image and target classes
    test_image = create_test_image()
    target_classes = ['airplane', 'bird', 'car', 'dog', 'cat']
    
    print(f"📊 Testing with {len(target_classes)} target classes: {target_classes}")
    
    # Method 1: Individual processing (old way - for comparison)
    individual_results = {}
    print("\n🔍 Computing individual mappings (old method)...")
    for target_class in target_classes:
        metadata = {'label': target_class, 'target_class': target_class}
        _, best_class, probability = metric.compute_with_best_mapping(test_image, metadata)
        individual_results[target_class] = (best_class, probability)
        print(f"  {target_class:>8} → {best_class} ({probability:.5f})")
    
    # Method 2: Batch processing (new optimized way)
    print("\n⚡ Computing batch mappings (optimized method)...")
    probabilities = metric.compute_image_probabilities_only(test_image)
    batch_results = metric.compute_all_mappings_from_probabilities(probabilities, target_classes)
    
    for target_class in target_classes:
        best_class, probability = batch_results[target_class]
        print(f"  {target_class:>8} → {best_class} ({probability:.5f})")
    
    # Verify results are identical
    print("\n🔍 Verifying results match...")
    all_match = True
    for target_class in target_classes:
        individual = individual_results[target_class]
        batch = batch_results[target_class]
        
        if individual != batch:
            print(f"❌ Mismatch for {target_class}: {individual} vs {batch}")
            all_match = False
        else:
            print(f"✅ {target_class}: Results match")
    
    if all_match:
        print("\n🎉 SUCCESS: Optimized batch processing produces identical results!")
        print("💡 The optimization correctly reduces EfficientNet calls from N to 1 per image.")
        return True
    else:
        print("\n❌ FAILURE: Results don't match between methods")
        return False

if __name__ == "__main__":
    try:
        success = test_optimization()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)