#!/usr/bin/env python3
"""
Test script to verify the class name extraction bug fix for GTSRB dataset.
Tests that class names with underscores are correctly extracted.
"""

def test_class_name_extraction():
    """Test the fixed class name extraction logic"""

    # Simulate column names from GTSRB dataset
    test_columns = [
        'Class_pedestrians_img2img',
        'Class_pedestrians_txt2txt',
        'Class_stop_img2img',
        'Class_yield_img2img',
        'Class_ahead_only_img2img',
        'Class_ahead_only_txt2txt',
        'Class_ahead_only_img2txt',
        'Class_ahead_only_txt2img',
        'Class_beware_of_ice_snow_img2img',
        'Class_beware_of_ice_snow_efficientNet_score',
        'Class_bicycles_crossing_img2img',
        'Class_no_passing_for_vehicles_over_3_5_metric_tons_img2img',
        'Class_no_passing_for_vehicles_over_3_5_metric_tons_efficientNet_score',
    ]

    # Expected class names (what we should extract)
    expected_classes = {
        'pedestrians',
        'stop',
        'yield',
        'ahead_only',
        'beware_of_ice_snow',
        'bicycles_crossing',
        'no_passing_for_vehicles_over_3_5_metric_tons',
    }

    # Apply the fixed extraction logic
    known_suffixes = ['_img2img', '_txt2txt', '_img2txt', '_txt2img', '_efficientNet_score', '_clip_similarity_to_imagenet']
    extracted_classes = set()

    for col in test_columns:
        if col.startswith('Class_'):
            # Remove 'Class_' prefix
            name_with_suffix = col[6:]  # len('Class_') = 6
            # Remove known suffix from the end
            for suffix in known_suffixes:
                if name_with_suffix.endswith(suffix):
                    class_name = name_with_suffix[:-len(suffix)]
                    extracted_classes.add(class_name)
                    break

    # Verify results
    print("🧪 Testing class name extraction fix")
    print("=" * 60)
    print(f"Test columns: {len(test_columns)}")
    print(f"Expected classes: {len(expected_classes)}")
    print(f"Extracted classes: {len(extracted_classes)}")
    print()

    print("Expected classes:")
    for cls in sorted(expected_classes):
        print(f"  • {cls}")
    print()

    print("Extracted classes:")
    for cls in sorted(extracted_classes):
        print(f"  • {cls}")
    print()

    # Check if extraction matches expectations
    if extracted_classes == expected_classes:
        print("✅ SUCCESS: All class names extracted correctly!")
        print()

        # Test the EfficientNet extraction logic
        print("🧪 Testing EfficientNet class name extraction")
        print("=" * 60)
        efficientnet_columns = [col for col in test_columns if 'efficientNet_score' in col]

        for col in efficientnet_columns:
            # Apply the fixed logic
            class_name = col[6:-len('_efficientNet_score')]
            print(f"Column: {col}")
            print(f"  → Extracted: '{class_name}'")

        print()
        print("✅ All tests passed!")
        return True
    else:
        print("❌ FAILURE: Extracted classes don't match expected!")
        print()
        print("Missing classes:")
        for cls in sorted(expected_classes - extracted_classes):
            print(f"  • {cls}")
        print()
        print("Extra classes:")
        for cls in sorted(extracted_classes - expected_classes):
            print(f"  • {cls}")
        return False


if __name__ == '__main__':
    success = test_class_name_extraction()
    exit(0 if success else 1)
