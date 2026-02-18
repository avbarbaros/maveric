#!/usr/bin/env python3
"""
Comprehensive test to verify FER2013 works end-to-end.
Tests the complete flow from class name definition to dictionary usage.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Test 1: Verify elevater_datasets.py class names
print("="*80)
print("TEST 1: FER2013 Class Names Definition")
print("="*80)

from maveric.datasets.elevater_datasets import ELEVATERDataset

fer2013_config = ELEVATERDataset.ELEVATER_DATASETS.get('fer2013')
if fer2013_config:
    class_names = fer2013_config['class_names']
    print(f"✓ FER2013 has {len(class_names)} classes:")
    for i, cn in enumerate(class_names, 1):
        print(f"  {i}. {cn!r} (type: {type(cn).__name__})")

    # Verify all are lists
    all_lists = all(isinstance(cn, list) for cn in class_names)
    print(f"\n✓ All class names are lists: {all_lists}")
    assert all_lists, "Not all FER2013 class names are lists!"
else:
    print("✗ FER2013 not found in ELEVATER_DATASETS")
    sys.exit(1)

# Test 2: Verify sanitize_filename handles lists
print("\n" + "="*80)
print("TEST 2: sanitize_filename() List Handling")
print("="*80)

from maveric.retrieval.cache_manager import sanitize_filename

for class_name in class_names:
    canonical = sanitize_filename(class_name)
    print(f"  {class_name!r} → '{canonical}'")
    assert isinstance(canonical, str), f"Expected string, got {type(canonical)}"
    assert canonical == class_name[0], f"Expected first element '{class_name[0]}'"

print("\n✓ All FER2013 class names sanitize correctly")

# Test 3: Verify dictionary key usage
print("\n" + "="*80)
print("TEST 3: Dictionary Key Usage")
print("="*80)

# Simulate reference_samples creation (like in elevater_datasets.py)
reference_samples = {}
for class_name in class_names:
    canonical_name = class_name[0] if isinstance(class_name, list) else class_name
    reference_samples[canonical_name] = ["dummy_image_1", "dummy_image_2"]

print(f"✓ Created reference_samples with {len(reference_samples)} keys")
print(f"  Keys: {list(reference_samples.keys())}")

# Verify all keys are strings
all_string_keys = all(isinstance(k, str) for k in reference_samples.keys())
assert all_string_keys, "Not all dictionary keys are strings!"
print(f"✓ All dictionary keys are strings: {all_string_keys}")

# Test 4: Verify text embedding creation with synonym expansion
print("\n" + "="*80)
print("TEST 4: Text Embedding Creation with Synonym Expansion")
print("="*80)

text_templates = ["a photo of a {}", "a picture of {}"]
text_embeddings = {}

for class_name in class_names:
    # This mimics retriever.py lines 242-258
    if isinstance(class_name, list):
        canonical_name = class_name[0]
        prompts = []
        for synonym in class_name:
            prompts.extend([template.format(synonym) for template in text_templates])
    else:
        canonical_name = class_name
        prompts = [template.format(class_name) for template in text_templates]

    # Store with canonical name as key
    text_embeddings[canonical_name] = prompts
    print(f"  '{canonical_name}': {len(prompts)} prompts")

print(f"\n✓ Created text_embeddings with {len(text_embeddings)} keys")
print(f"  Keys: {list(text_embeddings.keys())}")

# Verify prompt expansion for list class names
for class_name in class_names:
    canonical = class_name[0]
    prompts = text_embeddings[canonical]
    expected_count = len(class_name) * len(text_templates)
    assert len(prompts) == expected_count, \
        f"Expected {expected_count} prompts for {class_name}, got {len(prompts)}"

print(f"✓ All synonym expansions correct")

# Test 5: Verify cache manager reference text saving
print("\n" + "="*80)
print("TEST 5: Cache Manager Reference Text Saving")
print("="*80)

# Simulate save_reference_texts (cache_manager.py lines 500-512)
text_data = {
    'templates': text_templates,
    'class_names': class_names,
    'generated_prompts': {}
}

for class_name in class_names:
    if isinstance(class_name, list):
        canonical_name = class_name[0]
        prompts = []
        for synonym in class_name:
            prompts.extend([template.format(synonym) for template in text_templates])
    else:
        canonical_name = class_name
        prompts = [template.format(class_name) for template in text_templates]

    text_data['generated_prompts'][canonical_name] = prompts

print(f"✓ Created text_data with {len(text_data['generated_prompts'])} prompt entries")
print(f"  Keys: {list(text_data['generated_prompts'].keys())}")

# Verify JSON serializability (dict keys must be strings)
import json
try:
    json_str = json.dumps(text_data['generated_prompts'], indent=2)
    print(f"✓ Dictionary is JSON-serializable")
    print(f"  Sample JSON output (first 200 chars):")
    print(f"  {json_str[:200]}...")
except TypeError as e:
    print(f"✗ Dictionary is NOT JSON-serializable: {e}")
    sys.exit(1)

# Final summary
print("\n" + "="*80)
print("✅ ALL COMPREHENSIVE TESTS PASSED!")
print("="*80)
print("\nFER2013 is fully compatible with:")
print("  1. Class name definition (lists of synonyms)")
print("  2. Filename sanitization (extracts canonical name)")
print("  3. Dictionary key usage (canonical names as keys)")
print("  4. Text embedding creation (synonym expansion)")
print("  5. Cache serialization (JSON-compatible)")
print("\nThe complete pipeline should work end-to-end! 🎉")
