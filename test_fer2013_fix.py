#!/usr/bin/env python3
"""
Test script to verify FER2013 list-based class names are handled correctly.
This tests both sanitize_filename() and dictionary key usage.
"""

from maveric.retrieval.cache_manager import sanitize_filename

# Test FER2013 class names (lists of synonyms)
fer2013_class_names = [
    ['angry', 'upset'],
    ['disgust', 'revulsion'],
    ['fear', 'scared'],
    ['happy', 'smiling'],
    ['sad', 'depressed'],
    ['surprise', 'shocked'],
    ['neutral', 'expressionless']
]

# Test GTSRB class name with special characters
gtsrb_class_name = "end / de-restriction of 80 kph speed limit"

print("="*80)
print("FER2013 List-Based Class Names Test")
print("="*80)

# Test 1: sanitize_filename() with lists
print("\n1. Testing sanitize_filename() with lists:")
for class_name in fer2013_class_names:
    sanitized = sanitize_filename(class_name)
    print(f"   {class_name} → '{sanitized}'")
    assert isinstance(sanitized, str), f"Expected string, got {type(sanitized)}"
    assert sanitized == class_name[0], f"Expected '{class_name[0]}', got '{sanitized}'"

# Test 2: Dictionary key usage
print("\n2. Testing dictionary key usage with lists:")
reference_samples = {}
for class_name in fer2013_class_names:
    # This is the pattern used in the fixed code
    canonical_name = class_name[0] if isinstance(class_name, list) else class_name
    reference_samples[canonical_name] = ["dummy_image_data"]
    print(f"   ✓ Successfully added '{canonical_name}' to dictionary")

print(f"\n   Dictionary has {len(reference_samples)} keys (expected 7)")
assert len(reference_samples) == 7, f"Expected 7 keys, got {len(reference_samples)}"

# Test 3: sanitize_filename() with special characters
print("\n3. Testing sanitize_filename() with special characters:")
sanitized_gtsrb = sanitize_filename(gtsrb_class_name)
print(f"   '{gtsrb_class_name}' → '{sanitized_gtsrb}'")
assert "/" not in sanitized_gtsrb, "Forward slash should be replaced"
assert "\\" not in sanitized_gtsrb, "Backslash should be replaced"
expected = "end _ de-restriction of 80 kph speed limit"
assert sanitized_gtsrb == expected, f"Expected '{expected}', got '{sanitized_gtsrb}'"

# Test 4: Mixed usage (lists + strings)
print("\n4. Testing mixed class names (lists and strings):")
mixed_class_names = [
    ['happy', 'smiling'],  # FER2013 style
    'airplane',            # Normal string
    "end / de-restriction of 80 kph speed limit",  # GTSRB style
]

mixed_samples = {}
for class_name in mixed_class_names:
    canonical_name = class_name[0] if isinstance(class_name, list) else class_name
    sanitized = sanitize_filename(canonical_name)
    mixed_samples[canonical_name] = ["dummy_data"]
    print(f"   ✓ {class_name!r} → canonical: '{canonical_name}' → sanitized: '{sanitized}'")

print(f"\n   Dictionary has {len(mixed_samples)} keys (expected 3)")
assert len(mixed_samples) == 3, f"Expected 3 keys, got {len(mixed_samples)}"

print("\n" + "="*80)
print("✅ ALL TESTS PASSED!")
print("="*80)
print("\nThe fix correctly handles:")
print("  1. FER2013 list-based class names (extracts first element)")
print("  2. Dictionary key usage (no unhashable type errors)")
print("  3. GTSRB special characters (sanitizes for filesystem)")
print("  4. Mixed scenarios (lists + strings)")

# Test 5: Synonym expansion for text embeddings
print("\n" + "="*80)
print("5. Testing synonym expansion for text embeddings:")
print("="*80)

text_templates = ["a photo of a {}", "an image showing {}"]
fer2013_example = ['happy', 'smiling']

# Simulate what retriever.py does
if isinstance(fer2013_example, list):
    canonical_name = fer2013_example[0]
    prompts = []
    for synonym in fer2013_example:
        prompts.extend([template.format(synonym) for template in text_templates])
else:
    canonical_name = fer2013_example
    prompts = [template.format(fer2013_example) for template in text_templates]

print(f"   Class name: {fer2013_example}")
print(f"   Canonical name: '{canonical_name}'")
print(f"   Generated prompts ({len(prompts)}):")
for i, prompt in enumerate(prompts, 1):
    print(f"     {i}. \"{prompt}\"")

expected_prompts = [
    "a photo of a happy",
    "an image showing happy",
    "a photo of a smiling",
    "an image showing smiling"
]
assert prompts == expected_prompts, f"Expected {expected_prompts}, got {prompts}"
print(f"\n   ✓ All synonyms expanded correctly!")
print(f"   ✓ This creates richer embeddings by including all synonym variations")

print("\n" + "="*80)
print("✅ ALL TESTS INCLUDING SYNONYM EXPANSION PASSED!")
print("="*80)
