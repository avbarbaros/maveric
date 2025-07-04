"""Basic usage example for MAVERIC."""

from maveric import MAVERIC, MAVERICConfig, TrainingConfig

# Initialize MAVERIC
config = MAVERICConfig(
    cache_base_dir="/path/to/cache",
    clip_model="ViT-B/32",
    device="cuda"
)
maveric = MAVERIC(config)

# Step 1: Retrieve samples
print("Retrieving samples...")
retrieval_result = maveric.retrieve(
    dataset_name="react-vl/react-retrieval-datasets",
    target_dataset="cifar100",
    num_samples=100000  # Retrieve 100k samples
)
print(f"Retrieved {retrieval_result.total_samples} samples")

# Step 2: Apply quality control
print("\nApplying quality control...")
quality_result = maveric.quality_control(
    retrieval_result,
    thresholds={
        'sharpness_score': 0.85,
        'consistency': 0.80,
        'resolution_score': 0.4
    },
    balance_strategy='median'
)
print(f"Filtered to {quality_result.filtered_count} high-quality samples")
print(f"Retention rate: {quality_result.retention_rate:.1%}")

# Step 3: Customize model
print("\nCustomizing model...")
training_config = TrainingConfig(
    epochs=10,
    learning_rate=1e-5,
    use_augmentation=True
)

customization_result = maveric.customize_model(
    quality_result,
    training_config=training_config,
    target_dataset="cifar100"
)

print(f"\nResults:")
print(f"Zero-shot baseline: {customization_result.zero_shot_baseline:.2f}%")
print(f"After customization: {customization_result.test_accuracy:.2f}%")
print(f"Improvement: {customization_result.improvement:+.2f}%")

# Save results
quality_result.to_dataframe().to_csv("filtered_data.csv", index=False)
print("\nSaved filtered data to filtered_data.csv")
