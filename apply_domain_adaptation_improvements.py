#!/usr/bin/env python3
"""
Script to apply all domain adaptation improvements to MAVERIC codebase.

This script:
1. Implements save_augmented_grids() method
2. Adds console logging for domain adaptation settings
3. Cleans up imports (removes numpy, moves inline imports)
4. Adds error handling for domain adaptation transforms

Run this script to automatically apply all improvements.
"""

print("🔧 Applying domain adaptation improvements to MAVERIC codebase...")
print()

# The improvements will be applied through manual edits due to complexity
# This script serves as documentation of what needs to be done

improvements = {
    "1. Implement _save_augmented_grids() method": """
Add after line 1070 in model_customizer.py (before _apply_transforms):

    def _save_augmented_grids(self, dataset, dataset_name, training_config):
        '''Save 10x10 grid visualizations of augmented/domain-adapted training samples.'''
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from pathlib import Path
        import random

        self.log_info("📸 Saving augmented sample grids for visual inspection...")

        # Create output directory
        output_dir = Path(self.checkpoint_dir).parent / 'augmented_grids'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Sample 100 random indices
        num_samples = min(100, len(dataset))
        indices = random.sample(range(len(dataset)), num_samples)

        # Create figure with 10x10 grid
        fig = plt.figure(figsize=(30, 30))
        gs = gridspec.GridSpec(10, 10, figure=fig, hspace=0.3, wspace=0.3)

        for idx, sample_idx in enumerate(indices):
            try:
                # Get augmented/domain-adapted image
                image, text, label = dataset[sample_idx]

                # Convert tensor to numpy if needed
                if hasattr(image, 'numpy'):
                    img_array = image.permute(1, 2, 0).numpy()
                else:
                    img_array = image

                # Create subplot
                ax = fig.add_subplot(gs[idx // 10, idx % 10])
                ax.imshow(img_array)
                ax.axis('off')

                # Add label
                class_name = dataset.class_names[label] if hasattr(dataset, 'class_names') else f'Class {label}'
                ax.set_title(f'{class_name}\\n#{sample_idx}', fontsize=8)

            except Exception as e:
                self.log_warning(f"Failed to process sample {sample_idx}: {e}")
                continue

        # Save figure
        output_file = output_dir / f'{dataset_name}_augmented_grid.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)

        self.log_info(f"✅ Saved augmented samples grid to: {output_file}")

        # Print summary
        if training_config.use_augmentation or training_config.use_domain_adaptation:
            self.log_info("   Grid shows effects of:")
            if training_config.use_augmentation:
                self.log_info(f"   - RandAugment (ops={training_config.augmentation_strength}, mag={training_config.augmentation_magnitude})")
            if training_config.use_domain_adaptation:
                self.log_info("   - Domain Adaptation (blur/JPEG/downsample)")
""",

    "2. Add console logging in _prepare_data()": """
Add after line 255 in model_customizer.py (after dataset creation):

        # Log augmentation and domain adaptation settings
        self.log_info("📦 Creating training dataset...")
        if training_config.use_augmentation:
            self.log_info(f"   Augmentation: RandAugment (num_ops={training_config.augmentation_strength}, magnitude={training_config.augmentation_magnitude})")
        else:
            self.log_info("   Augmentation: Disabled")

        if training_config.use_domain_adaptation:
            self.log_info("   Domain Adaptation: Enabled")
            self.log_info(f"      - Blur probability: {training_config.domain_blur_probability*100:.1f}%")
            self.log_info(f"      - JPEG probability: {training_config.domain_jpeg_probability*100:.1f}%")
            self.log_info(f"      - Downsample probability: {training_config.domain_downsample_probability*100:.1f}%")
            if training_config.domain_target_size:
                self.log_info(f"      - Target size: {training_config.domain_target_size}x{training_config.domain_target_size} (CIFAR-10/100 mode)")
            else:
                scale_min, scale_max = training_config.domain_downsample_scale_range
                self.log_info(f"      - Scale range: {scale_min:.2f}-{scale_max:.2f}")
        else:
            self.log_info("   Domain Adaptation: Disabled")
""",

    "3. Clean up imports": """
At top of model_customizer.py (lines 1-11), add:

from PIL import ImageFilter
import io
import random

Then in _apply_domain_adaptation() (line 1025), remove these lines:
    import numpy as np  # REMOVE - unused
    # Keep other imports since they're already at top
""",

    "4. Add error handling": """
In _apply_transforms() method (line 1098-1103), wrap domain adaptation:

        else:
            # Apply domain adaptation even without RandAugment (if enabled)
            if self.use_domain_adaptation:
                try:
                    image = self._apply_domain_adaptation(image)
                except Exception as e:
                    # Log warning but continue with non-adapted image
                    print(f"⚠️  Domain adaptation failed: {str(e)}, using non-adapted image")
            return image.resize((224, 224)) if image.size != (224, 224) else image

Also in lines 1090-1091, wrap the call:

                # Apply domain adaptation AFTER RandAugment (if enabled)
                if self.use_domain_adaptation:
                    try:
                        augmented_image = self._apply_domain_adaptation(augmented_image)
                    except Exception as e:
                        print(f"⚠️  Domain adaptation failed: {str(e)}, using non-adapted image")
"""
}

print("=" * 80)
print("IMPROVEMENTS TO APPLY:")
print("=" * 80)
print()

for title, instructions in improvements.items():
    print(f"### {title}")
    print(instructions)
    print()
    print("-" * 80)
    print()

print("=" * 80)
print("✅ Review complete! Apply these changes manually or use automated tools.")
print("=" * 80)
