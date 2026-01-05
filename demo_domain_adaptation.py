#!/usr/bin/env python3
"""
Demo script to visualize domain adaptation effects on images.
Creates a side-by-side comparison of original vs. adapted images.
"""

import os
import sys
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import io
import random

def create_test_image(size=(224, 224)):
    """Create a colorful test image with patterns"""
    img = Image.new('RGB', size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Add colorful rectangles
    colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255),
              (255, 255, 100), (255, 100, 255), (100, 255, 255)]

    rect_size = size[0] // 6
    for i, color in enumerate(colors):
        x = (i % 3) * rect_size * 2
        y = (i // 3) * rect_size * 2
        draw.rectangle([x, y, x + rect_size, y + rect_size], fill=color)

    # Add some text
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()

    draw.text((10, size[1] - 30), "Original", fill=(0, 0, 0), font=font)

    return img


def apply_gaussian_blur(image, sigma=1.5):
    """Apply Gaussian blur"""
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def apply_jpeg_compression(image, quality=50):
    """Apply JPEG compression artifacts"""
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert('RGB')


def apply_downsample(image, target_size=32):
    """Apply resolution degradation"""
    original_size = image.size
    downsampled = image.resize((target_size, target_size), Image.BILINEAR)
    return downsampled.resize(original_size, Image.BILINEAR)


def add_label(image, text):
    """Add label to image"""
    img_with_label = image.copy()
    draw = ImageDraw.Draw(img_with_label)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()

    # Add white background for text
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    padding = 5
    draw.rectangle(
        [5, img_with_label.height - text_height - padding * 2 - 5,
         text_width + padding * 2 + 5, img_with_label.height - 5],
        fill=(255, 255, 255, 200)
    )

    draw.text((10, img_with_label.height - text_height - padding - 5),
              text, fill=(0, 0, 0), font=font)

    return img_with_label


def create_comparison_grid():
    """Create a grid showing original vs. adapted images"""
    print("Creating domain adaptation demonstration...")

    # Create original test image
    print("  1. Creating test image...")
    original = create_test_image((224, 224))

    # Apply domain adaptations
    print("  2. Applying Gaussian blur (sigma=1.5)...")
    blurred = apply_gaussian_blur(original, sigma=1.5)

    print("  3. Applying JPEG compression (quality=40)...")
    compressed = apply_jpeg_compression(original, quality=40)

    print("  4. Applying resolution degradation (32x32)...")
    downsampled = apply_downsample(original, target_size=32)

    print("  5. Applying all transforms together...")
    all_transforms = apply_gaussian_blur(original, sigma=1.0)
    all_transforms = apply_jpeg_compression(all_transforms, quality=50)
    all_transforms = apply_downsample(all_transforms, target_size=32)

    # Add labels
    print("  6. Adding labels...")
    original_labeled = add_label(original, "Original")
    blurred_labeled = add_label(blurred, "Gaussian Blur")
    compressed_labeled = add_label(compressed, "JPEG Compression")
    downsampled_labeled = add_label(downsampled, "Resolution Degradation")
    all_labeled = add_label(all_transforms, "All Transforms")

    # Create grid (2 rows x 3 columns)
    print("  7. Creating grid layout...")
    grid_width = 224 * 3 + 20  # 3 images + margins
    grid_height = 224 * 2 + 20  # 2 rows + margins

    grid = Image.new('RGB', (grid_width, grid_height), color=(240, 240, 240))

    # Paste images
    # Row 1
    grid.paste(original_labeled, (10, 10))
    grid.paste(blurred_labeled, (244, 10))
    grid.paste(compressed_labeled, (478, 10))

    # Row 2
    grid.paste(downsampled_labeled, (10, 244))
    grid.paste(all_labeled, (244, 244))

    # Add title
    draw = ImageDraw.Draw(grid)
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        title_font = ImageFont.load_default()

    title = "Domain Adaptation Effects"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (grid_width - title_width) // 2

    draw.rectangle([title_x - 10, 250, title_x + title_width + 10, 285],
                   fill=(255, 255, 255))
    draw.text((title_x, 252), title, fill=(0, 0, 0), font=title_font)

    # Save
    output_path = "domain_adaptation_demo.png"
    grid.save(output_path, quality=95)
    print(f"\n✅ Demo image saved to: {output_path}")
    print(f"   Size: {grid.size}")
    print(f"   Mode: {grid.mode}")

    return output_path


def print_statistics():
    """Print statistics about the transforms"""
    print("\n" + "=" * 60)
    print("DOMAIN ADAPTATION EFFECTS")
    print("=" * 60)

    print("\n1. Gaussian Blur (sigma=1.5)")
    print("   Purpose: Simulates low quality cameras, motion blur")
    print("   Effect: Reduces sharp edges, smooths textures")

    print("\n2. JPEG Compression (quality=40)")
    print("   Purpose: Simulates web images, compressed uploads")
    print("   Effect: Adds blocking artifacts, color banding")

    print("\n3. Resolution Degradation (32x32)")
    print("   Purpose: Simulates CIFAR-10/100 native resolution")
    print("   Effect: Pixelation, loss of fine details")

    print("\n4. All Transforms Combined")
    print("   Purpose: Realistic worst-case scenario")
    print("   Effect: Multiple degradations compounded")

    print("\n" + "=" * 60)
    print("USAGE IN CONFIG")
    print("=" * 60)
    print("""
training:
  use_domain_adaptation: true

  domain_blur_probability: 0.3       # 30% of images
  domain_blur_sigma_range: [0.1, 2.0]

  domain_jpeg_probability: 0.3       # 30% of images
  domain_jpeg_quality_range: [30, 95]

  domain_downsample_probability: 0.3  # 30% of images
  domain_target_size: 32             # CIFAR-10/100
""")
    print("=" * 60 + "\n")


def main():
    """Main demo function"""
    print("\n" + "=" * 60)
    print("DOMAIN ADAPTATION VISUALIZATION DEMO")
    print("=" * 60 + "\n")

    try:
        # Create comparison grid
        output_path = create_comparison_grid()

        # Print statistics
        print_statistics()

        print("✅ Demo completed successfully!")
        print(f"\nView the output: {os.path.abspath(output_path)}")

        return 0

    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
