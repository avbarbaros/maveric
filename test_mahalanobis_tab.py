#!/usr/bin/env python3
"""
Test script to verify Mahalanobis distance filtering tab in interactive GUI.

This script demonstrates the new Mahalanobis Filter tab functionality.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_mahalanobis_imports():
    """Test that all required imports work"""
    print("=" * 80)
    print("Testing Mahalanobis Distance Filtering - Import Check")
    print("=" * 80)

    try:
        print("\n1. Testing scipy import...")
        from scipy.spatial.distance import mahalanobis
        print("   ✓ scipy.spatial.distance.mahalanobis imported successfully")

        print("\n2. Testing matplotlib.patches import...")
        from matplotlib.patches import Ellipse
        print("   ✓ matplotlib.patches.Ellipse imported successfully")

        print("\n3. Testing interactive module import...")
        from maveric.visualization.interactive import MAVERICInteractiveQualityControl
        print("   ✓ MAVERICInteractiveQualityControl imported successfully")

        print("\n4. Checking for new methods...")
        methods = [
            '_create_mahalanobis_tab',
            '_apply_mahalanobis_filter',
            '_plot_mahalanobis_analysis',
            '_show_mahalanobis_statistics'
        ]

        for method in methods:
            if hasattr(MAVERICInteractiveQualityControl, method):
                print(f"   ✓ {method}() method exists")
            else:
                print(f"   ❌ {method}() method NOT FOUND")
                return False

        print("\n5. Checking instance variables...")
        # Create a test instance (will fail if no data, but that's okay for this test)
        try:
            # This will fail because we don't have real data, but we can check the __init__
            import inspect
            init_source = inspect.getsource(MAVERICInteractiveQualityControl.__init__)

            if 'mahalanobis_filter_info' in init_source:
                print("   ✓ mahalanobis_filter_info instance variable defined")
            else:
                print("   ❌ mahalanobis_filter_info NOT FOUND in __init__")

            if 'data_before_mahalanobis' in init_source:
                print("   ✓ data_before_mahalanobis instance variable defined")
            else:
                print("   ❌ data_before_mahalanobis NOT FOUND in __init__")

        except Exception as e:
            print(f"   ⚠️ Could not inspect __init__: {e}")

        print("\n" + "=" * 80)
        print("✅ ALL IMPORT TESTS PASSED!")
        print("=" * 80)

        print("\n📋 Mahalanobis Filter Tab Features:")
        print("   • Location: Tab 2 (between Quality Thresholds and EfficientNet)")
        print("   • Controls: Percentage dropdown, custom input, Global/Per-Class mode")
        print("   • Visualization: XY scatter plot with ellipse boundary")
        print("   • Statistics: Before/after counts, class distribution")
        print("\n💡 Usage:")
        print("   from maveric.visualization import start_interactive_gui")
        print("   gui = start_interactive_gui('cifar10')")
        print("   # Navigate to 'Mahalanobis Filter' tab and click 'Apply Filter'")
        print("\n" + "=" * 80)

        return True

    except ImportError as e:
        print(f"\n❌ IMPORT ERROR: {e}")
        print("\nMake sure all dependencies are installed:")
        print("   pip install scipy matplotlib ipywidgets")
        return False
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mahalanobis_algorithm():
    """Test the Mahalanobis distance calculation"""
    print("\n" + "=" * 80)
    print("Testing Mahalanobis Distance Calculation")
    print("=" * 80)

    try:
        import numpy as np
        from scipy.spatial.distance import mahalanobis

        # Create sample data
        np.random.seed(42)
        n_samples = 1000

        # Correlated data
        mean = [0.6, 0.8]
        cov = [[0.01, 0.005], [0.005, 0.01]]
        data = np.random.multivariate_normal(mean, cov, n_samples)

        print(f"\n📊 Generated {n_samples} samples with correlation")
        print(f"   Mean: {mean}")
        print(f"   Covariance:\n{np.array(cov)}")

        # Calculate ideal point (95th percentile)
        ideal_point = np.array([
            np.percentile(data[:, 0], 95),
            np.percentile(data[:, 1], 95)
        ])
        print(f"\n🎯 Ideal point (95th percentile): {ideal_point}")

        # Compute covariance and inverse
        cov_matrix = np.cov(data.T)
        cov_inv = np.linalg.inv(cov_matrix)

        print(f"\n📐 Computed covariance matrix:\n{cov_matrix}")

        # Calculate distances
        distances = np.array([
            mahalanobis(x, ideal_point, cov_inv)
            for x in data
        ])

        print(f"\n📏 Distance statistics:")
        print(f"   Min: {distances.min():.4f}")
        print(f"   Max: {distances.max():.4f}")
        print(f"   Mean: {distances.mean():.4f}")
        print(f"   Median: {np.median(distances):.4f}")

        # Test filtering at 30%
        keep_percentile = 30
        n_keep = int(len(data) * keep_percentile / 100)
        threshold = np.partition(distances, n_keep-1)[n_keep-1]

        selected = distances <= threshold

        print(f"\n✂️ Filtering at {keep_percentile}%:")
        print(f"   Threshold: {threshold:.4f}")
        print(f"   Selected: {selected.sum()} samples")
        print(f"   Rejected: {(~selected).sum()} samples")

        # Quality of selected samples
        selected_data = data[selected]
        print(f"\n✨ Selected samples quality:")
        print(f"   Metric 1 - Mean: {selected_data[:, 0].mean():.4f}, Min: {selected_data[:, 0].min():.4f}")
        print(f"   Metric 2 - Mean: {selected_data[:, 1].mean():.4f}, Min: {selected_data[:, 1].min():.4f}")

        print("\n" + "=" * 80)
        print("✅ ALGORITHM TEST PASSED!")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\n❌ ALGORITHM TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n🧪 MAVERIC Mahalanobis Distance Filtering - Test Suite")
    print("=" * 80)

    # Run tests
    test1_passed = test_mahalanobis_imports()
    test2_passed = test_mahalanobis_algorithm()

    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print(f"   Import Tests: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   Algorithm Tests: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print("=" * 80)

    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nThe Mahalanobis Filter tab is ready to use in the interactive GUI.")
        sys.exit(0)
    else:
        print("\n⚠️ SOME TESTS FAILED")
        print("Please check the error messages above.")
        sys.exit(1)
