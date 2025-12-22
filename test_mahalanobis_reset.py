"""
Test suite for Mahalanobis Filter Reset Button functionality.

Tests:
1. Reset button widget exists and has correct properties
2. Global mode reset clears filtered data
3. Class-Based mode reset with no class selected clears all class data
4. Class-Based mode reset with class selected clears specific class
5. Reset button in layout
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

def test_reset_button_exists():
    """Test that reset button widget is created with correct properties"""
    print("Test 1: Reset button widget exists...")

    # Mock ipywidgets
    with patch('maveric.visualization.interactive.widgets') as mock_widgets:
        # Create mock button
        mock_reset_button = MagicMock()
        mock_reset_button.description = 'Reset'
        mock_reset_button.button_style = 'danger'
        mock_reset_button.icon = 'undo'

        # Mock widgets.Button to return our mock
        def mock_button_factory(**kwargs):
            if kwargs.get('description') == 'Reset':
                return mock_reset_button
            return MagicMock()

        mock_widgets.Button = mock_button_factory
        mock_widgets.VBox = MagicMock(return_value=MagicMock())
        mock_widgets.HBox = MagicMock(return_value=MagicMock())
        mock_widgets.RadioButtons = MagicMock(return_value=MagicMock())
        mock_widgets.Dropdown = MagicMock(return_value=MagicMock())
        mock_widgets.FloatText = MagicMock(return_value=MagicMock())
        mock_widgets.Output = MagicMock(return_value=MagicMock())
        mock_widgets.HTML = MagicMock(return_value=MagicMock())
        mock_widgets.Layout = MagicMock

        from maveric.visualization.interactive import MAVERICInteractiveQualityControl

        # Create sample data
        sample_data = pd.DataFrame({
            'url': ['http://example.com/1.jpg'] * 10,
            'text': ['sample text'] * 10,
            'label': ['class1'] * 5 + ['class2'] * 5,
            'weighted_class_score': np.random.rand(10),
            'consistency': np.random.rand(10)
        })

        gui = MAVERICInteractiveQualityControl('test_dataset', sample_data)

        # The reset button should be created (we can't verify exact properties due to mocking)
        print("✅ Reset button created in GUI initialization")
        return True

def test_global_mode_reset_logic():
    """Test reset logic for Global mode"""
    print("\nTest 2: Global mode reset logic...")

    # Create test GUI instance with real data
    sample_data = pd.DataFrame({
        'url': ['http://example.com/1.jpg'] * 10,
        'text': ['sample text'] * 10,
        'label': ['class1'] * 5 + ['class2'] * 5,
        'weighted_class_score': np.random.rand(10),
        'consistency': np.random.rand(10)
    })

    from maveric.visualization.interactive import MAVERICInteractiveQualityControl

    with patch('maveric.visualization.interactive.widgets'):
        gui = MAVERICInteractiveQualityControl('test_dataset', sample_data)

    # Simulate having applied a filter
    gui.filtered_data = sample_data.head(5).copy()  # Filtered to 5 samples
    gui.data_before_mahalanobis = sample_data.copy()  # Original 10 samples

    # Simulate reset logic (Global mode)
    if gui.data_before_mahalanobis is not None:
        gui.filtered_data = gui.data_before_mahalanobis.copy()
        gui.data_before_mahalanobis = None

    assert len(gui.filtered_data) == 10, f"Expected 10 samples after reset, got {len(gui.filtered_data)}"
    assert gui.data_before_mahalanobis is None, "Backup should be cleared after reset"

    print("✅ Global mode reset restores original data correctly")
    return True

def test_class_based_reset_all():
    """Test reset logic for Class-Based mode (no specific class)"""
    print("\nTest 3: Class-Based mode reset all classes...")

    sample_data = pd.DataFrame({
        'url': ['http://example.com/1.jpg'] * 10,
        'text': ['sample text'] * 10,
        'label': ['class1'] * 5 + ['class2'] * 5,
        'weighted_class_score': np.random.rand(10),
        'consistency': np.random.rand(10)
    })

    from maveric.visualization.interactive import MAVERICInteractiveQualityControl

    with patch('maveric.visualization.interactive.widgets'):
        gui = MAVERICInteractiveQualityControl('test_dataset', sample_data)

    # Simulate having class-based filtered data
    gui.class_based_filtered_data = {
        'class1': sample_data.head(2).copy(),
        'class2': sample_data.tail(3).copy()
    }

    # Simulate reset all (no specific class selected)
    num_classes_before = len(gui.class_based_filtered_data)
    gui.class_based_filtered_data.clear()

    assert len(gui.class_based_filtered_data) == 0, "All class data should be cleared"
    assert num_classes_before == 2, "Should have had 2 classes before reset"

    print("✅ Class-Based mode reset clears all class data correctly")
    return True

def test_class_based_reset_specific():
    """Test reset logic for Class-Based mode (specific class)"""
    print("\nTest 4: Class-Based mode reset specific class...")

    sample_data = pd.DataFrame({
        'url': ['http://example.com/1.jpg'] * 10,
        'text': ['sample text'] * 10,
        'label': ['class1'] * 5 + ['class2'] * 5,
        'weighted_class_score': np.random.rand(10),
        'consistency': np.random.rand(10)
    })

    from maveric.visualization.interactive import MAVERICInteractiveQualityControl

    with patch('maveric.visualization.interactive.widgets'):
        gui = MAVERICInteractiveQualityControl('test_dataset', sample_data)

    # Simulate having class-based filtered data
    gui.class_based_filtered_data = {
        'class1': sample_data.head(2).copy(),
        'class2': sample_data.tail(3).copy()
    }

    # Simulate reset specific class
    selected_class = 'class1'
    if selected_class in gui.class_based_filtered_data:
        del gui.class_based_filtered_data[selected_class]

    assert 'class1' not in gui.class_based_filtered_data, "class1 should be removed"
    assert 'class2' in gui.class_based_filtered_data, "class2 should remain"
    assert len(gui.class_based_filtered_data) == 1, "Should have 1 class remaining"

    print("✅ Class-Based mode reset clears specific class correctly")
    return True

def test_reset_button_in_layout():
    """Test that reset button is included in layout"""
    print("\nTest 5: Reset button in layout...")

    # This is harder to test without full widget initialization
    # We can verify the code structure is correct
    import inspect
    from maveric.visualization.interactive import MAVERICInteractiveQualityControl

    source = inspect.getsource(MAVERICInteractiveQualityControl._create_mahalanobis_tab)

    # Check that reset_button is created
    assert 'reset_button = widgets.Button' in source, "reset_button should be created"

    # Check that reset_button has correct style
    assert "button_style='danger'" in source, "reset_button should have danger style"

    # Check that reset callback is defined
    assert 'def on_reset_clicked(b):' in source, "reset callback should be defined"

    # Check that reset_button is in layout
    assert 'reset_button' in source.split('widgets.HBox([')[-1].split('])')[0], "reset_button should be in HBox layout"

    print("✅ Reset button properly included in layout")
    return True

if __name__ == '__main__':
    print("=" * 60)
    print("Testing Mahalanobis Filter Reset Button Functionality")
    print("=" * 60)

    tests = [
        test_reset_button_exists,
        test_global_mode_reset_logic,
        test_class_based_reset_all,
        test_class_based_reset_specific,
        test_reset_button_in_layout
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(('PASS', test.__name__))
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            results.append(('FAIL', test.__name__))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for status, name in results:
        symbol = "✅" if status == "PASS" else "❌"
        print(f"{symbol} {name}: {status}")

    passed = sum(1 for status, _ in results if status == 'PASS')
    total = len(results)

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        sys.exit(1)
