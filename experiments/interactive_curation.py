#!/usr/bin/env python3
"""
MAVERIC Interactive Data Curation - Compact Notebook Script

This compact script can be executed directly in a Google Colab or Jupyter notebook cell
to create an interactive GUI for data curation with sliders, plots, and controls.

Usage in notebook cell:

# Method 1: Copy and paste this entire script into a cell and run
# (Modify the variables below as needed)

# Method 2: Execute from file
# exec(open('interactive_curation.py').read())

# Method 3: With custom parameters
# DATASET = 'cifar10'
# CONFIG = 'maveric_config.yaml'
# exec(open('interactive_curation.py').read())
"""

import sys
import os
from pathlib import Path

# Configuration - Modify these variables as needed
if 'DATASET' not in globals():
    DATASET = 'cifar10'  # Change to your dataset name

if 'CONFIG' not in globals():
    CONFIG = 'maveric_config.yaml'  # Change to your config file path

# Add MAVERIC to path if needed
maveric_path = Path(__file__).parent.parent
if str(maveric_path) not in sys.path:
    sys.path.insert(0, str(maveric_path))

try:
    # Import MAVERIC interactive GUI
    from maveric.visualization import create_interactive_gui, INTERACTIVE_AVAILABLE
    
    if not INTERACTIVE_AVAILABLE:
        print("❌ Interactive GUI not available!")
        print("📦 Please install required packages:")
        print("   pip install ipywidgets")
        print("   jupyter nbextension enable --py widgetsnbextension")
        sys.exit(1)
    
    # Create and display the interactive GUI
    print("🚀 Starting MAVERIC Interactive Data Curation")
    print("=" * 50)
    
    # Create the GUI - this will display automatically
    gui = create_interactive_gui(DATASET, CONFIG)
    
    print("\n" + "=" * 50)
    print("✅ Interactive GUI loaded successfully!")
    print("📱 Widgets should be visible above")
    print("\n🎛️ How to use:")
    print("1. Adjust thresholds in the 'Quality Thresholds' tab")
    print("2. Adjust weights in the 'Metric Weights' tab")  
    print("3. Click 'Apply Settings' to update plots and filtering")
    print("4. Click 'Show Samples' to view sample images")
    print("5. Click 'Save Config' to save your settings")
    print("\n💡 The plots will update automatically when you apply settings!")
    
except ImportError as e:
    print("❌ Error importing MAVERIC:")
    print(f"   {e}")
    print("\n🔧 Please ensure MAVERIC is properly installed and accessible")
    print("   You may need to adjust the path or install MAVERIC")
    
except Exception as e:
    print(f"❌ Error creating GUI: {e}")
    import traceback
    print("\n📝 Full error details:")
    traceback.print_exc()
    
    print("\n🔧 Troubleshooting:")
    print("1. Check that your dataset raw data exists")
    print("2. Verify the configuration file path")
    print("3. Ensure all required packages are installed")
    print("4. Make sure you're running in a Jupyter/Colab environment")

# Store GUI reference for later use
if 'gui' in locals():
    MAVERIC_GUI = gui