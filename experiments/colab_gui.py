"""
MAVERIC Interactive GUI for Google Colab

Simple script to run MAVERIC interactive data curation GUI in Google Colab.
Copy this entire cell content and paste into a Colab cell to run.
"""

import sys
import os
from pathlib import Path

# =============================================================================
# CONFIGURATION - MODIFY THESE PATHS FOR YOUR SETUP
# =============================================================================

# Path to MAVERIC directory (modify this for your setup)
MAVERIC_PATH = '/content/drive/MyDrive/MAVERIC/repo/maveric'

# Dataset and config (modify these for your needs)
DATASET_NAME = 'cifar10'
CONFIG_FILE = '/content/drive/MyDrive/MAVERIC/repo/maveric/experiments/maveric_config.yaml'

# =============================================================================
# SETUP AND EXECUTION
# =============================================================================

print("🚀 MAVERIC Interactive Data Curation for Google Colab")
print("=" * 60)

# Add MAVERIC to Python path
if MAVERIC_PATH not in sys.path:
    sys.path.insert(0, MAVERIC_PATH)
    print(f"✅ Added MAVERIC to path: {MAVERIC_PATH}")

# Verify MAVERIC directory exists
if not Path(MAVERIC_PATH).exists():
    print(f"❌ MAVERIC directory not found: {MAVERIC_PATH}")
    print("💡 Please update MAVERIC_PATH to point to your MAVERIC directory")
    sys.exit(1)

# Verify config file exists  
if not Path(CONFIG_FILE).exists():
    print(f"❌ Config file not found: {CONFIG_FILE}")
    print("💡 Please update CONFIG_FILE to point to your configuration file")
    sys.exit(1)

try:
    # Import MAVERIC components
    print("📦 Importing MAVERIC components...")
    from maveric.visualization import create_interactive_gui, INTERACTIVE_AVAILABLE
    
    if not INTERACTIVE_AVAILABLE:
        print("❌ Interactive widgets not available!")
        print("📋 Installing required packages...")
        print("Run this in a cell first:")
        print("!pip install ipywidgets")
        print("!jupyter nbextension enable --py widgetsnbextension")
        sys.exit(1)
    
    print(f"🎯 Dataset: {DATASET_NAME}")
    print(f"⚙️ Config: {CONFIG_FILE}")
    print("-" * 40)
    
    # Create the interactive GUI
    print("🎨 Creating interactive GUI...")
    gui = create_interactive_gui(DATASET_NAME, CONFIG_FILE)
    
    print("\n" + "=" * 60)
    print("✅ SUCCESS! Interactive GUI is now running above.")
    print("\n📖 How to use:")
    print("1. 📊 Use the 'Quality Thresholds' tab to adjust filtering thresholds")
    print("2. ⚖️ Use the 'Metric Weights' tab to adjust metric importance")
    print("3. 🔄 Click 'Apply Settings' to update plots and see filtering results")
    print("4. 🖼️ Click 'Show Samples' to view sample images from filtered data")
    print("5. 💾 Click 'Save Config' to save your threshold settings")
    print("\n💡 Tips:")
    print("- Adjust thresholds to balance quality vs quantity")
    print("- Watch the filtered count update as you change thresholds")
    print("- Use sample images to verify quality of filtered data")
    print("- Save your settings before running the actual data curation")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\n🔧 Troubleshooting:")
    print("1. Verify MAVERIC_PATH points to correct directory")
    print("2. Check that MAVERIC is properly installed")
    print("3. Ensure all dependencies are installed")
    
except Exception as e:
    print(f"❌ Error creating GUI: {e}")
    import traceback
    print("\n📝 Full error:")
    traceback.print_exc()
    print("\n🔧 Common fixes:")
    print("1. Check dataset raw data exists at expected location")
    print("2. Verify configuration file path and contents")
    print("3. Ensure you're running in a Jupyter/Colab environment")

print("\n" + "=" * 60)