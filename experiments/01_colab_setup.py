#!/usr/bin/env python3
"""
Google Colab Setup Script for MAVERIC
This script sets up the MAVERIC environment on Google Colab with T4 GPU support.
Follows the official installation guide from README.md
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description=""):
    """Run a shell command with error handling."""
    print(f"🔄 {description}")
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout.strip():
            print("Output:", result.stdout.strip())
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return None

def check_gpu():
    """Check if GPU is available and get GPU info."""
    print("🔍 Checking GPU availability...")
    gpu_info = run_command("nvidia-smi", "GPU check")
    if gpu_info:
        print("GPU Information:")
        print(gpu_info)
        return True
    else:
        print("⚠️  GPU not detected. Make sure you're using a GPU runtime.")
        return False

def install_system_dependencies():
    """Install system-level dependencies required for MAVERIC."""
    print("📦 Installing system dependencies...")
    
    # Update package list
    run_command("apt-get update -qq", "Updating package list")
    
    # Install system packages mentioned in README for headless environments
    packages = [
        "libgl1-mesa-glx",  # For OpenCV
        "libglib2.0-0",     # For OpenCV
        "libsm6",           # For OpenCV 
        "libxext6",         # For OpenCV
        "libxrender-dev",   # For OpenCV
        "libgomp1"          # For PyTorch
    ]
    
    for package in packages:
        run_command(f"apt-get install -y {package}", f"Installing {package}")

def setup_environment():
    """Set up environment variables for headless Colab environment."""
    print("🌍 Setting up environment variables...")
    
    # Set matplotlib backend for headless environment (as per README.md)
    os.environ['MPLBACKEND'] = 'Agg'
    
    # Set cache directories on Google Drive
    os.environ['MAVERIC_CACHE_DIR'] = '/content/drive/MyDrive/MAVERIC/maveric_cache'
    os.environ['HF_HOME'] = '/content/drive/MyDrive/MAVERIC/huggingface_cache'
    
    print("Environment variables set:")
    print(f"MPLBACKEND: {os.environ.get('MPLBACKEND')}")
    print(f"MAVERIC_CACHE_DIR: {os.environ.get('MAVERIC_CACHE_DIR')}")
    print(f"HF_HOME: {os.environ.get('HF_HOME')}")

def clone_maveric_repo():
    """Clone MAVERIC repository from GitHub."""
    print("📥 Cloning MAVERIC repository...")
    
    repo_url = "https://github.com/avbarbaros/maveric.git"
    
    # Remove existing directory if it exists
    if os.path.exists("/content/drive/MyDrive/MAVERIC/repo/maveric"):
        run_command("rm -rf /content/drive/MyDrive/MAVERIC/repo/maveric", "Removing existing maveric directory")
    
    # Clone the repository
    result = run_command(f"git clone {repo_url} /content/drive/MyDrive/MAVERIC/repo/maveric", "Cloning MAVERIC repository")

    if result is not None:
        # Change to maveric directory
        os.chdir("/content/drive/MyDrive/MAVERIC/repo/maveric")
        print("✅ Successfully cloned and changed to MAVERIC directory")
        return True
    else:
        print("❌ Failed to clone repository")
        return False

def install_maveric():
    """Install MAVERIC following the official README.md instructions."""
    print("🐍 Installing MAVERIC using official method...")
    
    # Make sure we're in the maveric directory
    if not os.path.exists("/content/drive/MyDrive/MAVERIC/repo/maveric/requirements.txt"):
        print("❌ requirements.txt not found. Please run clone_maveric_repo() first.")
        return False

    os.chdir("/content/drive/MyDrive/MAVERIC/repo/maveric")

    # Follow README.md installation instructions exactly:
    # 1. Install dependencies first
    result1 = run_command("pip install -r requirements.txt", "Installing dependencies from requirements.txt")
    
    # 2. Install in development mode
    result2 = run_command('pip install -e ".[dev]"', "Installing MAVERIC in development mode")
    
    return result1 is not None and result2 is not None

def test_installation():
    """Test if MAVERIC is properly installed."""
    print("🧪 Testing MAVERIC installation...")
    
    # Ensure we're in the correct directory for imports
    maveric_dir = "/content/drive/MyDrive/MAVERIC/repo/maveric"
    if os.path.exists(maveric_dir):
        os.chdir(maveric_dir)
        # Add maveric directory to Python path
        sys.path.insert(0, maveric_dir)
        print(f"✅ Changed to MAVERIC directory: {maveric_dir}")
    
    try:
        # Test PyTorch and CUDA
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU device: {torch.cuda.get_device_name(0)}")
        
        # Test MAVERIC import
        import maveric
        print("✅ MAVERIC imported successfully")
        
        # Test CLIP import
        import clip
        print("✅ CLIP imported successfully")
        
        # Test key components
        from maveric import MAVERIC, MAVERICConfig
        print("✅ MAVERIC core classes imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Tip: Make sure you're running this from the correct directory")
        print("💡 Try: cd /content/drive/MyDrive/MAVERIC/repo/maveric && python ../../01_colab_setup.py")
        return False

def main():
    """Main setup function following README.md guidelines."""
    print("🚀 Starting MAVERIC setup for Google Colab...")
    print("Following official installation guide from README.md")
    print("=" * 60)
    
    # Check GPU
    gpu_available = check_gpu()
    
    # Install system dependencies
    install_system_dependencies()
    
    # Set up environment (headless setup as per README.md)
    setup_environment()
    
    # Clone repository
    # if not clone_maveric_repo():
    #     print("❌ Failed to clone repository. Exiting.")
    #     return False
    
    # Install MAVERIC using official method
    if not install_maveric():
        print("❌ Failed to install MAVERIC. Exiting.")
        return False
    
    # Test installation
    if test_installation():
        print("\n" + "=" * 60)
        print("🎉 MAVERIC setup completed successfully!")
        print("📋 Installation Summary:")
        print("✅ System dependencies installed")
        print("✅ Environment configured for headless operation")
        print("✅ Dependencies installed from requirements.txt")
        print("✅ MAVERIC installed in development mode")
        print("✅ All imports working correctly")
        print("\n📝 Next steps:")
        print("1. Mount Google Drive: run google_drive_setup.py")
        print("2. Run ELEVATER experiments: run elevater_experiments.py")
        print("=" * 60)
        return True
    else:
        print("\n" + "=" * 60)
        print("❌ Setup completed with errors. Please check the output above.")
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)