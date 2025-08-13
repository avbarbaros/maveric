#!/usr/bin/env python3
"""
MAVERIC Setup Script for Google Colab
Comprehensive setup combining Colab environment setup and Google Drive integration.
"""

import subprocess
import sys
import os
import shutil
import yaml
import argparse
from pathlib import Path

def run_command(cmd, description=""):
    """Run a shell command with error handling."""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}: {e}")
        return None

def check_gpu():
    """Check GPU availability."""
    print("🔍 Checking GPU...")
    gpu_info = run_command("nvidia-smi", "GPU check")
    if gpu_info:
        print("✅ GPU detected")
        return True
    print("⚠️  GPU not detected")
    return False

def install_system_dependencies():
    """Install system dependencies from system-requirements.txt."""
    print("📦 Installing system dependencies...")
    run_command("apt-get update -qq", "Updating packages")
    
    # Read system requirements from file
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    system_req_path = os.path.join(repo_root, "system-requirements.txt")
    
    if os.path.exists(system_req_path):
        with open(system_req_path, 'r') as f:
            packages = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        
        for package in packages:
            run_command(f"apt-get install -y {package}", f"Installing {package}")
    else:
        print("⚠️  system-requirements.txt not found, skipping system dependencies")

def mount_google_drive():
    """Mount Google Drive."""
    print("📂 Checking Google Drive...")
    
    # Check if Google Drive is already mounted
    if os.path.exists('/content/drive/MyDrive'):
        print("✅ Google Drive already mounted")
        return True
    
    try:
        from google.colab import drive
        print("📂 Mounting Google Drive...")
        drive.mount('/content/drive')
        print("✅ Google Drive mounted")
        return True
    except ImportError:
        print("⚠️  Not running in Google Colab - skipping Drive mount")
        print("ℹ️  Make sure your paths point to accessible directories")
        return True  # Return True to continue execution
    except Exception as e:
        print(f"❌ Error mounting Drive: {e}")
        return False

def load_config(config_path):
    """Load MAVERIC configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Configuration loaded from: {config_path}")
        return config
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"❌ Error parsing YAML file: {e}")
        return None
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return None

def setup_environment(config, config_path):
    """Configure environment variables from config."""
    print("🌍 Setting up environment...")
    
    maveric_base_dir = config['maveric_base_dir']
    
    env_vars = {
        'MPLBACKEND': 'Agg',
        'MAVERIC_BASE_DIR': maveric_base_dir,
        'MAVERIC_CACHE_DIR': config['cache_base_dir'],
        'MAVERIC_RESULTS_DIR': config['results_dir'],
        'HF_HOME': f'{maveric_base_dir}/huggingface_cache',
        'MAVERIC_CONFIG_PATH': config_path
    }
    
    for var, value in env_vars.items():
        os.environ[var] = value
        print(f"✅ Set {var}")

def create_directories(config):
    """Create necessary directories based on config."""
    print("📁 Creating directories...")
    
    maveric_base_dir = config['maveric_base_dir']
    base_cache_dir = config['cache_base_dir']
    results_dir = config['results_dir']
    
    directories = [
        base_cache_dir,
        f"{base_cache_dir}/images",
        f"{base_cache_dir}/embeddings",
        f"{base_cache_dir}/models",
        f"{base_cache_dir}/datasets",
        results_dir,
        f"{results_dir}/experiments",
        f"{results_dir}/visualizations",
        f"{results_dir}/quality_reports",
        f"{maveric_base_dir}/huggingface_cache"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {directory}")

def install_maveric(config):
    """Install MAVERIC package."""
    print("🐍 Installing MAVERIC package and all dependencies...")
    
    # Clone if needed
    maveric_base_dir = config['maveric_base_dir']
    repo_path = f"{maveric_base_dir}/repo/maveric"
    if not os.path.exists(repo_path):
        repo_url = "https://github.com/avbarbaros/maveric.git"
        result = run_command(f"git clone {repo_url} {repo_path}", "Cloning MAVERIC")
        if not result:
            return False
    
    # Install
    os.chdir(repo_path)
    result1 = run_command("pip install -r requirements.txt", "Installing dependencies")
    result2 = run_command('pip install -e ".[dev]"', "Installing MAVERIC")
    
    return result1 is not None and result2 is not None

def test_installation():
    """Test MAVERIC installation."""
    print("🧪 Testing installation...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        print(f"✅ CUDA: {torch.cuda.is_available()}")
        
        # Force reload of modules in case they were cached
        import importlib
        import sys
        
        # Remove any cached modules
        modules_to_remove = [name for name in sys.modules.keys() if name.startswith('maveric')]
        for module in modules_to_remove:
            del sys.modules[module]
        
        # Try importing maveric
        import maveric
        from maveric.main import MAVERIC  
        from maveric.config import MAVERICConfig
        print("✅ MAVERIC imported")
        
        import clip
        print("✅ CLIP imported")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("ℹ️  This might be normal if running outside the repository")
        print("ℹ️  Try importing maveric in a new Python session")
        return False


def test_cache_access(config):
    """Test read/write access to cache directories."""
    print("🧪 Testing cache access...")
    
    cache_dir = config['cache_base_dir']
    test_file = f"{cache_dir}/test_access.txt"
    
    try:
        with open(test_file, 'w') as f:
            f.write("MAVERIC cache test")
        with open(test_file, 'r') as f:
            content = f.read()
        os.remove(test_file)
        print("✅ Cache access verified")
        return True
    except Exception as e:
        print(f"❌ Cache access test failed: {e}")
        return False

def show_summary(config):
    """Display setup summary with config details."""
    try:
        total, used, free = shutil.disk_usage("/content/drive/MyDrive")
        free_gb = free // (1024**3)
        print(f"💾 Free space: {free_gb:.1f} GB")
        if free_gb < 10:
            print("⚠️  Warning: Less than 10GB free space available")
    except:
        pass
    
    print("\n" + "=" * 60)
    print("🎉 MAVERIC setup completed!")
    print("✅ Environment configured")
    print("✅ Dependencies installed") 
    print("✅ Directories created")
    print("✅ Configuration loaded from YAML")
    print("✅ Cache access verified")
    print(f"\n📁 Cache directory: {config['cache_base_dir']}")
    print(f"📁 Results directory: {config['results_dir']}")
    print(f"🤖 CLIP model: {config['clip_model']}")
    print(f"📦 Batch size: {config['batch_size']}")
    print(f"\n🎯 Ready for MAVERIC experiments!")
    print("=" * 60)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MAVERIC Setup Script for Google Colab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 01_setup.py --config /path/to/maveric_config.yaml
  python 01_setup.py -c ./maveric_config.yaml
  python 01_setup.py --config /content/drive/MyDrive/MAVERIC/config.yaml
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to the MAVERIC configuration YAML file'
    )
    
    return parser.parse_args()

def main():
    """Main setup function."""
    # Parse command line arguments
    args = parse_arguments()
    
    print("🚀 Starting MAVERIC setup...")
    print(f"📋 Using config file: {args.config}")
    
    # Validate config file exists
    if not os.path.exists(args.config):
        print(f"❌ Configuration file not found: {args.config}")
        return False
    
    # System setup
    check_gpu()
    install_system_dependencies()
    
    # Google Drive integration
    if not mount_google_drive():
        print("❌ Drive mount failed")
        return False
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        print("❌ Failed to load configuration")
        return False
    
    # Environment and directories
    setup_environment(config, args.config)
    create_directories(config)
    
    # MAVERIC installation
    if not install_maveric(config):
        print("❌ MAVERIC installation failed")
        return False
    
    # Testing and validation
    # if not test_installation():
    #     print("❌ Installation test failed")
    #     return False
    
    if not test_cache_access(config):
        print("❌ Cache access test failed")
        return False
    
    
    # Show completion summary
    show_summary(config)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)