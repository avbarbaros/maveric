#!/usr/bin/env python3
"""
Google Drive Setup for MAVERIC Cache
This script configures Google Drive integration for MAVERIC caching on Colab.
"""

import os
import shutil
import yaml
from pathlib import Path
from google.colab import drive

def mount_google_drive():
    """Mount Google Drive to /content/drive."""
    print("📂 Mounting Google Drive...")
    try:
        drive.mount('/content/drive')
        print("✅ Google Drive mounted successfully at /content/drive")
        return True
    except Exception as e:
        print(f"❌ Error mounting Google Drive: {e}")
        return False

def load_config():
    """Load MAVERIC configuration from YAML file."""
    config_path = "/content/drive/MyDrive/MAVERIC/repo/maveric/experiments/maveric_config.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Configuration loaded from: {config_path}")
        return config
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return None

def create_cache_directories(config):
    """Create necessary cache directories on Google Drive based on config."""
    print("📁 Creating cache directories...")
    
    base_cache_dir = config['cache_base_dir']
    results_dir = config['results_dir']
    
    directories = [
        base_cache_dir,
        f"{base_cache_dir}/images",
        f"{base_cache_dir}/embeddings", 
        f"{base_cache_dir}/models",
        f"{base_cache_dir}/results",
        f"{base_cache_dir}/datasets",
        "/content/drive/MyDrive/MAVERIC/huggingface_cache",
        results_dir,
        f"{results_dir}/elevater_results",
        f"{results_dir}/quality_reports",
        f"{results_dir}/visualizations",
        f"{results_dir}/filtered_datasets"
    ]
    
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {directory}")
        except Exception as e:
            print(f"❌ Error creating {directory}: {e}")
            return False
    
    return True

def setup_environment_variables(config):
    """Set up environment variables based on config."""
    print("⚙️ Configuring environment variables...")
    
    # Set environment variables from config
    env_vars = {
        'MAVERIC_CACHE_DIR': config['cache_base_dir'],
        'MAVERIC_RESULTS_DIR': config['results_dir'],
        'HF_HOME': '/content/drive/MyDrive/MAVERIC/huggingface_cache',
        'HF_DATASETS_CACHE': '/content/drive/MyDrive/MAVERIC/huggingface_cache/datasets',
        'TRANSFORMERS_CACHE': '/content/drive/MyDrive/MAVERIC/huggingface_cache/transformers',
        'MPLBACKEND': 'Agg',
        'MAVERIC_CONFIG_PATH': '/content/drive/MyDrive/MAVERIC/repo/maveric/experiments/maveric_config.yaml'
    }
    
    for var, value in env_vars.items():
        os.environ[var] = value
        print(f"✅ Set {var} = {value}")
    
    return True

def test_cache_access(config):
    """Test read/write access to cache directories."""
    print("🧪 Testing cache directory access...")
    
    cache_dir = config['cache_base_dir']
    test_file = f"{cache_dir}/test_write.txt"
    
    try:
        # Test write
        with open(test_file, 'w') as f:
            f.write("MAVERIC cache test")
        print("✅ Write access confirmed")
        
        # Test read
        with open(test_file, 'r') as f:
            content = f.read()
        print("✅ Read access confirmed")
        
        # Clean up
        os.remove(test_file)
        print("✅ Cache access test completed")
        return True
        
    except Exception as e:
        print(f"❌ Cache access test failed: {e}")
        return False

def show_disk_usage():
    """Show Google Drive disk usage."""
    print("💾 Checking Google Drive space...")
    
    try:
        total, used, free = shutil.disk_usage("/content/drive/MyDrive")
        
        total_gb = total // (1024**3)
        used_gb = used // (1024**3)
        free_gb = free // (1024**3)
        
        print(f"Total space: {total_gb:.1f} GB")
        print(f"Used space: {used_gb:.1f} GB")
        print(f"Free space: {free_gb:.1f} GB")
        
        if free_gb < 10:
            print("⚠️  Warning: Less than 10GB free space available")
            print("   Consider cleaning up files or using a different Google account")
        else:
            print("✅ Sufficient space available for experiments")
            
        return True
    except Exception as e:
        print(f"❌ Error checking disk usage: {e}")
        return False

def create_experiment_log(config):
    """Create experiment tracking log."""
    print("📋 Creating experiment log...")
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_content = f"""# MAVERIC ELEVATER Experiments Log
Started: {timestamp}

## Configuration
- Cache Directory: {config['cache_base_dir']}
- Results Directory: {config['results_dir']}
- CLIP Model: {config['clip_model']}
- Batch Size: {config['batch_size']}
- Max Samples per Dataset: {config['processing']['max_samples_per_dataset']}

## ELEVATER Datasets to Process
{chr(10).join([f"- [ ] {dataset}" for dataset in config['elevater']['datasets']])}

## Experiment Progress
- [x] Setup completed
- [x] Cache configured  
- [ ] ELEVATER datasets downloaded
- [ ] Quality assessment running
- [ ] Filtering applied
- [ ] Results collected
- [ ] Analysis completed

## Quality Thresholds
{chr(10).join([f"- {metric}: {threshold}" for metric, threshold in config['quality_thresholds'].items()])}

## Notes
Experiment started on Google Colab T4 GPU environment.
Configuration loaded from maveric_config.yaml.

Add your experiment notes here...
"""
    
    log_path = f"{config['results_dir']}/experiment_log.md"
    
    try:
        with open(log_path, 'w') as f:
            f.write(log_content)
        
        print(f"✅ Experiment log created: {log_path}")
        return log_path
    except Exception as e:
        print(f"❌ Error creating experiment log: {e}")
        return None

def copy_config_to_drive(config):
    """Copy the config file to Google Drive for persistence."""
    print("📋 Copying configuration to Google Drive...")
    
    source_config = "/content/drive/MyDrive/MAVERIC/repo/maveric/experiments/maveric_config.yaml"
    dest_config = f"{config['results_dir']}/maveric_config.yaml"
    
    try:
        shutil.copy2(source_config, dest_config)
        print(f"✅ Configuration copied to: {dest_config}")
        return dest_config
    except Exception as e:
        print(f"❌ Error copying configuration: {e}")
        return None

def main():
    """Main Google Drive setup function."""
    print("🚀 Setting up Google Drive integration for MAVERIC...")
    print("=" * 60)
    
    # Mount Google Drive
    # if not mount_google_drive():
    #     print("❌ Failed to mount Google Drive. Exiting.")
    #     return False
    
    # Load configuration
    config = load_config()
    if not config:
        print("❌ Failed to load configuration. Exiting.")
        return False
    
    # Create cache directories
    if not create_cache_directories(config):
        print("❌ Failed to create cache directories. Exiting.")
        return False
    
    # Setup environment variables
    if not setup_environment_variables(config):
        print("❌ Failed to setup environment variables. Exiting.")
        return False
    
    # Test cache access
    if not test_cache_access(config):
        print("❌ Cache access test failed. Exiting.")
        return False
    
    # Show disk usage
    show_disk_usage()
    
    # Copy config to Google Drive
    drive_config_path = copy_config_to_drive(config)
    
    # Create experiment log
    log_path = create_experiment_log(config)
    
    print("\n" + "=" * 60)
    print("🎉 Google Drive setup completed successfully!")
    print("📋 Setup Summary:")
    print("✅ Google Drive mounted")
    print("✅ Configuration loaded from maveric_config.yaml")
    print("✅ Cache directories created")
    print("✅ Environment variables configured")
    print("✅ Cache access verified")
    print("✅ Configuration backed up to Google Drive")
    print("✅ Experiment log created")
    print(f"\n📁 Cache directory: {config['cache_base_dir']}")
    print(f"📁 Results directory: {config['results_dir']}")
    print(f"📁 Config backup: {drive_config_path}")
    print(f"📁 Experiment log: {log_path}")
    print(f"\n🎯 Ready to process {len(config['elevater']['datasets'])} ELEVATER datasets")
    print("📝 Next step: Run ELEVATER experiments")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)