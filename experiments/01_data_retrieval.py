#!/usr/bin/env python3
"""
MAVERIC Data Retrieval Experiment
Retrieves samples from source dataset and outputs JSON with all metrics.
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

# Add maveric to path if running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from maveric import MAVERIC
from maveric.config import MAVERICConfig
        

def configure_logging():
    """Configure logging to suppress download warnings."""
    # Set CacheManager warnings to ERROR level to suppress them
    cache_logger = logging.getLogger('maveric.CacheManager')
    cache_logger.setLevel(logging.ERROR)
    
    # Keep other maveric loggers at INFO level for important messages
    maveric_logger = logging.getLogger('maveric')
    maveric_logger.setLevel(logging.INFO)


def load_config_file(config_path: str) -> Dict:
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


def get_user_file_sequence(start_index: int) -> int:
    """Get user selection for starting file sequence number."""
    # Suggest a reasonable starting file number based on start_index
    suggested_file_id = max(1, (start_index // 500) + 1) if start_index > 0 else 1
    
    while True:
        print(f"\n📁 Output File Sequence Selection:")
        print(f"  • Starting index: {start_index:,}")
        print(f"  • Suggested file sequence: {suggested_file_id} (based on batch size 500)")
        print(f"  • Output format: datasetname_raw_maveric_{{sequence}}.json")
        print("  • Enter file sequence number (e.g., '1', '5', '10')")
        print("  • Enter 'q' to quit")
        
        user_input = input(f"\nFile sequence number (default: {suggested_file_id}): ").strip().lower()
        
        if user_input == 'q':
            print("👋 Exiting...")
            return None
        
        # Use suggested value if user just presses enter
        if user_input == '':
            user_input = str(suggested_file_id)
        
        try:
            file_seq = int(user_input)
            if file_seq >= 1:
                print(f"✅ Starting file sequence: {file_seq}")
                return file_seq
            else:
                print("❌ File sequence must be >= 1")
                continue
        except ValueError:
            print("❌ Invalid input. Please enter a valid number or 'q' to quit.")
            continue


def get_user_start_index(dataset_size: int) -> int:
    """Get user selection for starting index with dataset size information."""
    max_index = dataset_size - 1  # 0-based indexing
    
    while True:
        print(f"\n📍 Starting Index Selection:")
        print(f"  • Dataset size: {dataset_size:,} samples")
        print(f"  • Valid index range: 0 to {max_index:,}")
        print("  • Enter starting index (e.g., '0', '1000', '50000')")
        print("  • Enter '0' to start from the beginning")
        print("  • Enter 'q' to quit")
        
        user_input = input(f"\nStarting index (0-{max_index:,}): ").strip().lower()
        
        if user_input == 'q':
            print("👋 Exiting...")
            return None
        
        try:
            start_index = int(user_input)
            if 0 <= start_index <= max_index:
                print(f"✅ Starting from index: {start_index:,}")
                return start_index
            else:
                print(f"❌ Starting index must be between 0 and {max_index:,}")
                continue
        except ValueError:
            print("❌ Invalid input. Please enter a valid number or 'q' to quit.")
            continue


def get_user_dataset_selection(datasets: list) -> str:
    """Get user selection for which dataset to process."""
    while True:
        print("\n🎯 Dataset Selection:")
        print("  • Enter dataset number (e.g., '1', '5', '10')")
        print("  • Enter dataset name directly (e.g., 'cifar10', 'imagenet')")
        print("  • Enter 'q' to quit")
        
        user_input = input("\nYour selection: ").strip().lower()
        
        if user_input == 'q':
            print("👋 Exiting...")
            return None
        
        # Try to parse as number
        try:
            idx = int(user_input)
            if 1 <= idx <= len(datasets):
                selected_dataset = datasets[idx - 1]
                print(f"✅ Selected dataset: {selected_dataset}")
                return selected_dataset
            else:
                print(f"❌ Invalid number. Valid range: 1-{len(datasets)}")
                continue
        except ValueError:
            pass
        
        # Try to match as dataset name
        if user_input in datasets:
            print(f"✅ Selected dataset: {user_input}")
            return user_input
        
        # Try partial matching
        matches = [d for d in datasets if user_input in d.lower()]
        if len(matches) == 1:
            print(f"✅ Selected dataset: {matches[0]} (matched '{user_input}')")
            return matches[0]
        elif len(matches) > 1:
            print(f"❌ Multiple matches for '{user_input}': {', '.join(matches)}")
            print("Please be more specific.")
            continue
        
        print(f"❌ No dataset found matching '{user_input}'")
        print("Please enter a valid dataset number or name.")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MAVERIC Data Retrieval with ELEVATER Dataset Selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 01_data_retrieval.py --config maveric_config.yaml
  python 01_data_retrieval.py -c /path/to/config.yaml
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to MAVERIC configuration YAML file'
    )
    
    parser.add_argument(
        '--dataset-id',
        type=int,
        default=1,
        help='Dataset ID for output filename (default: 1)'
    )
    
    return parser.parse_args()


def setup_maveric(config: Dict) -> MAVERIC:
    """Setup MAVERIC instance with configuration."""
    print("🔧 Setting up MAVERIC...")
    
    try:
        # Create MAVERICConfig object from loaded config
        maveric_config = MAVERICConfig(
            cache_base_dir=config['cache_base_dir'],
            clip_model=config.get('clip_model', 'ViT-B/32'),
            batch_size=config.get('batch_size', 32),
            device=config.get('device', 'auto'),
            enable_image_cache=config.get('caching', {}).get('enable_image_cache', True),
            retrieval_rotation_size=config.get('retrieval_rotation_size', 1000)
        )
        
        # Initialize MAVERIC (real-time stats are enabled by default)
        maveric = MAVERIC(maveric_config)
        print("✅ MAVERIC initialized successfully")
        print(f"🔄 Rotation size configured: {maveric_config.retrieval_rotation_size}")
        return maveric
        
    except Exception as e:
        print(f"❌ Error setting up MAVERIC: {e}")
        return None


def main():
    """Main data retrieval function."""
    args = parse_arguments()
    
    # Configure logging to suppress download warnings
    configure_logging()
    
    print("🚀 Starting MAVERIC Data Retrieval...")
    print(f"📋 Configuration file: {args.config}")
    
    # Validate config file exists
    if not os.path.exists(args.config):
        print(f"❌ Configuration file not found: {args.config}")
        return False
    
    # Load configuration
    config = load_config_file(args.config)
    if not config:
        print("❌ Failed to load configuration")
        return False
    
    # Get number of samples from config
    num_samples = config.get('elevater', {}).get('retrieval', {}).get('num_samples', 1000)
    print(f"📈 Samples per retrieval: {num_samples}")
    
    # Get output directory from config
    output_dir = config.get('results_dir', './results')
    print(f"📁 Output directory: {output_dir}")
    
    try:
        # Setup MAVERIC 
        maveric = setup_maveric(config)
        if not maveric:
            print("❌ Failed to initialize MAVERIC")
            return False
        
        # Display ELEVATER datasets and get user selection
        available_datasets = MAVERIC.display_elevater_datasets()
        selected_dataset = get_user_dataset_selection(available_datasets)
        
        if not selected_dataset:
            print("❌ No dataset selected or user quit")
            return False
        
        print(f"🎯 Target dataset: {selected_dataset}")
        print(f"📊 Source: react-vl/react-retrieval-datasets")
        
        # Get dataset size to show user valid range
        print("📏 Getting dataset size...")
        from maveric.retrieval.dataset_handlers import REACTDatasetHandler
        dataset_handler = REACTDatasetHandler("react-vl/react-retrieval-datasets")
        dataset_size = len(dataset_handler)
        print(f"📊 Dataset contains {dataset_size:,} samples")
        
        # Get starting index from user
        start_index = get_user_start_index(dataset_size)
        if start_index is None:
            print("❌ No starting index selected or user quit")
            return False
        
        # Get starting file sequence from user
        start_file_sequence = get_user_file_sequence(start_index)
        if start_file_sequence is None:
            print("❌ No file sequence selected or user quit")
            return False
        
        # Create dataset-specific output directory
        dataset_output_dir = Path(output_dir) / selected_dataset.lower()
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Dataset output directory: {dataset_output_dir}")
        
        # Perform retrieval
        print("🔍 Starting retrieval process...")
        retrieval_result = maveric.retrieve(
            dataset_name="react-vl/react-retrieval-datasets",
            target_dataset=selected_dataset.lower(),
            num_samples=num_samples,
            start_index=start_index,
            start_file_id=start_file_sequence,
            cache_results=(start_index == 0),  # Only cache if starting from beginning
            export_rotation_files=True,
            rotation_export_dir=str(dataset_output_dir)
        )
        
        if retrieval_result is None or len(retrieval_result.samples) == 0:
            print(f"❌ No samples retrieved for {selected_dataset}")
            return False
        
        print(f"📊 Total samples retrieved: {len(retrieval_result.samples)}")
        print(f"✅ Rotation files automatically exported to: {dataset_output_dir}")
        
        # List the rotation files that were created
        output_dir_path = dataset_output_dir
        rotation_files = sorted(output_dir_path.glob(f"{selected_dataset.lower()}_raw_maveric_*.json"))
        if rotation_files:
            print(f"📋 Created {len(rotation_files)} rotation files:")
            for file_path in rotation_files:
                print(f"   ✅ {file_path.name}")
        else:
            print("⚠️  No rotation files found - samples may not have reached rotation size")
        
        # Display final cache statistics
        cache_stats = maveric.get_cache_stats()
        if cache_stats:
            print("\n📈 Final Download Summary:")
            successful = cache_stats.get('downloads_successful', 0)
            failed = cache_stats.get('downloads_failed', 0)
            cache_hits = cache_stats.get('cache_hits', 0)
            
            print(f"   ✅ Successful downloads: {successful}")
            print(f"   ❌ Failed downloads: {failed}")
            print(f"   🎯 Cache hits: {cache_hits}")
            
            if successful + failed > 0:
                success_rate = (successful / (successful + failed)) * 100
                print(f"   📊 Success rate: {success_rate:.1f}%")
            
            # Show other cache info
            cache_size = cache_stats.get('cache_size', {})
            if cache_size:
                images_mb = cache_size.get('images_mb', 0)
                if images_mb > 0:
                    print(f"   💾 Image cache size: {images_mb:.1f} MB")
        
        print("\n🎉 Data retrieval completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during data retrieval: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)