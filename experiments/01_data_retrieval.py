#!/usr/bin/env python3
"""
MAVERIC Data Retrieval Experiment
Retrieves samples from source dataset and outputs JSON with all metrics.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, Optional

# Add maveric to path if running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from maveric import MAVERIC
from maveric.config import MAVERICConfig


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
  python 01_data_retrieval.py -c /path/to/config.yaml --output-dir ./results
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to MAVERIC configuration YAML file'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./results',
        help='Output directory for results (default: ./results)'
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
            enable_image_cache=config.get('caching', {}).get('enable_image_cache', True)
        )
        
        # Initialize MAVERIC
        maveric = MAVERIC(maveric_config)
        print("✅ MAVERIC initialized successfully")
        return maveric
        
    except Exception as e:
        print(f"❌ Error setting up MAVERIC: {e}")
        return None


def main():
    """Main data retrieval function."""
    args = parse_arguments()
    
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
        
        # Perform retrieval
        print("🔍 Starting retrieval process...")
        retrieval_result = maveric.retrieve(
            dataset_name="react-vl/react-retrieval-datasets",
            target_dataset=selected_dataset.lower(),
            num_samples=num_samples,
            start_index=0,
            cache_results=True
        )
        
        if retrieval_result is None or len(retrieval_result.samples) == 0:
            print(f"❌ No samples retrieved for {selected_dataset}")
            return False
        
        # Export using the built-in method
        print("💾 Exporting retrieved dataset JSON...")
        output_path = retrieval_result.export_retrieved_dataset_json(
            target_dataset=selected_dataset,
            dataset_id=args.dataset_id,
            output_dir=args.output_dir
        )
        
        print(f"✅ Results saved to: {output_path}")
        print(f"📊 Total samples retrieved: {len(retrieval_result.samples)}")
        
        # Display cache statistics
        cache_stats = maveric.get_cache_stats()
        if cache_stats:
            print("\n📦 Cache Statistics:")
            for key, value in cache_stats.items():
                print(f"   {key}: {value}")
        
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