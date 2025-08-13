#!/usr/bin/env python3
"""
MAVERIC Data Curation Experiment
Processes retrieved dataset JSON and outputs training dataset JSON after quality control.
"""

import json
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
from maveric.core.interfaces import RetrievalResult


def load_config_file(config_path: str) -> Dict:
    """Load MAVERIC configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Configuration loaded from: {config_path}")
        return config
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return None


def load_retrieved_dataset(file_path: str) -> Optional[Dict]:
    """Load retrieved dataset JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"✅ Retrieved dataset loaded from: {file_path}")
        print(f"📊 Total samples: {len(data)}")
        return data
    except Exception as e:
        print(f"❌ Error loading retrieved dataset: {e}")
        return None


def convert_to_retrieval_result(data: list, source_dataset: str, target_dataset: str) -> RetrievalResult:
    """Convert retrieved dataset JSON to RetrievalResult object."""
    return RetrievalResult(
        samples=data,
        source_dataset=source_dataset,
        target_dataset=target_dataset
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MAVERIC Data Curation (Quality Control)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 02_data_curation.py --input cifar10_retrieved_maveric_dataset1.json --config maveric_config.yaml
  python 02_data_curation.py -i results/imagenet_retrieved_maveric_dataset1.json -c config.yaml
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to retrieved dataset JSON file'
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
    
    parser.add_argument(
        '--balance-strategy',
        type=str,
        default='median',
        choices=['none', 'median', 'min', 'max'],
        help='Dataset balancing strategy (default: median)'
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
            default_thresholds=config.get('quality_thresholds', {})
        )
        
        # Initialize MAVERIC
        maveric = MAVERIC(maveric_config)
        print("✅ MAVERIC initialized successfully")
        return maveric
        
    except Exception as e:
        print(f"❌ Error setting up MAVERIC: {e}")
        return None


def extract_dataset_name_from_filename(filename: str) -> str:
    """Extract dataset name from the input filename."""
    # Extract dataset name from filename like "cifar10_retrieved_maveric_dataset1.json"
    basename = Path(filename).stem  # Remove extension
    if '_retrieved_maveric_dataset' in basename:
        return basename.split('_retrieved_maveric_dataset')[0]
    return "unknown"


def main():
    """Main data curation function."""
    args = parse_arguments()
    
    print("🚀 Starting MAVERIC Data Curation (Quality Control)...")
    print(f"📁 Input file: {args.input}")
    print(f"📋 Configuration file: {args.config}")
    
    # Validate input files exist
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return False
    
    if not os.path.exists(args.config):
        print(f"❌ Configuration file not found: {args.config}")
        return False
    
    try:
        # Load configuration
        config = load_config_file(args.config)
        if not config:
            print("❌ Failed to load configuration")
            return False
        
        # Load retrieved dataset
        retrieved_data = load_retrieved_dataset(args.input)
        if not retrieved_data:
            print("❌ Failed to load retrieved dataset")
            return False
        
        # Extract dataset name from filename
        target_dataset = extract_dataset_name_from_filename(args.input)
        print(f"🎯 Target dataset: {target_dataset}")
        
        # Setup MAVERIC
        maveric = setup_maveric(config)
        if not maveric:
            print("❌ Failed to initialize MAVERIC")
            return False
        
        # Convert to RetrievalResult
        print("🔄 Converting to RetrievalResult...")
        retrieval_result = convert_to_retrieval_result(
            retrieved_data,
            source_dataset="react-vl/react-retrieval-datasets",
            target_dataset=target_dataset
        )
        
        # Apply quality control
        print("🔍 Applying quality control filtering...")
        quality_result = maveric.quality_control(
            data=retrieval_result,
            thresholds=config.get('quality_thresholds'),
            weights=config.get('metric_weights'),
            balance_strategy=args.balance_strategy
        )
        
        if quality_result is None or len(quality_result.filtered_samples) == 0:
            print("❌ Quality control failed or no samples passed filtering")
            return False
        
        print(f"✅ Quality control completed")
        print(f"📊 Original samples: {quality_result.original_count}")
        print(f"📊 Filtered samples: {quality_result.filtered_count}")
        print(f"📊 Retention rate: {quality_result.retention_rate:.1%}")
        
        # Display per-class statistics if available
        if quality_result.class_statistics:
            print("\n📋 Per-class retention:")
            for class_name, stats in sorted(quality_result.class_statistics.items()):
                print(f"   {class_name}: {stats['filtered_count']}/{stats['original_count']} ({stats['retention_rate']:.1%})")
        
        # Export using the built-in method
        print("💾 Exporting training dataset JSON...")
        output_path = quality_result.export_training_dataset_json(
            target_dataset=target_dataset,
            dataset_id=args.dataset_id,
            output_dir=args.output_dir
        )
        
        print(f"✅ Training dataset saved to: {output_path}")
        
        # Display sample from output
        if quality_result.filtered_samples:
            print("\n📋 Sample training data structure:")
            sample = quality_result.filtered_samples[0]
            for key in ['id', 'url', 'label', 'text', 'weighted_class_score', 'consistency']:
                if key in sample:
                    print(f"   {key}: {sample[key]}")
        
        print("\n🎉 Data curation completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during data curation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)