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
  # Auto-detects raw subfolder: reads from results_dir/cifar10/raw/, outputs to results_dir/cifar10/
  python 02_data_curation.py --input-dir /content/drive/MyDrive/MAVERIC/maveric_experiments/cifar10 --dataset-name cifar10 --config maveric_config.yaml
  
  # Explicitly specify raw folder
  python 02_data_curation.py --input-dir /content/drive/MyDrive/MAVERIC/maveric_experiments/cifar10/raw --dataset-name cifar10 --config maveric_config.yaml
  
  # Override with custom output directory
  python 02_data_curation.py -i ./custom_input -d imagenet -c config.yaml --output-dir ./custom_output
        """
    )
    
    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        required=True,
        help='Directory containing rotation files from data retrieval (should point to results_dir/datasetName/raw/)'
    )
    
    parser.add_argument(
        '--dataset-name', '-d',
        type=str,
        required=True,
        help='Target dataset name (e.g., cifar10, imagenet) to process rotation files'
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
        default=None,
        help='Output directory for results (default: results_dir/datasetName from config)'
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
            default_thresholds=config.get('quality_thresholds', {}),
            balance_min_samples=config.get('elevater', {}).get('quality_control', {}).get('min_samples_per_class', 15),
            # Additional configuration mappings
            retrieval_rotation_size=config.get('retrieval_rotation_size', 1000),
            enable_real_time_stats=config.get('enable_real_time_stats', True),
            metric_weights=config.get('metric_weights', {}),
            num_workers=config.get('performance', {}).get('num_workers', 4),
            log_level=config.get('logging', {}).get('level', 'INFO'),
            viz_save_figures=config.get('experiment', {}).get('save_visualizations', False)
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


def validate_and_adjust_input_dir(input_dir: str, dataset_name: str) -> str:
    """
    Validate input directory and auto-detect raw subfolder if needed.
    
    Args:
        input_dir: User-provided input directory
        dataset_name: Dataset name for validation
        
    Returns:
        Validated input directory path
    """
    input_path = Path(input_dir)
    
    # If the provided path exists and contains rotation files, use it directly
    if input_path.exists():
        pattern = f"{dataset_name.lower()}_raw_maveric_*.json"
        rotation_files = list(input_path.glob(pattern))
        if rotation_files:
            return str(input_path)
        
        # If no rotation files found, check if there's a 'raw' subfolder
        raw_subfolder = input_path / "raw"
        if raw_subfolder.exists():
            rotation_files = list(raw_subfolder.glob(pattern))
            if rotation_files:
                print(f"📁 Auto-detected raw subfolder: {raw_subfolder}")
                return str(raw_subfolder)
    
    # Return original path (will be validated later)
    return str(input_path)


def main():
    """Main data curation function."""
    args = parse_arguments()
    
    print("🚀 Starting MAVERIC Data Curation (Quality Control)...")
    print(f"🎯 Target dataset: {args.dataset_name}")
    print(f"📋 Configuration file: {args.config}")
    
    # Validate and adjust input directory
    validated_input_dir = validate_and_adjust_input_dir(args.input_dir, args.dataset_name)
    print(f"📁 Input directory: {validated_input_dir}")
    
    # Validate input directory and config file exist
    if not os.path.exists(validated_input_dir):
        print(f"❌ Input directory not found: {validated_input_dir}")
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
        
        # Setup MAVERIC
        maveric = setup_maveric(config)
        if not maveric:
            print("❌ Failed to initialize MAVERIC")
            return False
        
        # Determine output directory - use dataset-specific subdirectory for better organization
        if args.output_dir is not None:
            # User specified custom output directory
            output_dir = args.output_dir
        else:
            # Use results_dir from config and create dataset-specific subdirectory
            base_results_dir = config.get('results_dir', './results')
            output_dir = f"{base_results_dir}/{args.dataset_name}"
        
        print(f"📁 Output directory: {output_dir}")
        
        # Apply quality control using rotation files directly
        print("🔍 Loading rotation files and applying quality control filtering...")
        quality_result = maveric.quality_control(
            data=(args.dataset_name, validated_input_dir),
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
        
        # Export using the built-in method with rotation size from config
        rotation_size = config.get('retrieval_rotation_size', 1000)
        total_samples = len(quality_result.filtered_samples)
        
        if total_samples > rotation_size:
            num_files = (total_samples + rotation_size - 1) // rotation_size  # Ceiling division
            print(f"💾 Exporting training dataset JSON ({total_samples} samples → {num_files} files, {rotation_size} samples per file)...")
        else:
            print(f"💾 Exporting training dataset JSON ({total_samples} samples → 1 file)...")
            
        output_path = quality_result.export_training_dataset_json(
            target_dataset=args.dataset_name,
            dataset_id=args.dataset_id,
            output_dir=output_dir,
            rotation_size=rotation_size
        )
        
        if total_samples > rotation_size:
            print(f"✅ Training dataset files saved to: {output_dir}")
            print(f"   First file: {output_path}")
        else:
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