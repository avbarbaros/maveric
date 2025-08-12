#!/usr/bin/env python3
"""
ELEVATER Datasets Experiment Runner for MAVERIC
This script runs MAVERIC quality-driven filtering experiments on all ELEVATER datasets.
"""

import os
import json
import yaml
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Import MAVERIC components
from maveric import MAVERIC, MAVERICConfig
from maveric.datasets import get_dataset
from maveric.utils.logging import setup_logger

def load_experiment_config():
    """Load experiment configuration."""
    config_path = os.environ.get('MAVERIC_CONFIG_PATH', '/content/drive/MyDrive/MAVERIC/repo/maveric/experiments/maveric_config.yaml')

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Configuration loaded from: {config_path}")
        return config
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return None

def setup_maveric_from_config(config):
    """Initialize MAVERIC instance from configuration."""
    print("🔧 Setting up MAVERIC...")
    
    try:
        # Create MAVERICConfig object
        maveric_config = MAVERICConfig(
            cache_base_dir=config['cache_base_dir'],
            clip_model=config['clip_model'],
            batch_size=config['batch_size'],
            device=config['device'],
            enable_caching=config['caching']['enable_image_cache']
        )
        
        # Initialize MAVERIC
        maveric = MAVERIC(maveric_config)
        print("✅ MAVERIC initialized successfully")
        return maveric
        
    except Exception as e:
        print(f"❌ Error setting up MAVERIC: {e}")
        print(traceback.format_exc())
        return None

def run_single_dataset_experiment(maveric, dataset_name: str, config: Dict) -> Dict[str, Any]:
    """Run MAVERIC experiment on a single ELEVATER dataset."""
    print(f"\n{'='*60}")
    print(f"🎯 Processing dataset: {dataset_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    results = {
        'dataset_name': dataset_name,
        'start_time': datetime.now().isoformat(),
        'status': 'started',
        'error': None,
        'metrics': {},
        'filtering_results': {},
        'execution_time': 0
    }
    
    try:
        # Step 1: Retrieve dataset
        print(f"📥 Step 1: Retrieving {dataset_name} dataset...")
        retrieval_result = maveric.retrieve(
            dataset_name="react-vl/react-retrieval-datasets",  # Source dataset
            target_dataset=dataset_name.lower(),
            num_samples=config['elevater']['retrieval']['num_samples']
        )
        
        if retrieval_result is None or len(retrieval_result.retrieved_samples) == 0:
            raise ValueError(f"No samples retrieved for {dataset_name}")
        
        print(f"✅ Retrieved {len(retrieval_result.retrieved_samples)} samples")
        results['retrieved_samples'] = len(retrieval_result.retrieved_samples)
        
        # Step 2: Quality assessment
        print(f"🔍 Step 2: Running quality assessment...")
        quality_result = maveric.assess_quality(retrieval_result)
        
        if quality_result is None:
            raise ValueError(f"Quality assessment failed for {dataset_name}")
        
        print(f"✅ Quality assessment completed")
        results['quality_scores'] = {
            'mean_scores': quality_result.get_mean_scores(),
            'score_distribution': quality_result.get_score_distribution()
        }
        
        # Step 3: Apply filtering with configured thresholds
        print(f"🎛️ Step 3: Applying quality filtering...")
        filtered_result = maveric.quality_control(
            quality_result,
            thresholds=config['quality_thresholds']
        )
        
        if filtered_result is None:
            raise ValueError(f"Quality filtering failed for {dataset_name}")
        
        filtered_count = len(filtered_result.filtered_samples)
        original_count = len(retrieval_result.retrieved_samples)
        retention_rate = (filtered_count / original_count) * 100 if original_count > 0 else 0
        
        print(f"✅ Filtering completed: {filtered_count}/{original_count} samples retained ({retention_rate:.1f}%)")
        
        results['filtering_results'] = {
            'original_samples': original_count,
            'filtered_samples': filtered_count,
            'retention_rate': retention_rate,
            'applied_thresholds': config['quality_thresholds']
        }
        
        # Step 4: Save results
        print(f"💾 Step 4: Saving results...")
        dataset_results_dir = f"{config['results_dir']}/elevater_results/{dataset_name}"
        Path(dataset_results_dir).mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        detailed_results_path = f"{dataset_results_dir}/detailed_results.json"
        with open(detailed_results_path, 'w') as f:
            json.dump({
                'dataset_name': dataset_name,
                'retrieval_result': retrieval_result.to_dict(),
                'quality_result': quality_result.to_dict(),
                'filtered_result': filtered_result.to_dict(),
                'config': config
            }, f, indent=2)
        
        # Save summary
        summary_path = f"{dataset_results_dir}/summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate visualizations if enabled
        if config['experiment']['generate_visualizations']:
            print(f"📊 Step 5: Generating visualizations...")
            try:
                viz_dir = f"{dataset_results_dir}/visualizations"
                Path(viz_dir).mkdir(parents=True, exist_ok=True)
                
                # Quality distribution plots
                maveric.visualize_quality_distribution(
                    quality_result, 
                    save_path=f"{viz_dir}/quality_distribution.png"
                )
                
                # Sample gallery
                if config['experiment']['save_sample_images']:
                    maveric.create_sample_gallery(
                        filtered_result,
                        save_path=f"{viz_dir}/sample_gallery.png",
                        max_samples=config['experiment']['max_visualization_samples']
                    )
                
                print("✅ Visualizations generated")
                
            except Exception as viz_error:
                print(f"⚠️  Visualization generation failed: {viz_error}")
                results['visualization_error'] = str(viz_error)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        results['execution_time'] = execution_time
        results['status'] = 'completed'
        results['end_time'] = datetime.now().isoformat()
        
        print(f"🎉 {dataset_name} experiment completed successfully in {execution_time:.1f} seconds")
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"Error processing {dataset_name}: {str(e)}"
        print(f"❌ {error_msg}")
        print(traceback.format_exc())
        
        results.update({
            'status': 'failed',
            'error': error_msg,
            'traceback': traceback.format_exc(),
            'execution_time': execution_time,
            'end_time': datetime.now().isoformat()
        })
    
    return results

def save_experiment_summary(all_results: List[Dict], config: Dict, selected_datasets: List[str]):
    """Save overall experiment summary."""
    print("\n📋 Generating experiment summary...")
    
    summary = {
        'experiment_info': {
            'start_time': datetime.now().isoformat(),
            'total_datasets': len(selected_datasets),
            'selected_datasets': selected_datasets,
            'config': config
        },
        'results_summary': {
            'completed': len([r for r in all_results if r['status'] == 'completed']),
            'failed': len([r for r in all_results if r['status'] == 'failed']),
            'total_execution_time': sum(r['execution_time'] for r in all_results)
        },
        'detailed_results': all_results
    }
    
    # Calculate aggregate statistics for completed experiments
    completed_results = [r for r in all_results if r['status'] == 'completed']
    if completed_results:
        total_retrieved = sum(r.get('retrieved_samples', 0) for r in completed_results)
        total_filtered = sum(r['filtering_results'].get('filtered_samples', 0) for r in completed_results)
        avg_retention = sum(r['filtering_results'].get('retention_rate', 0) for r in completed_results) / len(completed_results)
        
        summary['aggregate_statistics'] = {
            'total_samples_retrieved': total_retrieved,
            'total_samples_filtered': total_filtered,
            'overall_retention_rate': (total_filtered / total_retrieved * 100) if total_retrieved > 0 else 0,
            'average_retention_rate': avg_retention
        }
    
    # Save summary
    summary_path = f"{config['results_dir']}/elevater_experiment_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Experiment summary saved to: {summary_path}")
    return summary

def display_dataset_options(datasets: List[str]) -> None:
    """Display all available ELEVATER dataset options to the user."""
    print("\n📊 Available ELEVATER Datasets:")
    print("=" * 50)
    for i, dataset in enumerate(datasets, 1):
        print(f"  {i:2d}. {dataset}")
    print("=" * 50)
    print(f"Total: {len(datasets)} datasets available")

def get_user_dataset_selection(datasets: List[str]) -> List[str]:
    """Get user selection for which datasets to process."""
    while True:
        print("\n🎯 Dataset Selection Options:")
        print("  • Enter numbers (e.g., '1,3,5' or '1-5,8,10')")
        print("  • Enter 'all' to run all datasets")
        print("  • Enter 'q' to quit")
        
        user_input = input("\nYour selection: ").strip().lower()
        
        if user_input == 'q':
            print("👋 Exiting...")
            return []
        
        if user_input == 'all':
            print(f"✅ Selected all {len(datasets)} datasets")
            return datasets
        
        try:
            selected_indices = []
            
            # Parse comma-separated values and ranges
            for part in user_input.split(','):
                part = part.strip()
                if '-' in part:
                    # Handle range (e.g., "1-5")
                    start, end = map(int, part.split('-'))
                    selected_indices.extend(range(start, end + 1))
                else:
                    # Handle single number
                    selected_indices.append(int(part))
            
            # Validate indices
            selected_indices = list(set(selected_indices))  # Remove duplicates
            invalid_indices = [i for i in selected_indices if i < 1 or i > len(datasets)]
            
            if invalid_indices:
                print(f"❌ Invalid selection(s): {invalid_indices}. Valid range: 1-{len(datasets)}")
                continue
            
            # Convert to dataset names
            selected_datasets = [datasets[i-1] for i in sorted(selected_indices)]
            
            print(f"\n✅ Selected {len(selected_datasets)} dataset(s):")
            for i, dataset in enumerate(selected_datasets, 1):
                print(f"  {i}. {dataset}")
            
            # Confirm selection
            confirm = input(f"\nProceed with these {len(selected_datasets)} datasets? (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                return selected_datasets
            else:
                print("Let's try again...")
                continue
                
        except ValueError:
            print("❌ Invalid input format. Please use numbers, ranges (1-5), or 'all'")
            continue
        except Exception as e:
            print(f"❌ Error parsing selection: {e}")
            continue

def update_experiment_log(dataset_name: str, status: str, config: Dict):
    """Update the experiment log with progress."""
    log_path = f"{config['results_dir']}/experiment_log.md"
    
    try:
        # Read current log
        with open(log_path, 'r') as f:
            log_content = f.read()
        
        # Update dataset status
        if status == 'completed':
            log_content = log_content.replace(f"- [ ] {dataset_name}", f"- [x] {dataset_name}")
        elif status == 'failed':
            log_content = log_content.replace(f"- [ ] {dataset_name}", f"- [❌] {dataset_name}")
        
        # Write back
        with open(log_path, 'w') as f:
            f.write(log_content)
            
    except Exception as e:
        print(f"⚠️  Could not update experiment log: {e}")

def main():
    """Main experiment runner."""
    print("🚀 Starting ELEVATER Datasets Experiments with MAVERIC")
    print("=" * 70)
    
    # Load configuration
    config = load_experiment_config()
    if not config:
        print("❌ Failed to load configuration. Exiting.")
        return False
    
    # Setup logger
    setup_logger(
        log_file=config['logging']['log_file'],
        level=config['logging']['level']
    )
    
    # Initialize MAVERIC
    maveric = setup_maveric_from_config(config)
    if not maveric:
        print("❌ Failed to initialize MAVERIC. Exiting.")
        return False
    
    # Get list of available datasets
    available_datasets = config['elevater']['datasets']
    
    # Display dataset options to user
    display_dataset_options(available_datasets)
    
    # Get user selection
    selected_datasets = get_user_dataset_selection(available_datasets)
    
    if not selected_datasets:
        print("❌ No datasets selected or user quit. Exiting.")
        return False
    
    print(f"\n🎯 Processing {len(selected_datasets)} selected dataset(s)")
    print("=" * 70)
    
    # Run experiments on selected datasets
    all_results = []
    
    for i, dataset_name in enumerate(selected_datasets, 1):
        print(f"\n🔄 Processing dataset {i}/{len(selected_datasets)}: {dataset_name}")
        
        try:
            # Run experiment for this dataset
            result = run_single_dataset_experiment(maveric, dataset_name, config)
            all_results.append(result)
            
            # Update experiment log
            update_experiment_log(dataset_name, result['status'], config)
            
            # Save intermediate results
            if config['experiment']['save_intermediate_results']:
                intermediate_path = f"{config['results_dir']}/intermediate_results.json"
                with open(intermediate_path, 'w') as f:
                    json.dump(all_results, f, indent=2)
            
            print(f"✅ Dataset {i}/{len(selected_datasets)} completed: {dataset_name}")
            
        except KeyboardInterrupt:
            print("\n⚠️  Experiment interrupted by user")
            break
        except Exception as e:
            print(f"❌ Unexpected error with {dataset_name}: {e}")
            result = {
                'dataset_name': dataset_name,
                'status': 'failed',
                'error': str(e),
                'execution_time': 0
            }
            all_results.append(result)
            update_experiment_log(dataset_name, 'failed', config)
    
    # Generate final summary
    print("\n" + "="*70)
    print("📊 Generating final experiment summary...")
    summary = save_experiment_summary(all_results, config, selected_datasets)
    
    # Print final results
    print("\n🎉 ELEVATER Experiments Completed!")
    print("="*70)
    print(f"📈 Results Summary:")
    print(f"  • Total datasets: {summary['experiment_info']['total_datasets']}")
    print(f"  • Completed successfully: {summary['results_summary']['completed']}")
    print(f"  • Failed: {summary['results_summary']['failed']}")
    print(f"  • Total execution time: {summary['results_summary']['total_execution_time']:.1f} seconds")
    
    if 'aggregate_statistics' in summary:
        stats = summary['aggregate_statistics']
        print(f"  • Total samples retrieved: {stats['total_samples_retrieved']:,}")
        print(f"  • Total samples after filtering: {stats['total_samples_filtered']:,}")
        print(f"  • Overall retention rate: {stats['overall_retention_rate']:.1f}%")
        print(f"  • Average retention rate: {stats['average_retention_rate']:.1f}%")
    
    print(f"\n📁 Results saved to: {config['results_dir']}")
    print("="*70)
    
    return True

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)