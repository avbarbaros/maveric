"""Command-line interface for MAVERIC."""

import argparse
import json
from pathlib import Path
import sys

from .main import MAVERIC
from .config import MAVERICConfig, TrainingConfig


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MAVERIC: Mahalanobis-based Adaptive Vision-language Efficient Retrieval with Integrated Curation"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Retrieve command
    retrieve_parser = subparsers.add_parser('retrieve', help='Retrieve samples from dataset')
    retrieve_parser.add_argument('--source', required=True, help='Source dataset name')
    retrieve_parser.add_argument('--target', required=True, help='Target dataset name')
    retrieve_parser.add_argument('--num-samples', type=int, help='Number of samples to retrieve')
    retrieve_parser.add_argument('--start-index', type=int, default=0, help='Starting index')
    retrieve_parser.add_argument('--config', help='Configuration file')
    retrieve_parser.add_argument('--output', help='Output file for results')
    
    # Quality control command
    qc_parser = subparsers.add_parser('quality-control', help='Apply quality control filtering')
    qc_parser.add_argument('--input', required=True, help='Input data file')
    qc_parser.add_argument('--thresholds', help='JSON file with thresholds')
    qc_parser.add_argument('--balance', choices=['none', 'median', 'mean', 'min', 'max'], 
                          default='median', help='Balance strategy')
    qc_parser.add_argument('--output', required=True, help='Output file for filtered data')
    qc_parser.add_argument('--config', help='Configuration file')
    
    # Customize command
    customize_parser = subparsers.add_parser('customize', help='Customize model with filtered data')
    customize_parser.add_argument('--input', required=True, help='Input filtered data')
    customize_parser.add_argument('--model', default='openai/clip-vit-base-patch32', 
                                 help='Base model name')
    customize_parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    customize_parser.add_argument('--output-dir', required=True, help='Output directory')
    customize_parser.add_argument('--config', help='Configuration file')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Visualize data distributions')
    viz_parser.add_argument('--input', required=True, help='Input data file')
    viz_parser.add_argument('--metrics', nargs='+', help='Metrics to visualize')
    viz_parser.add_argument('--output-dir', required=True, help='Output directory for plots')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == 'retrieve':
            cmd_retrieve(args)
        elif args.command == 'quality-control':
            cmd_quality_control(args)
        elif args.command == 'customize':
            cmd_customize(args)
        elif args.command == 'visualize':
            cmd_visualize(args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_retrieve(args):
    """Execute retrieve command."""
    # Load config
    if args.config:
        maveric = MAVERIC.from_config_file(args.config)
    else:
        maveric = MAVERIC()
    
    # Perform retrieval
    result = maveric.retrieve(
        dataset_name=args.source,
        target_dataset=args.target,
        num_samples=args.num_samples,
        start_index=args.start_index
    )
    
    # Save results
    if args.output:
        result.save(args.output)
        print(f"Saved {result.total_samples} samples to {args.output}")
    else:
        print(f"Retrieved {result.total_samples} samples")
        print(f"Class distribution: {result.class_distribution}")


def cmd_quality_control(args):
    """Execute quality control command."""
    # Load config
    if args.config:
        maveric = MAVERIC.from_config_file(args.config)
    else:
        maveric = MAVERIC()
    
    # Load thresholds
    thresholds = None
    if args.thresholds:
        with open(args.thresholds, 'r') as f:
            thresholds = json.load(f)
    
    # Apply quality control
    result = maveric.quality_control(
        data=args.input,
        thresholds=thresholds,
        balance_strategy=args.balance
    )
    
    # Save results
    output_path = Path(args.output)
    if output_path.suffix == '.json':
        maveric.quality_controller.save_filtered_data(args.output, format='json')
    else:
        maveric.quality_controller.save_filtered_data(args.output, format='csv')
    
    print(f"Saved {result.filtered_count} filtered samples to {args.output}")
    print(f"Retention rate: {result.retention_rate:.1%}")


def cmd_customize(args):
    """Execute model customization command."""
    # Load config
    if args.config:
        maveric = MAVERIC.from_config_file(args.config)
    else:
        maveric = MAVERIC()
    
    # Create training config
    training_config = TrainingConfig(
        epochs=args.epochs,
        checkpoint_dir=args.output_dir
    )
    
    # Load quality result
    from .core.interfaces import QualityResult
    import pandas as pd
    
    # Simple loading - in practice, would need proper deserialization
    data = pd.read_json(args.input) if args.input.endswith('.json') else pd.read_csv(args.input)
    
    quality_result = QualityResult(
        filtered_samples=data.to_dict('records'),
        original_samples=data.to_dict('records'),  # Simplified
        thresholds={},
        balance_strategy='none'
    )
    
    # Customize model
    result = maveric.customize_model(
        quality_result=quality_result,
        model_name=args.model,
        training_config=training_config
    )
    
    print(f"Training complete!")
    print(f"Test accuracy: {result.test_accuracy:.2f}%")
    print(f"Improvement over baseline: {result.improvement:+.2f}%")
    
    # Save summary
    summary_path = Path(args.output_dir) / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'test_accuracy': result.test_accuracy,
            'baseline_accuracy': result.zero_shot_baseline,
            'improvement': result.improvement,
            'training_samples': result.training_samples,
            'checkpoint': result.checkpoint_path
        }, f, indent=2)


def cmd_visualize(args):
    """Execute visualization command."""
    from .visualization import MetricsVisualizer
    import pandas as pd
    
    # Load data
    if args.input.endswith('.json'):
        data = pd.read_json(args.input)
    else:
        data = pd.read_csv(args.input)
    
    # Create visualizer
    viz = MetricsVisualizer()
    
    # Determine metrics
    if args.metrics:
        metrics = args.metrics
    else:
        metrics = [col for col in data.columns if 'score' in col or 'consistency' in col]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    for metric in metrics:
        if metric in data.columns:
            fig = viz.plot_metric_distribution(
                data, 
                metric,
                save_path=output_dir / f"{metric}_distribution.png"
            )
            print(f"Saved {metric} distribution plot")
    
    # Class distribution
    if 'label' in data.columns:
        from .visualization.plots import plot_class_distribution
        fig = plot_class_distribution(
            data,
            save_path=output_dir / "class_distribution.png"
        )
        print("Saved class distribution plot")
    
    print(f"All plots saved to {output_dir}")


if __name__ == "__main__":
    main()
