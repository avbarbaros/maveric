#!/usr/bin/env python3
"""
Results Analysis and Visualization for MAVERIC ELEVATER Experiments
This script analyzes and visualizes the results from ELEVATER dataset experiments.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import yaml

# Set matplotlib backend for headless environments
plt.style.use('default')
sns.set_palette("husl")

def load_experiment_config():
    """Load experiment configuration."""
    config_path = os.environ.get('MAVERIC_CONFIG_PATH', '/content/drive/MyDrive/MAVERIC/repo/maveric/experiments/maveric_config.yaml')
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return None

def load_experiment_results(results_dir: str) -> Dict[str, Any]:
    """Load experiment results from JSON files."""
    print("📊 Loading experiment results...")
    
    # Load main summary
    summary_path = f"{results_dir}/elevater_experiment_summary.json"
    if not os.path.exists(summary_path):
        print(f"❌ Summary file not found: {summary_path}")
        return None
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    print(f"✅ Loaded summary for {len(summary['detailed_results'])} datasets")
    
    # Load individual dataset results
    dataset_results = {}
    elevater_results_dir = f"{results_dir}/elevater_results"
    
    if os.path.exists(elevater_results_dir):
        for dataset_dir in os.listdir(elevater_results_dir):
            dataset_path = f"{elevater_results_dir}/{dataset_dir}"
            if os.path.isdir(dataset_path):
                detailed_results_path = f"{dataset_path}/detailed_results.json"
                if os.path.exists(detailed_results_path):
                    try:
                        with open(detailed_results_path, 'r') as f:
                            dataset_results[dataset_dir] = json.load(f)
                    except Exception as e:
                        print(f"⚠️  Could not load results for {dataset_dir}: {e}")
    
    print(f"✅ Loaded detailed results for {len(dataset_results)} datasets")
    
    return {
        'summary': summary,
        'dataset_results': dataset_results
    }

def create_results_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    """Create pandas DataFrame from experiment results."""
    print("🔄 Creating results DataFrame...")
    
    detailed_results = results['summary']['detailed_results']
    
    # Extract key metrics for each dataset
    data = []
    for result in detailed_results:
        row = {
            'dataset_name': result['dataset_name'],
            'status': result['status'],
            'execution_time': result.get('execution_time', 0),
            'retrieved_samples': result.get('retrieved_samples', 0)
        }
        
        # Add filtering results if available
        if 'filtering_results' in result:
            filtering = result['filtering_results']
            row.update({
                'filtered_samples': filtering.get('filtered_samples', 0),
                'retention_rate': filtering.get('retention_rate', 0)
            })
        else:
            row.update({
                'filtered_samples': 0,
                'retention_rate': 0
            })
        
        # Add quality scores if available
        if 'quality_scores' in result and result['quality_scores']:
            mean_scores = result['quality_scores'].get('mean_scores', {})
            for metric, score in mean_scores.items():
                row[f'quality_{metric}'] = score
        
        data.append(row)
    
    df = pd.DataFrame(data)
    print(f"✅ DataFrame created with {len(df)} rows and {len(df.columns)} columns")
    return df

def generate_summary_statistics(df: pd.DataFrame, config: Dict, experiment_info: Dict = None) -> Dict[str, Any]:
    """Generate summary statistics from results."""
    print("📈 Generating summary statistics...")
    
    # Filter successful experiments
    successful_df = df[df['status'] == 'completed']
    
    stats = {
        'experiment_overview': {
            'processed_datasets': len(df),
            'successful_experiments': len(successful_df),
            'failed_experiments': len(df) - len(successful_df),
            'success_rate': len(successful_df) / len(df) * 100 if len(df) > 0 else 0,
            'total_execution_time': df['execution_time'].sum(),
            'average_execution_time': df['execution_time'].mean()
        }
    }
    
    # Add selection information if available
    if experiment_info:
        stats['experiment_overview']['total_available_datasets'] = len(config.get('elevater', {}).get('datasets', []))
        stats['experiment_overview']['selected_datasets'] = experiment_info.get('selected_datasets', df['dataset_name'].tolist())
        stats['experiment_overview']['selection_count'] = len(stats['experiment_overview']['selected_datasets'])
        
        # Calculate selection rate
        total_available = stats['experiment_overview']['total_available_datasets']
        if total_available > 0:
            stats['experiment_overview']['selection_rate'] = (stats['experiment_overview']['selection_count'] / total_available) * 100
        else:
            stats['experiment_overview']['selection_rate'] = 100
    
    if len(successful_df) > 0:
        stats['data_statistics'] = {
            'total_samples_retrieved': successful_df['retrieved_samples'].sum(),
            'total_samples_filtered': successful_df['filtered_samples'].sum(),
            'overall_retention_rate': successful_df['retention_rate'].mean(),
            'retention_rate_std': successful_df['retention_rate'].std(),
            'min_retention_rate': successful_df['retention_rate'].min(),
            'max_retention_rate': successful_df['retention_rate'].max()
        }
        
        # Quality metrics statistics
        quality_cols = [col for col in successful_df.columns if col.startswith('quality_')]
        if quality_cols:
            stats['quality_statistics'] = {}
            for col in quality_cols:
                metric_name = col.replace('quality_', '')
                stats['quality_statistics'][metric_name] = {
                    'mean': successful_df[col].mean(),
                    'std': successful_df[col].std(),
                    'min': successful_df[col].min(),
                    'max': successful_df[col].max()
                }
    
    return stats

def create_visualizations(df: pd.DataFrame, results: Dict, output_dir: str):
    """Create comprehensive visualizations of the results."""
    print("🎨 Creating visualizations...")
    
    # Create output directory
    viz_dir = f"{output_dir}/analysis_visualizations"
    Path(viz_dir).mkdir(parents=True, exist_ok=True)
    
    # Set up plot style
    plt.figure(figsize=(15, 10))
    
    # 1. Success Rate Overview
    plt.subplot(2, 3, 1)
    success_counts = df['status'].value_counts()
    colors = ['#2ecc71', '#e74c3c']  # Green for completed, red for failed
    plt.pie(success_counts.values, labels=success_counts.index, autopct='%1.1f%%', colors=colors)
    plt.title('Experiment Success Rate')
    
    # 2. Retention Rate Distribution
    plt.subplot(2, 3, 2)
    successful_df = df[df['status'] == 'completed']
    if len(successful_df) > 0:
        plt.hist(successful_df['retention_rate'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(successful_df['retention_rate'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {successful_df["retention_rate"].mean():.1f}%')
        plt.xlabel('Retention Rate (%)')
        plt.ylabel('Number of Datasets')
        plt.title('Distribution of Retention Rates')
        plt.legend()
    
    # 3. Sample Counts by Dataset
    plt.subplot(2, 3, 3)
    if len(successful_df) > 0:
        plt.barh(successful_df['dataset_name'], successful_df['retrieved_samples'], 
                alpha=0.6, label='Retrieved', color='lightblue')
        plt.barh(successful_df['dataset_name'], successful_df['filtered_samples'], 
                alpha=0.8, label='Filtered', color='darkblue')
        plt.xlabel('Number of Samples')
        plt.title('Sample Counts by Dataset')
        plt.legend()
        plt.xticks(rotation=45)
    
    # 4. Execution Time Analysis
    plt.subplot(2, 3, 4)
    if len(df) > 0:
        plt.bar(range(len(df)), df['execution_time'], color='orange', alpha=0.7)
        plt.xlabel('Dataset Index')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Execution Time by Dataset')
        # Add dataset names on x-axis
        plt.xticks(range(len(df)), [name[:8] + '...' if len(name) > 8 else name 
                  for name in df['dataset_name']], rotation=45)
    
    # 5. Retention Rate by Dataset
    plt.subplot(2, 3, 5)
    if len(successful_df) > 0:
        plt.bar(range(len(successful_df)), successful_df['retention_rate'], 
               color='green', alpha=0.7)
        plt.xlabel('Dataset Index')
        plt.ylabel('Retention Rate (%)')
        plt.title('Retention Rate by Dataset')
        plt.xticks(range(len(successful_df)), 
                  [name[:8] + '...' if len(name) > 8 else name 
                   for name in successful_df['dataset_name']], rotation=45)
    
    # 6. Quality Scores Heatmap
    plt.subplot(2, 3, 6)
    quality_cols = [col for col in df.columns if col.startswith('quality_')]
    if quality_cols and len(successful_df) > 0:
        # Create quality scores matrix
        quality_data = successful_df[quality_cols + ['dataset_name']].set_index('dataset_name')
        quality_data.columns = [col.replace('quality_', '') for col in quality_data.columns]
        
        # Create heatmap
        sns.heatmap(quality_data.T, annot=True, fmt='.2f', cmap='viridis', 
                   cbar_kws={'label': 'Quality Score'})
        plt.title('Quality Scores Heatmap')
        plt.xlabel('Datasets')
        plt.ylabel('Quality Metrics')
    else:
        plt.text(0.5, 0.5, 'No quality scores available', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Quality Scores (Not Available)')
    
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/experiment_overview.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed retention rate comparison
    if len(successful_df) > 0:
        plt.figure(figsize=(12, 8))
        
        # Sort by retention rate for better visualization
        sorted_df = successful_df.sort_values('retention_rate', ascending=True)
        
        y_pos = np.arange(len(sorted_df))
        bars = plt.barh(y_pos, sorted_df['retention_rate'], color='steelblue', alpha=0.8)
        
        # Color bars based on retention rate
        for i, bar in enumerate(bars):
            rate = sorted_df.iloc[i]['retention_rate']
            if rate >= 80:
                bar.set_color('green')
            elif rate >= 60:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.yticks(y_pos, sorted_df['dataset_name'])
        plt.xlabel('Retention Rate (%)')
        plt.title('MAVERIC Quality Filtering: Retention Rates by Selected Dataset')
        plt.axvline(sorted_df['retention_rate'].mean(), color='black', linestyle='--', 
                   label=f'Average: {sorted_df["retention_rate"].mean():.1f}%')
        
        # Add value labels on bars
        for i, (_, row) in enumerate(sorted_df.iterrows()):
            plt.text(row['retention_rate'] + 1, i, f'{row["retention_rate"]:.1f}%', 
                    va='center', ha='left')
        
        plt.legend()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/retention_rates_detailed.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Visualizations saved to: {viz_dir}")

def generate_analysis_report(df: pd.DataFrame, stats: Dict, results: Dict, output_dir: str):
    """Generate comprehensive analysis report."""
    print("📝 Generating analysis report...")
    
    report_path = f"{output_dir}/maveric_analysis_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# MAVERIC ELEVATER Experiments Analysis Report\n\n")
        f.write(f"**Report Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        processed_count = stats['experiment_overview']['processed_datasets']
        f.write(f"This report analyzes the effectiveness of MAVERIC's quality-driven filtering ")
        f.write(f"approach across {processed_count} selected ELEVATER datasets.\n\n")
        
        # Add dataset selection information if available
        if 'selected_datasets' in stats['experiment_overview']:
            total_available = stats['experiment_overview']['total_available_datasets']
            selected_count = stats['experiment_overview']['selection_count']
            selection_rate = stats['experiment_overview']['selection_rate']
            
            f.write(f"**Dataset Selection:**\n")
            f.write(f"- **Available Datasets:** {total_available}\n")
            f.write(f"- **Selected for Processing:** {selected_count} ({selection_rate:.1f}%)\n")
            f.write(f"- **Selected Datasets:** {', '.join(stats['experiment_overview']['selected_datasets'])}\n\n")
        
        if 'data_statistics' in stats:
            f.write(f"**Key Findings:**\n")
            f.write(f"- **Overall Success Rate:** {stats['experiment_overview']['success_rate']:.1f}%\n")
            f.write(f"- **Average Retention Rate:** {stats['data_statistics']['overall_retention_rate']:.1f}%\n")
            f.write(f"- **Total Samples Processed:** {stats['data_statistics']['total_samples_retrieved']:,}\n")
            f.write(f"- **Samples After Filtering:** {stats['data_statistics']['total_samples_filtered']:,}\n\n")
        
        # Experiment Overview
        f.write("## Experiment Overview\n\n")
        exp_overview = stats['experiment_overview']
        f.write(f"- **Datasets Processed:** {exp_overview['processed_datasets']}\n")
        f.write(f"- **Successful Experiments:** {exp_overview['successful_experiments']}\n")
        f.write(f"- **Failed Experiments:** {exp_overview['failed_experiments']}\n")
        f.write(f"- **Success Rate:** {exp_overview['success_rate']:.1f}%\n")
        f.write(f"- **Total Execution Time:** {exp_overview['total_execution_time']:.1f} seconds\n")
        f.write(f"- **Average Execution Time:** {exp_overview['average_execution_time']:.1f} seconds per dataset\n\n")
        
        # Data Quality Results
        if 'data_statistics' in stats:
            f.write("## Data Quality Results\n\n")
            data_stats = stats['data_statistics']
            f.write(f"### Filtering Effectiveness\n")
            f.write(f"- **Total Samples Retrieved:** {data_stats['total_samples_retrieved']:,}\n")
            f.write(f"- **Total Samples After Filtering:** {data_stats['total_samples_filtered']:,}\n")
            f.write(f"- **Overall Data Reduction:** {100 - (data_stats['total_samples_filtered']/data_stats['total_samples_retrieved']*100):.1f}%\n\n")
            
            f.write(f"### Retention Rate Statistics\n")
            f.write(f"- **Mean Retention Rate:** {data_stats['overall_retention_rate']:.1f}% ± {data_stats['retention_rate_std']:.1f}%\n")
            f.write(f"- **Minimum Retention Rate:** {data_stats['min_retention_rate']:.1f}%\n")
            f.write(f"- **Maximum Retention Rate:** {data_stats['max_retention_rate']:.1f}%\n\n")
        
        # Quality Metrics Analysis
        if 'quality_statistics' in stats:
            f.write("## Quality Metrics Analysis\n\n")
            for metric, metric_stats in stats['quality_statistics'].items():
                f.write(f"### {metric.replace('_', ' ').title()}\n")
                f.write(f"- **Mean:** {metric_stats['mean']:.3f}\n")
                f.write(f"- **Standard Deviation:** {metric_stats['std']:.3f}\n")
                f.write(f"- **Range:** {metric_stats['min']:.3f} - {metric_stats['max']:.3f}\n\n")
        
        # Dataset-Specific Results
        f.write("## Dataset-Specific Results\n\n")
        successful_df = df[df['status'] == 'completed']
        
        if len(successful_df) > 0:
            # Sort by retention rate for analysis
            sorted_df = successful_df.sort_values('retention_rate', ascending=False)
            
            f.write("### Top Performing Datasets (Highest Retention Rates)\n")
            f.write("| Dataset | Retention Rate | Retrieved | Filtered | Execution Time |\n")
            f.write("|---------|----------------|-----------|----------|----------------|\n")
            
            for _, row in sorted_df.head(5).iterrows():
                f.write(f"| {row['dataset_name']} | {row['retention_rate']:.1f}% | ")
                f.write(f"{row['retrieved_samples']:,} | {row['filtered_samples']:,} | ")
                f.write(f"{row['execution_time']:.1f}s |\n")
            f.write("\n")
            
            f.write("### Datasets with Aggressive Filtering (Lowest Retention Rates)\n")
            f.write("| Dataset | Retention Rate | Retrieved | Filtered | Execution Time |\n")
            f.write("|---------|----------------|-----------|----------|----------------|\n")
            
            for _, row in sorted_df.tail(5).iterrows():
                f.write(f"| {row['dataset_name']} | {row['retention_rate']:.1f}% | ")
                f.write(f"{row['retrieved_samples']:,} | {row['filtered_samples']:,} | ")
                f.write(f"{row['execution_time']:.1f}s |\n")
            f.write("\n")
        
        # Failed Experiments
        failed_df = df[df['status'] == 'failed']
        if len(failed_df) > 0:
            f.write("### Failed Experiments\n")
            f.write("| Dataset | Error |\n")
            f.write("|---------|-------|\n")
            for _, row in failed_df.iterrows():
                error_msg = row.get('error', 'Unknown error')[:100]
                f.write(f"| {row['dataset_name']} | {error_msg}... |\n")
            f.write("\n")
        
        # Conclusions and Recommendations
        f.write("## Conclusions and Recommendations\n\n")
        
        if 'data_statistics' in stats:
            avg_retention = stats['data_statistics']['overall_retention_rate']
            
            if avg_retention > 75:
                f.write("### ✅ High Quality Filtering Performance\n")
                f.write(f"MAVERIC demonstrates conservative filtering with {avg_retention:.1f}% average retention rate, ")
                f.write("indicating that most data meets quality standards.\n\n")
                f.write("**Recommendations:**\n")
                f.write("- Consider slightly stricter thresholds to improve quality further\n")
                f.write("- Analyze which quality metrics are most effective for different dataset types\n\n")
                
            elif avg_retention > 50:
                f.write("### ⚖️ Balanced Filtering Performance\n") 
                f.write(f"MAVERIC shows balanced filtering with {avg_retention:.1f}% average retention rate, ")
                f.write("suggesting effective quality-based data curation.\n\n")
                f.write("**Recommendations:**\n")
                f.write("- Current thresholds appear well-calibrated\n")
                f.write("- Monitor model performance with filtered datasets\n")
                f.write("- Consider dataset-specific threshold tuning\n\n")
                
            else:
                f.write("### ⚠️ Aggressive Filtering Performance\n")
                f.write(f"MAVERIC applies aggressive filtering with {avg_retention:.1f}% average retention rate, ")
                f.write("indicating strict quality requirements.\n\n")
                f.write("**Recommendations:**\n")
                f.write("- Review quality thresholds - may be too strict\n")
                f.write("- Analyze filtered-out samples to understand quality issues\n")
                f.write("- Consider relaxing some thresholds while maintaining quality\n\n")
        
        f.write("### Next Steps\n")
        f.write("1. **Model Training:** Train vision-language models on filtered datasets\n")
        f.write("2. **Performance Comparison:** Compare model performance with and without MAVERIC filtering\n")
        f.write("3. **Threshold Optimization:** Fine-tune quality thresholds based on downstream task performance\n")
        f.write("4. **Quality Metric Analysis:** Identify which quality metrics are most predictive of model performance\n\n")
        
        f.write("---\n")
        f.write("*Report generated by MAVERIC Analysis Suite*\n")
    
    print(f"✅ Analysis report saved to: {report_path}")

def display_dataset_selection_info(experiment_info: Dict, config: Dict):
    """Display information about dataset selection."""
    if 'selected_datasets' in experiment_info:
        selected_datasets = experiment_info['selected_datasets']
        available_datasets = config.get('elevater', {}).get('datasets', [])
        
        print(f"\n📊 Dataset Selection Overview:")
        print(f"   • Available datasets: {len(available_datasets)}")
        print(f"   • Selected datasets: {len(selected_datasets)}")
        print(f"   • Selection rate: {len(selected_datasets) / len(available_datasets) * 100:.1f}%")
        
        if len(selected_datasets) < len(available_datasets):
            not_selected = [d for d in available_datasets if d not in selected_datasets]
            print(f"\n   Selected: {', '.join(selected_datasets)}")
            print(f"   Not selected: {', '.join(not_selected)}")
        else:
            print(f"   All available datasets were selected for processing")

def main():
    """Main analysis function."""
    print("🚀 Starting MAVERIC Results Analysis...")
    print("=" * 60)
    
    # Load configuration
    config = load_experiment_config()
    if not config:
        print("❌ Failed to load configuration. Exiting.")
        return False
    
    results_dir = config['results_dir']
    
    # Load experiment results
    results = load_experiment_results(results_dir)
    if not results:
        print("❌ Failed to load experiment results. Exiting.")
        return False
    
    # Create results DataFrame
    df = create_results_dataframe(results)
    
    # Extract experiment info for enhanced statistics
    experiment_info = results['summary'].get('experiment_info', {})
    
    # Display dataset selection information
    display_dataset_selection_info(experiment_info, config)
    
    # Generate statistics
    stats = generate_summary_statistics(df, config, experiment_info)
    
    # Create visualizations
    create_visualizations(df, results, results_dir)
    
    # Generate comprehensive report
    generate_analysis_report(df, stats, results, results_dir)
    
    # Print summary to console
    print("\n" + "="*60)
    print("🎉 MAVERIC Results Analysis Completed!")
    print("="*60)
    print("📊 Key Results:")
    exp_overview = stats['experiment_overview']
    print(f"   • Success Rate: {exp_overview['success_rate']:.1f}%")
    print(f"   • Processed Datasets: {exp_overview['processed_datasets']}")
    if 'selection_count' in exp_overview:
        print(f"   • Selected from: {exp_overview['total_available_datasets']} available datasets")
    print(f"   • Execution Time: {exp_overview['total_execution_time']:.1f}s")
    
    if 'data_statistics' in stats:
        data_stats = stats['data_statistics']
        print(f"   • Average Retention Rate: {data_stats['overall_retention_rate']:.1f}%")
        print(f"   • Total Samples Processed: {data_stats['total_samples_retrieved']:,}")
        print(f"   • Samples After Filtering: {data_stats['total_samples_filtered']:,}")
    
    print(f"\n📁 Analysis outputs saved to: {results_dir}")
    print("   • Analysis report: maveric_analysis_report.md")
    print("   • Visualizations: analysis_visualizations/")
    print("="*60)
    
    return True

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)