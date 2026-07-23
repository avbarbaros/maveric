"""
Multi-File Consistency Analysis

Analyze consistency scores across multiple raw retrieval files and aggregate results.
Useful for comprehensive analysis of large datasets split into rotation files.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from consistency_analysis import analyze_curated_data, generate_analysis_report


def analyze_multiple_files(data_dir: str,
                           file_pattern: str = "*_raw_maveric_dataset*.json",
                           max_files: int = None,
                           normalization: str = "none",
                           B: int = 1000,
                           seed: int = 0) -> Dict:
    """
    Analyze consistency across multiple raw retrieval files.

    Args:
        data_dir: Directory containing raw JSON files
        file_pattern: Glob pattern for file selection (default: all raw files)
        max_files: Maximum number of files to analyze (None = all files)
        normalization: Normalization method ("none" or "zscore")
        B: Number of permutation iterations per file
        seed: Random seed

    Returns:
        Dictionary with aggregated results per class

    Example:
        >>> results = analyze_multiple_files(
        ...     "../../maveric_experiments/cifar10/raw/",
        ...     max_files=10,  # Analyze first 10 files
        ...     normalization="none"
        ... )
    """
    data_path = Path(data_dir)
    files = sorted(data_path.glob(file_pattern))

    if not files:
        raise ValueError(f"No files found matching pattern: {file_pattern} in {data_dir}")

    if max_files:
        files = files[:max_files]

    print(f"📂 Found {len(files)} files to analyze")
    print(f"   Pattern: {file_pattern}")
    print(f"   Analyzing: {len(files)} files")
    print()

    # Aggregate results per class
    class_results = {}  # class_name -> list of rho_observed values

    for i, file_path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] Analyzing {file_path.name}...")

        try:
            # Analyze this file
            file_results = analyze_curated_data(
                str(file_path),
                normalization=normalization,
                B=B,
                seed=seed + i  # Different seed per file
            )

            # Aggregate results per class
            for cls, result in file_results.items():
                if cls not in class_results:
                    class_results[cls] = {
                        'rho_observed': [],
                        'rho_null_mean': [],
                        'p_value': [],
                        'n_samples': []
                    }

                class_results[cls]['rho_observed'].append(result['rho_observed'])
                class_results[cls]['rho_null_mean'].append(result['rho_null_mean'])
                class_results[cls]['p_value'].append(result['p_value'])
                class_results[cls]['n_samples'].append(result['n_samples'])

        except Exception as e:
            print(f"   ⚠️  Error analyzing {file_path.name}: {e}")
            continue

    print()
    print("=" * 80)
    print("AGGREGATED RESULTS ACROSS FILES")
    print("=" * 80)

    # Compute aggregated statistics
    aggregated = {}
    for cls, values in class_results.items():
        aggregated[cls] = {
            'rho_observed': float(np.mean(values['rho_observed'])),
            'rho_observed_std': float(np.std(values['rho_observed'])),
            'rho_observed_min': float(np.min(values['rho_observed'])),
            'rho_observed_max': float(np.max(values['rho_observed'])),
            'p_value_mean': float(np.mean(values['p_value'])),
            'p_value_median': float(np.median(values['p_value'])),
            'n_samples_total': int(np.sum(values['n_samples'])),
            'n_files': len(values['rho_observed']),
            'significant': np.mean(values['p_value']) < 0.05
        }

    return aggregated


def print_aggregated_report(results: Dict, output_path: str = None) -> str:
    """Generate formatted report for multi-file analysis."""

    lines = []
    lines.append("=" * 80)
    lines.append("MULTI-FILE CONSISTENCY ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")

    total_classes = len(results)
    significant_classes = sum(1 for r in results.values() if r['significant'])

    lines.append(f"Total classes analyzed: {total_classes}")
    lines.append(f"Classes with significant correlation (p < 0.05): {significant_classes} ({100*significant_classes/total_classes:.1f}%)")
    lines.append("")
    lines.append("-" * 80)
    lines.append("")

    lines.append("PER-CLASS AGGREGATED RESULTS:")
    lines.append("")
    lines.append(f"{'Class':<20} {'Files':>6} {'N':>8} {'ρ_mean':>8} {'ρ_std':>8} {'p_mean':>8} {'Sig':>5}")
    lines.append("-" * 80)

    for cls, res in sorted(results.items()):
        sig_mark = "***" if res['p_value_mean'] < 0.001 else "**" if res['p_value_mean'] < 0.01 else "*" if res['p_value_mean'] < 0.05 else ""
        lines.append(
            f"{cls:<20} {res['n_files']:>6} {res['n_samples_total']:>8} "
            f"{res['rho_observed']:>8.3f} {res['rho_observed_std']:>8.3f} "
            f"{res['p_value_mean']:>8.3f} {sig_mark:>5}"
        )

    lines.append("-" * 80)
    lines.append("")
    lines.append("Significance codes: *** p<0.001, ** p<0.01, * p<0.05")
    lines.append("")
    lines.append("ρ_mean: Average observed correlation across files")
    lines.append("ρ_std: Standard deviation of correlation across files")
    lines.append("p_mean: Average p-value across files")
    lines.append("")

    # Interpretation
    lines.append("=" * 80)
    lines.append("INTERPRETATION:")
    lines.append("=" * 80)
    if significant_classes / total_classes > 0.8:
        lines.append("✅ STRONG EVIDENCE for real multimodal quality structure")
        lines.append("   Most classes show significant correlation beyond mechanical effects.")
        lines.append("   Current consistency scores are valid without normalization.")
    elif significant_classes / total_classes > 0.5:
        lines.append("⚠️  MIXED EVIDENCE - correlation is partially mechanical")
        lines.append("   Consider z-score normalization to reduce mechanical correlation.")
    else:
        lines.append("⚠️  WEAK EVIDENCE - correlation appears largely mechanical")
        lines.append("   Z-score normalization is STRONGLY RECOMMENDED.")

    report = "\n".join(lines)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"📄 Multi-file report saved to: {output_path}")

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze consistency across multiple raw retrieval files"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing raw JSON files"
    )
    parser.add_argument(
        "--pattern",
        default="*_raw_maveric_dataset*.json",
        help="File pattern to match (default: *_raw_maveric_dataset*.json)"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Maximum number of files to analyze (default: all)"
    )
    parser.add_argument(
        "--normalization",
        default="none",
        choices=["none", "zscore"],
        help="Normalization method"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Permutation iterations per file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed base"
    )
    parser.add_argument(
        "--output",
        help="Output report path"
    )

    args = parser.parse_args()

    # Run multi-file analysis
    results = analyze_multiple_files(
        args.data_dir,
        file_pattern=args.pattern,
        max_files=args.max_files,
        normalization=args.normalization,
        B=args.iterations,
        seed=args.seed
    )

    # Print report
    report = print_aggregated_report(results, output_path=args.output)
    print(report)
