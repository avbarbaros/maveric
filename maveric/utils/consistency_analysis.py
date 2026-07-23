"""
Consistency Score Analysis Tools

This module provides tools for analyzing consistency scores in response to reviewer
concerns about mechanical correlation and scale normalization.

Key features:
1. Z-score normalization of similarity metrics before consistency calculation
2. Null-model permutation tests to assess mechanical vs. real correlation
3. Per-class analysis and visualization

Reference: Reviewer comment on scale normalization and null-model tests.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json


def compute_consistency_score(metrics: np.ndarray,
                              normalization: str = "none",
                              eps: float = 1e-8) -> np.ndarray:
    """
    Compute consistency scores with optional normalization.

    Args:
        metrics: Array of shape (N, 4) containing [img2img, txt2txt, img2txt, txt2img]
                 for ONE class
        normalization: "none" (raw scores) or "zscore" (per-metric z-score normalization)
        eps: Small constant to avoid division by zero

    Returns:
        Array of consistency scores (N,) where consistency = 1 - std(normalized_metrics)

    Note:
        - Weighted average (q_avg) should be computed on RAW metrics
        - Consistency uses normalized metrics when zscore is enabled
        - Per-class normalization addresses scale differences across metrics
    """
    if normalization == "zscore":
        # Per-class, per-metric z-score normalization
        mu = metrics.mean(axis=0)  # Shape: (4,) - mean per metric
        sd = metrics.std(axis=0)   # Shape: (4,) - std per metric
        normalized = (metrics - mu) / (sd + eps)
        return 1.0 - normalized.std(axis=1)  # Shape: (N,)
    else:
        # Raw consistency (default)
        return 1.0 - metrics.std(axis=1)


def null_test_correlation(metrics: np.ndarray,
                          B: int = 1000,
                          seed: int = 0,
                          normalization: str = "none") -> Dict:
    """
    Permutation test for correlation between weighted_score and consistency.

    Tests whether the observed correlation reflects real multimodal quality structure
    or is a mechanical artifact of the scoring formula.

    Args:
        metrics: Array of shape (N, 4) containing [img2img, txt2txt, img2txt, txt2img]
        B: Number of bootstrap iterations (default: 1000)
        seed: Random seed for reproducibility
        normalization: Normalization method to apply before computing consistency

    Returns:
        Dictionary containing:
            - rho_observed: Observed correlation between q_avg and q_cons
            - rho_null_mean: Mean of null distribution
            - rho_null_std: Std of null distribution
            - ci_95: 95% confidence interval [lower, upper]
            - p_value: Two-tailed p-value
            - significant: Whether correlation is significant (p < 0.05)

    Example:
        >>> metrics = np.random.rand(1000, 4)  # Simulated data
        >>> result = null_test_correlation(metrics, B=1000)
        >>> print(f"Observed ρ: {result['rho_observed']:.3f}")
        >>> print(f"p-value: {result['p_value']:.3f}")
    """
    rng = np.random.default_rng(seed)

    # Compute observed statistics
    q_avg = metrics.mean(axis=1)  # Weighted average on RAW metrics
    q_cons = compute_consistency_score(metrics, normalization=normalization)
    rho_obs = np.corrcoef(q_avg, q_cons)[0, 1]

    # Null distribution via permutation
    rho_null = np.empty(B)
    for b in range(B):
        # Permute each metric column independently
        perm = np.column_stack([rng.permutation(metrics[:, j]) for j in range(4)])

        # Compute statistics on permuted data
        perm_avg = perm.mean(axis=1)
        perm_cons = compute_consistency_score(perm, normalization=normalization)
        rho_null[b] = np.corrcoef(perm_avg, perm_cons)[0, 1]

    # Compute p-value (two-tailed)
    p_value = np.mean(np.abs(rho_null) >= np.abs(rho_obs))

    # 95% confidence interval
    ci_95 = np.quantile(rho_null, [0.025, 0.975])

    return {
        'rho_observed': float(rho_obs),
        'rho_null_mean': float(rho_null.mean()),
        'rho_null_std': float(rho_null.std()),
        'ci_95': [float(ci_95[0]), float(ci_95[1])],
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'null_distribution': rho_null.tolist()  # For visualization
    }


def analyze_curated_data(data_path: str,
                        class_column: str = 'label',
                        normalization: str = "none",
                        B: int = 1000,
                        seed: int = 0) -> Dict[str, Dict]:
    """
    Analyze consistency scores and correlations from RAW retrieval dataset.

    IMPORTANT: This analysis requires RAW retrieval data (not curated data).
    Raw data contains individual similarity metrics (img2img, txt2txt, img2txt, txt2img)
    for all classes, while curated data only has aggregated scores.

    Data format expected:
        - Raw data: Class_{class_name}_{metric} columns for all classes
        - Example: Class_airplane_img2img, Class_airplane_txt2txt, etc.

    Args:
        data_path: Path to RAW retrieval JSON/pickle file
        class_column: Ignored (kept for API compatibility)
        normalization: Normalization method ("none" or "zscore")
        B: Number of permutation iterations
        seed: Random seed

    Returns:
        Dictionary mapping class_name -> null_test_results

    Example:
        >>> # Use raw retrieval data, not curated data
        >>> results = analyze_curated_data(
        ...     "results/cifar10/raw/cifar10_raw_maveric_dataset1.json",
        ...     normalization="none",
        ...     B=1000
        ... )
        >>> for cls, res in results.items():
        ...     print(f"{cls}: ρ={res['rho_observed']:.3f}, p={res['p_value']:.3f}")
    """
    # Load data
    if data_path.endswith('.json'):
        with open(data_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        df = pd.read_pickle(data_path)

    # Extract classes from column names (Class_{class_name}_img2img)
    class_cols = [col for col in df.columns if col.startswith('Class_') and col.endswith('_img2img')]

    if not class_cols:
        raise ValueError(
            "❌ No class columns found! This analysis requires RAW retrieval data.\n"
            "   Expected columns: Class_{class_name}_img2img, Class_{class_name}_txt2txt, etc.\n"
            "   Curated data (with 'label' and 'consistency' columns only) cannot be analyzed.\n"
            "   Please use raw retrieval data from: results/{dataset}/raw/*.json"
        )

    classes = [col.replace('Class_', '').replace('_img2img', '') for col in class_cols]
    print(f"ℹ️  Found {len(classes)} classes in raw data")
    print(f"   Classes: {', '.join(classes[:5])}{'...' if len(classes) > 5 else ''}")
    print()

    results = {}
    for cls in classes:
        # Raw data: use all rows, extract this class's metric columns
        try:
            metrics = np.column_stack([
                df[f'Class_{cls}_img2img'].values,
                df[f'Class_{cls}_txt2txt'].values,
                df[f'Class_{cls}_img2txt'].values,
                df[f'Class_{cls}_txt2img'].values
            ])
        except KeyError as e:
            print(f"⚠️  Warning: Missing similarity columns for class '{cls}': {e}")
            continue

        # Run null test
        result = null_test_correlation(
            metrics,
            B=B,
            seed=seed,
            normalization=normalization
        )

        # Add sample count
        result['n_samples'] = len(metrics)

        results[cls] = result

    return results


def apply_zscore_normalization_to_data(data_path: str,
                                       output_path: str,
                                       class_column: str = 'label') -> None:
    """
    Apply z-score normalization to consistency scores in RAW data and save.

    IMPORTANT: This function requires RAW retrieval data (not curated data).
    It recomputes consistency scores using z-score normalized metrics.

    Args:
        data_path: Path to RAW retrieval JSON/pickle file
        output_path: Path to save updated JSON file
        class_column: Ignored (kept for API compatibility)

    Example:
        >>> # Use raw retrieval data
        >>> apply_zscore_normalization_to_data(
        ...     "results/cifar10/raw/cifar10_raw_maveric_dataset1.json",
        ...     "results/cifar10/raw/cifar10_raw_maveric_dataset1_zscore.json"
        ... )
    """
    # Load data
    if data_path.endswith('.json'):
        with open(data_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        df = pd.read_pickle(data_path)

    # Extract classes from column names
    class_cols = [col for col in df.columns if col.startswith('Class_') and col.endswith('_img2img')]

    if not class_cols:
        raise ValueError(
            "❌ No class columns found! This function requires RAW retrieval data.\n"
            "   Expected columns: Class_{class_name}_img2img, etc.\n"
            "   Please use raw retrieval data from: results/{dataset}/raw/*.json"
        )

    classes = [col.replace('Class_', '').replace('_img2img', '') for col in class_cols]
    print(f"ℹ️  Processing {len(classes)} classes in raw data")

    # Process each class
    for cls in classes:
        try:
            # Extract metrics for this class (from all rows)
            metrics = np.column_stack([
                df[f'Class_{cls}_img2img'].values,
                df[f'Class_{cls}_txt2txt'].values,
                df[f'Class_{cls}_img2txt'].values,
                df[f'Class_{cls}_txt2img'].values
            ])

            # Compute z-score normalized consistency
            consistency_zscore = compute_consistency_score(metrics, normalization="zscore")

            # Update the consistency column for this class
            df[f'Class_{cls}_consistency'] = consistency_zscore

            print(f"   ✓ {cls}: Updated consistency with z-score normalization")

        except KeyError as e:
            print(f"   ⚠️  Warning: Missing columns for class '{cls}': {e}")
            continue

    # Save updated data
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if output_path.endswith('.json'):
        df.to_json(output_path, orient='records', indent=2)
    else:
        df.to_pickle(output_path)

    print()
    print(f"✅ Z-score normalized data saved to: {output_path}")
    print(f"   Original consistency columns preserved, updated with z-score values")


def generate_analysis_report(results: Dict[str, Dict],
                            output_path: Optional[str] = None) -> str:
    """
    Generate a formatted analysis report from null-test results.

    Args:
        results: Output from analyze_curated_data()
        output_path: Optional path to save report as text file

    Returns:
        Formatted report string

    Example:
        >>> results = analyze_curated_data("data.json")
        >>> report = generate_analysis_report(results, "analysis_report.txt")
        >>> print(report)
    """
    lines = []
    lines.append("=" * 80)
    lines.append("CONSISTENCY SCORE NULL-MODEL ANALYSIS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("This analysis tests whether the correlation between weighted_class_score")
    lines.append("and consistency reflects real multimodal quality structure or is a")
    lines.append("mechanical artifact of the scoring formula.")
    lines.append("")
    lines.append("Null hypothesis: Correlation arises purely from formula mechanics")
    lines.append("Alternative: Correlation reflects real quality structure")
    lines.append("")
    lines.append("=" * 80)
    lines.append("")

    # Summary statistics
    total_classes = len(results)
    significant_classes = sum(1 for r in results.values() if r['significant'])

    lines.append(f"Total classes analyzed: {total_classes}")
    lines.append(f"Classes with significant correlation (p < 0.05): {significant_classes} ({100*significant_classes/total_classes:.1f}%)")
    lines.append("")
    lines.append("-" * 80)
    lines.append("")

    # Per-class results
    lines.append("PER-CLASS RESULTS:")
    lines.append("")
    lines.append(f"{'Class':<20} {'N':>8} {'ρ_obs':>8} {'ρ_null':>8} {'p-value':>8} {'Sig':>5}")
    lines.append("-" * 80)

    for cls, res in sorted(results.items()):
        sig_mark = "***" if res['p_value'] < 0.001 else "**" if res['p_value'] < 0.01 else "*" if res['p_value'] < 0.05 else ""
        lines.append(
            f"{cls:<20} {res['n_samples']:>8} {res['rho_observed']:>8.3f} "
            f"{res['rho_null_mean']:>8.3f} {res['p_value']:>8.3f} {sig_mark:>5}"
        )

    lines.append("-" * 80)
    lines.append("")
    lines.append("Significance codes: *** p<0.001, ** p<0.01, * p<0.05")
    lines.append("")

    # Interpretation
    lines.append("=" * 80)
    lines.append("INTERPRETATION:")
    lines.append("=" * 80)
    if significant_classes / total_classes > 0.8:
        lines.append("✅ STRONG EVIDENCE for real multimodal quality structure")
        lines.append("   Most classes show significant correlation beyond mechanical effects.")
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
        print(f"📄 Report saved to: {output_path}")

    return report


if __name__ == "__main__":
    """
    Example usage as standalone script.

    Run from command line:
        python -m maveric.utils.consistency_analysis \\
            --data results/cifar10/curated/training_data.json \\
            --normalization zscore \\
            --output analysis_report.txt
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze consistency scores with null-model permutation test"
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to curated JSON data file"
    )
    parser.add_argument(
        "--normalization",
        default="none",
        choices=["none", "zscore"],
        help="Normalization method for consistency calculation"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of permutation iterations (default: 1000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output",
        help="Path to save analysis report (optional)"
    )
    parser.add_argument(
        "--apply-zscore",
        help="Apply z-score normalization and save to this path (optional)"
    )

    args = parser.parse_args()

    print("🔬 Running consistency score null-model analysis...")
    print(f"   Data: {args.data}")
    print(f"   Normalization: {args.normalization}")
    print(f"   Iterations: {args.iterations}")
    print("")

    # Run analysis
    results = analyze_curated_data(
        args.data,
        normalization=args.normalization,
        B=args.iterations,
        seed=args.seed
    )

    # Generate report
    report = generate_analysis_report(
        results,
        output_path=args.output
    )

    print(report)

    # Apply z-score normalization if requested
    if args.apply_zscore:
        print("")
        print("📊 Applying z-score normalization to data...")
        apply_zscore_normalization_to_data(
            args.data,
            args.apply_zscore
        )
