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


def apply_zscore_normalization_to_directory(input_dir: str,
                                            output_dir: str,
                                            pattern: str = "*.json") -> None:
    """
    Apply per-class z-score normalization across ALL JSON files in a directory,
    treating them as one combined population, and write normalized files to
    output_dir.

    IMPORTANT: Retrieval output is typically split across multiple rotation
    files (e.g. dataset_001.json, dataset_002.json, ...), each holding only a
    fraction of the samples for any given class. Normalizing each file on its
    own (as apply_zscore_normalization_to_data() does) computes mean/std from
    an incomplete, biased subset of that class's samples. This function instead
    makes two passes over the directory:
      1. Scan every file to accumulate each class's raw similarity metrics
         (img2img, txt2txt, img2txt, txt2img) across the FULL population.
      2. Re-load each file and recompute its Class_{cls}_consistency column
         using the global per-class mean/std from step 1, writing the result
         to output_dir under the same filename.

    Args:
        input_dir: Directory containing raw retrieval JSON files
        output_dir: Directory to write z-score normalized JSON files (mirrors
            input_dir's filenames)
        pattern: Glob pattern to select files within input_dir (default: "*.json")

    Example:
        >>> apply_zscore_normalization_to_directory(
        ...     "results/hateful_memes/raw",
        ...     "results/hateful_memes/raw_zscore"
        ... )
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    json_files = sorted(input_path.glob(pattern))
    if not json_files:
        raise ValueError(f"❌ No JSON files found in {input_dir} matching pattern '{pattern}'")

    print(f"ℹ️  Found {len(json_files)} files in {input_dir}")

    # ---- Pass 1: accumulate per-class metrics across ALL files ----
    class_metrics_chunks: Dict[str, List[np.ndarray]] = {}

    for file_path in json_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)

        class_cols = [col for col in df.columns if col.startswith('Class_') and col.endswith('_img2img')]
        if not class_cols:
            print(f"   ⚠️  Skipping {file_path.name}: no Class_*_img2img columns found")
            continue

        file_classes = [col.replace('Class_', '').replace('_img2img', '') for col in class_cols]

        for cls in file_classes:
            try:
                metrics = np.column_stack([
                    df[f'Class_{cls}_img2img'].values,
                    df[f'Class_{cls}_txt2txt'].values,
                    df[f'Class_{cls}_img2txt'].values,
                    df[f'Class_{cls}_txt2img'].values
                ])
            except KeyError as e:
                print(f"   ⚠️  Warning: Missing columns for class '{cls}' in {file_path.name}: {e}")
                continue
            class_metrics_chunks.setdefault(cls, []).append(metrics)

    if not class_metrics_chunks:
        raise ValueError(
            "❌ No class columns found in any file! This function requires RAW retrieval data.\n"
            "   Expected columns: Class_{class_name}_img2img, etc.\n"
            f"   Checked directory: {input_dir}"
        )

    # Compute global per-class, per-metric mean/std across the full population
    eps = 1e-8
    class_stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for cls, chunks in class_metrics_chunks.items():
        all_metrics = np.concatenate(chunks, axis=0)  # (N_total, 4)
        mu = all_metrics.mean(axis=0)
        sd = all_metrics.std(axis=0)
        class_stats[cls] = (mu, sd)
        print(f"   ✓ {cls}: global stats computed from {len(all_metrics)} samples across {len(json_files)} files")

    # ---- Pass 2: re-load each file, apply global stats, write to output_dir ----
    output_path.mkdir(parents=True, exist_ok=True)

    for file_path in json_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)

        class_cols = [col for col in df.columns if col.startswith('Class_') and col.endswith('_img2img')]
        file_classes = [col.replace('Class_', '').replace('_img2img', '') for col in class_cols]

        for cls in file_classes:
            if cls not in class_stats:
                continue
            try:
                metrics = np.column_stack([
                    df[f'Class_{cls}_img2img'].values,
                    df[f'Class_{cls}_txt2txt'].values,
                    df[f'Class_{cls}_img2txt'].values,
                    df[f'Class_{cls}_txt2img'].values
                ])
            except KeyError:
                continue

            mu, sd = class_stats[cls]
            normalized = (metrics - mu) / (sd + eps)
            df[f'Class_{cls}_consistency'] = 1.0 - normalized.std(axis=1)

        out_file_path = output_path / file_path.name
        df.to_json(out_file_path, orient='records', indent=2)
        print(f"   → {file_path.name} → {out_file_path}")

    print()
    print(f"✅ Z-score normalized data (global per-class stats across {len(json_files)} files) saved to: {output_dir}")


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
        # Single-file analysis + optional z-score export
        python -m maveric.utils.consistency_analysis \\
            --data results/cifar10/curated/training_data.json \\
            --normalization zscore \\
            --output analysis_report.txt

        # Batch directory z-score normalization (multiple rotation files treated
        # as one combined population, output mirrors input filenames)
        python -m maveric.utils.consistency_analysis \\
            --input-dir results/hateful_memes/raw \\
            --output-dir results/hateful_memes/raw_zscore
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze consistency scores with null-model permutation test"
    )
    parser.add_argument(
        "--data",
        help="Path to curated JSON data file (single-file mode)"
    )
    parser.add_argument(
        "--input-dir",
        help="Directory containing raw retrieval JSON files to z-score normalize "
             "as one combined population (batch directory mode)"
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to write z-score normalized JSON files (required with --input-dir)"
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

    if args.input_dir:
        # Batch directory mode: z-score normalize every JSON file in a
        # directory using per-class stats pooled across all of them.
        if not args.output_dir:
            parser.error("--output-dir is required when using --input-dir")

        print("📊 Applying z-score normalization to directory...")
        print(f"   Input directory: {args.input_dir}")
        print(f"   Output directory: {args.output_dir}")
        print("")

        apply_zscore_normalization_to_directory(
            args.input_dir,
            args.output_dir
        )
    else:
        # Single-file mode: run null-model analysis (and optionally export a
        # z-score normalized copy of that one file).
        if not args.data:
            parser.error("--data is required (or use --input-dir/--output-dir for batch mode)")

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
