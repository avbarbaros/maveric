"""
05_unified_evaluation.py - Evaluate a unified model checkpoint per ELEVATER dataset.

Compares:
  - Baseline zero-shot CLIP (same backbone, no fine-tuning)
  - Customized unified model (loaded from checkpoint)

Both are evaluated on every dataset found in the checkpoint's metadata.

Usage:
    python experiments/05_unified_evaluation.py \
        --checkpoint  results/unified_training/models/unified_model_best.pth \
        --config      experiments/maveric_config.yaml

    # Evaluate only specific datasets:
    python experiments/05_unified_evaluation.py \
        --checkpoint  results/unified_training/models/unified_model_best.pth \
        --config      experiments/maveric_config.yaml \
        --datasets cifar10 cifar100 caltech101

    # Skip baseline evaluation (customized model only):
    python experiments/05_unified_evaluation.py \
        --checkpoint  results/unified_training/models/unified_model_best.pth \
        --config      experiments/maveric_config.yaml \
        --no-baseline

    # Save detailed per-class results:
    python experiments/05_unified_evaluation.py \
        --checkpoint  results/unified_training/models/unified_model_best.pth \
        --config      experiments/maveric_config.yaml \
        --detailed
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

# Make sure the repo root is on the path when running from experiments/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maveric.customization.evaluation import Evaluator
from maveric.customization.model_customizer import CustomizedCLIP, ModelCustomizer
from maveric.datasets.elevater_datasets import ELEVATERDataset


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate a unified CLIP checkpoint on every ELEVATER dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--checkpoint", "-c",
        required=True,
        help="Path to unified model checkpoint (.pth)"
    )
    parser.add_argument(
        "--config",
        default="experiments/maveric_config.yaml",
        help="Path to MAVERIC config YAML (default: experiments/maveric_config.yaml)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Evaluate only these datasets (default: all in checkpoint metadata)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for result JSON files (default: same directory as checkpoint)"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Also save per-class accuracy breakdown"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for evaluation (default: from config)"
    )
    parser.add_argument(
        "--no-templates",
        action="store_true",
        help="Disable REACT-style template ensembling (use single prompt)"
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip baseline zero-shot CLIP evaluation (evaluate customized model only)"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_unified_checkpoint(checkpoint_path: str, device: str):
    """
    Load a unified model checkpoint saved by run_unified_training().

    Returns
    -------
    checkpoint : dict  – raw checkpoint dict
    clip_model_name : str – e.g. 'ViT-B/32'
    dataset_metadata : dict – dataset_name → {class_names, ...}
    """
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    clip_model_name = checkpoint.get("clip_model", "ViT-B/32")
    dataset_metadata = checkpoint.get("dataset_metadata", {})

    if not dataset_metadata:
        print("⚠️  Checkpoint contains no dataset_metadata. "
              "Using all ELEVATER datasets as fallback.")

    print(f"   CLIP backbone : {clip_model_name}")
    print(f"   Datasets in checkpoint: {list(dataset_metadata.keys()) or '(none – using ELEVATER_DATASETS)'}")
    return checkpoint, clip_model_name, dataset_metadata


def build_baseline_model(clip_model_name: str, device: str) -> tuple:
    """Load a fresh (un-fine-tuned) CLIP model as baseline."""
    customizer = ModelCustomizer(
        base_model_name=clip_model_name,
        device=device,
        checkpoint_dir=None,
        cache_base_dir=None
    )
    baseline = CustomizedCLIP(
        customizer.model,
        customizer.processor,
        regularize=False
    ).to(device)
    return baseline, customizer.processor


def build_customized_model(checkpoint: dict, clip_model_name: str,
                           device: str, processor) -> CustomizedCLIP:
    """Restore the fine-tuned model from a checkpoint's state_dict."""
    # We need a fresh base CLIPModel to wrap inside CustomizedCLIP
    from transformers import CLIPModel

    model_mapping = {
        "ViT-B/32":        "openai/clip-vit-base-patch32",
        "ViT-B/16":        "openai/clip-vit-base-patch16",
        "ViT-L/14":        "openai/clip-vit-large-patch14",
        "ViT-L/14@336px":  "openai/clip-vit-large-patch14-336",
    }
    hf_name = model_mapping.get(clip_model_name, clip_model_name)
    base_clip = CLIPModel.from_pretrained(hf_name).to(device)

    customized = CustomizedCLIP(base_clip, processor, regularize=False).to(device)
    customized.load_state_dict(checkpoint["model_state_dict"])
    customized.eval()
    return customized


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------

def evaluate_one_dataset(model, dataset_name: str, class_names: list,
                         evaluator: Evaluator, customizer: ModelCustomizer,
                         use_templates: bool, detailed: bool,
                         metric_type: str = "accuracy"):
    """
    Create test loader then run evaluation using the dataset's official
    ELEVATER metric (accuracy / mean_per_class / roc_auc / voc11_map) -
    same evaluate_with_dataset_metric() path 03_model_customization.py uses,
    instead of always computing plain top-1 accuracy.

    Returns
    -------
    accuracy : float (0-100)
    per_class : dict | None
    num_samples : int
    """
    test_loader = customizer._create_test_loader(
        target_dataset_name=dataset_name,
        class_names=class_names
    )
    if test_loader is None:
        return None, None, 0

    templates = None
    if use_templates:
        try:
            handler = ELEVATERDataset(dataset_name, train=False)
            templates = handler.get_text_templates()
        except Exception:
            pass

    result = evaluator.evaluate_with_dataset_metric(
        model, test_loader, class_names, dataset_name,
        metric_type=metric_type, templates=templates
    )
    accuracy = result['accuracy']

    # Per-class breakdown only makes sense for single-label metrics -
    # voc11_map's labels are multi-hot vectors, which evaluate_detailed()
    # isn't equipped to break down per class.
    per_class = None
    if detailed and metric_type != "voc11_map":
        _, per_class = evaluator.evaluate_detailed(
            model, test_loader, class_names, templates=templates
        )

    num_samples = len(test_loader.dataset)
    return accuracy, per_class, num_samples


# ---------------------------------------------------------------------------
# Result display helpers
# ---------------------------------------------------------------------------

def print_comparison_table(results: dict):
    """Print a formatted side-by-side comparison table."""
    no_baseline = all(r.get("baseline_accuracy") is None for r in results.values()
                      if r.get("customized_accuracy") is not None)

    if no_baseline:
        header = f"{'Dataset':<20} {'Customized':>12} {'Samples':>10}"
    else:
        header = f"{'Dataset':<20} {'Baseline':>10} {'Customized':>12} {'Δ (pp)':>9}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    improvements = []
    customized_accs = []
    for dataset_name in sorted(results.keys()):
        r = results[dataset_name]
        baseline = r.get("baseline_accuracy")
        custom   = r.get("customized_accuracy")

        if custom is None:
            if no_baseline:
                print(f"  {dataset_name:<18} {'N/A':>12} {'N/A':>10}")
            else:
                print(f"  {dataset_name:<18} {'N/A':>10} {'N/A':>12} {'N/A':>9}")
            continue

        c_str = f"{custom:.2f}%"
        customized_accs.append(custom)

        if no_baseline:
            n = r.get("num_test_samples", 0)
            print(f"  {dataset_name:<18} {c_str:>12} {n:>10,}")
        else:
            b_str = f"{baseline:.2f}%" if baseline is not None else "N/A"
            if baseline is not None:
                delta = custom - baseline
                delta_str = f"{delta:+.2f}"
                improvements.append(delta)
            else:
                delta_str = "N/A"
            print(f"  {dataset_name:<18} {b_str:>10} {c_str:>12} {delta_str:>9}")

    print(sep)
    if customized_accs:
        avg_c = sum(customized_accs) / len(customized_accs)
        if no_baseline:
            print(f"  {'AVERAGE':<18} {avg_c:>11.2f}%")
        elif improvements:
            avg_b   = sum(r["baseline_accuracy"] for r in results.values()
                          if r.get("baseline_accuracy") is not None) / len(improvements)
            avg_imp = sum(improvements) / len(improvements)
            print(f"  {'AVERAGE':<18} {avg_b:>9.2f}% {avg_c:>11.2f}% {avg_imp:>+9.2f}")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_arguments()

    # ── Validate paths ────────────────────────────────────────────────────
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    if not os.path.exists(args.config):
        print(f"❌ Config not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = args.batch_size or config.get("batch_size", 32)
    use_templates = not args.no_templates
    cache_base_dir = config.get("cache_base_dir", "./maveric_cache")
    # Per-dataset official ELEVATER metric (accuracy / mean_per_class / roc_auc /
    # voc11_map) - same mapping MAVERICConfig.evaluation_metrics uses in main.py.
    evaluation_metrics_map = config.get("evaluation_metrics", {})

    print("=" * 70)
    print("  MAVERIC – Unified Model Evaluation")
    print("=" * 70)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Device     : {device}")
    print(f"  Batch size : {batch_size}")
    print(f"  Templates  : {'enabled (REACT-style)' if use_templates else 'disabled'}")
    print(f"  Baseline   : {'disabled (--no-baseline)' if args.no_baseline else 'enabled'}")
    print(f"  Detailed   : {args.detailed}")

    # ── Load checkpoint ───────────────────────────────────────────────────
    checkpoint, clip_model_name, dataset_metadata = load_unified_checkpoint(
        args.checkpoint, device
    )

    # ── Determine which datasets to evaluate ─────────────────────────────
    if args.datasets:
        eval_datasets = args.datasets
    elif dataset_metadata:
        eval_datasets = sorted(dataset_metadata.keys())
    else:
        eval_datasets = sorted(ELEVATERDataset.ELEVATER_DATASETS.keys())

    print(f"\n  Datasets to evaluate ({len(eval_datasets)}): {eval_datasets}")

    # ── Build baseline model (skip if --no-baseline) ─────────────────────
    if args.no_baseline:
        print(f"\n🔧 Loading CLIP processor for {clip_model_name} (no baseline evaluation)...")
        # Still need the processor; load it via ModelCustomizer without keeping a baseline model
        _tmp = ModelCustomizer(
            base_model_name=clip_model_name,
            device=device,
            checkpoint_dir=None,
            cache_base_dir=None
        )
        processor = _tmp.processor
        baseline_model = None
        del _tmp
    else:
        print(f"\n🔧 Loading baseline CLIP model ({clip_model_name})...")
        baseline_model, processor = build_baseline_model(clip_model_name, device)

    # ── Build customized model ────────────────────────────────────────────
    print(f"🔧 Restoring customized model from checkpoint...")
    customized_model = build_customized_model(checkpoint, clip_model_name, device, processor)

    # ── Shared ModelCustomizer shell (for _create_test_loader) ────────────
    # We reuse ModelCustomizer._create_test_loader without re-downloading a model.
    from maveric.core.base import BaseComponent
    from transformers import CLIPModel

    model_mapping = {
        "ViT-B/32":       "openai/clip-vit-base-patch32",
        "ViT-B/16":       "openai/clip-vit-base-patch16",
        "ViT-L/14":       "openai/clip-vit-large-patch14",
        "ViT-L/14@336px": "openai/clip-vit-large-patch14-336",
    }
    hf_name = model_mapping.get(clip_model_name, clip_model_name)
    # Reuse the underlying CLIPModel from baseline (if present) or load a fresh one
    underlying_clip = (baseline_model.clip_model if baseline_model is not None
                       else CLIPModel.from_pretrained(hf_name).to(device))

    customizer = ModelCustomizer.__new__(ModelCustomizer)
    BaseComponent.__init__(customizer, "ModelCustomizer")
    customizer.base_model_name = clip_model_name
    customizer.device = device
    customizer.checkpoint_dir = None
    customizer.cache_base_dir = cache_base_dir
    customizer.model = underlying_clip
    customizer.processor = processor
    customizer.trainer = None
    customizer.evaluator = None

    evaluator = Evaluator(device=device)

    # ── Output directory ──────────────────────────────────────────────────
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Evaluate each dataset ─────────────────────────────────────────────
    results = {}

    for dataset_name in eval_datasets:
        print(f"\n{'─'*60}")
        print(f"📊 Evaluating: {dataset_name}")

        # Authoritative class names from ELEVATER_DATASETS
        if dataset_name in ELEVATERDataset.ELEVATER_DATASETS:
            class_names = ELEVATERDataset.ELEVATER_DATASETS[dataset_name]["class_names"]
        elif dataset_name in dataset_metadata:
            class_names = dataset_metadata[dataset_name]["class_names"]
        else:
            print(f"   ⚠️  Unknown dataset '{dataset_name}', skipping.")
            continue

        metric_type = evaluation_metrics_map.get(dataset_name.lower(), "accuracy")

        results[dataset_name] = {
            "num_classes": len(class_names),
            "metric_type": metric_type,
            "baseline_accuracy": None,
            "customized_accuracy": None,
            "improvement": None,
            "num_test_samples": 0,
        }

        # — Baseline (skipped when --no-baseline) ─────────────────────────
        b_acc = None
        if not args.no_baseline:
            print(f"   🔷 Baseline CLIP ({metric_type}) ...")
            b_acc, b_per_class, n_samples = evaluate_one_dataset(
                baseline_model, dataset_name, class_names,
                evaluator, customizer, use_templates, args.detailed,
                metric_type=metric_type
            )
            if b_acc is None:
                print(f"   ⚠️  Test data unavailable, skipping dataset.")
                continue

            results[dataset_name]["baseline_accuracy"] = round(b_acc, 4)
            results[dataset_name]["num_test_samples"]  = n_samples
            if args.detailed and b_per_class:
                results[dataset_name]["baseline_per_class"] = {
                    k: round(v, 2) for k, v in b_per_class.items()
                }
            print(f"      Baseline  : {b_acc:.2f}%  ({n_samples:,} samples)")

        # — Customized ————————————————————————————————————————————————————
        print(f"   🔶 Customized model ({metric_type}) ...")
        c_acc, c_per_class, n_samples_c = evaluate_one_dataset(
            customized_model, dataset_name, class_names,
            evaluator, customizer, use_templates, args.detailed,
            metric_type=metric_type
        )
        if c_acc is None:
            print(f"   ⚠️  Test data unavailable, skipping dataset.")
            continue

        results[dataset_name]["customized_accuracy"] = round(c_acc, 4)
        results[dataset_name]["num_test_samples"]    = n_samples_c
        if args.detailed and c_per_class:
            results[dataset_name]["customized_per_class"] = {
                k: round(v, 2) for k, v in c_per_class.items()
            }

        if b_acc is not None:
            improvement = c_acc - b_acc
            results[dataset_name]["improvement"] = round(improvement, 4)
            sign = "+" if improvement >= 0 else ""
            print(f"      Customized: {c_acc:.2f}%  ({sign}{improvement:.2f} pp)")
        else:
            print(f"      Customized: {c_acc:.2f}%  ({n_samples_c:,} samples)")

    # ── Summary table ─────────────────────────────────────────────────────
    print_comparison_table(results)

    # ── Save JSON results ─────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"unified_evaluation_{timestamp}.json"

    evaluated = [r for r in results.values() if r.get("customized_accuracy") is not None]
    evaluated_with_baseline = [r for r in evaluated if r.get("baseline_accuracy") is not None]

    avg_customized  = (sum(r["customized_accuracy"] for r in evaluated) / len(evaluated)) if evaluated else 0
    avg_baseline    = (sum(r["baseline_accuracy"]   for r in evaluated_with_baseline) / len(evaluated_with_baseline)) if evaluated_with_baseline else None
    avg_improvement = (avg_customized - avg_baseline) if avg_baseline is not None else None

    summary = {"avg_customized_accuracy": round(avg_customized, 4)}
    if avg_baseline is not None:
        summary["avg_baseline_accuracy"] = round(avg_baseline, 4)
        summary["avg_improvement"]       = round(avg_improvement, 4)

    output_data = {
        "evaluation_date": datetime.now().isoformat(),
        "checkpoint": str(args.checkpoint),
        "clip_model": clip_model_name,
        "use_templates": use_templates,
        "baseline_evaluated": not args.no_baseline,
        "num_datasets_evaluated": len(evaluated),
        "summary": summary,
        "per_dataset": results,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")
    print(f"\n🏁 Done. Evaluated {len(evaluated)}/{len(eval_datasets)} datasets.")
    if avg_baseline is not None:
        print(f"   Average baseline  : {avg_baseline:.2f}%")
    print(f"   Average customized: {avg_customized:.2f}%")
    if avg_improvement is not None:
        print(f"   Average improvement: {avg_improvement:+.2f} pp")


if __name__ == "__main__":
    main()
