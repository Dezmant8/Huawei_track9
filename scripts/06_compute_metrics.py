"""Compute TEDS/TEDS-struct metrics with statistical analysis.

Features: bootstrap CI, Wilcoxon signed-rank tests, error analysis,
comparison tables, and visualization (boxplot, heatmap, histogram).

Usage:
    python scripts/06_compute_metrics.py --predictions results/baseline_trivia.json
    python scripts/06_compute_metrics.py --compare
    python scripts/06_compute_metrics.py --predictions results/baseline_trivia.json --visualize
"""

import os
import sys
import json
import logging
import argparse
from collections import defaultdict

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BENCHMARK_DIR, RESULTS_DIR, ANGLES, PARTS, BOOTSTRAP_N, BOOTSTRAP_CI, SIGNIFICANCE_ALPHA
from utils.teds import compute_teds, compute_teds_struct
from utils.html_utils import normalize_html_for_teds

logger = logging.getLogger(__name__)


def load_benchmark_meta() -> dict:
    """Load benchmark metadata as {index: entry}."""
    meta_path = os.path.join(BENCHMARK_DIR, "benchmark_meta.jsonl")
    entries = {}
    with open(meta_path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            entries[entry["index"]] = entry
    return entries


def compute_metrics(predictions_file: str, benchmark_meta: dict) -> list:
    """Compute TEDS and TEDS-struct for all predictions."""
    with open(predictions_file, "r") as f:
        predictions = json.load(f)

    results = []
    for pred in tqdm(predictions, desc="Computing TEDS"):
        idx = pred["index"]
        gt_entry = benchmark_meta[idx]
        gt_html = gt_entry["gt_html"]
        pred_html = normalize_html_for_teds(pred.get("pred_html", ""))

        try:
            teds_score = compute_teds(pred_html, gt_html)
            teds_struct_score = compute_teds_struct(pred_html, gt_html)
        except Exception as e:
            logger.warning(f"TEDS computation failed for index {idx}: {e}")
            teds_score = 0.0
            teds_struct_score = 0.0

        is_empty = not pred.get("pred_html", "").strip()

        results.append({
            "index": idx,
            "filename": pred["filename"],
            "part": pred["part"],
            "rotation_angle": pred["rotation_angle"],
            "teds": teds_score,
            "teds_struct": teds_struct_score,
            "is_empty_prediction": is_empty,
        })

    return results


def compute_stats(values: list) -> dict:
    """Compute descriptive statistics: mean, std, median, quartiles, min, max."""
    if not values:
        return {"mean": 0, "std": 0, "median": 0, "q25": 0, "q75": 0,
                "min": 0, "max": 0, "count": 0}

    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": len(values),
    }


def bootstrap_ci(values: list, n_bootstrap: int = BOOTSTRAP_N,
                  ci: float = BOOTSTRAP_CI) -> dict:
    """Compute bootstrap confidence interval for the mean."""
    if len(values) < 2:
        m = np.mean(values) if values else 0.0
        return {"mean": float(m), "ci_lower": float(m), "ci_upper": float(m)}

    arr = np.array(values)
    rng = np.random.RandomState(42)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        bootstrap_means.append(np.mean(sample))

    bootstrap_means = np.array(bootstrap_means)
    alpha = (1 - ci) / 2

    return {
        "mean": float(np.mean(arr)),
        "ci_lower": float(np.percentile(bootstrap_means, alpha * 100)),
        "ci_upper": float(np.percentile(bootstrap_means, (1 - alpha) * 100)),
    }


def significance_test(scores_a: list, scores_b: list,
                       alpha: float = SIGNIFICANCE_ALPHA) -> dict:
    """Wilcoxon signed-rank test with Cohen's d effect size."""
    from scipy import stats

    if len(scores_a) != len(scores_b):
        return {"error": "Lists must have equal length"}

    arr_a = np.array(scores_a)
    arr_b = np.array(scores_b)
    diff = arr_b - arr_a

    try:
        stat, p_value = stats.wilcoxon(diff, alternative="two-sided")
    except ValueError:
        stat, p_value = 0.0, 1.0

    pooled_std = np.sqrt((np.var(arr_a) + np.var(arr_b)) / 2)
    cohens_d = (np.mean(arr_b) - np.mean(arr_a)) / max(pooled_std, 1e-8)

    return {
        "wilcoxon_statistic": float(stat),
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "cohens_d": float(cohens_d),
        "mean_diff": float(np.mean(diff)),
        "median_diff": float(np.median(diff)),
    }


def aggregate_results(results: list) -> dict:
    """Aggregate metrics by angle and dataset part."""
    groups = {
        "overall": {"teds": [], "teds_struct": []},
        "by_angle": defaultdict(lambda: {"teds": [], "teds_struct": []}),
        "by_part": defaultdict(lambda: {"teds": [], "teds_struct": []}),
    }

    empty_count = 0
    for r in results:
        groups["overall"]["teds"].append(r["teds"])
        groups["overall"]["teds_struct"].append(r["teds_struct"])
        groups["by_angle"][r["rotation_angle"]]["teds"].append(r["teds"])
        groups["by_angle"][r["rotation_angle"]]["teds_struct"].append(r["teds_struct"])
        groups["by_part"][r["part"]]["teds"].append(r["teds"])
        groups["by_part"][r["part"]]["teds_struct"].append(r["teds_struct"])
        if r.get("is_empty_prediction"):
            empty_count += 1

    summary = {
        "overall": {
            "teds": compute_stats(groups["overall"]["teds"]),
            "teds_struct": compute_stats(groups["overall"]["teds_struct"]),
            "teds_ci": bootstrap_ci(groups["overall"]["teds"]),
            "teds_struct_ci": bootstrap_ci(groups["overall"]["teds_struct"]),
            "empty_predictions": empty_count,
        },
        "by_angle": {},
        "by_part": {},
    }

    for angle in sorted(groups["by_angle"]):
        d = groups["by_angle"][angle]
        summary["by_angle"][angle] = {
            "teds": compute_stats(d["teds"]),
            "teds_struct": compute_stats(d["teds_struct"]),
        }

    for part in sorted(groups["by_part"]):
        d = groups["by_part"][part]
        summary["by_part"][part] = {
            "teds": compute_stats(d["teds"]),
            "teds_struct": compute_stats(d["teds_struct"]),
        }

    return summary


def error_analysis(results: list, top_k: int = 10) -> dict:
    """Categorize errors and find worst predictions."""
    categories = {"empty": 0, "poor_structure": 0, "poor_content": 0, "acceptable": 0}

    for r in results:
        if r.get("is_empty_prediction"):
            categories["empty"] += 1
        elif r["teds_struct"] < 0.5:
            categories["poor_structure"] += 1
        elif r["teds"] < 0.5:
            categories["poor_content"] += 1
        else:
            categories["acceptable"] += 1

    sorted_by_teds = sorted(results, key=lambda x: x["teds"])
    worst = sorted_by_teds[:top_k]

    return {
        "error_categories": categories,
        "worst_samples": [
            {"index": w["index"], "filename": w["filename"], "part": w["part"],
             "angle": w["rotation_angle"], "teds": round(w["teds"], 4),
             "teds_struct": round(w["teds_struct"], 4)}
            for w in worst
        ],
    }


def create_visualizations(results: list, output_dir: str, label: str):
    """Generate boxplot, histogram, and heatmap plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not installed, skipping visualizations")
        return

    os.makedirs(output_dir, exist_ok=True)

    teds_scores = [r["teds"] for r in results]

    # Boxplot by angle
    _, ax = plt.subplots(figsize=(8, 5))
    data_by_angle = {a: [] for a in ANGLES}
    for r in results:
        data_by_angle[r["rotation_angle"]].append(r["teds"])
    ax.boxplot(
        [data_by_angle[a] for a in ANGLES],
        labels=[f"{a}" for a in ANGLES],
    )
    ax.set_xlabel("Rotation Angle")
    ax.set_ylabel("TEDS Score")
    ax.set_title(f"TEDS Distribution by Rotation Angle - {label}")
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{label}_boxplot_angles.png"), dpi=150)
    plt.close()

    # Histogram
    _, ax = plt.subplots(figsize=(8, 5))
    ax.hist(teds_scores, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(np.mean(teds_scores), color="red", linestyle="--",
               label=f"Mean: {np.mean(teds_scores):.4f}")
    ax.axvline(np.median(teds_scores), color="blue", linestyle="--",
               label=f"Median: {np.median(teds_scores):.4f}")
    ax.set_xlabel("TEDS Score")
    ax.set_ylabel("Count")
    ax.set_title(f"TEDS Score Distribution - {label}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{label}_histogram.png"), dpi=150)
    plt.close()

    # Heatmap: part x angle
    _, ax = plt.subplots(figsize=(8, 5))
    heatmap_data = np.zeros((len(PARTS), len(ANGLES)))
    for i, part in enumerate(PARTS):
        for j, angle in enumerate(ANGLES):
            vals = [r["teds"] for r in results
                    if r["part"] == part and r["rotation_angle"] == angle]
            heatmap_data[i, j] = np.mean(vals) if vals else 0

    sns.heatmap(
        heatmap_data, annot=True, fmt=".3f", cmap="YlOrRd_r",
        xticklabels=[f"{a}" for a in ANGLES],
        yticklabels=PARTS,
        ax=ax, vmin=0, vmax=1,
    )
    ax.set_title(f"Mean TEDS by Part x Angle - {label}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{label}_heatmap.png"), dpi=150)
    plt.close()

    logger.info(f"Visualizations saved to {output_dir}")


def print_report(summary: dict, label: str = ""):
    """Print formatted metrics report."""
    print(f"\n{'='*70}")
    if label:
        print(f"  {label}")
    print(f"{'='*70}")

    o = summary["overall"]
    ts = o["teds"]
    tss = o["teds_struct"]
    tci = o["teds_ci"]
    tsci = o["teds_struct_ci"]

    print(f"\n  Overall ({ts['count']} samples, {o['empty_predictions']} empty predictions):")
    print(f"    TEDS:        {ts['mean']:.4f} +/- {ts['std']:.4f}  "
          f"[95% CI: {tci['ci_lower']:.4f} - {tci['ci_upper']:.4f}]")
    print(f"    TEDS-struct: {tss['mean']:.4f} +/- {tss['std']:.4f}  "
          f"[95% CI: {tsci['ci_lower']:.4f} - {tsci['ci_upper']:.4f}]")
    print(f"    Median TEDS: {ts['median']:.4f}  (Q25={ts['q25']:.4f}, Q75={ts['q75']:.4f})")

    print(f"\n  By rotation angle:")
    print(f"    {'Angle':>8}  {'TEDS':>8}  {'std':>7}  {'TEDS-s':>8}  {'std':>7}  {'N':>5}")
    print(f"    {'-'*52}")
    for angle in sorted(summary["by_angle"], key=int):
        d = summary["by_angle"][angle]
        print(f"    {angle:>7}  {d['teds']['mean']:>8.4f}  {d['teds']['std']:>7.4f}  "
              f"{d['teds_struct']['mean']:>8.4f}  {d['teds_struct']['std']:>7.4f}  "
              f"{d['teds']['count']:>5}")

    print(f"\n  By dataset part:")
    print(f"    {'Part':>12}  {'TEDS':>8}  {'std':>7}  {'TEDS-s':>8}  {'std':>7}  {'N':>5}")
    print(f"    {'-'*56}")
    for part in sorted(summary["by_part"]):
        d = summary["by_part"][part]
        print(f"    {part:>12}  {d['teds']['mean']:>8.4f}  {d['teds']['std']:>7.4f}  "
              f"{d['teds_struct']['mean']:>8.4f}  {d['teds_struct']['std']:>7.4f}  "
              f"{d['teds']['count']:>5}")


def compare_results(results_dir: str, benchmark_meta: dict):
    """Print comparison table for all result files + significance tests."""
    result_files = sorted([
        f for f in os.listdir(results_dir)
        if f.endswith(".json") and not f.startswith("metrics_")
    ])

    if not result_files:
        print("No result files found in", results_dir)
        return

    print(f"\n{'='*90}")
    print("  COMPARISON TABLE")
    print(f"{'='*90}")

    header = (f"{'Model':>25}  {'TEDS':>7}  {'std':>6}  {'TEDS-s':>7}  "
              f"{'@0':>6}  {'@90':>6}  {'@180':>6}  {'@270':>6}")
    print(f"\n  {header}")
    print(f"  {'-'*len(header)}")

    all_results = {}
    for fname in result_files:
        filepath = os.path.join(results_dir, fname)
        label = fname.replace(".json", "")
        try:
            results = compute_metrics(filepath, benchmark_meta)
            summary = aggregate_results(results)
            all_results[label] = results

            o = summary["overall"]["teds"]
            angles = summary["by_angle"]
            t0 = angles.get(0, {}).get("teds", {}).get("mean", 0)
            t90 = angles.get(90, {}).get("teds", {}).get("mean", 0)
            t180 = angles.get(180, {}).get("teds", {}).get("mean", 0)
            t270 = angles.get(270, {}).get("teds", {}).get("mean", 0)
            print(f"  {label:>25}  {o['mean']:>7.4f}  {o['std']:>6.4f}  "
                  f"{summary['overall']['teds_struct']['mean']:>7.4f}  "
                  f"{t0:>6.4f}  {t90:>6.4f}  {t180:>6.4f}  {t270:>6.4f}")
        except Exception as e:
            print(f"  {label:>25}  ERROR: {e}")

    # Significance tests: baseline vs pipeline
    pairs = []
    for model in ["trivia", "qwen"]:
        baseline_key = f"baseline_{model}"
        pipeline_key = f"pipeline_{model}"
        if baseline_key in all_results and pipeline_key in all_results:
            pairs.append((baseline_key, pipeline_key))

    if pairs:
        print(f"\n  STATISTICAL SIGNIFICANCE TESTS")
        print(f"  {'-'*60}")
        for base_key, pipe_key in pairs:
            base_scores = [r["teds"] for r in sorted(all_results[base_key], key=lambda x: x["index"])]
            pipe_scores = [r["teds"] for r in sorted(all_results[pipe_key], key=lambda x: x["index"])]

            if len(base_scores) == len(pipe_scores):
                test = significance_test(base_scores, pipe_scores)
                sig_marker = "***" if test["significant"] else "n.s."
                print(f"  {base_key} vs {pipe_key}:")
                print(f"    Mean diff: {test['mean_diff']:+.4f}, "
                      f"Cohen's d: {test['cohens_d']:.3f}, "
                      f"p={test['p_value']:.4f} {sig_marker}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute TEDS metrics and statistical analysis"
    )
    parser.add_argument("--predictions", type=str, default=None)
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

    benchmark_meta = load_benchmark_meta()

    if args.compare:
        compare_results(RESULTS_DIR, benchmark_meta)
        return

    if not args.predictions:
        parser.error("--predictions required (or use --compare)")

    label = args.label or os.path.basename(args.predictions).replace(".json", "")

    logger.info(f"Computing metrics for: {args.predictions}")
    results = compute_metrics(args.predictions, benchmark_meta)
    summary = aggregate_results(results)
    errors = error_analysis(results)

    print_report(summary, label=label)

    print(f"\n  Error categories:")
    for cat, count in errors["error_categories"].items():
        pct = count / max(1, len(results)) * 100
        print(f"    {cat:>16}: {count:>4} ({pct:.1f}%)")

    print(f"\n  Worst {len(errors['worst_samples'])} samples by TEDS:")
    for w in errors["worst_samples"]:
        print(f"    idx={w['index']:>4} {w['filename']:>35} "
              f"part={w['part']:>10} angle={w['angle']:>3} "
              f"TEDS={w['teds']:.4f} TEDS-s={w['teds_struct']:.4f}")

    if args.visualize:
        viz_dir = os.path.join(RESULTS_DIR, "plots")
        create_visualizations(results, viz_dir, label)

    output_path = args.output or args.predictions.replace(".json", "_metrics.json")
    output_data = {
        "label": label,
        "summary": summary,
        "error_analysis": errors,
        "per_sample": results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    logger.info(f"Detailed metrics saved to: {output_path}")


if __name__ == "__main__":
    main()
