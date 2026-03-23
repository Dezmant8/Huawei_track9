"""Build rotated-table benchmark from SynthTabNet test splits.

Takes 250 test samples per part, applies deterministic rotation,
saves rotated images and normalized GT HTML.

Usage:
    python scripts/02_build_benchmark.py
    python scripts/02_build_benchmark.py --samples-per-part 50
"""

import os
import re
import sys
import json
import logging
import argparse
from collections import defaultdict

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PARTS, SAMPLES_PER_PART, RAW_DIR, BENCHMARK_DIR, BENCHMARK_IMAGES_DIR
from utils.html_utils import format_html, normalize_html_for_teds
from utils.rotation import rotate_image, angle_from_index

logger = logging.getLogger(__name__)


def load_test_samples(part: str, n_samples: int) -> list:
    """Load first n_samples test entries from SynthTabNet, sorted by filename."""
    jsonl_path = os.path.join(RAW_DIR, part, "synthetic_data.jsonl")
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(
            f"Annotation file not found: {jsonl_path}. "
            f"Run 01_download_dataset.py first."
        )

    test_entries = []
    with open(jsonl_path, "r") as f:
        for line in f:
            raw = line.strip()
            # Fix trailing commas in SynthTabNet JSON arrays
            raw = re.sub(r",\s*([}\]])", r"\1", raw)
            entry = json.loads(raw)
            if entry["split"] == "test":
                test_entries.append(entry)

    test_entries.sort(key=lambda x: x["filename"])

    if len(test_entries) < n_samples:
        logger.warning(
            f"[{part}] Only {len(test_entries)} test samples available "
            f"(requested {n_samples})"
        )

    return test_entries[:n_samples]


def compute_table_complexity(ann: dict) -> dict:
    """Compute table complexity stats from annotation."""
    cells = ann["html"]["cells"]
    tokens = ann["html"]["structure"]["tokens"]

    n_cells = len(cells)
    n_rows = tokens.count("</tr>")
    n_cols = max(1, n_cells // max(1, n_rows)) if n_rows > 0 else 0

    n_spans = sum(1 for c in cells if "span" in c and c["span"])
    n_header_cells = sum(1 for c in cells if c.get("is_header", False))
    n_empty_cells = sum(1 for c in cells if not "".join(c.get("tokens", [])).strip())

    return {
        "n_cells": n_cells,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_spans": n_spans,
        "n_header_cells": n_header_cells,
        "n_empty_cells": n_empty_cells,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build rotated-table benchmark from SynthTabNet test splits"
    )
    parser.add_argument(
        "--samples-per-part", type=int, default=SAMPLES_PER_PART,
        help=f"Number of test samples per part (default: {SAMPLES_PER_PART})"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    os.makedirs(BENCHMARK_IMAGES_DIR, exist_ok=True)

    benchmark_meta = []
    image_sizes = []
    complexity_stats = []
    failed_samples = 0
    global_idx = 0

    for part in PARTS:
        logger.info(f"Processing {part}...")
        samples = load_test_samples(part, args.samples_per_part)
        logger.info(f"  Loaded {len(samples)} test samples")

        for local_idx, entry in enumerate(tqdm(samples, desc=f"  {part}")):
            angle = angle_from_index(global_idx)
            img_path = os.path.join(RAW_DIR, part, "images", "test", entry["filename"])

            try:
                img = Image.open(img_path).convert("RGB")
                original_size = img.size
                rotated_img = rotate_image(img, angle)

                out_filename = f"{part}_{local_idx:04d}_rot{angle}.png"
                out_path = os.path.join(BENCHMARK_IMAGES_DIR, out_filename)
                rotated_img.save(out_path)

                gt_html = format_html(entry)
                gt_html_normalized = normalize_html_for_teds(gt_html)
                complexity = compute_table_complexity(entry)

                benchmark_meta.append({
                    "index": global_idx,
                    "filename": out_filename,
                    "part": part,
                    "original_filename": entry["filename"],
                    "rotation_angle": angle,
                    "image_width": original_size[0],
                    "image_height": original_size[1],
                    "gt_html": gt_html_normalized,
                    **complexity,
                })

                image_sizes.append(original_size)
                complexity_stats.append(complexity)

            except Exception as e:
                logger.error(f"Failed to process {entry['filename']}: {e}")
                failed_samples += 1

            global_idx += 1

    # Save metadata
    meta_path = os.path.join(BENCHMARK_DIR, "benchmark_meta.jsonl")
    with open(meta_path, "w") as f:
        for entry in benchmark_meta:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Validate
    total = len(benchmark_meta)
    angle_counts = defaultdict(int)
    part_counts = defaultdict(int)
    for entry in benchmark_meta:
        angle_counts[entry["rotation_angle"]] += 1
        part_counts[entry["part"]] += 1

    expected_total = args.samples_per_part * len(PARTS)
    issues = []
    if total != expected_total:
        issues.append(f"Expected {expected_total} samples, got {total}")
    for angle in [0, 90, 180, 270]:
        expected_per_angle = total // 4
        if abs(angle_counts[angle] - expected_per_angle) > 1:
            issues.append(f"Angle {angle}: {angle_counts[angle]} (expected ~{expected_per_angle})")
    for part in PARTS:
        if part_counts[part] != args.samples_per_part:
            issues.append(f"Part {part}: {part_counts[part]} (expected {args.samples_per_part})")

    empty_gt = sum(1 for e in benchmark_meta if not e["gt_html"].strip())
    if empty_gt > 0:
        issues.append(f"{empty_gt} samples have empty GT HTML")

    # Compute and save stats
    if image_sizes:
        widths = [s[0] for s in image_sizes]
        heights = [s[1] for s in image_sizes]

        stats = {
            "total_samples": total,
            "failed_samples": failed_samples,
            "samples_per_part": dict(part_counts),
            "samples_per_angle": dict(angle_counts),
            "image_stats": {
                "width": {"min": min(widths), "max": max(widths),
                          "mean": sum(widths) / len(widths)},
                "height": {"min": min(heights), "max": max(heights),
                           "mean": sum(heights) / len(heights)},
            },
            "table_complexity": {
                "avg_cells": sum(c["n_cells"] for c in complexity_stats) / len(complexity_stats),
                "avg_rows": sum(c["n_rows"] for c in complexity_stats) / len(complexity_stats),
                "avg_cols": sum(c["n_cols"] for c in complexity_stats) / len(complexity_stats),
                "avg_spans": sum(c["n_spans"] for c in complexity_stats) / len(complexity_stats),
                "max_cells": max(c["n_cells"] for c in complexity_stats),
                "max_rows": max(c["n_rows"] for c in complexity_stats),
            },
            "validation_issues": issues,
        }

        stats_path = os.path.join(BENCHMARK_DIR, "benchmark_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Benchmark created: {total} samples")
    print(f"{'='*60}")
    print(f"\n  Rotation distribution:")
    for angle in sorted(angle_counts):
        print(f"    {angle:>4}: {angle_counts[angle]} samples")
    print(f"\n  Part distribution:")
    for part in PARTS:
        print(f"    {part:>12}: {part_counts[part]} samples")
    if image_sizes:
        print(f"\n  Image sizes: {min(widths)}x{min(heights)} - {max(widths)}x{max(heights)}")
        print(f"  Avg table complexity: {stats['table_complexity']['avg_cells']:.1f} cells, "
              f"{stats['table_complexity']['avg_rows']:.1f} rows, "
              f"{stats['table_complexity']['avg_spans']:.1f} spans")
    if failed_samples:
        print(f"\n  WARNING: {failed_samples} samples failed to process")
    if issues:
        print(f"\n  Validation issues:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"\n  Validation: PASSED")
    print(f"\n  Metadata: {meta_path}")
    print(f"  Images: {BENCHMARK_IMAGES_DIR}")
    if image_sizes:
        print(f"  Statistics: {stats_path}")


if __name__ == "__main__":
    main()
