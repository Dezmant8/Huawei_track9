"""Download dataset (4 parts, ~42 GB total).

Usage:
    python scripts/01_download_dataset.py
    python scripts/01_download_dataset.py --parts sparse pubtabnet
"""

import os
import sys
import json
import time
import zipfile
import logging
import argparse
import requests
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATASET_URLS, RAW_DIR, PARTS

logger = logging.getLogger(__name__)


def download_file(url: str, dest_path: str, max_retries: int = 3) -> None:
    """Download file with progress bar and exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))

            with open(dest_path, "wb") as f, tqdm(
                total=total_size, unit="B", unit_scale=True,
                desc=os.path.basename(dest_path)
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
            return

        except (requests.RequestException, IOError) as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"Failed to download {url} after {max_retries} attempts")


def verify_extraction(part_dir: str, part_name: str) -> bool:
    """Verify extracted dataset: check JSONL, image dirs, and format."""
    jsonl_path = os.path.join(part_dir, "synthetic_data.jsonl")
    if not os.path.exists(jsonl_path):
        logger.error(f"[{part_name}] synthetic_data.jsonl not found")
        return False

    for split in ("train", "val", "test"):
        split_dir = os.path.join(part_dir, "images", split)
        if not os.path.isdir(split_dir):
            logger.error(f"[{part_name}] images/{split}/ directory not found")
            return False

    try:
        with open(jsonl_path, "r") as f:
            first_line = json.loads(f.readline())
            required_keys = {"filename", "split", "html"}
            if not required_keys.issubset(first_line.keys()):
                logger.error(f"[{part_name}] JSONL missing keys: {required_keys - first_line.keys()}")
                return False
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"[{part_name}] JSONL format error: {e}")
        return False

    split_counts = {"train": 0, "val": 0, "test": 0}
    with open(jsonl_path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            split_counts[entry["split"]] = split_counts.get(entry["split"], 0) + 1

    logger.info(
        f"[{part_name}] Verified: train={split_counts['train']}, "
        f"val={split_counts['val']}, test={split_counts['test']}"
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="Download SynthTabNet v2.0.0 dataset")
    parser.add_argument(
        "--parts", nargs="+", default=PARTS, choices=PARTS,
        help="Which parts to download (default: all)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    os.makedirs(RAW_DIR, exist_ok=True)

    success = []
    failed = []

    for part in args.parts:
        part_dir = os.path.join(RAW_DIR, part)
        jsonl_path = os.path.join(part_dir, "synthetic_data.jsonl")

        if os.path.exists(jsonl_path):
            logger.info(f"[{part}] Already downloaded, skipping.")
            success.append(part)
            continue

        url = DATASET_URLS[part]
        zip_path = os.path.join(RAW_DIR, f"{part}.zip")

        try:
            logger.info(f"[{part}] Downloading from {url}")
            download_file(url, zip_path)

            logger.info(f"[{part}] Extracting...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(RAW_DIR)

            if verify_extraction(part_dir, part):
                success.append(part)
                logger.info(f"[{part}] OK")
            else:
                failed.append(part)

            os.remove(zip_path)

        except Exception as e:
            failed.append(part)
            logger.error(f"[{part}] Failed: {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path)

    print(f"\n{'='*50}")
    print(f"Download complete: {len(success)}/{len(args.parts)} parts successful")
    if success:
        print(f"  OK: {', '.join(success)}")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
    print(f"Data location: {RAW_DIR}")


if __name__ == "__main__":
    main()
