"""Evaluate VLM baselines on rotated table benchmark (no rotation correction).

Supported models:
- TRivia-3B: table-specialized VLM, outputs OTSL -> converted to HTML
- Qwen2.5-VL-3B-Instruct: general-purpose VLM, outputs HTML directly

Usage:
    python scripts/03_evaluate_baselines.py --model trivia
    python scripts/03_evaluate_baselines.py --model qwen --limit 10
    python scripts/03_evaluate_baselines.py --model trivia --resume
"""

import os
import sys
import re
import json
import time
import logging
import argparse
from abc import ABC, abstractmethod

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    BENCHMARK_DIR, RESULTS_DIR, TRIVIA_DIR,
    TRIVIA_SYSTEM_PROMPT, QWEN_TABLE_PROMPT,
    MIN_PIXELS, MAX_PIXELS, GENERATION_PARAMS,
)
from utils.seed import get_device, set_seed, setup_logging

sys.path.insert(0, TRIVIA_DIR)

logger = logging.getLogger(__name__)

CHECKPOINT_EVERY = 50


class VLMEvaluator(ABC):
    """Base class for VLM evaluation (Template Method pattern)."""

    def __init__(self, checkpoint: str, device: str):
        self.checkpoint = checkpoint
        self.device = device
        self.model = None
        self.processor = None

    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def build_messages(self, image: Image.Image) -> list:
        pass

    @abstractmethod
    def postprocess(self, raw_output: str) -> str:
        pass

    def run_inference(self, image: Image.Image) -> dict:
        """Run single-image inference. Returns raw_output, pred_html, inference_time_s."""
        from qwen_vl_utils import process_vision_info

        messages = self.build_messages(image)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        start_time = time.time()
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **self.generation_params)
        inference_time = time.time() - start_time

        input_len = inputs["input_ids"].shape[1]
        output_ids = generated_ids[:, input_len:]
        raw_output = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        pred_html = self.postprocess(raw_output)

        del inputs, generated_ids, output_ids
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        return {
            "raw_output": raw_output,
            "pred_html": pred_html,
            "inference_time_s": round(inference_time, 2),
        }


class TRiviaEvaluator(VLMEvaluator):
    """TRivia-3B evaluator. Outputs OTSL, converted to HTML via otsl_utils."""

    def __init__(self, checkpoint: str = "opendatalab/TRivia-3B", device: str = "cpu"):
        super().__init__(checkpoint, device)
        self.generation_params = GENERATION_PARAMS["trivia"]

    def load_model(self) -> None:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        logger.info(f"Loading TRivia-3B from {self.checkpoint}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.checkpoint,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device == "mps":
            self.model = self.model.to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.checkpoint)
        logger.info("TRivia-3B loaded")

    def build_messages(self, image: Image.Image) -> list:
        return [{
            "role": "user",
            "content": [
                {"type": "text", "text": TRIVIA_SYSTEM_PROMPT},
                {"type": "image", "image": image,
                 "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
            ],
        }]

    def postprocess(self, raw_output: str) -> str:
        from otsl_utils import convert_otsl_to_html

        if not raw_output.strip():
            return ""
        try:
            return convert_otsl_to_html(raw_output)
        except Exception as e:
            logger.warning(f"OTSL->HTML conversion failed: {e}")
            return ""


class QwenEvaluator(VLMEvaluator):
    """Qwen2.5-VL-3B-Instruct evaluator. Outputs HTML directly."""

    def __init__(self, checkpoint: str = "Qwen/Qwen2.5-VL-3B-Instruct", device: str = "cpu"):
        super().__init__(checkpoint, device)
        self.generation_params = GENERATION_PARAMS["qwen"]

    def load_model(self) -> None:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        logger.info(f"Loading Qwen2.5-VL-3B from {self.checkpoint}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.checkpoint,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device == "mps":
            self.model = self.model.to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.checkpoint)
        logger.info("Qwen2.5-VL-3B loaded")

    def build_messages(self, image: Image.Image) -> list:
        return [{
            "role": "user",
            "content": [
                {"type": "image", "image": image,
                 "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
                {"type": "text", "text": QWEN_TABLE_PROMPT},
            ],
        }]

    def postprocess(self, raw_output: str) -> str:
        if not raw_output.strip():
            return ""
        table_match = re.search(
            r"<table.*?</table>", raw_output, re.DOTALL | re.IGNORECASE
        )
        if table_match:
            return table_match.group(0)
        return raw_output


def load_benchmark_meta() -> list:
    """Load benchmark metadata from JSONL."""
    meta_path = os.path.join(BENCHMARK_DIR, "benchmark_meta.jsonl")
    entries = []
    with open(meta_path, "r") as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    return entries


def load_existing_results(output_file: str) -> dict:
    """Load previously saved results for --resume. Returns {index: result}."""
    if not os.path.exists(output_file):
        return {}
    with open(output_file, "r") as f:
        results = json.load(f)
    return {r["index"]: r for r in results}


def save_results(results: list, output_file: str) -> None:
    """Atomic save via temp file."""
    tmp_file = output_file + ".tmp"
    with open(tmp_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    os.replace(tmp_file, output_file)


def evaluate_model(
    evaluator: VLMEvaluator,
    entries: list,
    output_file: str,
    resume: bool = False,
) -> list:
    """Run model evaluation on benchmark entries."""
    existing = load_existing_results(output_file) if resume else {}
    if existing:
        logger.info(f"Resuming: {len(existing)} samples already processed")

    evaluator.load_model()

    results = list(existing.values())
    processed_indices = set(existing.keys())
    total_time = 0.0
    errors = 0

    for entry in tqdm(entries, desc="Inference"):
        if entry["index"] in processed_indices:
            continue

        img_path = os.path.join(BENCHMARK_DIR, "images", entry["filename"])

        try:
            image = Image.open(img_path).convert("RGB")
            result = evaluator.run_inference(image)

            results.append({
                "index": entry["index"],
                "filename": entry["filename"],
                "part": entry["part"],
                "rotation_angle": entry["rotation_angle"],
                "pred_html": result["pred_html"],
                "raw_output": result["raw_output"],
                "inference_time_s": result["inference_time_s"],
            })
            total_time += result["inference_time_s"]

        except Exception as e:
            logger.error(f"Error processing {entry['filename']}: {e}")
            errors += 1
            results.append({
                "index": entry["index"],
                "filename": entry["filename"],
                "part": entry["part"],
                "rotation_angle": entry["rotation_angle"],
                "pred_html": "",
                "raw_output": f"ERROR: {str(e)}",
                "inference_time_s": 0.0,
            })

        if len(results) % CHECKPOINT_EVERY == 0:
            save_results(results, output_file)
            logger.info(f"Checkpoint saved: {len(results)} samples")

    save_results(results, output_file)

    non_empty = sum(1 for r in results if r.get("pred_html", "").strip())
    avg_time = total_time / max(1, len(results) - len(existing))

    logger.info(f"Evaluation complete: {len(results)} total, {non_empty} non-empty, "
                f"{errors} errors, {avg_time:.2f}s avg")
    logger.info(f"Results saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VLM baselines on rotated table benchmark"
    )
    parser.add_argument("--model", required=True, choices=["trivia", "qwen"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    setup_logging(log_file=os.path.join(RESULTS_DIR, f"baseline_{args.model}.log"))
    set_seed()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = get_device()

    if args.model == "trivia":
        ckpt = args.checkpoint or "opendatalab/TRivia-3B"
        evaluator = TRiviaEvaluator(checkpoint=ckpt, device=device)
        output_file = os.path.join(RESULTS_DIR, "baseline_trivia.json")
    else:
        ckpt = args.checkpoint or "Qwen/Qwen2.5-VL-3B-Instruct"
        evaluator = QwenEvaluator(checkpoint=ckpt, device=device)
        output_file = os.path.join(RESULTS_DIR, "baseline_qwen.json")

    entries = load_benchmark_meta()
    if args.limit:
        entries = entries[:args.limit]
        logger.info(f"Limited to {args.limit} samples")

    evaluate_model(evaluator, entries, output_file, resume=args.resume)


if __name__ == "__main__":
    main()
