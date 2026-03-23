"""Pipeline: rotation classifier + VLM inference.

Two-phase approach to avoid MPS memory pressure:
  Phase 1: classify all rotation angles, then unload classifier
  Phase 2: load VLM and run inference on corrected images

Usage:
    python scripts/05_run_pipeline.py --model trivia
    python scripts/05_run_pipeline.py --model qwen --resume
"""

import os
import sys
import json
import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    BENCHMARK_DIR, MODEL_DIR, RESULTS_DIR, ANGLES,
    CLASSIFIER_RESIZE, CLASSIFIER_CROP,
    IMAGENET_MEAN, IMAGENET_STD, NUM_ROTATION_CLASSES,
)
from utils.rotation import correct_rotation
from utils.seed import get_device, set_seed, setup_logging

logger = logging.getLogger(__name__)

CHECKPOINT_EVERY = 50


def _import_evaluators():
    """Lazy import of VLM evaluators from 03_evaluate_baselines.py."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "evaluate_baselines",
        os.path.join(os.path.dirname(__file__), "03_evaluate_baselines.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_rotation_classifier(device: str) -> nn.Module:
    """Load trained ResNet-18 rotation classifier."""
    model_path = os.path.join(MODEL_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Rotation classifier not found at {model_path}. "
            f"Run 04_train_rotation_classifier.py first."
        )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_ROTATION_CLASSES)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info(f"Rotation classifier loaded from {model_path}")
    return model


def predict_rotation(
    classifier: nn.Module,
    image: Image.Image,
    device: str,
    transform: transforms.Compose,
) -> dict:
    """Predict rotation angle with confidence scores."""
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = classifier(img_tensor)
        probs = F.softmax(logits, dim=1)[0]
        pred_class = probs.argmax().item()

    predicted_angle = ANGLES[pred_class]
    confidence = probs[pred_class].item()
    all_probs = {str(ANGLES[i]): round(probs[i].item(), 4) for i in range(len(ANGLES))}

    return {
        "predicted_angle": predicted_angle,
        "confidence": round(confidence, 4),
        "all_probs": all_probs,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run rotation correction + VLM pipeline"
    )
    parser.add_argument("--model", required=True, choices=["trivia", "qwen"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    setup_logging(log_file=os.path.join(RESULTS_DIR, f"pipeline_{args.model}.log"))
    set_seed()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = get_device()

    # Load rotation classifier
    logger.info("Loading rotation classifier...")
    classifier = load_rotation_classifier(device)

    clf_transform = transforms.Compose([
        transforms.Resize(CLASSIFIER_RESIZE),
        transforms.CenterCrop(CLASSIFIER_CROP),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # Load benchmark
    baselines_module = _import_evaluators()
    entries = baselines_module.load_benchmark_meta()
    if args.limit:
        entries = entries[:args.limit]
        logger.info(f"Limited to {args.limit} samples")

    if args.model == "trivia":
        output_file = os.path.join(RESULTS_DIR, "pipeline_trivia.json")
    else:
        output_file = os.path.join(RESULTS_DIR, "pipeline_qwen.json")

    existing = {}
    if args.resume and os.path.exists(output_file):
        with open(output_file, "r") as f:
            existing = {r["index"]: r for r in json.load(f)}
        logger.info(f"Resuming: {len(existing)} samples already processed")

    processed_indices = set(existing.keys())

    # Phase 1: classify all rotation angles (fast, ~0.01s/sample)
    logger.info("Phase 1: Predicting rotation angles...")
    rotation_predictions = {}
    for entry in tqdm(entries, desc="Classifying"):
        if entry["index"] in processed_indices:
            continue
        img_path = os.path.join(BENCHMARK_DIR, "images", entry["filename"])
        image = Image.open(img_path).convert("RGB")
        rot_pred = predict_rotation(classifier, image, device, clf_transform)
        rotation_predictions[entry["index"]] = rot_pred

    rotation_correct_count = sum(
        1 for entry in entries
        if entry["index"] in rotation_predictions
        and rotation_predictions[entry["index"]]["predicted_angle"] == entry["rotation_angle"]
    )
    rotation_total = len(rotation_predictions)
    logger.info(f"Rotation accuracy: {rotation_correct_count}/{rotation_total}")

    # Unload classifier before loading VLM
    del classifier
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    import gc; gc.collect()
    logger.info("Classifier unloaded, cache cleared")

    # Phase 2: VLM inference on corrected images
    logger.info("Phase 2: Loading VLM and running inference...")
    if args.model == "trivia":
        ckpt = args.checkpoint or "opendatalab/TRivia-3B"
        evaluator = baselines_module.TRiviaEvaluator(checkpoint=ckpt, device=device)
    else:
        ckpt = args.checkpoint or "Qwen/Qwen2.5-VL-3B-Instruct"
        evaluator = baselines_module.QwenEvaluator(checkpoint=ckpt, device=device)

    evaluator.load_model()

    results = list(existing.values())
    errors = 0

    for entry in tqdm(entries, desc="Pipeline"):
        if entry["index"] in processed_indices:
            continue

        img_path = os.path.join(BENCHMARK_DIR, "images", entry["filename"])
        rot_pred = rotation_predictions[entry["index"]]

        try:
            image = Image.open(img_path).convert("RGB")
            corrected_image = correct_rotation(image, rot_pred["predicted_angle"])
            is_correct = rot_pred["predicted_angle"] == entry["rotation_angle"]

            vlm_result = evaluator.run_inference(corrected_image)

            results.append({
                "index": entry["index"],
                "filename": entry["filename"],
                "part": entry["part"],
                "rotation_angle": entry["rotation_angle"],
                "predicted_rotation": rot_pred["predicted_angle"],
                "rotation_confidence": rot_pred["confidence"],
                "rotation_probs": rot_pred["all_probs"],
                "rotation_correct": is_correct,
                "pred_html": vlm_result["pred_html"],
                "raw_output": vlm_result["raw_output"],
                "inference_time_s": vlm_result["inference_time_s"],
            })

        except Exception as e:
            logger.error(f"Error processing {entry['filename']}: {e}")
            errors += 1
            results.append({
                "index": entry["index"],
                "filename": entry["filename"],
                "part": entry["part"],
                "rotation_angle": entry["rotation_angle"],
                "predicted_rotation": -1,
                "rotation_confidence": 0.0,
                "rotation_correct": False,
                "pred_html": "",
                "raw_output": f"ERROR: {str(e)}",
                "inference_time_s": 0.0,
            })

        if len(results) % CHECKPOINT_EVERY == 0:
            baselines_module.save_results(results, output_file)

    baselines_module.save_results(results, output_file)

    rot_acc = rotation_correct_count / max(1, rotation_total)
    non_empty = sum(1 for r in results if r.get("pred_html", "").strip())
    rotation_correct_count = sum(1 for r in results if r.get("rotation_correct"))

    print(f"\n{'='*60}")
    print(f"  Pipeline complete: {len(results)} samples")
    print(f"{'='*60}")
    print(f"  Rotation classifier accuracy: {rotation_correct_count}/{rotation_total} ({rot_acc:.4f})")
    print(f"  Non-empty predictions: {non_empty}/{len(results)}")
    print(f"  Errors: {errors}")
    print(f"  Results saved to: {output_file}")


if __name__ == "__main__":
    main()
