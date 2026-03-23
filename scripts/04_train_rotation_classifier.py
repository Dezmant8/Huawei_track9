"""Train rotation angle classifier (0/90/180/270) on SynthTabNet images.

Supports ResNet-18/34 and EfficientNet-B0 via --arch flag.
Includes early stopping, per-class metrics, and confusion matrix.

Usage:
    python scripts/04_train_rotation_classifier.py
    python scripts/04_train_rotation_classifier.py --arch resnet34 --epochs 15
    python scripts/04_train_rotation_classifier.py --max-per-part 1000
"""

import os
import sys
import re
import json
import random
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    SEED, PARTS, ANGLES, NUM_ROTATION_CLASSES, RAW_DIR, MODEL_DIR,
    CLASSIFIER_LR, CLASSIFIER_EPOCHS, CLASSIFIER_BATCH_SIZE,
    CLASSIFIER_PATIENCE, CLASSIFIER_RESIZE, CLASSIFIER_CROP,
    CLASSIFIER_TRAIN_PER_PART, CLASSIFIER_VAL_PER_PART,
    IMAGENET_MEAN, IMAGENET_STD,
)
from utils.rotation import rotate_image
from utils.seed import set_seed, get_device, setup_logging

logger = logging.getLogger(__name__)


class RotatedTableDataset(Dataset):
    """Dataset that creates 4 rotated variants per original image."""

    def __init__(self, image_paths: list, transform=None):
        self.samples = []
        for path in image_paths:
            for label, angle in enumerate(ANGLES):
                self.samples.append((path, label, angle))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, angle = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = rotate_image(img, angle)
        if self.transform:
            img = self.transform(img)
        return img, label


def collect_image_paths(split: str, max_per_part: int) -> list:
    """Collect image paths from SynthTabNet for given split, with subsampling."""
    all_paths = []
    for part in PARTS:
        jsonl_path = os.path.join(RAW_DIR, part, "synthetic_data.jsonl")
        part_paths = []
        with open(jsonl_path, "r") as f:
            for line in f:
                raw = line.strip()
                # Fix trailing commas in SynthTabNet JSON
                raw = re.sub(r",\s*([}\]])", r"\1", raw)
                entry = json.loads(raw)
                if entry["split"] == split:
                    img_path = os.path.join(RAW_DIR, part, "images", split, entry["filename"])
                    if os.path.exists(img_path):
                        part_paths.append(img_path)

        if len(part_paths) > max_per_part:
            rng = random.Random(SEED)
            part_paths = rng.sample(part_paths, max_per_part)

        logger.info(f"  {part} ({split}): {len(part_paths)} images")
        all_paths.extend(part_paths)

    return all_paths


SUPPORTED_ARCHS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, 512),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1, 512),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1, 1280),
}


def build_model(arch: str, num_classes: int = NUM_ROTATION_CLASSES) -> nn.Module:
    """Build classifier from pretrained backbone with replaced head."""
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"Unsupported architecture: {arch}. Choose from {list(SUPPORTED_ARCHS)}")

    model_fn, weights, num_features = SUPPORTED_ARCHS[arch]
    model = model_fn(weights=weights)

    if arch.startswith("resnet"):
        model.fc = nn.Linear(num_features, num_classes)
    elif arch.startswith("efficientnet"):
        model.classifier[-1] = nn.Linear(num_features, num_classes)

    logger.info(f"Built {arch} with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def train_one_epoch(model, loader, criterion, optimizer, device, max_grad_norm=1.0):
    """Single training epoch with gradient clipping."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model. Returns loss, accuracy, confusion matrix, per-class metrics."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    confusion = torch.zeros(NUM_ROTATION_CLASSES, NUM_ROTATION_CLASSES, dtype=torch.long)

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Eval", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += images.size(0)

            for t, p in zip(labels, predicted):
                confusion[t.item()][p.item()] += 1

    accuracy = correct / total
    loss = total_loss / total

    per_class = {}
    for i, angle in enumerate(ANGLES):
        tp = confusion[i][i].item()
        fp = confusion[:, i].sum().item() - tp
        fn = confusion[i, :].sum().item() - tp
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        per_class[angle] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": confusion[i, :].sum().item(),
        }

    return loss, accuracy, confusion, per_class


def print_confusion_matrix(confusion, angles=ANGLES):
    print(f"\n  Confusion matrix (rows=true, cols=predicted):")
    print(f"  {'':>8}", end="")
    for a in angles:
        print(f"{a:>8}", end="")
    print()
    for i, a in enumerate(angles):
        print(f"  {a:>7}", end="")
        for j in range(len(angles)):
            print(f"{confusion[i][j].item():>9}", end="")
        print()


def save_confusion_matrix_plot(confusion, angles, save_path):
    """Save confusion matrix as a heatmap image."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        _, ax = plt.subplots(figsize=(6, 5))
        cm_np = confusion.numpy()
        cm_norm = cm_np.astype(float) / cm_np.sum(axis=1, keepdims=True)

        sns.heatmap(
            cm_norm, annot=True, fmt=".2%", cmap="Blues",
            xticklabels=[f"{a}" for a in angles],
            yticklabels=[f"{a}" for a in angles],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Rotation Classifier Confusion Matrix")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Confusion matrix plot saved to {save_path}")
    except ImportError:
        logger.warning("matplotlib/seaborn not installed, skipping plot")


def main():
    parser = argparse.ArgumentParser(
        description="Train rotation angle classifier for table images"
    )
    parser.add_argument("--arch", type=str, default="resnet18",
                        choices=list(SUPPORTED_ARCHS.keys()))
    parser.add_argument("--epochs", type=int, default=CLASSIFIER_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=CLASSIFIER_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=CLASSIFIER_LR)
    parser.add_argument("--patience", type=int, default=CLASSIFIER_PATIENCE)
    parser.add_argument("--max-per-part", type=int, default=CLASSIFIER_TRAIN_PER_PART)
    parser.add_argument("--val-max-per-part", type=int, default=CLASSIFIER_VAL_PER_PART)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    setup_logging(log_file=os.path.join(MODEL_DIR, "training.log"))
    set_seed(SEED)

    device = get_device()
    logger.info(f"Config: arch={args.arch}, epochs={args.epochs}, lr={args.lr}, "
                f"batch_size={args.batch_size}, device={device}")

    # No RandomHorizontalFlip — it would break orientation labels
    train_transform = transforms.Compose([
        transforms.Resize(CLASSIFIER_RESIZE),
        transforms.RandomResizedCrop(CLASSIFIER_CROP, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(CLASSIFIER_RESIZE),
        transforms.CenterCrop(CLASSIFIER_CROP),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    logger.info("Collecting training images...")
    train_paths = collect_image_paths("train", args.max_per_part)
    logger.info(f"Total train: {len(train_paths)} originals x 4 = {len(train_paths)*4} samples")

    logger.info("Collecting validation images...")
    val_paths = collect_image_paths("val", args.val_max_per_part)
    logger.info(f"Total val: {len(val_paths)} originals x 4 = {len(val_paths)*4} samples")

    train_dataset = RotatedTableDataset(train_paths, transform=train_transform)
    val_dataset = RotatedTableDataset(val_paths, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = build_model(args.arch).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(MODEL_DIR, exist_ok=True)

    logger.info("Trivial baseline (always predict 0): accuracy = 0.2500")

    best_acc = 0.0
    epochs_without_improvement = 0
    training_log = []

    logger.info(f"Training for {args.epochs} epochs (early stopping patience={args.patience})...")

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _, per_class = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch+1}/{args.epochs}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={lr:.6f}"
        )

        training_log.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 5),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 5),
            "lr": lr,
            "per_class": per_class,
        })

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pth"))
            logger.info(f"  New best model (val_acc={val_acc:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                logger.info(f"  Early stopping: no improvement for {args.patience} epochs")
                break

    # Final report
    print(f"\n{'='*60}")
    print(f"  Training complete")
    print(f"{'='*60}")
    print(f"  Architecture: {args.arch}")
    print(f"  Best validation accuracy: {best_acc:.4f}")
    print(f"  Trivial baseline: 0.2500")
    print(f"  Improvement over baseline: +{best_acc - 0.25:.4f}")

    model.load_state_dict(
        torch.load(os.path.join(MODEL_DIR, "best_model.pth"),
                    map_location=device, weights_only=True)
    )
    _, _, final_confusion, final_per_class = evaluate(
        model, val_loader, criterion, device
    )

    print_confusion_matrix(final_confusion)

    print(f"\n  Per-class metrics:")
    print(f"  {'Angle':>8}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Support':>8}")
    print(f"  {'-'*48}")
    for angle in ANGLES:
        m = final_per_class[angle]
        print(f"  {angle:>7}  {m['precision']:>10.4f}  {m['recall']:>8.4f}  "
              f"{m['f1']:>8.4f}  {m['support']:>8}")

    save_confusion_matrix_plot(
        final_confusion, ANGLES,
        os.path.join(MODEL_DIR, "confusion_matrix.png")
    )

    log_data = {
        "config": {
            "arch": args.arch,
            "epochs_trained": len(training_log),
            "lr": args.lr,
            "batch_size": args.batch_size,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "seed": SEED,
            "device": device,
        },
        "best_val_acc": best_acc,
        "trivial_baseline_acc": 0.25,
        "final_per_class": final_per_class,
        "training_log": training_log,
    }
    log_path = os.path.join(MODEL_DIR, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    logger.info(f"Training log saved to {log_path}")


if __name__ == "__main__":
    main()
