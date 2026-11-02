"""Train and evaluate the optional CNN comparison model.

Uses the identical build_manifest() + load_and_preprocess() pipeline as the
traditional-ML baseline (src/models/train.py), and the same random_state /
test_size / stratify split, so the Training-split validation comparison
between the two model families is apples-to-apples (same held-out images).

Usage:
    python -m src.models.train_cnn
"""
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import build_manifest
from src.data.preprocessing import load_and_preprocess
from src.evaluation.metrics import compute_metrics, format_metrics_summary
from src.models.cnn import SimpleCNN
from src.utils.config import (
    BRAIN_TUMOR_CLASSES,
    EXPERIMENTS_RESULTS,
    MODELS_DIR,
    RANDOM_SEED,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

LABEL_TO_IDX = {label: i for i, label in enumerate(BRAIN_TUMOR_CLASSES)}
EPOCHS = 6
BATCH_SIZE = 64
LR = 1e-3


class ArrayDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx]).unsqueeze(0)  # (1, H, W)
        label = int(self.labels[idx])
        return img, label


def load_images(manifest) -> tuple[np.ndarray, np.ndarray]:
    """Preload every valid image via the shared preprocessing pipeline."""
    valid = manifest[manifest["is_valid"]].reset_index(drop=True)
    images = np.zeros((len(valid), 128, 128), dtype=np.float32)
    labels = np.zeros(len(valid), dtype=np.int64)
    for i, row in valid.iterrows():
        images[i] = load_and_preprocess(row["path"])
        labels[i] = LABEL_TO_IDX[row["label"]]
        if (i + 1) % 1000 == 0:
            logger.info("Loaded %d/%d images", i + 1, len(valid))
    return images, labels


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * images.size(0)
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, all_preds, all_labels


def main():
    device = "cpu"
    logger.info("Building manifests...")
    train_manifest = build_manifest("Training")
    test_manifest = build_manifest("Testing")

    logger.info("Preloading training images...")
    t0 = time.time()
    X_all, y_all = load_images(train_manifest)
    logger.info("Loaded %d training images in %.1fs", len(X_all), time.time() - t0)

    logger.info("Preloading testing images...")
    X_test, y_test = load_images(test_manifest)
    logger.info("Loaded %d testing images", len(X_test))

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=RANDOM_SEED, stratify=y_all
    )

    train_loader = DataLoader(ArrayDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ArrayDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(ArrayDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    torch.manual_seed(RANDOM_SEED)
    model = SimpleCNN(num_classes=len(BRAIN_TUMOR_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = []
    best_val_f1 = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_preds, val_labels = run_epoch(model, val_loader, criterion, None, device)

        val_preds_named = [BRAIN_TUMOR_CLASSES[p] for p in val_preds]
        val_labels_named = [BRAIN_TUMOR_CLASSES[l] for l in val_labels]
        val_metrics = compute_metrics(val_labels_named, val_preds_named, BRAIN_TUMOR_CLASSES)

        elapsed = time.time() - t0
        logger.info(
            "Epoch %d/%d: train_loss=%.4f val_loss=%.4f val_acc=%.3f val_macro_f1=%.3f (%.1fs)",
            epoch, EPOCHS, train_loss, val_loss, val_metrics["accuracy"], val_metrics["macro_f1"], elapsed,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    # Final evaluation on the held-out Testing split, never used for training
    # or model selection above.
    _, test_preds, test_labels = run_epoch(model, test_loader, criterion, None, device)
    test_preds_named = [BRAIN_TUMOR_CLASSES[p] for p in test_preds]
    test_labels_named = [BRAIN_TUMOR_CLASSES[l] for l in test_labels]
    test_metrics = compute_metrics(test_labels_named, test_preds_named, BRAIN_TUMOR_CLASSES)

    print("\n=== CNN held-out Testing split results (best-val-F1 checkpoint) ===")
    print(format_metrics_summary(test_metrics))

    EXPERIMENTS_RESULTS.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENTS_RESULTS / "cnn_test_results.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    with open(EXPERIMENTS_RESULTS / "cnn_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "cnn_classifier.pt")
    with open(MODELS_DIR / "cnn_model_metadata.json", "w") as f:
        json.dump(
            {
                "model_name": "simple_cnn",
                "classes": BRAIN_TUMOR_CLASSES,
                "input": "1x128x128 grayscale, same preprocessing as the RF baseline",
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "best_val_macro_f1": best_val_f1,
                "task": "educational_existing_tumor_classification",
                "NOT": "future_tumor_risk_prediction",
            },
            f,
            indent=2,
        )
    logger.info("Saved CNN artifacts to %s", MODELS_DIR)


if __name__ == "__main__":
    main()
