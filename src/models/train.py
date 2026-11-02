"""Train and compare traditional-ML baselines on the educational
existing-tumor-detection task (NOT the future-risk task — see
docs/dataset_feasibility.md).

Usage:
    python -m src.models.train
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data.dataset import build_manifest, save_manifest
from src.data.feature_extraction import FEATURE_NAMES, extract_features
from src.data.preprocessing import load_and_preprocess
from src.evaluation.metrics import compute_metrics, format_metrics_summary
from src.models.baseline import MajorityClassBaseline
from src.utils.config import (
    BRAIN_TUMOR_CLASSES,
    DATA_MANIFESTS,
    DATA_PROCESSED,
    EXPERIMENTS_RESULTS,
    MODELS_DIR,
    RANDOM_SEED,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def extract_feature_matrix(manifest: pd.DataFrame) -> pd.DataFrame:
    """Load every valid image in the manifest and extract its feature vector."""
    valid = manifest[manifest["is_valid"]].reset_index(drop=True)
    rows = []
    for i, row in valid.iterrows():
        image = load_and_preprocess(row["path"])
        feats = extract_features(image)
        feats["label"] = row["label"]
        feats["path"] = row["path"]
        rows.append(feats)
        if (i + 1) % 500 == 0:
            logger.info("Extracted features for %d/%d images", i + 1, len(valid))
    return pd.DataFrame(rows)


def main():
    logger.info("Building manifests...")
    train_manifest = build_manifest("Training")
    test_manifest = build_manifest("Testing")
    save_manifest(train_manifest, DATA_MANIFESTS / "training_manifest.csv")
    save_manifest(test_manifest, DATA_MANIFESTS / "testing_manifest.csv")

    logger.info("Extracting features (Training)...")
    train_features = extract_feature_matrix(train_manifest)
    logger.info("Extracting features (Testing)...")
    test_features = extract_feature_matrix(test_manifest)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    train_features.to_csv(DATA_PROCESSED / "train_features.csv", index=False)
    test_features.to_csv(DATA_PROCESSED / "test_features.csv", index=False)

    # .to_numpy(dtype=...) forces plain numpy arrays: pandas 3.x backs string
    # columns with PyArrow by default, and sklearn's fancy-indexing in
    # train_test_split does not support ArrowStringArray.
    X_all = train_features[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y_all = train_features["label"].to_numpy(dtype=object)

    # Image-level split for model selection (no patient ID available in this
    # public dataset — see docs/limitations.md). The held-out Testing split
    # from Kaggle is used as the final, untouched evaluation set.
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=RANDOM_SEED, stratify=y_all
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)

    candidates = {
        "majority_baseline": MajorityClassBaseline(),
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED),
        "svm_rbf": SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED),
    }

    results = {}
    for name, model in candidates.items():
        model.fit(X_train_s if name != "majority_baseline" else X_train, y_train)
        preds = model.predict(X_val_s if name != "majority_baseline" else X_val)
        metrics = compute_metrics(y_val, preds, BRAIN_TUMOR_CLASSES)
        results[name] = metrics
        logger.info("[%s] val macro_f1=%.3f acc=%.3f", name, metrics["macro_f1"], metrics["accuracy"])

    best_name = max(
        (n for n in results if n != "majority_baseline"),
        key=lambda n: results[n]["macro_f1"],
    )
    logger.info("Best model on validation: %s", best_name)

    best_model = candidates[best_name]
    X_test = test_features[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y_test = test_features["label"].to_numpy(dtype=object)
    X_test_s = scaler.transform(X_test)
    test_preds = best_model.predict(X_test_s)
    test_metrics = compute_metrics(y_test, test_preds, BRAIN_TUMOR_CLASSES)

    print("\n=== Validation results (all candidates) ===")
    for name, m in results.items():
        print(f"\n[{name}]")
        print(format_metrics_summary(m))

    print(f"\n=== Held-out Testing split results ({best_name}) ===")
    print(format_metrics_summary(test_metrics))

    EXPERIMENTS_RESULTS.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENTS_RESULTS / "validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(EXPERIMENTS_RESULTS / "test_results.json", "w") as f:
        json.dump({"model": best_name, **test_metrics}, f, indent=2)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "tumor_classifier.joblib")
    joblib.dump(scaler, MODELS_DIR / "feature_scaler.joblib")
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(
            {
                "model_name": best_name,
                "feature_names": FEATURE_NAMES,
                "classes": BRAIN_TUMOR_CLASSES,
                "task": "educational_existing_tumor_classification",
                "NOT": "future_tumor_risk_prediction",
            },
            f,
            indent=2,
        )
    logger.info("Saved model artifacts to %s", MODELS_DIR)


if __name__ == "__main__":
    main()
