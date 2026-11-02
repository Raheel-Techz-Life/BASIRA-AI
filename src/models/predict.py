"""Load the trained educational classifier and run inference on one image.

Used by both the Streamlit app and tests, so training and inference always
go through the identical preprocessing + feature extraction code path
(master prompt Section 7).
"""
import json
from pathlib import Path

import joblib
import numpy as np

from src.data.feature_extraction import FEATURE_NAMES, extract_features
from src.data.preprocessing import load_and_preprocess
from src.utils.config import MODELS_DIR


class TumorClassifierUnavailable(RuntimeError):
    pass


def load_model():
    model_path = MODELS_DIR / "tumor_classifier.joblib"
    scaler_path = MODELS_DIR / "feature_scaler.joblib"
    meta_path = MODELS_DIR / "model_metadata.json"
    if not (model_path.exists() and scaler_path.exists() and meta_path.exists()):
        raise TumorClassifierUnavailable(
            "No trained model found. Run `python -m src.models.train` first."
        )
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(meta_path) as f:
        metadata = json.load(f)
    return model, scaler, metadata


def predict_image(path_or_array, model, scaler, metadata) -> dict:
    """path_or_array: a filesystem path to an image."""
    image = load_and_preprocess(path_or_array)
    feats = extract_features(image)
    x = np.array([[feats[name] for name in metadata["feature_names"]]])
    x_scaled = scaler.transform(x)

    pred = model.predict(x_scaled)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba_arr = model.predict_proba(x_scaled)[0]
        proba = {cls: float(p) for cls, p in zip(model.classes_, proba_arr)}

    return {
        "predicted_label": pred,
        "class_probabilities": proba,
        "features": feats,
        "preprocessed_image": image,
    }
