"""End-to-end inference test against the actual trained model artifacts.

Skips cleanly if the model hasn't been trained yet (e.g. a fresh clone
before the dataset is downloaded), rather than failing the whole suite.
"""
import pytest

from src.models.predict import TumorClassifierUnavailable, load_model, predict_image
from src.utils.config import BRAIN_TUMOR_MRI_DIR


def test_predict_on_a_real_test_image():
    try:
        model, scaler, metadata = load_model()
    except TumorClassifierUnavailable:
        pytest.skip("Model not trained yet — run `python -m src.models.train` first.")

    sample_dir = BRAIN_TUMOR_MRI_DIR / "Testing" / "notumor"
    if not sample_dir.exists():
        pytest.skip("Dataset not present.")
    sample_path = next(sample_dir.iterdir())

    result = predict_image(sample_path, model, scaler, metadata)
    assert result["predicted_label"] in metadata["classes"]
    assert result["class_probabilities"] is not None
    assert abs(sum(result["class_probabilities"].values()) - 1.0) < 1e-3
