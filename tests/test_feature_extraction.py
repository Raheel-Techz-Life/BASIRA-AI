import numpy as np

from src.data.feature_extraction import FEATURE_NAMES, extract_features


def test_extract_features_returns_all_expected_keys():
    image = np.random.rand(32, 32).astype(np.float32)
    feats = extract_features(image)
    assert set(feats.keys()) == set(FEATURE_NAMES)
    for v in feats.values():
        assert np.isfinite(v)


def test_uniform_image_has_zero_contrast():
    image = np.full((32, 32), 0.5, dtype=np.float32)
    feats = extract_features(image)
    assert feats["glcm_contrast"] == 0.0
    assert feats["intensity_std"] == 0.0
