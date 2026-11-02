import numpy as np
from PIL import Image

from src.data.preprocessing import preprocess_image


def test_preprocess_resizes_and_normalizes():
    img = Image.fromarray((np.random.rand(64, 64) * 255).astype(np.uint8))
    out = preprocess_image(img, size=(32, 32))
    assert out.shape == (32, 32)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_preprocess_constant_image_does_not_divide_by_zero():
    img = Image.fromarray(np.full((16, 16), 100, dtype=np.uint8))
    out = preprocess_image(img, size=(16, 16))
    assert np.all(out == 0.0)
