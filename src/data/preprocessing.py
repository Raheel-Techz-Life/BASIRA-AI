"""Image loading and preprocessing shared by training and inference.

Fit-free, deterministic transforms only (resize, grayscale, intensity
normalization) so the exact same function can run on training data and on a
single uploaded image at inference time (master prompt Section 7).
"""
from pathlib import Path

import numpy as np
from PIL import Image

from src.utils.config import IMAGE_SIZE


def load_image(path: Path | str) -> Image.Image:
    """Load an image file and convert to 8-bit grayscale."""
    img = Image.open(path)
    img = img.convert("L")
    return img


def preprocess_image(img: Image.Image, size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Resize and intensity-normalize a grayscale image to a float32 array in [0, 1].

    Normalization is min-max per image. This is appropriate for 8-bit JPEG/PNG
    MRI renderings (already windowed by the source). It would NOT be
    appropriate for raw NIfTI intensities without proper modality-aware
    normalization — flagged here for future NIfTI support.
    """
    img = img.resize(size, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    min_val, max_val = arr.min(), arr.max()
    if max_val > min_val:
        arr = (arr - min_val) / (max_val - min_val)
    else:
        arr = np.zeros_like(arr)
    return arr


def load_and_preprocess(path: Path | str, size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    return preprocess_image(load_image(path), size=size)
