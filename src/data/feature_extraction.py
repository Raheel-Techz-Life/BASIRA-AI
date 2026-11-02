"""Quantitative feature extraction from a preprocessed MRI slice.

Feature families and justification (master prompt Section 6):

- Intensity statistics: mean/std/percentiles summarize overall signal
  distribution; large masses often shift local intensity relative to normal
  tissue.
- GLCM texture (contrast, homogeneity, energy, correlation): captures
  tissue heterogeneity — tumors frequently present as texturally irregular
  regions compared to normal parenchyma.
- Simple morphology (foreground area fraction via Otsu threshold, centroid
  offset from image center): a coarse proxy for asymmetric mass effect,
  cheap to compute and to sanity-check by eye.

These are NOT radiomics-grade features (no PyRadiomics here — out of scope
for a 2-day sprint) and are not claimed to be sufficient for any clinical
task. They exist to make the pipeline mechanically real and inspectable.
"""
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import threshold_otsu

FEATURE_NAMES = [
    "intensity_mean",
    "intensity_std",
    "intensity_p10",
    "intensity_p50",
    "intensity_p90",
    "glcm_contrast",
    "glcm_homogeneity",
    "glcm_energy",
    "glcm_correlation",
    "foreground_area_fraction",
    "centroid_offset_x",
    "centroid_offset_y",
]


def extract_features(image: np.ndarray) -> dict:
    """image: 2D float32 array in [0, 1], as produced by preprocess_image()."""
    flat = image.flatten()
    p10, p50, p90 = np.percentile(flat, [10, 50, 90])

    gray_u8 = (image * 255).astype(np.uint8)
    glcm = graycomatrix(
        gray_u8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True
    )
    contrast = graycoprops(glcm, "contrast")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    energy = graycoprops(glcm, "energy")[0, 0]
    correlation = graycoprops(glcm, "correlation")[0, 0]

    try:
        thresh = threshold_otsu(image)
        mask = image > thresh
    except ValueError:
        mask = image > image.mean()

    area_fraction = float(mask.mean())
    h, w = image.shape
    if mask.any():
        ys, xs = np.nonzero(mask)
        centroid_y, centroid_x = ys.mean(), xs.mean()
    else:
        centroid_y, centroid_x = h / 2, w / 2
    centroid_offset_x = float((centroid_x - w / 2) / w)
    centroid_offset_y = float((centroid_y - h / 2) / h)

    return {
        "intensity_mean": float(flat.mean()),
        "intensity_std": float(flat.std()),
        "intensity_p10": float(p10),
        "intensity_p50": float(p50),
        "intensity_p90": float(p90),
        "glcm_contrast": float(contrast),
        "glcm_homogeneity": float(homogeneity),
        "glcm_energy": float(energy),
        "glcm_correlation": float(correlation),
        "foreground_area_fraction": area_fraction,
        "centroid_offset_x": centroid_offset_x,
        "centroid_offset_y": centroid_offset_y,
    }
