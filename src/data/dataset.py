"""
Manifest building and validation for the educational existing-tumor-detection
dataset (Kaggle "Brain Tumor MRI Dataset", masoudnickparvar).

This is NOT the future-tumor-risk dataset. See docs/dataset_feasibility.md.

Expected layout after manual download+unzip:
    data/raw/brain_tumor_mri/Training/{glioma,meningioma,notumor,pituitary}/*.jpg
    data/raw/brain_tumor_mri/Testing/{glioma,meningioma,notumor,pituitary}/*.jpg

NOTE ON LEAKAGE: this public dataset ships as independent image files with no
patient identifier, so true patient-level splitting (master prompt Section 5E)
is not possible here. We use the dataset's own Training/Testing split and
additionally split Training into train/validation. This is a documented
limitation of the educational demo only (see docs/limitations.md) and does
not apply to the (not-yet-built) future-risk model, which will require real
patient IDs before any split is performed.
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

from src.utils.config import BRAIN_TUMOR_CLASSES, BRAIN_TUMOR_MRI_DIR
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class ValidationResult:
    path: Path
    is_valid: bool
    reason: str = ""


def validate_image_file(path: Path) -> ValidationResult:
    """Check a single MRI image file is readable and non-trivial."""
    if path.suffix.lower() not in VALID_EXTENSIONS:
        return ValidationResult(path, False, f"unsupported extension {path.suffix}")
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            width, height = img.size
            if width < 16 or height < 16:
                return ValidationResult(path, False, "image too small")
    except (UnidentifiedImageError, OSError) as exc:
        return ValidationResult(path, False, f"unreadable: {exc}")
    return ValidationResult(path, True)


def build_manifest(split: str = "Training") -> pd.DataFrame:
    """Scan data/raw/brain_tumor_mri/<split>/<class>/*.jpg and validate each file.

    Returns a DataFrame with columns: path, label, split, is_valid, reason.
    Invalid files are kept in the manifest (flagged) rather than silently
    dropped, so the feasibility/quality of the raw data is auditable.
    """
    split_dir = BRAIN_TUMOR_MRI_DIR / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Expected dataset split not found: {split_dir}. "
            f"Download the Kaggle 'Brain Tumor MRI Dataset' (masoudnickparvar) "
            f"and unzip it into {BRAIN_TUMOR_MRI_DIR}."
        )

    rows = []
    for label in BRAIN_TUMOR_CLASSES:
        class_dir = split_dir / label
        if not class_dir.exists():
            logger.warning("Missing expected class folder: %s", class_dir)
            continue
        for path in sorted(class_dir.iterdir()):
            if not path.is_file():
                continue
            result = validate_image_file(path)
            rows.append(
                {
                    "path": str(path),
                    "label": label,
                    "split": split,
                    "is_valid": result.is_valid,
                    "reason": result.reason,
                }
            )

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError(f"No image files found under {split_dir}")

    n_invalid = (~manifest["is_valid"]).sum()
    logger.info(
        "Manifest built for split=%s: %d files, %d invalid", split, len(manifest), n_invalid
    )
    return manifest


def save_manifest(manifest: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_path, index=False)
    logger.info("Saved manifest to %s", out_path)
