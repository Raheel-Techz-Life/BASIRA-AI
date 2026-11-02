from pathlib import Path

import numpy as np
from PIL import Image

from src.data.dataset import validate_image_file


def test_valid_image_passes(tmp_path: Path):
    p = tmp_path / "ok.jpg"
    Image.fromarray((np.random.rand(32, 32) * 255).astype(np.uint8)).save(p)
    result = validate_image_file(p)
    assert result.is_valid


def test_unsupported_extension_rejected(tmp_path: Path):
    p = tmp_path / "scan.nii"
    p.write_bytes(b"not a real nifti")
    result = validate_image_file(p)
    assert not result.is_valid
    assert "extension" in result.reason


def test_corrupt_file_rejected(tmp_path: Path):
    p = tmp_path / "corrupt.jpg"
    p.write_bytes(b"this is not a jpeg")
    result = validate_image_file(p)
    assert not result.is_valid


def test_tiny_image_rejected(tmp_path: Path):
    p = tmp_path / "tiny.png"
    Image.fromarray((np.random.rand(4, 4) * 255).astype(np.uint8)).save(p)
    result = validate_image_file(p)
    assert not result.is_valid
    assert "small" in result.reason
