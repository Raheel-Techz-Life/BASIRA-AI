"""Central paths and constants for Basira AI."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_MANIFESTS = PROJECT_ROOT / "data" / "manifests"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"
EXPERIMENTS_RESULTS = PROJECT_ROOT / "experiments" / "results"
EXPERIMENTS_FIGURES = PROJECT_ROOT / "experiments" / "figures"

# Educational existing-tumor-detection demo dataset (NOT the future-risk task).
BRAIN_TUMOR_MRI_DIR = DATA_RAW / "brain_tumor_mri"
BRAIN_TUMOR_CLASSES = ["notumor", "glioma", "meningioma", "pituitary"]

IMAGE_SIZE = (128, 128)  # resize target for feature extraction
RANDOM_SEED = 42
