# Limitations — Basira AI (current state)

## Research-level limitations (fundamental, not fixable by Wednesday)

1. **No future-risk model exists.** No accessible public dataset links a
   verified tumor-free baseline MRI to a verified future tumor diagnosis.
   See `docs/dataset_feasibility.md`. Everything the app shows under
   "Educational Baseline Demo" is a different task (see below).
2. **The academic title does not match the implemented work.** "Reliable
   Brain Tumor Segmentation from MRI Using Deep Learning" describes neither
   future-risk prediction (the stated research question) nor existing-tumor
   *classification* (what is actually implemented). Flagged for mentor
   discussion, not silently resolved.

## Educational baseline demo limitations

1. **No patient identifiers.** The Kaggle "Brain Tumor MRI Dataset" ships as
   independent image files with no patient ID, so true patient-level
   train/test splitting (required by the master prompt for the real
   future-risk task) is not possible here. The demo uses the dataset's own
   Training/Testing split plus a stratified image-level validation split.
   If any images in this dataset are in fact multiple slices from the same
   patient, this could optimistically bias validation results. This is a
   property of the public dataset, not of our code.
2. **This classifies an already-visible tumor, not a future one.** A
   correct prediction here says nothing about whether an initially
   tumor-free scan predicts later tumor development.
3. **Feature set is intentionally small.** Intensity statistics + GLCM
   texture + simple morphology — not clinical-grade radiomics (no
   PyRadiomics), chosen for a 2-day timeline. Documented in
   `src/data/feature_extraction.py`.
4. **No calibration analysis performed** (master prompt Section 8) given
   the timeline — only accuracy, macro-F1, per-class precision/recall, and
   confusion matrix are reported. Treat any class-probability output as an
   uncalibrated model score, not a true probability.
5. **No clinical validation.** This is a software/ML-methodology
   demonstration only.

## What is genuinely tested as of this writing

- Image validation (corrupt file, wrong extension, too-small image): unit
  tested, **passing** (`tests/test_dataset.py`).
- Preprocessing determinism and normalization edge cases: unit tested,
  **passing** (`tests/test_preprocessing.py`).
- Feature extraction correctness on synthetic images: unit tested,
  **passing** (`tests/test_feature_extraction.py`).
- End-to-end training/evaluation on the real Kaggle dataset: **pending** —
  requires the dataset to be manually downloaded and placed under
  `data/raw/brain_tumor_mri/` (Kaggle requires account credentials Claude
  cannot and should not handle).
