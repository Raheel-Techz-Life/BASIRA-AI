# Architecture — Basira AI

## System diagram (current, implemented state)

```
                        ┌─────────────────────────┐
                        │   Kaggle MRI dataset     │
                        │  (manual download only;  │
                        │   not automated — needs  │
                        │   user's own Kaggle acct)│
                        └────────────┬────────────┘
                                     │
                     data/raw/brain_tumor_mri/{Training,Testing}/
                                     │
                                     ▼
                        ┌─────────────────────────┐
                        │ src/data/dataset.py      │  validate + manifest
                        └────────────┬────────────┘
                                     ▼
                        ┌─────────────────────────┐
                        │ src/data/preprocessing.py│  grayscale, resize,
                        └────────────┬────────────┘  normalize
                                     ▼
                        ┌─────────────────────────┐
                        │src/data/feature_extraction│ 12 features
                        └────────────┬────────────┘  (intensity/GLCM/shape)
                                     ▼
                ┌────────────────────┴────────────────────┐
                ▼                                           ▼
   ┌─────────────────────────┐              ┌─────────────────────────┐
   │ src/models/train.py      │              │ src/models/predict.py   │
   │ (offline, run once)      │              │ (used by app.py)        │
   │ - fit StandardScaler     │   joblib     │ - load saved scaler +   │
   │ - compare 4 models       │ ───────────▶ │   model + metadata      │
   │ - pick best by val F1    │  artifacts   │ - run the SAME          │
   │ - eval on held-out Test  │              │   preprocess + feature  │
   └────────────┬────────────┘              │   extraction functions  │
                ▼                            └────────────┬────────────┘
   ┌─────────────────────────┐                            ▼
   │ models/checkpoints/      │              ┌─────────────────────────┐
   │  tumor_classifier.joblib │              │        app.py           │
   │  feature_scaler.joblib   │              │  Streamlit UI:          │
   │  model_metadata.json     │              │  upload → preview →     │
   └─────────────────────────┘              │  features → prediction  │
                                             └─────────────────────────┘
```

## Why this shape

- **Train and inference share code, not duplicate it.** `preprocess_image()`
  and `extract_features()` are called identically by `train.py` (batch, from
  disk) and `predict.py` (single image, from an upload). This is the master
  prompt's Section 7 requirement — it rules out an entire class of
  train/serve skew bugs by construction.
- **The model artifacts are the only coupling between training and the
  app.** `app.py` never re-implements scoring logic; it calls
  `src/models/predict.py`, which loads whatever `train.py` last saved. Retrain
  and the app picks up the new model with no code change.
- **Manifests are a paper trail, not just a file list.** `build_manifest()`
  records *why* a file was excluded (unreadable, too small, wrong format)
  rather than silently dropping it, so dataset quality is auditable.

## Not yet in this diagram

- A future-risk model and its training path — doesn't exist, because no
  valid dataset exists for it (`docs/dataset_feasibility.md`). If one is
  obtained later (e.g. via a UK Biobank access application), it plugs in
  as a parallel branch reusing the same `preprocessing.py` /
  `feature_extraction.py` modules, with its own `train_future_risk.py` and
  its own patient-level splitting logic (not image-level, unlike the
  current demo — see `docs/limitations.md`).
- NIfTI/DICOM volume support — `nibabel` is installed and listed as a
  dependency for this reason, but no code path uses it yet; the current
  demo only handles 2D JPG/PNG slices.
- Any deployment/hosting layer — this is a local Streamlit prototype only.
