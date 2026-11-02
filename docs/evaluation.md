# Evaluation Results — Educational Baseline Demo

**These are real results from an actual run**, `python -m src.models.train`,
2026-11-02 19:24–19:25. Raw output: `experiments/results/validation_results.json`,
`experiments/results/test_results.json`. Not from the future-risk task — see
`docs/dataset_feasibility.md`.

## Data

- Training: 5,600 images (1,400/class), Testing: 1,600 images (400/class).
- Stratified 80/20 image-level train/validation split within Training
  (4,480 train / 1,120 validation).
- Testing split held out entirely until final evaluation.

## Model comparison (validation set, n=1,120)

| Model | Accuracy | Macro F1 |
|---|---|---|
| Majority-class baseline | 0.250 | 0.100 |
| Logistic Regression | 0.679 | 0.681 |
| **Random Forest (200 trees)** | **0.886** | **0.885** |
| SVM (RBF kernel) | 0.799 | 0.799 |

Random Forest selected as best model by validation macro-F1.

## Final evaluation on held-out Testing split (n=1,600, Random Forest)

**Accuracy: 0.833 — Macro F1: 0.828**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| notumor | 0.846 | 1.000 | 0.916 |
| glioma | 0.826 | 0.627 | 0.713 |
| meningioma | 0.755 | 0.777 | 0.766 |
| pituitary | 0.903 | 0.927 | 0.915 |

## Honest read of these numbers

- All traditional-ML models clearly beat the majority-class baseline
  (0.100 macro-F1) — the 12 hand-engineered features carry real signal for
  this classification task.
- Random Forest generalizes reasonably from validation (0.885) to the truly
  held-out Testing split (0.828) — some drop, as expected, but not a
  collapse.
- **Glioma is the weak class**: 62.7% recall on Testing means over a third
  of glioma images are misclassified (mostly likely confused with
  meningioma, which is common in these datasets given overlapping texture).
  We have not performed a confusion-matrix deep-dive here due to the
  timeline — the raw confusion matrix is saved in
  `experiments/results/test_results.json` for anyone who wants to inspect it.
- `notumor` recall is a perfect 1.000 on Testing, the easiest class to
  separate given the feature set (texture/intensity features between tissue
  and no-tissue are a large, easy signal).
- **Confusion matrix breakdown** (rows = true, columns = predicted; order
  notumor/glioma/meningioma/pituitary):

  ```
                notumor  glioma  meningioma  pituitary
  notumor          400       0           0          0
  glioma            49     251          81         19
  meningioma        21      47         311         21
  pituitary          3       6          20        371
  ```

  Glioma and meningioma are mutually confused (81 glioma→meningioma, 47
  meningioma→glioma) — consistent with the literature review in
  `docs/references.md` (item 3), which found glioma/meningioma texture
  features overlap more than either does with pituitary. A secondary
  pattern: 49 of 400 glioma images were predicted `notumor`, suggesting
  some glioma slices have subtle/small lesions where the 128×128 downsample
  and our 12-feature set lose discriminative signal. `notumor` and
  `pituitary` are comparatively easy to separate.
- No calibration analysis was performed (master prompt Section 8) — the
  model's `predict_proba` output is an uncalibrated score, not a validated
  probability, and the app UI says so.
- These are detection-of-visible-tumor results. **They say nothing about
  future tumor risk prediction**, which remains unbuilt.
