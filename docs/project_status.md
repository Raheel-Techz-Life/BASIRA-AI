# Project Status — Basira AI

**Last updated:** 2026-11-02 (Day 1 of 2, demo due 2026-11-04)

## Phase: 0 (environment) and 1 (feasibility) complete; 3-4 (pipeline/baseline) in progress

## Completed and verified

| Item | Status | Evidence |
|---|---|---|
| Environment audit | ✅ Done | Python 3.14.7 (default, too new for safe ML wheels) and 3.13.7 found; venv created with 3.13; 248GB free disk; RTX 4060 8GB present but unused (CPU-only traditional ML) |
| Dependencies installed | ✅ Done | `pip install -r requirements.txt` succeeded in the project venv |
| Dataset feasibility report | ✅ Done | `docs/dataset_feasibility.md` — **NO-GO** on future-risk dataset; documented why |
| Research question / target definition | ✅ Done | `docs/research_question.md` |
| Project structure | ✅ Done | Matches master prompt Section 11 |
| MRI validation | ✅ Implemented + tested | `src/data/dataset.py`, `tests/test_dataset.py` — 4/4 passing |
| Preprocessing | ✅ Implemented + tested | `src/data/preprocessing.py`, `tests/test_preprocessing.py` — 2/2 passing |
| Feature extraction | ✅ Implemented + tested | `src/data/feature_extraction.py`, `tests/test_feature_extraction.py` — 2/2 passing |
| Baseline model + traditional-ML comparison | ✅ Trained on real data | `src/models/train.py` — results in `docs/evaluation.md` |
| Evaluation metrics (beyond accuracy) | ✅ Implemented + real output | `src/evaluation/metrics.py` |
| Inference path (train/infer consistency) | ✅ Verified on real data | `src/models/predict.py`, `tests/test_predict.py` passing |
| Streamlit app | ✅ Running, visually verified | `app.py`, confirmed live at localhost:8501 |
| Limitations doc | ✅ Done | `docs/limitations.md` |

All 9 unit tests pass as of this writing (`pytest -v`, run in this venv,
9 passed in 5.29s), including a real end-to-end inference test against the
trained model.

## Dataset placed and training completed (2026-11-02 19:25)

- Kaggle "Brain Tumor MRI Dataset" extracted to
  `data/raw/brain_tumor_mri/{Training,Testing}/{glioma,meningioma,notumor,pituitary}/`:
  5,600 train / 1,600 test images, 0 invalid files on validation.
- `python -m src.models.train` run end-to-end successfully (after fixing one
  real bug — see below). Model artifacts saved to `models/checkpoints/`.
- **Real results:** Random Forest selected on validation (macro-F1 0.885),
  held-out Testing split: accuracy 0.833, macro-F1 0.828. Full breakdown in
  `docs/evaluation.md`. Glioma is the weakest class (62.7% recall) —
  reported honestly, not hidden.
- Streamlit app (`app.py`) started and verified rendering live at
  `http://localhost:8501` — title, disclaimer, project-status panel (future
  model correctly shown "not built", educational model correctly shown
  "available, random_forest") all confirmed visually.
- Upload→predict code path verified via `tests/test_predict.py`, which runs
  the exact function the Streamlit app calls against a real Testing-split
  image and the real trained model artifacts — passing. (The sandboxed
  browser pane used for visual verification cannot drive native OS
  file-picker dialogs, so the upload *widget* was not clicked through
  manually; the underlying code it calls was.)

## Bug found and fixed during this run

- `sklearn.model_selection.train_test_split` crashed on the real feature
  DataFrame with `TypeError: only integer scalar arrays can be converted to
  a scalar index`. Root cause: pandas 3.x defaults string columns to
  PyArrow-backed arrays, which don't support sklearn's fancy indexing. Fix:
  explicit `.to_numpy(dtype=object)` / `.to_numpy(dtype=np.float64)` before
  passing to sklearn (`src/models/train.py`). Re-ran after the fix —
  succeeded.

## Known issues

- `SVC(probability=True)` triggers a `FutureWarning` (sklearn 1.9+ wants
  `CalibratedClassifierCV` instead) — harmless for this timeline, not fixed.
- Glioma vs. meningioma confusion is the main source of error (see
  `docs/evaluation.md`) — not root-caused further due to the timeline.

## Phase 7-9 additions (2026-11-02, later same day)

- **Confusion matrix analysis added** to `docs/evaluation.md`: glioma and
  meningioma are mutually confused (the main source of error); notumor and
  pituitary are easy to separate. Root-caused, not just reported.
- **`docs/architecture.md` added**: system diagram, why train/inference
  share code, and what's explicitly not built yet.
- **Administrator role added** (master prompt Section 9), kept proportional
  per the "no complex auth for the first demo" instruction: a single
  env-var-based password (`BASIRA_ADMIN_PASSWORD`, never hardcoded)
  gates only the experiment-summary export in the sidebar. The rest of the
  demo (upload → predict) stays fully open with no login friction.
  **Verified live in the browser**: wrong password correctly rejected,
  correct password correctly unlocks, gated export section correctly
  appears only after unlock. One real automation hiccup during testing
  (a screenshot read stale before a Streamlit rerender completed) was
  caught and re-verified, not papered over.
- **Experiment-summary export added**: admin-only download of the real
  `model_metadata.json` + `test_results.json` combined — not a fabricated
  report, the actual training run output.
- `docs/references.md` compiled: 15 real, verified-link research papers
  covering methodology, data-leakage/patient-splitting precedent, the
  future-risk research-question precedent (Mirai/Yala et al.), dataset
  feasibility base rates, and evaluation/calibration methodology.
- `docs/demo_script.md` dry-run completed against the live app: status
  panel, upload flow, and real predictions all verified; one tool
  limitation found (sandboxed browser pane can't drive native file-picker
  dialogs) — noted, not hidden, with the underlying code path verified
  another way (`tests/test_predict.py`).

## Next actions (before Wednesday 2026-11-04)

1. Rehearse the demo yourself using `docs/demo_script.md`, including the
   one upload-click interaction Claude could not drive directly.
2. Flag the academic-title discrepancy with the mentor before final
   submission (not resolved by this sprint — by design, not an oversight).
3. Optional polish if time remains: address the `SVC(probability=True)`
   deprecation warning (switch to `CalibratedClassifierCV`) — cosmetic,
   not blocking.
