# Basira AI

**Illuminating Hidden Patterns in Brain MRI**

> ⚠️ Research prototype. Not a medical device. See disclaimer below.

## What this project actually is right now

This repo pursues the research question: *can quantitative features from an
initially tumor-free brain MRI help estimate the risk of a future brain
tumor diagnosis?*

**That model has not been built**, because no accessible public dataset
currently links a verified tumor-free baseline MRI to a verified future
tumor diagnosis. Full investigation: [`docs/dataset_feasibility.md`](docs/dataset_feasibility.md).

What *is* implemented and working:

- A real MRI ingestion pipeline: file validation → preprocessing → feature
  extraction, generic enough to plug into a true longitudinal dataset later.
- An **Educational Baseline Demo**: a traditional-ML classifier (Logistic
  Regression / Random Forest / SVM, compared against a majority-class
  baseline) that classifies a *currently visible* tumor type
  (glioma / meningioma / pituitary / no_tumor) on the public Kaggle
  "Brain Tumor MRI Dataset". This is explicitly **not** future-risk
  prediction — it proves the pipeline mechanics on real, verifiable labels.
- A Streamlit interface exposing both the pipeline and the baseline demo,
  with disclaimers that keep the two tasks clearly separated.

See [`docs/limitations.md`](docs/limitations.md) for the full, honest list
of what is and isn't validated.

## Setup (Windows)

> Note: this folder is still named `CerebroNoor-AI` on disk (the project was
> renamed to Basira AI after the folder and Python venv were created;
> renaming the folder would break the venv's Windows launcher scripts,
> which embed absolute paths). The app, docs, and all branding say
> "Basira AI" — only the folder path is the old name.

```bash
cd CerebroNoor-AI
C:\Python313\python.exe -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Get the dataset (required for the educational demo)

1. Download **Brain Tumor MRI Dataset** (masoudnickparvar) from Kaggle:
   https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
2. Unzip it so you end up with:
   ```
   data/raw/brain_tumor_mri/Training/{glioma,meningioma,notumor,pituitary}/
   data/raw/brain_tumor_mri/Testing/{glioma,meningioma,notumor,pituitary}/
   ```

## Run the tests

```bash
.\venv\Scripts\python.exe -m pytest -v
```

## Train the educational baseline classifier

```bash
.\venv\Scripts\python.exe -m src.models.train
```

This builds manifests, extracts features, trains/compares majority-baseline,
logistic regression, random forest, and SVM, and saves the best model to
`models/checkpoints/`. Results are saved to `experiments/results/`.

## Optional: train the CNN comparison model

```bash
.\venv\Scripts\python.exe -m src.models.train_cnn
```

Trains a small CNN on the same split as the Random Forest baseline and
saves a head-to-head comparison to `experiments/results/cnn_test_results.json`
(see `docs/evaluation.md`). The live demo still uses the Random Forest —
the CNN did not outperform it in testing, so swapping it in would not be
honest. This is optional and not required to run the app.

## Run the web app

```bash
.\venv\Scripts\streamlit.exe run app.py
```

Optional: to enable the administrator-only experiment-summary export, set a
password via an environment variable before launching (no password is
hardcoded in the code):

```powershell
$env:BASIRA_ADMIN_PASSWORD = "choose-your-own-password"
.\venv\Scripts\streamlit.exe run app.py
```

Without this variable set, the app runs normally — the admin export section
is simply unavailable, and the full upload → predict demo still works.

## Project structure

See `docs/architecture.md` (and the master prompt) for the full intended
structure. Only the pieces described above are implemented so far.

## Disclaimer

> Basira AI is an experimental research prototype. Its outputs are not
> a medical diagnosis, do not establish that a tumor will develop, and must
> not be used as the sole basis for clinical decisions. Independent clinical
> validation is required.
