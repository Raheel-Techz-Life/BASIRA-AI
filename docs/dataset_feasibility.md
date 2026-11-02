# Dataset Feasibility Report — Basira AI

**Date:** 2026-11-02
**Phase:** 0/1 (Environment Audit + Research Feasibility)
**Status: NO-GO for the future incident-tumor-risk prediction task, within the Wednesday 2026-11-04 timeline.**

## Research question under test

> Can quantitative features extracted from an initially tumor-free brain MRI,
> potentially combined with clinical risk factors, help estimate the risk of a
> subsequently diagnosed brain tumor within a defined follow-up period?

To answer this validly we need a dataset that provides, per patient:

1. A baseline MRI *confirmed* tumor-free at acquisition time.
2. A persistent patient identifier linking that baseline to later timepoints.
3. A verified outcome: new tumor diagnosed within a defined follow-up window,
   OR adequately documented follow-up with no diagnosis (true negative — not
   merely "no later scan supplied").
4. Sufficient positive-outcome count to support any supervised model.
5. Usable licensing for an academic prototype.

## Candidates investigated

| Dataset | Tumor-free baseline? | Linked future outcome? | Verdict |
|---|---|---|---|
| [IXI](https://brain-development.org/ixi-dataset/) | Yes (healthy volunteers) | No — cross-sectional, no follow-up diagnosis records | **Unsuitable.** Normal-brain dataset, not an outcome dataset. |
| [OASIS Brain](https://sites.wustl.edu/oasisbrains/) | Yes (aging/dementia cohort) | No — follow-up tracks cognitive/Alzheimer's status, not tumor incidence | **Unsuitable** for this target. |
| [TCIA](https://www.cancerimagingarchive.net/) collections (e.g. UCSF-ALPTDG, Gamma Knife recurrence sets, the 2025 11,000-study brain-metastasis longitudinal release) | **No** — all cohorts enter the dataset *after* a tumor is already diagnosed | Longitudinal, but tracks *recurrence/progression*, not *first incidence* | **Unsuitable** — this is the recurrence-prediction task the master prompt explicitly says not to substitute. |
| UK Biobank imaging (population cohort with incidental-finding follow-up) | Plausible | Plausible, in principle | Requires a formal data-access application (weeks, review board, fee) — **not obtainable in 2 days.** |

Web search on 2026-11-02 for "longitudinal baseline MRI tumor-free linked future brain tumor diagnosis" confirms the same conclusion: no current public release pairs a verified-tumor-free baseline with a verified future incident-tumor outcome. Every public longitudinal brain-tumor dataset found (TCIA brain-metastasis release, UCSF-ALPTDG, Gamma Knife recurrence set, pediatric glioma recurrence cohort) starts from patients who **already have** a diagnosed tumor.

## Conclusion

**Do not train or display a future-tumor-risk model for the Wednesday demo.** Doing so would require either (a) fabricating outcome labels on a normal-brain dataset — explicitly forbidden by the master prompt, or (b) silently substituting a recurrence-prediction dataset for an incident-prediction task — also explicitly forbidden.

## What we build instead for Wednesday

Per the master prompt's own fallback clause (Section 5 / revised instructions):

> "...implement the imaging, preprocessing, feature extraction, and interface
> components. Clearly identify future-risk model training and validation as
> pending. You may implement a separate educational baseline classifier using
> a suitable existing-tumor dataset only if it is explicitly labeled as a
> separate demonstration and never presented as the future-risk model."

So the demo consists of two clearly separated pieces:

1. **The real, working pipeline** (MRI loading → validation → preprocessing →
   feature extraction → Streamlit interface) — built generically so it will
   plug into a true future-outcome dataset later, if/when one becomes
   accessible (e.g. via a UK Biobank application).
2. **An explicitly labeled "Educational Baseline Demo: Existing-Tumor
   Detection"** — a traditional-ML classifier (SVM / Logistic Regression /
   Random Forest) trained on a public *existing-tumor* MRI dataset
   (glioma / meningioma / pituitary / no_tumor — Kaggle
   "Brain Tumor MRI Dataset", masoudnickparvar, DOI 10.34740/KAGGLE/DSV/1183165,
   combining Figshare + SARTAJ + Br35H sources). This demonstrates the ML
   methodology (feature extraction → traditional classifier → evaluation)
   end-to-end on real images with real, verifiable labels — but it is
   **detecting an already-visible tumor in the image**, which is a different
   task from predicting future incident tumor development. The interface
   labels this module explicitly as such, with no claim of future-risk
   prediction.

Flag for mentor discussion: the finalized academic title
("Reliable Brain Tumor Segmentation from MRI Using Deep Learning") describes
neither the future-risk prediction research question nor the existing-tumor
*classification* baseline actually implemented. This discrepancy predates this
2-day sprint and should be resolved with the mentor before the final
submission — it is not something to silently paper over.
