# Research Question & Target Definition — Basira AI

## Primary research question (long-term)

Can quantitative features extracted from an initially tumor-free brain MRI,
potentially combined with clinical risk factors, help estimate the risk of a
subsequently diagnosed brain tumor within a defined follow-up period?

### Target definition (required before any training — Section 4A)

- **Positive outcome:** any new, histologically or radiologically confirmed
  primary brain tumor diagnosed within the follow-up window, in a patient
  whose baseline MRI showed no visible tumor.
- **Negative outcome:** no tumor diagnosed, with follow-up imaging/clinical
  contact documented for at least the full follow-up window (not merely "no
  further scan on file").
- **Follow-up horizon:** to be fixed once a real cohort is identified (e.g. 2
  or 5 years) — not yet defined, because no dataset exists to fix it against.
- **Baseline tumor-free verification:** radiologist read or equivalent
  structured report confirming no visible lesion at the baseline scan.
- **Exclusions:** patients with insufficient/unknown follow-up are excluded
  from training and reported separately — never coded as negative by default.
- **Status: UNRESOLVED.** See [dataset_feasibility.md](dataset_feasibility.md)
  — no accessible dataset currently supports this target. This document will
  be completed once/if a valid longitudinal cohort is obtained (e.g. via a UK
  Biobank access application).

## Secondary, implemented task (Wednesday demo only)

**This is a different task, built only to demonstrate the ML pipeline
mechanics.** It is NOT the research question above.

- **Task:** classify a brain MRI slice as `no_tumor`, `glioma`, `meningioma`,
  or `pituitary`, given a *currently visible* tumor (or its absence).
- **Why this is not future-risk prediction:** the label is determined by what
  is already visible in the same image being classified. There is no
  baseline-to-future linkage, no patient followed over time, and no claim
  about tumor development.
- **Purpose:** prove out MRI loading, preprocessing, feature extraction,
  traditional-ML model training/evaluation, and the Streamlit interface on
  real images with real, verifiable labels, so the same code paths are ready
  to receive a genuine longitudinal dataset later.
