# Demo Script — Basira AI (Wednesday)

**Keep this to ~5 minutes.** Structure: research question → honest data
limitation → what's actually working → live demo → limitations → next steps.

## 1. Research question (30s)

"Basira AI asks whether an MRI that currently shows no tumor can help
predict a *future* tumor diagnosis. This is a prediction task, not a
detection task — the baseline scan looks normal."

## 2. The honest data finding (45s)

"Before building anything, we audited whether a dataset for this actually
exists publicly. It doesn't: IXI and OASIS are healthy-brain cohorts with no
tumor outcomes; every public longitudinal brain-tumor dataset we found
(TCIA collections, UCSF-ALPTDG, Gamma Knife recurrence data) starts *after*
a tumor is already diagnosed — that's a different task (recurrence), not
first-time incidence. Full writeup in `docs/dataset_feasibility.md`. The
closest real option, UK Biobank, requires a formal access application we
can't complete in two days."

## 3. What we built instead, and why it's honest (30s)

"Rather than fabricate labels or quietly swap the research question, we
built the full real pipeline — MRI validation, preprocessing, feature
extraction — generically, so it's ready for a true longitudinal dataset
later. And we built a clearly separate 'Educational Baseline Demo' that
proves the ML methodology on real, verifiable labels: classifying a
*currently visible* tumor type. It is explicitly labeled as not being the
future-risk model, on every screen it appears."

## 4. Live demo (2 min)

1. Open the Streamlit app (`streamlit run app.py`).
2. Point at the "Project status" section — show future-risk model is
   honestly marked "not built" with the reason.
3. Upload a sample MRI slice from `data/raw/brain_tumor_mri/Testing/`.
4. Walk through: original → preprocessed (resized/normalized) → extracted
   feature table → educational classifier prediction + class probabilities.
5. Point out the disclaimer text and the "not a calibrated probability"
   caption.

## 5. Real results (30s)

Read directly from `docs/evaluation.md` / `experiments/results/test_results.json`
— state the actual accuracy/macro-F1 numbers once training has run. Do not
round up or omit a weak class.

## 6. Limitations and next steps (45s)

- No patient IDs in the public dataset → no true patient-level split for
  the baseline demo (documented, not hidden).
- No future-risk model yet; next step is a UK Biobank (or similar) access
  application if the mentor wants to pursue the original question.
- Academic title currently doesn't match either task — flag for mentor
  decision.

## Anticipated questions

- *"Why not just use the tumor dataset as the future-risk model?"* — Because
  the label would come from the same image being classified, not from a
  verified future outcome. That's detection, not prediction; using it as if
  it were prediction would be a fabricated claim.
- *"Could you simulate/synthesize future outcomes for the demo?"* — We
  could, but only with an explicit, visible disclaimer that it's simulated
  and not evidence of real predictive capability; we chose not to, to avoid
  any ambiguity in a research demo.
