# References — Basira AI

Compiled 2026-11-02 via web search; verify full text before citing in a
final submission (search snippets were not independently fact-checked
against the original PDFs). Grouped by what each supports in this project.

## A. Methodology for the implemented Educational Baseline Demo

1. Cheng J, Huang W, Cao S, et al. (2015). "Enhanced Performance of Brain
   Tumor Classification via Tumor Region Augmentation and Partition." *PLoS
   ONE* 10(10): e0140381. https://pmc.ncbi.nlm.nih.gov/articles/PMC4598126
   — Origin of the Figshare glioma/meningioma/pituitary MRI collection that
   feeds into the Kaggle dataset we trained on.

2. Frontiers in Computational Neuroscience (2024). Multi-task CNN on the
   combined Figshare + SARTAJ + Br35H, 7,023-image set (glioma, meningioma,
   pituitary, no-tumor).
   https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2024.1418546/epub
   — Describes the exact dataset composition (including the SARTAJ
   glioma-label swap) used in this project.

3. Comparative study (2026), GLCM + GLRLM texture features with SVM, Naive
   Bayes, MLP, Random Forest on glioma/meningioma/pituitary MRI.
   https://journal.unnes.ac.id/journals/jf/article/view/39267 — Closest
   direct precedent for our texture-feature + classical-ML approach;
   reports Random Forest as the strongest classifier, matching our result.

4. Zenodo preprint (2026). GLCM + LBP feature ensemble (5 classifiers) on
   the same 7,023-image Kaggle dataset; Random Forest and XGBoost strongest
   single models (~90%). https://zenodo.org/records/20048527

5. Frontiers in Neuroscience (2021). Texture-feature SVM classifier,
   tumor vs. healthy tissue, 40 patients.
   https://www.frontiersin.org/articles/10.3389/fnins.2021.634926/pdf

6. AJNR (2018). "Radiomics in Brain Tumor: Image Assessment, Quantitative
   Feature Descriptors, and Machine-Learning Approaches."
   https://www.ajnr.org/content/39/2/208 — Justifies our feature-family
   choices (intensity, texture, shape) in `feature_extraction.py`.

7. Latif G. et al. (2022). "Recent Advances in Classification of Brain
   Tumor from MR Images — State of the Art Review from 2017 to 2021."
   *Current Medical Imaging*.
   https://www.brainlife.org/fulltext/2022/Latif_G220117_CurrMedImaging.pdf

## B. Data leakage / patient-level splitting (master prompt Section 5E)

8. Rumala D. (2023). "How You Split Matters: Data Leakage and Subject
   Characteristics Studies in Longitudinal Brain MRI Analysis." arXiv
   2309.00350. https://arxiv.org/pdf/2309.00350

9. Bussola N. et al. "AI slipping on tiles: data leakage in digital
   pathology." arXiv 1909.06539. https://arxiv.org/pdf/1909.06539 —
   Quantifies inflated performance from improper (non-patient-level)
   splitting; directly supports why we flagged this as a limitation in
   `docs/limitations.md`.

10. medRxiv (2020). "Training machine learning models on patient level
    data segregation is crucial in practical clinical applications."
    https://www.medrxiv.org/content/10.1101/2020.04.23.20076406.full.pdf

## C. Precedent for predicting future disease from imaging with no visible disease yet

11. Yala A. et al. (2021). "Toward Robust Mammography-Based Models for
    Breast Cancer Risk." *Science Translational Medicine* 13(578). Project
    page: https://jclinic.mit.edu/research-project/toward-robust-mammography-based-models-for-breast-cancer-risk/
    — The clearest existing precedent for the shape of our (unbuilt)
    research question: a model (Mirai) estimating future cancer risk from
    an image with no diagnosed cancer yet, trained on genuine longitudinal
    outcome data (210,819 mammograms, 56,786 patients, multi-site
    validation). Shows what a *valid* version of our research question
    requires: a real linked longitudinal cohort of this scale.

12. Burger B. et al. (2023). Deep-learning anomaly detection on
    high-risk breast MRI predicting future lesion emergence (AUC 0.804,
    small sample). *European Radiology Experimental*.
    https://doaj.org/article/24e445bfa58447a4b00c2d266fbfe43a

## D. Base-rate evidence for dataset feasibility (supports our no-go finding)

13. Morris Z. et al. (2009 and 2018 updates, BMJ). "Potentially serious
    incidental findings on brain and body MRI of apparently asymptomatic
    adults: systematic review and meta-analysis."
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6249611/ — Pooled
    prevalence of potentially serious incidental brain findings: ~1.4%;
    earlier brain-only review found neoplastic incidental findings in
    ~0.70% of asymptomatic people. Supports why a usable future-incident-
    tumor cohort needs very large, purpose-built longitudinal imaging
    studies (e.g. UK Biobank scale), not an off-the-shelf Kaggle dataset.

14. Earlier brain-only incidental finding meta-analysis (BMJ, 2009).
    https://pmc.ncbi.nlm.nih.gov/articles/PMC2728201

## E. Evaluation methodology (calibration / class imbalance — master prompt Section 8)

15. Andersen et al. (2026). "Tipping the Balance: Impact of Class
    Imbalance Correction on the Performance of Clinical Risk Prediction
    Models." arXiv 2603.00208. https://arxiv.org/pdf/2603.00208 —
    Resampling/correction methods did not improve discrimination and
    degraded calibration across 605,842 patients/10 datasets; relevant to
    why we flag our model's `predict_proba` output as uncalibrated rather
    than treating it as a true probability (`docs/evaluation.md`).

---

**Note on verification:** these were located via web search snippets, not
full-text review. Before citing any of these in a final academic
submission, open the actual PDF/DOI and confirm the claim, authors, venue,
and year — per the master prompt's "never invent citations" rule, this list
is a verified starting point, not a substitute for reading the papers.
