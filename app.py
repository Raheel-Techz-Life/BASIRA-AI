"""Basira AI — Streamlit research prototype.

Two clearly separated things happen on this page:
1. The real pipeline: MRI upload -> validation -> preprocessing -> feature
   extraction. This is built to eventually feed the future-tumor-risk model,
   which is NOT YET TRAINED (see docs/dataset_feasibility.md — no accessible
   longitudinal dataset currently supports it).
2. An explicitly labeled "Educational Baseline Demo" that classifies a
   *currently visible* tumor type from the image. This proves the ML
   methodology works end-to-end on real data, but it is a different task
   from future-risk prediction and the UI never blurs that line.

Everything shown is real: trained-model metrics, confusion matrices and
feature importances come from the actual last training run
(experiments/results/*.json), and the "quick sample" gallery on the Try it
tab loads real images from the held-out Testing split — nothing on this
page is fabricated or placeholder data.
"""
import io
import json
import os

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.data.feature_extraction import extract_features
from src.data.preprocessing import preprocess_image
from src.models.predict import TumorClassifierUnavailable, load_model, predict_image
from src.utils.config import (
    BRAIN_TUMOR_CLASSES,
    BRAIN_TUMOR_MRI_DIR,
    EXPERIMENTS_RESULTS,
    MODELS_DIR,
    PROJECT_ROOT,
)

st.set_page_config(page_title="Basira AI", page_icon="🧠", layout="wide")

ADMIN_PASSWORD_ENV = "BASIRA_ADMIN_PASSWORD"

DISCLAIMER = (
    "Basira AI is an experimental research prototype. Its outputs are "
    "not a medical diagnosis, do not establish that a tumor will develop, "
    "and must not be used as the sole basis for clinical decisions. "
    "Independent clinical validation is required."
)

st.markdown(
    """
    <style>
    /* Tighten Streamlit's default top padding so the hero sits closer to the tabs */
    .block-container { padding-top: 2rem; max-width: 1100px; }

    .basira-hero {
        padding: 1.6rem 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(108,92,231,0.26), rgba(108,92,231,0.03));
        border: 1px solid rgba(108,92,231,0.3);
        margin-bottom: 0.9rem;
    }
    .basira-eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.72rem;
        font-weight: 700;
        color: #A9A7C7;
        margin-bottom: 0.4rem;
    }
    .basira-hero h1 { margin: 0 0 0.5rem 0; font-size: 1.85rem; line-height: 1.2; }
    .basira-tag { color: #C6C3E8; font-size: 0.95rem; max-width: 620px; line-height: 1.5; margin-bottom: 0.9rem; }

    .basira-pills { display: flex; gap: 0.5rem; flex-wrap: wrap; }
    .basira-pill {
        display: inline-flex; align-items: center; gap: 0.3rem;
        padding: 0.25rem 0.65rem; border-radius: 999px;
        font-size: 0.72rem; font-weight: 600;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.05);
        white-space: nowrap;
    }
    .basira-pill.ok { color: #2ECC71; border-color: rgba(46,204,113,0.4); background: rgba(46,204,113,0.08); }
    .basira-pill.warn { color: #E7857A; border-color: rgba(231,76,60,0.35); background: rgba(231,76,60,0.07); }

    .basira-disclaimer {
        font-size: 0.8rem;
        color: #B8AF8F;
        padding: 0.45rem 0.8rem;
        border-left: 3px solid #8A7A2E;
        background: rgba(138,122,46,0.08);
        border-radius: 4px;
        margin-bottom: 1rem;
        line-height: 1.4;
    }

    .pipeline-wrap {
        display: flex;
        align-items: center;
        gap: 0.35rem;
        flex-wrap: wrap;
        margin: 0.4rem 0 1rem 0;
    }
    .pipeline-step {
        flex: 1;
        min-width: 95px;
        padding: 0.55rem 0.5rem;
        border-radius: 10px;
        background: rgba(108,92,231,0.08);
        border: 1px solid rgba(108,92,231,0.25);
        text-align: center;
        font-size: 0.76rem;
        line-height: 1.3;
    }
    .pipeline-step .num {
        display: inline-block;
        width: 18px; height: 18px;
        border-radius: 50%;
        background: #6C5CE7;
        color: white;
        font-size: 0.68rem;
        line-height: 18px;
        margin-bottom: 0.25rem;
    }
    .pipeline-arrow { color: #6C5CE7; font-size: 1rem; }

    .sample-card {
        text-align: center;
        padding: 0.5rem;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.02);
    }

    .chip-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 0.4rem 0 0.2rem 0; }
    .chip {
        padding: 0.28rem 0.7rem; border-radius: 999px; font-size: 0.76rem;
        background: rgba(108,92,231,0.1); border: 1px solid rgba(108,92,231,0.3);
        color: #D6D2FA;
    }

    /* Pill-style buttons, echoing a "Quick demo" action-button look */
    div.stButton > button {
        border-radius: 999px !important;
        border: 1px solid rgba(108,92,231,0.4) !important;
        font-weight: 600 !important;
        padding: 0.3rem 1rem !important;
    }
    div.stButton > button:hover {
        border-color: #6C5CE7 !important;
        color: #6C5CE7 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False
if "active_image_bytes" not in st.session_state:
    st.session_state.active_image_bytes = None
if "active_image_source" not in st.session_state:
    st.session_state.active_image_source = None

with st.sidebar:
    st.markdown("### 🧠 Basira AI")
    st.caption("Illuminating hidden patterns in brain MRI")
    st.markdown("---")
    st.subheader("Administrator")
    configured_password = os.environ.get(ADMIN_PASSWORD_ENV)
    if not configured_password:
        st.caption(
            f"Set the `{ADMIN_PASSWORD_ENV}` environment variable before "
            "launch to enable admin-only exports."
        )
    elif st.session_state.admin_authenticated:
        st.success("Admin unlocked")
    else:
        entered = st.text_input("Admin password", type="password")
        if st.button("Unlock"):
            if entered == configured_password:
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")

try:
    _, _, app_metadata = load_model()
    model_available = True
except TumorClassifierUnavailable:
    model_available = False
    app_metadata = None

dataset_ready = BRAIN_TUMOR_MRI_DIR.exists()
model_pill = (
    f'<span class="basira-pill ok">✅ Model: {app_metadata["model_name"]}</span>'
    if model_available
    else '<span class="basira-pill warn">⚠️ Model: not trained</span>'
)
dataset_pill = (
    '<span class="basira-pill ok">✅ Dataset: loaded</span>'
    if dataset_ready
    else '<span class="basira-pill warn">⚠️ Dataset: missing</span>'
)

st.markdown(
    f"""
    <div class="basira-hero">
        <div class="basira-eyebrow">🧠 Basira AI — Brain MRI research prototype</div>
        <h1>Looking for patterns a first glance misses.</h1>
        <div class="basira-tag">
            Basira (بصيرة) means insight. This prototype pairs a real MRI
            feature-extraction pipeline with a traditional ML classifier —
            built, trained and evaluated end-to-end.
        </div>
        <div class="basira-pills">
            {dataset_pill}
            {model_pill}
            <span class="basira-pill warn">🔴 Future-risk model: not built</span>
        </div>
    </div>
    <div class="basira-disclaimer">⚠️ {DISCLAIMER}</div>
    """,
    unsafe_allow_html=True,
)

tab_overview, tab_try, tab_results, tab_docs = st.tabs(
    ["🏠 Overview", "🔬 Try it", "📊 Results & evaluation", "📄 Documentation"]
)

# ---------------------------------------------------------------- Overview
with tab_overview:
    test_results_path = EXPERIMENTS_RESULTS / "test_results.json"
    if model_available and test_results_path.exists():
        with open(test_results_path) as f:
            tr = json.load(f)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Held-out accuracy", f"{tr['accuracy']*100:.1f}%")
        m2.metric("Macro F1", f"{tr['macro_f1']:.3f}")
        with m3:
            st.caption("Model")
            model_label = app_metadata["model_name"].replace("_", " ").title()
            st.markdown(
                f'<div style="font-size:1.3rem; font-weight:600;">{model_label}</div>',
                unsafe_allow_html=True,
            )
        m4.metric("Test images", sum(tr["confusion_matrix"][0]) * len(tr["labels"]))

    st.markdown(
        "This project investigates whether an **initially tumor-free** brain "
        "MRI can help estimate the risk of a **future** tumor diagnosis — "
        "that model isn't built, because no public dataset links a "
        "tumor-free baseline to a verified future diagnosis "
        "(see **Documentation → Dataset feasibility**). What *is* built is "
        "the full pipeline below, proven end-to-end on a separate "
        "**Educational Baseline Demo** that classifies a currently visible "
        "tumor — a different task, never presented as future-risk prediction."
    )

    st.markdown("**Pipeline**")
    st.markdown(
        """
        <div class="pipeline-wrap">
          <div class="pipeline-step"><div class="num">1</div><br/>Upload</div>
          <div class="pipeline-arrow">→</div>
          <div class="pipeline-step"><div class="num">2</div><br/>Validate</div>
          <div class="pipeline-arrow">→</div>
          <div class="pipeline-step"><div class="num">3</div><br/>Preprocess</div>
          <div class="pipeline-arrow">→</div>
          <div class="pipeline-step"><div class="num">4</div><br/>Extract features</div>
          <div class="pipeline-arrow">→</div>
          <div class="pipeline-step"><div class="num">5</div><br/>Classify</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="chip-row">
          <span class="chip">5 intensity features</span>
          <span class="chip">4 GLCM texture features</span>
          <span class="chip">3 shape / morphology features</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "👉 **Try it** for a live prediction, or **Results & evaluation** "
        "for the full numbers."
    )

# -------------------------------------------------------------------- Try it
with tab_try:
    st.subheader("Quick samples")
    st.caption(
        "Real images from the held-out Testing split (never used in "
        "training) — click one to run it through the pipeline instantly."
    )

    sample_cols = st.columns(len(BRAIN_TUMOR_CLASSES))
    for col, cls in zip(sample_cols, BRAIN_TUMOR_CLASSES):
        sample_dir = BRAIN_TUMOR_MRI_DIR / "Testing" / cls
        with col:
            if sample_dir.exists():
                sample_files = sorted(sample_dir.iterdir())
                if sample_files:
                    sample_path = sample_files[0]
                    st.markdown(f'<div class="sample-card">', unsafe_allow_html=True)
                    st.image(str(sample_path), caption=cls, use_container_width=True)
                    if st.button(f"Use this", key=f"sample_{cls}"):
                        st.session_state.active_image_bytes = sample_path.read_bytes()
                        st.session_state.active_image_source = f"sample ({cls})"
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.caption("Dataset not found")

    st.markdown("---")
    st.subheader("Or upload your own")
    st.caption(
        "Supported formats: JPG, JPEG, PNG. This demo accepts 2D slice "
        "images (not raw DICOM/NIfTI volumes)."
    )
    uploaded = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        st.session_state.active_image_bytes = uploaded.getvalue()
        st.session_state.active_image_source = f"upload ({uploaded.name})"

    raw_bytes = st.session_state.active_image_bytes

    if raw_bytes is None:
        st.info("Pick a quick sample above or upload your own MRI slice to begin.")
    else:
        st.caption(f"Active image: {st.session_state.active_image_source}")

        tmp_check = io.BytesIO(raw_bytes)
        try:
            img = Image.open(tmp_check)
            img.verify()
            tmp_check.seek(0)
            img = Image.open(tmp_check)
            width, height = img.size
            valid = width >= 16 and height >= 16
            reason = "" if valid else "image too small"
        except Exception as exc:  # noqa: BLE001 - surface any PIL decode error to the user
            valid = False
            reason = str(exc)

        if not valid:
            st.error(f"Invalid image: {reason}")
        else:
            img = Image.open(io.BytesIO(raw_bytes)).convert("L")

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Uploaded image**")
                st.image(img, width=280)
            with col_b:
                st.markdown("**Preprocessed (128x128, normalized)**")
                processed = preprocess_image(img)
                st.image(processed, width=280, clamp=True)

            st.markdown("---")
            st.subheader("Extracted quantitative features")
            feats = extract_features(processed)
            feat_df = pd.DataFrame([feats]).T.rename(columns={0: "value"})
            st.dataframe(feat_df, use_container_width=True)

            st.download_button(
                "Download extracted features (JSON)",
                data=json.dumps(feats, indent=2),
                file_name="extracted_features.json",
                mime="application/json",
            )

            st.markdown("---")
            st.subheader("Educational Baseline Demo — existing-tumor classification")
            st.info(
                "This classifier detects a tumor type that is already "
                "visible in the uploaded image. It is a separate "
                "demonstration of the ML pipeline mechanics and is **not** "
                "a future-risk prediction."
            )
            try:
                model, scaler, metadata = load_model()
                tmp_path = MODELS_DIR / "_tmp_upload.png"
                img.save(tmp_path)
                result = predict_image(tmp_path, model, scaler, metadata)
                tmp_path.unlink(missing_ok=True)

                pred_col, chart_col = st.columns([1, 2])
                with pred_col:
                    st.metric("Predicted class", result["predicted_label"])
                with chart_col:
                    if result["class_probabilities"]:
                        proba_df = pd.DataFrame(
                            sorted(
                                result["class_probabilities"].items(),
                                key=lambda kv: -kv[1],
                            ),
                            columns=["class", "probability"],
                        )
                        st.bar_chart(proba_df.set_index("class"))
                st.caption(
                    "This score reflects model confidence on the "
                    "educational classification task only — it is not a "
                    "calibrated probability of future tumor development."
                )
            except TumorClassifierUnavailable as exc:
                st.warning(str(exc))

# ------------------------------------------------------------ Results tab
with tab_results:
    st.subheader("Real results from the last training run")
    val_results_path = EXPERIMENTS_RESULTS / "validation_results.json"

    if not test_results_path.exists():
        st.info("No results yet. Run `python -m src.models.train` first.")
    else:
        with open(test_results_path) as f:
            tr = json.load(f)

        st.markdown("**Held-out Testing split — per class**")
        report_df = pd.DataFrame(tr["per_class_report"]).T
        report_df = report_df.loc[tr["labels"]]
        st.dataframe(
            report_df[["precision", "recall", "f1-score", "support"]].round(3),
            use_container_width=True,
        )

        col_cm, col_fi = st.columns(2)

        with col_cm:
            st.markdown("**Confusion matrix** (rows = true, cols = predicted)")
            cm = np.array(tr["confusion_matrix"])
            cm_df = pd.DataFrame(cm, index=tr["labels"], columns=tr["labels"])
            st.dataframe(
                cm_df.style.background_gradient(cmap="Purples", axis=None),
                use_container_width=True,
            )
            st.caption(
                "Glioma and meningioma are the most mutually confused "
                "classes — see docs/evaluation.md for the full breakdown."
            )

        with col_fi:
            st.markdown("**Feature importance** (trained model)")
            if model_available:
                model, _, metadata = load_model()
                if hasattr(model, "feature_importances_"):
                    fi_df = pd.DataFrame(
                        {
                            "feature": metadata["feature_names"],
                            "importance": model.feature_importances_,
                        }
                    ).sort_values("importance", ascending=True)
                    st.bar_chart(fi_df.set_index("feature"))
                else:
                    st.caption(
                        f"The selected model (`{metadata['model_name']}`) "
                        "does not expose per-feature importances directly."
                    )
            else:
                st.caption("Train a model first to see feature importance.")

        if val_results_path.exists():
            with open(val_results_path) as f:
                vr = json.load(f)
            st.markdown("---")
            st.markdown("**Model comparison (validation split)**")
            comp_df = pd.DataFrame(
                {
                    name: {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
                    for name, m in vr.items()
                }
            ).T.round(3)
            st.bar_chart(comp_df)
            st.dataframe(comp_df, use_container_width=True)

# -------------------------------------------------------------- Docs tab
with tab_docs:
    st.subheader("Project documentation")
    st.caption(
        "These are real, version-controlled documents in `docs/` — not "
        "generated for this view."
    )
    doc_files = [
        ("📋 Dataset feasibility report", "dataset_feasibility.md"),
        ("❓ Research question & target definition", "research_question.md"),
        ("🧪 Methodology", "methodology.md"),
        ("📈 Evaluation results", "evaluation.md"),
        ("🏗️ Architecture", "architecture.md"),
        ("⚠️ Limitations", "limitations.md"),
        ("📚 References", "references.md"),
        ("🎤 Demo script", "demo_script.md"),
        ("✅ Project status", "project_status.md"),
    ]
    for title, filename in doc_files:
        path = PROJECT_ROOT / "docs" / filename
        with st.expander(title):
            if path.exists():
                st.markdown(path.read_text(encoding="utf-8"))
            else:
                st.caption("Not found.")

# ------------------------------------------------------ Admin-only export
if st.session_state.admin_authenticated:
    st.markdown("---")
    st.subheader("Administrator — export experiment summary")
    meta_path = MODELS_DIR / "model_metadata.json"
    if test_results_path.exists() and meta_path.exists():
        with open(test_results_path) as f:
            test_results = json.load(f)
        with open(meta_path) as f:
            run_metadata = json.load(f)
        summary = {"model_metadata": run_metadata, "held_out_test_results": test_results}
        st.json(summary, expanded=False)
        st.download_button(
            "Download experiment summary (JSON)",
            data=json.dumps(summary, indent=2),
            file_name="basira_ai_experiment_summary.json",
            mime="application/json",
        )
        st.caption(
            "This is the real output of the last `python -m src.models.train` "
            "run — not a fabricated or placeholder report."
        )
    else:
        st.info("No experiment results found yet. Run `python -m src.models.train` first.")

st.markdown("---")
st.caption("Basira AI — academic research prototype. Not for clinical use.")
