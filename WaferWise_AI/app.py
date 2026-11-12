"""WaferWise AI - Home dashboard. Run with: streamlit run app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

from src import config  # noqa: E402
from src.data_loader import dataset_signature  # noqa: E402
from src.ui import card, fmt_pct, get_artifacts, setup_page, sidebar_footer  # noqa: E402

setup_page("WaferWise AI", "Intelligent semiconductor defect prediction and explainable manufacturing quality analysis")
sidebar_footer()

art = get_artifacts(required=False)

# ------------------------------------------------------------- status / stats ---
if art is None:
    st.warning(
        "**Setup needed:** no trained model was found. Add the SECOM dataset to `data/` (see `data/README.md`) and run "
        "`python -m src.train`. Until then the pages below show their layout but no results."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset files in data/", "found" if dataset_signature(config.get_data_dir()) else "not found")
    c2.metric("Trained model", "not found")
    c3.metric("Results shown", "none")
else:
    md, mt = art.metadata, art.metrics
    ds, sel = md["dataset"], mt["models"][mt["selected_model"]]
    st.caption(f"Statistics below are computed from the dataset used for training ({md['data_source']}).")
    c = st.columns(5)
    c[0].metric("Samples", f"{ds['n_samples']:,}")
    c[1].metric("Process features", f"{ds['n_features']:,}")
    c[2].metric("FAIL samples", f"{ds['n_fail']:,}", fmt_pct(ds["fail_pct"]), delta_color="off")
    c[3].metric("Missing cells", fmt_pct(ds["missing_cells_pct"]))
    c[4].metric("Final model", md["selected_model_display"])
    c = st.columns(5)
    c[0].metric("Features used by model", md["n_model_input_features"])
    c[1].metric("CV PR-AUC (train)", f"{sel['cv']['pr_auc_mean']:.3f}")
    c[2].metric("Test FAIL recall", f"{sel['test_tuned']['recall']:.3f}")
    c[3].metric("Test FAIL precision", f"{sel['test_tuned']['precision']:.3f}")
    c[4].metric("Test PR-AUC", f"{sel['test_tuned']['pr_auc']:.3f}")
    st.caption(
        f"Held-out test set: {mt['test_set']['n']} samples, {mt['test_set']['n_fail']} FAIL — small counts mean "
        "wide uncertainty (see Model Comparison for bootstrap intervals)."
    )

st.divider()

# ------------------------------------------------------------------ narrative ---
left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("Project introduction")
    st.write(
        "WaferWise AI is an academic machine-learning system that predicts whether a semiconductor manufacturing "
        "sample will **PASS** or **FAIL** in-line testing from hundreds of process-sensor measurements, and explains "
        "which measurements drive each prediction."
    )
    st.subheader("Problem statement")
    st.write(
        "Modern fabrication lines record hundreds of signals per production unit, but only a few carry information "
        "about yield. Failures are rare, so a model that always says *PASS* looks accurate while catching nothing. "
        "The task is to detect the rare FAIL class reliably from noisy, partly missing sensor data, and to make the "
        "result interpretable enough for process engineers to act on."
    )
    st.subheader("Objectives")
    st.markdown(
        "- Build a leakage-free pipeline for cleaning, imputation, feature filtering and selection.\n"
        "- Compare Logistic Regression, Random Forest and XGBoost with class-imbalance-aware metrics.\n"
        "- Prioritise FAIL-class recall, precision, F1 and PR-AUC rather than accuracy.\n"
        "- Explain predictions globally and per sample with SHAP.\n"
        "- Offer single and batch prediction with validated CSV input and downloadable results."
    )
with right:
    st.subheader("Navigate")
    st.markdown(
        "1. **Dataset Explorer** – upload or inspect data\n"
        "2. **Exploratory Data Analysis** – missingness, class balance, correlations\n"
        "3. **Model Comparison** – measured results for all three models\n"
        "4. **Defect Prediction** – single and batch PASS/FAIL\n"
        "5. **Explainable AI** – SHAP importance and per-sample reasons\n"
        "6. **Manufacturing Insights** – predicted defect rates and drivers\n"
        "7. **Downloadable Reports** – CSV / Markdown / HTML exports"
    )

st.subheader("Semiconductor manufacturing in brief")
a, b, c = st.columns(3)
a.markdown(card("Many process steps, many signals",
                "Wafers pass through deposition, lithography, etching and other steps. Sensors and metrology points "
                "log pressures, temperatures, flows, voltages and more for every production entity."), unsafe_allow_html=True)
b.markdown(card("Small drifts, real yield loss",
                "Tiny shifts in process conditions can reduce yield. Most signals are irrelevant or noisy, so "
                "selecting informative ones matters as much as the classifier."), unsafe_allow_html=True)
c.markdown(card("Rare failures",
                "Only a small share of units fail in-line testing. The strong class imbalance drives every design "
                "choice here: metrics, thresholds and class weighting."), unsafe_allow_html=True)

st.subheader("Machine-learning methodology")
st.markdown(
    "<span class='ww-pill'>1 Split</span><span class='ww-pill'>2 Clean &amp; impute</span>"
    "<span class='ww-pill'>3 Filter &amp; select</span><span class='ww-pill'>4 Tune (CV, PR-AUC)</span>"
    "<span class='ww-pill'>5 Threshold</span><span class='ww-pill'>6 Evaluate</span>"
    "<span class='ww-pill'>7 Explain</span>",
    unsafe_allow_html=True,
)
st.write(
    "A stratified train/test split is made first. Every preprocessing step (missing-value filter, median imputation, "
    "constant and highly-correlated feature removal, univariate selection, scaling for the linear model) is part of a "
    "scikit-learn pipeline, so it is re-fitted inside each cross-validation fold. Class imbalance is handled with class "
    "weights (SMOTE is optional and only ever applied inside training folds). The decision threshold is chosen from "
    "out-of-fold training predictions, and the final model is selected by cross-validated PR-AUC — not by test results "
    "or accuracy."
)

st.subheader("Dataset overview")
st.write(
    "The **SECOM** dataset (McCann & Johnston, 2008; UCI Machine Learning Repository, CC BY 4.0) contains measurements "
    "from a semiconductor manufacturing process with a pass/fail label per production entity. See `data/README.md` "
    "for how to obtain it; this repository does not redistribute the raw data."
)
st.info("Model outputs are statistical estimates learned from historical data. They are not actual factory inspection "
        "results, and explanations describe model behaviour, not proven physical causes.")
