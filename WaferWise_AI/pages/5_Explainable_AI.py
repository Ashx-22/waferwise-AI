"""Explainable AI - global SHAP importance, summary plot and per-sample explanations."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from src import config, explainability  # noqa: E402
from src.ui import CYAN, FAIL_COLOR, get_artifacts, setup_page, show_df, show_fig, sidebar_footer, style_fig  # noqa: E402

setup_page("Explainable AI", "Which measurements drive the model's PASS / FAIL predictions?")
sidebar_footer()

art = get_artifacts()
cache = art.shap_cache
st.info(
    "**Read with care:** SHAP values describe how this *trained model* uses each feature. They show statistical "
    "association learned from historical data — not proof that a sensor reading *causes* a defect. Treat them as "
    "leads for process engineers to investigate."
)
if cache is None:
    st.warning("No explanation cache was saved with the model (training with `--shap-rows 0`, or the file is "
               "missing). Showing built-in model importances only.")
    if art.importance is not None:
        top = art.importance.head(20).iloc[::-1]
        fig = px.bar(top, x="importance", y="feature", orientation="h")
        fig.update_traces(marker_color=CYAN)
        show_fig(style_fig(fig, "Model feature importance", 520))
    st.stop()

names = cache["feature_names"]
is_shap = cache["method"] == "shap"
if not is_shap:
    st.warning(f"SHAP could not be computed for this model ({cache.get('error', 'unknown reason')}). "
               "Falling back to the model's built-in importances; per-sample SHAP plots are unavailable.")
else:
    st.caption(f"Explainer for **{art.metadata['selected_model_display']}** · values are in **{cache['units']}** "
               f"units toward the FAIL class · computed on {len(cache['X'])} held-out test rows.")

tab_g, tab_s, tab_i = st.tabs(["Global importance", "SHAP summary plot", "Individual explanation"])

with tab_g:
    imp = cache["importance"]
    n = st.slider("Features to show", 5, min(40, len(imp)), min(15, len(imp)))
    top = imp.head(n).iloc[::-1]
    fig = px.bar(top, x="importance", y="feature", orientation="h",
                 labels={"importance": "Mean |SHAP value|" if is_shap else "Model importance"})
    fig.update_traces(marker_color=CYAN)
    show_fig(style_fig(fig, "Most influential features (global)", max(320, 24 * n + 80)))
    show_df(imp.head(100), height=280)

with tab_s:
    if not is_shap:
        st.info("Summary plot requires SHAP values.")
    else:
        n = st.slider("Features in plot", 5, min(30, len(names)), min(15, len(names)), key="summary_n")
        fig = explainability.summary_figure(cache["values"], cache["X"], names, n)
        st.pyplot(fig, width="stretch")
        st.caption("Each dot is one sample. Horizontal position = effect on the FAIL prediction (right pushes "
                   "toward FAIL); colour = the feature's value (red high, blue low).")

with tab_i:
    if not is_shap:
        st.info("Individual explanations require SHAP values.")
        st.stop()
    options = ["Test-set sample (from explanation cache)"]
    if st.session_state.get("ww_single_row") is not None:
        options.append("Last single-sample prediction")
    if st.session_state.get("ww_pred_features") is not None:
        options.append("Row from the last batch upload")
    src = st.radio("Explain", options, horizontal=True)

    values = x_row = proba = base = None
    title = ""
    try:
        if src.startswith("Test-set"):
            y_c, p_c = cache["y"], cache["proba"]
            order = np.argsort(-p_c)
            labels = [f"#{i}  ·  P(FAIL)={p_c[i]:.2f}  ·  actual {'FAIL' if y_c[i] == 1 else 'PASS'}" for i in order]
            pick = st.selectbox("Sample (sorted by predicted FAIL probability)", range(len(order)),
                                format_func=lambda j: labels[j])
            i = int(order[pick])
            values, x_row, proba, base = cache["values"][i], cache["X"][i], float(p_c[i]), cache["base_value"]
            title = f"Test-set sample #{i}"
        else:
            if src.startswith("Last single"):
                raw = st.session_state["ww_single_row"]
                title = "Your single-sample input"
            else:
                feats = st.session_state["ww_pred_features"]
                idx = st.number_input("Row number (1-based)", 1, len(feats), 1) - 1
                raw = feats.iloc[[int(idx)]]
                title = f"Uploaded row {int(idx) + 1}"
            res = explainability.explain_raw(art.pipeline, raw, cache["background"])
            values, x_row, proba, base = res["values"][0], res["X"][0], float(res["proba"][0]), res["base_value"]
    except Exception as exc:  # keep the app alive on unexpected shapes
        st.error(f"Could not compute the explanation: {exc}")
        st.stop()

    thr = art.threshold
    c = st.columns(3)
    c[0].metric("Predicted FAIL probability", f"{proba:.1%}")
    c[1].metric("Model decision", config.FAIL_LABEL if proba >= thr else config.PASS_LABEL)
    c[2].metric(f"Baseline ({cache['units']})", f"{base:.3f}")
    contrib = explainability.contribution_frame(values, x_row, names, 15)
    contrib["colour"] = np.where(contrib["shap_value"] > 0, "towards FAIL", "towards PASS")
    fig = px.bar(contrib.iloc[::-1], x="shap_value", y="feature", orientation="h", color="colour",
                 color_discrete_map={"towards FAIL": FAIL_COLOR, "towards PASS": CYAN},
                 labels={"shap_value": f"SHAP value ({cache['units']})"}, hover_data=["feature_value"])
    show_fig(style_fig(fig, f"{title}: top feature contributions", 520))
    show_df(contrib.drop(columns="colour").round(4), height=300)
    st.caption("Feature values shown are those seen by the classifier after imputation (and scaling for the "
               "linear model). Positive SHAP values push the prediction toward FAIL.")
