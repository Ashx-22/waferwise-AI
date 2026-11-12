"""Defect Prediction - batch CSV scoring and single-sample prediction."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src import config  # noqa: E402
from src.data_loader import DatasetFormatError, read_csv_bytes  # noqa: E402
from src.predict import (  # noqa: E402
    InputValidationError, predict_frame, single_sample_frame, template_csv, validate_features,
)
from src.ui import (  # noqa: E402
    CYAN, FAIL_COLOR, fmt_pct, get_artifacts, setup_page, show_df, show_fig, sidebar_footer, style_fig,
)

setup_page("Defect Prediction", "Predict PASS / FAIL for manufacturing samples. Outputs are model estimates, not inspection results.")
sidebar_footer()

art = get_artifacts()
expected = art.expected_features
stats = art.feature_stats.set_index("feature") if art.feature_stats is not None else None
default_thr = float(np.clip(art.threshold, 0.01, 0.99))

threshold = st.slider(
    "Decision threshold (predict FAIL when probability ≥ threshold)", 0.01, 0.99, default_thr, 0.01,
    help="Default = threshold selected from out-of-fold training predictions. Lower it to catch more defects at the "
         "cost of more false alarms.")
st.caption(f"Model: **{art.metadata['selected_model_display']}** · expects {len(expected)} feature columns · "
           f"trained threshold {default_thr:.2f}")

tab_batch, tab_single = st.tabs(["Batch prediction (CSV)", "Single sample"])

# ------------------------------------------------------------------- batch ---
with tab_batch:
    medians = stats["median"] if stats is not None else None
    st.download_button("Download CSV template", template_csv(expected, medians), "waferwise_template.csv",
                       "text/csv", help="Header row with the exact feature names the model expects.")
    st.write("Upload a CSV with one row per sample and the model's feature columns. Label columns such as "
             "`Pass/Fail` are removed automatically; extra columns are ignored.")
    up = st.file_uploader("Manufacturing measurements (CSV)", type=["csv"], key="batch_upload")
    if up is not None:
        try:
            raw = read_csv_bytes(up, up.name)
            val = validate_features(raw, expected)
        except (DatasetFormatError, InputValidationError) as exc:
            st.error(f"Cannot score this file: {exc}")
        else:
            for m in val.messages:
                st.warning(m)
            res = predict_frame(art.pipeline, val.features, threshold)
            id_col = val.ignored_id_columns[0] if val.ignored_id_columns else None
            if id_col is not None and id_col in raw.columns:
                res.insert(1, id_col, raw[id_col].values)
            st.session_state.update(ww_pred_results=res, ww_pred_features=val.features, ww_pred_threshold=threshold,
                                    ww_pred_source=up.name)
            n_fail = int((res["prediction"] == config.FAIL_LABEL).sum())
            c = st.columns(4)
            c[0].metric("Samples scored", f"{len(res):,}")
            c[1].metric("Predicted FAIL", f"{n_fail:,}")
            c[2].metric("Predicted PASS", f"{len(res) - n_fail:,}")
            c[3].metric("Predicted defect rate", fmt_pct(n_fail / len(res) * 100))
            fig = px.histogram(res, x="fail_probability", nbins=40, color="prediction",
                               color_discrete_map={"PASS": CYAN, "FAIL": FAIL_COLOR})
            fig.add_vline(x=threshold, line_dash="dash", line_color="#e2e8f0", annotation_text="threshold")
            show_fig(style_fig(fig, "Predicted FAIL probability", 320))
            include = st.checkbox("Include input features in the downloaded CSV", value=False)
            out = res.join(val.features.add_prefix("in_")) if include else res
            show_df(out.head(1000), height=360)
            st.download_button("Download predictions (CSV)", out.to_csv(index=False).encode(),
                               "waferwise_predictions.csv", "text/csv", type="primary")
            st.caption("Predictions are statistical estimates from a trained model. The same results feed the "
                       "Explainable AI, Manufacturing Insights and Reports pages.")

# ------------------------------------------------------------------ single ---
with tab_single:
    ranked = [f for f in art.metadata.get("top_features", []) if f in expected] or expected[:20]
    n = st.slider("Number of key features to enter", 3, min(20, len(ranked)), min(9, len(ranked)))
    st.write("Enter values for the features the model relies on most. Every other feature is left **missing** and "
             "filled with its training-set median by the pipeline. Defaults are training medians.")
    with st.form("single_form"):
        cols = st.columns(3)
        values = {}
        for i, feat in enumerate(ranked[:n]):
            med, lo, hi = 0.0, None, None
            if stats is not None and feat in stats.index:
                med = float(stats.loc[feat, "median"]) if np.isfinite(stats.loc[feat, "median"]) else 0.0
                lo, hi = stats.loc[feat, "p01"], stats.loc[feat, "p99"]
            help_txt = f"Typical range (1st–99th percentile): {lo:.4g} – {hi:.4g}" if lo is not None else None
            values[feat] = cols[i % 3].number_input(feat, value=med, format="%g", help=help_txt, key=f"single_{feat}")
        submitted = st.form_submit_button("Predict", type="primary")
    if submitted:
        try:
            row = single_sample_frame(values, expected)
            res = predict_frame(art.pipeline, row, threshold).iloc[0]
        except (InputValidationError, ValueError) as exc:
            st.error(f"Prediction failed: {exc}")
        else:
            st.session_state.update(ww_single_row=row, ww_single_threshold=threshold)
            p = float(res["fail_probability"])
            c1, c2 = st.columns([1, 1])
            with c1:
                css = "ww-fail" if res["prediction"] == config.FAIL_LABEL else "ww-pass"
                st.markdown(f"### Predicted: <span class='{css}'>{res['prediction']}</span>", unsafe_allow_html=True)
                st.metric("FAIL probability", f"{p:.1%}")
                st.metric("PASS probability", f"{1 - p:.1%}")
                st.caption("Open **Explainable AI → Individual explanation** to see which features drove this result.")
            with c2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=p * 100, number={"suffix": "%"},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": FAIL_COLOR if p >= threshold else CYAN},
                           "threshold": {"line": {"color": "#e2e8f0", "width": 3}, "value": threshold * 100}}))
                show_fig(style_fig(fig, "FAIL probability", 300))
            st.download_button("Download this prediction (CSV)",
                               predict_frame(art.pipeline, row, threshold).to_csv(index=False).encode(),
                               "waferwise_single_prediction.csv", "text/csv")
