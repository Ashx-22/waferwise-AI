"""Downloadable Reports - exports of predictions, metrics and a summary report."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from src.reports import build_report, model_table  # noqa: E402
from src.ui import get_artifacts, setup_page, show_df, sidebar_footer  # noqa: E402

setup_page("Downloadable Reports", "Export predictions, measured metrics and a shareable summary report.")
sidebar_footer()

art = get_artifacts()
preds = st.session_state.get("ww_pred_results")
report = build_report(art, preds, st.session_state.get("ww_pred_source"), st.session_state.get("ww_pred_threshold"))
md_text = report.to_markdown()

st.subheader("Summary report")
c1, c2 = st.columns(2)
c1.download_button("Download report (Markdown)", md_text.encode(), "waferwise_report.md", "text/markdown",
                   type="primary")
c2.download_button("Download report (HTML)", report.to_html().encode(), "waferwise_report.html", "text/html")
with st.expander("Preview report"):
    st.markdown(md_text)

st.subheader("Data exports")
c1, c2, c3 = st.columns(3)
if preds is not None:
    c1.download_button("Predictions (CSV)", preds.to_csv(index=False).encode(), "waferwise_predictions.csv",
                       "text/csv")
else:
    c1.button("Predictions (CSV)", disabled=True, help="Run a batch prediction first.")
c2.download_button("Model metrics (CSV)", model_table(art.metrics, "test_tuned").to_csv(index=False).encode(),
                   "waferwise_model_metrics.csv", "text/csv")
c3.download_button("Model metadata (JSON)", json.dumps(art.metadata, indent=2).encode(), "waferwise_metadata.json",
                   "application/json")
if preds is None:
    st.info("No batch predictions are available yet. Score a CSV on the **Defect Prediction** page and the "
            "prediction summary and CSV export will appear here.")
else:
    st.caption(f"Latest batch predictions: {len(preds):,} samples from {st.session_state.get('ww_pred_source')}.")
    show_df(preds.head(200), height=260)
