"""Manufacturing Insights - predicted defect rates and feature-level analysis."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from src import config, eda  # noqa: E402
from src.predict import InputValidationError, predict_frame, validate_features  # noqa: E402
from src.ui import (  # noqa: E402
    CYAN, FAIL_COLOR, fmt_pct, get_active_dataset, get_artifacts, setup_page, show_df, show_fig, sidebar_footer,
    style_fig,
)

setup_page("Manufacturing Insights", "Predicted defect counts, rates and the measurements that separate flagged samples.")
sidebar_footer()
st.warning("Everything on this page is a **model estimate**, not an actual factory inspection result.")

art = get_artifacts()
sources = []
if st.session_state.get("ww_pred_results") is not None:
    sources.append("Last batch upload (Defect Prediction page)")
sources.append("Score the loaded dataset (local data/ or Explorer upload)")
choice = st.radio("Samples to analyse", sources, horizontal=False)

res = feats = timestamps = None
if choice.startswith("Last batch"):
    res = st.session_state["ww_pred_results"]
    feats = st.session_state["ww_pred_features"]
    st.caption(f"Source: {st.session_state.get('ww_pred_source', 'uploaded file')} · threshold "
               f"{st.session_state.get('ww_pred_threshold', art.threshold):.2f}")
else:
    ds, err = get_active_dataset()
    if ds is None:
        st.info("No dataset is loaded. Upload a CSV on the Dataset Explorer page or place the SECOM files in `data/`.")
        st.stop()
    try:
        val = validate_features(ds.X, art.expected_features)
    except InputValidationError as exc:
        st.error(f"The loaded dataset does not match the model's expected features: {exc}")
        st.stop()
    thr = st.slider("Decision threshold", 0.01, 0.99, float(np.clip(art.threshold, 0.01, 0.99)), 0.01)
    res, feats, timestamps = predict_frame(art.pipeline, val.features, thr), val.features, ds.timestamps
    st.caption(f"Source: {ds.source}")
    if ds.source.startswith(("UCI", "CSV")):
        st.info("If this is the same data the model was trained on, scores for training rows are optimistic. Use new, "
                "unseen samples for an honest picture.")

n = len(res)
n_fail = int((res["prediction"] == config.FAIL_LABEL).sum())
rate = n_fail / n * 100
train_rate = art.metadata["dataset"].get("fail_pct")
c = st.columns(4)
c[0].metric("Samples analysed", f"{n:,}")
c[1].metric("Predicted FAIL", f"{n_fail:,}")
c[2].metric("Predicted PASS", f"{n - n_fail:,}")
c[3].metric("Predicted defect rate", fmt_pct(rate), None if train_rate is None else f"training FAIL rate {train_rate:.2f}%",
            delta_color="off")

a, b = st.columns(2)
with a:
    fig = px.histogram(res, x="fail_probability", nbins=40, color="prediction",
                       color_discrete_map={"PASS": CYAN, "FAIL": FAIL_COLOR})
    show_fig(style_fig(fig, "Distribution of predicted FAIL probability", 340))
with b:
    counts = res["prediction"].value_counts().rename_axis("prediction").reset_index(name="count")
    fig = px.pie(counts, names="prediction", values="count", hole=0.55, color="prediction",
                 color_discrete_map={"PASS": CYAN, "FAIL": FAIL_COLOR})
    show_fig(style_fig(fig, "Predicted PASS vs FAIL", 340))

if timestamps is not None and timestamps.notna().sum() > 3:
    t = pd.DataFrame({"time": pd.to_datetime(timestamps).values, "fail": (res["prediction"] == "FAIL").values})
    t = t.dropna()
    if t["time"].dt.normalize().nunique() >= 3:
        daily = t.set_index("time")["fail"].resample("D").agg(["mean", "count"]).dropna()
        daily = daily[daily["count"] > 0].reset_index()
        daily["predicted_defect_rate_%"] = daily["mean"] * 100
        fig = px.line(daily, x="time", y="predicted_defect_rate_%", markers=True)
        fig.update_traces(line_color=CYAN)
        show_fig(style_fig(fig, "Predicted defect rate over time (daily)", 320))

st.subheader("Highest-risk samples")
show_df(res.sort_values("fail_probability", ascending=False).head(20).reset_index(drop=True), height=300)

st.subheader("Feature-level analysis")
model_feats = [f for f in art.metadata["model_input_features"] if f in feats.columns]
group = (res["prediction"] == config.FAIL_LABEL).astype(int).reset_index(drop=True)
if group.sum() < 2 or (group == 0).sum() < 2:
    st.info("Feature comparison needs at least two predicted FAIL and two predicted PASS samples.")
else:
    smd = eda.standardized_mean_difference(feats[model_feats], group)
    top = smd.head(15)
    fig = px.bar(top.iloc[::-1], x="smd", y="feature", orientation="h",
                 labels={"smd": "Standardised mean difference (FAIL − PASS)"})
    fig.update_traces(marker_color=CYAN)
    show_fig(style_fig(fig, "Features that differ most between predicted-FAIL and predicted-PASS samples", 460))
    pick = st.selectbox("Compare a feature", top["feature"].tolist())
    plot = pd.DataFrame({pick: feats[pick].values, "prediction": res["prediction"].values}).dropna()
    fig = px.box(plot, x="prediction", y=pick, color="prediction",
                 color_discrete_map={"PASS": CYAN, "FAIL": FAIL_COLOR})
    show_fig(style_fig(fig, f"{pick} by predicted class", 360))

st.subheader("Quality insights")
bullets = [f"**{n_fail:,} of {n:,}** samples ({rate:.2f}%) are predicted FAIL at the chosen threshold."]
if art.importance is not None and len(art.importance):
    bullets.append("Features the model relies on most overall: "
                   + ", ".join(f"`{f}`" for f in art.importance["feature"].head(5)) + ".")
if train_rate is not None and abs(rate - train_rate) > 0:
    bullets.append(f"The predicted rate is {'higher' if rate > train_rate else 'lower'} than the FAIL rate observed in "
                   f"the training data ({train_rate:.2f}%). A shift can reflect process change, different data, or "
                   "threshold choice — it is a prompt to investigate, not a diagnosis.")
st.markdown("\n".join(f"- {b}" for b in bullets))
