"""Dataset Explorer - load (local or uploaded CSV), preview and profile the data."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from src import eda  # noqa: E402
from src.data_loader import (  # noqa: E402
    AmbiguousTargetError, DatasetFormatError, build_dataset, detect_target_column, read_csv_bytes,
)
from src.ui import (  # noqa: E402
    FAIL_COLOR, CYAN, dataset_missing_notice, fmt_pct, get_local_dataset, setup_page, show_df, show_fig,
    sidebar_footer, style_fig,
)

setup_page("Dataset Explorer", "Inspect the local SECOM data or upload your own CSV of manufacturing measurements.")
sidebar_footer()

source = st.radio("Data source", ["Local data/ folder", "Upload a CSV"], horizontal=True)
ds, err = None, None

if source == "Upload a CSV":
    up = st.file_uploader("CSV file (one row per sample, one column per feature, optional label column)",
                          type=["csv"], key="explorer_upload")
    if up is None:
        st.info("Upload a CSV to explore it. Nothing is stored on the server beyond this browser session.")
        if st.session_state.get("ww_dataset") is not None:
            ds = st.session_state["ww_dataset"]
            st.caption(f"Showing the previously uploaded dataset: {ds.source}")
    else:
        try:
            raw = read_csv_bytes(up, up.name)
        except DatasetFormatError as exc:
            st.error(f"Could not read the file: {exc}")
            st.stop()
        guess = detect_target_column(raw.columns)
        options = ["(no label column)"] + [str(c) for c in raw.columns]
        choice = st.selectbox("Label column (PASS/FAIL)", options,
                              index=options.index(str(guess)) if guess is not None else 0)
        fail_value = None
        target = None if choice == options[0] else choice
        try:
            ds = build_dataset(raw, f"Uploaded file '{up.name}'", target, auto_target=False)
        except AmbiguousTargetError as exc:
            st.warning(str(exc))
            fail_value = st.radio("Which value means FAIL?", exc.values, horizontal=True)
            try:
                ds = build_dataset(raw, f"Uploaded file '{up.name}'", target, fail_value, auto_target=False)
            except DatasetFormatError as exc2:
                st.error(str(exc2))
                st.stop()
        except DatasetFormatError as exc:
            st.error(f"Could not use this file: {exc}")
            st.stop()
        st.session_state["ww_dataset"] = ds
        for n in ds.notes:
            st.caption(n)
else:
    st.session_state["ww_dataset"] = None
    ds, err = get_local_dataset()

if ds is None:
    dataset_missing_notice(err)
    st.stop()

X, y = ds.X, ds.y
st.success(f"Loaded: {ds.source}")
if ds.target_info:
    st.caption(f"Target encoding: raw values {ds.target_info.get('raw_values')} → {ds.target_info.get('rule')} "
               f"(internal: {ds.target_info.get('internal_encoding')}).")
if ds.dropped_columns:
    st.caption(f"Columns set aside (ids, timestamps, non-numeric): {ds.dropped_columns[:8]}")

summary = eda.summarize(X, y)
c = st.columns(5)
c[0].metric("Rows", f"{summary['n_samples']:,}")
c[1].metric("Feature columns", f"{summary['n_features']:,}")
c[2].metric("Missing cells", fmt_pct(summary["missing_cells_pct"]))
c[3].metric("Duplicate rows", f"{summary['duplicate_rows']:,}")
c[4].metric("Constant features", f"{summary['constant_features']:,}")

tab_prev, tab_miss, tab_target, tab_stats = st.tabs(["Preview", "Missing values", "Target distribution",
                                                      "Feature statistics"])
with tab_prev:
    n = st.slider("Rows to preview", 5, min(200, len(X)), min(20, len(X))) if len(X) > 5 else len(X)
    prev = X.head(n).copy()
    if y is not None:
        prev.insert(0, "target (1=FAIL)", y.head(n).values)
    show_df(prev, height=420)

with tab_miss:
    miss = eda.missing_by_column(X)
    k1, k2, k3 = st.columns(3)
    k1.metric("Columns with any missing", f"{summary['columns_with_missing']:,}")
    k2.metric("Columns > 40% missing", f"{summary['columns_over_40pct_missing']:,}")
    k3.metric("Rows with any missing", fmt_pct(summary["rows_with_missing_pct"]))
    top = miss.head(30)
    if top["missing_pct"].max() > 0:
        fig = px.bar(top, x="feature", y="missing_pct", labels={"missing_pct": "% missing", "feature": ""})
        fig.update_traces(marker_color=CYAN)
        show_fig(style_fig(fig, "Top 30 columns by missing percentage"))
    else:
        st.info("No missing values found.")
    show_df(miss, height=300)

with tab_target:
    if y is None:
        st.info("This dataset has no label column, so no class distribution can be shown.")
    else:
        dist = eda.class_distribution(y)
        a, b = st.columns([1, 1])
        with a:
            fig = px.bar(dist, x="class", y="count", color="class", text="count",
                         color_discrete_map={"PASS": CYAN, "FAIL": FAIL_COLOR})
            show_fig(style_fig(fig, "Class counts", 340))
        with b:
            show_df(dist.round(2))
            ratio = dist.set_index("class")["count"]
            if {"PASS", "FAIL"} <= set(ratio.index) and ratio["FAIL"] > 0:
                st.write(f"Imbalance ratio PASS : FAIL ≈ **{ratio['PASS'] / ratio['FAIL']:.1f} : 1**. "
                         "Accuracy alone is therefore misleading; see Model Comparison for FAIL-focused metrics.")

with tab_stats:
    q = st.text_input("Filter features by name")
    desc = X.describe().T
    desc["missing_pct"] = X.isna().mean() * 100
    desc.index.name = "feature"
    desc = desc.reset_index()
    if q:
        desc = desc[desc["feature"].str.contains(q, case=False, regex=False)]
    show_df(desc, height=420)
