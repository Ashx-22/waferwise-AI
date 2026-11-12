"""Exploratory Data Analysis - missingness, relevance, distributions and correlations."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from src import eda  # noqa: E402
from src.ui import (  # noqa: E402
    CYAN, FAIL_COLOR, dataset_missing_notice, get_active_dataset, setup_page, show_df, show_fig, sidebar_footer,
    style_fig,
)

setup_page("Exploratory Data Analysis", "Understand data quality, class imbalance and which signals separate FAIL from PASS.")
sidebar_footer()

ds, err = get_active_dataset()
if ds is None:
    dataset_missing_notice(err)
    st.stop()
X, y = ds.X, ds.y
st.caption(f"Dataset: {ds.source}. EDA is descriptive and uses all rows; model training uses a separate "
           "stratified train/test split, so nothing here leaks into evaluation.")


@st.cache_data(show_spinner="Ranking features…")
def _ranking(X_, y_):
    return eda.feature_target_ranking(X_, y_)


tab_q, tab_rel, tab_dist, tab_corr = st.tabs(["Data quality", "Feature relevance", "Distributions", "Correlations"])

with tab_q:
    miss = eda.missing_by_column(X)
    fig = px.histogram(miss, x="missing_pct", nbins=40, labels={"missing_pct": "% missing in column"})
    fig.update_traces(marker_color=CYAN)
    show_fig(style_fig(fig, "How sparse are the columns?"))
    rows_missing = X.isna().mean(axis=1) * 100
    fig = px.histogram(rows_missing, nbins=40, labels={"value": "% of features missing in the row"})
    fig.update_traces(marker_color=CYAN)
    fig.update_layout(showlegend=False)
    show_fig(style_fig(fig, "How incomplete are the rows?"))
    s = eda.summarize(X, y)
    st.write(f"**{s['constant_features']}** constant columns carry no information and "
             f"**{s['duplicate_rows']}** duplicate rows were found. The model pipeline drops columns with more than "
             "40% missing values and imputes the rest with the training-set median.")

with tab_rel:
    if y is None:
        st.info("A label column is required to rank feature relevance.")
    else:
        rank = _ranking(X, y)
        n = st.slider("Features to show", 5, min(50, max(5, len(rank))), min(20, max(5, len(rank))))
        top = rank.head(n)
        fig = px.bar(top.iloc[::-1], x="f_score", y="feature", orientation="h", labels={"f_score": "ANOVA F-score"})
        fig.update_traces(marker_color=CYAN)
        show_fig(style_fig(fig, "Univariate association with FAIL (descriptive)", max(320, 22 * n + 80)))
        show_df(rank.head(200), height=300)
        st.caption("A high F-score means the feature's mean differs between PASS and FAIL. It is an association, "
                   "not evidence of causation.")

with tab_dist:
    if y is not None:
        rank = _ranking(X, y)
        default_features = rank["feature"].head(30).tolist() or list(X.columns[:30])
    else:
        default_features = list(X.columns[:30])
    feat = st.selectbox("Feature", list(X.columns), index=list(X.columns).index(default_features[0]))
    plot_df = X[[feat]].copy()
    if y is not None:
        plot_df["class"] = y.map({0: "PASS", 1: "FAIL"}).values
        a, b = st.columns(2)
        with a:
            fig = px.histogram(plot_df.dropna(), x=feat, color="class", barmode="overlay", opacity=0.65, nbins=40,
                               color_discrete_map={"PASS": CYAN, "FAIL": FAIL_COLOR}, histnorm="probability density")
            show_fig(style_fig(fig, f"{feat} by class"))
        with b:
            fig = px.box(plot_df.dropna(), x="class", y=feat, color="class",
                         color_discrete_map={"PASS": CYAN, "FAIL": FAIL_COLOR})
            show_fig(style_fig(fig, "Box plot"))
    else:
        fig = px.histogram(plot_df.dropna(), x=feat, nbins=40)
        fig.update_traces(marker_color=CYAN)
        show_fig(style_fig(fig, f"Distribution of {feat}"))
    st.caption(f"{int(X[feat].isna().sum())} missing values in this column.")

with tab_corr:
    n = st.slider("Number of features in the heat-map", 5, min(40, X.shape[1]), min(20, X.shape[1]))
    feats = (_ranking(X, y)["feature"].head(n).tolist() if y is not None else list(X.columns[:n]))
    corr = eda.top_correlation_matrix(X, feats)
    fig = px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto")
    show_fig(style_fig(fig, "Pearson correlation (most FAIL-associated features)" if y is not None
                       else "Pearson correlation", 560))
    pairs = eda.highly_correlated_pairs(X, feats, 0.9)
    st.write(f"Feature pairs with |r| ≥ 0.9 among these: **{len(pairs)}**. Strongly correlated features are largely "
             "redundant; the pipeline removes one of each pair above |r| = 0.95.")
    if len(pairs):
        show_df(pairs, height=240)
