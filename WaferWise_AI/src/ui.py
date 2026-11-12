"""Shared Streamlit helpers: theme, cached loaders and layout utilities."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import config
from .data_loader import (
    AmbiguousTargetError, DatasetFormatError, DatasetNotFoundError, LoadedDataset, dataset_signature, load_dataset,
)
from .predict import Artifacts, ArtifactsMissingError, load_artifacts

NAVY, PANEL, BORDER = "#0a1628", "#0f2038", "#1e3a5f"
CYAN, FAIL_COLOR, MUTED, TEXT = "#22d3ee", "#fb7185", "#94a3b8", "#f1f5f9"
COLORWAY = [CYAN, FAIL_COLOR, "#a78bfa", "#fbbf24", "#34d399", "#60a5fa"]

_CSS = f"""
<style>
.block-container {{padding-top: 2.2rem; max-width: 1300px;}}
h1, h2, h3 {{letter-spacing: -0.01em;}}
h1 {{font-weight: 700;}}
[data-testid="stMetric"] {{background:{PANEL}; border:1px solid {BORDER}; border-radius:10px; padding:14px 16px;}}
[data-testid="stMetricLabel"] {{color:{MUTED};}}
[data-testid="stSidebar"] {{border-right: 1px solid {BORDER};}}
.ww-sub {{color:{MUTED}; margin-top:-0.6rem; margin-bottom:1.2rem; font-size:1.02rem;}}
.ww-card {{background:{PANEL}; border:1px solid {BORDER}; border-radius:12px; padding:18px 20px; height:100%;}}
.ww-card h4 {{margin:0 0 .4rem 0; color:{CYAN}; font-size:1rem;}}
.ww-card p {{margin:0; color:#cbd5e1; font-size:.93rem; line-height:1.5;}}
.ww-pill {{display:inline-block; padding:2px 10px; border-radius:999px; font-size:.8rem;
          border:1px solid {CYAN}; color:{CYAN}; margin-right:6px;}}
.ww-fail {{color:{FAIL_COLOR}; font-weight:700;}} .ww-pass {{color:{CYAN}; font-weight:700;}}
</style>
"""


def setup_page(title: str, subtitle: str | None = None, icon: str = "🔬") -> None:
    """Page config, theme CSS, sidebar brand and header. Call first on every page."""
    st.set_page_config(page_title=f"{title} · WaferWise AI", page_icon=icon, layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(_CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown(f"### 🔬 WaferWise AI\n<span style='color:{MUTED};font-size:.85rem'>"
                    "Semiconductor defect prediction &amp; explainable quality analysis</span>",
                    unsafe_allow_html=True)
        st.divider()
    st.title(title)
    if subtitle:
        st.markdown(f"<div class='ww-sub'>{subtitle}</div>", unsafe_allow_html=True)


def sidebar_footer() -> None:
    with st.sidebar:
        st.divider()
        st.caption("Predictions are model estimates, not factory inspection results.")


def card(title: str, body: str) -> str:
    return f"<div class='ww-card'><h4>{title}</h4><p>{body}</p></div>"


def show_fig(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", key=key)


def show_df(df: pd.DataFrame, height: int | None = None, **kw) -> None:
    if height is not None:
        kw["height"] = height
    st.dataframe(df, width="stretch", **kw)


def style_fig(fig: go.Figure, title: str | None = None, height: int = 380) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,32,56,0.55)",
        font=dict(color=TEXT, family="Inter, Segoe UI, Roboto, sans-serif"), colorway=COLORWAY, height=height,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10), title=dict(text=title or "", x=0.01),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#1b2f4d", zerolinecolor="#1b2f4d")
    fig.update_yaxes(gridcolor="#1b2f4d", zerolinecolor="#1b2f4d")
    return fig


# ------------------------------------------------------------------ artifacts ---
SETUP_HELP = """
**Trained model files were not found.** WaferWise needs the SECOM dataset and a one-off training run:

1. Put the dataset in `data/` — see `data/README.md` (UCI files `secom.data` + `secom_labels.data`, or a labelled CSV).
2. Run `python -m src.train` from the project folder (add `--quick` for a short smoke test).
3. Restart or refresh this app.
"""


@st.cache_resource(show_spinner="Loading trained model…")
def _cached_artifacts(models_dir: str, stamp: float) -> Artifacts:
    return load_artifacts(models_dir)


def get_artifacts(required: bool = True) -> Artifacts | None:
    """Cached artifacts. If missing: show setup help and stop the page (when ``required``)."""
    d = config.get_models_dir()
    meta = d / config.METADATA_JSON
    try:
        if not meta.exists():
            raise ArtifactsMissingError(f"No trained model found in '{d}'.")
        return _cached_artifacts(str(d), meta.stat().st_mtime)
    except ArtifactsMissingError as exc:
        if required:
            st.warning(SETUP_HELP)
            st.caption(f"Details: {exc}")
            st.stop()
        return None


@st.cache_data(show_spinner="Loading dataset…")
def _cached_local_dataset(data_dir: str, signature: tuple):
    try:
        return load_dataset(data_dir), None
    except (DatasetNotFoundError, DatasetFormatError, AmbiguousTargetError) as exc:
        return None, str(exc)


def get_local_dataset() -> tuple[LoadedDataset | None, str | None]:
    d = config.get_data_dir()
    return _cached_local_dataset(str(d), dataset_signature(d))


def get_active_dataset() -> tuple[LoadedDataset | None, str | None]:
    """Uploaded dataset (from the Dataset Explorer) if any, else the local data/ folder."""
    up = st.session_state.get("ww_dataset")
    if up is not None:
        return up, None
    return get_local_dataset()


def dataset_missing_notice(err: str | None) -> None:
    st.info(
        "No dataset is loaded. Upload a CSV on the **Dataset Explorer** page, or place the SECOM files in `data/` "
        "(see `data/README.md`)." + (f"\n\nDetails: {err}" if err else "")
    )


def fmt_pct(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}%"
