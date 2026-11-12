"""SHAP-based explanations with a model-importance fallback.

SHAP values describe how the *trained model* arrives at a prediction. They are
associations learned from historical data, not proof of physical causes.
"""
from __future__ import annotations

import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from .preprocessing import get_feature_names, transform_features  # noqa: E402


def fail_class_values(sv) -> np.ndarray:
    """Return a (n_samples, n_features) array of SHAP values for the FAIL class."""
    if hasattr(sv, "values"):  # shap.Explanation
        sv = sv.values
    if isinstance(sv, list):
        sv = sv[1] if len(sv) == 2 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1] if sv.shape[2] == 2 else sv[:, :, 0]
    return sv


def _base_value(expected) -> float:
    arr = np.atleast_1d(np.asarray(expected, dtype=float))
    return float(arr[1] if arr.size == 2 else arr[0])


def build_explainer(clf, background: np.ndarray):
    """Create the appropriate SHAP explainer; returns (explainer, output_units)."""
    import shap

    if isinstance(clf, LogisticRegression):
        return shap.LinearExplainer(clf, background), "log-odds"
    units = "log-odds" if clf.__class__.__name__.startswith("XGB") else "probability"
    return shap.TreeExplainer(clf), units


def compute_shap(clf, X_t: np.ndarray, background: np.ndarray) -> dict:
    """SHAP values (FAIL class) for already-transformed features."""
    explainer, units = build_explainer(clf, background)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            sv = explainer.shap_values(X_t, check_additivity=False)
        except TypeError:
            sv = explainer.shap_values(X_t)
    return {"values": fail_class_values(sv), "base_value": _base_value(explainer.expected_value), "units": units}


def model_importance(clf, feature_names: list[str]) -> pd.DataFrame:
    """Fallback: built-in importances (trees) or |coefficients| (linear)."""
    if hasattr(clf, "feature_importances_"):
        imp = np.asarray(clf.feature_importances_, dtype=float)
    elif hasattr(clf, "coef_"):
        imp = np.abs(np.asarray(clf.coef_)).ravel()
    else:
        imp = np.zeros(len(feature_names))
    df = pd.DataFrame({"feature": feature_names, "importance": imp})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def importance_table(values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Mean absolute SHAP value per feature, ranked."""
    df = pd.DataFrame({"feature": feature_names, "importance": np.abs(values).mean(axis=0)})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def build_shap_cache(pipeline, X_raw: pd.DataFrame, y, input_features, n_rows: int = 200,
                     seed: int = 42, background_rows: int = 100) -> dict:
    """Compute and return everything the Explainable-AI page needs.

    Falls back to model importances if SHAP fails, so the app never breaks.
    """
    names = get_feature_names(pipeline, input_features)
    clf = pipeline[-1]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(X_raw), size=min(n_rows, len(X_raw)), replace=False))
    X_t = transform_features(pipeline, X_raw.iloc[idx])
    bg_idx = rng.choice(len(X_t), size=min(background_rows, len(X_t)), replace=False)
    cache = {
        "feature_names": names, "X": X_t, "y": np.asarray(y)[idx], "background": X_t[bg_idx],
        "proba": pipeline.predict_proba(X_raw.iloc[idx])[:, 1], "model_class": clf.__class__.__name__,
    }
    try:
        res = compute_shap(clf, X_t, cache["background"])
        cache.update(method="shap", values=res["values"], base_value=res["base_value"], units=res["units"],
                     importance=importance_table(res["values"], names))
    except Exception as exc:  # SHAP incompatibility must not break training
        cache.update(method="model_importance", values=None, base_value=None, units="n/a",
                     importance=model_importance(clf, names).assign(rank=lambda d: np.arange(1, len(d) + 1)),
                     error=f"{type(exc).__name__}: {exc}")
    return cache


def explain_raw(pipeline, X_raw: pd.DataFrame, background: np.ndarray) -> dict:
    """Live SHAP explanation for new raw rows (already validated / aligned)."""
    X_t = transform_features(pipeline, X_raw)
    res = compute_shap(pipeline[-1], X_t, background)
    res.update(X=X_t, proba=pipeline.predict_proba(X_raw)[:, 1])
    return res


def contribution_frame(values_row: np.ndarray, x_row: np.ndarray, names: list[str], top: int = 15) -> pd.DataFrame:
    """Top features by |SHAP| for one prediction."""
    df = pd.DataFrame({"feature": names, "feature_value": x_row, "shap_value": values_row})
    df["abs"] = df["shap_value"].abs()
    df = df.sort_values("abs", ascending=False).head(top).drop(columns="abs").reset_index(drop=True)
    df["direction"] = np.where(df["shap_value"] > 0, "towards FAIL", "towards PASS")
    return df


def summary_figure(values: np.ndarray, X: np.ndarray, names: list[str], max_display: int = 15):
    """Matplotlib SHAP beeswarm summary plot (dark-theme friendly)."""
    import shap

    plt.close("all")
    shap.summary_plot(values, X, feature_names=names, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.patch.set_facecolor("#0a1628")
    for ax in fig.axes:
        ax.set_facecolor("#0a1628")
        ax.tick_params(colors="#e2e8f0")
        ax.xaxis.label.set_color("#e2e8f0")
        ax.yaxis.label.set_color("#e2e8f0")
        for spine in ax.spines.values():
            spine.set_color("#334155")
    fig.set_size_inches(9, max(4, 0.35 * max_display + 1.5))
    fig.tight_layout()
    return fig


def save_cache(cache: dict, path) -> None:
    joblib.dump(cache, path, compress=3)


def load_cache(path) -> dict:
    return joblib.load(path)
