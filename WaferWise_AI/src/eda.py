"""Descriptive analysis helpers (pure pandas/numpy, no Streamlit)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif


def summarize(X: pd.DataFrame, y: pd.Series | None = None) -> dict:
    """Headline dataset statistics, all computed from the supplied data."""
    n_rows, n_cols = X.shape
    miss = X.isna()
    nunique = X.nunique(dropna=True)
    out = {
        "n_samples": int(n_rows),
        "n_features": int(n_cols),
        "missing_cells_pct": float(miss.to_numpy().mean() * 100) if X.size else 0.0,
        "columns_with_missing": int(miss.any().sum()),
        "columns_over_40pct_missing": int((miss.mean() > 0.4).sum()),
        "rows_with_missing_pct": float(miss.any(axis=1).mean() * 100) if n_rows else 0.0,
        "constant_features": int((nunique <= 1).sum()),
        "duplicate_rows": int(X.duplicated().sum()),
    }
    if y is not None and len(y):
        n_fail = int((y == 1).sum())
        out.update(n_fail=n_fail, n_pass=int(len(y) - n_fail), fail_pct=float(n_fail / len(y) * 100))
    return out


def missing_by_column(X: pd.DataFrame) -> pd.DataFrame:
    """Missing count and percentage per column, sorted descending."""
    m = X.isna().sum()
    df = pd.DataFrame({"feature": m.index, "missing_count": m.values,
                       "missing_pct": m.values / max(len(X), 1) * 100})
    return df.sort_values("missing_pct", ascending=False).reset_index(drop=True)


def class_distribution(y: pd.Series) -> pd.DataFrame:
    counts = y.map({0: "PASS", 1: "FAIL"}).value_counts()
    df = counts.rename_axis("class").reset_index(name="count")
    df["percent"] = df["count"] / df["count"].sum() * 100
    return df


def feature_target_ranking(X: pd.DataFrame, y: pd.Series, top: int | None = None) -> pd.DataFrame:
    """Univariate ANOVA F-score of each feature against FAIL (median-imputed).

    Descriptive only - it is computed on whatever data is passed in.
    """
    num = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    filled = num.fillna(num.median()).fillna(0.0)
    keep = filled.columns[filled.std() > 0]
    if len(keep) == 0 or y.nunique() < 2:
        return pd.DataFrame(columns=["feature", "f_score", "p_value"])
    f, p = f_classif(filled[keep], y)
    df = pd.DataFrame({"feature": keep, "f_score": np.nan_to_num(f), "p_value": p})
    df = df.sort_values("f_score", ascending=False).reset_index(drop=True)
    return df.head(top) if top else df


def top_correlation_matrix(X: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Pearson correlation among the chosen features (pairwise complete)."""
    return X[features].apply(pd.to_numeric, errors="coerce").corr()


def highly_correlated_pairs(X: pd.DataFrame, features: list[str], threshold: float = 0.9) -> pd.DataFrame:
    corr = top_correlation_matrix(X, features).abs()
    rows = []
    for i, a in enumerate(features):
        for b in features[i + 1:]:
            v = corr.loc[a, b]
            if pd.notna(v) and v >= threshold:
                rows.append((a, b, float(v)))
    return pd.DataFrame(rows, columns=["feature_a", "feature_b", "abs_correlation"]).sort_values(
        "abs_correlation", ascending=False).reset_index(drop=True)


def standardized_mean_difference(X: pd.DataFrame, group: pd.Series) -> pd.DataFrame:
    """Cohen's-d style difference of feature means between group==1 and group==0."""
    num = X.apply(pd.to_numeric, errors="coerce")
    a, b = num[group == 1], num[group == 0]
    if len(a) < 2 or len(b) < 2:
        return pd.DataFrame(columns=["feature", "smd", "mean_group1", "mean_group0"])
    pooled = np.sqrt((a.var() + b.var()) / 2).replace(0, np.nan)
    smd = (a.mean() - b.mean()) / pooled
    df = pd.DataFrame({"feature": smd.index, "smd": smd.values, "mean_group1": a.mean().values,
                       "mean_group0": b.mean().values}).dropna(subset=["smd"])
    return df.reindex(df["smd"].abs().sort_values(ascending=False).index).reset_index(drop=True)
