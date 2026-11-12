"""Pipeline construction, train/test splitting and feature bookkeeping."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .feature_engineering import CorrelationFilter, MissingnessFilter, TopKSelector


def build_pipeline(
    classifier,
    scale: bool = False,
    k: int = 40,
    use_smote: bool = False,
    seed: int = 42,
    max_missing: float = 0.4,
    corr_threshold: float = 0.95,
):
    """Full leakage-safe pipeline: clean -> impute -> filter -> select -> (SMOTE) -> classifier.

    All steps are fitted on the training data only. When ``use_smote`` is set an
    imbalanced-learn pipeline is returned so that SMOTE is applied to the
    training folds only (never at predict time).
    """
    steps = [
        ("missing", MissingnessFilter(max_missing=max_missing)),
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(threshold=1e-12)),
        ("correlation", CorrelationFilter(threshold=corr_threshold)),
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("select", TopKSelector(k=k)))
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImbPipeline
        except ImportError as exc:  # pragma: no cover
            raise ImportError("SMOTE requested but imbalanced-learn is not installed.") from exc
        steps.append(("smote", SMOTE(random_state=seed)))
        steps.append(("clf", classifier))
        return ImbPipeline(steps)
    steps.append(("clf", classifier))
    return Pipeline(steps)


def get_feature_names(pipeline, input_features) -> list[str]:
    """Names of the features that reach the classifier, in order."""
    names = np.asarray(list(input_features), dtype=object)
    for _, step in pipeline.steps[:-1]:
        if step is None or step == "passthrough" or not hasattr(step, "get_feature_names_out"):
            continue
        names = np.asarray(step.get_feature_names_out(names), dtype=object)
    return [str(n) for n in names]


def transform_features(pipeline, X: pd.DataFrame) -> np.ndarray:
    """Apply every transforming step before the classifier.

    Resamplers (SMOTE) have no ``transform`` and are skipped, so this is safe for
    pipelines trained with oversampling.
    """
    out = X
    for _, step in pipeline.steps[:-1]:
        if step is None or step == "passthrough" or not hasattr(step, "transform"):
            continue
        out = step.transform(out)
    return out


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, seed: int = 42, min_per_class: int = 10):
    """Stratified split with a sanity check on the minority class size."""
    counts = y.value_counts()
    if len(counts) != 2:
        raise ValueError("Target must contain both PASS and FAIL samples.")
    if counts.min() < min_per_class:
        raise ValueError(
            f"Only {int(counts.min())} samples in the minority class; need at least {min_per_class} "
            "for a stratified split and cross-validation."
        )
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def compute_feature_stats(X_train: pd.DataFrame) -> pd.DataFrame:
    """Per-feature summary of the training data (used for input widgets and docs)."""
    num = X_train.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    stats = pd.DataFrame({
        "median": num.median(),
        "mean": num.mean(),
        "std": num.std(),
        "min": num.min(),
        "p01": num.quantile(0.01),
        "p99": num.quantile(0.99),
        "max": num.max(),
        "missing_pct": num.isna().mean() * 100,
    })
    stats.index.name = "feature"
    return stats.reset_index()
