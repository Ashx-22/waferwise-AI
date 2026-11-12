"""Custom scikit-learn transformers used inside the modelling pipeline.

Every statistic (missing-value fractions, correlations, univariate scores) is
learned in ``fit`` and therefore, when the pipeline is used with
cross-validation, only from the training fold - no train/test leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif


class MissingnessFilter(BaseEstimator, TransformerMixin):
    """Coerce to numeric, replace +/-inf by NaN and drop sparse columns.

    Columns whose missing fraction in the *training* data exceeds
    ``max_missing`` (default 40 %) are removed. Returns a DataFrame so the
    following imputer keeps column names.
    """

    def __init__(self, max_missing: float = 0.4):
        self.max_missing = max_missing

    def _to_frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X
        else:
            arr = np.asarray(X)
            names = getattr(self, "feature_names_in_", None)
            df = pd.DataFrame(arr, columns=names if names is not None and len(names) == arr.shape[1] else None)
        df = df.apply(pd.to_numeric, errors="coerce")
        return df.replace([np.inf, -np.inf], np.nan)

    def fit(self, X, y=None):
        df = self._to_frame(X)
        self.feature_names_in_ = np.array([str(c) for c in df.columns], dtype=object)
        self.missing_fraction_ = df.isna().mean().to_numpy()
        self.keep_mask_ = self.missing_fraction_ <= self.max_missing
        if not self.keep_mask_.any():
            raise ValueError("All columns exceed the allowed missing-value fraction.")
        return self

    def transform(self, X):
        df = self._to_frame(X)
        if isinstance(X, pd.DataFrame):
            missing = [c for c in self.feature_names_in_ if c not in df.columns]
            if missing:
                raise ValueError(f"Missing expected columns: {missing[:10]}")
            df = df[list(self.feature_names_in_)]
        elif df.shape[1] != len(self.feature_names_in_):
            raise ValueError(f"Expected {len(self.feature_names_in_)} columns, got {df.shape[1]}.")
        else:
            df.columns = list(self.feature_names_in_)
        return df.loc[:, self.keep_mask_]

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_[self.keep_mask_]


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop one of each pair of features with |Pearson r| above ``threshold``.

    Greedy and order dependent (the earlier column is kept). Works on numeric
    arrays with no missing values (place it after the imputer).
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.abs(np.nan_to_num(np.corrcoef(arr, rowvar=False)))
        n = arr.shape[1]
        keep: list[int] = []
        for i in range(n):
            if not keep or corr[i, keep].max() <= self.threshold:
                keep.append(i)
        self.keep_idx_ = np.array(keep, dtype=int)
        self.n_features_in_ = n
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_idx_]

    def get_feature_names_out(self, input_features=None):
        names = np.asarray(input_features if input_features is not None
                           else [f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        return names[self.keep_idx_]


class TopKSelector(BaseEstimator, TransformerMixin):
    """Keep the ``k`` features with the highest ANOVA F-score against the target.

    ``k`` larger than the number of available features simply keeps them all,
    so the same search grid works for datasets of any width.
    """

    def __init__(self, k: int = 40):
        self.k = k

    def fit(self, X, y):
        arr = np.asarray(X, dtype=float)
        scores, _ = f_classif(arr, np.asarray(y))
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0)
        k = min(int(self.k), arr.shape[1])
        order = np.argsort(-scores, kind="stable")[:k]
        self.scores_ = scores
        self.keep_idx_ = np.sort(order)
        self.n_features_in_ = arr.shape[1]
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_idx_]

    def get_feature_names_out(self, input_features=None):
        names = np.asarray(input_features if input_features is not None
                           else [f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        return names[self.keep_idx_]
