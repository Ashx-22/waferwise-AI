"""Metrics for the imbalanced PASS/FAIL problem. FAIL is the positive class (1)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    """All headline metrics at a given decision threshold.

    ``roc_auc`` and ``pr_auc`` are threshold independent. ``accuracy`` is
    reported for completeness only - it is misleading on imbalanced data.
    """
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) == 2
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)) if both else 0.0,
        "roc_auc": float(roc_auc_score(y_true, proba)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y_true, proba)) if both else float("nan"),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def best_threshold(y_true, proba, metric: str = "f1") -> float:
    """Decision threshold maximising F1 (or F2) on the supplied probabilities.

    Call it only with out-of-fold training predictions, never with test data.
    """
    beta = 2.0 if metric == "f2" else 1.0
    prec, rec, thr = precision_recall_curve(np.asarray(y_true).astype(int), np.asarray(proba, dtype=float))
    prec, rec = prec[:-1], rec[:-1]  # align with thresholds
    if len(thr) == 0:
        return 0.5
    denom = beta**2 * prec + rec
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.where(denom > 0, (1 + beta**2) * prec * rec / denom, 0.0)
    return float(thr[int(np.argmax(score))])


def curves(y_true, proba, max_points: int = 200) -> dict:
    """ROC and precision-recall curve coordinates (down-sampled for storage)."""
    y_true = np.asarray(y_true).astype(int)
    fpr, tpr, _ = roc_curve(y_true, proba)
    prec, rec, _ = precision_recall_curve(y_true, proba)

    def thin(*arrs):
        n = len(arrs[0])
        idx = np.unique(np.linspace(0, n - 1, min(n, max_points)).astype(int))
        return [np.round(a[idx], 5).tolist() for a in arrs]

    f, t = thin(fpr, tpr)
    p, r = thin(prec, rec)
    return {"roc": {"fpr": f, "tpr": t}, "pr": {"precision": p, "recall": r}}


def bootstrap_ci(y_true, proba, metric_fn, n_boot: int = 200, seed: int = 42, alpha: float = 0.05):
    """Percentile bootstrap interval for a threshold-independent metric."""
    rng = np.random.default_rng(seed)
    y_true, proba = np.asarray(y_true).astype(int), np.asarray(proba, dtype=float)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(metric_fn(y_true[idx], proba[idx]))
    if not vals:
        return [float("nan"), float("nan")]
    return [float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))]
