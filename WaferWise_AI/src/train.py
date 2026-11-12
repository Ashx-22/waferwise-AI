"""Train, tune, evaluate and save the WaferWise models.

Usage (from the repository root)::

    python -m src.train                 # full run on data/
    python -m src.train --quick         # small search, for smoke tests only

Protocol (no leakage):
  1. Stratified train/test split; the test set is touched only for final reporting.
  2. Every pre-processing / feature-selection step lives inside the pipeline, so
     it is re-fitted on each cross-validation training fold.
  3. Hyper-parameters are tuned with stratified CV, optimising PR-AUC.
  4. The decision threshold is chosen from out-of-fold *training* predictions.
  5. The final model is chosen by cross-validated PR-AUC (then F1) - never by
     test-set results or accuracy.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict

from . import config, eda, explainability
from .data_loader import load_dataset
from .evaluate import best_threshold, bootstrap_ci, compute_metrics, curves
from .preprocessing import build_pipeline, compute_feature_stats, split_data

SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc", "f1": "f1", "recall": "recall",
           "precision": "precision", "accuracy": "accuracy"}
K_GRID = [20, 40, 80]


def model_specs(seed: int, scale_pos_weight: float) -> dict:
    """name -> (needs_scaling, classifier, search grid). Imbalance handled via class weights."""
    from xgboost import XGBClassifier

    spw = max(float(scale_pos_weight), 1.0)
    return {
        "logistic_regression": (
            True,
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed),
            {"select__k": K_GRID, "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            False,
            RandomForestClassifier(class_weight="balanced_subsample", random_state=seed, n_jobs=1),
            {"select__k": K_GRID, "clf__n_estimators": [200, 400], "clf__max_depth": [None, 6, 12],
             "clf__min_samples_leaf": [1, 3, 5], "clf__max_features": ["sqrt", 0.3]},
        ),
        "xgboost": (
            False,
            XGBClassifier(eval_metric="logloss", tree_method="hist", n_jobs=1, random_state=seed, verbosity=0),
            {"select__k": K_GRID, "clf__n_estimators": [150, 300], "clf__max_depth": [3, 5],
             "clf__learning_rate": [0.03, 0.1], "clf__subsample": [0.8, 1.0],
             "clf__colsample_bytree": [0.6, 0.9], "clf__min_child_weight": [1, 5],
             "clf__scale_pos_weight": sorted({1.0, round(math.sqrt(spw), 2), round(spw, 2)})},
        ),
    }


def _grid_size(grid: dict) -> int:
    return int(np.prod([len(v) for v in grid.values()]))


def _clean(obj):
    """Make nested structures JSON serialisable."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if (isinstance(obj, float) and math.isnan(obj)) else float(obj)
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj


def _versions() -> dict:
    import shap
    import xgboost

    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__, "numpy": np.__version__, "xgboost": xgboost.__version__,
            "shap": shap.__version__}


def run_training(
    data_dir=None, models_dir=None, seed: int = config.RANDOM_SEED, test_size: float = 0.2,
    cv_folds: int = 5, n_iter: int = 15, quick: bool = False, use_smote: bool = False,
    threshold_metric: str = "f1", target_col: str | None = None, fail_value=None, n_jobs: int = 1,
    models: tuple[str, ...] = tuple(config.MODEL_FILES), shap_rows: int = 200, verbose: bool = True,
) -> dict:
    """Run the whole protocol and write artifacts to ``models_dir``."""
    log = print if verbose else (lambda *a, **k: None)
    warnings.filterwarnings("ignore")
    models_dir = Path(models_dir) if models_dir else config.get_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    if quick:
        cv_folds, n_iter = min(cv_folds, 3), min(n_iter, 3)

    # ---- 1. data ---------------------------------------------------------
    ds = load_dataset(data_dir, target_col=target_col, fail_value=fail_value)
    X, y = ds.X, ds.y
    log(f"[data] {ds.source}: {X.shape[0]} samples x {X.shape[1]} features; "
        f"FAIL={int(y.sum())} ({y.mean() * 100:.2f}%); target rule: {ds.target_info.get('rule')}")
    summary = eda.summarize(X, y)
    X_tr, X_te, y_tr, y_te = split_data(X, y, test_size, seed, min_per_class=max(10, 2 * cv_folds))
    log(f"[split] train={len(X_tr)} (FAIL={int(y_tr.sum())})  test={len(X_te)} (FAIL={int(y_te.sum())})")

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    specs = model_specs(seed, spw)

    results: dict = {}
    fitted: dict = {}
    for name in models:
        needs_scale, clf, grid = specs[name]
        t0 = time.time()
        pipe = build_pipeline(clf, scale=needs_scale, use_smote=use_smote, seed=seed)
        search = RandomizedSearchCV(
            pipe, grid, n_iter=min(n_iter, _grid_size(grid)), scoring=SCORING, refit="pr_auc", cv=skf,
            n_jobs=n_jobs, random_state=seed, error_score="raise",
        ).fit(X_tr, y_tr)
        i = search.best_index_
        cv = {}
        for m in SCORING:
            cv[f"{m}_mean"] = float(search.cv_results_[f"mean_test_{m}"][i])
            cv[f"{m}_std"] = float(search.cv_results_[f"std_test_{m}"][i])

        best = search.best_estimator_
        oof = cross_val_predict(clone(best), X_tr, y_tr, cv=skf, method="predict_proba", n_jobs=n_jobs)[:, 1]
        thr = best_threshold(y_tr, oof, threshold_metric)
        proba_te = best.predict_proba(X_te)[:, 1]
        results[name] = {
            "display_name": config.MODEL_DISPLAY[name],
            "best_params": {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()},
            "n_candidates_searched": int(min(n_iter, _grid_size(grid))),
            "cv": cv,
            "oof_threshold": thr,
            "oof_train": compute_metrics(y_tr, oof, thr),
            "test_tuned": compute_metrics(y_te, proba_te, thr),
            "test_default": compute_metrics(y_te, proba_te, 0.5),
            "curves": curves(y_te, proba_te),
            "bootstrap_ci_95": {
                "pr_auc": bootstrap_ci(y_te, proba_te, average_precision_score, seed=seed),
                "roc_auc": bootstrap_ci(y_te, proba_te, roc_auc_score, seed=seed),
            },
            "fit_seconds": round(time.time() - t0, 1),
        }
        fitted[name] = best
        joblib.dump(best, models_dir / config.MODEL_FILES[name], compress=3)
        t = results[name]["test_tuned"]
        log(f"[{name}] CV PR-AUC={cv['pr_auc_mean']:.3f}+/-{cv['pr_auc_std']:.3f}  "
            f"test@thr={thr:.3f}: recall={t['recall']:.3f} precision={t['precision']:.3f} "
            f"F1={t['f1']:.3f} PR-AUC={t['pr_auc']:.3f}  ({results[name]['fit_seconds']}s)")

    # ---- 2. model selection (cross-validation only) ------------------------
    selected = max(results, key=lambda n: (results[n]["cv"]["pr_auc_mean"], results[n]["cv"]["f1_mean"]))
    rule = ("Highest mean cross-validated PR-AUC on the training split (ties broken by CV F1). "
            "Test-set results and accuracy are not used for selection.")
    final = fitted[selected]
    joblib.dump(final, models_dir / config.FINAL_PIPELINE, compress=3)
    log(f"[select] final model: {selected}")

    # ---- 3. explainability + feature tables --------------------------------
    input_features = list(X.columns)
    shap_info = {"method": "none"}
    if shap_rows > 0:
        cache = explainability.build_shap_cache(final, X_te, y_te, input_features, n_rows=shap_rows, seed=seed)
        explainability.save_cache(cache, models_dir / config.SHAP_CACHE)
        cache["importance"].to_csv(models_dir / config.FEATURE_IMPORTANCE_CSV, index=False)
        names = cache["feature_names"]
        top_features = cache["importance"]["feature"].head(30).tolist()
        shap_info = {"method": cache["method"], "units": cache["units"], "rows_cached": int(len(cache["X"]))}
        if "error" in cache:
            shap_info["fallback_reason"] = cache["error"]
        log(f"[explain] method={cache['method']}")
    else:
        from .preprocessing import get_feature_names
        names = get_feature_names(final, input_features)
        imp = explainability.model_importance(final[-1], names)
        top_features = imp["feature"].head(30).tolist()
        imp.to_csv(models_dir / config.FEATURE_IMPORTANCE_CSV, index=False)

    compute_feature_stats(X_tr).to_csv(models_dir / config.FEATURE_STATS_CSV, index=False)

    # ---- 4. metadata / metrics ---------------------------------------------
    metadata = {
        "project": "WaferWise AI",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "data_source": ds.source,
        "dataset": summary,
        "target_info": ds.target_info,
        "split": {"test_size": test_size, "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
                  "fail_train": int(y_tr.sum()), "fail_test": int(y_te.sum())},
        "config": {"cv_folds": cv_folds, "n_iter": n_iter, "quick": quick, "use_smote": use_smote,
                   "threshold_metric": threshold_metric, "k_grid": K_GRID,
                   "max_missing": 0.4, "corr_threshold": 0.95},
        "selected_model": selected,
        "selected_model_display": config.MODEL_DISPLAY[selected],
        "selection_rule": rule,
        "decision_threshold": results[selected]["oof_threshold"],
        "expected_features": input_features,
        "model_input_features": names,
        "n_model_input_features": len(names),
        "top_features": top_features,
        "explainability": shap_info,
        "has_timestamps": ds.timestamps is not None,
        "versions": _versions(),
    }
    metrics = {
        "models": results, "selected_model": selected, "selection_rule": rule,
        "test_set": {"n": int(len(y_te)), "n_fail": int(y_te.sum()), "fail_prevalence": float(y_te.mean())},
        "primary_metrics_note": "FAIL is the positive class. PR-AUC, recall, precision and F1 are prioritised; "
                                "accuracy is shown for completeness only.",
    }
    (models_dir / config.METADATA_JSON).write_text(json.dumps(_clean(metadata), indent=2))
    (models_dir / config.METRICS_JSON).write_text(json.dumps(_clean(metrics), indent=2))
    log(f"[done] artifacts written to {models_dir}")
    return {"metadata": metadata, "metrics": metrics}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Train WaferWise models")
    ap.add_argument("--data-dir", default=None, help="folder with the SECOM files (default: ./data)")
    ap.add_argument("--models-dir", default=None, help="output folder (default: ./models)")
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--n-iter", type=int, default=15, help="random-search candidates per model")
    ap.add_argument("--quick", action="store_true", help="tiny search (smoke test, not for reporting)")
    ap.add_argument("--smote", action="store_true", help="apply SMOTE inside training folds only")
    ap.add_argument("--threshold-metric", choices=["f1", "f2"], default="f1")
    ap.add_argument("--target-col", default=None, help="CSV target column (auto-detected if omitted)")
    ap.add_argument("--fail-value", default=None, help="raw value meaning FAIL (needed for 0/1 targets)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--models", nargs="+", default=list(config.MODEL_FILES), choices=list(config.MODEL_FILES))
    ap.add_argument("--shap-rows", type=int, default=200, help="rows cached for SHAP plots (0 disables)")
    a = ap.parse_args(argv)
    run_training(a.data_dir, a.models_dir, a.seed, a.test_size, a.cv_folds, a.n_iter, a.quick, a.smote,
                 a.threshold_metric, a.target_col, a.fail_value, a.n_jobs, tuple(a.models), a.shap_rows)


if __name__ == "__main__":
    main()
