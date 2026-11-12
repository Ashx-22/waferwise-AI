"""End-to-end training tests on synthetic data (artifact structure, reproducibility, guards)."""
import json

import joblib
import numpy as np
import pytest

from src import config
from src.data_loader import AmbiguousTargetError, DatasetNotFoundError
from src.evaluate import best_threshold, compute_metrics
from src.train import run_training
from tests.conftest import make_synthetic, write_uci_files


def test_all_artifacts_written(trained):
    _, models_dir = trained
    expected = [config.FINAL_PIPELINE, config.METADATA_JSON, config.METRICS_JSON, config.FEATURE_STATS_CSV,
                config.FEATURE_IMPORTANCE_CSV, config.SHAP_CACHE, *config.MODEL_FILES.values()]
    for name in expected:
        assert (models_dir / name).exists(), name


def test_three_models_evaluated_with_required_metrics(trained):
    metrics = json.loads((trained[1] / config.METRICS_JSON).read_text())
    assert set(metrics["models"]) == {"logistic_regression", "random_forest", "xgboost"}
    for m in metrics["models"].values():
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "confusion_matrix"):
            assert key in m["test_tuned"] and key in m["test_default"]
        assert np.array(m["test_tuned"]["confusion_matrix"]).sum() == metrics["test_set"]["n"]
        assert {"roc", "pr"} <= set(m["curves"])
        assert m["cv"]["pr_auc_mean"] >= 0


def test_selection_is_by_cv_pr_auc_not_test_or_accuracy(trained):
    metrics = json.loads((trained[1] / config.METRICS_JSON).read_text())
    best = max(metrics["models"], key=lambda k: (metrics["models"][k]["cv"]["pr_auc_mean"],
                                                 metrics["models"][k]["cv"]["f1_mean"]))
    assert metrics["selected_model"] == best
    meta = json.loads((trained[1] / config.METADATA_JSON).read_text())
    assert meta["selected_model"] == best and "Test-set" in meta["selection_rule"]


def test_final_pipeline_is_complete_and_matches_selected(trained, synthetic):
    X, _ = synthetic
    final = joblib.load(trained[1] / config.FINAL_PIPELINE)
    sel = json.loads((trained[1] / config.METADATA_JSON).read_text())["selected_model"]
    other = joblib.load(trained[1] / config.MODEL_FILES[sel])
    np.testing.assert_allclose(final.predict_proba(X.iloc[:20]), other.predict_proba(X.iloc[:20]))
    assert [n for n, _ in final.steps][:2] == ["missing", "imputer"]  # raw data goes in; preprocessing is inside


def test_training_is_reproducible(tmp_path, synthetic):
    X, y = synthetic
    d = tmp_path / "d"
    d.mkdir()
    write_uci_files(d, X, y)
    a = run_training(d, tmp_path / "a", quick=True, models=("logistic_regression", "random_forest"),
                     shap_rows=0, verbose=False)
    b = run_training(d, tmp_path / "b", quick=True, models=("logistic_regression", "random_forest"),
                     shap_rows=0, verbose=False)
    for k in a["metrics"]["models"]:
        assert a["metrics"]["models"][k]["test_tuned"] == b["metrics"]["models"][k]["test_tuned"]


def test_smote_training_runs(tmp_path, synthetic):
    d = tmp_path / "d"
    d.mkdir()
    write_uci_files(d, *synthetic)
    out = run_training(d, tmp_path / "m", quick=True, use_smote=True, models=("logistic_regression",),
                       shap_rows=50, verbose=False)
    assert out["metadata"]["config"]["use_smote"] is True
    assert (tmp_path / "m" / config.FINAL_PIPELINE).exists()


def test_training_fails_cleanly_without_data_or_with_ambiguous_target(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        run_training(tmp_path / "empty", tmp_path / "m", quick=True, verbose=False)
    X, y = make_synthetic(200, 10, 0.1, 3)
    d = tmp_path / "csv"
    d.mkdir()
    X.assign(label=y.values).to_csv(d / "train.csv", index=False)  # 0/1 target: which value is FAIL?
    with pytest.raises(AmbiguousTargetError):
        run_training(d, tmp_path / "m2", quick=True, verbose=False)
    out = run_training(d, tmp_path / "m3", quick=True, fail_value=1, models=("logistic_regression",),
                       shap_rows=0, verbose=False)
    assert out["metadata"]["target_info"]["rule"].startswith("explicit")


def test_metric_helpers():
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9])
    m = compute_metrics(y, p, 0.5)
    assert m["recall"] == 1.0 and m["precision"] == pytest.approx(2 / 3) and m["confusion_matrix"] == [[3, 1], [0, 2]]
    assert 0.5 <= best_threshold(y, p) <= 0.9
