"""Tests for artifact loading, input validation, prediction and explanations (synthetic data)."""
import io
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src import config, explainability
from src.data_loader import DatasetFormatError, read_csv_bytes
from src.predict import (
    ArtifactsMissingError, InputValidationError, load_artifacts, predict_frame, single_sample_frame, template_csv,
    validate_features,
)
from tests.conftest import make_synthetic


@pytest.fixture(scope="module")
def art(trained):
    return load_artifacts(trained[1])


def test_missing_artifacts_raise_helpful_error(tmp_path):
    with pytest.raises(ArtifactsMissingError, match="src.train"):
        load_artifacts(tmp_path)


def test_corrupt_artifacts_raise_helpful_error(trained, tmp_path):
    for f in (config.FINAL_PIPELINE, config.METADATA_JSON, config.METRICS_JSON):
        (tmp_path / f).write_text("not valid")
    with pytest.raises(ArtifactsMissingError, match="unreadable"):
        load_artifacts(tmp_path)


def test_artifacts_load_and_metadata_consistent(art):
    assert art.metadata["selected_model"] in config.MODEL_FILES
    assert len(art.expected_features) == art.metadata["dataset"]["n_features"]
    assert 0 < art.threshold < 1
    assert art.metadata["target_info"]["internal_encoding"] == "1 = FAIL, 0 = PASS"
    json.dumps(art.metadata)  # serialisable
    assert set(art.metrics["models"]) == set(config.MODEL_FILES)


def test_batch_prediction_on_unseen_data(art):
    X_new, _ = make_synthetic(n=80, seed=123)  # unseen rows
    val = validate_features(X_new, art.expected_features)
    res = predict_frame(art.pipeline, val.features, art.threshold)
    assert len(res) == 80
    assert res["fail_probability"].between(0, 1).all()
    assert set(res["prediction"]) <= {"PASS", "FAIL"}
    np.testing.assert_allclose(res["fail_probability"] + res["pass_probability"], 1.0, atol=1e-5)
    # a lower threshold can only increase the number of predicted FAIL
    low = predict_frame(art.pipeline, val.features, 0.05)
    assert (low["prediction"] == "FAIL").sum() >= (res["prediction"] == "FAIL").sum()


def test_column_order_does_not_matter(art):
    X_new, _ = make_synthetic(n=30, seed=5)
    shuffled = X_new[list(reversed(X_new.columns))]
    a = predict_frame(art.pipeline, validate_features(X_new, art.expected_features).features)
    b = predict_frame(art.pipeline, validate_features(shuffled, art.expected_features).features)
    np.testing.assert_allclose(a["fail_probability"], b["fail_probability"])


def test_target_column_is_removed_from_inputs(art):
    X_new, y_new = make_synthetic(n=30, seed=6)
    X_new["Pass/Fail"] = np.where(y_new == 1, 1, -1)
    val = validate_features(X_new, art.expected_features)
    assert "Pass/Fail" not in val.features.columns and val.dropped_label_columns == ["Pass/Fail"]
    assert any("label" in m.lower() for m in val.messages)


def test_validation_errors(art):
    exp = art.expected_features
    X_new, _ = make_synthetic(n=10, seed=7)
    with pytest.raises(InputValidationError, match="missing"):
        validate_features(X_new.drop(columns=exp[:3]), exp)
    with pytest.raises(InputValidationError, match="header"):
        validate_features(pd.DataFrame(np.zeros((3, len(exp)))), exp)
    with pytest.raises(InputValidationError, match="no rows"):
        validate_features(pd.DataFrame(columns=exp), exp)
    dup = pd.concat([X_new, X_new[[exp[0]]]], axis=1)
    with pytest.raises(InputValidationError, match="Duplicate"):
        validate_features(dup, exp)
    empty_row = X_new.copy()
    empty_row.iloc[0] = np.nan
    with pytest.raises(InputValidationError, match="no numeric"):
        validate_features(empty_row, exp)


def test_non_numeric_cells_and_extra_columns_are_handled(art):
    X_new, _ = make_synthetic(n=12, seed=8)
    X_new = X_new.astype(object)
    X_new.iloc[0, 0] = "abc"
    X_new["notes"] = "ok"
    val = validate_features(X_new, art.expected_features)
    assert val.non_numeric_cells == 1 and "notes" in val.extra_columns
    assert predict_frame(art.pipeline, val.features).shape[0] == 12


def test_invalid_csv_uploads_rejected():
    for payload in (b"", b"just one line", b"\x00\x01\x02garbage\xff"):
        with pytest.raises(DatasetFormatError):
            read_csv_bytes(io.BytesIO(payload), "bad.csv")


def test_single_sample_prediction(art):
    stats = art.feature_stats.set_index("feature")
    top = art.metadata["top_features"][:5]
    row = single_sample_frame({f: stats.loc[f, "median"] for f in top}, art.expected_features)
    assert row.shape == (1, len(art.expected_features)) and row[top].notna().all().all()
    res = predict_frame(art.pipeline, row, art.threshold)
    assert res.loc[0, "prediction"] in ("PASS", "FAIL")
    with pytest.raises(InputValidationError):
        single_sample_frame({"nope": 1.0}, art.expected_features)


def test_template_csv_header_matches_expected(art):
    tpl = pd.read_csv(io.BytesIO(template_csv(art.expected_features, art.feature_stats.set_index("feature")["median"])))
    assert list(tpl.columns) == art.expected_features and len(tpl) == 1
    validate_features(tpl, art.expected_features)


@pytest.mark.parametrize("model_key", list(config.MODEL_FILES))
def test_shap_works_for_every_model_type(trained, synthetic, model_key):
    """SHAP (or its documented fallback) must work for LR, RF and XGBoost."""
    X, y = synthetic
    pipe = joblib.load(trained[1] / config.MODEL_FILES[model_key])
    cache = explainability.build_shap_cache(pipe, X.iloc[:150], y.iloc[:150], list(X.columns), n_rows=60)
    assert cache["method"] == "shap", cache.get("error")
    assert cache["values"].shape == cache["X"].shape
    live = explainability.explain_raw(pipe, X.iloc[:2], cache["background"])
    assert live["values"].shape == (2, len(cache["feature_names"]))
    row = explainability.contribution_frame(live["values"][0], live["X"][0], cache["feature_names"], 5)
    assert len(row) == 5


def test_shap_fallback_does_not_break(trained, synthetic, monkeypatch):
    X, y = synthetic
    pipe = joblib.load(trained[1] / config.MODEL_FILES["random_forest"])

    def boom(*a, **k):
        raise RuntimeError("shap unavailable")

    monkeypatch.setattr(explainability, "compute_shap", boom)
    cache = explainability.build_shap_cache(pipe, X.iloc[:100], y.iloc[:100], list(X.columns), n_rows=50)
    assert cache["method"] == "model_importance" and "shap unavailable" in cache["error"]
    assert len(cache["importance"]) == len(cache["feature_names"])
