"""Tests for dataset loading, target encoding and the leakage-safe preprocessing pipeline."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.data_loader import (
    AmbiguousTargetError, DatasetFormatError, DatasetNotFoundError, build_dataset, load_dataset, normalize_target,
    read_csv_bytes,
)
from src.feature_engineering import CorrelationFilter, MissingnessFilter, TopKSelector
from src.preprocessing import build_pipeline, get_feature_names, split_data, transform_features
from tests.conftest import make_synthetic, write_uci_files


# ------------------------------------------------------------------ target ---
def test_uci_target_encoding():
    y, info = normalize_target(pd.Series([-1, 1, -1, -1, 1]))
    assert y.tolist() == [0, 1, 0, 0, 1]
    assert info["fail_value"] == 1 and "UCI" in info["rule"]


def test_zero_one_target_is_ambiguous_without_fail_value():
    with pytest.raises(AmbiguousTargetError):
        normalize_target(pd.Series([0, 1, 0, 1]))
    y, _ = normalize_target(pd.Series([0, 1, 0, 1]), fail_value=0)
    assert y.tolist() == [1, 0, 1, 0]


def test_text_target_and_bad_targets():
    y, _ = normalize_target(pd.Series(["Pass", "Fail", "pass"]))
    assert y.tolist() == [0, 1, 0]
    with pytest.raises(AmbiguousTargetError):
        normalize_target(pd.Series(["a", "b"]))
    with pytest.raises(DatasetFormatError):
        normalize_target(pd.Series([1, 2, 3]))


# ----------------------------------------------------------------- loading ---
def test_load_uci_roundtrip(tmp_path, synthetic):
    X, y = synthetic
    write_uci_files(tmp_path, X, y)
    ds = load_dataset(tmp_path)
    assert ds.X.shape == X.shape and ds.y.sum() == y.sum()
    assert ds.timestamps is not None and ds.X.isna().sum().sum() == X.isna().sum().sum()


def test_uci_misaligned_files_rejected(tmp_path, synthetic):
    X, y = synthetic
    write_uci_files(tmp_path, X, y)
    lines = (tmp_path / "secom_labels.data").read_text().splitlines()[:-5]
    (tmp_path / "secom_labels.data").write_text("\n".join(lines) + "\n")
    with pytest.raises(DatasetFormatError, match="do not match"):
        load_dataset(tmp_path)


def test_missing_dataset_raises_informative_error(tmp_path):
    with pytest.raises(DatasetNotFoundError, match="data/README"):
        load_dataset(tmp_path)
    with pytest.raises(DatasetNotFoundError):
        load_dataset(tmp_path / "does_not_exist")


def test_csv_layout_detects_target_ids_and_times(tmp_path):
    df = pd.DataFrame({"id": [1, 2, 3, 4], "Time": ["19/07/2008 11:55:00"] * 4, "f1": [1.0, 2, 3, 4],
                       "f2": ["1", "x", "3", "4"], "Pass/Fail": [-1, 1, -1, -1]})
    df.to_csv(tmp_path / "train.csv", index=False)
    ds = load_dataset(tmp_path)
    assert list(ds.X.columns) == ["f1", "f2"] and ds.y.tolist() == [0, 1, 0, 0]
    assert ds.timestamps is not None and ds.X["f2"].isna().sum() == 1


def test_build_dataset_unknown_target_column():
    with pytest.raises(DatasetFormatError):
        build_dataset(pd.DataFrame({"a": [1, 2], "b": [3, 4]}), "x", target_col="nope")


@pytest.mark.parametrize("payload", [b"", b"   \n", b"\x00\x01\x02\xff\xfe"])
def test_read_csv_bytes_rejects_bad_input(payload):
    with pytest.raises(DatasetFormatError):
        read_csv_bytes(payload, "bad.csv")


# ------------------------------------------------------------- transformers ---
def test_missingness_filter_uses_training_statistics_only():
    rng = np.random.default_rng(1)
    train = pd.DataFrame(rng.normal(size=(100, 3)), columns=list("abc"))
    train.loc[:70, "c"] = np.nan  # 71% missing in train
    test = pd.DataFrame(rng.normal(size=(10, 3)), columns=list("abc"))  # c fully observed in test
    f = MissingnessFilter(0.4).fit(train)
    assert list(f.transform(test).columns) == ["a", "b"]
    with pytest.raises(ValueError):
        f.transform(test[["a", "b"]])


def test_correlation_filter_and_topk():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(200, 1))
    X = np.hstack([a, a * 3 + 1, rng.normal(size=(200, 4))])
    cf = CorrelationFilter(0.95).fit(X)
    assert cf.transform(X).shape[1] == 5
    y = (X[:, 2] > 0.5).astype(int)
    sel = TopKSelector(k=100).fit(X, y)  # k larger than n_features keeps all
    assert sel.transform(X).shape[1] == 6
    sel2 = TopKSelector(k=1).fit(X, y)
    assert sel2.keep_idx_.tolist() == [2]


def test_pipeline_fits_on_train_only_and_names_align(synthetic):
    X, y = synthetic
    X_tr, X_te, y_tr, y_te = split_data(X, y, 0.2, 42)
    pipe = build_pipeline(LogisticRegression(max_iter=500), scale=True, k=15).fit(X_tr, y_tr)
    kept = pipe.named_steps["missing"].get_feature_names_out()
    np.testing.assert_allclose(pipe.named_steps["imputer"].statistics_, X_tr[list(kept)].median().to_numpy())
    names = get_feature_names(pipe, X.columns)
    assert len(names) == pipe[-1].n_features_in_ == 15
    assert "feature_011" not in names  # constant column removed
    assert "feature_021" not in names  # >40% missing column removed
    assert pipe.predict_proba(X_te).shape == (len(X_te), 2)


def test_smote_runs_only_inside_training(synthetic):
    X, y = synthetic
    X_tr, X_te, y_tr, _ = split_data(X, y, 0.2, 42)
    pipe = build_pipeline(LogisticRegression(max_iter=500), scale=True, k=10, use_smote=True).fit(X_tr, y_tr)
    assert type(pipe).__module__.startswith("imblearn")
    assert pipe.predict_proba(X_te).shape[0] == len(X_te)  # no resampling at predict time
    assert transform_features(pipe, X_te).shape == (len(X_te), 10)


def test_split_is_stratified_and_guards_small_classes(synthetic):
    X, y = synthetic
    _, _, y_tr, y_te = split_data(X, y, 0.2, 42)
    assert abs(y_tr.mean() - y_te.mean()) < 0.03
    with pytest.raises(ValueError, match="minority"):
        split_data(make_synthetic(100, 5, 0.0, 1)[0], pd.Series([0] * 98 + [1] * 2))
